"""
================================================================================
 generate_tier1_masks.py — Phase 2: SAM2 Auto-Masking + QA  [v3]
================================================================================
 PURPOSE:
   Apply SAM2 to all 15,000 Tier 1 images to generate precise binary leaf
   silhouette masks. Masks are stored as BOTH:
     - float32 .npy  (raw SAM2 probability map — used as soft targets)
     - uint8 .png    (binarized at 0.5 — for visualization)

 AUTO-PROMPTING STRATEGY:
   For each image:
   1. Convert to HSV. Detect BOTH healthy-green AND yellowed/necrotic tissue.
      Yellow detection is critical: MSV and MLN destroy green tissue, so the
      v1 green-only approach rejected exactly the images we most need masks for.
   2. Pick the best available centroid (green → yellow → combined → center).
      Center-of-image is used as a last-resort fallback instead of rejecting.
   3. Use four image corners (10px inset) → background prompts (label=0).
   Fully automatic, deterministic, and reproducible.

 QA FILTERS (v2 — relaxed vs v1):
   Filter 1 — Coverage range  : foreground must be 5%–90% of image
                                 (was 10% — diseased leaves are sparse)
   Filter 2 — Mean confidence : mean prob of foreground region ≥ 0.65
                                 (unchanged)
   Filter 3 — Aspect ratio    : mask bounding box ratio ≥ 1.05
                                 (was 1.20 — overhead shots can be squarish)
   Target rejection rate: < 8% of total images.

 CHANGELOG (v2 → v3):
   [NEW]  YOLO-guided prompting: loads checkpoints/yolo/best.pt if present.
          When a leaf is detected, SAM2 receives a tight box prompt AND
          foreground points whose HSV centroid search is spatially constrained
          to pixels inside the YOLO box — eliminating background centroid drift.
   [NEW]  Calibrated QA confidence threshold: reads logs/yolo_qa_calibration.csv
          (written by train_yolo_detector.py) to replace the fixed 0.65 with a
          threshold derived from actual gold-standard IoU measurements.
   [NEW]  QA report gains two extra columns: prompt_mode, yolo_box.
   [NEW]  Prompt strategy prefixed with "yolo_" when YOLO detection is used.
   [KEEP] Full v2 HSV-only fallback when YOLO weights are absent, detection
          confidence is too low, or the detected box is degenerate.

 CHANGELOG (v1 → v2):
   [FIX]  get_leaf_centroid: adds yellow HSV range (H=15–45) for MSV/MLN tissue
   [FIX]  Minimum leaf-tissue coverage threshold: 0.10 → 0.05
   [FIX]  Center-of-image fallback instead of immediate rejection
   [FIX]  Aspect ratio QA threshold: 1.20 → 1.05
   [FIX]  Duplicate REPORTS_DIR import removed

 OUTPUTS:
   data/tier1_leaf_masks/{stem}_softmask.npy   ← float32 probability map
   data/tier1_leaf_masks/{stem}_mask.png        ← uint8 binary visualization
   tier1_qa_report.csv                          ← per-image QA log
                                                   (+ prompt_mode, yolo_box columns)
================================================================================
"""

import csv
import time
from collections import defaultdict
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from image_utils import load_image_rgb, to_hsv   # EXIF correction

from config import (
    SEED,
    TIER1_RAW_DIR, TIER1_MASKS_DIR, TIER1_QA_REPORT, REPORTS_DIR,
    SAM2_CHECKPOINT, SAM2_CONFIG,
    SAM2_GREEN_H_MIN, SAM2_GREEN_H_MAX,
    SAM2_GREEN_S_MIN, SAM2_GREEN_V_MIN,
    SAM2_YELLOW_H_MIN, SAM2_YELLOW_S_MIN, SAM2_YELLOW_V_MIN,
    SAM2_BROWN_H_MIN, SAM2_BROWN_H_MAX,
    SAM2_BROWN_S_MIN, SAM2_BROWN_V_MIN,
    SAM2_QA_MIN_COVERAGE, SAM2_QA_MAX_COVERAGE,
    SAM2_QA_MIN_CONFIDENCE, SAM2_QA_MIN_ASPECT_RATIO,
    SAM2_QA_MAX_REJECT_RATE,
    VALID_EXTENSIONS,
    YOLO_WEIGHTS_BEST, YOLO_CONF_THRESHOLD, YOLO_IOU_NMS,
    YOLO_QA_CALIB_FILE,
)

# ── QA thresholds (v3: calibrated confidence if available) ───────────────────
_QA_MIN_COVERAGE       = 0.02   # 99.2% of coverage_low rejections were 2–3% (YOLO-confirmed leaf); 3% was too strict
_QA_MAX_COVERAGE       = 1.00   # no hard cap — full-frame close-up leaves are valid; caught by confidence gate below
_QA_MAX_COVERAGE_BOX   = 1.00   # same: no hard cap for box-relative coverage
_QA_HIGH_COVERAGE_THR  = 0.97   # above this, apply elevated confidence gate instead of outright rejection
_QA_HIGH_COVERAGE_CONF = 0.80   # min mean confidence required when coverage >= _QA_HIGH_COVERAGE_THR
_QA_MIN_ASPECT         = 1.00   # effectively disabled: all aspect_low rejections were overhead square shots (1.000–1.008)

# Runtime YOLO box quality gates (applied in get_yolo_leaf_box).
# These are intentional runtime overrides — config.py is left unchanged.
_YOLO_MIN_CONF_FOR_BOX       = 0.35
_YOLO_MIN_CONF_FOR_LARGE_BOX = 0.50
_YOLO_MAX_BOX_AREA_FRAC      = 0.98
_YOLO_LARGE_BOX_AREA_THRESH  = 0.70

# ── Throughput / GPU-utilisation constants ────────────────────────────────────
# Tuned for 8 GB VRAM (RTX 5060 / RTX 3070 / RTX 4060).
# SAM2-Hiera-Large at bfloat16 uses ~3.2 GB model weights.
# Remaining ~4.3 GB headroom is filled by encoder activations for the batch.
#
# _YOLO_BATCH_SIZE  — images per YOLO GPU forward pass.
# _SAM2_ENCODE_BATCH — images encoded SIMULTANEOUSLY by SAM2's ViT encoder.
#   This is the main lever for VRAM utilisation: each additional image in the
#   encoder batch costs ~300-500 MB peak activation memory.
#   8 images  → ~5.5-7 GB total VRAM → target 85-100 % utilisation on 8 GB.
#   Lower to 4 if you see OOM errors.
# _PREFETCH_WORKERS — CPU threads loading images while the GPU is busy.
# _VRAM_FRACTION    — fraction of GPU memory reserved for this process.
# _VRAM_CLEAR_EVERY — torch.cuda.empty_cache() interval (images processed).
#
# DO NOT add torch.compile or cudnn.benchmark here:
#   torch.compile introduces graph breaks that LOWER GPU utilisation on
#   models with dynamic control flow (SAM2 + YOLO + variable image sizes).
#   cudnn.benchmark runs algorithm searches on the first call of each new
#   input size — SAM2 sees many sizes, so the searches dominate early runtime.
_YOLO_BATCH_SIZE   = 25    # 25 × 20 = 500 exactly → progress prints at 500, 1000, 1500 …
_SAM2_ENCODE_BATCH = 8    # lower to 4 if OOM; raise to 16 if VRAM < 80 %
_PREFETCH_WORKERS  = 8
_SAM2_DTYPE        = torch.bfloat16 if torch.cuda.is_available() else None
_VRAM_CLEAR_EVERY  = 1000
_VRAM_FRACTION     = 0.95

# Confidence threshold: loaded from calibration file if present,
# else falls back to the v2 default of 0.65.
_QA_MIN_CONFIDENCE = 0.65   # default; overwritten below if calib file exists
_QA_CALIB_SOURCE   = "default"
try:
    import pandas as _pd
    _calib = _pd.read_csv(YOLO_QA_CALIB_FILE)
    _chosen = _calib[_calib["chosen"] == True]
    if not _chosen.empty:
        _QA_MIN_CONFIDENCE = float(_chosen.iloc[0]["conf_threshold"])
        _QA_CALIB_SOURCE   = "yolo_calibration"
except Exception:
    pass   # file missing or malformed — use default

# ── Yellow/necrotic HSV detection ranges (v2 addition) ────────────────────────
_YELLOW_H_MIN = SAM2_YELLOW_H_MIN    # 15
_YELLOW_H_MAX = 45                   # wider than SAM2_YELLOW_H_MAX (38)
_YELLOW_S_MIN = SAM2_YELLOW_S_MIN    # 40
_YELLOW_V_MIN = SAM2_YELLOW_V_MIN    # 80

# ── Brown/necrotic HSV detection range (v3 addition) ─────────────────────────
_BROWN_H_MIN = SAM2_BROWN_H_MIN    # 5
_BROWN_H_MAX = SAM2_BROWN_H_MAX    # 20
_BROWN_S_MIN = SAM2_BROWN_S_MIN    # 30
_BROWN_V_MIN = SAM2_BROWN_V_MIN    # 50

_MIN_TISSUE_FRACTION        = 0.05
_MIN_TISSUE_FRACTION_IN_BOX = 0.01


# ══════════════════════════════════════════════════════════════════════════════
# YOLO-GUIDED PROMPTING  (v3)
# ══════════════════════════════════════════════════════════════════════════════

_yolo_model      = None
_QA_CALIB_SOURCE = "default"


def load_yolo_detector():
    global _yolo_model
    if not YOLO_WEIGHTS_BEST.exists():
        return None
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(str(YOLO_WEIGHTS_BEST))
        print(f"  YOLO detector loaded: {YOLO_WEIGHTS_BEST}")
        return _yolo_model
    except ImportError:
        print("  [INFO] ultralytics not installed — YOLO prompting disabled.")
        return None
    except Exception as e:
        print(f"  [WARN] Could not load YOLO weights: {e}")
        return None


def _extract_box_from_result(result, img_rgb: np.ndarray):
    """Apply quality gates to one YOLO result; return best box or None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy  = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    best  = int(np.argmax(confs))
    if confs[best] < _YOLO_MIN_CONF_FOR_BOX:
        return None
    x1, y1, x2, y2 = [int(v) for v in xyxy[best]]
    h_img, w_img = img_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return None
    box_area_frac = ((x2 - x1) * (y2 - y1)) / max(h_img * w_img, 1)
    if box_area_frac >= _YOLO_LARGE_BOX_AREA_THRESH:
        if confs[best] < _YOLO_MIN_CONF_FOR_LARGE_BOX:
            return None
    if box_area_frac > _YOLO_MAX_BOX_AREA_FRAC:
        return None
    return (x1, y1, x2, y2)


def get_yolo_leaf_box(img_rgb: np.ndarray):
    """Single-image YOLO inference (fallback)."""
    if _yolo_model is None:
        return None
    try:
        results = _yolo_model(img_rgb, conf=YOLO_CONF_THRESHOLD,
                              iou=YOLO_IOU_NMS, verbose=False)
        return _extract_box_from_result(results[0], img_rgb)
    except Exception:
        return None


def get_yolo_boxes_batch(imgs_rgb: list) -> list:
    """
    Batch YOLO inference — _YOLO_BATCH_SIZE images in one GPU forward pass.
    Returns a parallel list of boxes (or None for each image).
    """
    if _yolo_model is None:
        return [None] * len(imgs_rgb)
    valid = [(i, img) for i, img in enumerate(imgs_rgb) if img is not None]
    if not valid:
        return [None] * len(imgs_rgb)
    try:
        batch_imgs = [img for _, img in valid]
        results    = _yolo_model(batch_imgs, conf=YOLO_CONF_THRESHOLD,
                                 iou=YOLO_IOU_NMS, verbose=False)
        out = [None] * len(imgs_rgb)
        for (orig_idx, img), result in zip(valid, results):
            out[orig_idx] = _extract_box_from_result(result, img)
        return out
    except Exception:
        return [None] * len(imgs_rgb)


def get_leaf_centroid_in_box(img_rgb: np.ndarray, box: tuple):
    """
    v3 centroid logic constrained to pixels inside the YOLO box.
    Returns full-image (cx, cy) and strategy string.
    """
    x1, y1, x2, y2 = box
    roi_rgb = img_rgb[y1:y2, x1:x2]

    if roi_rgb.size == 0:
        centroid, strategy = get_leaf_centroid(img_rgb)
        return centroid, "box_degenerate_" + strategy

    roi_hsv = to_hsv(roi_rgb)
    h_ch, s_ch, v_ch = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]
    total_px = roi_rgb.shape[0] * roi_rgb.shape[1]

    green_mask = (
        (h_ch >= SAM2_GREEN_H_MIN) & (h_ch <= SAM2_GREEN_H_MAX) &
        (s_ch >= SAM2_GREEN_S_MIN) & (v_ch >= SAM2_GREEN_V_MIN)
    ).astype(np.uint8)
    yellow_mask = (
        (h_ch >= _YELLOW_H_MIN) & (h_ch <= _YELLOW_H_MAX) &
        (s_ch >= _YELLOW_S_MIN) & (v_ch >= _YELLOW_V_MIN)
    ).astype(np.uint8)
    brown_mask = (
        (h_ch >= _BROWN_H_MIN) & (h_ch <= _BROWN_H_MAX) &
        (s_ch >= _BROWN_S_MIN) & (v_ch >= _BROWN_V_MIN)
    ).astype(np.uint8)

    thr = _MIN_TISSUE_FRACTION_IN_BOX
    has_green  = green_mask.sum()  / total_px >= thr
    has_yellow = yellow_mask.sum() / total_px >= thr
    has_brown  = brown_mask.sum()  / total_px >= thr

    if has_green or has_yellow or has_brown:
        parts, label_parts = [], []
        if has_green:  parts.append(green_mask);  label_parts.append("green")
        if has_yellow: parts.append(yellow_mask); label_parts.append("yellow")
        if has_brown:  parts.append(brown_mask);  label_parts.append("brown")
        if len(parts) == 1:
            tissue, strategy = parts[0], label_parts[0] + "_centroid"
        else:
            tissue   = np.clip(sum(parts), 0, 1).astype(np.uint8)
            strategy = "combined_centroid"
    else:
        cx = x1 + (x2 - x1) // 2
        cy = y1 + (y2 - y1) // 2
        return (cx, cy), "box_center_fallback"

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(tissue, 8)
    if num_labels < 2:
        cx = x1 + (x2 - x1) // 2
        cy = y1 + (y2 - y1) // 2
        return (cx, cy), "box_center_fallback"

    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx = x1 + int(centroids[best_label][0])
    cy = y1 + int(centroids[best_label][1])
    return (cx, cy), strategy


# ══════════════════════════════════════════════════════════════════════════════
# SAM2 LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_sam2(checkpoint: Path, config: str):
    """
    Load SAM2 model. Requires Meta's segment-anything-2 package.
    Install: pip install git+https://github.com/facebookresearch/segment-anything-2
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model     = build_sam2(config, str(checkpoint), device="cuda")
        predictor = SAM2ImagePredictor(model)
        # Probe whether the batch API is available (SAM2 >= v1.0 public release)
        has_batch = (hasattr(predictor, "set_image_batch") and
                     hasattr(predictor, "predict_batch"))
        print(f"  SAM2 loaded. Batch encode API: {'YES' if has_batch else 'NO (sequential fallback)'}")
        return predictor, has_batch
    except ImportError:
        raise ImportError(
            "SAM2 not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything-2\n"
            "  and download weights to sam2/sam2_hiera_large.pt"
        )


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-PROMPTING  (v2 — disease-aware, multi-point foreground)
# ══════════════════════════════════════════════════════════════════════════════

def get_leaf_centroid(img_rgb: np.ndarray) -> tuple[tuple[int, int], str]:
    """
    Find the centroid of the dominant leaf-tissue region (HSV-only fallback).

    Detection priority (v3 — adds brown MLN necrotic tissue):
      1. Combined green + yellow + brown mask (any qualifying tissue)
      2. Green-only  (clearly healthy leaf)
      3. Yellow-only (chlorotic/MSV-streaked tissue)
      4. Brown-only  (heavily necrotic MLN — dark amber dead tissue)
      5. Center of image (last resort — still lets SAM2 attempt segmentation)
    """
    img_hsv  = to_hsv(img_rgb)
    h, s, v  = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    total_px = img_rgb.shape[0] * img_rgb.shape[1]

    green_mask = (
        (h >= SAM2_GREEN_H_MIN) & (h <= SAM2_GREEN_H_MAX) &
        (s >= SAM2_GREEN_S_MIN) & (v >= SAM2_GREEN_V_MIN)
    ).astype(np.uint8)
    yellow_mask = (
        (h >= _YELLOW_H_MIN) & (h <= _YELLOW_H_MAX) &
        (s >= _YELLOW_S_MIN) & (v >= _YELLOW_V_MIN)
    ).astype(np.uint8)
    brown_mask = (
        (h >= _BROWN_H_MIN) & (h <= _BROWN_H_MAX) &
        (s >= _BROWN_S_MIN) & (v >= _BROWN_V_MIN)
    ).astype(np.uint8)

    has_green  = int(green_mask.sum())  / total_px >= _MIN_TISSUE_FRACTION
    has_yellow = int(yellow_mask.sum()) / total_px >= _MIN_TISSUE_FRACTION
    has_brown  = int(brown_mask.sum())  / total_px >= _MIN_TISSUE_FRACTION

    if has_green or has_yellow or has_brown:
        parts, label_parts = [], []
        if has_green:  parts.append(green_mask);  label_parts.append("green")
        if has_yellow: parts.append(yellow_mask); label_parts.append("yellow")
        if has_brown:  parts.append(brown_mask);  label_parts.append("brown")
        if len(parts) == 1:
            tissue_mask = parts[0]
            strategy    = label_parts[0] + "_centroid"
        else:
            tissue_mask = np.clip(sum(parts), 0, 1).astype(np.uint8)
            strategy    = "combined_centroid"
    else:
        cy = img_rgb.shape[0] // 2
        cx = img_rgb.shape[1] // 2
        return (cx, cy), "center_fallback"

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        tissue_mask, connectivity=8)
    if num_labels < 2:
        cy = img_rgb.shape[0] // 2
        cx = img_rgb.shape[1] // 2
        return (cx, cy), "center_fallback"

    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx = int(centroids[best_label][0])
    cy = int(centroids[best_label][1])
    return (cx, cy), strategy


def build_prompts(
    img_rgb: np.ndarray,
    centroid: tuple[int, int],
    yolo_box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build SAM2 point prompts (horizontal spread inside box, or vertical axis)."""
    h, w   = img_rgb.shape[:2]
    pad    = 10
    cx, cy = centroid

    if yolo_box is not None:
        bx1, by1, bx2, by2 = yolo_box
        bw      = max(bx2 - bx1, 1)
        left_x  = bx1 + bw // 3
        right_x = bx2 - bw // 3
        fg_points = np.array([
            [cx,      cy],
            [left_x,  cy],
            [right_x, cy],
        ], dtype=np.float32)
        margin = 5
        bg_candidates = [
            [w // 2,  pad    ],
            [w // 2,  h - pad],
            [pad,     h // 2 ],
            [w - pad, h // 2 ],
        ]
        bg_outside = [
            pt for pt in bg_candidates
            if not (bx1 - margin <= pt[0] <= bx2 + margin and
                    by1 - margin <= pt[1] <= by2 + margin)
        ]
        if not bg_outside:
            bg_outside = bg_candidates
        bg_points = np.array(bg_outside, dtype=np.float32)
        bg_labels = np.zeros(len(bg_points), dtype=np.int32)
    else:
        upper_y = (0 + cy) // 2
        lower_y = (cy + h) // 2
        fg_points = np.array([
            [cx, cy     ],
            [cx, upper_y],
            [cx, lower_y],
        ], dtype=np.float32)
        bg_points = np.array([
            [pad,     pad    ],
            [w - pad, pad    ],
            [pad,     h - pad],
            [w - pad, h - pad],
        ], dtype=np.float32)
        bg_labels = np.array([0, 0, 0, 0], dtype=np.int32)

    fg_labels  = np.ones(len(fg_points), dtype=np.int32)
    all_points = np.vstack([fg_points, bg_points])
    all_labels = np.concatenate([fg_labels, bg_labels])
    return all_points, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# QA FILTERS  (v2 — relaxed thresholds)
# ══════════════════════════════════════════════════════════════════════════════

def qa_check(
    prob_map: np.ndarray,
    yolo_box: tuple[int, int, int, int] | None = None,
) -> tuple[bool, str]:
    """Apply three QA filters to a SAM2 probability map."""
    h_img, w_img = prob_map.shape[:2]
    binary = (prob_map >= 0.5).astype(np.uint8)
    fg_px  = int(binary.sum())

    _is_large_box = False
    if yolo_box is not None:
        bx1, by1, bx2, by2 = yolo_box
        box_area      = max((bx2 - bx1) * (by2 - by1), 1)
        img_area      = max(h_img * w_img, 1)
        fg_in_box     = int(binary[by1:by2, bx1:bx2].sum())
        _is_large_box = (box_area / img_area) >= _YOLO_LARGE_BOX_AREA_THRESH
        # For large boxes the box-relative coverage is misleadingly tiny
        # (a 2 % whole-image leaf in an 80 %-of-image box = only 2.5 % box-coverage).
        # Switch to whole-image coverage so the threshold means the same thing
        # regardless of how much of the frame YOLO's box happens to cover.
        if _is_large_box:
            coverage = fg_px / img_area
        else:
            coverage = fg_in_box / box_area
    else:
        coverage = fg_px / max(h_img * w_img, 1)

    if coverage < _QA_MIN_COVERAGE:
        return False, f"coverage_low:{coverage:.3f}"

    if fg_px == 0:
        return False, "no_foreground"
    mean_conf = float(prob_map[binary == 1].mean())

    # High-coverage gate: instead of a hard ceiling, apply a stricter confidence
    # requirement when coverage is very high.  A genuine close-up leaf filling the
    # frame will produce tight, high-confidence SAM2 logits (mean_conf ≥ 0.80).
    # A runaway blob that hallucinates the whole background tends to have lower
    # edge confidence even when binary coverage = 100%.
    if coverage >= _QA_HIGH_COVERAGE_THR:
        if mean_conf < _QA_HIGH_COVERAGE_CONF:
            return False, f"coverage_high_low_conf:{coverage:.3f}|conf:{mean_conf:.3f}"
        # Passes elevated gate — flag for easy spot-check in validate_masks.py
        return True, f"ok_high_coverage:{coverage:.3f}|conf:{mean_conf:.3f}"
    elif mean_conf < _QA_MIN_CONFIDENCE:
        return False, f"confidence_low:{mean_conf:.3f}"

    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
        mask_h = max(rmax - rmin + 1, 1)
        mask_w = max(cmax - cmin + 1, 1)
        aspect = max(mask_h, mask_w) / min(mask_h, mask_w)
        if yolo_box is not None:
            bx1, by1, bx2, by2 = yolo_box
            bh_box    = max(by2 - by1, 1)
            bw_box    = max(bx2 - bx1, 1)
            box_aspect = max(bh_box, bw_box) / min(bh_box, bw_box)
            if not (box_aspect < 1.15 or _is_large_box):
                if aspect < _QA_MIN_ASPECT:
                    return False, f"aspect_low:{aspect:.3f}"
        else:
            if aspect < _QA_MIN_ASPECT:
                return False, f"aspect_low:{aspect:.3f}"

    return True, "passed"


# ══════════════════════════════════════════════════════════════════════════════
# SAM2 BATCH INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_sam2_batch(predictor, items: list, autocast_ctx) -> list:
    """
    Run SAM2 on a mini-batch using set_image_batch + predict_batch.

    This is the PRIMARY performance path for 8 GB GPUs:
      - set_image_batch encodes _SAM2_ENCODE_BATCH images simultaneously,
        filling 5-7 GB VRAM and keeping tensor cores saturated at ~100%.
      - predict_batch runs all mask decoders back-to-back in the same context.

    items: list of dicts with keys: img, points, labels, box (or None)

    Returns: list of (prob_map: np.ndarray | None, error: str | None)
    """
    imgs   = [it["img"] for it in items]
    pts_b  = [it["points"] for it in items]
    lbl_b  = [it["labels"] for it in items]
    box_b  = [
        np.array(it["box"], dtype=np.float32) if it["box"] is not None else None
        for it in items
    ]
    # If no image has a box, pass None to predict_batch (avoids empty-list edge case)
    box_arg = box_b if any(b is not None for b in box_b) else None

    with torch.inference_mode(), autocast_ctx:
        predictor.set_image_batch(imgs)
        masks_all, scores_all, logits_all = predictor.predict_batch(
            point_coords_batch=pts_b,
            point_labels_batch=lbl_b,
            box_batch=box_arg,
            multimask_output=True,
        )

    results = []
    for scores, logits in zip(scores_all, logits_all):
        best = int(np.argmax(scores))
        pm   = (1.0 / (1.0 + np.exp(-logits[best]))).squeeze().astype(np.float32)
        results.append((pm, None))
    return results


def _run_sam2_sequential(predictor, items: list, autocast_ctx) -> list:
    """
    Fallback: sequential set_image + predict per image.
    Used when set_image_batch / predict_batch are unavailable.
    """
    results = []
    for it in items:
        try:
            with torch.inference_mode(), autocast_ctx:
                predictor.set_image(it["img"])
                kwargs = dict(
                    point_coords=it["points"],
                    point_labels=it["labels"],
                    multimask_output=True,
                )
                if it["box"] is not None:
                    kwargs["box"] = np.array(it["box"], dtype=np.float32)
                masks, scores, logits = predictor.predict(**kwargs)
            best = int(np.argmax(scores))
            pm   = (1.0 / (1.0 + np.exp(-logits[best]))).squeeze().astype(np.float32)
            results.append((pm, None))
        except Exception as exc:
            results.append((None, str(exc)[:120]))
    return results


def _infer_sam2(predictor, items: list, autocast_ctx, use_batch_api: bool) -> list:
    """
    Route to batch or sequential SAM2 inference.
    If batch API raises (e.g. due to SAM2 version mismatch), falls back
    automatically to sequential so the run never aborts.
    """
    if use_batch_api and len(items) > 0:
        try:
            return _run_sam2_batch(predictor, items, autocast_ctx)
        except Exception as e:
            print(f"\n  [WARN] SAM2 batch API failed ({e}); falling back to sequential.")
    return _run_sam2_sequential(predictor, items, autocast_ctx)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_global = time.time()

    print("=" * 72)
    print("  Yellow MAIze | Phase 2: SAM2 Tier 1 Masking  [v3]")
    print("=" * 72)
    print("  QA thresholds (v3):")
    print(f"    Coverage     : {_QA_MIN_COVERAGE:.0%} – {_QA_MAX_COVERAGE:.0%}  "
          f"(box-relative ceiling: {_QA_MAX_COVERAGE_BOX:.0%})")
    print(f"    Confidence   : ≥ {_QA_MIN_CONFIDENCE:.2f}  (source: {_QA_CALIB_SOURCE})")
    print(f"    Aspect ratio : ≥ {_QA_MIN_ASPECT:.2f}  (v1 was 1.20)")
    print()

    # ── GPU setup ─────────────────────────────────────────────────────────────
    # TF32: ~8× faster matmuls vs FP32 on Ada Lovelace with negligible
    # numeric difference. Safe for both SAM2 and YOLO.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_VRAM_FRACTION)
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU VRAM        : {total/1024**3:.1f} GB total  "
              f"({free/1024**3:.1f} GB free before model load)")
        print(f"  VRAM reserved   : {_VRAM_FRACTION*100:.0f}% of device")
    autocast_ctx = (torch.autocast("cuda", dtype=_SAM2_DTYPE)
                    if _SAM2_DTYPE is not None else nullcontext())
    print(f"  SAM2 dtype      : {'bfloat16 (≈40% faster encoder)' if _SAM2_DTYPE else 'float32 (no CUDA)'}")
    print(f"  YOLO batch size : {_YOLO_BATCH_SIZE} images / GPU pass")
    print(f"  SAM2 enc. batch : {_SAM2_ENCODE_BATCH} images / encoder forward pass")
    print(f"  Prefetch threads: {_PREFETCH_WORKERS}")
    print()

    # ── Preflight checks ──────────────────────────────────────────────────────
    if not TIER1_RAW_DIR.is_dir() or not any(TIER1_RAW_DIR.iterdir()):
        print(f"[FATAL] {TIER1_RAW_DIR} is empty. Run sample_15000.py first.")
        return
    if not SAM2_CHECKPOINT.exists():
        print(f"[FATAL] SAM2 checkpoint not found: {SAM2_CHECKPOINT}")
        print("  Download: https://dl.fbaipublicfiles.com/segment_anything_v2/")
        return
    TIER1_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    _yolo_available = load_yolo_detector() is not None
    print(f"  YOLO prompting  : {'ENABLED' if _yolo_available else 'DISABLED (HSV-only fallback)'}")
    predictor, _sam2_batch_api = load_sam2(SAM2_CHECKPOINT, SAM2_CONFIG)
    print()

    # ── Collect images ────────────────────────────────────────────────────────
    images = sorted([
        p for p in TIER1_RAW_DIR.iterdir()
        if p.suffix.lower() in [e.lower() for e in VALID_EXTENSIONS]
    ])
    if not images:
        print(f"[FATAL] No valid images found in {TIER1_RAW_DIR}.")
        return
    print(f"  Images to process: {len(images):,}\n")

    # ── Processing loop ───────────────────────────────────────────────────────
    #
    # Per YOLO chunk (_YOLO_BATCH_SIZE images):
    #   1. NEXT chunk images pre-load on CPU threads (hidden behind GPU work).
    #   2. YOLO runs on the full chunk in one batched GPU pass.
    #   3. Centroids + prompts computed (fast numpy, CPU).
    #   4. SAM2 encodes + predicts in mini-batches of _SAM2_ENCODE_BATCH:
    #        set_image_batch  → all ViT encoders run simultaneously
    #        predict_batch    → all mask decoders run back-to-back
    #      This keeps 5-7 GB VRAM occupied and tensor cores near 100%.
    #   5. QA + save (CPU).

    qa_rows         = []
    n_passed        = 0
    n_rejected      = 0
    reject_reasons  = defaultdict(int)
    strategy_counts = defaultdict(int)
    t_start          = time.time()
    i_global         = -1
    _next_print_at   = 500  # print when done crosses 500, 1000, 1500 …

    executor      = ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS)
    chunk_paths   = images[:_YOLO_BATCH_SIZE]
    chunk_futures = [executor.submit(load_image_rgb, p) for p in chunk_paths]
    chunk_start   = 0

    try:
        while chunk_start < len(images):
            chunk_imgs = [f.result() for f in chunk_futures]

            # Pre-load NEXT chunk while this one is on the GPU
            next_start    = chunk_start + _YOLO_BATCH_SIZE
            next_paths    = images[next_start : next_start + _YOLO_BATCH_SIZE]
            next_futures  = [executor.submit(load_image_rgb, p) for p in next_paths]

            # ── 1. Batch YOLO ─────────────────────────────────────────────────
            yolo_boxes_chunk = (get_yolo_boxes_batch(chunk_imgs)
                                if _yolo_available else [None] * len(chunk_imgs))

            # ── 2. Centroids + prompts (CPU) ──────────────────────────────────
            # Build a list of "items" for valid images; track failed ones separately.
            valid_items  = []   # will be passed to SAM2 in mini-batches
            failed_items = []   # load errors

            for img_path, img_rgb, yolo_box in zip(
                    chunk_paths, chunk_imgs, yolo_boxes_chunk):
                i_global += 1
                stem     = img_path.stem
                category = stem.split("_")[0]

                if img_rgb is None:
                    failed_items.append({
                        "path": img_path, "stem": stem, "category": category,
                        "reason": "corrupt_or_truncated",
                    })
                    continue

                prompt_mode = "hsv_fallback"
                if yolo_box is not None:
                    centroid, strategy = get_leaf_centroid_in_box(img_rgb, yolo_box)
                    strategy    = "yolo_" + strategy
                    prompt_mode = "yolo"
                else:
                    centroid, strategy = get_leaf_centroid(img_rgb)

                points, labels = build_prompts(img_rgb, centroid, yolo_box=yolo_box)
                valid_items.append({
                    "path":        img_path,
                    "stem":        stem,
                    "category":    category,
                    "img":         img_rgb,
                    "points":      points,
                    "labels":      labels,
                    "box":         yolo_box,
                    "strategy":    strategy,
                    "prompt_mode": prompt_mode,
                })

            # Record load errors
            for fi in failed_items:
                qa_rows.append({
                    "filename":        fi["path"].name,
                    "category":        fi["category"],
                    "status":          "load_error",
                    "reason":          fi["reason"],
                    "prompt_strategy": "n/a",
                    "prompt_mode":     "n/a",
                    "yolo_box":        "n/a",
                    "coverage":        -1,
                    "mean_conf":       -1,
                })
                reject_reasons[fi["reason"]] += 1
                n_rejected += 1

            # ── 3. SAM2 in mini-batches of _SAM2_ENCODE_BATCH ─────────────────
            for batch_start in range(0, len(valid_items), _SAM2_ENCODE_BATCH):
                mini_batch = valid_items[batch_start : batch_start + _SAM2_ENCODE_BATCH]

                try:
                    sam2_results = _infer_sam2(
                        predictor, mini_batch, autocast_ctx, _sam2_batch_api
                    )
                except Exception as exc:
                    # Whole mini-batch failed — mark all as sam2_error
                    for it in mini_batch:
                        qa_rows.append({
                            "filename":        it["path"].name,
                            "category":        it["category"],
                            "status":          "sam2_error",
                            "reason":          f"sam2_error:{str(exc)[:80]}",
                            "prompt_strategy": it["strategy"],
                            "prompt_mode":     it["prompt_mode"],
                            "yolo_box":        "n/a",
                            "coverage":        -1,
                            "mean_conf":       -1,
                        })
                        reject_reasons["sam2_error"] += 1
                        n_rejected += 1
                    continue

                # ── 4. QA + save ──────────────────────────────────────────────
                for it, (prob_map, err) in zip(mini_batch, sam2_results):
                    strategy_counts[it["strategy"]] += 1

                    if err is not None or prob_map is None:
                        qa_rows.append({
                            "filename":        it["path"].name,
                            "category":        it["category"],
                            "status":          "sam2_error",
                            "reason":          f"sam2_error:{err or 'unknown'}",
                            "prompt_strategy": it["strategy"],
                            "prompt_mode":     it["prompt_mode"],
                            "yolo_box":        "n/a",
                            "coverage":        -1,
                            "mean_conf":       -1,
                        })
                        reject_reasons["sam2_error"] += 1
                        n_rejected += 1
                        continue

                    passed, reason = qa_check(prob_map, yolo_box=it["box"])
                    binary    = (prob_map >= 0.5).astype(np.uint8)
                    coverage  = float(binary.sum()) / (prob_map.shape[0] * prob_map.shape[1])
                    mean_conf = float(prob_map[binary == 1].mean()) if binary.sum() > 0 else 0.0
                    box_str   = "{},{},{},{}".format(*it["box"]) if it["box"] else "n/a"

                    if not passed:
                        reason_key = reason.split(":")[0]
                        reject_reasons[reason_key] += 1
                        qa_rows.append({
                            "filename":        it["path"].name,
                            "category":        it["category"],
                            "status":          "rejected",
                            "reason":          reason,
                            "prompt_strategy": it["strategy"],
                            "prompt_mode":     it["prompt_mode"],
                            "yolo_box":        box_str,
                            "coverage":        round(coverage, 4),
                            "mean_conf":       round(mean_conf, 4),
                        })
                        n_rejected += 1
                        continue

                    np.save(str(TIER1_MASKS_DIR / f"{it['stem']}_softmask.npy"), prob_map)
                    cv2.imwrite(
                        str(TIER1_MASKS_DIR / f"{it['stem']}_mask.png"),
                        (binary * 255).astype(np.uint8),
                    )
                    n_passed += 1
                    qa_rows.append({
                        "filename":        it["path"].name,
                        "category":        it["category"],
                        "status":          "passed",
                        "reason":          "ok",
                        "prompt_strategy": it["strategy"],
                        "prompt_mode":     it["prompt_mode"],
                        "yolo_box":        box_str,
                        "coverage":        round(coverage, 4),
                        "mean_conf":       round(mean_conf, 4),
                    })

            # ── Periodic VRAM defrag ──────────────────────────────────────────
            if torch.cuda.is_available() and (i_global + 1) % _VRAM_CLEAR_EVERY == 0:
                torch.cuda.empty_cache()

            chunk_start   = next_start
            chunk_paths   = next_paths
            chunk_futures = next_futures

            # ── Progress every ~500 images (+ always on last chunk) ─────────────
            # _next_print_at advances by exactly 500 each time, so triggers
            # align to 500, 1000, 1500 … regardless of chunk-boundary drift.
            done = i_global + 1
            if done >= _next_print_at or chunk_start >= len(images):
                while _next_print_at <= done:   # skip missed boundaries if chunk > 500
                    _next_print_at += 500
                elapsed     = time.time() - t_start
                rate        = done / max(elapsed, 1e-6)
                eta         = (len(images) - done) / max(rate, 1e-6)
                reject_rate = n_rejected / max(done, 1)
                vram_str    = ""
                if torch.cuda.is_available():
                    vf, vt = torch.cuda.mem_get_info()
                    vram_str = f"  |  VRAM {(vt-vf)/1024**3:.1f}/{vt/1024**3:.1f} GB"
                print(
                    f"  [{done:>6}/{len(images)}]  "
                    f"passed {n_passed:,}  |  "
                    f"rejected {n_rejected:,} ({reject_rate*100:.1f}%)  |  "
                    f"ETA {eta / 60:.1f} min{vram_str}"
                )

    finally:
        executor.shutdown(wait=False)

    # ── Write QA report ───────────────────────────────────────────────────────
    if qa_rows:
        with open(TIER1_QA_REPORT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(qa_rows[0].keys()))
            writer.writeheader()
            writer.writerows(qa_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    total       = n_passed + n_rejected
    reject_rate = n_rejected / max(total, 1)
    elapsed_min = (time.time() - t_global) / 60

    print(f"\n{'─' * 72}")
    print(f"  Total processed : {total:,}")
    print(f"  Passed QA       : {n_passed:,}  ({(1 - reject_rate) * 100:.1f}%)")
    print(f"  Rejected        : {n_rejected:,}  ({reject_rate * 100:.1f}%)")
    print(f"  Elapsed         : {elapsed_min:.1f} min")
    print(f"  QA report       : {TIER1_QA_REPORT}")

    if reject_reasons:
        print(f"\n  Rejection breakdown:")
        print(f"  {'Reason':<32} {'Count':>7}  {'%':>6}")
        print(f"  {'─' * 32} {'─' * 7}  {'─' * 6}")
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<32} {count:>7,}  {count / max(total, 1) * 100:>5.1f}%")

    if strategy_counts:
        print(f"\n  Prompt strategy breakdown:")
        print(f"  {'Strategy':<32} {'Count':>7}  {'%':>6}")
        print(f"  {'─' * 32} {'─' * 7}  {'─' * 6}")
        for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f"  {strat:<32} {count:>7,}  {count / max(total, 1) * 100:>5.1f}%")

    print()
    if reject_rate > SAM2_QA_MAX_REJECT_RATE:
        print(f"  [WARN] Rejection rate {reject_rate * 100:.1f}% exceeds "
              f"{SAM2_QA_MAX_REJECT_RATE * 100:.0f}% target.")
        print("         Check the rejection breakdown above and review the QA report.")
    else:
        print(f"  [OK] Rejection rate within acceptable range "
              f"({reject_rate * 100:.1f}% ≤ {SAM2_QA_MAX_REJECT_RATE * 100:.0f}%).")

    yolo_ct = sum(v for k, v in strategy_counts.items() if k.startswith("yolo_"))
    hsv_ct  = sum(v for k, v in strategy_counts.items() if not k.startswith("yolo_"))
    if yolo_ct + hsv_ct > 0:
        print(f"\n  Prompting mode breakdown:")
        print(f"    YOLO-guided : {yolo_ct:>7,}  ({yolo_ct/(yolo_ct+hsv_ct)*100:.1f}%)")
        print(f"    HSV fallback: {hsv_ct:>7,}  ({hsv_ct/(yolo_ct+hsv_ct)*100:.1f}%)")

    print(f"\n  NEXT STEP: python validate_masks.py  (review QA report visually)")
    print(f"  THEN     : python train_teacher.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
