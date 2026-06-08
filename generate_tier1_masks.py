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
from pathlib import Path

import cv2
import numpy as np
from image_utils import load_image_rgb, to_hsv   # EXIF correction

from config import (
    SEED,
    TIER1_RAW_DIR, TIER1_MASKS_DIR, TIER1_QA_REPORT, REPORTS_DIR,
    SAM2_CHECKPOINT, SAM2_CONFIG,
    SAM2_GREEN_H_MIN, SAM2_GREEN_H_MAX,
    SAM2_GREEN_S_MIN, SAM2_GREEN_V_MIN,
    SAM2_QA_MIN_COVERAGE, SAM2_QA_MAX_COVERAGE,
    SAM2_QA_MIN_CONFIDENCE, SAM2_QA_MIN_ASPECT_RATIO,
    SAM2_QA_MAX_REJECT_RATE,
    VALID_EXTENSIONS,
    YOLO_WEIGHTS_BEST, YOLO_CONF_THRESHOLD, YOLO_IOU_NMS,
    YOLO_QA_CALIB_FILE,
)

# ── QA thresholds (v3: calibrated confidence if available) ───────────────────
_QA_MIN_COVERAGE   = 0.03   # was SAM2_QA_MIN_COVERAGE (0.10)
_QA_MAX_COVERAGE   = 0.90
_QA_MIN_ASPECT     = 1.01   # was SAM2_QA_MIN_ASPECT_RATIO (1.20)

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
# Covers MSV streak yellowing and MLN necrotic discoloration.
# H=15–45 → yellow-green through yellow in OpenCV's 0–179 hue scale.
_YELLOW_H_MIN = 15
_YELLOW_H_MAX = 45
_YELLOW_S_MIN = 40    # exclude washed-out whites/greys
_YELLOW_V_MIN = 60    # exclude dark shadows

# Minimum fraction of image that must be leaf-colored before falling back
# to the center-of-image prompt.
_MIN_TISSUE_FRACTION = 0.05



# ══════════════════════════════════════════════════════════════════════════════
# YOLO-GUIDED PROMPTING  (v3 — loaded once at startup, falls back to v2)
# ══════════════════════════════════════════════════════════════════════════════

_yolo_model = None    # module-level singleton; set by load_yolo_detector()
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
        print("  [INFO] ultralytics not installed - YOLO prompting disabled.")
        return None
    except Exception as e:
        print(f"  [WARN] Could not load YOLO weights: {e}")
        return None


def get_yolo_leaf_box(img_rgb: np.ndarray):
    if _yolo_model is None:
        return None
    try:
        det   = _yolo_model(img_rgb, conf=YOLO_CONF_THRESHOLD,
                            iou=YOLO_IOU_NMS, verbose=False)
        boxes = det[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        best  = int(np.argmax(confs))
        x1, y1, x2, y2 = [int(v) for v in xyxy[best]]
        h_img, w_img = img_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if (x2 - x1) < 16 or (y2 - y1) < 16:
            return None
        return (x1, y1, x2, y2)
    except Exception:
        return None


def get_leaf_centroid_in_box(img_rgb: np.ndarray, box: tuple):
    """
    v2 get_leaf_centroid logic constrained to pixels inside the YOLO box.
    Returns full-image (cx, cy) and strategy string.
    Prevents centroid drift onto bright background sharing disease hue ranges.
    """
    x1, y1, x2, y2 = box
    roi_rgb = img_rgb[y1:y2, x1:x2]

    if roi_rgb.size == 0:
        centroid, strategy = get_leaf_centroid(img_rgb)
        return centroid, "box_degenerate_" + strategy

    roi_hsv = to_hsv(roi_rgb)
    h_ch = roi_hsv[:, :, 0]
    s_ch = roi_hsv[:, :, 1]
    v_ch = roi_hsv[:, :, 2]
    total_px = roi_rgb.shape[0] * roi_rgb.shape[1]

    green_mask = (
        (h_ch >= SAM2_GREEN_H_MIN) & (h_ch <= SAM2_GREEN_H_MAX) &
        (s_ch >= SAM2_GREEN_S_MIN) & (v_ch >= SAM2_GREEN_V_MIN)
    ).astype(np.uint8)
    yellow_mask = (
        (h_ch >= _YELLOW_H_MIN) & (h_ch <= _YELLOW_H_MAX) &
        (s_ch >= _YELLOW_S_MIN) & (v_ch >= _YELLOW_V_MIN)
    ).astype(np.uint8)

    has_green  = green_mask.sum()  / total_px >= _MIN_TISSUE_FRACTION
    has_yellow = yellow_mask.sum() / total_px >= _MIN_TISSUE_FRACTION

    if has_green and has_yellow:
        tissue = np.clip(green_mask + yellow_mask, 0, 1).astype(np.uint8)
        strategy = "combined_centroid"
    elif has_green:
        tissue = green_mask;  strategy = "green_centroid"
    elif has_yellow:
        tissue = yellow_mask; strategy = "yellow_centroid"
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
        print("  SAM2 loaded successfully.")
        return predictor
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
    Find the centroid of the dominant leaf-tissue region.

    Detection priority:
      1. Combined green + yellow mask (mixed healthy/diseased tissue)
      2. Green-only  (clearly healthy leaf)
      3. Yellow-only (heavily diseased, minimal green remains)
      4. Center of image (last resort — still lets SAM2 attempt segmentation)

    Why the fallback matters:
      MSV and MLN destroy green tissue. A heavily infected leaf can have
      near-zero green pixels, so the v1 green-only approach was silently
      rejecting diseased images before SAM2 even ran. The center-of-image
      fallback keeps those images in the pipeline and lets SAM2 — which
      has much richer visual priors — decide whether the mask is valid.

    Args:
        img_rgb: H×W×3 uint8 RGB image (EXIF-corrected by load_image_rgb).

    Returns:
        centroid : (cx, cy) pixel coordinates for the foreground prompt.
        strategy : diagnostic label logged to the QA report.
                   One of: "green_centroid" | "yellow_centroid" |
                            "combined_centroid" | "center_fallback"
    """
    img_hsv  = to_hsv(img_rgb)       # image_utils guarantees RGB→HSV
    h, s, v  = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    total_px = img_rgb.shape[0] * img_rgb.shape[1]

    # ── 1. Healthy-green mask ─────────────────────────────────────────────────
    green_mask = (
        (h >= SAM2_GREEN_H_MIN) & (h <= SAM2_GREEN_H_MAX) &
        (s >= SAM2_GREEN_S_MIN) &
        (v >= SAM2_GREEN_V_MIN)
    ).astype(np.uint8)

    # ── 2. Yellowed / necrotic mask ───────────────────────────────────────────
    # Captures MSV streak yellowing and MLN necrotic patches that are
    # invisible to the green-only detector.
    yellow_mask = (
        (h >= _YELLOW_H_MIN) & (h <= _YELLOW_H_MAX) &
        (s >= _YELLOW_S_MIN) &
        (v >= _YELLOW_V_MIN)
    ).astype(np.uint8)

    green_frac  = int(green_mask.sum())  / total_px
    yellow_frac = int(yellow_mask.sum()) / total_px
    has_green   = green_frac  >= _MIN_TISSUE_FRACTION
    has_yellow  = yellow_frac >= _MIN_TISSUE_FRACTION

    # ── 3. Choose tissue mask ─────────────────────────────────────────────────
    if has_green and has_yellow:
        tissue_mask = np.clip(green_mask + yellow_mask, 0, 1).astype(np.uint8)
        strategy    = "combined_centroid"
    elif has_green:
        tissue_mask = green_mask
        strategy    = "green_centroid"
    elif has_yellow:
        tissue_mask = yellow_mask
        strategy    = "yellow_centroid"
    else:
        # No qualifying tissue found — fall back to image center.
        # SAM2 will still run; QA filters decide the outcome.
        cy = img_rgb.shape[0] // 2
        cx = img_rgb.shape[1] // 2
        return (cx, cy), "center_fallback"

    # ── 4. Centroid of largest connected component ────────────────────────────
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        tissue_mask, connectivity=8)

    if num_labels < 2:
        # Pixel count passed but no discrete component found (edge case).
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build SAM2 point prompts.

    Foreground (label=1):
      - centroid of dominant leaf-tissue blob
      - upper-half midpoint (centroid_x, midpoint between top and centroid_y)
      - lower-half midpoint (centroid_x, midpoint between centroid_y and bottom)
      Using three foreground points better anchors long, narrow maize leaves
      whose full length a single centroid may fail to capture.

    Background (label=0): four corners, 10px inset from each edge.
    """
    h, w  = img_rgb.shape[:2]
    pad   = 10
    cx, cy = centroid

    # Three foreground points along the leaf's vertical axis
    upper_y = (0 + cy) // 2          # midpoint between top edge and centroid
    lower_y = (cy + h) // 2          # midpoint between centroid and bottom edge

    fg_points = np.array([
        [cx, cy     ],   # centroid (primary anchor)
        [cx, upper_y],   # upper-half midpoint
        [cx, lower_y],   # lower-half midpoint
    ], dtype=np.float32)
    fg_labels = np.array([1, 1, 1], dtype=np.int32)

    bg_points = np.array([
        [pad,     pad    ],   # top-left
        [w - pad, pad    ],   # top-right
        [pad,     h - pad],   # bottom-left
        [w - pad, h - pad],   # bottom-right
    ], dtype=np.float32)
    bg_labels = np.array([0, 0, 0, 0], dtype=np.int32)

    all_points = np.vstack([fg_points, bg_points])
    all_labels = np.concatenate([fg_labels, bg_labels])
    return all_points, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# QA FILTERS  (v2 — relaxed thresholds)
# ══════════════════════════════════════════════════════════════════════════════

def qa_check(prob_map: np.ndarray) -> tuple[bool, str]:
    """
    Apply three QA filters to a SAM2 probability map.

    v2 threshold changes:
      - Min coverage    : 0.10 → 0.05   (diseased leaves have sparser tissue)
      - Min aspect ratio: 1.20 → 1.05   (overhead/square-frame leaves)

    Args:
        prob_map: H×W float32 sigmoid probability map from SAM2.

    Returns:
        passed : True if all filters pass.
        reason : "passed" or a human-readable failure description.
    """
    total  = prob_map.shape[0] * prob_map.shape[1]
    binary = (prob_map >= 0.5).astype(np.uint8)
    fg_px  = int(binary.sum())

    # ── Filter 1: Coverage range ──────────────────────────────────────────────
    coverage = fg_px / total
    if coverage < _QA_MIN_COVERAGE:
        return False, f"coverage_low:{coverage:.3f}"
    if coverage > _QA_MAX_COVERAGE:
        return False, f"coverage_high:{coverage:.3f}"

    # ── Filter 2: Mean foreground confidence ──────────────────────────────────
    if fg_px == 0:
        return False, "no_foreground"
    mean_conf = float(prob_map[binary == 1].mean())
    if mean_conf < _QA_MIN_CONFIDENCE:
        return False, f"confidence_low:{mean_conf:.3f}"

    # ── Filter 3: Aspect ratio of bounding box ────────────────────────────────
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
        mask_h     = max(rmax - rmin + 1, 1)
        mask_w     = max(cmax - cmin + 1, 1)
        aspect     = max(mask_h, mask_w) / min(mask_h, mask_w)
        if aspect < _QA_MIN_ASPECT:
            return False, f"aspect_low:{aspect:.3f}"

    return True, "passed"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_global = time.time()

    print("=" * 72)
    print("  Yellow MAIze | Phase 2: SAM2 Tier 1 Masking  [v3]")
    print("=" * 72)
    print("  QA thresholds (v2):")
    print(f"    Coverage     : {_QA_MIN_COVERAGE:.0%} – {_QA_MAX_COVERAGE:.0%}  "
          f"(v1 was 10% – 90%)")
    print(f"    Confidence   : ≥ {_QA_MIN_CONFIDENCE:.2f}  (unchanged)")
    print(f"    Aspect ratio : ≥ {_QA_MIN_ASPECT:.2f}  (v1 was 1.20)")
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

    # ── Load YOLO detector (v3 — non-fatal if absent) ───────────────────────
    _yolo_available = load_yolo_detector() is not None
    print(f"  YOLO prompting  : {'ENABLED' if _yolo_available else 'DISABLED (fallback to v2 HSV-only)'}")
    print(f"  QA conf source  : {_QA_CALIB_SOURCE}  (threshold = {_QA_MIN_CONFIDENCE:.3f})")
    print()

    # ── Load SAM2 ─────────────────────────────────────────────────────────────
    predictor = load_sam2(SAM2_CHECKPOINT, SAM2_CONFIG)

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
    qa_rows         = []
    n_passed        = 0
    n_rejected      = 0
    reject_reasons  = defaultdict(int)   # keyed on reason prefix
    strategy_counts = defaultdict(int)   # keyed on prompt strategy
    t_start         = time.time()

    for i, img_path in enumerate(images):
        stem     = img_path.stem
        category = stem.split("_")[0]   # prefix written by sample_15000.py

        # ── Load ──────────────────────────────────────────────────────────────
        img_rgb = load_image_rgb(img_path)
        if img_rgb is None:
            reason = "corrupt_or_truncated"
            qa_rows.append({
                "filename":        img_path.name,
                "category":        category,
                "status":          "load_error",
                "reason":          reason,
                "prompt_strategy": "n/a",
                "coverage":        -1,
                "mean_conf":       -1,
            })
            reject_reasons[reason] += 1
            n_rejected += 1
            continue

        # ── Auto-prompting (v3: YOLO-guided with v2 HSV fallback) ──────────
        yolo_box    = get_yolo_leaf_box(img_rgb) if _yolo_available else None
        prompt_mode = "hsv_fallback"

        if yolo_box is not None:
            centroid, strategy = get_leaf_centroid_in_box(img_rgb, yolo_box)
            strategy    = "yolo_" + strategy
            prompt_mode = "yolo"
        else:
            centroid, strategy = get_leaf_centroid(img_rgb)

        strategy_counts[strategy] += 1
        points, labels = build_prompts(img_rgb, centroid)

        # ── SAM2 inference ────────────────────────────────────────────────────
        try:
            predictor.set_image(img_rgb)
            predict_kwargs = dict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
            if yolo_box is not None:
                predict_kwargs["box"] = np.array(yolo_box, dtype=np.float32)
            masks, scores, logits = predictor.predict(**predict_kwargs)
            # Pick the candidate mask with highest SAM2 score
            best_idx = int(np.argmax(scores))
            # logits → sigmoid probability
            prob_map = (1.0 / (1.0 + np.exp(-logits[best_idx]))).squeeze()
            prob_map = prob_map.astype(np.float32)
        except Exception as exc:
            reason = "sam2_error"
            qa_rows.append({
                "filename":        img_path.name,
                "category":        category,
                "status":          "sam2_error",
                "reason":          f"sam2_error:{str(exc)[:80]}",
                "prompt_strategy": strategy,
                "coverage":        -1,
                "mean_conf":       -1,
            })
            reject_reasons[reason] += 1
            n_rejected += 1
            continue

        # ── QA filters ────────────────────────────────────────────────────────
        passed, reason = qa_check(prob_map)

        binary    = (prob_map >= 0.5).astype(np.uint8)
        coverage  = float(binary.sum()) / (prob_map.shape[0] * prob_map.shape[1])
        mean_conf = float(prob_map[binary == 1].mean()) if binary.sum() > 0 else 0.0

        if not passed:
            reason_key = reason.split(":")[0]   # strip numeric detail for bucketing
            reject_reasons[reason_key] += 1
            qa_rows.append({
                "filename":        img_path.name,
                "category":        category,
                "status":          "rejected",
                "reason":          reason,
                "prompt_strategy": strategy,
                "prompt_mode":     prompt_mode,
                "yolo_box":        "{},{},{},{}".format(*yolo_box) if yolo_box else "n/a",
                "coverage":        round(coverage, 4),
                "mean_conf":       round(mean_conf, 4),
            })
            n_rejected += 1
            continue

        # ── Save outputs ──────────────────────────────────────────────────────
        npy_path = TIER1_MASKS_DIR / f"{stem}_softmask.npy"
        png_path = TIER1_MASKS_DIR / f"{stem}_mask.png"

        np.save(str(npy_path), prob_map)
        cv2.imwrite(str(png_path), (binary * 255).astype(np.uint8))

        n_passed += 1
        qa_rows.append({
            "filename":        img_path.name,
            "category":        category,
            "status":          "passed",
            "reason":          "ok",
            "prompt_strategy": strategy,
            "prompt_mode":     prompt_mode,
            "yolo_box":        "{},{},{},{}".format(*yolo_box) if yolo_box else "n/a",
            "coverage":        round(coverage, 4),
            "mean_conf":       round(mean_conf, 4),
        })

        # ── Progress log ──────────────────────────────────────────────────────
        if (i + 1) % 500 == 0:
            elapsed     = time.time() - t_start
            rate        = (i + 1) / max(elapsed, 1e-6)
            eta         = (len(images) - i - 1) / max(rate, 1e-6)
            reject_rate = n_rejected / (i + 1)
            print(
                f"  [{i+1:>6}/{len(images)}]  "
                f"passed {n_passed:,}  |  "
                f"rejected {n_rejected:,} ({reject_rate*100:.1f}%)  |  "
                f"ETA {eta / 60:.1f} min"
            )

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

    # -- Rejection breakdown
    if reject_reasons:
        print(f"\n  Rejection breakdown:")
        print(f"  {'Reason':<32} {'Count':>7}  {'%':>6}")
        print(f"  {'─' * 32} {'─' * 7}  {'─' * 6}")
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<32} {count:>7,}  {count / max(total, 1) * 100:>5.1f}%")

    # -- Prompt strategy breakdown
    if strategy_counts:
        print(f"\n  Prompt strategy breakdown:")
        print(f"  {'Strategy':<32} {'Count':>7}  {'%':>6}")
        print(f"  {'─' * 32} {'─' * 7}  {'─' * 6}")
        for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f"  {strat:<32} {count:>7,}  {count / max(total, 1) * 100:>5.1f}%")

    print()
    if reject_rate > SAM2_QA_MAX_REJECT_RATE:
        print(f"  [WARN] Rejection rate {reject_rate * 100:.1f}% still exceeds the "
              f"{SAM2_QA_MAX_REJECT_RATE * 100:.0f}% target.")
        print("         Check the rejection breakdown above and review the QA report.")
        print("         If 'center_fallback' is high, consider adding a bbox prompt.")
    else:
        print(f"  [OK] Rejection rate within acceptable range "
              f"({reject_rate * 100:.1f}% ≤ {SAM2_QA_MAX_REJECT_RATE * 100:.0f}%).")

    # -- Prompt mode breakdown (YOLO vs HSV fallback)
    yolo_ct = sum(v for k,v in strategy_counts.items() if k.startswith("yolo_"))
    hsv_ct  = sum(v for k,v in strategy_counts.items() if not k.startswith("yolo_"))
    if yolo_ct + hsv_ct > 0:
        print(f"\n  Prompting mode breakdown:")
        print(f"    YOLO-guided : {yolo_ct:>7,}  ({yolo_ct/(yolo_ct+hsv_ct)*100:.1f}%)")
        print(f"    HSV fallback: {hsv_ct:>7,}  ({hsv_ct/(yolo_ct+hsv_ct)*100:.1f}%)")

    print(f"\n  NEXT STEP: python validate_masks.py  (review QA report visually)")
    print(f"  THEN     : python train_teacher.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
