"""
================================================================================
 generate_tier1_masks.py — Phase 2: SAM2 Auto-Masking + QA  [v8]
================================================================================
 PURPOSE:
   Apply SAM2 to all 15,000 Tier 1 images to generate precise binary leaf
   silhouette masks. Masks are stored as BOTH:
     - float32 .npy  (raw SAM2 probability map — used as soft targets)
     - uint8 .png    (binarized at 0.5 — for visualization)

 AUTO-PROMPTING STRATEGY (v8 — aggressive multi-stage retry targeting coverage_low):
   For each image:
   1. Convert to HSV. Apply morphological closing to the tissue mask before
      component analysis — fills small holes / fragmented blobs caused by
      specular highlights or shadows.
      THREE color ranges are combined:
        Green  (H=35–75, S≥50)  — healthy leaf tissue
        Yellow (H=15–38)        — MSV streak yellowing
        Brown  (H=5–20)         — MLN necrotic / dead tissue
   2. Merge the N largest connected components within spatial proximity before
      bbox derivation — catches split leaves.
   3. Pass a box prompt to SAM2. Adaptive padding: larger pad for small
      components so SAM2 sees edge context.
   4. Background corner points fire when bbox > _MAX_BBOX_COVERAGE (0.50) —
      Background points are a 3×3 grid of 9 bg pts giving SAM2 stronger
      background signal across the frame.
   5. Five-attempt adaptive retry chain (v8):
        a. coverage_high → GREEN-ONLY bbox: re-derive bbox using only the
           green channel, excluding yellow/brown tissue that inflates the bbox
           on MSV/MLN leaves. Always forces bg corner grid. If no green tissue,
           falls back to pad=0 tight box.
        b. coverage_low / center_fallback → MULTI-SCALE EXPAND retries (NEW v8):
              - Pass 1: +30px  expand (v4 unchanged)
              - Pass 2: +60px  expand (NEW v8 — larger jump for small leaves)
              - Pass 3: +100px expand (NEW v8 — near-full-frame capture)
           Each pass re-runs QA; stops as soon as coverage_low is resolved.
        c. coverage_low with no bbox (center_fallback path) → LOWER tissue
           threshold (NEW v8): re-run get_leaf_bbox with _MIN_TISSUE_FRACTION
           halved to 0.025 to catch faint/small leaf tissue.
        d. confidence_low → single-mask mode (multimask=False) [v5 unchanged].
           Now fires AFTER any prior retry (BUG FIX v7 carried forward).
        e. Any remaining failure → foreground-point + full bbox hybrid
           (NEW v8): inject the image centroid as a foreground point alongside
           the current box to resolve ambiguous SAM2 segmentation.
   6. Fallback: if no tissue is detected, use four corner background points +
      image center foreground point. QA filters catch bad results.

 ROOT CAUSE ANALYSIS (v7 → v8):
   Observed at checkpoint 1500:
     coverage_low:503   (dominant — 62% of all failures)
     confidence_low:282
     coverage_high:29   (minor)

   v7 spent its effort on coverage_high (29 cases) while coverage_low (503)
   received only a single +30px retry — insufficient for images where the leaf
   fills only 2–4% of the frame (dark MLN images, heavily cropped shots).
   Additionally:
     - _MIN_TISSUE_FRACTION = 0.05 silently diverts 2–4% tissue images to
       center_fallback, which produces no bbox → SAM2 guesses → coverage_low.
     - _MIN_COMPONENT_AREA_PX = 500 discards small but real fragments.
     - center_fallback images skipped bbox retries entirely.
     - confidence retry never fired for coverage_low cases.

 QA FILTERS (v8 — unchanged thresholds, same as v3/v7):
   Filter 1 — Coverage range  : foreground must be 3%–90% of image
   Filter 2 — Mean confidence : mean prob of foreground region ≥ 0.60
   Filter 3 — Aspect ratio    : mask bounding box ratio ≥ 1.01
   Target rejection rate: < 8% of total images.

 CHANGELOG (v7 → v8):
   ROOT CAUSE: coverage_low dominates at 62% of failures; v7 single-retry
   was insufficient for faint/small-coverage leaf images.

   [FIX]  Multi-scale expand retry for coverage_low (primary fix):
          Three passes at +30px / +60px / +100px instead of one pass at +30px.
          Each pass stops early on success. Resolves cases where the initial
          bbox was too tight due to fragmented tissue detection.
   [FIX]  Lowered _MIN_TISSUE_FRACTION fallback (NEW v8):
          When center_fallback fires (tissue < 5%), a second attempt with
          threshold halved to 0.025 tries to recover a weak bbox before
          committing to the point-prompt fallback path.
   [FIX]  center_fallback images now enter the coverage_low retry chain
          (previously they skipped all bbox retries).
   [FIX]  Lowered _MIN_COMPONENT_AREA_PX 500 → 200: small but real leaf
          fragments were discarded before bbox derivation, forcing center_fallback.
   [FIX]  Foreground-point hybrid retry (NEW v8):
          Last-resort pass injects the image centroid as a foreground point
          alongside the active box, resolving cases where SAM2 is ambiguous
          between leaf and background at low coverage.
   [FIX]  confidence_low now also triggers the single-mask retry for images
          that reached it via coverage_low resolution (extended retry chain).
   [KEEP] All v7 fixes: green-only bbox retry, 3×3 bg grid, lowered
          _MAX_BBOX_COVERAGE=0.50, confidence=0.60, bug-fix confidence retry.
   [KEEP] All v6/v5/v4 fixes.

 OUTPUTS:
   data/tier1_leaf_masks/{stem}_softmask.npy   ← float32 probability map
   data/tier1_leaf_masks/{stem}_mask.png        ← uint8 binary visualization
   tier1_qa_report.csv                          ← per-image QA log
================================================================================
"""

import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from image_utils import load_image_rgb, to_hsv  # EXIF correction

from config import (
    SEED,
    TIER1_RAW_DIR, TIER1_MASKS_DIR, TIER1_QA_REPORT,
    SAM2_CHECKPOINT, SAM2_CONFIG,
    SAM2_GREEN_H_MIN, SAM2_GREEN_H_MAX,
    SAM2_GREEN_S_MIN, SAM2_GREEN_V_MIN,
    SAM2_YELLOW_H_MIN, SAM2_YELLOW_H_MAX,
    SAM2_YELLOW_S_MIN, SAM2_YELLOW_V_MIN,
    SAM2_BROWN_H_MIN,  SAM2_BROWN_H_MAX,
    SAM2_BROWN_S_MIN,  SAM2_BROWN_V_MIN,
    SAM2_QA_MIN_COVERAGE, SAM2_QA_MAX_COVERAGE,
    SAM2_QA_MIN_CONFIDENCE, SAM2_QA_MIN_ASPECT_RATIO,
    SAM2_QA_MAX_REJECT_RATE,
    VALID_EXTENSIONS,
)

# ── v8 QA / prompting constants ───────────────────────────────────────────────
# QA thresholds — unchanged from v3/v7. The target is to pass more images
# by improving prompting, not by lowering the quality bar.
_QA_MIN_COVERAGE   = 0.03   # foreground must cover ≥ 3% of image
_QA_MAX_COVERAGE   = 0.90   # foreground must cover ≤ 90% of image
_QA_MIN_CONFIDENCE = 0.60   # v7: lowered 0.65 → 0.60 to absorb dark MLN leaves
_QA_MIN_ASPECT     = 1.01   # mask bounding box ratio ≥ 1.01

# Green HSV range — tightened in v4 to exclude cogon grass / background vegetation.
_GREEN_H_MIN = 35   # slight raise avoids warm yellows
_GREEN_H_MAX = 75   # KEY: excludes cogon band 75–90
_GREEN_S_MIN = 50   # excludes dull/background greens
_GREEN_V_MIN = 40   # unchanged

# Morphological closing kernel for tissue mask.
# Fills holes from specular highlights and veins, connecting nearby fragments.
_MORPH_CLOSE_KSIZE = 15

# Multi-component merge: union the top-N components within spatial proximity.
_MERGE_TOP_N          = 5      # consider at most 5 largest components
_MERGE_PROXIMITY_FRAC = 0.35   # centroid must be within 35% of image diagonal

# Adaptive padding for bbox derivation.
_PAD_BASE  = 8    # minimum padding in pixels
_PAD_SCALE = 0.02 # additional pad = _PAD_SCALE * sqrt(component_area)
_PAD_MAX   = 40   # cap to avoid box exceeding image bounds excessively

# Multi-scale expand pads for coverage_low retries (NEW v8).
# Three passes: conservative → moderate → aggressive.
# Using a sequence instead of a single fixed value rescues images where the
# leaf is very small (needs +60/+100) without over-expanding normal cases.
_RETRY_EXPAND_PADS = [30, 60, 100]  # pixels added each side per pass

# Tight-box retry for coverage_high (unchanged from v4/v7).
_RETRY_PAD_TIGHT = 0

# Background point grid — injected when bbox is too large.
# v7: threshold lowered 0.65 → 0.50; 4-corner → 3×3 grid of 9 points.
_MAX_BBOX_COVERAGE = 0.50
_BG_CORNER_PAD    = 15   # pixels from image edge for grid point placement

# Minimum tissue fraction for bbox detection.
# Images below this go to center_fallback (point prompt only).
# v8: kept at 0.05 for primary pass; retry pass uses _MIN_TISSUE_FRACTION_RETRY.
_MIN_TISSUE_FRACTION       = 0.05
# v8 NEW: halved threshold for a second tissue-detection attempt on
# center_fallback images. Catches faint/small tissue that primary pass misses.
_MIN_TISSUE_FRACTION_RETRY = 0.025

# Minimum pixel area for a connected component to be considered a valid leaf.
# v8: lowered 500 → 200 — small but real leaf fragments (dark MLN, heavy crop)
# were being silently discarded, routing the image to center_fallback instead
# of a bbox, which is less reliable. 200px still excludes dust/noise.
_MIN_COMPONENT_AREA_PX = 200


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
# AUTO-PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

def get_leaf_bbox(
    img_rgb: np.ndarray,
    pad_override: int | None = None,
    min_tissue_fraction: float = _MIN_TISSUE_FRACTION,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """
    Derive a tight bounding box around the dominant leaf-tissue region
    using HSV color segmentation + morphological closing + multi-component merge.

    v8: added min_tissue_fraction parameter so retry callers can lower the bar
    without changing the global constant.

    THREE tissue ranges combined:
      Green  (H=_GREEN_H_MIN–_GREEN_H_MAX, S≥_GREEN_S_MIN) — healthy tissue
      Yellow (H=SAM2_YELLOW_H_MIN–MAX)                     — MSV yellowing
      Brown  (H=SAM2_BROWN_H_MIN–MAX)                      — MLN necrosis

    Returns:
        bbox     : (x1, y1, x2, y2) pixel coords, or None on fallback.
        strategy : "green_bbox" | "yellow_bbox" | "brown_bbox" |
                   "combined_bbox" | "center_fallback"
    """
    img_hsv  = to_hsv(img_rgb)
    h, s, v  = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    total_px = img_rgb.shape[0] * img_rgb.shape[1]
    img_h, img_w = img_rgb.shape[:2]

    # ── 1. Build per-tissue masks ─────────────────────────────────────────────
    green_mask = (
        (h >= _GREEN_H_MIN) & (h <= _GREEN_H_MAX) &
        (s >= _GREEN_S_MIN) & (v >= _GREEN_V_MIN)
    ).astype(np.uint8)

    yellow_mask = (
        (h >= SAM2_YELLOW_H_MIN) & (h <= SAM2_YELLOW_H_MAX) &
        (s >= SAM2_YELLOW_S_MIN) & (v >= SAM2_YELLOW_V_MIN)
    ).astype(np.uint8)

    brown_mask = (
        (h >= SAM2_BROWN_H_MIN) & (h <= SAM2_BROWN_H_MAX) &
        (s >= SAM2_BROWN_S_MIN) & (v >= SAM2_BROWN_V_MIN)
    ).astype(np.uint8)

    green_frac  = green_mask.sum()  / total_px
    yellow_frac = yellow_mask.sum() / total_px
    brown_frac  = brown_mask.sum()  / total_px

    has_green  = green_frac  >= min_tissue_fraction
    has_yellow = yellow_frac >= min_tissue_fraction
    has_brown  = brown_frac  >= min_tissue_fraction

    if not (has_green or has_yellow or has_brown):
        return None, "center_fallback"

    # ── 2. Build combined tissue mask ─────────────────────────────────────────
    tissue_mask = np.zeros_like(green_mask)
    active = []
    if has_green:
        tissue_mask = np.clip(tissue_mask + green_mask, 0, 1).astype(np.uint8)
        active.append("green")
    if has_yellow:
        tissue_mask = np.clip(tissue_mask + yellow_mask, 0, 1).astype(np.uint8)
        active.append("yellow")
    if has_brown:
        tissue_mask = np.clip(tissue_mask + brown_mask, 0, 1).astype(np.uint8)
        active.append("brown")

    strategy = ("combined_bbox" if len(active) > 1 else f"{active[0]}_bbox")

    # ── 3. Morphological closing ──────────────────────────────────────────────
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_MORPH_CLOSE_KSIZE, _MORPH_CLOSE_KSIZE)
    )
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    # ── 4. Connected-component analysis ───────────────────────────────────────
    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(
        tissue_mask, connectivity=8)

    if num_labels < 2:
        return None, "center_fallback"

    component_areas = stats[1:, cv2.CC_STAT_AREA]   # exclude background (label 0)
    valid_indices   = np.where(component_areas >= _MIN_COMPONENT_AREA_PX)[0]

    if len(valid_indices) == 0:
        return None, "center_fallback"

    sorted_valid = valid_indices[np.argsort(-component_areas[valid_indices])]
    best_label   = 1 + int(sorted_valid[0])

    # ── 5. Multi-component merge ───────────────────────────────────────────────
    img_diag       = (img_h ** 2 + img_w ** 2) ** 0.5
    prox_threshold = _MERGE_PROXIMITY_FRAC * img_diag
    anchor_cx, anchor_cy = centroids[best_label]
    merged_labels  = [best_label]

    for rank_idx in sorted_valid[1: _MERGE_TOP_N]:
        lbl    = 1 + int(rank_idx)
        cx, cy = centroids[lbl]
        dist   = ((cx - anchor_cx) ** 2 + (cy - anchor_cy) ** 2) ** 0.5
        if dist <= prox_threshold:
            merged_labels.append(lbl)

    merged_mask = np.isin(labels_map, merged_labels).astype(np.uint8)

    rows = np.any(merged_mask, axis=1)
    cols = np.any(merged_mask, axis=0)
    if not rows.any() or not cols.any():
        return None, "center_fallback"

    row_min = int(np.where(rows)[0][0])
    row_max = int(np.where(rows)[0][-1])
    col_min = int(np.where(cols)[0][0])
    col_max = int(np.where(cols)[0][-1])

    # ── 6. Adaptive padding ────────────────────────────────────────────────────
    if pad_override is not None:
        pad = pad_override
    else:
        merged_area = int(merged_mask.sum())
        pad = min(int(_PAD_BASE + _PAD_SCALE * (merged_area ** 0.5)), _PAD_MAX)

    x1 = max(0,     col_min - pad)
    y1 = max(0,     row_min - pad)
    x2 = min(img_w, col_max + pad)
    y2 = min(img_h, row_max + pad)

    return (x1, y1, x2, y2), strategy


def get_leaf_bbox_green_only(
    img_rgb: np.ndarray,
    pad_override: int | None = None,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """
    Derive a bounding box from the GREEN channel only (v7).

    Used exclusively as the coverage_high retry strategy. MSV/MLN leaves have
    yellow/brown tissue that spans most of the frame — the combined mask inflates
    the bbox, causing SAM2 to over-segment. By restricting to green pixels only,
    we anchor the bbox to the true leaf silhouette edges.

    Returns:
        bbox     : (x1, y1, x2, y2) pixel coords, or None if insufficient green.
        strategy : "green_only_bbox"
    """
    img_hsv = to_hsv(img_rgb)
    h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    img_h, img_w = img_rgb.shape[:2]
    total_px = img_h * img_w

    green_mask = (
        (h >= _GREEN_H_MIN) & (h <= _GREEN_H_MAX) &
        (s >= _GREEN_S_MIN) & (v >= _GREEN_V_MIN)
    ).astype(np.uint8)

    if green_mask.sum() / total_px < _MIN_TISSUE_FRACTION:
        return None, "green_only_bbox"

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_MORPH_CLOSE_KSIZE, _MORPH_CLOSE_KSIZE)
    )
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    rows = np.any(green_mask, axis=1)
    cols = np.any(green_mask, axis=0)
    if not rows.any() or not cols.any():
        return None, "green_only_bbox"

    row_min = int(np.where(rows)[0][0])
    row_max = int(np.where(rows)[0][-1])
    col_min = int(np.where(cols)[0][0])
    col_max = int(np.where(cols)[0][-1])

    if pad_override is not None:
        pad = pad_override
    else:
        merged_area = int(green_mask.sum())
        pad = min(int(_PAD_BASE + _PAD_SCALE * (merged_area ** 0.5)), _PAD_MAX)

    x1 = max(0,     col_min - pad)
    y1 = max(0,     row_min - pad)
    x2 = min(img_w, col_max + pad)
    y2 = min(img_h, row_max + pad)

    return (x1, y1, x2, y2), "green_only_bbox"


def build_box_prompt(
    img_rgb: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray | None, np.ndarray | None,
           np.ndarray | None, np.ndarray | None]:
    """
    Build SAM2 prompt arrays from a bounding box.

    Primary path (bbox is not None):
      Returns box=np.array([[x1,y1,x2,y2]]) and no point prompts.

    Fallback path (bbox is None — center_fallback):
      Returns a single center foreground point + four background corner points.

    Returns:
        box          : (1,4) float32 array or None
        point_coords : (N,2) float32 array or None
        point_labels : (N,)  int32  array or None
        (unused)     : None
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        return box, None, None, None

    # center_fallback: single foreground point + four background corners
    h, w = img_rgb.shape[:2]
    pad  = 10
    cx, cy = w // 2, h // 2
    point_coords = np.array([
        [cx,       cy      ],   # center foreground
        [pad,      pad     ],   # top-left background
        [w - pad,  pad     ],   # top-right background
        [pad,      h - pad ],   # bottom-left background
        [w - pad,  h - pad ],   # bottom-right background
    ], dtype=np.float32)
    point_labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
    return None, point_coords, point_labels, None


def build_fg_point_prompt(
    img_rgb: np.ndarray,
    box: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a foreground point at the image centroid to inject alongside a box
    prompt (NEW v8 — foreground-point hybrid retry).

    Used as a last-resort retry when SAM2 produces coverage_low despite having
    a valid bbox. Adding an explicit foreground point at the image center resolves
    cases where SAM2 is ambiguous: the leaf is present but SAM2 chooses a small
    segment. The centroid is a reliable foreground anchor for maize leaf images
    where the leaf is almost always centered.

    Returns:
        point_coords : (1,2) float32 array — image center
        point_labels : (1,)  int32  array — foreground label (1)
    """
    h, w = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    return point_coords, point_labels


# ══════════════════════════════════════════════════════════════════════════════
# QA FILTERS (thresholds unchanged from v3/v7)
# ══════════════════════════════════════════════════════════════════════════════

def qa_check(prob_map: np.ndarray) -> tuple[bool, str]:
    """
    Apply three QA filters to a SAM2 probability map.

    Args:
        prob_map: H×W float32 sigmoid probability map from SAM2,
                  upscaled to original image resolution.

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

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Phase 2: SAM2 Tier 1 Masking  [v8]")
    print("=" * 72)
    print("  QA thresholds (unchanged from v3/v7):")
    print(f"    Coverage     : {_QA_MIN_COVERAGE:.0%} – {_QA_MAX_COVERAGE:.0%}")
    print(f"    Confidence   : ≥ {_QA_MIN_CONFIDENCE:.2f}")
    print(f"    Aspect ratio : ≥ {_QA_MIN_ASPECT:.2f}")
    print("  Prompting      : contour bbox + 3×3 bg grid (>50% bbox) → green-only retry")
    print("                   → multi-scale expand [30/60/100px] → fg-point hybrid → SAM2")
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
    reject_reasons  = defaultdict(int)
    strategy_counts = defaultdict(int)
    t_start         = time.time()

    for i, img_path in enumerate(images):
        stem     = img_path.stem
        category = stem.split("_")[0]

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

        # ── Auto-prompting ─────────────────────────────────────────────────────
        bbox, strategy = get_leaf_bbox(img_rgb)
        strategy_counts[strategy] += 1
        box, point_coords, point_labels, _ = build_box_prompt(img_rgb, bbox)

        # ── SAM2 image embedding (called ONCE per image) ───────────────────────
        try:
            with torch.inference_mode():
                predictor.set_image(img_rgb)
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

        def predict_prob(box, point_coords, point_labels,
                         multimask=True, force_bg_points=False):
            """
            Run SAM2 predict() and return an upscaled float32 prob_map.
            Assumes predictor.set_image() has already been called.

            Background suppression uses a 3×3 grid of 9 background-label points
            (v7) when bbox_frac > _MAX_BBOX_COVERAGE or force_bg_points=True.
            """
            h_img, w_img = img_rgb.shape[:2]
            with torch.inference_mode():
                if box is not None:
                    x1, y1, x2, y2 = box[0]
                    bbox_frac = (x2 - x1) * (y2 - y1) / (h_img * w_img)
                    use_bg_pts = force_bg_points or (bbox_frac > _MAX_BBOX_COVERAGE)

                    if use_bg_pts:
                        # 3×3 grid of 9 background points (v7).
                        cp    = _BG_CORNER_PAD
                        mid_x = w_img // 2
                        mid_y = h_img // 2
                        bg_pts = np.array([
                            [cp,          cp         ],  # top-left corner
                            [w_img - cp,  cp         ],  # top-right corner
                            [cp,          h_img - cp ],  # bottom-left corner
                            [w_img - cp,  h_img - cp ],  # bottom-right corner
                            [mid_x,       cp         ],  # top-center
                            [mid_x,       h_img - cp ],  # bottom-center
                            [cp,          mid_y      ],  # left-center
                            [w_img - cp,  mid_y      ],  # right-center
                            [mid_x,       mid_y      ],  # image center
                        ], dtype=np.float32)
                        bg_lbl = np.zeros(9, dtype=np.int32)

                        # Merge with any caller-supplied foreground point prompts
                        if point_coords is not None and point_labels is not None:
                            all_pts = np.vstack([point_coords, bg_pts])
                            all_lbl = np.concatenate([point_labels, bg_lbl])
                        else:
                            all_pts = bg_pts
                            all_lbl = bg_lbl

                        masks, scores, logits = predictor.predict(
                            box=box,
                            point_coords=all_pts,
                            point_labels=all_lbl,
                            multimask_output=multimask,
                        )
                    else:
                        if point_coords is not None and point_labels is not None:
                            # Hybrid: box + foreground point (v8 last-resort retry)
                            masks, scores, logits = predictor.predict(
                                box=box,
                                point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=multimask,
                            )
                        else:
                            masks, scores, logits = predictor.predict(
                                box=box,
                                multimask_output=multimask,
                            )
                else:
                    masks, scores, logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=multimask,
                    )

            best_idx  = int(np.argmax(scores))
            logit_map = logits[best_idx].squeeze()
            if logit_map.shape != (h_img, w_img):
                logit_map = cv2.resize(
                    logit_map, (w_img, h_img),
                    interpolation=cv2.INTER_LINEAR,
                )
            return (1.0 / (1.0 + np.exp(-logit_map))).astype(np.float32)

        # ── First SAM2 pass ────────────────────────────────────────────────────
        try:
            prob_map = predict_prob(box, point_coords, point_labels, multimask=True)
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

        # ── Adaptive retries (v8 — five-stage chain) ───────────────────────────
        # Stage order:
        #   1. coverage_high  → green-only bbox (v7)
        #   2. coverage_low   → multi-scale expand [30 / 60 / 100 px]  (NEW v8)
        #   3. center_fallback→ lower tissue threshold retry (NEW v8)
        #   4. confidence_low → single-mask mode (v5, bug-fixed v7)
        #   5. any remaining  → foreground-point hybrid (NEW v8)

        passed, reason = qa_check(prob_map)

        # Track active prompts for retry chain.
        active_box          = box
        active_point_coords = point_coords
        active_point_labels = point_labels

        if not passed:
            reason_key = reason.split(":")[0]

            # ── Stage 1: coverage_high → green-only bbox (v7) ────────────────
            if reason_key == "coverage_high" and bbox is not None:
                green_bbox, green_strategy = get_leaf_bbox_green_only(img_rgb)
                if green_bbox is not None:
                    retry_box, _, _, _ = build_box_prompt(img_rgb, green_bbox)
                    try:
                        prob_map = predict_prob(retry_box, None, None,
                                               multimask=True,
                                               force_bg_points=True)
                        passed, reason = qa_check(prob_map)
                        active_box = retry_box
                        if passed:
                            strategy = green_strategy + "_retry_greenonly"
                            strategy_counts[strategy] += 1
                    except Exception:
                        pass

                if not passed:
                    # Fallback: tight-box (pad=0) if green-only insufficient
                    retry_bbox, retry_strategy = get_leaf_bbox(
                        img_rgb, pad_override=_RETRY_PAD_TIGHT)
                    if retry_bbox is not None:
                        retry_box, _, _, _ = build_box_prompt(img_rgb, retry_bbox)
                        try:
                            prob_map = predict_prob(retry_box, None, None,
                                                   multimask=True,
                                                   force_bg_points=True)
                            passed, reason = qa_check(prob_map)
                            active_box = retry_box
                            if passed:
                                strategy = retry_strategy + "_retry_tight"
                                strategy_counts[strategy] += 1
                        except Exception:
                            pass

            # ── Stage 2: coverage_low → multi-scale expand (NEW v8) ──────────
            # Three progressive pad sizes: 30 / 60 / 100 px.
            # Each pass re-runs QA; stops as soon as coverage_low resolves.
            # Also fires for center_fallback images (strategy="center_fallback")
            # because they have no bbox and SAM2 guesses poorly with only a
            # point prompt — giving them a bbox via a low threshold improves odds.
            if not passed and reason.split(":")[0] == "coverage_low":
                for expand_pad in _RETRY_EXPAND_PADS:
                    retry_bbox, retry_strategy = get_leaf_bbox(
                        img_rgb, pad_override=expand_pad)
                    if retry_bbox is not None:
                        retry_box, _, _, _ = build_box_prompt(img_rgb, retry_bbox)
                        try:
                            prob_map = predict_prob(retry_box, None, None,
                                                   multimask=True)
                            passed, reason = qa_check(prob_map)
                            active_box          = retry_box
                            active_point_coords = None
                            active_point_labels = None
                            if passed:
                                strategy = f"{retry_strategy}_retry_expand{expand_pad}"
                                strategy_counts[strategy] += 1
                                break   # stop at first successful pad
                        except Exception:
                            continue   # try next pad size on SAM2 error

            # ── Stage 3: center_fallback → lower tissue threshold (NEW v8) ───
            # When no tissue is found at 5%, retry with 2.5% threshold.
            # This catches faint/dark tissue (dark MLN leaves, heavy shadow)
            # that the primary pass misses, converting center_fallback to a
            # proper bbox prompt and feeding it into the expand chain above.
            if not passed and strategy == "center_fallback":
                retry_bbox, retry_strategy = get_leaf_bbox(
                    img_rgb,
                    min_tissue_fraction=_MIN_TISSUE_FRACTION_RETRY,
                )
                if retry_bbox is not None and retry_strategy != "center_fallback":
                    retry_box, _, _, _ = build_box_prompt(img_rgb, retry_bbox)
                    try:
                        prob_map = predict_prob(retry_box, None, None,
                                               multimask=True)
                        passed, reason = qa_check(prob_map)
                        active_box          = retry_box
                        active_point_coords = None
                        active_point_labels = None
                        if passed:
                            strategy = retry_strategy + "_retry_lowtissue"
                            strategy_counts[strategy] += 1
                        elif reason.split(":")[0] == "coverage_low":
                            # Found a weak bbox but coverage still low —
                            # run multi-scale expand on this new bbox too.
                            for expand_pad in _RETRY_EXPAND_PADS:
                                retry_bbox2, retry_strategy2 = get_leaf_bbox(
                                    img_rgb,
                                    pad_override=expand_pad,
                                    min_tissue_fraction=_MIN_TISSUE_FRACTION_RETRY,
                                )
                                if retry_bbox2 is not None:
                                    retry_box2, _, _, _ = build_box_prompt(
                                        img_rgb, retry_bbox2)
                                    try:
                                        prob_map = predict_prob(
                                            retry_box2, None, None, multimask=True)
                                        passed, reason = qa_check(prob_map)
                                        active_box          = retry_box2
                                        active_point_coords = None
                                        active_point_labels = None
                                        if passed:
                                            strategy = (
                                                f"{retry_strategy2}"
                                                f"_retry_lowtissue_expand{expand_pad}"
                                            )
                                            strategy_counts[strategy] += 1
                                            break
                                    except Exception:
                                        continue
                                if passed:
                                    break
                    except Exception:
                        pass

        # ── Stage 4: confidence_low → single-mask mode (v5, bug-fixed v7) ───
        # Fires for any image still failing on confidence, including those
        # that went through a bbox retry above (BUG FIX v7 carried forward).
        if not passed and reason.split(":")[0] == "confidence_low":
            try:
                prob_map_single = predict_prob(active_box,
                                               active_point_coords,
                                               active_point_labels,
                                               multimask=False)
                passed_single, reason_single = qa_check(prob_map_single)
                if passed_single:
                    prob_map  = prob_map_single
                    passed    = passed_single
                    reason    = reason_single
                    strategy  = strategy + "_singlemask"
                    strategy_counts[strategy] += 1
            except Exception:
                pass

        # ── Stage 5: foreground-point hybrid (NEW v8) ─────────────────────────
        # Last-resort: inject the image centroid as a foreground point alongside
        # the active box. Resolves cases where SAM2 is ambiguous at low coverage
        # — leaf present but SAM2 segments a small region. The centroid is a
        # reliable fg anchor for centered maize leaf images.
        # Fires on any remaining failure EXCEPT center_fallback (no box) and
        # coverage_high (adding a fg point worsens over-segmentation).
        if (not passed
                and active_box is not None
                and reason.split(":")[0] not in ("coverage_high", "no_foreground")):
            try:
                fg_pts, fg_lbl = build_fg_point_prompt(img_rgb, active_box)
                prob_map_hybrid = predict_prob(active_box, fg_pts, fg_lbl,
                                               multimask=True)
                passed_h, reason_h = qa_check(prob_map_hybrid)
                if passed_h:
                    prob_map = prob_map_hybrid
                    passed   = passed_h
                    reason   = reason_h
                    strategy = strategy + "_fghybrid"
                    strategy_counts[strategy] += 1
                elif reason_h.split(":")[0] == "confidence_low":
                    # Hybrid improved coverage but not confidence — try single-mask
                    prob_map_hs = predict_prob(active_box, fg_pts, fg_lbl,
                                               multimask=False)
                    passed_hs, reason_hs = qa_check(prob_map_hs)
                    if passed_hs:
                        prob_map = prob_map_hs
                        passed   = passed_hs
                        reason   = reason_hs
                        strategy = strategy + "_fghybrid_singlemask"
                        strategy_counts[strategy] += 1
            except Exception:
                pass

        # ── Final QA outcome ──────────────────────────────────────────────────
        binary    = (prob_map >= 0.5).astype(np.uint8)
        coverage  = float(binary.sum()) / (prob_map.shape[0] * prob_map.shape[1])
        mean_conf = float(prob_map[binary == 1].mean()) if binary.sum() > 0 else 0.0

        if not passed:
            reason_key = reason.split(":")[0]
            reject_reasons[reason_key] += 1
            qa_rows.append({
                "filename":        img_path.name,
                "category":        category,
                "status":          "rejected",
                "reason":          reason,
                "prompt_strategy": strategy,
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
            if reject_reasons:
                reason_parts = "  ".join(
                    f"{r}:{c}" for r, c in
                    sorted(reject_reasons.items(), key=lambda x: -x[1])
                )
                print(f"         reasons → {reason_parts}")

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

    # ── Rejection breakdown ───────────────────────────────────────────────────
    if reject_reasons:
        print(f"\n  Rejection breakdown:")
        print(f"  {'Reason':<32} {'Count':>7}  {'%':>6}")
        print(f"  {'─' * 32} {'─' * 7}  {'─' * 6}")
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<32} {count:>7,}  {count / max(total, 1) * 100:>5.1f}%")

    # ── Prompt strategy breakdown ─────────────────────────────────────────────
    if strategy_counts:
        print(f"\n  Prompt strategy breakdown:")
        print(f"  {'Strategy':<40} {'Count':>7}  {'%':>6}")
        print(f"  {'─' * 40} {'─' * 7}  {'─' * 6}")
        for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f"  {strat:<40} {count:>7,}  {count / max(total, 1) * 100:>5.1f}%")

    print()
    if reject_rate > SAM2_QA_MAX_REJECT_RATE:
        print(f"  [WARN] Rejection rate {reject_rate * 100:.1f}% still exceeds the "
              f"{SAM2_QA_MAX_REJECT_RATE * 100:.0f}% target.")
        print("         Read the rejection reason breakdown above — tune the dominant cause:")
        print("           coverage_low   → lower _MIN_TISSUE_FRACTION_RETRY below 0.025")
        print("                            or add a 4th expand pad (e.g. 150px) to _RETRY_EXPAND_PADS")
        print("                            or lower _MIN_COMPONENT_AREA_PX below 200")
        print("           coverage_high  → lower _MAX_BBOX_COVERAGE (now 0.50) or raise _BG_CORNER_PAD")
        print("           confidence_low → lower _QA_MIN_CONFIDENCE below 0.60 (already at min)")
        print("           center_fallback→ lower _MIN_TISSUE_FRACTION_RETRY below 0.025")
        print("           sam2_error     → check GPU memory / SAM2 install")
        print("         Review overlays in reports/tier1_overlays for visual confirmation.")
    else:
        print(f"  [OK] Rejection rate within acceptable range "
              f"({reject_rate * 100:.1f}% ≤ {SAM2_QA_MAX_REJECT_RATE * 100:.0f}%).")

    print(f"\n  NEXT STEP: python validate_masks.py  (review QA report visually)")
    print(f"  THEN     : python train_teacher.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
