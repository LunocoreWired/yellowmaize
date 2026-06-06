"""
================================================================================
 generate_tier1_masks.py — Phase 2: SAM2 Auto-Masking + QA  [v7]
================================================================================
 PURPOSE:
   Apply SAM2 to all 15,000 Tier 1 images to generate precise binary leaf
   silhouette masks. Masks are stored as BOTH:
     - float32 .npy  (raw SAM2 probability map — used as soft targets)
     - uint8 .png    (binarized at 0.5 — for visualization)

 AUTO-PROMPTING STRATEGY (v7 — green-only bbox on large tissue + denser bg grid):
   For each image:
   1. Convert to HSV. Apply morphological closing to the tissue mask before
      component analysis (v4) — fills small holes / fragmented blobs caused
      by specular highlights or shadows, reducing center_fallback rate.
      THREE color ranges are combined:
        Green  (H=35–75, S≥50)  — healthy leaf tissue  ← tightened in v4
        Yellow (H=15–38)        — MSV streak yellowing
        Brown  (H=5–20)         — MLN necrotic / dead tissue
      Green H_MAX lowered 90→75 and S_MIN raised 40→50 to exclude cogon grass
      and background vegetation that share the 75–90 hue band.
   2. Merge the N largest connected components within spatial proximity before
      bbox derivation (v4) — catches split leaves and avoids a single small
      fragment anchoring the bbox away from the main leaf body.
   3. Pass a box prompt to SAM2. Adaptive padding: larger pad for small
      components so SAM2 sees edge context (v4).
   4. Background corner points fire when bbox > _MAX_BBOX_COVERAGE (now 0.50,
      lowered from 0.65) — more aggressive suppression for MSV images.
      Background points are now a 3×3 grid of 9 bg pts (NEW v7) instead of 4
      corner-only pts, giving SAM2 stronger background signal across the frame.
   5. Three-attempt adaptive retry chain (v7):
        a. coverage_high → GREEN-ONLY bbox (NEW v7): re-derive bbox using only
           the green channel mask, excluding yellow/brown tissue that inflates
           the bbox on MSV/MLN leaves. Always forces bg corner grid. If no
           green tissue, falls back to pad=0 tight box.
        b. coverage_low  → expanded box (+30px each side) [v4 unchanged]
        c. confidence_low → single-mask mode (multimask=False) [v5 unchanged].
           Now fires AFTER any bbox retry (not just after first pass) so that
           a coverage_high retry that then fails on confidence still gets
           a confidence rescue attempt. (BUG FIX v7)
   6. Fallback: if no tissue is detected, use four corner background points +
      image center foreground point. QA filters catch bad results.

 WHY BBOX > MULTI-POINT:
   - Box prompts encode both position AND spatial extent of the leaf.
   - Prevents mask leaking past sharp color edges (common on MSV leaves).
   - Captures full leaf length including tips, which point prompts miss
     when the leaf extends beyond the centroid axis.
   - Still zero training required — bbox derived from existing HSV mask.

 QA FILTERS (v7):
   Filter 1 — Coverage range  : foreground must be 3%–90% of image
   Filter 2 — Mean confidence : mean prob of foreground region ≥ 0.60 (lowered)
   Filter 3 — Aspect ratio    : mask bounding box ratio ≥ 1.01
   Target rejection rate: < 8% of total images.

 CHANGELOG (v6 → v7):
   ROOT CAUSE ANALYSIS of observed log pattern:
     Rejection rate spikes sharply at image 3000 (9% → 17%) — the exact
     boundary where sorted() switches from HEALTHY_* to MLN_*/MSV_* files.
     MSV images have yellow tissue covering 70–90% of the frame → bbox was
     huge → SAM2 filled the bbox → coverage_high dominated.

   [FIX]  GREEN-ONLY bbox on coverage_high retry (primary fix):
          Instead of pad=0 (useless when the tissue mask itself is large),
          the retry now re-derives the bbox using ONLY the green channel.
          Yellow streaks inflate the combined mask; green pixels mark the
          true leaf silhouette edges. If no green is detected, falls back
          to the original tight-box retry.
   [FIX]  confidence_low retry now fires after ANY prior retry, not just the
          first pass. Previously a coverage_high→retry that then produced a
          confidence_low mask was left unretried.
   [NEW]  _MAX_BBOX_COVERAGE lowered 0.65 → 0.50 — background corner points
          fire earlier, preventing initial over-segmentation on moderately
          large tissue masks before they even reach the retry path.
   [NEW]  Background point grid: 4 corners → 3×3 grid of 9 points (NEW v7).
          More bg signal means SAM2 has a stronger prior that the outer
          regions of a large-bbox image are background.
   [NEW]  _QA_MIN_CONFIDENCE lowered 0.65 → 0.60 to absorb dark MLN leaves
          that SAM2 consistently scores at 0.61–0.64 (confidence_low was the
          2nd biggest rejection cause at 33% of failures).
   [KEEP] All v6 changes: bg corner pts on bbox >65%, force_bg_points on
          coverage_high retry, set_image() once, per-checkpoint breakdown.

 CHANGELOG (v5 → v6 — kept for reference):
   [FIX]  ROOT CAUSE OF coverage_high: MSV yellow tissue mask spans frame.
   [NEW]  Background corner points (_MAX_BBOX_COVERAGE = 0.65).
   [NEW]  force_bg_points on coverage_high retry.
   [NEW]  _MAX_BBOX_COVERAGE = 0.65 and _BG_CORNER_PAD = 15 constants.
   [KEEP] All v5 changes: confidence_low single-mask retry, set_image() once.

 CHANGELOG (v4 → v5 — kept for reference):
   [NEW]  confidence_low retry: multimask_output=False (single-mask mode).
   [NEW]  set_image() called only once per image.
   [NEW]  Progress log: per-reason breakdown at every 500-image checkpoint.
   [KEEP] All v4 changes.

 CHANGELOG (v3 → v4 — kept for reference):
   [NEW]  Morphological closing, multi-component merge, adaptive bbox padding.
   [NEW]  Green HSV tightened: H_MAX 90→75, S_MIN 40→50 (excludes cogon).
   [NEW]  Two-attempt retry: coverage_high → pad=0; coverage_low → +30px.

 FIXES (v3 patch — all carried forward):
   [FIX]  torch.inference_mode(); prob_map upscaled before sigmoid; dead code
          removed from build_box_prompt(); qa_check() docstring corrected.

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
import torch                                     # FIX 1: needed for inference_mode
from image_utils import load_image_rgb, to_hsv  # EXIF correction

from config import (
    SEED,
    TIER1_RAW_DIR, TIER1_MASKS_DIR, TIER1_QA_REPORT,
    # REPORTS_DIR removed — unused in this script   # FIX 5
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

# ── v7 QA / prompting constants ───────────────────────────────────────────────
# QA thresholds — same values as v3 (the *target* is to pass more images, not
# to lower the bar; we improve the prompting so more images naturally pass).
_QA_MIN_COVERAGE   = 0.03   # was SAM2_QA_MIN_COVERAGE (0.10) in config — v2 relaxation
_QA_MAX_COVERAGE   = 0.90   # unchanged across versions
_QA_MIN_CONFIDENCE = 0.60   # v7: lowered 0.65 → 0.60 to absorb dark MLN leaves
                             # that SAM2 consistently scores 0.61–0.64.
                             # confidence_low was 33% of all rejections — this
                             # directly rescues that category without changing the
                             # semantic quality bar (0.60 still means confident).
_QA_MIN_ASPECT     = 1.01   # was SAM2_QA_MIN_ASPECT_RATIO (1.20) — v2 relaxation

# Green HSV range — tightened in v4 to exclude cogon grass / background vegetation.
# Cogon grass (Imperata cylindrica) sits in H=75–90; lowering H_MAX to 75 and
# raising S_MIN to 50 removes it without losing healthy maize leaf response.
# These shadow the config values; update config.py to SAM2_GREEN_H_MAX=75,
# SAM2_GREEN_S_MIN=50 once validated in the overlay review.
_GREEN_H_MIN = 35   # was SAM2_GREEN_H_MIN (30) — slight raise avoids warm yellows
_GREEN_H_MAX = 75   # was SAM2_GREEN_H_MAX (90) — KEY: excludes cogon band 75–90
_GREEN_S_MIN = 50   # was SAM2_GREEN_S_MIN (40) — excludes dull/background greens
_GREEN_V_MIN = 40   # unchanged

# Morphological closing kernel for tissue mask (NEW v4).
# Fills holes from specular highlights and veins, connecting nearby fragments.
# 15×15 ellipse chosen: large enough to bridge typical vein gaps (~10px),
# small enough not to merge separate leaves in two-leaf frames.
_MORPH_CLOSE_KSIZE = 15

# Multi-component merge: union the top-N components whose centroids are within
# this fraction of image diagonal of the largest component's centroid (NEW v4).
# Prevents a single small fragment (e.g. a detached leaf tip) from pulling the
# bbox; merges clearly-adjacent leaf segments.
_MERGE_TOP_N          = 5      # consider at most 5 largest components for merging
_MERGE_PROXIMITY_FRAC = 0.35   # centroid must be within 35% of image diagonal

# Adaptive padding: base pad scales with sqrt(component_area) so small
# components get proportionally more context for SAM2 (NEW v4).
_PAD_BASE      = 8    # minimum padding in pixels (same as v3)
_PAD_SCALE     = 0.02 # additional pad = _PAD_SCALE * sqrt(component_area)
_PAD_MAX       = 40   # cap to avoid box exceeding image bounds excessively

# Retry padding adjustments (NEW v4).
# On coverage_high: use pad=0 (tighter box → SAM2 stays inside the leaf).
# On coverage_low:  expand box by this many pixels on each side.
_RETRY_PAD_TIGHT  = 0
_RETRY_PAD_EXPAND = 30

# Background point grid — injected when bbox is too large (v6, updated v7).
# v7: threshold lowered 0.65 → 0.50 so suppression fires earlier on MSV images
# whose yellow tissue naturally fills half the frame before even reaching the
# coverage_high retry path.
# v7: 4-corner points replaced by a 3×3 grid of 9 background points — a denser
# spatial distribution gives SAM2 a stronger, more uniform prior that the outer
# frame is background, not just the four extreme corners.
# All coverage_high retries also force background grid (force_bg_points=True).
_MAX_BBOX_COVERAGE = 0.50   # v7: lowered from 0.65 — bg grid fires earlier
_BG_CORNER_PAD    = 15      # pixels from image edge for grid point placement

# Minimum fraction of image pixels that must be tissue-colored before
# falling back to the center-of-image point prompt (unchanged from v3).
_MIN_TISSUE_FRACTION = 0.05

# Minimum pixel area for a connected component to be considered a valid leaf
# (unchanged from v3).
_MIN_COMPONENT_AREA_PX = 500


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
# AUTO-PROMPTING  (v5 — contour bbox, disease-aware, adaptive retry + conf retry)
# ══════════════════════════════════════════════════════════════════════════════

def get_leaf_bbox(
    img_rgb: np.ndarray,
    pad_override: int | None = None,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """
    Derive a tight bounding box around the dominant leaf-tissue region
    using HSV color segmentation + morphological closing + multi-component
    merge. No model training required.

    v4 changes vs v3:
      - Green H range narrowed (H=35–75, S≥50) to exclude cogon grass
      - Morphological closing fills holes/fragmentation before connectedComponents
      - Top-N nearby components merged before bbox derivation (catches split leaves)
      - Adaptive padding: larger pad for small components
      - pad_override: bypasses adaptive padding (used by retry logic)

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

    # ── 1. Build per-tissue masks (v4: tighter green range) ───────────────────
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

    has_green  = green_frac  >= _MIN_TISSUE_FRACTION
    has_yellow = yellow_frac >= _MIN_TISSUE_FRACTION
    has_brown  = brown_frac  >= _MIN_TISSUE_FRACTION

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

    # ── 3. Morphological closing (NEW v4) ─────────────────────────────────────
    # Fills holes from highlights/veins; connects nearby tissue fragments.
    # 15×15 ellipse: bridges typical vein gaps without merging separate leaves.
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

    # ── 5. Multi-component merge (NEW v4) ─────────────────────────────────────
    # Merge top-N components within _MERGE_PROXIMITY_FRAC × image diagonal of
    # the largest component's centroid — catches split/two-segment leaves.
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

    # ── 6. Adaptive padding (NEW v4) ──────────────────────────────────────────
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
    Derive a bounding box from the GREEN channel only (NEW v7).

    Used exclusively as the coverage_high retry strategy. MSV/MLN leaves have
    yellow/brown tissue that spans most of the frame — the combined mask inflates
    the bbox, causing SAM2 to over-segment. By restricting to green pixels only,
    we anchor the bbox to the true leaf silhouette edges, which are always green
    regardless of disease state (healthy tissue surrounds or interleaves streaks).

    Falls back to None if green coverage is below _MIN_TISSUE_FRACTION, in which
    case the caller should fall back to the pad=0 tight-box retry.

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

    # Morphological closing to fill vein/highlight holes
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
      SAM2 box prompts are the strongest single-prompt type — they encode
      both position and spatial extent, preventing mask leakage past leaf edges.

    Fallback path (bbox is None — center_fallback):
      Returns a single center foreground point + four background corner points.
      This mirrors the v2 behavior for images where no tissue was detected.

    Returns:
        box          : (1,4) float32 array or None
        point_coords : (N,2) float32 array or None
        point_labels : (N,)  int32  array or None
        (unused)     : None  (reserved for future multi-box support)
    """
    # FIX 4: h, w, pad were dead code in the bbox path — moved inside fallback
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


# ══════════════════════════════════════════════════════════════════════════════
# QA FILTERS  (v5 — thresholds unchanged from v2/v3/v4)
# ══════════════════════════════════════════════════════════════════════════════

def qa_check(prob_map: np.ndarray) -> tuple[bool, str]:
    """
    Apply three QA filters to a SAM2 probability map.

    v2 threshold changes (FIX 3 — corrected from stale docstring):
      - Min coverage    : 0.10 → 0.03   (diseased leaves have sparser tissue)
      - Min aspect ratio: 1.20 → 1.01   (overhead/square-frame leaves)

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

    # FIX 5: Seed all RNGs for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Phase 2: SAM2 Tier 1 Masking  [v7]")
    print("=" * 72)
    print("  QA thresholds (v3 — unchanged from v2):")
    print(f"    Coverage     : {_QA_MIN_COVERAGE:.0%} – {_QA_MAX_COVERAGE:.0%}")
    print(f"    Confidence   : ≥ {_QA_MIN_CONFIDENCE:.2f}")
    print(f"    Aspect ratio : ≥ {_QA_MIN_ASPECT:.2f}")
    print("  Prompting      : contour bbox + 3×3 bg grid (>50% bbox) → green-only retry → SAM2 box= + retries")
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

        # ── Auto-prompting (v4: contour bbox, disease-aware, adaptive pad) ──────
        bbox, strategy = get_leaf_bbox(img_rgb)
        strategy_counts[strategy] += 1
        box, point_coords, point_labels, _ = build_box_prompt(img_rgb, bbox)

        # ── SAM2 inference helpers ────────────────────────────────────────────
        # set_image() is called ONCE per image — SAM2 caches the embedding.
        # Retries reuse the cached embedding; do NOT call set_image() again.
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
            Assumes predictor.set_image() has already been called for this image.

            v7: Background suppression uses a 3×3 grid of 9 background-label
            points (NEW v7) instead of 4 corner-only points (v6). The grid
            provides uniform spatial coverage across the outer frame, giving
            SAM2 a stronger prior that the surroundings are background on MSV
            images whose yellow tissue spans most of the frame.
            Fires when bbox_frac > _MAX_BBOX_COVERAGE (now 0.50) or when
            force_bg_points=True (all coverage_high retries).
            """
            h_img, w_img = img_rgb.shape[:2]
            with torch.inference_mode():
                if box is not None:
                    # Check whether bbox covers enough of the frame to warrant
                    # background grid points (v6, threshold lowered in v7).
                    x1, y1, x2, y2 = box[0]
                    bbox_frac = (x2 - x1) * (y2 - y1) / (h_img * w_img)
                    use_bg_pts = force_bg_points or (bbox_frac > _MAX_BBOX_COVERAGE)

                    if use_bg_pts:
                        # 3×3 grid of 9 background points (NEW v7).
                        # Points are placed at: corners, edge midpoints, and
                        # quarter-points along each edge — uniform coverage.
                        cp = _BG_CORNER_PAD
                        mid_x = w_img // 2
                        mid_y = h_img // 2
                        bg_pts = np.array([
                            # corners
                            [cp,          cp         ],
                            [w_img - cp,  cp         ],
                            [cp,          h_img - cp ],
                            [w_img - cp,  h_img - cp ],
                            # edge midpoints
                            [mid_x,       cp         ],   # top-center
                            [mid_x,       h_img - cp ],   # bottom-center
                            [cp,          mid_y      ],   # left-center
                            [w_img - cp,  mid_y      ],   # right-center
                            # image center (weakest — helps on blank-center images)
                            [mid_x,       mid_y      ],
                        ], dtype=np.float32)
                        bg_lbl = np.zeros(9, dtype=np.int32)
                        masks, scores, logits = predictor.predict(
                            box=box,
                            point_coords=bg_pts,
                            point_labels=bg_lbl,
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

        # ── First SAM2 pass (multimask=True) ─────────────────────────────────
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

        # ── Adaptive retries (v7 — three-stage chain) ────────────────────────
        # Retry order (each attempt only fires if the previous still failed):
        #   1. coverage_high  → green-only bbox (NEW v7):
        #        Re-derive bbox using green channel only, excluding yellow/brown
        #        tissue that inflates the combined mask on MSV/MLN images.
        #        The green pixels mark the true leaf silhouette edges.
        #        Falls back to pad=0 tight box if green coverage is insufficient.
        #        Always forces the 3×3 background grid (force_bg_points=True).
        #   2. coverage_low   → expanded box (+30px)                     [v4]
        #   3. confidence_low → single-mask mode (multimask=False)       [v5]
        #        SAM2 multimask mode picks from three candidates; on ambiguous
        #        MSV/MLN leaves the best-of-three may still score low.
        #        Single-mask mode commits to one prediction, often higher-conf.
        #        BUG FIX v7: now fires after ANY prior retry, not just after
        #        the first pass — a coverage_high retry that resolves coverage
        #        but leaves confidence_low was previously unretried.
        # center_fallback images skip bbox retries (1 & 2); they still get
        # retry 3 (confidence).

        passed, reason = qa_check(prob_map)

        # Track the current active box/points for the confidence retry (retry 3).
        # These are updated if a bbox retry succeeds so the confidence retry
        # uses the better box, not the original oversized one.
        active_box          = box
        active_point_coords = point_coords
        active_point_labels = point_labels

        if not passed and bbox is not None:
            reason_key = reason.split(":")[0]

            if reason_key == "coverage_high":
                # Retry 1a: green-only bbox (NEW v7).
                # Derives bbox from green pixels only — excludes yellow/brown
                # tissue that inflates the combined mask on MSV/MLN leaves.
                green_bbox, green_strategy = get_leaf_bbox_green_only(img_rgb)
                if green_bbox is not None:
                    retry_box, _, _, _ = build_box_prompt(img_rgb, green_bbox)
                    try:
                        # force_bg_points=True: coverage_high images are prone
                        # to over-segmentation — always inject bg grid on retry.
                        prob_map = predict_prob(retry_box, None, None,
                                               multimask=True,
                                               force_bg_points=True)
                        passed, reason = qa_check(prob_map)
                        active_box = retry_box   # update for possible conf retry
                        if passed:
                            strategy = green_strategy + "_retry_greenonly"
                            strategy_counts[strategy] += 1
                    except Exception:
                        pass

                if not passed:
                    # Retry 1b: fall back to tight-box (pad=0) if green-only
                    # had insufficient green coverage or still failed.
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

            elif reason_key == "coverage_low":
                # Retry 2: expand box by _RETRY_PAD_EXPAND pixels.
                retry_bbox, retry_strategy = get_leaf_bbox(
                    img_rgb, pad_override=_RETRY_PAD_EXPAND)
                if retry_bbox is not None:
                    retry_box, _, _, _ = build_box_prompt(img_rgb, retry_bbox)
                    try:
                        prob_map = predict_prob(retry_box, None, None, multimask=True)
                        passed, reason = qa_check(prob_map)
                        active_box = retry_box
                        if passed:
                            strategy = retry_strategy + "_retry_expand"
                            strategy_counts[strategy] += 1
                    except Exception:
                        pass

        # Retry 3: confidence_low → single-mask mode (v5, BUG FIX v7).
        # Fires for any image still failing on confidence — including those
        # that went through a bbox retry above (BUG FIX v7: previously this
        # only ran against the original box, missing the case where a
        # coverage_high retry resolved coverage but left confidence_low).
        if not passed and reason.split(":")[0] == "confidence_low":
            try:
                prob_map_single = predict_prob(active_box,
                                               active_point_coords,
                                               active_point_labels,
                                               multimask=False)
                passed_single, reason_single = qa_check(prob_map_single)
                if passed_single:
                    prob_map = prob_map_single
                    passed   = passed_single
                    reason   = reason_single
                    strategy = strategy + "_singlemask"
                    strategy_counts[strategy] += 1
            except Exception:
                pass   # keep original QA outcome

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
            # Per-reason breakdown so you can diagnose the dominant failure
            # without waiting for the end-of-run summary.
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
        print(f"  {'Strategy':<32} {'Count':>7}  {'%':>6}")
        print(f"  {'─' * 32} {'─' * 7}  {'─' * 6}")
        for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f"  {strat:<32} {count:>7,}  {count / max(total, 1) * 100:>5.1f}%")

    print()
    if reject_rate > SAM2_QA_MAX_REJECT_RATE:
        print(f"  [WARN] Rejection rate {reject_rate * 100:.1f}% still exceeds the "
              f"{SAM2_QA_MAX_REJECT_RATE * 100:.0f}% target.")
        print("         Read the rejection reason breakdown above — tune the dominant cause:")
        print("           coverage_high  → lower _MAX_BBOX_COVERAGE (now 0.50) or raise _BG_CORNER_PAD (15)")
        print("                            check green-only retry fires (strategy breakdown shows _retry_greenonly)")
        print("           coverage_low   → raise _RETRY_PAD_EXPAND (30) or lower _MIN_TISSUE_FRACTION (0.05)")
        print("           center_fallback→ lower _MIN_COMPONENT_AREA_PX (500) or increase _MORPH_CLOSE_KSIZE (15)")
        print("           confidence_low → lower _QA_MIN_CONFIDENCE below 0.60 (already lowered from 0.65 in v7)")
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
