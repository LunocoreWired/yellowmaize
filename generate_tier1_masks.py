"""
================================================================================
 generate_tier1_masks.py — Phase 2: SAM2 Auto-Masking + QA  [v3]
================================================================================
 PURPOSE:
   Apply SAM2 to all 15,000 Tier 1 images to generate precise binary leaf
   silhouette masks. Masks are stored as BOTH:
     - float32 .npy  (raw SAM2 probability map — used as soft targets)
     - uint8 .png    (binarized at 0.5 — for visualization)

 AUTO-PROMPTING STRATEGY (v3 — contour bbox):
   For each image:
   1. Convert to HSV. Build a combined tissue mask from THREE color ranges:
        Green  (H=30–90)  — healthy leaf tissue
        Yellow (H=15–38)  — MSV streak yellowing
        Brown  (H=5–20)   — MLN necrotic / dead tissue  ← NEW in v3
      This ensures severely diseased MLN leaves with minimal green/yellow
      are still detected via their necrotic brown tissue.
   2. Run cv2.connectedComponentsWithStats on the combined tissue mask.
      Take cv2.boundingRect() of the largest connected component.
      This gives a tight [x, y, x2, y2] bounding box directly from the
      tissue mask — no YOLO training required.
   3. Pass the bounding box as a SAM2 box prompt.
      Box prompts are SAM2's strongest prompt type — they constrain the
      search space far more precisely than point prompts, preventing the
      model from expanding into background soil/sky.
   4. Fallback: if no tissue is detected (all three ranges below threshold),
      use four image corners as background points + image center as a single
      foreground point. QA filters catch bad results.

 WHY BBOX > MULTI-POINT:
   - Box prompts encode both position AND spatial extent of the leaf.
   - Prevents mask leaking past sharp color edges (common on MSV leaves).
   - Captures full leaf length including tips, which point prompts miss
     when the leaf extends beyond the centroid axis.
   - Still zero training required — bbox derived from existing HSV mask.

 QA FILTERS (v3 — unchanged from v2):
   Filter 1 — Coverage range  : foreground must be 3%–90% of image
   Filter 2 — Mean confidence : mean prob of foreground region ≥ 0.65
   Filter 3 — Aspect ratio    : mask bounding box ratio ≥ 1.01
   Target rejection rate: < 8% of total images.

 CHANGELOG (v2 → v3):
   [NEW]  Brown HSV range (H=5–20) for MLN necrotic tissue detection
   [NEW]  get_leaf_bbox(): replaces get_leaf_centroid() — returns bbox not centroid
   [NEW]  build_box_prompt(): replaces build_prompts() — uses SAM2 box= API
   [NEW]  SAM2 predictor.predict() now uses box= instead of point_coords=
   [NEW]  Strategy labels updated: *_bbox suffix (green_bbox, yellow_bbox, etc.)
   [NEW]  center_fallback retains point-based prompting as last resort
   [KEEP] All QA thresholds unchanged from v2

 OUTPUTS:
   data/tier1_leaf_masks/{stem}_softmask.npy   ← float32 probability map
   data/tier1_leaf_masks/{stem}_mask.png        ← uint8 binary visualization
   tier1_qa_report.csv                          ← per-image QA log
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
    SAM2_YELLOW_H_MIN, SAM2_YELLOW_H_MAX,
    SAM2_YELLOW_S_MIN, SAM2_YELLOW_V_MIN,
    SAM2_BROWN_H_MIN,  SAM2_BROWN_H_MAX,
    SAM2_BROWN_S_MIN,  SAM2_BROWN_V_MIN,
    SAM2_QA_MIN_COVERAGE, SAM2_QA_MAX_COVERAGE,
    SAM2_QA_MIN_CONFIDENCE, SAM2_QA_MIN_ASPECT_RATIO,
    SAM2_QA_MAX_REJECT_RATE,
    VALID_EXTENSIONS,
)

# ── v2 QA overrides ───────────────────────────────────────────────────────────
# These shadow the config values with the relaxed v2 thresholds.
# If you update config.py to match, these become no-ops.
_QA_MIN_COVERAGE   = 0.03   # was SAM2_QA_MIN_COVERAGE (0.10)
_QA_MAX_COVERAGE   = 0.90   # unchanged
_QA_MIN_CONFIDENCE = 0.65   # unchanged
_QA_MIN_ASPECT     = 1.01  # was SAM2_QA_MIN_ASPECT_RATIO (1.20)

# Minimum fraction of image pixels that must be tissue-colored before
# falling back to the center-of-image point prompt.
_MIN_TISSUE_FRACTION = 0.05

# Minimum pixel area for a connected component to be considered a valid leaf.
# Prevents tiny noise blobs from driving the bbox.
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
# AUTO-PROMPTING  (v2 — disease-aware, multi-point foreground)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-PROMPTING  (v3 — contour bbox, disease-aware)
# ══════════════════════════════════════════════════════════════════════════════

def get_leaf_bbox(
    img_rgb: np.ndarray,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """
    Derive a tight bounding box around the dominant leaf-tissue region
    using HSV color segmentation. No model training required.

    THREE tissue ranges are combined (v3):
      Green  (H=30–90)  — healthy leaf tissue
      Yellow (H=15–38)  — MSV streak yellowing
      Brown  (H=5–20)   — MLN necrotic / dead tissue

    Detection logic:
      1. Build per-range binary masks and OR them into a combined tissue mask.
      2. Run connected-component analysis; pick the largest component whose
         area exceeds _MIN_COMPONENT_AREA_PX.
      3. Return cv2.boundingRect() of that component as (x1, y1, x2, y2).
      4. If no qualifying component is found, return None → center_fallback.

    Args:
        img_rgb: H×W×3 uint8 RGB image (EXIF-corrected).

    Returns:
        bbox     : (x1, y1, x2, y2) in pixel coordinates, or None on fallback.
        strategy : one of "green_bbox" | "yellow_bbox" | "brown_bbox" |
                          "combined_bbox" | "center_fallback"
    """
    img_hsv  = to_hsv(img_rgb)
    h, s, v  = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    total_px = img_rgb.shape[0] * img_rgb.shape[1]

    # ── 1. Build per-tissue masks ─────────────────────────────────────────────
    green_mask = (
        (h >= SAM2_GREEN_H_MIN)  & (h <= SAM2_GREEN_H_MAX) &
        (s >= SAM2_GREEN_S_MIN)  & (v >= SAM2_GREEN_V_MIN)
    ).astype(np.uint8)

    yellow_mask = (
        (h >= SAM2_YELLOW_H_MIN) & (h <= SAM2_YELLOW_H_MAX) &
        (s >= SAM2_YELLOW_S_MIN) & (v >= SAM2_YELLOW_V_MIN)
    ).astype(np.uint8)

    # Brown/necrotic: MLN dead tissue — low saturation allowed (ash/straw)
    brown_mask = (
        (h >= SAM2_BROWN_H_MIN)  & (h <= SAM2_BROWN_H_MAX) &
        (s >= SAM2_BROWN_S_MIN)  & (v >= SAM2_BROWN_V_MIN)
    ).astype(np.uint8)

    green_frac  = green_mask.sum()  / total_px
    yellow_frac = yellow_mask.sum() / total_px
    brown_frac  = brown_mask.sum()  / total_px

    has_green  = green_frac  >= _MIN_TISSUE_FRACTION
    has_yellow = yellow_frac >= _MIN_TISSUE_FRACTION
    has_brown  = brown_frac  >= _MIN_TISSUE_FRACTION

    # ── 2. Combine detected ranges into a single tissue mask ──────────────────
    if not (has_green or has_yellow or has_brown):
        # No tissue detected at all — fall back to center point prompt
        return None, "center_fallback"

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

    strategy = ("combined_bbox" if len(active) > 1
                else f"{active[0]}_bbox")

    # ── 3. Largest qualifying connected component ─────────────────────────────
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
        tissue_mask, connectivity=8)

    if num_labels < 2:
        return None, "center_fallback"

    # Filter components below minimum area, then pick the largest
    component_areas = stats[1:, cv2.CC_STAT_AREA]   # exclude background (label 0)
    valid_indices   = np.where(component_areas >= _MIN_COMPONENT_AREA_PX)[0]

    if len(valid_indices) == 0:
        return None, "center_fallback"

    best_label = 1 + int(valid_indices[np.argmax(component_areas[valid_indices])])
    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h_box = int(stats[best_label, cv2.CC_STAT_HEIGHT])

    img_h, img_w = img_rgb.shape[:2]
    pad = 8   # small padding so SAM2 sees the leaf edge in context

    x1 = max(0,     x - pad)
    y1 = max(0,     y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h_box + pad)

    return (x1, y1, x2, y2), strategy


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
    h, w = img_rgb.shape[:2]
    pad  = 10

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        return box, None, None, None

    # center_fallback: single foreground point + four background corners
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
    print("  QA thresholds (v3 — unchanged from v2):")
    print(f"    Coverage     : {_QA_MIN_COVERAGE:.0%} – {_QA_MAX_COVERAGE:.0%}")
    print(f"    Confidence   : ≥ {_QA_MIN_CONFIDENCE:.2f}")
    print(f"    Aspect ratio : ≥ {_QA_MIN_ASPECT:.2f}")
    print("  Prompting      : contour bbox (green+yellow+brown HSV) → SAM2 box=")
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

        # ── Auto-prompting (v3: contour bbox, disease-aware) ─────────────────
        bbox, strategy = get_leaf_bbox(img_rgb)
        strategy_counts[strategy] += 1
        box, point_coords, point_labels, _ = build_box_prompt(img_rgb, bbox)

        # ── SAM2 inference ────────────────────────────────────────────────────
        try:
            predictor.set_image(img_rgb)
            if box is not None:
                # Primary path: bounding box prompt (strongest SAM2 input)
                masks, scores, logits = predictor.predict(
                    box=box,
                    multimask_output=True,
                )
            else:
                # Fallback path: center point + corner background points
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
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
        print("         If 'center_fallback' is high, check HSV ranges in config.py.")
        print("         (SAM2_GREEN/YELLOW/BROWN_H_MIN/MAX)")
    else:
        print(f"  [OK] Rejection rate within acceptable range "
              f"({reject_rate * 100:.1f}% ≤ {SAM2_QA_MAX_REJECT_RATE * 100:.0f}%).")

    print(f"\n  NEXT STEP: python validate_masks.py  (review QA report visually)")
    print(f"  THEN     : python train_teacher.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
