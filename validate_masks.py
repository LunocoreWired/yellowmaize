"""
================================================================================
 validate_masks.py — Phase 2b: QA Validation Report
================================================================================
 PURPOSE:
   Reads tier1_qa_report.csv and produces:
     1. Summary statistics (pass rate, rejection reasons breakdown)
     2. Per-class rejection analysis
     3. 15 qualitative overlay figures (5 per class) saved to reports/
        for manual review and thesis appendix

 Run after generate_tier1_masks.py. Review the overlays before training
 the Teacher — if masks look wrong, adjust SAM2 prompting strategy.
================================================================================
"""

import random
from pathlib import Path
from collections import Counter
import time
from image_utils import load_image_rgb

import cv2
import numpy as np
import pandas as pd

from config import (
    SEED, TIER1_QA_REPORT, TIER1_RAW_DIR,
    TIER1_MASKS_DIR, REPORTS_DIR, CLASSES,
)

OVERLAY_PER_CLASS = 5       # qualitative overlays per class
OVERLAY_ALPHA     = 0.45    # mask overlay transparency


def set_seeds(seed: int) -> None:
    random.seed(seed)


def draw_overlay(img_rgb: np.ndarray,
                 prob_map: np.ndarray,
                 category: str,
                 filename: str) -> np.ndarray:
    """
    Draw SAM2 soft probability map as a green overlay on the original image.
    Boundary contour drawn in red. Probability gradient shown as heatmap.
    """
    h, w    = img_rgb.shape[:2]
    binary  = (prob_map >= 0.5).astype(np.uint8)

    # Resize prob_map to image size if needed
    if prob_map.shape != (h, w):
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
        binary   = (prob_map >= 0.5).astype(np.uint8)

    # Green overlay for foreground
    overlay = img_rgb.copy()
    overlay[binary == 1] = (
        overlay[binary == 1] * (1 - OVERLAY_ALPHA) +
        np.array([0, 200, 0]) * OVERLAY_ALPHA
    ).astype(np.uint8)

    # Red boundary contour
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (220, 0, 0), 2)

    # Probability heatmap strip (right 30px)
    heatmap = cv2.applyColorMap(
        (prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Text label
    label = f"{category} | {filename}"
    cv2.putText(overlay, label, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(overlay, label, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return overlay


def main() -> None:
    _t_start = time.time()
    set_seeds(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Phase 2b: Mask QA Validation")
    print("=" * 72)

    if not TIER1_QA_REPORT.exists():
        print(f"[FATAL] QA report not found: {TIER1_QA_REPORT}")
        print("        Run generate_tier1_masks.py first.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    overlay_dir = REPORTS_DIR / "tier1_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TIER1_QA_REPORT)

    # ── Summary statistics ────────────────────────────────────────────────────
    total    = len(df)
    passed   = (df["status"] == "passed").sum()
    rejected = total - passed

    print(f"\n  Total images : {total:,}")
    print(f"  Passed       : {passed:,}  ({100*passed/total:.1f}%)")
    print(f"  Rejected     : {rejected:,}  ({100*rejected/total:.1f}%)")

    if rejected > 0:
        reasons = Counter(df[df["status"] != "passed"]["reason"].tolist())
        print(f"\n  Rejection reasons:")
        for reason, count in reasons.most_common():
            print(f"    {reason:<30} {count:>5,}  ({100*count/total:.1f}%)")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print(f"\n  {'Class':<12} {'Total':>7} {'Passed':>7} {'Rejected':>8} {'Pass%':>6}")
    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*8} {'─'*6}")
    for cls in CLASSES:
        cls_df   = df[df["category"] == cls]
        cls_pass = (cls_df["status"] == "passed").sum()
        cls_tot  = len(cls_df)
        cls_rej  = cls_tot - cls_pass
        pct      = 100 * cls_pass / max(cls_tot, 1)
        print(f"  {cls:<12} {cls_tot:>7,} {cls_pass:>7,} {cls_rej:>8,} {pct:>5.1f}%")

    # ── Coverage and confidence stats for passed images ────────────────────────
    passed_df = df[df["status"] == "passed"]
    if not passed_df.empty:
        print(f"\n  Passed image statistics:")
        print(f"    Coverage  — mean: {passed_df['coverage'].mean():.3f}"
              f"  std: {passed_df['coverage'].std():.3f}"
              f"  min: {passed_df['coverage'].min():.3f}"
              f"  max: {passed_df['coverage'].max():.3f}")
        print(f"    Confidence — mean: {passed_df['mean_conf'].mean():.3f}"
              f"  std: {passed_df['mean_conf'].std():.3f}"
              f"  min: {passed_df['mean_conf'].min():.3f}"
              f"  max: {passed_df['mean_conf'].max():.3f}")

    # ── Qualitative overlays (5 per class) ───────────────────────────────────
    print(f"\n  Generating {OVERLAY_PER_CLASS} overlay figures per class ...")

    for cls in CLASSES:
        cls_passed = passed_df[passed_df["category"] == cls]
        if cls_passed.empty:
            print(f"  [WARN] No passed images for {cls}. Skipping overlays.")
            continue

        samples = cls_passed.sample(
            min(OVERLAY_PER_CLASS, len(cls_passed)), random_state=SEED)

        for _, row in samples.iterrows():
            fname   = row["filename"]
            stem    = Path(fname).stem

            img_path = TIER1_RAW_DIR / fname
            npy_path = TIER1_MASKS_DIR / f"{stem}_softmask.npy"

            if not img_path.exists() or not npy_path.exists():
                continue

            try:
                # EXIF-corrected load with truncation guard
                img_rgb = load_image_rgb(img_path)
                if img_rgb is None:
                    continue   # corrupt/truncated — skip overlay
                prob_map = np.load(str(npy_path))
                overlay  = draw_overlay(img_rgb, prob_map, cls, fname)

                # Save as BGR for cv2.imwrite
                out_path = overlay_dir / f"{stem}_overlay.jpg"
                cv2.imwrite(str(out_path),
                            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"  [WARN] Could not generate overlay for {fname}: {e}")

    print(f"  Overlays saved to: {overlay_dir}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    reject_rate = rejected / max(total, 1)
    print(f"\n{'─' * 72}")
    if reject_rate > 0.08:
        print(f"  [WARN] Rejection rate {reject_rate*100:.1f}% is above 8%.")
        print("         Review overlays. Consider adjusting SAM2 HSV thresholds.")
        print("         If cogon / background is being segmented, tighten green")
        print("         HSV range in config.py (SAM2_GREEN_H_MIN/MAX).")
    else:
        print(f"  [OK] Rejection rate {reject_rate*100:.1f}% — acceptable.")
        print("       Review overlays visually before proceeding.")

    print(f"\n  NEXT STEP: python train_teacher.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
