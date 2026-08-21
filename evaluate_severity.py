"""
================================================================================
 evaluate_severity.py — Phase 6b: Inter-rater Severity Reliability Check
================================================================================
 PURPOSE:
   Ground the severity MAE disclaimer with empirical evidence.
   Protocol:
     1. Sample 60 test-split images (20 per class).
     2. Two raters independently assign 0–3 severity scores.
     3. Compute inter-rater Cohen's Kappa (human agreement).
     4. Compute Spearman correlation between human ratings and Factory
        pseudo-label severity (the pipeline's automated severity %, sourced
        from the Symptom Teacher via factory_master.py — not HSV
        thresholding; the HSV-band functions in factory_master.py are
        dead code and were never the actual source).

   A Spearman ρ ≥ 0.60 supports "moderate correlation with expert assessment."
   If ρ < 0.50, the severity disclaimer must be made stronger in Chapter 4.

 USAGE:
   python evaluate_severity.py --sample       # Step 1: generate sample CSV
   python evaluate_severity.py --analyze      # Step 2: analyze after rating

 RATER INSTRUCTIONS (add to reports/severity_rating_guide.txt):
   0 = No visible symptoms (healthy green tissue throughout)
   1 = Mild — < 25% of visible leaf area shows symptoms
   2 = Moderate — 25–60% of visible leaf area shows symptoms
   3 = Severe — > 60% of visible leaf area shows symptoms

 OUTPUTS:
   reports/severity_sample.csv     — 60 images for raters to fill in
   reports/severity_analysis.csv   — correlation results
   logs/severity_reliability.csv   — kappa + spearman for thesis table
================================================================================
"""

import argparse
import csv
import time as _time
import random
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    SEED, GLOBAL_MANIFEST, PSEUDO_DIR,
    REPORTS_DIR, LOGS_DIR,
    CLASSES, CLASS_TO_IDX,
    STUDENT_BEST_VARIANT, STUDENT_FACTORY_MODE,
    SEVERITY_EVAL_N_IMAGES, SEVERITY_EVAL_SCALE_MAX,
)


SAMPLE_CSV   = REPORTS_DIR / "severity_sample.csv"
ANALYSIS_CSV = REPORTS_DIR / "severity_analysis.csv"
RELIAB_CSV   = LOGS_DIR    / "severity_reliability.csv"
RATING_GUIDE = REPORTS_DIR / "severity_rating_guide.txt"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE SAMPLE
# ══════════════════════════════════════════════════════════════════════════════

def generate_sample() -> None:
    """
    Sample 60 test-split images (20 per class) and write a CSV for raters.
    Raters fill in 'rater1_score' and 'rater2_score' columns manually.
    """
    random.seed(SEED)

    print("  Generating severity rating sample ...")

    # Find best mode from stage 2 comparison if available
    mode = STUDENT_FACTORY_MODE
    stage2_comp = LOGS_DIR / "student_comparison_stage2.csv"
    if stage2_comp.exists():
        df2  = pd.read_csv(stage2_comp)
        mode = df2.loc[df2["best_composite"].idxmax(), "mode"]

    mode_dir = PSEUDO_DIR / mode

    # Load test-split manifest
    global_df = pd.read_csv(GLOBAL_MANIFEST)
    test_df   = global_df[global_df["split"] == "test"]

    rows = []
    for cls in CLASSES:
        cls_df = test_df[test_df["category"] == cls]

        # Filter to images that have severity scores
        valid = []
        for _, row in cls_df.iterrows():
            stem     = Path(row["source_path"]).stem
            sev_path = mode_dir / f"{stem}_sev.txt"
            if sev_path.exists():
                sev = float(sev_path.read_text().strip())
                if sev >= 0:
                    valid.append({
                        "source_path": row["source_path"],
                        "stem":        stem,
                        "category":    cls,
                        "factory_severity":round(sev, 2),
                    })

        n_sample = min(SEVERITY_EVAL_N_IMAGES, len(valid))
        sampled  = random.sample(valid, n_sample)
        rows.extend(sampled)

    # Write sample CSV with empty rater columns
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stem", "source_path", "category",
        "factory_mode",  # recorded so same images rated even if mode changes
        "factory_severity",
        "rater1_score",  # FILL IN: 0, 1, 2, or 3
        "rater2_score",  # FILL IN: 0, 1, 2, or 3
        "notes",
    ]
    with open(SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "stem":         row["stem"],
                "source_path":  row["source_path"],
                "category":     row["category"],
                "factory_mode": mode,
                "factory_severity": row["factory_severity"],
                "rater1_score": "",
                "rater2_score": "",
                "notes":        "",
            })

    # Write rating guide
    RATING_GUIDE.write_text("""
SEVERITY RATING GUIDE — Yellow MAIze Thesis
============================================

Rate each maize leaf image on a 0–3 scale:

  0 = No visible symptoms
      Leaf is uniformly green. No streaks, chlorosis, or necrosis visible.

  1 = Mild  (< 25% of visible leaf area affected)
      Fine chlorotic streaks visible but limited. Leaf is mostly green.
      For MLN: scattered small necrotic spots or slight edge browning.

  2 = Moderate  (25–60% of visible leaf area affected)
      Prominent streaks or chlorosis covering roughly one quarter to
      more than half the leaf area. Leaf colour clearly uneven.

  3 = Severe  (> 60% of visible leaf area affected)
      Extensive streaking, bleaching, or necrosis dominating the leaf.
      Leaf may appear pale yellow, white, or brown throughout.

INSTRUCTIONS:
  - Rate based on what is visible in the image, not crop stage or leaf age.
  - If the leaf is partially out of frame, rate only the visible portion.
  - Assign scores independently (Rater 1 and Rater 2 should not consult each other).
  - Fill in 'rater1_score' and 'rater2_score' columns in severity_sample.csv.
  - Add any notes (occlusion, poor lighting, etc.) in the 'notes' column.
""")

    print(f"  Sample CSV   : {SAMPLE_CSV}  ({len(rows)} images)")
    print(f"  Rating guide : {RATING_GUIDE}")
    print(f"\n  NEXT STEPS:")
    print(f"    1. Two raters open {SAMPLE_CSV}")
    print(f"    2. Each rates every image independently (0–3 scale)")
    print(f"    3. Fill in rater1_score and rater2_score columns")
    print(f"    4. Save the CSV and run:")
    print(f"       python evaluate_severity.py --analyze")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ANALYZE RATINGS
# ══════════════════════════════════════════════════════════════════════════════

def cohen_kappa(r1: np.ndarray, r2: np.ndarray,
                n_categories: int = 4) -> float:
    """
    Cohen's Kappa for inter-rater agreement.
    Accounts for chance agreement unlike raw percent agreement.
    """
    n = len(r1)
    if n == 0:
        return float("nan")

    # Observed agreement
    p_o = np.mean(r1 == r2)

    # Expected agreement
    p_e = 0.0
    for cat in range(n_categories):
        p_e += (np.sum(r1 == cat) / n) * (np.sum(r2 == cat) / n)

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman rank correlation coefficient.
    Measures monotonic relationship between Factory pseudo-label severity
    and human ratings.
    """
    from scipy.stats import spearmanr
    rho, pval = spearmanr(x, y)
    return float(rho), float(pval)


def analyze_ratings() -> None:
    """
    Analyze rater scores from filled-in severity_sample.csv.
    Compute inter-rater Cohen's Kappa and Spearman ρ vs Factory pseudo-label severity.
    """
    if not SAMPLE_CSV.exists():
        print(f"[FATAL] {SAMPLE_CSV} not found. Run --sample first.")
        return

    df = pd.read_csv(SAMPLE_CSV)

    # Check rater columns are filled
    missing = df["rater1_score"].isna().sum() + df["rater2_score"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} rater scores are missing. "
              f"Proceeding with available data.")

    df = df.dropna(subset=["rater1_score", "rater2_score", "factory_severity"])
    if df.empty:
        print("[FATAL] No complete rows found. Fill in rater scores first.")
        return

    r1  = df["rater1_score"].astype(int).values
    r2  = df["rater2_score"].astype(int).values
    factory_sev = df["factory_severity"].astype(float).values

    # Average human rating (normalized to [0,1])
    human_avg  = (r1 + r2) / 2.0
    human_norm = human_avg / SEVERITY_EVAL_SCALE_MAX
    factory_norm   = factory_sev / 100.0

    # Inter-rater agreement
    kappa = cohen_kappa(r1, r2)
    raw_agree = float(np.mean(r1 == r2))

    # Spearman correlation
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(human_norm, factory_norm)
    except ImportError:
        # Manual Spearman if scipy unavailable
        rho  = np.corrcoef(
            pd.Series(human_norm).rank(),
            pd.Series(factory_norm).rank()
        )[0, 1]
        pval = float("nan")

    # Per-class analysis
    print(f"\n  Inter-rater reliability:")
    print(f"    Raw agreement : {raw_agree:.3f}  ({raw_agree*100:.1f}%)")
    print(f"    Cohen's Kappa : {kappa:.4f}")
    if kappa < 0.4:
        print(f"    [WARN] Kappa < 0.40 — poor agreement. Consider re-rating.")
    elif kappa < 0.6:
        print(f"    [OK]   Kappa 0.40–0.60 — moderate agreement.")
    else:
        print(f"    [GOOD] Kappa ≥ 0.60 — substantial agreement.")

    print(f"\n  Factory pseudo-label severity vs. human rating:")
    print(f"    Spearman ρ    : {rho:.4f}")
    print(f"    p-value       : {pval:.4f}" if not np.isnan(pval) else "    p-value: N/A")
    if abs(rho) >= 0.60:
        print(f"    [GOOD] ρ ≥ 0.60 — moderate-to-strong correlation.")
        print(f"           Factory pseudo-label severity is a reasonable proxy for human assessment.")
    elif abs(rho) >= 0.40:
        print(f"    [OK]   ρ 0.40–0.60 — weak-to-moderate correlation.")
        print(f"           Strengthen severity disclaimer in Chapter 4.")
    else:
        print(f"    [WARN] ρ < 0.40 — weak correlation.")
        print(f"           Factory pseudo-label severity is a poor proxy. Revise severity framing.")

    # Per-class Spearman
    print(f"\n  Per-class Spearman ρ:")
    for cls in CLASSES:
        cls_df   = df[df["category"] == cls]
        if len(cls_df) < 5:
            print(f"    {cls:<12}: insufficient data")
            continue
        h_cls    = ((cls_df["rater1_score"] + cls_df["rater2_score"]) / 2.0 /
                    SEVERITY_EVAL_SCALE_MAX).values
        factory_cls  = (cls_df["factory_severity"] / 100.0).values
        try:
            from scipy.stats import spearmanr as sr
            rho_c, _ = sr(h_cls, factory_cls)
        except ImportError:
            rho_c = np.corrcoef(
                pd.Series(h_cls).rank(), pd.Series(factory_cls).rank())[0, 1]
        print(f"    {cls:<12}: ρ = {rho_c:.4f}")

    # Write per-image analysis
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    analysis_rows = []
    for _, row in df.iterrows():
        r1v = int(row["rater1_score"])
        r2v = int(row["rater2_score"])
        analysis_rows.append({
            "stem":           row["stem"],
            "category":       row["category"],
            "rater1":         r1v,
            "rater2":         r2v,
            "avg_human":      round((r1v + r2v) / 2.0, 2),
            "agree":          int(r1v == r2v),
            "factory_severity":   round(float(row["factory_severity"]), 2),
            "human_norm":     round((r1v + r2v) / 2.0 / SEVERITY_EVAL_SCALE_MAX, 4),
            "factory_norm":       round(float(row["factory_severity"]) / 100.0, 4),
        })

    with open(ANALYSIS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=analysis_rows[0].keys())
        writer.writeheader()
        writer.writerows(analysis_rows)

    # Write reliability summary for thesis table
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    reliab = {
        "n_images":       len(df),
        "n_per_class":    SEVERITY_EVAL_N_IMAGES,
        "raw_agreement":  round(raw_agree, 4),
        "cohen_kappa":    round(kappa, 4),
        "spearman_rho":   round(float(rho), 4),
        "spearman_pval":  round(float(pval), 4) if not np.isnan(pval) else "nan",
        "rating_scale":   f"0–{SEVERITY_EVAL_SCALE_MAX}",
        "factory_scale":      "0–100%",
    }
    with open(RELIAB_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reliab.keys())
        writer.writeheader()
        writer.writerow(reliab)

    print(f"\n  Analysis saved    : {ANALYSIS_CSV}")
    print(f"  Reliability table : {RELIAB_CSV}")
    print(f"  (Cite this table in Chapter 4 alongside severity MAE disclaimer)")

    # Chapter 4 suggested sentence
    interp = ("moderate" if abs(rho) >= 0.60 else
              "weak-to-moderate" if abs(rho) >= 0.40 else "weak")
    print(f"\n  Suggested Chapter 4 sentence:")
    print(f"  \"Factory pseudo-label severity scores showed {interp} correlation")
    print(f"   (Spearman ρ = {rho:.3f}) with expert visual ratings on a")
    print(f"   {len(df)}-image sample rated by two independent assessors")
    print(f"   (Cohen's κ = {kappa:.3f}), supporting their use as a training")
    print(f"   proxy while acknowledging they do not substitute for")
    print(f"   agronomically validated disease severity ratings.\"")

    print(f"\n  NEXT STEP: python export_tflite.py")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _t_start = _time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",  action="store_true",
                        help="Generate 60-image sample CSV for raters")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze filled-in rater scores")
    args = parser.parse_args()

    print("=" * 72)
    print("  Yellow MAIze | Phase 6b: Severity Inter-Rater Reliability")
    print("=" * 72)

    if args.sample:
        generate_sample()
    elif args.analyze:
        analyze_ratings()
    else:
        print("  Usage:")
        print("    python evaluate_severity.py --sample    # generate rating CSV")
        print("    python evaluate_severity.py --analyze   # after raters fill in scores")

    print("=" * 72)


if __name__ == "__main__":
    main()
