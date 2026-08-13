"""
================================================================================
 select_best_pipeline.py — Best Pipeline Selection & Consolidated Report
================================================================================
 PURPOSE:
   Run after ALL training stages complete. This script:

   1. Reads all comparison CSVs (Bouncer, Teacher, Student Stage 1 + 2)
   2. Selects the best variant per component by primary metric
   3. Copies the winning checkpoint for each component to checkpoints/final/
      with canonical names used by all downstream scripts:
        checkpoints/final/bouncer_best.pth
        checkpoints/final/teacher_best.pth
        checkpoints/final/student_best.pth
   4. Auto-updates STUDENT_BEST_VARIANT and STUDENT_FACTORY_MODE in config.py
   5. Writes:
        reports/best_pipeline_summary.csv  — all winners + their metrics
        reports/best_pipeline_summary.txt  — human-readable thesis table
        reports/all_variants_ranked.csv    — every variant ranked per component
   6. Prints a thesis-ready results table with timing included

 RUN AFTER:
   python train_bouncer.py
   python train_teacher.py
   python train_student.py --stage 1
   python train_student.py --stage 2

 PRIMARY METRICS PER COMPONENT:
   Bouncer  → Specificity (maximise) subject to maize recall ≥ 95%
   Teacher  → Dice (maximise)
   Student  → Composite score (0.5×mIoU + 0.35×MSV_F1 + 0.15×(1−NormMAE))
================================================================================
"""

import csv
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import torch

from config import (
    LOGS_DIR, REPORTS_DIR, CHECKPOINTS_DIR,
    BOUNCER_CKPT_DIR, TEACHER_CKPT_DIR, STUDENT_CKPT_DIR,
    BOUNCER_MIN_MAIZE_RECALL,
    CLASSES,
)

FINAL_DIR    = CHECKPOINTS_DIR / "final"
SUMMARY_CSV  = REPORTS_DIR / "best_pipeline_summary.csv"
SUMMARY_TXT  = REPORTS_DIR / "best_pipeline_summary.txt"
RANKED_CSV   = REPORTS_DIR / "all_variants_ranked.csv"
MOBILE_RANKED_CSV = REPORTS_DIR / "student_mobile_ranking.csv"
CONFIG_PATH  = Path(__file__).parent / "config.py"

# ══════════════════════════════════════════════════════════════════════════════
# MOBILE-AWARE COMPOSITE SCORE WEIGHTS & TARGETS
# ══════════════════════════════════════════════════════════════════════════════
# Weights reflect deployment priority for a Philippine smallholder farmer app:
#   Clinical accuracy is primary — MSV_F1 gets highest single weight.
#   Segmentation quality matters for XAI overlay visualisation.
#   Speed is significant — total pipeline must be usable on mid-range devices.
#   Model size affects download and storage — minor differentiator here.
#   Severity calibration improves trust but is a secondary feature.
#
# Formula:
#   mobile_composite = W_MSV_F1   × msv_f1
#                    + W_MIOU     × sil_mIoU
#                    + W_SPEED    × speed_score
#                    + W_SEV      × (1 − norm_mae)
#                    + W_SIZE     × size_score
#
#   speed_score = clamp(TARGET_LATENCY_MS / cpu_lat_mean_ms, 0, 1)
#   size_score  = clamp(TARGET_SIZE_MB    / tflite_size_mb,  0, 1)
#
# Rationale for 150ms / 15MB targets:
#   150ms: real-time "point and diagnose" feel on mid-range Android (Snapdragon 680)
#          total pipeline (Bouncer ~30ms + Student ~120ms) fits under 250ms
#   15MB:  comfortable for app size; MobileNet variants will score 1.0 here

W_MSV_F1   = 0.32   # Primary clinical metric — MSV detection is the thesis claim
W_ROC_AUC  = 0.18   # MSV-vs-rest ROC-AUC — threshold-independent detection quality
W_MIOU     = 0.18   # Segmentation quality — directly visible as overlay in app
W_SPEED    = 0.16   # Inference speed — critical for mobile usability
W_MLN_F1   = 0.08   # MLN detection — prevents degenerate MSV-only model
W_SEV      = 0.05   # Severity calibration — secondary feature
W_SIZE     = 0.03   # Model size — MobileNet variants all score near 1.0 here
# Sum = 1.00

TARGET_LATENCY_MS = 150.0   # ms — target CPU latency per image
TARGET_SIZE_MB    = 15.0    # MB — target TFLite model file size
TFLITE_VARIANTS_INCOMPATIBLE = {"mobilevit_xxs"}  # excluded from deployment


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [WARN] Not found: {path.name}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"  [WARN] Empty: {path.name}")
        return None
    return df


def copy_canonical(src: Path, dest: Path, label: str) -> bool:
    if not src.exists():
        print(f"  [WARN] Source checkpoint not found: {src}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  ✓ {label} → {dest.name}")
    return True


def update_config(key: str, value: str) -> None:
    """Update a string config value in config.py."""
    content = CONFIG_PATH.read_text()
    # Match: KEY = "old_value" or KEY = 'old_value'
    pattern = rf'^({re.escape(key)}\s*=\s*)["\']([^"\']*)["\']'
    replacement = rf'\g<1>"{value}"'
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    if new_content == content:
        print(f"  [WARN] Could not update {key} in config.py — update manually")
    else:
        CONFIG_PATH.write_text(new_content)
        print(f"  ✓ config.py: {key} = \"{value}\"")


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT SELECTORS
# ══════════════════════════════════════════════════════════════════════════════

def select_best_bouncer() -> dict | None:
    """
    Primary metric: Specificity (TN rate — non-maize rejection).
    Constraint: maize_recall ≥ BOUNCER_MIN_MAIZE_RECALL (0.95).
    """
    df = load_csv_safe(LOGS_DIR / "bouncer_comparison.csv")
    if df is None:
        return None

    # Convert metrics to numeric
    df["specificity"]  = pd.to_numeric(df["specificity"],  errors="coerce")
    df["maize_recall"] = pd.to_numeric(df["maize_recall"], errors="coerce")
    df["roc_auc"]      = pd.to_numeric(df.get("roc_auc", pd.Series(dtype=float)),
                                        errors="coerce")

    # Filter: only neural variants (skip gabor_lbp, patchcore for deployment)
    neural = df[df["variant"].isin(["mobilenet_v3_large", "edgevit_xxs"])].copy()
    neural = neural.dropna(subset=["specificity", "maize_recall"])

    # Apply recall constraint
    eligible = neural[neural["maize_recall"] >= BOUNCER_MIN_MAIZE_RECALL]
    if eligible.empty:
        print("  [WARN] No Bouncer variant meets recall ≥ 0.95 — "
              "selecting by specificity only")
        eligible = neural

    # Primary: highest specificity
    # Tiebreaker: ROC-AUC when specificity within 0.005 of best
    best_spec = eligible["specificity"].max()
    near_best = eligible[eligible["specificity"] >= best_spec - 0.005]
    if len(near_best) > 1 and "roc_auc" in near_best.columns:
        near_best = near_best.dropna(subset=["roc_auc"])
        if not near_best.empty:
            best_row = near_best.loc[near_best["roc_auc"].idxmax()]
        else:
            best_row = eligible.loc[eligible["specificity"].idxmax()]
    else:
        best_row = eligible.loc[eligible["specificity"].idxmax()]
    return best_row.to_dict()


def select_best_teacher() -> dict | None:
    """Primary metric: Dice coefficient."""
    df = load_csv_safe(LOGS_DIR / "teacher_comparison.csv")
    if df is None:
        return None
    df["best_dice"] = pd.to_numeric(df["best_dice"], errors="coerce")
    best_row = df.loc[df["best_dice"].idxmax()]
    return best_row.to_dict()


def mobile_composite_score(test_metrics: dict,
                            tflite_size_mb: float | None = None) -> float:
    """
    Mobile-aware composite score for Student model selection.
    Balances clinical accuracy with real-world mobile deployment requirements.

    Weights (sum = 1.0):
      MSV F1        0.38  — primary clinical metric (thesis claim)
      Silhouette IoU 0.22 — segmentation quality (XAI overlay visibility)
      Speed score   0.22  — inference latency (mobile usability)
      Severity MAE  0.10  — severity calibration (secondary feature)
      Size score    0.08  — TFLite model size (download / storage)

    Speed score  = clamp(TARGET_LATENCY_MS / cpu_lat_mean_ms, 0, 1)
    Size score   = clamp(TARGET_SIZE_MB    / tflite_size_mb,  0, 1)
      → If TFLite size unknown: size_score defaults to 1.0 (optimistic)
        (conservative assumption — MobileNet variants are typically small enough)

    Citation basis:
      Clinical accuracy weights: Cruz et al. (2024), Mushayi et al. (2025)
      Latency target (150ms): realistic for Snapdragon 680-class devices
      Size target (15MB): comfortable for app store distribution
    """
    def _f(key: str, default: float = 0.0) -> float:
        v = test_metrics.get(key, default)
        try:
            return float(str(v).replace("N/A", str(default)) or default)
        except (ValueError, TypeError):
            return default

    msv_f1    = _f("msv_f1")
    mln_f1    = _f("mln_f1")
    sil_miou  = _f("sil_mIoU")
    msv_roc   = _f("msv_roc_auc", msv_f1)  # fallback to msv_f1 if AUC not yet computed
    sev_mae   = _f("sev_mae_pct", 100.0)   # percentage — normalise to [0,1]
    lat_ms    = _f("cpu_lat_mean_ms", 999.0)

    # Severity: lower MAE is better → convert to 0–1 score
    norm_mae    = min(sev_mae / 100.0, 1.0)

    # Speed score: 150ms target, capped at 1.0
    speed_score = min(TARGET_LATENCY_MS / max(lat_ms, 1.0), 1.0)

    # Size score: 15MB target, capped at 1.0; default 1.0 if unknown
    # (MobileNet variants are all small — optimistic default is more defensible
    #  than 0.80 which would arbitrarily penalise unexported variants)
    if tflite_size_mb and tflite_size_mb > 0:
        size_score = min(TARGET_SIZE_MB / max(tflite_size_mb, 0.1), 1.0)
    else:
        size_score = 1.0   # optimistic default — noted in report

    score = (W_MSV_F1  * msv_f1
           + W_ROC_AUC * msv_roc
           + W_MIOU    * sil_miou
           + W_SPEED   * speed_score
           + W_MLN_F1  * mln_f1
           + W_SEV     * (1.0 - norm_mae)
           + W_SIZE    * size_score)

    return round(score, 6)


def get_tflite_size(variant: str, mode: str) -> float | None:
    """Try to get TFLite size from export report or estimate from checkpoint."""
    from config import EXPORTS_DIR
    exp_csv = EXPORTS_DIR / "export_report.csv"
    if exp_csv.exists():
        df = pd.read_csv(exp_csv)
        if not df.empty and "tflite_size_mb" in df.columns:
            row = df.iloc[0]
            if str(row.get("encoder_variant","")) == variant:
                v = pd.to_numeric(row.get("tflite_size_mb", None), errors="coerce")
                if pd.notna(v):
                    return float(v)
    # Estimate from checkpoint size (~3× for FP16 conversion overhead)
    for ckpt_path in [
        STUDENT_CKPT_DIR / f"student_{variant}_{mode}_best.pth",
        STUDENT_CKPT_DIR / "stage1" / f"student_{variant}_{mode}_best.pth",
    ]:
        if ckpt_path.exists():
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            return round(size_mb * 0.5, 2)   # FP16 ≈ half the FP32 size
    return None


def select_best_student() -> dict | None:
    """
    Two-stage selection with mobile-aware re-ranking.

    Stage 1 (during training): Quality composite selects best checkpoint.
    Stage 2 (here, post-training): Mobile-aware composite re-ranks all
      completed runs using accuracy + speed + size simultaneously.
      This is the correct point for mobile-aware selection because
      CPU latency is now available from student_test_metrics CSVs.

    TFLite-incompatible variants (mobilevit_xxs) are excluded from
    deployment selection but their metrics are still reported for comparison.
    """
    # ── Collect all completed test metrics CSVs ──────────────────────────────
    all_runs = []
    for f in LOGS_DIR.glob("student_test_metrics_*.csv"):
        df = pd.read_csv(f)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        variant = str(row.get("variant",""))
        mode    = str(row.get("mode",""))
        if not variant or not mode:
            continue
        # Get TFLite size estimate
        tflite_mb = get_tflite_size(variant, mode)
        # Compute mobile composite
        mob_score = mobile_composite_score(row, tflite_mb)
        all_runs.append({
            **row,
            "tflite_size_mb_est": tflite_mb,
            "mobile_composite":   mob_score,
            "tflite_compatible":  variant not in TFLITE_VARIANTS_INCOMPATIBLE,
        })

    if all_runs:
        # Filter to TFLite-compatible variants for deployment selection
        deployable = [r for r in all_runs
                      if r.get("tflite_compatible", True)]
        if not deployable:
            deployable = all_runs   # fallback — all for reference

        # Best by mobile composite
        best = max(deployable, key=lambda r: r["mobile_composite"])

        # Save full mobile ranking CSV
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        sorted_runs = sorted(all_runs,
                              key=lambda r: r["mobile_composite"], reverse=True)
        if sorted_runs:
            with open(MOBILE_RANKED_CSV, "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(
                    f, fieldnames=sorted_runs[0].keys(),
                    extrasaction="ignore")
                writer.writeheader()
                writer.writerows(sorted_runs)
            print(f"  Mobile ranking → {MOBILE_RANKED_CSV}")

        # Print comparison table
        print(f"\n  {'Encoder':<22} {'Mode':<8} {'MSV_F1':>7} "
              f"{'mIoU':>6} {'CPU ms':>7} {'MB':>5} {'Mobile':>8} {'Deploy':>7}")
        print(f"  {'─'*22} {'─'*8} {'─'*7} {'─'*6} {'─'*7} {'─'*5} {'─'*8} {'─'*7}")
        for r in sorted_runs:
            v   = str(r.get("variant",""))[:22]
            m   = str(r.get("mode",""))[:8]
            f1  = f"{float(r.get('msv_f1',0)):.4f}"
            iou = f"{float(r.get('sil_mIoU',0)):.4f}"
            lat = f"{r.get('cpu_lat_mean_ms','?')}"
            mb  = f"{r.get('tflite_size_mb_est','?')}"
            mob = f"{r.get('mobile_composite',0):.4f}"
            dep = "YES" if r.get("tflite_compatible") else "no"
            marker = " ←" if r is best else ""
            print(f"  {v:<22} {m:<8} {f1:>7} {iou:>6} "
                  f"{lat:>7} {mb:>5} {mob:>8}{marker}  {dep:>7}")

        return {
            "encoder":          best.get("variant",""),
            "mode":             best.get("mode","mode_b"),
            "best_composite":   best.get("composite", 0),
            "mobile_composite": best.get("mobile_composite", 0),
        }

    # ── Fallback: use comparison CSVs if no test metrics exist yet ───────────
    print("  [WARN] No student_test_metrics CSVs found. "
          "Run train_student.py --stage 2 first for mobile-aware selection.")
    stage2 = load_csv_safe(LOGS_DIR / "student_comparison_stage2.csv")
    if stage2 is not None:
        stage2["best_composite"] = pd.to_numeric(
            stage2["best_composite"], errors="coerce")
        best_row = stage2.loc[stage2["best_composite"].idxmax()]
        return best_row.to_dict()

    stage1 = load_csv_safe(LOGS_DIR / "student_comparison_stage1.csv")
    if stage1 is not None:
        stage1["best_composite"] = pd.to_numeric(
            stage1["best_composite"], errors="coerce")
        best_row = stage1.loc[stage1["best_composite"].idxmax()]
        result   = best_row.to_dict()
        result["mode"] = "mode_b"
        return result
    return None


def get_student_test_metrics(variant: str, mode: str) -> dict:
    """Load the student test metrics CSV for the best variant+mode."""
    path = LOGS_DIR / f"student_test_metrics_{variant}_{mode}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if not df.empty else {}


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT PROMOTION
# ══════════════════════════════════════════════════════════════════════════════

def promote_checkpoints(best_bouncer: dict | None,
                         best_teacher: dict | None,
                         best_student: dict | None) -> dict:
    """
    Copy best checkpoints to checkpoints/final/ with canonical names.
    Returns dict of destination paths.
    """
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    if best_bouncer:
        variant = best_bouncer.get("variant", "")
        src     = BOUNCER_CKPT_DIR / f"bouncer_{variant}_best.pth"
        dest    = FINAL_DIR / "bouncer_best.pth"
        if copy_canonical(src, dest, f"Bouncer ({variant})"):
            paths["bouncer"] = dest

    if best_teacher:
        # Teacher best is already at teacher_model_best.pth
        src  = TEACHER_CKPT_DIR / "teacher_model_best.pth"
        dest = FINAL_DIR / "teacher_best.pth"
        if copy_canonical(src, dest, f"Teacher ({best_teacher.get('variant','')})"):
            paths["teacher"] = dest

    if best_student:
        variant = best_student.get("encoder", best_student.get("variant", ""))
        mode    = best_student.get("mode", "mode_b")
        # Check Stage 2 first, then Stage 1
        src = STUDENT_CKPT_DIR / f"student_{variant}_{mode}_best.pth"
        if not src.exists():
            src = STUDENT_CKPT_DIR / "stage1" / f"student_{variant}_{mode}_best.pth"
        dest = FINAL_DIR / "student_best.pth"
        if copy_canonical(src, dest, f"Student ({variant}, {mode})"):
            paths["student"] = dest

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# RANKED TABLE FOR ALL VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def build_ranked_table() -> list[dict]:
    ranked = []

    # Bouncer variants
    df = load_csv_safe(LOGS_DIR / "bouncer_comparison.csv")
    if df is not None:
        df["component"] = "Bouncer"
        df["primary_metric"] = "specificity"
        df["primary_value"] = pd.to_numeric(df["specificity"], errors="coerce")
        ranked.append(df.assign(rank=df["primary_value"].rank(
            ascending=False, method="min").astype(int)))

    # Teacher variants
    df = load_csv_safe(LOGS_DIR / "teacher_comparison.csv")
    if df is not None:
        df = df.rename(columns={"variant": "variant"})
        df["component"] = "Teacher"
        df["primary_metric"] = "dice"
        df["primary_value"] = pd.to_numeric(df["best_dice"], errors="coerce")
        ranked.append(df.assign(rank=df["primary_value"].rank(
            ascending=False, method="min").astype(int)))

    # Student Stage 1
    df = load_csv_safe(LOGS_DIR / "student_comparison_stage1.csv")
    if df is not None:
        df["component"] = "Student_Encoder"
        df["primary_metric"] = "composite"
        df["primary_value"] = pd.to_numeric(df["best_composite"], errors="coerce")
        ranked.append(df.assign(rank=df["primary_value"].rank(
            ascending=False, method="min").astype(int)))

    # Student Stage 2 — ranked by mobile composite (from test metrics CSVs)
    mobile_df = load_csv_safe(MOBILE_RANKED_CSV)
    if mobile_df is not None:
        mobile_df["component"]      = "Student_Mobile"
        mobile_df["primary_metric"] = "mobile_composite"
        mobile_df["primary_value"]  = pd.to_numeric(
            mobile_df["mobile_composite"], errors="coerce")
        ranked.append(mobile_df.assign(rank=mobile_df["primary_value"].rank(
            ascending=False, method="min").astype(int)))
    else:
        df = load_csv_safe(LOGS_DIR / "student_comparison_stage2.csv")
        if df is not None:
            df["component"]      = "Student_Mode"
            df["primary_metric"] = "composite"
            df["primary_value"]  = pd.to_numeric(df["best_composite"], errors="coerce")
            ranked.append(df.assign(rank=df["primary_value"].rank(
                ascending=False, method="min").astype(int)))

    if not ranked:
        return []

    combined = pd.concat(ranked, ignore_index=True, sort=False)
    return combined.to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(best_bouncer, best_teacher, best_student,
                   test_metrics: dict, canonical_paths: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # ── CSV ────────────────────────────────────────────────────────────────────
    rows = []
    if best_bouncer:
        rows.append({
            "component":      "Bouncer",
            "winner":         best_bouncer.get("variant",""),
            "primary_metric": "specificity",
            "primary_value":  best_bouncer.get("specificity",""),
            "maize_recall":   best_bouncer.get("maize_recall",""),
            "roc_auc":        best_bouncer.get("roc_auc",""),
            "canonical_ckpt": str(canonical_paths.get("bouncer","")),
        })
    if best_teacher:
        rows.append({
            "component":      "Teacher",
            "winner":         best_teacher.get("variant",""),
            "primary_metric": "dice",
            "primary_value":  best_teacher.get("best_dice",""),
            "lat_cpu_ms":     best_teacher.get("lat_cpu_ms",""),
            "canonical_ckpt": str(canonical_paths.get("teacher","")),
        })
    if best_student:
        rows.append({
            "component":        "Student",
            "winner_encoder":   best_student.get("encoder",
                                best_student.get("variant","")),
            "winner_mode":      best_student.get("mode",""),
            "quality_composite":best_student.get("best_composite",""),
            "mobile_composite": best_student.get("mobile_composite",""),
            "selection_basis":  "mobile_composite (MSV_F1×0.32 + ROC_AUC×0.18 + mIoU×0.18 + speed×0.16 + MLN_F1×0.08 + sev×0.05 + size×0.03)",
            "canonical_ckpt":   str(canonical_paths.get("student","")),
            **{f"test_{k}": v for k, v in test_metrics.items()
               if k not in ("variant","mode")},
        })

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        if rows:
            all_keys = list(dict.fromkeys(
                k for r in rows for k in r.keys()))
            writer = csv.DictWriter(f, fieldnames=all_keys,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    # ── TXT (thesis-ready table) ───────────────────────────────────────────────
    lines = [
        "Yellow MAIze — Best Pipeline Summary",
        f"Generated: {ts}",
        "=" * 65,
        "",
        "BEST MODELS PER COMPONENT",
        "─" * 65,
    ]

    if best_bouncer:
        lines += [
            f"  Bouncer : {best_bouncer.get('variant','')}",
            f"    Specificity  : {best_bouncer.get('specificity','N/A')}",
            f"    Maize Recall : {best_bouncer.get('maize_recall','N/A')}",
            f"    ROC-AUC      : {best_bouncer.get('roc_auc','N/A')}",
            f"    Checkpoint   : {canonical_paths.get('bouncer','not promoted')}",
            "",
        ]

    if best_teacher:
        lines += [
            f"  Teacher : {best_teacher.get('variant','')}",
            f"    Val Dice     : {best_teacher.get('best_dice','N/A')}",
            f"    CPU Latency  : {best_teacher.get('lat_cpu_ms','N/A')} ms/image",
            f"    Checkpoint   : {canonical_paths.get('teacher','not promoted')}",
            "",
        ]

    if best_student and test_metrics:
        enc  = best_student.get("encoder", best_student.get("variant",""))
        mode = best_student.get("mode","")
        lines += [
            f"  Student : {enc}  |  Mode: {mode}",
            "  ── Segmentation (silhouette) ──────────────────────────",
            f"    mIoU       : {test_metrics.get('sil_mIoU','N/A')}",
            f"    Dice       : {test_metrics.get('sil_dice','N/A')}",
            f"    Recall     : {test_metrics.get('sil_recall','N/A')}",
            f"    Precision  : {test_metrics.get('sil_precision','N/A')}",
            f"    Specificity: {test_metrics.get('sil_specificity','N/A')}",
            "  ── Segmentation (symptom mask) ────────────────────────",
            f"    mIoU       : {test_metrics.get('sym_mIoU','N/A')}",
            f"    Dice       : {test_metrics.get('sym_dice','N/A')}",
            f"    Recall     : {test_metrics.get('sym_recall','N/A')}",
            "  ── Classification ─────────────────────────────────────",
            f"    Accuracy   : {test_metrics.get('cls_accuracy','N/A')}",
            f"    MCC        : {test_metrics.get('mcc','N/A')}",
            f"    Macro F1   : {test_metrics.get('macro_f1','N/A')}",
            f"    Weighted F1: {test_metrics.get('weighted_f1','N/A')}",
            "  ── Per-Class (MSV is primary) ──────────────────────────",
            f"    HEALTHY    P:{test_metrics.get('healthy_prec','?')}  "
            f"R:{test_metrics.get('healthy_rec','?')}  "
            f"F1:{test_metrics.get('healthy_f1','?')}",
            f"    MSV        P:{test_metrics.get('msv_prec','?')}  "
            f"R:{test_metrics.get('msv_rec','?')}  "
            f"F1:{test_metrics.get('msv_f1','?')}",
            f"    MLN        P:{test_metrics.get('mln_prec','?')}  "
            f"R:{test_metrics.get('mln_rec','?')}  "
            f"F1:{test_metrics.get('mln_f1','?')}",
            "  ── Severity Regression ────────────────────────────────",
            f"    MAE        : {test_metrics.get('sev_mae_pct','N/A')}%",
            f"    RMSE       : {test_metrics.get('sev_rmse_pct','N/A')}%",
            f"    R²         : {test_metrics.get('sev_r2','N/A')}",
            "  ── Quality Composite (training criterion) ─────────────",
            f"    Score      : {test_metrics.get('composite','N/A')}",
            f"    Formula    : 0.50×mIoU + 0.35×MSV_F1 + 0.15×(1−NormMAE)",
            "  ── Mobile Composite (deployment selection) ────────────",
            f"    Score      : {best_student.get('mobile_composite','N/A')}",
            f"    Formula    : MSV_F1×0.32 + ROC_AUC×0.18 + mIoU×0.18 + speed×0.16 + MLN_F1×0.08 + sev×0.05 + size×0.03",
            f"    Target lat : {TARGET_LATENCY_MS}ms  |  Target size: {TARGET_SIZE_MB}MB",
            "  ── Inference Latency (CPU, simulates mobile) ──────────",
            f"    Mean       : {test_metrics.get('cpu_lat_mean_ms','N/A')} ms",
            f"    Std        : {test_metrics.get('cpu_lat_std_ms','N/A')} ms",
            f"    FPS        : {test_metrics.get('cpu_fps','N/A')}",
            f"    Eval time  : {test_metrics.get('eval_duration_s','N/A')}s",
            f"    Checkpoint : {canonical_paths.get('student','not promoted')}",
            f"    Mobile rank: {MOBILE_RANKED_CSV}",
            "",
        ]

    lines += [
        "=" * 65,
        "FILES",
        f"  Summary CSV  : {SUMMARY_CSV}",
        f"  Ranked table : {RANKED_CSV}",
        f"  Final ckpts  : {FINAL_DIR}",
        "=" * 65,
    ]

    SUMMARY_TXT.write_text("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("  Yellow MAIze | Best Pipeline Selection")
    print("=" * 65)

    # ── Select winners ─────────────────────────────────────────────────────────
    print("\n  Selecting best Bouncer ...")
    best_bouncer = select_best_bouncer()
    if best_bouncer:
        print(f"  Winner: {best_bouncer.get('variant')}  "
              f"spec={best_bouncer.get('specificity')}  "
              f"recall={best_bouncer.get('maize_recall')}")

    print("\n  Selecting best Teacher ...")
    best_teacher = select_best_teacher()
    if best_teacher:
        print(f"  Winner: {best_teacher.get('variant')}  "
              f"dice={best_teacher.get('best_dice')}")

    print("\n  Selecting best Student ...")
    best_student = select_best_student()
    if best_student:
        enc  = best_student.get("encoder", best_student.get("variant",""))
        mode = best_student.get("mode","")
        print(f"  Winner: {enc}  mode={mode}  "
              f"composite={best_student.get('best_composite')}")

    # ── Load test metrics for best student ────────────────────────────────────
    test_metrics = {}
    if best_student:
        enc  = best_student.get("encoder", best_student.get("variant",""))
        mode = best_student.get("mode","mode_b")
        test_metrics = get_student_test_metrics(enc, mode)
        if not test_metrics:
            print(f"  [WARN] Test metrics not found for {enc}/{mode}. "
                  f"Run train_student.py --stage 2 first.")

    # ── Promote checkpoints ────────────────────────────────────────────────────
    print("\n  Promoting best checkpoints to checkpoints/final/ ...")
    canonical = promote_checkpoints(best_bouncer, best_teacher, best_student)

    # ── Update config.py ──────────────────────────────────────────────────────
    if best_student:
        enc  = best_student.get("encoder", best_student.get("variant",""))
        mode = best_student.get("mode","mode_b")
        print("\n  Updating config.py ...")
        update_config("STUDENT_BEST_VARIANT", enc)
        update_config("STUDENT_FACTORY_MODE", mode)

    # ── Ranked table ──────────────────────────────────────────────────────────
    print("\n  Building ranked comparison table ...")
    ranked = build_ranked_table()
    if ranked:
        with open(RANKED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(dict.fromkeys(
                    k for r in ranked for k in r.keys())),
                extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ranked)
        print(f"  Ranked table → {RANKED_CSV}")

    # ── Write summary ──────────────────────────────────────────────────────────
    write_summary(best_bouncer, best_teacher, best_student,
                   test_metrics, canonical)

    # ── Print thesis-ready summary ─────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(SUMMARY_TXT.read_text())

    print(f"  Summary CSV → {SUMMARY_CSV}")
    print(f"  Summary TXT → {SUMMARY_TXT}")
    print(f"  Final ckpts → {FINAL_DIR}")
    print(f"\n  NEXT STEP: python export_tflite.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
