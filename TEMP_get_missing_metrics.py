"""
================================================================================
 TEMP_get_missing_metrics.py — THROWAWAY script, not part of the pipeline
================================================================================
 PURPOSE:
   Fills the three data points still missing for Chapter 4 (Bouncer + Teacher),
   using checkpoints/annotations that should already exist on disk from your
   earlier training runs. Nothing here retrains any model.

   1. Table 4.5  — Bouncer admission rate on held-out test-split maize images.
   2. Table 4.4  — Bouncer CPU inference latency (lat_cpu_ms) for all 3
                   already-trained neural variants.
   3. Table 4.11 — Teacher gold-standard IoU (delegated to the existing,
                   already-correct validate_gold_standard.py — not
                   reimplemented here).

 UPDATE: both fixes below have since landed in the real train_bouncer.py:
   - evaluate_admission_rate() is now called automatically from main() —
     it's no longer dead code. A normal full retrain will now produce
     Table 4.5's data on its own.
   - Its torch.load() call now includes weights_only=False (PyTorch 2.6
     changed that default; the checkpoints contain a numpy scalar the new
     default's safe-unpickler blocks).

 WHAT THIS SCRIPT IS STILL FOR:
   Getting admission rate + latency WITHOUT a full retrain, against
   checkpoints you already have on disk. If you're going to retrain
   train_bouncer.py anyway for some other reason, you don't need this
   script anymore for Table 4.5 — just use the updated train_bouncer.py
   directly. This script remains useful if you don't want to retrain.

   The latency benchmark (benchmark_latency() below) is still NOT wired
   into train_bouncer.py as a standalone function — it only runs inside
   train_variant() during a full retrain. This script's version is still
   the only way to get lat_cpu_ms for EXISTING checkpoints without
   retraining.

 USAGE:
   python TEMP_get_missing_metrics.py
   python TEMP_get_missing_metrics.py --skip-teacher   # Bouncer only, no subprocess call
================================================================================
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import BOUNCER_CKPT_DIR, BOUNCER_IMG_SIZE, LOGS_DIR, GOLD_ANNOTATION_FILE
from train_bouncer import VARIANT_BUILDERS, evaluate_admission_rate


# ══════════════════════════════════════════════════════════════════════════════
# 1. BOUNCER — CPU LATENCY ON EXISTING CHECKPOINTS (no retraining)
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_latency(variant: str) -> dict:
    """
    Same protocol as the one added to train_bouncer.py's train_variant():
    20 forward passes on CPU with a dummy input, first 5 discarded as warmup,
    mean of the remaining 15. Run here against an ALREADY-SAVED checkpoint.
    """
    ckpt_path = BOUNCER_CKPT_DIR / f"bouncer_{variant}_best.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] {variant}: no checkpoint found at {ckpt_path}")
        return {"variant": variant, "lat_cpu_ms": "N/A"}

    model = VARIANT_BUILDERS[variant]().eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    dummy = torch.zeros(1, 3, BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE)
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)
    avg_lat = round(float(np.mean(latencies[5:])), 2)
    print(f"  {variant:<20} {avg_lat:>8.2f} ms/image  (CPU, {BOUNCER_IMG_SIZE}px)")
    return {"variant": variant, "lat_cpu_ms": avg_lat}


def run_bouncer_metrics() -> None:
    print("\n" + "=" * 72)
    print("  [1/2] Bouncer — Admission Rate (Table 4.5)")
    print("=" * 72)
    admission = evaluate_admission_rate("mobilenet_v3_large")
    if not admission:
        print("  [WARN] Admission rate could not be computed — see warnings above.")
        print("         (Most likely: global_split_manifest.csv or the deployed")
        print("         checkpoint bouncer_mobilenet_v3_large_best.pth not found.)")
    else:
        print(f"\n  → Saved to: {LOGS_DIR / 'bouncer_admission_rate_mobilenet_v3_large.csv'}")
        print("    This file already has exactly the columns Table 4.5 needs.")

    print("\n" + "=" * 72)
    print("  [2/2] Bouncer — CPU Latency, all variants (Table 4.4's lat_cpu_ms)")
    print("=" * 72)
    rows = [benchmark_latency(v) for v in VARIANT_BUILDERS]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / "bouncer_latency_TEMP.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "lat_cpu_ms"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n  → Saved to: {out_path}")
    print("    Note: Gabor+LBP is NOT included here — it has no saved checkpoint")
    print("    to reload (its SVM is fit fresh each run), and it was already ruled")
    print("    out on accuracy grounds, so its latency isn't load-bearing for the")
    print("    deployment decision. Skip it unless you specifically need it.")
    print("\n  Merge this file's lat_cpu_ms column into logs/bouncer_comparison.csv")
    print("  by hand for now (join on 'variant') — or just re-run the patched")
    print("  train_bouncer.py later, which will write it correctly on its own.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEACHER — GOLD-STANDARD IoU (delegated, not reimplemented)
# ══════════════════════════════════════════════════════════════════════════════

def run_teacher_metrics() -> None:
    print("\n" + "=" * 72)
    print("  [3/3] Teacher — Gold-Standard IoU (Table 4.11)")
    print("=" * 72)

    if not Path(GOLD_ANNOTATION_FILE).exists():
        print(f"  [SKIP] {GOLD_ANNOTATION_FILE} not found.")
        print("         This means the human leaf-silhouette annotation pass on the")
        print("         501 gold-standard images (Label Studio/CVAT) hasn't been")
        print("         exported yet — Table 4.11 can't be produced without it,")
        print("         regardless of what script runs. See sample_gold_standard.py.")
        return

    print("  Calling the existing validate_gold_standard.py directly — it already")
    print("  does everything needed here; no new logic required.\n")
    result = subprocess.run([sys.executable, "validate_gold_standard.py"])
    if result.returncode == 0:
        print("\n  → reports/gold_standard_iou_summary.csv now has Table 4.11's data.")
    else:
        print("\n  [WARN] validate_gold_standard.py exited with an error — see output above.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Only run the Bouncer metrics; skip the "
                             "validate_gold_standard.py subprocess call.")
    args = parser.parse_args()

    print("=" * 72)
    print("  TEMPORARY SCRIPT — fills currently-missing Chapter 4 data")
    print("  (Bouncer admission rate + latency, Teacher gold-standard IoU)")
    print("  Delete this file once real numbers are in the pipeline properly.")
    print("=" * 72)

    run_bouncer_metrics()
    if not args.skip_teacher:
        run_teacher_metrics()

    print("\n" + "=" * 72)
    print("  DONE. Outputs to check:")
    print(f"    {LOGS_DIR / 'bouncer_admission_rate_mobilenet_v3_large.csv'}   → Table 4.5")
    print(f"    {LOGS_DIR / 'bouncer_latency_TEMP.csv'}                         → Table 4.4 (lat_cpu_ms)")
    print("    reports/gold_standard_iou_summary.csv                           → Table 4.11")
    print("=" * 72)


if __name__ == "__main__":
    main()
