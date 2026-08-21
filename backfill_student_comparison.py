"""
================================================================================
 backfill_student_comparison.py — One-off repair for the train_one() bug
================================================================================
 PROBLEM:
   train_one() in train_student.py used to build its returned row with:
     **{f"test_{k}": v for k, v in test_results.items() if k.startswith("test_")}
   test_results' keys (sil_mIoU, msv_f1, mln_f1, composite, etc.) never
   started with "test_" in the first place, so this filter silently dropped
   EVERY metric. logs/student_comparison_stage{1,2}.csv ended up with only
   encoder/mode/best_composite — none of the metrics generate_charts.py's
   _student_comparison_bar() needs, hence "no expected columns found".

 FIX (already applied in train_student.py): the filter is now
     **{f"test_{k}": v for k, v in test_results.items()}
   so all future runs write the full column set. This script backfills the
   EXISTING comparison CSVs for runs you already completed, by pulling the
   correct data from logs/student_test_metrics_{variant}_{mode}.csv (the
   per-run file, which was NEVER affected by the bug — it's written directly
   from evaluate_test_split()'s own results dict).

 USAGE:
   python backfill_student_comparison.py
   (run once, from the project root, after pulling the train_student.py fix)

 WHAT IT DOES:
   For each row in student_comparison_stage{1,2}.csv, looks up
   student_test_metrics_{encoder}_{mode}.csv and merges its metrics in with
   a "test_" prefix — reproducing exactly what a fresh run would now write,
   without re-training anything.
================================================================================
"""
import csv
from pathlib import Path

from config import LOGS_DIR


def backfill(stage: int) -> None:
    comp_path = LOGS_DIR / f"student_comparison_stage{stage}.csv"
    if not comp_path.exists():
        print(f"  [SKIP] {comp_path.name} not found.")
        return

    with open(comp_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"  [SKIP] {comp_path.name} is empty.")
        return

    fixed_rows = []
    n_backfilled = n_missing = 0

    for row in rows:
        encoder = row.get("encoder", "")
        mode    = row.get("mode", "")
        metrics_path = LOGS_DIR / f"student_test_metrics_{encoder}_{mode}.csv"

        if not metrics_path.exists():
            print(f"  [WARN] No {metrics_path.name} found for "
                  f"{encoder}/{mode} — leaving row as-is (metrics stay missing).")
            fixed_rows.append(row)
            n_missing += 1
            continue

        with open(metrics_path, newline="", encoding="utf-8") as mf:
            test_results = next(csv.DictReader(mf), None)

        if not test_results:
            print(f"  [WARN] {metrics_path.name} is empty — skipping.")
            fixed_rows.append(row)
            n_missing += 1
            continue

        merged = {
            "encoder":        encoder,
            "mode":           mode,
            "best_composite": row.get("best_composite", ""),
            **{f"test_{k}": v for k, v in test_results.items()
              if k not in ("variant", "mode")},
        }
        fixed_rows.append(merged)
        n_backfilled += 1

    # Union of all keys across rows, preserving first-seen order
    all_keys = []
    for r in fixed_rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fixed_rows)

    print(f"  ✓ {comp_path.name}: backfilled {n_backfilled}, "
          f"left as-is {n_missing} (of {len(rows)} rows)")
    print(f"    Columns now: {all_keys}")


def main() -> None:
    print("=" * 72)
    print("  Backfilling student_comparison_stage{1,2}.csv")
    print("=" * 72)
    backfill(1)
    backfill(2)
    print("\n  Done. Re-run: python generate_charts.py --student")
    print("=" * 72)


if __name__ == "__main__":
    main()
