"""
================================================================================
 sample_15000.py — Phase 1: Tier 1 Stratified Sampler
================================================================================
 PURPOSE:
   Select 15,000 images from maize_dataset in a stratified way for SAM2
   masking. Images are drawn ONLY from the global train+val split —
   no test-split images are ever included in Tier 1.

 COMPOSITION:
   HEALTHY : 3,000  (pure random sample — no severity gradient needed)
   MSV     : 7,500  (evenly spaced across sorted filenames — max diversity)
   MLN     : 4,500  (evenly spaced across sorted filenames — max diversity)
   TOTAL   : 15,000

 STRATIFICATION STRATEGY:
   For MSV and MLN: evenly-spaced indices across sorted filename list.
   This maximises diversity of lighting, severity, and leaf positions.
   For HEALTHY: pure random — seeded by random.seed(SEED=42), deterministic.

 REPRODUCIBILITY:
   Given the same global_split_manifest.csv, this script always produces
   the same tier1_manifest.csv:
     - HEALTHY: random.sample() seeded by random.seed(42) — deterministic
     - MSV/MLN: evenly_spaced_sample() is pure math — no randomness at all
     - Filename collision handler: random.randint() is also seeded

   ⚠ WARNING: If partition_dataset.py is re-run, global_split_manifest.csv
   changes and this script will silently produce a DIFFERENT tier1 set.
   That would also invalidate any gold standard annotations already done
   in Label Studio (sample_gold_standard.py depends on tier1_manifest.csv).

   GUARD: On first run, a SHA-256 hash of global_split_manifest.csv is
   written to data/tier1_raw/_manifest_hash.txt. On subsequent runs the
   hash is re-checked. If it has changed, the script aborts rather than
   silently re-sampling.

 OUTPUTS:
   data/tier1_raw/          — images copied here for SAM2 processing
   tier1_manifest.csv       — records selected images with split from global manifest
================================================================================
"""

import csv
import hashlib
import time as _time
import random
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

from config import (
    SEED, GLOBAL_MANIFEST, TIER1_MANIFEST,
    TIER1_RAW_DIR, TIER1_PER_CLASS, REPORTS_DIR,
    CLASSES, VALID_EXTENSIONS,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def evenly_spaced_sample(items: list, n: int) -> list:
    """
    Select n items from a sorted list using evenly-spaced indices.
    Maximises coverage of the full filename-space (diversity proxy).
    Purely deterministic — no randomness involved.
    """
    if n >= len(items):
        return items
    step    = len(items) / n
    indices = [int(i * step) for i in range(n)]
    return [items[i] for i in indices]


def _manifest_hash(path: Path) -> str:
    """SHA-256 hash of a manifest CSV file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _t_start = _time.time()
    set_seeds(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Phase 1: Tier 1 Stratified Sampler")
    print("=" * 72)

    # ── Load global manifest ──────────────────────────────────────────────────
    if not GLOBAL_MANIFEST.exists():
        print(f"[FATAL] {GLOBAL_MANIFEST} not found. Run partition_dataset.py first.")
        return

    # ── Manifest hash guard ───────────────────────────────────────────────────
    # Prevents silent re-sampling if partition_dataset.py is re-run after
    # SAM2 masking or gold standard annotation has already begun.
    TIER1_RAW_DIR.mkdir(parents=True, exist_ok=True)
    hash_file    = TIER1_RAW_DIR / "_manifest_hash.txt"
    current_hash = _manifest_hash(GLOBAL_MANIFEST)

    if hash_file.exists():
        saved_hash = hash_file.read_text().strip()
        if saved_hash != current_hash:
            print()
            print("  [FATAL] global_split_manifest.csv has changed since Tier 1")
            print("          was last sampled.")
            print()
            print("  This means partition_dataset.py was re-run after Tier 1 sampling")
            print("  began. Re-sampling would produce a DIFFERENT set of 15,000 images,")
            print("  invalidating any SAM2 masks or gold standard annotations already")
            print("  generated from the previous Tier 1 set.")
            print()
            print("  If you intend to start the entire pipeline from scratch:")
            print(f"    Delete {hash_file} (and data/tier1_leaf_masks/, data/gold_standard/)")
            print("    then re-run this script.")
            print()
            return

    # ── Prepare output directory ───────────────────────────────────────────────
    # Check if tier1_raw already has images from a previous run
    existing_images = [
        p for p in TIER1_RAW_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ] if TIER1_RAW_DIR.exists() else []

    if existing_images:
        answer = input(
            f"\n  {TIER1_RAW_DIR} already contains {len(existing_images):,} images.\n"
            f"  Clear and resample? [y/N]: "
        )
        if answer.strip().lower() != "y":
            print("  Aborted.")
            return
        # Remove images but preserve hash file
        for p in TIER1_RAW_DIR.iterdir():
            if p.is_file() and p.name != "_manifest_hash.txt":
                p.unlink()

    manifest = pd.read_csv(GLOBAL_MANIFEST)

    # Only train+val images are eligible for Tier 1
    eligible = manifest[manifest["split"] != "test"].copy()
    print(f"\n  Eligible images (train+val): {len(eligible):,}")

    # ── Sample per class ──────────────────────────────────────────────────────
    manifest_rows = []
    total_copied  = 0

    for category, target_n in TIER1_PER_CLASS.items():
        cat_rows = eligible[eligible["category"] == category]
        if cat_rows.empty:
            print(f"  [WARN] No eligible images for {category}. Skipping.")
            continue

        # Sort by filename for reproducible evenly-spaced selection
        cat_paths = sorted(
            [Path(p) for p in cat_rows["source_path"].tolist()],
            key=lambda p: p.name,
        )
        available = len(cat_paths)

        # HEALTHY: pure random (seeded above via set_seeds)
        # MSV/MLN: evenly spaced (deterministic math, no randomness)
        if category == "HEALTHY":
            selected = random.sample(cat_paths, min(target_n, available))
        else:
            selected = evenly_spaced_sample(cat_paths, min(target_n, available))

        n_selected = len(selected)
        print(f"\n  [{category}]")
        print(f"    Available : {available:,}")
        print(f"    Target    : {target_n:,}")
        print(f"    Selected  : {n_selected:,}")

        # Copy to tier1_raw with category prefix
        for src in selected:
            dest_name = f"{category}_{src.name}"
            dest      = TIER1_RAW_DIR / dest_name

            # Handle rare filename collision — seeded so even this is deterministic
            if dest.exists():
                stem = src.stem
                dest_name = f"{category}_{stem}_{random.randint(10000, 99999)}{src.suffix}"
                dest      = TIER1_RAW_DIR / dest_name

            shutil.copy2(src, dest)
            total_copied += 1

            # Get split from manifest
            match = manifest[manifest["source_path"] == str(src)]
            split = match["split"].iloc[0] if not match.empty else "train"

            manifest_rows.append({
                "dest_filename": dest_name,
                "source_path":   str(src),
                "category":      category,
                "split":         split,
                "tier":          1,
            })

    # ── Write Tier 1 manifest ─────────────────────────────────────────────────
    fieldnames = ["dest_filename", "source_path", "category", "split", "tier"]
    with open(TIER1_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # ── Write manifest hash — locks this Tier 1 set to the current manifest ──
    hash_file.write_text(current_hash)
    print(f"\n  Manifest hash locked : {hash_file.name}  ({current_hash[:16]}...)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  Total copied : {total_copied:,} images → {TIER1_RAW_DIR}")
    print(f"  Manifest     : {TIER1_MANIFEST}")

    by_class = {}
    for r in manifest_rows:
        by_class.setdefault(r["category"], 0)
        by_class[r["category"]] += 1

    print(f"\n  {'Class':<12} {'Count':>8}")
    print(f"  {'─'*12} {'─'*8}")
    for cls, cnt in by_class.items():
        print(f"  {cls:<12} {cnt:>8,}")
    print(f"  {'─'*12} {'─'*8}")
    print(f"  {'TOTAL':<12} {total_copied:>8,}")

    elapsed = round(_time.time() - _t_start, 1)
    print(f"\n  Duration     : {elapsed}s")
    print(f"\n  NEXT STEP: python generate_tier1_masks.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
