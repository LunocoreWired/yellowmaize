"""
================================================================================
 sample_gold_standard.py — Extract 300 Images for Human Validation
================================================================================
 PURPOSE:
   Pulls a mathematically reproducible, perfectly balanced stratified sample
   (100 HEALTHY, 100 MSV, 100 MLN) from the Tier 1 dataset.
   Copies and RENAMES the raw images to include their ground-truth class,
   making manual annotation in Label Studio foolproof.

 REPRODUCIBILITY:
   Given the same tier1_manifest.csv, this script always selects the same
   300 images — pd.DataFrame.sample(random_state=SEED) is deterministic.

   ⚠ WARNING: If sample_15000.py is re-run, tier1_manifest.csv changes and
   this script will silently select a DIFFERENT 300 images. Any Label Studio
   annotations already done would be invalidated.

   GUARD: On first run, a SHA-256 hash of tier1_manifest.csv is written to
   data/gold_standard/images/_manifest_hash.txt. On subsequent runs the hash
   is re-checked. If it has changed, the script aborts with a clear error
   rather than silently producing a mismatched sample.

 OUTPUT DIRECTORY:
   Images are written to GOLD_IMAGES_DIR (data/gold_standard/images/) so that
   validate_gold_standard.py can find them directly — no manual file moving
   required after Label Studio annotation.

 WORKFLOW:
   1. Run this script right after sample_15000.py (no masks needed).
   2. Import data/gold_standard/images/ into Label Studio and annotate.
   3. Export annotations as JSON → data/gold_standard/annotations/annotations.json
   4. Run validate_gold_standard.py (after Teacher + Student are trained).
================================================================================
"""

import hashlib
import shutil
from pathlib import Path
import pandas as pd

from config import TIER1_MANIFEST, GOLD_IMAGES_DIR, GOLD_MANIFEST, CLASSES, SEED


def _manifest_hash(path: Path) -> str:
    """SHA-256 hash of tier1_manifest.csv contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    print("=" * 72)
    print("  Yellow MAIze | Generating 300-Image Gold Standard Validation Set")
    print("=" * 72)

    # 1. Create the destination directory (data/gold_standard/images/)
    GOLD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Manifest hash guard ────────────────────────────────────────────────────
    # Protects against silent re-sampling if sample_15000.py is re-run after
    # Label Studio annotation has already begun.
    hash_file = GOLD_IMAGES_DIR / "_manifest_hash.txt"

    if not Path(TIER1_MANIFEST).exists():
        print(f"  [FATAL] Could not find {TIER1_MANIFEST}")
        print("          Run sample_15000.py first.")
        return

    current_hash = _manifest_hash(Path(TIER1_MANIFEST))

    if hash_file.exists():
        saved_hash = hash_file.read_text().strip()
        if saved_hash != current_hash:
            print()
            print("  [FATAL] tier1_manifest.csv has changed since the gold standard")
            print("          set was last generated.")
            print()
            print("  This means sample_15000.py was re-run after annotation began.")
            print("  Re-generating would produce a DIFFERENT 300 images, making any")
            print("  existing Label Studio annotations invalid.")
            print()
            print("  If you intend to start fresh (annotations not yet done):")
            print(f"    Delete {hash_file} and re-run this script.")
            print()
            return

    # Clear out old images only (preserve hash file if it exists)
    print("  Clearing old export folder...")
    for item in GOLD_IMAGES_DIR.iterdir():
        if item.is_file() and item.name != "_manifest_hash.txt":
            item.unlink()

    # 2. Load the Tier 1 manifest
    df = pd.read_csv(TIER1_MANIFEST)
    print(f"  Loaded Tier 1 Manifest: {len(df):,} images available.")

    # 3. Stratified sampling (100 per class)
    # pd.DataFrame.sample(random_state=SEED) is fully deterministic —
    # same manifest + same SEED always produces the same rows.
    samples_per_class = 100
    sampled_rows = []

    for cls in CLASSES:
        cls_df = df[df["category"] == cls]
        if len(cls_df) < samples_per_class:
            print(f"  [WARN] {cls} has only {len(cls_df)} images in Tier 1 "
                  f"(need {samples_per_class}). Using all.")
        sampled = cls_df.sample(
            n=min(samples_per_class, len(cls_df)),
            random_state=SEED,
        )
        sampled_rows.append(sampled)

    gold_df = pd.concat(sampled_rows).copy()

    # 4. Copy and rename the physical files
    print(f"\n  Copying and renaming {len(gold_df)} images to {GOLD_IMAGES_DIR} ...")

    success_count = 0
    new_filenames = []

    for _, row in gold_df.iterrows():
        src_path = Path(row["source_path"])
        category = row["category"]

        # Prepend the category to the filename so annotators see the class.
        # Example: MSV_original_filename.jpg
        dest_filename = f"{category}_{src_path.name}"
        dest_path = GOLD_IMAGES_DIR / dest_filename

        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            new_filenames.append(dest_filename)
            success_count += 1
        else:
            print(f"  [WARN] Missing source file: {src_path}")
            new_filenames.append("ERROR_MISSING")

    gold_df["gold_filename"] = new_filenames

    # 5. Save gold manifest (picked up by validate_gold_standard.py)
    GOLD_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    gold_df.to_csv(GOLD_MANIFEST, index=False)

    # 6. Write (or confirm) manifest hash — locks this sample to the current
    #    tier1_manifest.csv so future runs detect if it ever changes.
    hash_file.write_text(current_hash)
    print(f"  Manifest hash locked : {hash_file.name}  ({current_hash[:16]}...)")

    print(f"\n{'─' * 72}")
    print(f"  Successfully copied {success_count} / {len(gold_df)} images.")
    print(f"  Folder ready for Label Studio : {GOLD_IMAGES_DIR}")
    print(f"  Gold manifest saved to        : {GOLD_MANIFEST}")
    print()
    print("  NEXT STEPS:")
    print("    1. Import images into Label Studio and annotate leaf silhouettes")
    print("       (polygonlabels). Export JSON to:")
    print(f"      {GOLD_IMAGES_DIR.parent / 'annotations' / 'annotations.json'}")
    print("    2. Run validate_gold_standard.py after training is complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
