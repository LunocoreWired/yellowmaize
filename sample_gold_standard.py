"""
================================================================================
 sample_gold_standard.py — Extract 300 Images for Human Validation
================================================================================
 PURPOSE:
   Pulls a mathematically reproducible, perfectly balanced stratified sample
   (100 HEALTHY, 100 MSV, 100 MLN) from the Tier 1 dataset.
   Copies and RENAMES the raw images to include their ground-truth class,
   making manual annotation in Label Studio foolproof.
================================================================================
"""

import shutil
from pathlib import Path

import pandas as pd

# Import your existing configurations
from config import CLASSES, DATA_DIR, SEED, TIER1_MANIFEST


def main():
    print("=" * 72)
    print("  Yellow MAIze | Generating 300-Image Gold Standard Validation Set")
    print("=" * 72)

    # 1. Create the destination directory
    gold_dir = DATA_DIR / "label_studio_import"
    gold_dir.mkdir(parents=True, exist_ok=True)

    # Clear out the old 150 images to prevent mixing
    print("  Clearing old export folder...")
    for item in gold_dir.iterdir():
        if item.is_file():
            item.unlink()

    # 2. Load the Tier 1 manifest
    if not Path(TIER1_MANIFEST).exists():
        print(f" [FATAL] Could not find {TIER1_MANIFEST}")
        return

    df = pd.read_csv(TIER1_MANIFEST)
    print(f"  Loaded Tier 1 Manifest: {len(df):,} images available.")

    # 3. Stratified sampling (100 per class now)
    samples_per_class = 100
    sampled_rows = []

    for cls in CLASSES:
        cls_df = df[df["category"] == cls]
        # Use your exact global SEED for reproducibility
        sampled = cls_df.sample(n=samples_per_class, random_state=SEED)
        sampled_rows.append(sampled)

    gold_df = pd.concat(sampled_rows).copy()

    # 4. Copy and Rename the physical files
    print(f"\n  Copying and renaming {len(gold_df)} images to {gold_dir} ...")

    success_count = 0
    new_filenames = []

    for _, row in gold_df.iterrows():
        src_path = Path(row["source_path"])
        category = row["category"]

        # MAGIC TRICK: Prepend the category to the filename!
        # Example: MSV_original_filename.jpg
        dest_filename = f"{category}_{src_path.name}"
        dest_path = gold_dir / dest_filename

        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            new_filenames.append(dest_filename)
            success_count += 1
        else:
            print(f"  [WARN] Missing source file: {src_path}")
            new_filenames.append("ERROR_MISSING")

    # Add the new names to the manifest so we don't lose track of them
    gold_df["label_studio_filename"] = new_filenames

    # 5. Save a mini-manifest just for this gold standard set
    gold_manifest_path = gold_dir / "_gold_standard_manifest_300.csv"
    gold_df.to_csv(gold_manifest_path, index=False)

    print(f"\n{'─' * 72}")
    print(f"  Successfully copied {success_count} images.")
    print(f"  Folder ready for Label Studio: {gold_dir}")
    print(f"  Manifest saved to: {gold_manifest_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
