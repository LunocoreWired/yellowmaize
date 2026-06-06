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
   data/label_studio_import/_manifest_hash.txt. On subsequent runs the hash
   is re-checked. If it has changed, the script aborts with a clear error
   rather than silently producing a mismatched sample.
================================================================================
"""

import hashlib
import shutil
from pathlib import Path
import pandas as pd

from config import TIER1_MANIFEST, DATA_DIR, CLASSES, SEED


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

    # 1. Create the destination directory
    gold_dir = DATA_DIR / "label_studio_import"
    gold_dir.mkdir(parents=True, exist_ok=True)

    # ── Manifest hash guard ────────────────────────────────────────────────────
    # Protects against silent re-sampling if sample_15000.py is re-run after
    # Label Studio annotation has already begun.
    hash_file = gold_dir / "_manifest_hash.txt"

    if not Path(TIER1_MANIFEST).exists():
        print(f"  [FATAL] Could not find {TIER1_MANIFEST}")
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
    for item in gold_dir.iterdir():
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

    # 4. Copy and Rename the physical files
    print(f"\n  Copying and renaming {len(gold_df)} images to {gold_dir} ...")

    success_count = 0
    new_filenames = []

    for _, row in gold_df.iterrows():
        src_path = Path(row["source_path"])
        category = row["category"]

        # Prepend the category to the filename so annotators see the class.
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

    # 5. Save mini-manifest
    gold_manifest_path = gold_dir / "_gold_standard_manifest_300.csv"
    gold_df.to_csv(gold_manifest_path, index=False)

    # 6. Write (or confirm) manifest hash — locks this sample to the current
    #    tier1_manifest.csv so future runs detect if it ever changes.
    hash_file.write_text(current_hash)
    print(f"  Manifest hash locked : {hash_file.name}  ({current_hash[:16]}...)")

    print(f"\n{'─' * 72}")
    print(f"  Successfully copied {success_count} / {len(gold_df)} images.")
    print(f"  Folder ready for Label Studio: {gold_dir}")
    print(f"  Manifest saved to: {gold_manifest_path}")
    print()
    print("  NEXT STEP: Upload images to Label Studio and annotate leaf")
    print("  silhouettes (polygonlabels). Export JSON to:")
    print("  data/gold_standard/annotations/annotations.json")
    print("=" * 72)


if __name__ == "__main__":
    main()
