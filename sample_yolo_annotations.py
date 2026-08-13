"""
================================================================================
 sample_yolo_annotations.py — Extract 501 Images for YOLO BBox Annotation
================================================================================
 PURPOSE:
   Pulls a mathematically reproducible, perfectly balanced stratified sample
   (167 HEALTHY, 167 MSV, 167 MLN = 501 total) from the Tier 1 dataset.
   Copies and RENAMES the raw images to include their ground-truth class,
   making manual bounding box annotation in Label Studio foolproof.

   These 501 images are intended for YOLO bounding box training — annotators
   draw a single tight bounding box around the PRIMARY leaf only. This dataset
   replaces the HSV-based bbox prompting in generate_tier1_masks.py with a
   learned YOLO detector for more accurate SAM2 prompting (especially for
   MLN images where necrotic brown tissue confuses the HSV mask).

 REPRODUCIBILITY:
   Given the same tier1_manifest.csv, this script always selects the same
   501 images — pd.DataFrame.sample(random_state=SEED) is deterministic.

   ⚠ WARNING: If sample_15000.py is re-run, tier1_manifest.csv changes and
   this script will silently select a DIFFERENT 501 images. Any Label Studio
   annotations already done would be invalidated.

   GUARD: On first run, a SHA-256 hash of tier1_manifest.csv is written to
   data/yolo_annotations/images/_manifest_hash.txt. On subsequent runs the
   hash is re-checked. If it has changed, the script aborts with a clear error
   rather than silently producing a mismatched sample.

 OUTPUT DIRECTORY:
   Images are written to YOLO_IMAGES_DIR (data/yolo_annotations/images/).

 ANNOTATION INSTRUCTIONS FOR LABEL STUDIO:
   - Task type : Object Detection (bounding box)
   - Label     : "leaf" (one label only)
   - Draw ONE tight bounding box around the PRIMARY leaf only.
   - Ignore background leaves, stems, and hands.
   - Export format: YOLO (txt) or COCO JSON.

 WORKFLOW:
   1. Run this script right after sample_15000.py (no masks needed).
   2. Import data/yolo_annotations/images/ into Label Studio.
   3. Annotate bounding boxes (primary leaf only, label = "leaf").
   4. Export annotations → data/yolo_annotations/labels/ (YOLO format).
   5. Run train_yolo.py to train the YOLO detector.
   6. Integrate trained YOLO into generate_tier1_masks.py to replace HSV bbox.
================================================================================
"""

import hashlib
import shutil
from pathlib import Path
import pandas as pd

from config import TIER1_MANIFEST, CLASSES, SEED

# ── Output directories for YOLO annotation set ────────────────────────────────
# Add YOLO_IMAGES_DIR and YOLO_MANIFEST to config.py if not already present,
# or they default to the paths below.
try:
    from config import YOLO_IMAGES_DIR, YOLO_MANIFEST
except ImportError:
    YOLO_IMAGES_DIR = Path("data/yolo_annotations/images")
    YOLO_MANIFEST   = Path("data/yolo_annotations/yolo_manifest.csv")


def _manifest_hash(path: Path) -> str:
    """SHA-256 hash of tier1_manifest.csv contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    print("=" * 72)
    print("  Yellow MAIze | Generating 501-Image YOLO Annotation Set")
    print("  (167 HEALTHY + 167 MSV + 167 MLN)")
    print("=" * 72)

    # 1. Create the destination directory (data/yolo_annotations/images/)
    YOLO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Manifest hash guard ────────────────────────────────────────────────────
    # Protects against silent re-sampling if sample_15000.py is re-run after
    # Label Studio annotation has already begun.
    hash_file = YOLO_IMAGES_DIR / "_manifest_hash.txt"

    if not Path(TIER1_MANIFEST).exists():
        print(f"  [FATAL] Could not find {TIER1_MANIFEST}")
        print("          Run sample_15000.py first.")
        return

    current_hash = _manifest_hash(Path(TIER1_MANIFEST))

    if hash_file.exists():
        saved_hash = hash_file.read_text().strip()
        if saved_hash != current_hash:
            print()
            print("  [FATAL] tier1_manifest.csv has changed since the YOLO annotation")
            print("          set was last generated.")
            print()
            print("  This means sample_15000.py was re-run after annotation began.")
            print("  Re-generating would produce a DIFFERENT 501 images, making any")
            print("  existing Label Studio annotations invalid.")
            print()
            print("  If you intend to start fresh (annotations not yet done):")
            print(f"    Delete {hash_file} and re-run this script.")
            print()
            return

    # Clear out old images only (preserve hash file if it exists)
    print("  Clearing old export folder...")
    for item in YOLO_IMAGES_DIR.iterdir():
        if item.is_file() and item.name != "_manifest_hash.txt":
            item.unlink()

    # 2. Load the Tier 1 manifest
    df = pd.read_csv(TIER1_MANIFEST)
    print(f"  Loaded Tier 1 Manifest: {len(df):,} images available.")

    # 3. Stratified sampling (167 per class = 501 total)
    # pd.DataFrame.sample(random_state=SEED) is fully deterministic —
    # same manifest + same SEED always produces the same rows.
    samples_per_class = 167
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
        print(f"  Sampled {len(sampled):>3} images for {cls}")

    gold_df = pd.concat(sampled_rows).copy()
    print(f"  Total : {len(gold_df)} images")

    # 4. Copy and rename the physical files
    print(f"\n  Copying and renaming {len(gold_df)} images to {YOLO_IMAGES_DIR} ...")

    success_count = 0
    new_filenames = []

    for _, row in gold_df.iterrows():
        src_path = Path(row["source_path"])
        category = row["category"]

        # Prepend the category to the filename so annotators see the class.
        # Example: MLN_original_filename.jpg
        dest_filename = f"{category}_{src_path.name}"
        dest_path = YOLO_IMAGES_DIR / dest_filename

        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            new_filenames.append(dest_filename)
            success_count += 1
        else:
            print(f"  [WARN] Missing source file: {src_path}")
            new_filenames.append("ERROR_MISSING")

    gold_df["yolo_filename"] = new_filenames

    # 5. Save YOLO manifest
    YOLO_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    gold_df.to_csv(YOLO_MANIFEST, index=False)

    # 6. Write (or confirm) manifest hash — locks this sample to the current
    #    tier1_manifest.csv so future runs detect if it ever changes.
    hash_file.write_text(current_hash)
    print(f"  Manifest hash locked : {hash_file.name}  ({current_hash[:16]}...)")

    print(f"\n{'─' * 72}")
    print(f"  Successfully copied {success_count} / {len(gold_df)} images.")
    print(f"  Folder ready for Label Studio : {YOLO_IMAGES_DIR}")
    print(f"  YOLO manifest saved to        : {YOLO_MANIFEST}")
    print()
    print("  NEXT STEPS:")
    print("    1. Import data/yolo_annotations/images/ into Label Studio.")
    print("    2. Annotate ONE bounding box per image (primary leaf only).")
    print("       Label Studio task type: Object Detection | label: 'leaf'")
    print("    3. Export annotations in YOLO format to:")
    print(f"       {YOLO_IMAGES_DIR.parent / 'labels'}/")
    print("    4. Run train_yolo.py to train the YOLO leaf detector.")
    print("    5. Integrate into generate_tier1_masks.py to replace HSV bbox.")
    print("=" * 72)


if __name__ == "__main__":
    main()
