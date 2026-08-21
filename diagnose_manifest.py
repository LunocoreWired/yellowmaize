"""
Run this from your project root (venv active) to characterize the
category-duplication pattern in global_split_manifest.csv before we
decide on a fix.
"""
import pandas as pd
from config import GLOBAL_MANIFEST

df = pd.read_csv(GLOBAL_MANIFEST)
print(f"Total rows:            {len(df)}")
print(f"Unique filenames:      {df['filename'].nunique()}")
print(f"Columns:               {df.columns.tolist()}")
print()

per_file_cats = df.groupby("filename")["category"].nunique()
print("Distribution of #categories per filename:")
print(per_file_cats.value_counts().sort_index())
print()

# How many rows are TRUE duplicates (same filename + same category, e.g.
# a copy-paste dupe) vs DIFFERENT categories for the same filename?
exact_dupes = df.duplicated(subset=["filename", "category"]).sum()
print(f"Exact (filename+category) duplicate rows: {exact_dupes}")
print()

# Does severity/score differ across the category-rows for the same image?
# If there's a severity or score column, print it per category for a
# few multi-category images — tells us if these are independent scores
# (multi-label design) or literal copy/paste duplicates (bug).
candidate_cols = [c for c in df.columns if c not in ("filename", "category", "split")]
print(f"Other columns to inspect: {candidate_cols}")
print()

multi = per_file_cats[per_file_cats > 1].index[:5]
for fn in multi:
    print(df[df["filename"] == fn])
    print()
