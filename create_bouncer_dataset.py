"""
================================================================================
 create_bouncer_dataset.py — Phase 0a: Build Bouncer Dataset
================================================================================
 PURPOSE:
   Build a balanced 50k binary dataset for Bouncer training:
     - 25,000 maize images   (positive class — train+val split only)
     - 25,000 non-maize images (negative class — intel + natural + crop neighbors)

 IMPORTANT:
   Reads global_split_manifest.csv to exclude test-split maize images
   from the positive class. Test-split images must never be seen by
   the Bouncer during training.

 OUTPUT:
   data/bouncer_dataset/maize/       ← 25k maize positives
   data/bouncer_dataset/not_maize/   ← 25k non-maize negatives
================================================================================
"""

import random
import shutil
from pathlib import Path

import pandas as pd

from config import (
    SEED, GLOBAL_MANIFEST, BOUNCER_DATASET_DIR, BOUNCER_TARGET_PER_CLASS,
    NON_MAIZE_SOURCES, VALID_EXTENSIONS, LOGS_DIR,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def collect_images(directories: list[Path]) -> list[Path]:
    images = []
    for d in directories:
        if not d.is_dir():
            print(f"  [WARN] Directory not found, skipping: {d}")
            continue
        for ext in VALID_EXTENSIONS:
            images.extend(d.rglob(f"*{ext}"))
    return images


def copy_images(src_list: list[Path], dest_dir: Path,
                prefix: str, label: str) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i, src in enumerate(src_list):
        dest = dest_dir / f"{prefix}_{i:06d}{src.suffix.lower()}"
        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception as e:
            print(f"  [WARN] Could not copy {src.name}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"    {label}: {i+1:,} / {len(src_list):,} copied")
    return copied


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# NON-MAIZE SOURCE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_nonmaize_sources(images: list) -> tuple[list, list]:
    """
    Run lightweight preprocessing validation on non-maize source images.
    Applies Steps 1-6 from partition_dataset.py (zero-byte, magic bytes,
    truncation, resolution, uniformity). No pHash dedup needed — all negatives.

    Returns (valid_images, rejected_records).
    """
    from PIL import Image as _PIL, ImageFile as _PILF, ImageOps as _PILO
    import hashlib as _hl

    _PILF.LOAD_TRUNCATED_IMAGES = False

    MAGIC = {
        ".jpg":  b"\xff\xd8\xff",
        ".jpeg": b"\xff\xd8\xff",
        ".png":  b"\x89PNG",
    }

    valid, rejected = [], []

    for i, p in enumerate(images):
        p = Path(p)
        reason = None

        # Step 1 — zero byte
        try:
            sz = p.stat().st_size
            if sz < 100:
                reason = f"zero_or_tiny:{sz}b"
        except OSError as e:
            reason = f"stat_error:{e}"

        # Step 2 — magic bytes
        if not reason:
            ext = p.suffix.lower()
            magic = MAGIC.get(ext)
            if magic:
                try:
                    with open(p, "rb") as f:
                        header = f.read(8)
                    if not header[:len(magic)] == magic:
                        reason = f"magic_mismatch:{header[:4].hex()}"
                except OSError:
                    reason = "read_error"

        # Step 3 — truncation + open
        img = None
        if not reason:
            try:
                img = _PIL.open(p)
                img.load()
                img = _PILO.exif_transpose(img)
            except Exception as e:
                reason = f"truncated:{str(e)[:60]}"
                img = None

        if img is not None:
            w, h = img.size

            # Step 4 — resolution
            if not reason:
                if min(w, h) < 64:
                    reason = f"too_small:{w}x{h}"
                elif max(w, h) > 4096:
                    reason = f"too_large:{w}x{h}"
                elif max(w,h) / max(min(w,h),1) > 8.0:
                    reason = f"extreme_aspect:{w}x{h}"

            # Step 5 — colour mode
            if not reason and img.mode == "1":
                reason = "binary_1bit"

            # Step 6 — near-uniform
            if not reason:
                import numpy as _np
                arr = _np.array(img.convert("RGB"), dtype=_np.float32)
                if arr.std() < 5.0:
                    reason = f"near_uniform:std={arr.std():.1f}"

            img.close()

        if reason:
            rejected.append({"path": str(p), "reason": reason})
        else:
            valid.append(p)

        if (i + 1) % 5000 == 0:
            print(f"    Validated {i+1:,}/{len(images):,} non-maize images ...")

    return valid, rejected


def main() -> None:
    random.seed(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Phase 0a: Create Bouncer Dataset")
    print("=" * 72)

    # ── 1. Load manifest — get train+val maize images only ────────────────────
    if not GLOBAL_MANIFEST.exists():
        print(f"[FATAL] {GLOBAL_MANIFEST} not found. Run partition_dataset.py first.")
        return

    manifest = pd.read_csv(GLOBAL_MANIFEST)
    trainval  = manifest[manifest["split"] != "test"]
    maize_paths = [Path(p) for p in trainval["source_path"].tolist()]

    print(f"\n  Maize (train+val only): {len(maize_paths):,} available")

    # ── 2. Sample maize positives ─────────────────────────────────────────────
    target = BOUNCER_TARGET_PER_CLASS
    if len(maize_paths) >= target:
        maize_sample = random.sample(maize_paths, target)
    else:
        print(f"  [WARN] Only {len(maize_paths):,} maize images available "
              f"(target {target:,}). Using all.")
        maize_sample = maize_paths
    print(f"  Maize sample          : {len(maize_sample):,}")

    # ── 3. Collect non-maize negatives ────────────────────────────────────────
    non_maize_paths = collect_images(NON_MAIZE_SOURCES)
    print(f"\n  Non-maize available   : {len(non_maize_paths):,}")

    # Log per-source counts
    for src in NON_MAIZE_SOURCES:
        if src.is_dir():
            count = sum(1 for ext in VALID_EXTENSIONS
                        for _ in src.rglob(f"*{ext}"))
            print(f"    {src.name:<20}: {count:,}")

    if len(non_maize_paths) >= target:
        non_maize_sample = random.sample(non_maize_paths, target)
    else:
        print(f"  [WARN] Only {len(non_maize_paths):,} non-maize images available "
              f"(target {target:,}). Using all.")
        non_maize_sample = non_maize_paths
    print(f"  Non-maize sample      : {len(non_maize_sample):,}")

    # ── 4. Clear existing dataset if present ─────────────────────────────────
    maize_out    = BOUNCER_DATASET_DIR / "maize"
    nonmaize_out = BOUNCER_DATASET_DIR / "not_maize"

    for d in [maize_out, nonmaize_out]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # ── 5. Copy images ────────────────────────────────────────────────────────
    print(f"\n  Copying maize images ...")
    n_maize = copy_images(maize_sample, maize_out, "maize", "maize")

    print(f"  Copying non-maize images ...")
    n_non   = copy_images(non_maize_sample, nonmaize_out, "not_maize", "not_maize")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  Bouncer dataset ready:")
    print(f"    Maize     : {n_maize:,}  →  {maize_out}")
    print(f"    Non-maize : {n_non:,}  →  {nonmaize_out}")
    print(f"    Total     : {n_maize + n_non:,}")
    print(f"\n  NEXT STEP: python train_bouncer.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
