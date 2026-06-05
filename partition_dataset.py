"""
================================================================================
 partition_dataset.py — Phase 0-pre: Global Dataset Split + Full Preprocessing
================================================================================
 PURPOSE:
   1. Scan maize_dataset/ recursively for all valid images.
   2. Run a complete preprocessing validation pipeline:
        Step 1 — Zero-byte / empty file check
        Step 2 — Magic bytes / format mismatch check
        Step 3 — Truncated image detection (full pixel load)
        Step 4 — Resolution checks (min/max dimensions, extreme aspect ratio)
        Step 5 — Colour mode check (grayscale, palette flagged)
        Step 6 — Near-uniform / solid colour detection
        Step 7 — MD5 exact duplicate removal (within + across classes)
        Step 8 — pHash near-duplicate removal (within class)
        Step 9 — Cross-class pHash duplicate detection
        Step 10 — Low green content flag (non-plant suspicion)
   3. Apply stratified 70/15/15 split per class.
   4. Write global_split_manifest.csv — single source of truth.
   5. Write reports/preprocessing_report.csv + preprocessing_summary.txt.

 RUN ONCE before any other script. Never re-run after training begins.

 OUTPUTS:
   global_split_manifest.csv          — every valid image assigned to split
   reports/preprocessing_report.csv  — per-rejected-image log
   reports/preprocessing_summary.txt — human-readable summary for Chapter 3
   quarantine/<reason>/<class>/       — rejected images moved here (not deleted)
================================================================================
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import csv
import hashlib
import random
import shutil
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from config import (
    SEED, MAIZE_DIR, GLOBAL_MANIFEST, REPORTS_DIR,
    CLASSES, VALID_EXTENSIONS, SPLIT_RATIOS,
)

# Quarantine root — rejected files are moved here instead of deleted
QUARANTINE_DIR = MAIZE_DIR.parent / "quarantine"

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL DEPENDENCY
# ══════════════════════════════════════════════════════════════════════════════
try:
    import imagehash
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False
    print("[WARN] imagehash not installed — pHash dedup skipped.")
    print("       pip install imagehash")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
PHASH_HAMMING_THRESHOLD  = 2      # near-duplicate threshold
MIN_DIMENSION            = 64     # px — smaller images rejected
MAX_DIMENSION            = 4096   # px — larger images rejected
MAX_ASPECT_RATIO         = 8.0    # max(w,h)/min(w,h) — beyond this, rejected
UNIFORM_STD_THRESHOLD    = 5.0    # pixel std below this → near-uniform, rejected
GREEN_CONTENT_THRESHOLD  = 0.05   # < 5% green pixels → flagged (not rejected)

# Magic bytes for format validation
MAGIC_BYTES = {
    ".jpg":  [(0, b"\xff\xd8\xff")],
    ".jpeg": [(0, b"\xff\xd8\xff")],
    ".png":  [(0, b"\x89PNG")],
    ".JPG":  [(0, b"\xff\xd8\xff")],
    ".JPEG": [(0, b"\xff\xd8\xff")],
    ".PNG":  [(0, b"\x89PNG")],
}

# HSV green range for plant content check
GREEN_H_MIN, GREEN_H_MAX = 30, 90
GREEN_S_MIN, GREEN_V_MIN = 40, 40


# ══════════════════════════════════════════════════════════════════════════════
# SEED
# ══════════════════════════════════════════════════════════════════════════════
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ══════════════════════════════════════════════════════════════════════════════
# QUARANTINE — move rejected files rather than deleting them
# ══════════════════════════════════════════════════════════════════════════════
def quarantine_file(path: Path, reason: str, cls: str) -> None:
    """
    Move *path* to QUARANTINE_DIR/<reason_prefix>/<cls>/<filename>.
    Safe to call even if the file was already moved (silently skips).
    """
    reason_prefix = reason.split(":")[0]          # e.g. "exact_duplicate_of"
    dest_dir = QUARANTINE_DIR / reason_prefix / cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    # Avoid collisions: append a counter suffix if name already exists
    if dest.exists():
        stem, suffix = path.stem, path.suffix
        for i in range(1, 10_000):
            dest = dest_dir / f"{stem}_{i}{suffix}"
            if not dest.exists():
                break
    try:
        shutil.move(str(path), dest)
    except (FileNotFoundError, shutil.Error):
        pass   # already moved or inaccessible — not fatal


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ZERO-BYTE CHECK
# ══════════════════════════════════════════════════════════════════════════════
def check_zero_byte(path: Path) -> str | None:
    """Returns rejection reason string or None if OK."""
    try:
        size = path.stat().st_size
        if size == 0:
            return "zero_byte"
        if size < 100:   # < 100 bytes is almost certainly not a valid image
            return f"too_small_bytes:{size}"
    except OSError as e:
        return f"stat_error:{e}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MAGIC BYTES / FORMAT MISMATCH
# ══════════════════════════════════════════════════════════════════════════════
def check_magic_bytes(path: Path) -> str | None:
    ext    = path.suffix
    checks = MAGIC_BYTES.get(ext)
    if checks is None:
        return None   # extension not in our map — skip magic check
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        for offset, magic in checks:
            if not header[offset:offset + len(magic)] == magic:
                return f"magic_mismatch:expected_{magic.hex()}_got_{header[:4].hex()}"
    except OSError as e:
        return f"read_error:{e}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRUNCATION CHECK (full pixel load)
# ══════════════════════════════════════════════════════════════════════════════
def check_truncated(path: Path) -> tuple[str | None, Image.Image | None]:
    """
    Attempts a full pixel load. Returns (reason_or_None, image_or_None).
    PIL silently opens truncated files — only .load() forces full decoding.
    """
    # Disable truncation tolerance
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        img = Image.open(path)
        img.load()   # forces full decode — raises on truncation
        return None, img
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ("truncat", "decompression", "corrupt",
                                   "invalid", "broken", "error")):
            return f"truncated_or_corrupt:{str(e)[:80]}", None
        return f"open_error:{str(e)[:80]}", None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RESOLUTION CHECKS
# ══════════════════════════════════════════════════════════════════════════════
def check_resolution(img: Image.Image) -> str | None:
    w, h = img.size
    if min(w, h) < MIN_DIMENSION:
        return f"too_small:{w}x{h}"
    if max(w, h) > MAX_DIMENSION:
        return f"too_large:{w}x{h}"
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT_RATIO:
        return f"extreme_aspect:{aspect:.1f}:1 ({w}x{h})"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — COLOUR MODE CHECK
# ══════════════════════════════════════════════════════════════════════════════
def check_colour_mode(img: Image.Image) -> tuple[str | None, str]:
    """
    Returns (rejection_reason_or_None, flag_or_empty).
    Grayscale (L) → flag but keep (convert to RGB during training).
    Palette (P) without alpha → flag (may have colour quantisation artefacts).
    Binary (1-bit) → reject.
    """
    mode = img.mode
    if mode == "1":
        return "binary_1bit_image", ""
    if mode == "L":
        return None, "grayscale_converted_to_rgb"
    if mode == "P":
        return None, "palette_mode_may_have_artefacts"
    return None, ""


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — NEAR-UNIFORM DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def check_uniform(img: Image.Image) -> str | None:
    """
    Compute pixel standard deviation across all channels.
    Images with std < threshold are near-uniform (solid colour, blank).
    """
    try:
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        std = float(arr.std())
        if std < UNIFORM_STD_THRESHOLD:
            return f"near_uniform:std={std:.2f}"
    except Exception:
        return "uniform_check_failed"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — MD5 EXACT DUPLICATE
# ══════════════════════════════════════════════════════════════════════════════
def compute_md5(path: Path) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _md5_worker(args: tuple[Path, str]) -> tuple[Path, str, str | None]:
    """Return (path, cls, md5_or_None) — runs in a thread pool."""
    p, cls = args
    return p, cls, compute_md5(p)


def remove_exact_duplicates(all_images: dict[str, list[Path]],
                             rejected: list[dict]) -> dict[str, list[Path]]:
    """
    MD5 dedup across ALL classes simultaneously.
    Hashing is I/O-bound → parallelised with ThreadPoolExecutor.
    If same MD5 appears in two different classes → cross-class conflict.
    """
    # Flatten to (path, cls) pairs preserving original class order
    flat: list[tuple[Path, str]] = [
        (p, cls) for cls, paths in all_images.items() for p in paths
    ]

    max_workers = max(1, (os.cpu_count() or 1) - 2)
    md5_results: dict[Path, tuple[str, str | None]] = {}   # path → (cls, md5)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for p, cls, md5 in executor.map(_md5_worker, flat):
            md5_results[p] = (cls, md5)

    md5_to_first: dict[str, tuple[str, Path]] = {}   # md5 → (class, path)
    result: dict[str, list[Path]] = {cls: [] for cls in all_images}

    for p, cls in flat:
        cls_r, md5 = md5_results[p]
        if md5 is None:
            rejected.append({"path": str(p), "category": cls,
                              "reason": "md5_read_error"})
            quarantine_file(p, "md5_read_error", cls)
            continue
        if md5 in md5_to_first:
            first_cls, first_path = md5_to_first[md5]
            if first_cls != cls:
                reason = (f"cross_class_exact_duplicate:"
                          f"also_in_{first_cls}:{first_path.name}")
            else:
                reason = f"exact_duplicate_of:{first_path.name}"
            rejected.append({"path": str(p), "category": cls, "reason": reason})
            quarantine_file(p, reason, cls)
        else:
            md5_to_first[md5] = (cls, p)
            result[cls].append(p)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — pHASH NEAR-DUPLICATE (within class)
# ══════════════════════════════════════════════════════════════════════════════
def compute_phash(path: Path) -> str | None:
    if not PHASH_AVAILABLE:
        return None
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img.convert("RGB"), hash_size=8))
    except Exception:
        return None


def _phash_worker(args: tuple[Path, str]) -> tuple[Path, str, str | None]:
    """Return (path, cls, phash_or_None) — runs in a process pool."""
    p, cls = args
    return p, cls, compute_phash(p)


def _phashes_to_uint64(hashes: list[str]) -> np.ndarray:
    """
    Convert a list of 16-char hex pHash strings to a uint64 numpy array.
    Each 64-bit integer encodes the full hash for fast XOR comparisons.
    """
    return np.array([int(h, 16) for h in hashes], dtype=np.uint64)


def _popcount_array(x: np.ndarray) -> np.ndarray:
    """Vectorised popcount (Hamming weight) using the bit-twiddling method."""
    # Brian Kernighan / lookup-table approach via numpy
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


def phash_dedup_within_class(paths: list[Path], cls: str,
                              rejected: list[dict]) -> list[Path]:
    """
    Near-duplicate removal within one class.

    Hash computation is I/O + PIL decode → ThreadPoolExecutor (not Process).
    Using a ProcessPoolExecutor here causes worker processes to re-evaluate the
    module-level `try: import imagehash` and silently set PHASH_AVAILABLE=False
    in the subprocess, returning None for every hash and skipping all dedup.

    Duplicate detection uses vectorised numpy XOR + popcount → O(n) per image
    against a growing array of kept hashes, vs the old O(n²) Python loop.
    """
    if not PHASH_AVAILABLE:
        return paths

    max_workers = max(1, (os.cpu_count() or 1) - 2)

    # ── Compute hashes in parallel (threads, not processes) ───────────────────
    hash_map: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as tex:
        futures = {tex.submit(compute_phash, p): p for p in paths}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                h = fut.result()
            except Exception as exc:
                print(f"  [WARN] pHash failed for {p.name}: {exc}")
                h = None
            if h is not None:
                hash_map[p] = h
            else:
                rejected.append({"path": str(p), "category": cls,
                                  "reason": "phash_failed"})
                quarantine_file(p, "phash_failed", cls)

    # ── Vectorised dedup ──────────────────────────────────────────────────────
    kept: list[Path]    = []
    kept_ints: list[int] = []          # uint64 ints for XOR comparisons

    for p in paths:                    # iterate in original order for determinism
        h = hash_map.get(p)
        if h is None:
            continue                   # already logged above
        h_int = int(h, 16)
        if kept_ints:
            arr       = np.array(kept_ints, dtype=np.uint64)
            distances = _popcount_array(arr ^ np.uint64(h_int))
            if int(distances.min()) <= PHASH_HAMMING_THRESHOLD:
                rejected.append({"path": str(p), "category": cls,
                                  "reason": "phash_near_duplicate"})
                quarantine_file(p, "phash_near_duplicate", cls)
                continue
        kept.append(p)
        kept_ints.append(h_int)

    return kept


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — CROSS-CLASS pHASH DUPLICATE
# ══════════════════════════════════════════════════════════════════════════════
def phash_cross_class_check(all_images: dict[str, list[Path]],
                             rejected: list[dict]) -> dict[str, list[Path]]:
    """
    After within-class dedup, check for near-duplicates across classes.
    These are especially dangerous — same image, conflicting labels.

    Old approach: O(n²) nested Python loop.
    New approach:
      - Compute hashes in parallel with ThreadPoolExecutor (avoids worker
        re-import issues that silently break imagehash in ProcessPoolExecutor).
      - For each image, do a single vectorised XOR against ALL other-class hashes
        in one numpy call → effectively O(n) per image with tiny constants.
    """
    if not PHASH_AVAILABLE:
        return all_images

    max_workers = max(1, (os.cpu_count() or 1) - 2)

    # ── Compute hashes in parallel (threads, not processes) ───────────────────
    flat: list[tuple[Path, str]] = [
        (p, cls) for cls, paths in all_images.items() for p in paths
    ]
    hash_map: dict[Path, tuple[str, str]] = {}   # path → (cls, hex_hash)
    with ThreadPoolExecutor(max_workers=max_workers) as tex:
        futures = {tex.submit(compute_phash, p): (p, cls) for p, cls in flat}
        for fut in as_completed(futures):
            p, cls = futures[fut]
            try:
                h = fut.result()
            except Exception as exc:
                print(f"  [WARN] pHash failed for {p.name}: {exc}")
                h = None
            if h is not None:
                hash_map[p] = (cls, h)

    # ── Build per-class uint64 arrays ─────────────────────────────────────────
    class_paths:  dict[str, list[Path]] = {cls: [] for cls in all_images}
    class_hashes: dict[str, list[int]]  = {cls: [] for cls in all_images}
    for p, (cls, h) in hash_map.items():
        class_paths[cls].append(p)
        class_hashes[cls].append(int(h, 16))

    class_arrays: dict[str, np.ndarray] = {
        cls: np.array(ints, dtype=np.uint64) if ints else np.array([], dtype=np.uint64)
        for cls, ints in class_hashes.items()
    }

    # ── Vectorised cross-class comparison ─────────────────────────────────────
    conflicts: set[Path] = set()
    classes = list(all_images.keys())

    for i, cls_a in enumerate(classes):
        arr_a = class_arrays[cls_a]
        if arr_a.size == 0:
            continue
        # Concatenate all other-class hashes into a single array for one XOR pass
        other_arrays = [class_arrays[cls_b] for cls_b in classes if cls_b != cls_a]
        if not any(a.size for a in other_arrays):
            continue
        arr_other = np.concatenate([a for a in other_arrays if a.size])

        # Build a reverse index: position in arr_other → (cls_b, path)
        other_meta: list[tuple[str, Path]] = []
        for cls_b in classes:
            if cls_b == cls_a:
                continue
            for p_b in class_paths[cls_b]:
                other_meta.append((cls_b, p_b))

        for idx_a, (p_a, h_a) in enumerate(zip(class_paths[cls_a], arr_a)):
            distances = _popcount_array(arr_other ^ h_a)
            near = np.where(distances <= PHASH_HAMMING_THRESHOLD)[0]
            for idx_b in near:
                cls_b, p_b = other_meta[int(idx_b)]
                if p_a not in conflicts:
                    rejected.append({
                        "path": str(p_a), "category": cls_a,
                        "reason": (f"cross_class_near_duplicate:"
                                   f"similar_to_{cls_b}:{p_b.name}"),
                    })
                    quarantine_file(p_a, "cross_class_near_duplicate", cls_a)
                    conflicts.add(p_a)
                if p_b not in conflicts:
                    rejected.append({
                        "path": str(p_b), "category": cls_b,
                        "reason": (f"cross_class_near_duplicate:"
                                   f"similar_to_{cls_a}:{p_a.name}"),
                    })
                    quarantine_file(p_b, "cross_class_near_duplicate", cls_b)
                    conflicts.add(p_b)

    result = {
        cls: [p for p in paths if p not in conflicts]
        for cls, paths in all_images.items()
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — LOW GREEN CONTENT FLAG (non-reject — manual review)
# ══════════════════════════════════════════════════════════════════════════════
def check_green_content(img: Image.Image) -> str:
    """
    Returns a flag string if green coverage is very low.
    Does NOT reject — heavily bleached MSV images may have low green.
    Returns empty string if green content is normal.
    """
    try:
        import cv2
        arr_rgb = np.array(img.convert("RGB"))
        arr_hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = arr_hsv[:, :, 0], arr_hsv[:, :, 1], arr_hsv[:, :, 2]
        green = ((h >= GREEN_H_MIN) & (h <= GREEN_H_MAX) &
                 (s >= GREEN_S_MIN) & (v >= GREEN_V_MIN))
        coverage = green.sum() / max(arr_rgb.shape[0] * arr_rgb.shape[1], 1)
        if coverage < GREEN_CONTENT_THRESHOLD:
            return f"low_green:{coverage:.3f}"
    except Exception:
        pass
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# FULL VALIDATION PIPELINE FOR ONE IMAGE
# ══════════════════════════════════════════════════════════════════════════════
def validate_image(path: Path, cls: str) -> tuple[bool, str, str, dict]:
    """
    Run all per-image checks (Steps 1–6 + 10).
    Returns (is_valid, rejection_reason, warning_flag, metadata_dict).
    metadata contains: width, height, mode, file_size_kb.
    """
    meta = {"width": 0, "height": 0, "mode": "", "file_size_kb": 0}

    # Step 1 — zero byte
    reason = check_zero_byte(path)
    if reason:
        return False, reason, "", meta

    meta["file_size_kb"] = round(path.stat().st_size / 1024, 1)

    # Step 2 — magic bytes
    reason = check_magic_bytes(path)
    if reason:
        return False, reason, "", meta

    # Step 3 — truncation (opens image fully)
    reason, img = check_truncated(path)
    if reason or img is None:
        return False, reason or "open_failed", "", meta

    meta["width"], meta["height"] = img.size
    meta["mode"] = img.mode

    # Step 4 — resolution
    reason = check_resolution(img)
    if reason:
        img.close()
        return False, reason, "", meta

    # Step 5 — colour mode
    reason, flag = check_colour_mode(img)
    if reason:
        img.close()
        return False, reason, flag, meta

    # Step 6 — near-uniform
    reason = check_uniform(img)
    if reason:
        img.close()
        return False, reason, flag, meta

    # Step 10 — low green content flag (no rejection)
    green_flag = check_green_content(img)
    combined_flag = " | ".join(f for f in [flag, green_flag] if f)
    
    img.close()
    # Calculate MD5
    md5_hash = compute_md5(path)
    
    # Calculate pHash
    phash = compute_phash(path) if PHASH_AVAILABLE else None
    return True, "", combined_flag, meta


# ══════════════════════════════════════════════════════════════════════════════
# STRATIFIED SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def stratified_split(images: list[Path],
                     ratios: dict) -> dict[str, list[Path]]:
    shuffled = images.copy()
    random.shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train: n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    set_seeds(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Phase 0-pre: Dataset Preprocessing + Partition")
    print("=" * 72)

    if not MAIZE_DIR.is_dir():
        print(f"[FATAL] maize_dataset/ not found at: {MAIZE_DIR}")
        return

    if GLOBAL_MANIFEST.exists():
        answer = input(f"  {GLOBAL_MANIFEST.name} already exists. Overwrite? [y/N]: ")
        if answer.strip().lower() != "y":
            print("  Aborted.")
            return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Quarantine dir   : {QUARANTINE_DIR}")

    # Leave 2 cores free so your Fedora system remains responsive
    max_workers = max(1, (os.cpu_count() or 1) - 2)

    # ── Phase A: Per-image validation (Multiprocessed) ─────────────────────────
    print("\n  Phase A: Per-image validation (Multiprocessed)...")
    raw_valid:   dict[str, list[Path]] = {}
    all_rejected: list[dict]           = []
    all_flagged:  list[dict]           = []

    for cls in CLASSES:
        cat_dir = MAIZE_DIR / cls
        if not cat_dir.is_dir():
            print(f"  [WARN] {cat_dir} not found, skipping.")
            continue

        all_files = sorted([
            p for p in cat_dir.rglob("*")
            if p.suffix in set(VALID_EXTENSIONS) and p.is_file()
        ])
        print(f"\n  [{cls}] {len(all_files):,} files found")

        valid_paths = []
        n_rej = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(validate_image, p, cls): p for p in all_files}

            for i, future in enumerate(as_completed(future_to_path)):
                p = future_to_path[future]
                try:
                    is_valid, reason, flag, meta = future.result()
                    if not is_valid:
                        all_rejected.append({
                            "path": str(p), "category": cls,
                            "reason": reason,
                            **meta,
                        })
                        quarantine_file(p, reason, cls)
                        n_rej += 1
                    else:
                        valid_paths.append(p)
                        if flag:
                            all_flagged.append({
                                "path": str(p), "category": cls,
                                "flag": flag,
                                **meta,
                            })
                except Exception as exc:
                    print(f"  [ERROR] {p.name} generated an exception: {exc}")

                if (i + 1) % 5000 == 0:
                    print(f"    Validated {i+1:,}/{len(all_files):,} "
                          f"({n_rej:,} rejected so far)")

        raw_valid[cls] = valid_paths
        print(f"    Valid: {len(valid_paths):,}  |  Rejected: {n_rej:,}")

    # ── Phase B: MD5 exact duplicates (Steps 7 — cross-class) ─────────────────
    print("\n  Phase B: MD5 exact duplicate removal (parallelised) ...")
    after_md5 = remove_exact_duplicates(raw_valid, all_rejected)
    for cls in CLASSES:
        removed = len(raw_valid.get(cls, [])) - len(after_md5.get(cls, []))
        if removed > 0:
            print(f"    [{cls}] removed {removed:,} MD5 duplicates")

    # ── Phase C: pHash near-duplicate within class (Step 8) ───────────────────
    print("\n  Phase C: pHash near-duplicate removal (within class, parallelised + vectorised) ...")
    after_phash: dict[str, list[Path]] = {}
    for cls in CLASSES:
        paths = after_md5.get(cls, [])
        before = len(paths)
        after_phash[cls] = phash_dedup_within_class(paths, cls, all_rejected)
        removed = before - len(after_phash[cls])
        if removed > 0:
            print(f"    [{cls}] removed {removed:,} pHash near-duplicates")

    # ── Phase D: Cross-class pHash duplicate detection (Step 9) ───────────────
    print("\n  Phase D: Cross-class pHash duplicate detection (vectorised) ...")
    after_crossclass = phash_cross_class_check(after_phash, all_rejected)
    for cls in CLASSES:
        removed = len(after_phash.get(cls, [])) - len(after_crossclass.get(cls, []))
        if removed > 0:
            print(f"    [{cls}] removed {removed:,} cross-class conflicts")

    # ── Phase E: Stratified split ──────────────────────────────────────────────
    print("\n  Phase E: Stratified 70/15/15 split ...")
    manifest_rows: list[dict] = []
    split_counts:  dict       = {}

    for cls in CLASSES:
        paths = after_crossclass.get(cls, [])
        if not paths:
            print(f"  [WARN] No valid images for {cls} after preprocessing.")
            continue
        splits = stratified_split(paths, SPLIT_RATIOS)
        split_counts[cls] = {s: len(v) for s, v in splits.items()}
        for split_name, split_paths in splits.items():
            for p in split_paths:
                manifest_rows.append({
                    "source_path": str(p),
                    "filename":    p.name,
                    "category":    cls,
                    "split":       split_name,
                })
        print(f"  [{cls}] train {split_counts[cls]['train']:,} | "
              f"val {split_counts[cls]['val']:,} | "
              f"test {split_counts[cls]['test']:,}")

    # ── Write global manifest ──────────────────────────────────────────────────
    with open(GLOBAL_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source_path", "filename", "category", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # ── Write preprocessing report ────────────────────────────────────────────
    report_path = REPORTS_DIR / "preprocessing_report.csv"
    if all_rejected:
        fields = list(all_rejected[0].keys())
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rejected)

    flagged_path = REPORTS_DIR / "preprocessing_flagged.csv"
    if all_flagged:
        fields = list(all_flagged[0].keys())
        with open(flagged_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_flagged)

    # ── Rejection reason breakdown ────────────────────────────────────────────
    from collections import Counter
    reason_counts = Counter(
        r["reason"].split(":")[0] for r in all_rejected)

    # ── Write summary txt ──────────────────────────────────────────────────────
    summary_path = REPORTS_DIR / "preprocessing_summary.txt"
    lines = [
        "Yellow MAIze — Preprocessing Summary",
        "=" * 50,
        f"Total files scanned     : {sum(len(v) for v in raw_valid.values()) + len(all_rejected):,}",
        f"Rejected (all reasons)  : {len(all_rejected):,}",
        f"Flagged (review needed) : {len(all_flagged):,}",
        f"Final valid images      : {len(manifest_rows):,}",
        "",
        "Rejection reasons:",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"  {reason:<35} {count:>6,}")
    lines += [
        "",
        "Split breakdown:",
        f"  {'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8}",
    ]
    for cls, counts in split_counts.items():
        lines.append(
            f"  {cls:<12} {counts.get('train',0):>8,} "
            f"{counts.get('val',0):>8,} {counts.get('test',0):>8,}")
    summary_path.write_text("\n".join(lines))

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  Manifest         : {GLOBAL_MANIFEST}  ({len(manifest_rows):,} images)")
    print(f"  Rejected         : {len(all_rejected):,}  (moved to {QUARANTINE_DIR})")
    print(f"  Flagged          : {len(all_flagged):,}  (kept, needs review)")
    if reason_counts:
        print(f"\n  Rejection breakdown:")
        for reason, count in reason_counts.most_common():
            print(f"    {reason:<35} {count:>6,}")
    print(f"\n  Reports:")
    print(f"    {report_path}")
    print(f"    {flagged_path}")
    print(f"    {summary_path}")
    print(f"\n  NEXT STEP: python create_bouncer_dataset.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
