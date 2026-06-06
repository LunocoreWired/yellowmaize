"""
================================================================================
 factory_master.py — Phase 4: Pseudo-Label Generation (All 4 Modes)
================================================================================
 PURPOSE:
   Process all train+val maize images to generate pseudo-labels for
   Student training. Runs all 4 Factory modes in sequence, each writing
   to its own subfolder so all mode outputs coexist on disk.

 MODES:
   mode_a — Otsu threshold silhouette + hard binary HSV symptom mask
   mode_b — SAM2 hard binary silhouette + hard binary HSV symptom mask
   mode_c — SAM2 soft float silhouette + hard binary HSV symptom mask
   mode_d — SAM2 soft float silhouette + soft HSV confidence symptom map

 PER-IMAGE PROCESSING (Stages):
   Stage 1 — Bouncer gate (heuristic pre-filter + neural classifier)
   Stage 2 — Tier 1 check: if Tier 1, load SAM2 .npy; else run Teacher
   Stage 3 — Silhouette refinement (threshold + morphological ops)
   Stage 4 — Coverage guard → reliability weight
   Stage 5 — HSV symptom masking (hard or soft depending on mode)
   Stage 6 — Severity computation

 OUTPUTS per mode subfolder:
   {stem}_silhouette.npy   ← float32 leaf silhouette probability
   {stem}_symptom.npy      ← float32 symptom confidence (mode_d) OR
   {stem}_symptom.png      ← uint8 binary symptom mask (modes a/b/c)
   {stem}_sev.txt          ← severity % or -1 (sentinel)
   {stem}_weight.txt       ← reliability weight [0.3, 0.7, 1.0] or -1
   phase4_report_{mode}.csv ← per-image log
================================================================================
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from image_utils import load_image_rgb, load_image_clahe, to_hsv, rgb_to_bgr
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from config import (
    SEED,
    GLOBAL_MANIFEST, TIER1_MANIFEST,
    TIER1_MASKS_DIR, PSEUDO_DIR, REPORTS_DIR,
    TEACHER_CKPT_DIR,
    BOUNCER_CKPT_DIR, BOUNCER_DEPLOYED_VARIANT,
    TEACHER_DEPLOYED_VARIANT, TEACHER_IMG_SIZE,
    STUDENT_IMG_SIZE,
    FACTORY_MODES,
    FACTORY_SILHOUETTE_THRESHOLD,
    FACTORY_MIN_LEAF_COVERAGE,
    FACTORY_WEIGHT_BRACKETS,
    FACTORY_R3_MIN_AREA_PX,
    FACTORY_MORPH_KERNEL_SIZE,
    HSV_GREEN_EXCL,
    HSV_MSV_RANGES, HSV_MLN_RANGES,
    BOUNCER_MIN_ASPECT_RATIO, BOUNCER_MAX_ASPECT_RATIO,
    BOUNCER_MIN_GREEN_COVERAGE, BOUNCER_THRESHOLD,
    VALID_EXTENSIONS, CLASSES,
)
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision import models


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_INFER_TF = A.Compose([
    A.LongestMaxSize(max_size=TEACHER_IMG_SIZE),
    A.PadIfNeeded(TEACHER_IMG_SIZE, TEACHER_IMG_SIZE, border_mode=0, value=0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_bouncer() -> tuple[nn.Module, float]:
    ckpt_path = BOUNCER_CKPT_DIR / f"bouncer_{BOUNCER_DEPLOYED_VARIANT}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Bouncer checkpoint not found: {ckpt_path}")

    model = models.mobilenet_v3_large(weights=None)
    in_f  = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 1)

    ckpt      = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    threshold = BOUNCER_THRESHOLD
    model.to(DEVICE).eval()
    return model, threshold


def load_teacher() -> nn.Module:
    ckpt_path = TEACHER_CKPT_DIR / "teacher_model_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    variant = ckpt.get("variant", TEACHER_DEPLOYED_VARIANT)

    encoder_map = {
        "resnet50":       "resnet50",
        "efficientnet-b2":"efficientnet-b2",
        "mit_b2":         "mit_b2",
    }
    model = smp.Unet(
        encoder_name=encoder_map.get(variant, "efficientnet-b2"),
        encoder_weights=None,
        in_channels=3, classes=1, activation=None,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — BOUNCER GATE
# Shared helpers imported from scripts/bouncer_inference.py — single source
# of truth used by both factory_master.py and train_bouncer.py.
# ══════════════════════════════════════════════════════════════════════════════

from scripts.bouncer_inference import heuristic_prefilter, neural_bouncer  # noqa: E402



# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — TEACHER INFERENCE / TIER 1 LOAD
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def teacher_infer(img_rgb: np.ndarray, teacher: nn.Module,
                  orig_h: int, orig_w: int) -> np.ndarray:
    """Run Teacher inference. Returns float32 probability map at orig resolution."""
    tensor  = TEACHER_INFER_TF(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    logits  = teacher(tensor)
    prob    = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
    # Resize back to original image dimensions
    prob    = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return prob


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3-4 — SILHOUETTE REFINEMENT + COVERAGE WEIGHT
# ══════════════════════════════════════════════════════════════════════════════

def refine_silhouette(prob_map: np.ndarray) -> np.ndarray:
    """
    Threshold + morphological close/open to clean up silhouette.
    Uses FACTORY_SILHOUETTE_THRESHOLD (0.35) — lower than 0.5 to
    catch darker leaves.
    """
    binary = (prob_map >= FACTORY_SILHOUETTE_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (FACTORY_MORPH_KERNEL_SIZE, FACTORY_MORPH_KERNEL_SIZE))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    return binary


def get_reliability_weight(leaf_coverage: float) -> float | None:
    """
    Map leaf coverage fraction to a reliability weight.
    Returns None if image should be excluded (sentinel -1).
    """
    for lo, hi, weight in FACTORY_WEIGHT_BRACKETS:
        if lo <= leaf_coverage < hi:
            return weight
    return FACTORY_WEIGHT_BRACKETS[-1][2]   # > last bracket high


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — HSV SYMPTOM MASKING
# ══════════════════════════════════════════════════════════════════════════════

def _in_range(h, s, v, rng: dict) -> np.ndarray:
    h_lo, h_hi = rng["h"]
    s_lo, s_hi = rng["s"]
    v_lo, v_hi = rng["v"]
    return (
        (h >= h_lo) & (h <= h_hi) &
        (s >= s_lo) & (s <= s_hi) &
        (v >= v_lo) & (v <= v_hi)
    )


def _green_exclusion(h, s, v) -> np.ndarray:
    ge = HSV_GREEN_EXCL
    return (
        (h >= ge["h_min"]) & (h <= ge["h_max"]) &
        (s >= ge["s_min"]) &
        (v >= ge["v_min"]) & (v <= ge["v_max"])
    )


def _apply_r3_area_filter(mask: np.ndarray) -> np.ndarray:
    """Remove small connected components from R3 (near-white bleached range)."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= FACTORY_R3_MIN_AREA_PX:
            filtered[labels == lbl] = 1
    return filtered


def compute_hsv_hard_mask(img_rgb: np.ndarray,
                           silhouette: np.ndarray,
                           category: str) -> np.ndarray:
    """
    Hard binary symptom mask (modes a, b, c).
    Returns uint8 array of 0/255.
    """
    img_hsv = to_hsv(img_rgb)   # image_utils: guarantees RGB→HSV
    h = img_hsv[:, :, 0].astype(np.int32)
    s = img_hsv[:, :, 1].astype(np.int32)
    v = img_hsv[:, :, 2].astype(np.int32)

    green_excl = _green_exclusion(h, s, v)
    ranges     = HSV_MSV_RANGES if category == "MSV" else HSV_MLN_RANGES

    symptom = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for i, rng in enumerate(ranges):
        hit = _in_range(h, s, v, rng)
        # Apply R3 area filter for MSV only
        if category == "MSV" and i == 2:   # R3 index
            hit = _apply_r3_area_filter(hit.astype(np.uint8)).astype(bool)
        symptom |= hit.astype(np.uint8)

    # Remove green exclusion + apply leaf silhouette mask
    symptom &= (~green_excl).astype(np.uint8)
    symptom &= silhouette
    return (symptom * 255).astype(np.uint8)


def compute_hsv_soft_confidence(img_rgb: np.ndarray,
                                 silhouette: np.ndarray,
                                 category: str) -> np.ndarray:
    """
    Soft HSV confidence map (mode d).
    confidence = (Σ range_hit × normalized_distance_to_center) / N_ranges
    Returns float32 array in [0,1].
    """
    img_hsv = to_hsv(img_rgb)   # image_utils: guarantees RGB→HSV
    h = img_hsv[:, :, 0].astype(np.float32)
    s = img_hsv[:, :, 1].astype(np.float32)
    v = img_hsv[:, :, 2].astype(np.float32)

    green_excl = _green_exclusion(
        h.astype(np.int32), s.astype(np.int32), v.astype(np.int32))
    ranges     = HSV_MSV_RANGES if category == "MSV" else HSV_MLN_RANGES
    n_ranges   = len(ranges)

    conf_accum = np.zeros(img_rgb.shape[:2], dtype=np.float32)

    for i, rng in enumerate(ranges):
        h_lo, h_hi = rng["h"]
        s_lo, s_hi = rng["s"]
        v_lo, v_hi = rng["v"]

        hit = _in_range(h.astype(np.int32), s.astype(np.int32),
                        v.astype(np.int32), rng)

        if category == "MSV" and i == 2:
            hit = _apply_r3_area_filter(hit.astype(np.uint8)).astype(bool)

        # Normalized distance to range center (1.0 at center, 0.0 at edge)
        h_center = (h_lo + h_hi) / 2.0
        s_center = (s_lo + s_hi) / 2.0
        v_center = (v_lo + v_hi) / 2.0

        h_half = max((h_hi - h_lo) / 2.0, 1.0)
        s_half = max((s_hi - s_lo) / 2.0, 1.0)
        v_half = max((v_hi - v_lo) / 2.0, 1.0)

        d_h = 1.0 - np.abs(h - h_center) / h_half
        d_s = 1.0 - np.abs(s - s_center) / s_half
        d_v = 1.0 - np.abs(v - v_center) / v_half
        d   = np.clip((d_h + d_s + d_v) / 3.0, 0.0, 1.0)

        conf_accum += hit.astype(np.float32) * d

    conf_map  = conf_accum / n_ranges
    conf_map *= (~green_excl).astype(np.float32)
    conf_map *= silhouette.astype(np.float32)
    return np.clip(conf_map, 0.0, 1.0)


def compute_severity(symptom_mask: np.ndarray,
                     silhouette: np.ndarray) -> float:
    """severity = (symptom pixels / leaf pixels) × 100"""
    leaf_px    = float(silhouette.sum())
    if leaf_px < 1:
        return -1.0
    symptom_px = float((symptom_mask > 0).sum())
    return (symptom_px / leaf_px) * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# OTSU SILHOUETTE (Mode A)
# ══════════════════════════════════════════════════════════════════════════════

def otsu_silhouette(img_rgb: np.ndarray) -> np.ndarray:
    """Otsu threshold on grayscale. Returns binary uint8 mask."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS ONE IMAGE — ALL MODES
# ══════════════════════════════════════════════════════════════════════════════

def process_image(img_path: Path, category: str,
                  is_tier1: bool,
                  tier1_stem_set: set[str],
                  bouncer_model: nn.Module, bouncer_thresh: float,
                  teacher_model: nn.Module,
                  mode_dirs: dict[str, Path]) -> dict:
    """
    Process one image through all 4 Factory modes simultaneously.
    Returns a result dict with per-mode outputs.
    """
    result = {
        "filename": img_path.name,
        "category": category,
        "is_tier1": is_tier1,
    }

    # ── Load image — EXIF-corrected + CLAHE enhanced ──────────────────────────
    # CLAHE improves HSV masking accuracy on low-contrast / overexposed images
    img_rgb = load_image_clahe(img_path)
    if img_rgb is None:
        result["status"] = "load_error:corrupt_or_truncated"
        return result

    orig_h, orig_w = img_rgb.shape[:2]
    stem = img_path.stem

    # ── Stage 1: Bouncer gate ─────────────────────────────────────────────────
    if not heuristic_prefilter(img_rgb):
        result["status"] = "filtered_heuristic"
        return result

    if not neural_bouncer(img_rgb, bouncer_model, bouncer_thresh):
        result["status"] = "filtered_bouncer"
        return result

    # ── Stage 2: Get leaf silhouette ──────────────────────────────────────────
    if is_tier1:
        # Use pre-existing SAM2 .npy — higher quality than Teacher
        # Match dest_filename stem pattern used in tier1_masks
        npy_candidates = list(TIER1_MASKS_DIR.glob(f"*{img_path.stem}*softmask.npy"))
        if npy_candidates:
            soft_prob = np.load(str(npy_candidates[0])).astype(np.float32)
            soft_prob = cv2.resize(soft_prob, (orig_w, orig_h),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback to Teacher if SAM2 mask not found
            soft_prob = teacher_infer(img_rgb, teacher_model, orig_h, orig_w)
    else:
        # Tier 2: Teacher inference
        soft_prob = teacher_infer(img_rgb, teacher_model, orig_h, orig_w)

    # Hard binary silhouette (used for modes a, b, and metric computation)
    binary_sil = refine_silhouette(soft_prob)

    # ── Stage 3: Otsu silhouette (mode_a only) ────────────────────────────────
    otsu_sil = otsu_silhouette(img_rgb)

    # ── Stage 4: Coverage → reliability weight ────────────────────────────────
    total_px   = orig_h * orig_w
    leaf_px    = float(binary_sil.sum())
    coverage   = leaf_px / max(total_px, 1)
    weight     = get_reliability_weight(coverage)

    result["coverage"] = round(coverage, 4)
    result["weight"]   = weight if weight is not None else -1

    # ── Stage 5+6: Compute symptom masks and severity for each mode ───────────
    modes_data = {}

    for mode in FACTORY_MODES:
        if mode == "mode_a":
            sil = otsu_sil
        elif mode in ("mode_b", "mode_c"):
            sil = binary_sil
        else:  # mode_d
            sil = binary_sil

        # Symptom mask
        if mode == "mode_d":
            symptom = compute_hsv_soft_confidence(img_rgb, sil, category)
        else:
            symptom = compute_hsv_hard_mask(img_rgb, sil, category)

        # Severity (always from binary symptom)
        if mode == "mode_d":
            sym_binary = (symptom >= 0.3).astype(np.uint8)
        else:
            sym_binary = (symptom > 0).astype(np.uint8)

        sev = compute_severity(sym_binary, sil) if weight is not None else -1.0

        modes_data[mode] = {
            "silhouette": soft_prob if mode in ("mode_c", "mode_d") else sil.astype(np.float32),
            "symptom":    symptom,
            "severity":   sev,
        }

    # ── Write outputs for each mode ───────────────────────────────────────────
    for mode, data in modes_data.items():
        mode_dir = mode_dirs[mode]

        # Silhouette (Downscaled to save storage)
        sil_out = cv2.resize(data["silhouette"].astype(np.float32), 
                             (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE), 
                             interpolation=cv2.INTER_LINEAR)
        sil_path = mode_dir / f"{stem}_silhouette.npy"
        np.save(str(sil_path), sil_out)

        # Symptom mask
        if mode == "mode_d":
            sym_out = cv2.resize(data["symptom"].astype(np.float32), 
                                 (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE), 
                                 interpolation=cv2.INTER_LINEAR)
            sym_path = mode_dir / f"{stem}_symptom.npy"
            np.save(str(sym_path), sym_out)
        else:
            sym_path = mode_dir / f"{stem}_symptom.png"
            # Downscale the PNG mask using NEAREST to preserve binary 0/255 boundaries
            sym_out = cv2.resize(data["symptom"], 
                                 (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE), 
                                 interpolation=cv2.INTER_NEAREST)
            
            if sym_out.dtype == np.uint8 and len(sym_out.shape) == 2:
                cv2.imwrite(str(sym_path), sym_out)   # grayscale — no conversion
            else:
                cv2.imwrite(str(sym_path), sym_out)

        # Severity
        sev_path = mode_dir / f"{stem}_sev.txt"
        sev_path.write_text(str(round(data["severity"], 4)))

        # Reliability weight
        wt_path = mode_dir / f"{stem}_weight.txt"
        wt_path.write_text(str(result["weight"]))

    result["status"] = "processed"
    result.update({
        f"{m}_sev": round(modes_data[m]["severity"], 4)
        for m in FACTORY_MODES
    })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# FACTORY SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def generate_factory_summary(report_rows: list[dict],
                              mode_dirs: dict) -> None:
    """
    Aggregate pseudo-label quality statistics per mode × class.
    Writes reports/factory_summary.csv — evidence for ablation table.

    Metrics per mode × class:
      - n_images, n_processed, pct_processed
      - mean_severity, std_severity, median_severity
      - pct_symptomatic (severity > 0)
      - mean_weight, pct_excluded (weight == -1)
      - mean_silhouette_confidence (modes c/d only)
    """
    import pandas as pd, csv as _csv, numpy as _np
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(report_rows)
    summary_rows = []

    for mode in FACTORY_MODES:
        sev_col = f"{mode}_sev"
        if sev_col not in df.columns:
            continue

        for cls in CLASSES:
            cls_df = df[df["category"] == cls].copy() if "category" in df.columns else df

            # Processed only (status == "processed")
            proc = cls_df[cls_df.get("status", pd.Series(["processed"]*len(cls_df))) == "processed"]

            sevs      = pd.to_numeric(proc[sev_col], errors="coerce").dropna()
            valid_sev = sevs[sevs >= 0]
            excluded  = (proc["weight"] == -1).sum() if "weight" in proc.columns else 0

            row = {
                "mode":             mode,
                "class":            cls,
                "n_total":          len(cls_df),
                "n_processed":      len(proc),
                "pct_processed":    round(100*len(proc)/max(len(cls_df),1), 1),
                "mean_severity":    round(valid_sev.mean(), 2) if len(valid_sev) else "N/A",
                "std_severity":     round(valid_sev.std(),  2) if len(valid_sev) else "N/A",
                "median_severity":  round(valid_sev.median(),2) if len(valid_sev) else "N/A",
                "pct_symptomatic":  round(100*(valid_sev>0).mean(),1) if len(valid_sev) else "N/A",
                "n_excluded":       int(excluded),
                "pct_excluded":     round(100*excluded/max(len(proc),1),1),
            }

            # Silhouette confidence for soft modes (c, d)
            if mode in ("mode_c", "mode_d"):
                conf_vals = []
                for _, r in proc.head(500).iterrows():   # sample 500 for speed
                    stem    = Path(r.get("filename","")).stem if "filename" in r else ""
                    npy_p   = mode_dirs.get(mode, Path(".")) / f"{stem}_silhouette.npy"
                    if npy_p.exists():
                        try:
                            arr = _np.load(str(npy_p))
                            conf_vals.append(float(arr.mean()))
                        except Exception:
                            pass
                row["mean_sil_confidence"] = round(_np.mean(conf_vals), 4) if conf_vals else "N/A"

            summary_rows.append(row)

    if summary_rows:
        summary_path = REPORTS_DIR / "factory_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=summary_rows[0].keys(),
                                extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"  Factory summary → {summary_path}")

        # Quick per-mode print
        print(f"\n  {'Mode':<8} {'Class':<12} {'Mean Sev':>9} {'% Symptomatic':>14}")
        print(f"  {'─'*8} {'─'*12} {'─'*9} {'─'*14}")
        for r in summary_rows:
            print(f"  {r['mode']:<8} {r['class']:<12} "
                  f"{str(r['mean_severity']):>9} {str(r['pct_symptomatic']):>13}%")



def main() -> None:
    import random, time
    random.seed(42)   # reproducibility anchor for any future sampling
    _t_factory_start = time.time()
    print("=" * 72)
    print("  Yellow MAIze | Phase 4: Factory Pseudo-Label Generation")
    print("=" * 72)

    # ── Check prerequisites ───────────────────────────────────────────────────
    for path, name in [
        (GLOBAL_MANIFEST,  "global_split_manifest.csv"),
        (TIER1_MANIFEST,   "tier1_manifest.csv"),
    ]:
        if not path.exists():
            print(f"[FATAL] {name} not found.")
            return

    # ── Create mode output directories ───────────────────────────────────────
    mode_dirs = {}
    for mode in FACTORY_MODES:
        d = PSEUDO_DIR / mode
        d.mkdir(parents=True, exist_ok=True)
        mode_dirs[mode] = d

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n  Loading models ...")
    bouncer_model, bouncer_thresh = load_bouncer()
    teacher_model = load_teacher()
    print(f"  Bouncer threshold: {bouncer_thresh:.4f}")

    # ── Build image list (train+val only) ─────────────────────────────────────
    global_df = pd.read_csv(GLOBAL_MANIFEST)
    trainval  = global_df[global_df["split"] != "test"]

    # Tier 1 filename set (original source filenames)
    tier1_df    = pd.read_csv(TIER1_MANIFEST)
    tier1_fnames = set(
        Path(p).name for p in tier1_df["source_path"].tolist()
    )

    print(f"\n  Images to process : {len(trainval):,}")
    print(f"  Tier 1 images     : {len(tier1_fnames):,} (Teacher skipped)")

    # ── Process all images ────────────────────────────────────────────────────
    report_rows = []
    t_start     = time.time()
    n_processed = 0
    n_filtered  = 0
    n_errors    = 0

    for i, (_, row) in enumerate(trainval.iterrows()):
        img_path = Path(row["source_path"])
        category = row["category"]
        is_tier1 = img_path.name in tier1_fnames

        if not img_path.exists():
            n_errors += 1
            continue

        result = process_image(
            img_path, category, is_tier1,
            tier1_fnames,
            bouncer_model, bouncer_thresh,
            teacher_model, mode_dirs,
        )
        report_rows.append(result)

        if result["status"] == "processed":
            n_processed += 1
        elif "filter" in result.get("status", ""):
            n_filtered += 1
        else:
            n_errors += 1

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / max(elapsed, 1)
            eta     = (len(trainval) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1:>7}/{len(trainval)}] "
                  f"processed {n_processed:,} | "
                  f"filtered {n_filtered:,} | "
                  f"ETA {eta/60:.1f} min")

    # ── Write per-mode reports ────────────────────────────────────────────────
    for mode in FACTORY_MODES:
        report_path = REPORTS_DIR / f"phase4_report_{mode}.csv"
        if report_rows:
            keys = [k for k in report_rows[0].keys()]
            with open(report_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(report_rows)
        print(f"  Report: {report_path}")

    # ── Filter breakdown summary ──────────────────────────────────────────────
    from collections import Counter
    status_counts = Counter(r.get("status","unknown") for r in report_rows)

    # ── Summary ───────────────────────────────────────════════════════════════
    total = len(report_rows)
    print(f"\n{'─' * 72}")
    print(f"  Total images     : {total:,}")
    print(f"  Processed        : {n_processed:,}  ({100*n_processed/max(total,1):.1f}%)")
    print(f"  Filtered out     : {n_filtered:,}   ({100*n_filtered/max(total,1):.1f}%)")
    print(f"  Errors           : {n_errors:,}")
    print(f"\n  Filter breakdown by stage:")
    for status, cnt in status_counts.most_common():
        print(f"    {status:<30} {cnt:>7,}")

    # Save filter breakdown
    breakdown_path = REPORTS_DIR / "factory_filter_breakdown.csv"
    import csv as _csv
    with open(breakdown_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["status", "count", "pct"])
        for status, cnt in status_counts.most_common():
            w.writerow([status, cnt, round(100*cnt/max(total,1), 2)])
    print(f"\n  Filter breakdown: {breakdown_path}")
    print(f"  Mode outputs written to:")
    for mode, d in mode_dirs.items():
        print(f"    {mode}: {d}")
    # ── Factory summary statistics ─────────────────────────────────────────────
    print(f"\n  Generating factory summary statistics ...")
    generate_factory_summary(report_rows, mode_dirs)

    _duration = round(time.time() - _t_factory_start, 1)
    print(f"\n  Total Factory duration: {_duration}s")
    print(f"\n  NEXT STEP: python train_student.py --stage 1")
    print("=" * 72)


if __name__ == "__main__":
    main()
