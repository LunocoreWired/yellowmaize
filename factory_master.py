"""
factory_master.py — Phase 4: Pseudo-Label Generation (Optimized & Fixed)
Hardware Target: Ryzen 5 3600X (12T) / RTX 5060 (8GB) / 16GB RAM / WSL2

Change-log (v4 → v5):
  Fix 1  — LAB-based symptom detection replaces HSV (compute_lab_hard_mask /
            compute_lab_soft_confidence). L* normalised per-image; a*/b*
            thresholded for red-shift / yellowing. HSV functions retained as
            labelled legacy fallback.
  Fix 2  — Directional morphological opening (1×15 vertical kernel) applied
            inside LAB mask functions to suppress continuous midrib structure
            while keeping broken MSV/MLN streaks.
  Fix 3  — Conditional HEALTHY zeroing: symptom signal < 2 % of leaf area is
            treated as a false positive and zeroed.  Signal ≥ 2 % is preserved
            for early-symptom learning (per Luno's guidance).
  Fix 4  — HSV S floors tightened in config.py (interim until LAB is fully
            validated in production).
  Fix 6  — Reminder: verify train_student.py includes aggressive augmentation:
              A.RandomBrightnessContrast(p=0.5)
              A.HueSaturationValue(p=0.5)
              A.RandomGamma(p=0.3)

Change-log (v5 → v6):
  MSV-1  — Gabor now acts as a soft weight (raw response 0–1 multiplied into
            the LAB mask; product thresholded at 0.4) instead of a hard binary
            AND gate.  Strong LAB hits survive borderline Gabor response.
  MSV-2  — Directional morphological kernel changed from (1, 15) → (1, 7)
            for the MSV branch: shorter kernel preserves broken, shorter
            streaks on yellow maize that the 15-px opening was destroying.
  MLN-1  — Added dark-necrosis condition: (L_norm < 110) & (a ≥ 125).
            OR-ed with the existing yellowing gate before green exclusion so
            that low-L* necrotic patches are captured.
  MLN-2  — (5×5) elliptical closing applied after MLN mask and before
            directional opening to bridge fragmented necrotic patches that
            would otherwise fall below FACTORY_R3_MIN_AREA_PX.
  HLT-1  — HEALTHY branch uses its own stricter LAB thresholds (a ≥ 140,
            b ≥ 145) instead of the MSV thresholds to avoid flagging normal
            warm-yellow leaf coloration as symptom signal.
  HLT-2  — Replaced the single 2%-floor zero with a three-band rule:
              < 1 %  → zero completely (noise)
              1–4 %  → multiply by 0.3 (confidence dampening)
              ≥ 4 %  → preserve at full strength (genuine early signal)
  CROSS-1 — _normalize_L() skips stretching when image L* range > 180 to
            prevent over-amplification of lighting variation on uniformly-
            yellow images.
  CROSS-2 — LAB_GREEN_A_MAX raised from 124 → 121 to stop correctly
            excluding chlorotic-yellow pixels in the 121–130 a* range.
"""
import os
# CRITICAL: Prevent OpenCV/NumPy from hijacking all CPU cores and choking DataLoader workers
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import csv
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchvision import models

cv2.setNumThreads(2)  # Force OpenCV to respect thread limits

from image_utils import load_image_clahe, to_hsv
from config import (
    SEED, GLOBAL_MANIFEST, TIER1_MANIFEST, TIER1_MASKS_DIR, PSEUDO_DIR, REPORTS_DIR,
    TEACHER_CKPT_DIR, BOUNCER_CKPT_DIR, BOUNCER_DEPLOYED_VARIANT,
    TEACHER_DEPLOYED_VARIANT, TEACHER_IMG_SIZE, STUDENT_IMG_SIZE,
    FACTORY_MODES, FACTORY_SILHOUETTE_THRESHOLD, FACTORY_MIN_LEAF_COVERAGE,
    FACTORY_WEIGHT_BRACKETS, FACTORY_R3_MIN_AREA_PX, FACTORY_MORPH_KERNEL_SIZE, HSV_GREEN_EXCL,
    HSV_MSV_RANGES, HSV_MLN_RANGES, BOUNCER_THRESHOLD, VALID_EXTENSIONS, CLASSES,
    GABOR_KERNEL_SIZE, GABOR_SIGMA, GABOR_LAMBDA, GABOR_GAMMA, GABOR_PSI, GABOR_NORMS,
    GABOR_THETAS, GABOR_THRESHOLD, BOUNCER_IMG_SIZE
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
TEACHER_INFER_TF = A.Compose([
    A.LongestMaxSize(max_size=TEACHER_IMG_SIZE),
    A.PadIfNeeded(TEACHER_IMG_SIZE, TEACHER_IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

BOUNCER_INFER_TF = A.Compose([
    A.Resize(BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
def _find_ckpt(base_dir: Path, variant: str, prefix: str) -> Path:
    """Check final/ first, then fallback to variant-specific dir."""
    final_path = base_dir.parent / "final" / f"{prefix}_best.pth"
    if final_path.exists():
        return final_path
    var_path = base_dir / f"{prefix}_{variant}_best.pth"
    if var_path.exists():
        return var_path
    raise FileNotFoundError(f"Checkpoint not found for {prefix} (variant: {variant})")

def load_bouncer() -> tuple[nn.Module, float]:
    ckpt_path = _find_ckpt(BOUNCER_CKPT_DIR, BOUNCER_DEPLOYED_VARIANT, "bouncer")
    model = models.mobilenet_v3_large(weights=None)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 1)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model, BOUNCER_THRESHOLD

def load_teacher() -> nn.Module:
    ckpt_path = _find_ckpt(TEACHER_CKPT_DIR, TEACHER_DEPLOYED_VARIANT, "teacher")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    variant = ckpt.get("variant", TEACHER_DEPLOYED_VARIANT)
    
    if variant == "deeplabv3plus-eb2":
        model = smp.DeepLabV3Plus(encoder_name="efficientnet-b2", encoder_weights=None, in_channels=3, classes=1, activation=None)
    else:
        encoder_map = {"resnet50": "resnet50", "efficientnet-b2": "efficientnet-b2", "mit_b2": "mit_b2"}
        model = smp.Unet(encoder_name=encoder_map.get(variant, "efficientnet-b2"), encoder_weights=None, in_channels=3, classes=1, activation=None)
        
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET & COLLATE (Fixes Varying-Shape RuntimeError)
# ═══════════════════════════════════════════════════════════════════════════════
class FactoryDataset(Dataset):
    def __init__(self, df, tier1_fnames, test_fnames):
        # Exclude Tier 1 images that belong to the test split (Blueprint 10.1)
        self.df = df[~df["source_path"].apply(lambda p: Path(p).name in test_fnames)].reset_index(drop=True)
        self.tier1_fnames = tier1_fnames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["source_path"])
        category = row["category"]
        is_tier1 = img_path.name in self.tier1_fnames
        
        img_rgb = load_image_clahe(img_path)
        if img_rgb is None:
            return None
            
        orig_h, orig_w = img_rgb.shape[:2]
        
        b_tensor = BOUNCER_INFER_TF(image=img_rgb)["image"]
        t_tensor = TEACHER_INFER_TF(image=img_rgb)["image"]
        
        tier1_mask = None
        if is_tier1:
            npy_candidates = list(TIER1_MASKS_DIR.glob(f"*{img_path.stem}*softmask.npy"))
            if npy_candidates:
                tier1_mask = np.load(str(npy_candidates[0])).astype(np.float32)
                
        return {
            "img_path": str(img_path), "stem": img_path.stem, "category": category,
            "is_tier1": is_tier1, "orig_h": orig_h, "orig_w": orig_w,
            "img_rgb": img_rgb, "b_tensor": b_tensor, "t_tensor": t_tensor, "tier1_mask": tier1_mask
        }

def safe_collate(batch):
    """Custom collate to handle mixed tensors and variable-shape numpy arrays."""
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    collated = {}
    elem = batch[0]
    for key in elem:
        if key in ("img_rgb", "tier1_mask"):
            # Keep variable-shape arrays as Python lists
            collated[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch])
        elif isinstance(elem[key], (int, float, bool, str)):
            collated[key] = [d[key] for d in batch]
        else:
            collated[key] = [d[key] for d in batch]
    return collated

# ═══════════════════════════════════════════════════════════════════════════════
# CPU & I/O HEAVY FUNCTIONS (Run in ThreadPool)
# ═══════════════════════════════════════════════════════════════════════════════
def refine_silhouette(prob_map: np.ndarray) -> np.ndarray:
    binary = (prob_map >= FACTORY_SILHOUETTE_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FACTORY_MORPH_KERNEL_SIZE, FACTORY_MORPH_KERNEL_SIZE))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    return binary

def get_reliability_weight(leaf_coverage: float) -> float:
    if leaf_coverage < FACTORY_MIN_LEAF_COVERAGE:
        return None  # Explicitly gate on config constant rather than relying on bracket sentinel
    for lo, hi, weight in FACTORY_WEIGHT_BRACKETS:
        if lo <= leaf_coverage < hi: return weight
    return FACTORY_WEIGHT_BRACKETS[-1][2]

def _in_range(h, s, v, rng: dict) -> np.ndarray:
    return (h >= rng["h"][0]) & (h <= rng["h"][1]) & (s >= rng["s"][0]) & (s <= rng["s"][1]) & (v >= rng["v"][0]) & (v <= rng["v"][1])

def _green_exclusion(h, s, v) -> np.ndarray:
    ge = HSV_GREEN_EXCL
    return (h >= ge["h_min"]) & (h <= ge["h_max"]) & (s >= ge["s_min"]) & (v >= ge["v_min"]) & (v <= ge["v_max"])

def _apply_r3_area_filter(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= FACTORY_R3_MIN_AREA_PX:
            filtered[labels == lbl] = 1
    return filtered

def _compute_gabor_combined_mask(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = img_rgb.shape[:2]
    combined = np.zeros((h, w), dtype=np.float32)
    
    for theta in GABOR_THETAS:
        for norm_freq in GABOR_NORMS:
            lambd = GABOR_LAMBDA  # config constant — image-size-independent, reproducible
            # FIX: cv2.CV_32F prevents memory corruption on 1-channel grayscale
            kernel = cv2.getGaborKernel((GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE), GABOR_SIGMA, theta, lambd, GABOR_GAMMA, GABOR_PSI, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            filtered = np.abs(filtered)
            if filtered.max() > 0:
                filtered = 255 * (filtered / filtered.max())
            combined += filtered.astype(np.float32) / 255.0

    combined /= (len(GABOR_THETAS) * len(GABOR_NORMS))
    return (combined >= GABOR_THRESHOLD).astype(np.uint8)

# ─── LAB-based symptom detection (Fix 1) ────────────────────────────────────
# Replaces HSV for symptom isolation.  LAB separates luminance (L*) from
# colour (a*, b*) so flash, shadow, and midrib specular shine don't shift the
# chrominance channels — the root cause of HSV false positives on midribs.
#
# Thresholds (OpenCV scale: L* 0-255, a*/b* 0-255 centred at 128):
#   a* > 128  → red-shift  (MSV chlorotic streaks, necrotic tissue)
#   b* > 128  → yellow-shift (both MSV and MLN yellowing)
#   a* < 128  → green-shift  (healthy — used for exclusion)
#
# L* is normalised per-image before thresholding to remove lighting variance.

LAB_MSV_A_MIN   = 133   # slight red-shift: chlorotic/necrotic streaks
LAB_MSV_B_MIN   = 135   # yellowing component
LAB_MLN_A_MIN   = 130   # MLN lesions trend slightly less red than MSV
LAB_MLN_B_MIN   = 138   # MLN necrosis more yellow-brown
LAB_GREEN_A_MAX = 121   # pixels greener than this are healthy tissue (raised 124→121: tighter exclusion preserves chlorotic-yellow pixels in a* 121–130 range)

def _normalize_L(lab: np.ndarray) -> np.ndarray:
    """Stretch L* channel to [0, 255] per-image to remove lighting/flash bias.

    Guard: if the image already spans more than 180 L* units it has good
    contrast on its own.  Stretching further amplifies small lighting
    variations into large L* swings that shift which pixels cross the a*/b*
    thresholds — a known source of false positives on uniformly-yellow maize
    images.  Skip normalization in that case and return raw L* values.
    """
    L = lab[:, :, 0].astype(np.float32)
    lo, hi = L.min(), L.max()
    if hi - lo < 1e-3:
        return lab
    # NEW: skip stretching when image already has wide L* range
    if hi - lo > 180:
        return lab
    lab = lab.copy()
    lab[:, :, 0] = np.clip((L - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return lab

def compute_lab_hard_mask(img_rgb: np.ndarray, silhouette: np.ndarray, category: str) -> np.ndarray:
    """LAB hard binary symptom mask with directional morphological opening (Fix 1 + Fix 2).

    Changes vs v5:
      MSV  — Gabor acts as a soft weight (multiply LAB mask by raw Gabor
             response 0–1) rather than a hard AND gate; threshold result at
             0.4 so strong LAB hits survive borderline Gabor.
           — Directional opening kernel changed from (1, 15) → (1, 7): shorter
             kernel preserves the broken, shorter streaks typical of MSV on
             yellow maize.
      MLN  — Added dark-necrosis condition: L_norm < 110 AND a* ≥ 125 captures
             necrotic patches with low L* that the yellowing condition misses.
           — Small (5×5) elliptical closing applied before the directional
             opening to bridge fragmented necrotic patches that would otherwise
             fall below FACTORY_R3_MIN_AREA_PX and be dropped.
      HEALTHY — Uses stricter own thresholds (a ≥ 140, b ≥ 145) instead of
             the MSV thresholds so normal warm-yellow coloration does not
             trigger a non-zero mask before the area check.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab = _normalize_L(lab)
    L_norm = lab[:, :, 0].astype(np.int32)  # 0-255 after normalization
    a = lab[:, :, 1].astype(np.int32)       # 0-255, neutral=128
    b = lab[:, :, 2].astype(np.int32)       # 0-255, neutral=128

    green_excl = (a < LAB_GREEN_A_MAX)

    if category == "MSV":
        # LAB colour gate
        lab_mask = ((a >= LAB_MSV_A_MIN) | (b >= LAB_MSV_B_MIN)).astype(np.uint8)
        lab_mask &= (~green_excl).astype(np.uint8)
        lab_mask &= silhouette

        # Gabor as a soft weight rather than a binary AND gate.
        # _compute_gabor_combined_mask() returns a uint8 binary mask; we need
        # the raw continuous response (0.0–1.0) that underlies it.  We
        # recompute the raw combined response here and threshold the product.
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h_, w_ = img_rgb.shape[:2]
        gabor_raw = np.zeros((h_, w_), dtype=np.float32)
        for theta in GABOR_THETAS:
            for norm_freq in GABOR_NORMS:
                kernel = cv2.getGaborKernel(
                    (GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE),
                    GABOR_SIGMA, theta, GABOR_LAMBDA, GABOR_GAMMA, GABOR_PSI,
                    ktype=cv2.CV_32F,
                )
                filtered = np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel))
                if filtered.max() > 0:
                    filtered = filtered / filtered.max()
                gabor_raw += filtered
        gabor_raw /= len(GABOR_THETAS) * len(GABOR_NORMS)  # now in [0, 1]

        # Multiply LAB mask by raw Gabor response; threshold product at 0.4.
        # Strong LAB hits survive even when Gabor is borderline.
        weighted = lab_mask.astype(np.float32) * gabor_raw
        symptom = (weighted >= 0.4).astype(np.uint8)

        # Directional opening: (1, 7) — shorter than the legacy (1, 15) so
        # broken, shorter MSV streaks on yellow maize are not destroyed.
        if symptom.any():
            kernel_dir = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
            symptom = cv2.morphologyEx(symptom, cv2.MORPH_OPEN, kernel_dir)

    elif category == "MLN":
        # Primary yellowing / red-shift condition (unchanged)
        yellowing = (a >= LAB_MLN_A_MIN) | (b >= LAB_MLN_B_MIN)

        # NEW: dark necrosis condition — low L* (dark patches) with slight
        # red-shift capture necrotic tissue that has low b* and would be missed
        # by the yellowing gate.
        dark_necrosis = (L_norm < 110) & (a >= 125)

        symptom = (yellowing | dark_necrosis).astype(np.uint8)
        symptom &= (~green_excl).astype(np.uint8)
        symptom &= silhouette

        # NEW: small closing to bridge fragmented necrotic patches before the
        # directional opening removes them as isolated specks.
        if symptom.any():
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            symptom = cv2.morphologyEx(symptom, cv2.MORPH_CLOSE, kernel_close)

        # Directional opening (same (1, 15) as before for MLN — streaks are
        # longer and less broken than on yellow maize).
        if symptom.any():
            kernel_dir = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            symptom = cv2.morphologyEx(symptom, cv2.MORPH_OPEN, kernel_dir)

    else:
        # HEALTHY: use stricter own thresholds (a ≥ 140, b ≥ 145) so normal
        # warm-yellow leaf coloration does not trigger a non-zero mask before
        # the area check in process_single_image_cpu().
        # The legacy code reused LAB_MSV_A_MIN / LAB_MSV_B_MIN here, which
        # caused almost every yellow-maize healthy leaf to produce a non-zero
        # mask, defeating the 2%-floor noise gate.
        symptom = ((a >= 140) | (b >= 145)).astype(np.uint8)
        symptom &= (~green_excl).astype(np.uint8)
        symptom &= silhouette

        if symptom.any():
            kernel_dir = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            symptom = cv2.morphologyEx(symptom, cv2.MORPH_OPEN, kernel_dir)

    return (symptom * 255).astype(np.uint8)

def compute_lab_soft_confidence(img_rgb: np.ndarray, silhouette: np.ndarray, category: str) -> np.ndarray:
    """LAB soft confidence map (analogous to compute_hsv_soft_confidence, for mode_d)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab = _normalize_L(lab)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    green_excl = (a < float(LAB_GREEN_A_MAX))

    if category == "MSV":
        a_min, b_min = float(LAB_MSV_A_MIN), float(LAB_MSV_B_MIN)
    else:
        a_min, b_min = float(LAB_MLN_A_MIN), float(LAB_MLN_B_MIN)

    # Soft confidence: linear ramp from threshold to channel ceiling (255)
    a_conf = np.clip((a - a_min) / max(255.0 - a_min, 1.0), 0.0, 1.0)
    b_conf = np.clip((b - b_min) / max(255.0 - b_min, 1.0), 0.0, 1.0)
    conf_map = np.maximum(a_conf, b_conf)
    conf_map *= (~green_excl).astype(np.float32)
    conf_map *= silhouette.astype(np.float32)

    if category == "MSV":
        conf_map *= _compute_gabor_combined_mask(img_rgb).astype(np.float32)

    # Fix 2 — directional opening on binarised version, then reapply to conf
    if conf_map.max() > 0:
        binary_hint = (conf_map > 0).astype(np.uint8)
        kernel_dir = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        binary_hint = cv2.morphologyEx(binary_hint, cv2.MORPH_OPEN, kernel_dir)
        conf_map *= binary_hint.astype(np.float32)

    return np.clip(conf_map, 0.0, 1.0)

# ─── Legacy HSV detection (kept as fallback / comparison baseline) ────────────
def compute_hsv_hard_mask(img_rgb, silhouette, category):
    img_hsv = to_hsv(img_rgb)
    h, s, v = img_hsv[:,:,0].astype(np.int32), img_hsv[:,:,1].astype(np.int32), img_hsv[:,:,2].astype(np.int32)
    green_excl = _green_exclusion(h, s, v)
    ranges = HSV_MSV_RANGES if category == "MSV" else HSV_MLN_RANGES
    
    symptom = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for i, rng in enumerate(ranges):
        hit = _in_range(h, s, v, rng)
        if category == "MSV" and i == 2: 
            hit = _apply_r3_area_filter(hit.astype(np.uint8)).astype(bool)
        symptom |= hit.astype(np.uint8)
        
    symptom &= (~green_excl).astype(np.uint8)
    symptom &= silhouette
    if category == "MSV": symptom &= _compute_gabor_combined_mask(img_rgb)
    return (symptom * 255).astype(np.uint8)

def compute_hsv_soft_confidence(img_rgb, silhouette, category):
    img_hsv = to_hsv(img_rgb)
    h, s, v = img_hsv[:,:,0].astype(np.float32), img_hsv[:,:,1].astype(np.float32), img_hsv[:,:,2].astype(np.float32)
    green_excl = _green_exclusion(h.astype(np.int32), s.astype(np.int32), v.astype(np.int32))
    ranges = HSV_MSV_RANGES if category == "MSV" else HSV_MLN_RANGES
    
    conf_accum = np.zeros(img_rgb.shape[:2], dtype=np.float32)
    hit_count  = np.zeros(img_rgb.shape[:2], dtype=np.float32)  # FIX: track per-pixel hit count
    for i, rng in enumerate(ranges):
        h_lo, h_hi = rng["h"]; s_lo, s_hi = rng["s"]; v_lo, v_hi = rng["v"]
        hit = _in_range(h.astype(np.int32), s.astype(np.int32), v.astype(np.int32), rng)
        if category == "MSV" and i == 2: hit = _apply_r3_area_filter(hit.astype(np.uint8)).astype(bool)
        
        h_center, s_center, v_center = (h_lo+h_hi)/2.0, (s_lo+s_hi)/2.0, (v_lo+v_hi)/2.0
        h_half, s_half, v_half = max((h_hi-h_lo)/2.0, 1.0), max((s_hi-s_lo)/2.0, 1.0), max((v_hi-v_lo)/2.0, 1.0)
        
        d = np.clip((1.0 - np.abs(h-h_center)/h_half + 1.0 - np.abs(s-s_center)/s_half + 1.0 - np.abs(v-v_center)/v_half)/3.0, 0.0, 1.0)
        hit_f = hit.astype(np.float32)
        conf_accum += hit_f * d
        hit_count  += hit_f  # FIX: count how many ranges fired per pixel
        
    # FIX: normalize by actual hit count (max 1 where any range fired) instead of total
    # len(ranges), which was capping MLN pixels at 0.20 (1/5) and preventing the 0.3 threshold
    safe_count = np.maximum(hit_count, 1.0)
    conf_map = (conf_accum / safe_count) * (~green_excl).astype(np.float32) * silhouette.astype(np.float32)
    if category == "MSV": conf_map *= _compute_gabor_combined_mask(img_rgb).astype(np.float32)
    return np.clip(conf_map, 0.0, 1.0)

def process_single_image_cpu(args):
    """Runs entirely on CPU threads to unblock the GPU pipeline"""
    img_path, stem, category, is_tier1, orig_h, orig_w, img_rgb, soft_prob, mode_dirs = args
    
    # mode_a: Otsu hard threshold on grayscale
    # mode_b: morphology-refined binary from teacher soft_prob  (close + open)
    # mode_c: raw soft_prob thresholded without morphology cleanup
    # mode_d: same sil as mode_b, but soft confidence symptom map
    binary_sil    = refine_silhouette(soft_prob)                          # morph-refined  (mode_b, mode_d)

    # Convex hull fill — recovers leaf area lost to necrotic patches that
    # break silhouette connectivity. Maize leaves are convex along their
    # long axis, so the hull is a valid geometric constraint.
    _contours, _ = cv2.findContours(binary_sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if _contours:
        _hull = cv2.convexHull(max(_contours, key=cv2.contourArea))
        binary_sil = cv2.drawContours(
            np.zeros_like(binary_sil), [_hull], -1, 1, -1)

    raw_thresh_sil = (soft_prob >= FACTORY_SILHOUETTE_THRESHOLD).astype(np.uint8)  # no morph (mode_c)
    otsu_sil      = (cv2.threshold(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0).astype(np.uint8)
    
    # Coverage and weight are derived from the refined silhouette (mode_b baseline)
    coverage = float(binary_sil.sum()) / (orig_h * orig_w)
    weight = get_reliability_weight(coverage)
    
    result = {"filename": stem, "category": category, "coverage": round(coverage, 4), "weight": weight if weight is not None else -1}
    modes_data = {}
    
    SIL_MAP = {
        "mode_a": otsu_sil,
        "mode_b": binary_sil,
        "mode_c": raw_thresh_sil,   # FIX: was binary_sil — mode_c must differ from mode_b
        "mode_d": binary_sil,
    }
    
    for mode in FACTORY_MODES:
        mode = mode.strip()  # Safeguard against OCR spaces in config
        sil = SIL_MAP[mode]

        # Fix 1: use LAB-based detection for all modes (HSV kept as fallback).
        # mode_d → soft confidence map; all others → hard binary mask.
        symptom = compute_lab_soft_confidence(img_rgb, sil, category) if mode == "mode_d" else compute_lab_hard_mask(img_rgb, sil, category)

        # Fix 3 (conditional): zero symptom signal for HEALTHY images.
        # Rationale: confirmed-HEALTHY labels have no annotated disease by
        # definition; any LAB/HSV signal is a false positive that corrupts
        # regression targets and inflates severity.
        # EXCEPTION (per Luno): early-stage / sub-clinical cases may carry
        # faint real signal.  We therefore zero only when the detected
        # symptom area is below a noise floor (< 2 % of leaf area), treating
        # those as artefacts.  Genuine very-early symptoms (≥ 2 % leaf area)
        # are preserved so the student can learn from them.
        if category == "HEALTHY":
            leaf_px_for_check = float(sil.sum())
            if leaf_px_for_check > 0:
                sym_area_frac = float((symptom > 0).sum()) / leaf_px_for_check
                if sym_area_frac < 0.01:
                    # Below 1 % → almost certainly noise; zero completely.
                    if mode == "mode_d":
                        symptom = np.zeros_like(symptom, dtype=np.float32)
                    else:
                        symptom = np.zeros_like(symptom, dtype=np.uint8)
                elif sym_area_frac < 0.04:
                    # 1 %–4 % → ambiguous early-symptom signal.  Apply
                    # confidence dampening (×0.3) rather than zeroing so the
                    # student sees the spatial pattern but at down-weighted
                    # contribution.  The weight file already down-weights this
                    # image; the 0.3 factor adds an additional signal-level
                    # guard without destroying spatial information.
                    if mode == "mode_d":
                        symptom = (symptom.astype(np.float32) * 0.3)
                    else:
                        symptom = np.clip(
                            (symptom.astype(np.float32) * 0.3).astype(np.uint8),
                            0, 255,
                        )
                # ≥ 4 % → preserved at full strength (genuine early-stage signal)
            else:
                symptom = np.zeros_like(symptom)

        sym_binary = (symptom >= 0.3).astype(np.uint8) if mode == "mode_d" else (symptom > 0).astype(np.uint8)

        leaf_px = float(sil.sum())
        sev = (float(sym_binary.sum()) / leaf_px) * 100.0 if leaf_px >= 1 and weight is not None else -1.0
        
        # Silhouette saved to disk: mode_c and mode_d store the continuous soft_prob
        # so the student can learn from the full probability map, not just the binary mask
        disk_sil = soft_prob if mode in ("mode_c", "mode_d") else sil.astype(np.float32)
        modes_data[mode] = {"silhouette": disk_sil, "symptom": symptom, "severity": sev}

    # Async Disk I/O
    for mode, data in modes_data.items():
        mode_dir = mode_dirs[mode]
        sil_out = cv2.resize(data["silhouette"].astype(np.float32), (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        np.save(str(mode_dir / f"{stem}_silhouette.npy"), sil_out)
        
        if mode == "mode_d":
            sym_out = cv2.resize(data["symptom"].astype(np.float32), (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            np.save(str(mode_dir / f"{stem}_symptom.npy"), sym_out)
        else:
            sym_out = cv2.resize(data["symptom"], (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(mode_dir / f"{stem}_symptom.png"), sym_out)
            
        (mode_dir / f"{stem}_sev.txt").write_text(str(round(data["severity"], 4)))
        (mode_dir / f"{stem}_weight.txt").write_text(str(result["weight"]))
        
    result["status"] = "processed"
    result.update({f"{m}_sev": round(modes_data[m]["severity"], 4) for m in modes_data})
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATORS (Blueprint 10.7)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_filter_breakdown(report_rows):
    from collections import Counter
    status_counts = Counter(r.get("status", "unknown") for r in report_rows)
    total = sum(status_counts.values())
    
    breakdown_path = REPORTS_DIR / "factory_filter_breakdown.csv"
    with open(breakdown_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["status", "count", "pct"])
        for status, cnt in status_counts.most_common():
            w.writerow([status, cnt, round(100*cnt/max(total,1), 2)])
    print(f"  Filter breakdown → {breakdown_path}")

def generate_factory_summary(report_rows):
    proc = [r for r in report_rows if r.get("status") == "processed"]
    if not proc: return
    
    df = pd.DataFrame(proc)
    summary_rows = []
    
    for mode in FACTORY_MODES:
        mode = mode.strip()
        sev_col = f"{mode}_sev"
        if sev_col not in df.columns: continue
        
        for cls in df["category"].unique():
            cls_df = df[df["category"] == cls]
            valid_sev = cls_df[sev_col].replace(-1.0, np.nan).dropna()
            
            summary_rows.append({
                "mode": mode, "class": cls,
                "n_total": len(cls_df), "n_processed": len(cls_df[cls_df["status"]=="processed"]),
                "pct_processed": round(100*len(cls_df[cls_df["status"]=="processed"])/max(len(cls_df),1), 1),
                "mean_severity": round(valid_sev.mean(), 2) if len(valid_sev) else "N/A",
                "std_severity": round(valid_sev.std(), 2) if len(valid_sev) else "N/A",
                "median_severity": round(valid_sev.median(), 2) if len(valid_sev) else "N/A",
                "pct_symptomatic": round(100*(valid_sev > 0).mean(), 1) if len(valid_sev) else 0.0,
                "n_excluded": len(cls_df[cls_df["weight"] == -1]),
                "pct_excluded": round(100*len(cls_df[cls_df["weight"] == -1])/max(len(cls_df),1), 1)
            })
            
    if summary_rows:
        summary_path = REPORTS_DIR / "factory_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys(), extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"  Factory summary → {summary_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 72, "\n  Yellow MAIze | Phase 4: Factory Pseudo-Label Generation\n" + "=" * 72)
    
    if not GLOBAL_MANIFEST.exists() or not TIER1_MANIFEST.exists():
        print("[FATAL] Missing manifest files. Run partition_dataset.py and sample_15000.py first.")
        return
        
    global_df = pd.read_csv(GLOBAL_MANIFEST)
    trainval = global_df[global_df["split"] != "test"].reset_index(drop=True)
    test_fnames = set(global_df[global_df["split"] == "test"]["filename"].tolist())
    tier1_fnames = set(Path(p).name for p in pd.read_csv(TIER1_MANIFEST)["source_path"].tolist())
    
    mode_dirs = {m.strip(): PSEUDO_DIR / m.strip() for m in FACTORY_MODES}
    for d in mode_dirs.values(): d.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n  Loading models to RTX 5060...")
    bouncer_model, bouncer_thresh = load_bouncer()
    teacher_model = load_teacher()
    
    dataset = FactoryDataset(trainval, tier1_fnames, test_fnames)
    loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True, collate_fn=safe_collate, prefetch_factor=2)
    
    report_rows = []
    t_start = time.time()
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        for i, batch in enumerate(loader):
            if batch is None: continue
            
            # 1. Batched Bouncer Inference (GPU)
            b_tensors = batch["b_tensor"].to(DEVICE, non_blocking=True)
            with torch.no_grad():
                b_probs = torch.sigmoid(bouncer_model(b_tensors)).view(-1)
                b_passed = b_probs >= bouncer_thresh
            
            passed_indices = torch.where(b_passed)[0].cpu().numpy()
            passed_set = set(passed_indices.tolist())  # FIX: O(1) lookup instead of O(n) numpy scan
            
            for idx in range(len(b_passed)):
                if idx not in passed_set:
                    report_rows.append({"filename": Path(batch["img_path"][idx]).stem, "status": "filtered_bouncer", "category": batch["category"][idx]})
            
            if len(passed_indices) == 0: continue
                
            # 2. Batched Teacher Inference (GPU)
            t_tensors = torch.stack([batch["t_tensor"][idx] for idx in passed_indices]).to(DEVICE, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher_model(t_tensors)
                t_probs_batch = torch.sigmoid(t_logits).squeeze(1).cpu().numpy()
                
            # 3. Dispatch to CPU ThreadPool
            futures = []
            for i_rel, idx in enumerate(passed_indices):
                orig_h = int(batch["orig_h"][idx])
                orig_w = int(batch["orig_w"][idx])
                is_tier1 = bool(batch["is_tier1"][idx])
                prob = t_probs_batch[i_rel]
                
                if is_tier1 and batch["tier1_mask"][idx] is not None:
                    soft_prob = cv2.resize(batch["tier1_mask"][idx], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                else:
                    soft_prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    
                args = (batch["img_path"][idx], batch["stem"][idx], batch["category"][idx], 
                        is_tier1, orig_h, orig_w, batch["img_rgb"][idx], soft_prob, mode_dirs)
                futures.append(executor.submit(process_single_image_cpu, args))
                
            for future in futures:
                report_rows.append(future.result())
                
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-3)
                eta = (len(loader) - i - 1) / rate
                print(f"  [{i+1:>4}/{len(loader)} Batches] | ETA: {eta/60:.1f} min | Processed: {len(report_rows)}")

    print("\n  Writing reports...")
    # Write one unified per-image report (all mode severities are already columns in report_rows)
    if report_rows:
        report_path = REPORTS_DIR / "phase4_report.csv"
        keys = list(report_rows[0].keys())
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"  Per-image report → {report_path}")
                
    generate_filter_breakdown(report_rows)
    generate_factory_summary(report_rows)
    
    print(f"\n  Total duration: {round(time.time() - t_start, 1)}s")
    print(f"  NEXT STEP: python train_student.py --stage 1")
    print("=" * 72)

if __name__ == "__main__":
    main()