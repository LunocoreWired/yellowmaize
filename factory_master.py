"""
factory_master.py — Phase 4: Pseudo-Label Generation (Optimized & Fixed)
Hardware Target: Ryzen 5 3600X (12T) / RTX 5060 (8GB) / 16GB RAM / WSL2

Change-log (v8 → v9):
  REPL-1 — Symptom source replaced: LAB/HSV color thresholding (compute_lab_hard_mask
            / compute_lab_soft_confidence) was structurally unable to separate
            early chlorosis from healthy yellow-maize tissue, or tip-burn from
            MLN necrosis, after eight rounds of threshold tuning (v1->v8) — each
            fix traded one failure mode for another rather than converging.
            A Symptom Teacher (EfficientNet-B2 UNet, train_symptom_model.py)
            trained on ~400-500 human-verified polygon masks now supplies the
            symptom probability map for ALL modes when SYMPTOM_TEACHER_DEPLOYED
            is True (config.py) and checkpoints exist. LAB functions are kept
            in this file UNCHANGED as automatic fallback (checkpoints missing)
            and as an explicit comparison source (FACTORY_SYMPTOM_COMPARE_LAB).
  REPL-2 — Symptom Teacher input is RGB + a HealthyAE reconstruction-error map
            (4 channels) — the AE is trained on HEALTHY images only and supplies
            a lighting-invariant anomaly prior the LAB pipeline never had.
            Ground truth for the Symptom Teacher is always the human mask, so
            it cannot inherit LAB's systematic biases the way a model trained
            on LAB-derived pseudo-labels would.
  REPL-3 — predict_symptom_mask() output is restricted to the leaf silhouette
            (sil) before use, matching the spatial constraint LAB masks already
            had implicitly (LAB functions take sil as an argument).
  FIX-1  — compute_lab_soft_confidence() MLN branch: added Otsu-adaptive
            thresholding (matching compute_lab_hard_mask()'s MLN branch).
            Fixed threshold divergence between mode_d and modes b/c under
            variable lighting — corrupted the cross-mode ablation study.
  FIX-2  — compute_lab_soft_confidence() MLN branch: added dark necrosis soft
            confidence channel `(L* < 110) & (a* >= 125)` to match the hard
            mask's `dark_necrosis` gate. Dark necrotic patches now score
            non-zero confidence in mode_d, consistent with modes b/c.

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

Change-log (v6 → v7):
  BUG-1  — compute_lab_soft_confidence: HEALTHY branch added with stricter
            thresholds (a_min=140, b_min=145) matching compute_lab_hard_mask.
            Previously fell into else→MLN branch, producing over-broad soft
            maps for HEALTHY images before the three-band noise gate.
  BUG-2  — compute_lab_soft_confidence: MSV directional kernel corrected
            from (1,15) → (1,7) to match compute_lab_hard_mask. Short broken
            streaks were being destroyed in mode_d but preserved in mode_b.
  NEW-1  — Otsu-guided adaptive thresholding added to compute_lab_hard_mask.
            Per-image optimal threshold computed on masked a*/b* pixels then
            clamped to fixed floor (LAB_MSV_A_MIN / LAB_MLN_A_MIN) so Otsu
            cannot drift into healthy-tissue range on heavily diseased leaves.
  NEW-2  — CLAHE applied to L* channel within leaf mask before LAB symptom
            detection via _clahe_L(). Locally adaptive contrast on the leaf
            region only — more targeted than full-image CLAHE in load_image_clahe.
  NEW-3  — CIMMYT severity grading added. _sev_to_cimmyt_grade() maps
            continuous severity % to published CIMMYT 1–9 (MSV) and 1–5 (MLN)
            scales. Written as _grade.txt alongside _sev.txt per image.

Change-log (v7 → v8):
  VEIN-1 — Frangi sigma range extended to (0.5, 1.5, 3.0, 6.0, 10.0) via
            FRANGI_SIGMAS in config. Sigma=6.0 and 10.0 explicitly target the
            midrib (primary false positive source) which was under-represented
            at previous scale range. Vein suppression weights now computed at
            all 5 scales simultaneously before the leaf-normalised maximum.
  VEIN-2 — Frangi now also runs on the normalised a* channel (black_ridges=True)
            to capture chromatic vein ridges (veins are greener = lower a* on
            yellow maize). Pixel-wise maximum of L*-Frangi and a*-Frangi taken
            before soft suppression formula. Catches veins invisible in L*.
  MORPH-1 — Angled morphological opening bank added for MSV branch. Six
            orientations (0°, 30°, 60°, 90°, 120°, 150°) via MORPH_OPEN_ANGLES.
            Union of all 6 responses replaces the fixed (1,N) vertical kernel.
            MLN branch unchanged (blob morphology, not streak).
  GABOR-1 — Dual-channel Gabor: raw response now computed on a* channel
            (normalized [0,255]) in addition to grayscale. Weighted sum
            (1-GABOR_A_CHANNEL_WEIGHT)*gray + GABOR_A_CHANNEL_WEIGHT*a_channel
            before threshold. Chromatic streak texture more discriminative than
            luminance under overcast conditions.
  OTSU-1  — 2D joint Otsu on (a*, b*) histogram replaces independent 1D Otsu
            for MSV. _otsu_2d() uses vectorized cumsum on OTSU_2D_BIN_COUNT×
            OTSU_2D_BIN_COUNT histogram to find optimal (ta, tb) threshold
            surface. Falls back to 1D Otsu when leaf pixel count < 200.
  CLAHE-1 — CLAHE applied to a* channel within leaf mask (_clahe_a()) with
            CLAHE_A_CLIP_LIMIT=1.0 (lower than L* clip=2.0). Applied
            immediately after _clahe_L(). Enhances local chromatic contrast
            of early MSV streaks in low-contrast captures.
  MLN-3   — Convex-hull margin erosion mask (_margin_erosion_mask()) applied
            to MLN branch only. Erodes leaf perimeter by
            MARGIN_EROSION_FRAC=8% of sqrt(leaf_area) pixels. Suppresses
            tip burn false positives (physiologically distinct from MLN).
  TRAIN-1 — STUDENT_CKPT_W_MSV_F1 (0.35) replaced by three severity-stratified
            weights in config: EARLY (0.20), MID (0.10), SEVERE (0.05).
            Total budget unchanged. train_student.py must be updated separately
            to compute and log grade-stratified MSV F1 using CIMMYT brackets.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision import models

cv2.setNumThreads(2)  # Force OpenCV to respect thread limits

from image_utils import load_image_clahe, to_hsv
from skimage.filters import frangi

from config import (
    BOUNCER_CKPT_DIR,
    BOUNCER_DEPLOYED_VARIANT,
    BOUNCER_IMG_SIZE,
    BOUNCER_THRESHOLD,
    CIMMYT_MLN_BRACKETS,
    CIMMYT_MSV_BRACKETS,
    CLAHE_A_CLIP_LIMIT,
    CLASSES,
    FACTORY_MIN_LEAF_COVERAGE,
    FACTORY_MODES,
    FACTORY_MORPH_KERNEL_SIZE,
    FACTORY_R3_MIN_AREA_PX,
    FACTORY_SILHOUETTE_THRESHOLD,
    FACTORY_WEIGHT_BRACKETS,
    FRANGI_SIGMAS,
    GABOR_A_CHANNEL_WEIGHT,
    GABOR_GAMMA,
    GABOR_KERNEL_SIZE,
    GABOR_LAMBDA,
    GABOR_NORMS,
    GABOR_PSI,
    GABOR_SIGMA,
    GABOR_THETAS,
    GABOR_THRESHOLD,
    GLOBAL_MANIFEST,
    HSV_GREEN_EXCL,
    HSV_MLN_RANGES,
    HSV_MSV_RANGES,
    MARGIN_EROSION_FRAC,
    MORPH_OPEN_ANGLES,
    OTSU_2D_BIN_COUNT,
    PSEUDO_DIR,
    REPORTS_DIR,
    SEED,
    STUDENT_IMG_SIZE,
    SYMPTOM_TEACHER_DEPLOYED,
    FACTORY_SYMPTOM_COMPARE_LAB,
    TEACHER_CKPT_DIR,
    TEACHER_DEPLOYED_VARIANT,
    TEACHER_IMG_SIZE,
    TIER1_MANIFEST,
    TIER1_MASKS_DIR,
    VALID_EXTENSIONS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level singletons for the Symptom Teacher (Phase 3b). Populated by
# load_symptom_models() once in main() before the ThreadPoolExecutor starts;
# read (never written) by process_single_image_cpu() across all worker
# threads. None/None means "use the legacy LAB pipeline" — checked once per
# call rather than per-pixel so the fallback costs nothing when disabled.
_SYMPTOM_MODEL = None
_AE_MODEL      = None

# Transforms
TEACHER_INFER_TF = A.Compose(
    [
        A.LongestMaxSize(max_size=TEACHER_IMG_SIZE),
        A.PadIfNeeded(
            TEACHER_IMG_SIZE, TEACHER_IMG_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

BOUNCER_INFER_TF = A.Compose(
    [
        A.Resize(BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


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
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b2",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
    else:
        encoder_map = {
            "resnet50": "resnet50",
            "efficientnet-b2": "efficientnet-b2",
            "mit_b2": "mit_b2",
        }
        model = smp.Unet(
            encoder_name=encoder_map.get(variant, "efficientnet-b2"),
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model


def load_symptom_models() -> tuple[object | None, object | None]:
    """
    Load the Symptom Teacher + HealthyAE (Phase 3b, train_symptom_model.py)
    if SYMPTOM_TEACHER_DEPLOYED is True and checkpoints exist.

    Returns (symptom_model, ae_model), both None if unavailable — callers
    must fall back to the legacy LAB pipeline in that case. Import is local
    (not at module top) so factory_master.py and train_symptom_model.py can
    import from each other without a circular-import error: the only thing
    needed here is the loader + inference helper, both pure functions of
    the checkpoint file.

    NOTE: predict_symptom_mask() runs PyTorch GPU inference and is called
    per-image from the CPU ThreadPoolExecutor workers in the main loop
    below — this is correctness-safe (CUDA calls across Python threads
    serialize automatically) but is NOT batched the way Bouncer/Teacher
    inference is, so symptom inference is the per-image latency floor when
    this is enabled. Acceptable for 150k images on the strength of the
    accuracy gain over LAB; revisit with batched inference if throughput
    becomes the bottleneck.
    """
    if not SYMPTOM_TEACHER_DEPLOYED:
        return None, None
    try:
        from train_symptom_model import load_symptom_teacher, load_healthy_ae
    except ImportError as e:
        print(f"  [WARN] Could not import train_symptom_model.py: {e}")
        print("         Falling back to legacy LAB symptom detection.")
        return None, None

    ae_model = load_healthy_ae()
    symptom_model = load_symptom_teacher()

    if symptom_model is None or ae_model is None:
        print("  [INFO] Symptom Teacher / HealthyAE checkpoint(s) not found.")
        print("         Run train_symptom_model.py first, or set")
        print("         SYMPTOM_TEACHER_DEPLOYED = False in config.py to use")
        print("         the legacy LAB pipeline intentionally.")
        return None, None

    print("  Symptom Teacher loaded — replacing LAB/HSV symptom detection.")
    return symptom_model, ae_model


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET & COLLATE (Fixes Varying-Shape RuntimeError)
# ═══════════════════════════════════════════════════════════════════════════════
class FactoryDataset(Dataset):
    def __init__(self, df, tier1_fnames, test_fnames):
        # Exclude Tier 1 images that belong to the test split (Blueprint 10.1)
        self.df = df[
            ~df["source_path"].apply(lambda p: Path(p).name in test_fnames)
        ].reset_index(drop=True)
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
            npy_candidates = list(
                TIER1_MASKS_DIR.glob(f"*{img_path.stem}*softmask.npy")
            )
            if npy_candidates:
                tier1_mask = np.load(str(npy_candidates[0])).astype(np.float32)

        return {
            "img_path": str(img_path),
            "stem": img_path.stem,
            "category": category,
            "is_tier1": is_tier1,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "img_rgb": img_rgb,
            "b_tensor": b_tensor,
            "t_tensor": t_tensor,
            "tier1_mask": tier1_mask,
        }


def factory_collate(batch):
    """Custom collate to handle mixed tensors and variable-shape numpy arrays.
    Named factory_collate (not safe_collate) to avoid shadowing scripts/safe_collate.py.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    collated = {}
    elem = batch[0]
    for key in elem:
        if key in ("img_rgb", "tier1_mask"):
            # Keep variable-shape arrays as Python lists (variable image dimensions)
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
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (FACTORY_MORPH_KERNEL_SIZE, FACTORY_MORPH_KERNEL_SIZE)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def get_reliability_weight(leaf_coverage: float) -> float:
    if leaf_coverage < FACTORY_MIN_LEAF_COVERAGE:
        return None  # Explicitly gate on config constant rather than relying on bracket sentinel
    for lo, hi, weight in FACTORY_WEIGHT_BRACKETS:
        if lo <= leaf_coverage < hi:
            return weight
    return FACTORY_WEIGHT_BRACKETS[-1][2]


def _in_range(h, s, v, rng: dict) -> np.ndarray:
    return (
        (h >= rng["h"][0])
        & (h <= rng["h"][1])
        & (s >= rng["s"][0])
        & (s <= rng["s"][1])
        & (v >= rng["v"][0])
        & (v <= rng["v"][1])
    )


def _green_exclusion(h, s, v) -> np.ndarray:
    ge = HSV_GREEN_EXCL
    return (
        (h >= ge["h_min"])
        & (h <= ge["h_max"])
        & (s >= ge["s_min"])
        & (v >= ge["v_min"])
        & (v <= ge["v_max"])
    )


def _apply_r3_area_filter(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(mask)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= FACTORY_R3_MIN_AREA_PX:
            filtered[labels == lbl] = 1
    return filtered


def _compute_gabor_raw_response(channel_img: np.ndarray) -> np.ndarray:
    """Raw continuous Gabor response (0-1) for a single-channel uint8 image.

    Runs the full GABOR_THETAS × GABOR_NORMS kernel bank and averages the
    normalized magnitude response. Factored out of _compute_gabor_combined_mask
    so the same kernel loop can be reused on grayscale and on the a* channel
    (Priority 3 — chromatic Gabor texture for MSV).
    """
    h, w = channel_img.shape[:2]
    combined = np.zeros((h, w), dtype=np.float32)

    for theta in GABOR_THETAS:
        for norm_freq in GABOR_NORMS:
            lambd = (
                GABOR_LAMBDA  # config constant — image-size-independent, reproducible
            )
            # FIX: cv2.CV_32F prevents memory corruption on 1-channel grayscale
            kernel = cv2.getGaborKernel(
                (GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE),
                GABOR_SIGMA,
                theta,
                lambd,
                GABOR_GAMMA,
                GABOR_PSI,
                ktype=cv2.CV_32F,
            )
            filtered = cv2.filter2D(channel_img, cv2.CV_32F, kernel)
            filtered = np.abs(filtered)
            if filtered.max() > 0:
                filtered = filtered / filtered.max()
            combined += filtered.astype(np.float32)

    combined /= len(GABOR_THETAS) * len(GABOR_NORMS)
    return combined  # in [0, 1]


def _compute_gabor_dual_channel_raw(
    img_rgb: np.ndarray, lab: np.ndarray | None = None
) -> np.ndarray:
    """Raw Gabor response combining grayscale texture with a*-channel texture.

    Priority 3: for MSV, the chlorotic streak has a distinctive chromatic
    texture in the a* channel (oscillation between chlorotic and healthy
    tissue) that can be more discriminative than luminance texture alone,
    especially under overcast lighting where luminance contrast is low.
    The a* channel is rescaled to span the full [0, 255] range (via
    min-max normalization) before running the same kernel bank used on
    grayscale, since Gabor magnitude response is sensitive to the dynamic
    range of the input.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gabor_raw = _compute_gabor_raw_response(gray)

    if lab is None:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a_channel = lab[:, :, 1].astype(np.float32)
    a_norm = cv2.normalize(a_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gabor_raw_a = _compute_gabor_raw_response(a_norm)

    gabor_combined = (
        1 - GABOR_A_CHANNEL_WEIGHT
    ) * gabor_raw + GABOR_A_CHANNEL_WEIGHT * gabor_raw_a
    return gabor_combined


def _compute_gabor_combined_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Grayscale-only Gabor binary mask — legacy/HSV-path helper.

    Retained as-is (grayscale only, thresholded) for the legacy HSV
    functions below, which were not part of the Priority 3 dual-channel
    change. compute_lab_hard_mask / compute_lab_soft_confidence use
    _compute_gabor_dual_channel_raw() + _compute_gabor_combined_mask_dual()
    instead.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    combined = _compute_gabor_raw_response(gray)
    return (combined >= GABOR_THRESHOLD).astype(np.uint8)


def _compute_gabor_combined_mask_dual(
    img_rgb: np.ndarray, lab: np.ndarray | None = None
) -> np.ndarray:
    """Dual-channel (grayscale + a*) Gabor binary mask, thresholded at GABOR_THRESHOLD.

    Used by compute_lab_soft_confidence()'s MSV branch in place of the
    grayscale-only _compute_gabor_combined_mask() (Priority 3).
    """
    gabor_combined = _compute_gabor_dual_channel_raw(img_rgb, lab)
    return (gabor_combined >= GABOR_THRESHOLD).astype(np.uint8)


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

LAB_MSV_A_MIN = 133  # slight red-shift: chlorotic/necrotic streaks
LAB_MSV_B_MIN = 135  # yellowing component
LAB_MLN_A_MIN = 130  # MLN lesions trend slightly less red than MSV
LAB_MLN_B_MIN = 138  # MLN necrosis more yellow-brown
LAB_GREEN_A_MAX = 121  # pixels greener than this are healthy tissue (raised 124→121: tighter exclusion preserves chlorotic-yellow pixels in a* 121–130 range)

# Multi-scale morphological union threshold.
# The large-kernel (1×15) opening is only added to the union when the
# symptomatic area fraction exceeds this value. Below it, only the small
# kernel (1×7) is used to preserve early-stage flecks.
MULTISCALE_LARGE_KERNEL_THRESHOLD = 0.08  # 8% of leaf silhouette area


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


def _clahe_L(lab: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
    """Apply CLAHE to L* channel within the leaf mask region only.

    More targeted than full-image CLAHE in load_image_clahe() — equalization
    is computed from leaf pixels only, so background tone does not shift the
    contrast curve for the leaf region.

    FIX: tileGridSize is now proportional to image resolution.
    Fixed (4,4) on a 3000×4000 field image produces tiles of ~750×1000px —
    effectively global histogram equalization, defeating the local adaptation.
    We target ~56px tiles (matching Student input resolution scale) clamped
    to a minimum of 4 and rounded to even numbers for OpenCV compatibility.
    """
    lab = lab.copy()
    L = lab[:, :, 0]
    mask = silhouette > 0
    if mask.sum() < 100:
        return lab  # too few leaf pixels — skip

    # Proportional tile size: target ~1/56th of each dimension, min 4, even numbers
    h, w = L.shape[:2]
    tile_h = max(4, (h // 56) // 2 * 2)  # round down to even
    tile_w = max(4, (w // 56) // 2 * 2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_w, tile_h))

    L_eq = clahe.apply(L)
    # Apply equalization only within leaf mask — preserve background L*
    L_out = L.copy()
    L_out[mask] = L_eq[mask]
    lab[:, :, 0] = L_out
    return lab


def _clahe_a(lab: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the a* channel within the leaf mask region only.

    Mirrors _clahe_L() but operates on the a* channel (index 1) with a
    lower clip limit (CLAHE_A_CLIP_LIMIT = 1.0 vs L*'s 2.0) to enhance
    local chromatic contrast without amplifying chromatic noise.

    Priority 5 (CLAHE-1): in low-contrast (overcast) captures, mild MSV
    streaks may have a* values nearly identical to surrounding healthy tissue.
    CLAHE on the a* channel within the leaf mask makes early streaks more
    distinguishable before thresholding.

    Pixels outside the leaf silhouette retain their original a* value so
    that background chromatics are never affected.
    """
    lab = lab.copy()
    a = lab[:, :, 1]
    mask = silhouette > 0
    if mask.sum() < 100:
        return lab  # too few leaf pixels — skip

    # Reuse the same proportional tile size strategy as _clahe_L()
    h, w = a.shape[:2]
    tile_h = max(4, (h // 56) // 2 * 2)
    tile_w = max(4, (w // 56) // 2 * 2)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_A_CLIP_LIMIT, tileGridSize=(tile_w, tile_h))

    a_eq = clahe.apply(a)
    # Apply equalization only within leaf mask — preserve background a*
    a_out = a.copy()
    a_out[mask] = a_eq[mask]
    lab[:, :, 1] = a_out
    return lab


def _margin_erosion_mask(silhouette: np.ndarray) -> np.ndarray:
    """Convex-hull margin erosion mask to suppress tip-burn MLN false positives.

    Leaf tip burn (marginal chlorosis/necrosis) is phenotypically similar to
    MLN but is a distinct physiological response that occurs at leaf margins
    and tips. It is the second-ranked false positive source for MLN detection.

    This function computes the convex hull of the leaf silhouette, then erodes
    it inward by MARGIN_EROSION_FRAC × sqrt(leaf_area) pixels. The result is
    an interior mask that excludes the outer ring of the leaf from MLN
    detection, suppressing tip-burn signals without touching the leaf interior
    where genuine MLN necrosis appears.

    Priority 6 (MLN-3): applied to the MLN branch only in
    compute_lab_hard_mask() and compute_lab_soft_confidence(). MSV and
    HEALTHY branches are unaffected.

    Returns a uint8 mask (1 = interior, 0 = margin/outside).
    """
    contours, _ = cv2.findContours(
        silhouette.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return silhouette.astype(np.uint8)

    # Draw filled convex hull
    hull_mask = np.zeros_like(silhouette, dtype=np.uint8)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    cv2.drawContours(hull_mask, [hull], -1, 1, thickness=cv2.FILLED)

    # Erosion radius: MARGIN_EROSION_FRAC × sqrt(leaf_area), minimum 1 px
    leaf_area = int(silhouette.sum())
    radius = max(1, int(MARGIN_EROSION_FRAC * np.sqrt(leaf_area)))

    # Circular erosion kernel
    diameter = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    eroded = cv2.erode(hull_mask, kernel, iterations=1)

    return eroded.astype(np.uint8)


def _sev_to_cimmyt_grade(severity_pct: float, category: str) -> int:
    """Map continuous severity % to CIMMYT published agronomic grade.

    Brackets are read from config.py (CIMMYT_MSV_BRACKETS / CIMMYT_MLN_BRACKETS)
    so config remains the single source of truth — see config.py's
    "CIMMYT SEVERITY GRADING" section for the published bracket definitions.

    MSV scale (CIMMYT, 1–9):
      1=<5%  3=5–25%  5=25–50%  7=50–75%  9=>75%
    MLN scale (CIMMYT, 1–5):
      1=<10%  2=10–25%  3=25–50%  4=50–75%  5=>75%
    HEALTHY: always grade 0 (no disease).
    Returns -1 for excluded images (severity_pct < 0).
    """
    if severity_pct < 0:
        return -1
    if category == "HEALTHY":
        return 0
    brackets = CIMMYT_MSV_BRACKETS if category == "MSV" else CIMMYT_MLN_BRACKETS
    for lo, hi, grade in brackets:
        if lo <= severity_pct < hi:
            return grade
    # Severity at or beyond the last bracket's upper bound (e.g. exactly 100
    # with an exclusive 101 ceiling never hit due to float edge cases) —
    # fall back to the highest defined grade.
    return brackets[-1][2]


def _compute_raw_vesselness(lab_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frangi vesselness filter on L* and a* channels — pure function of the
    raw (non-CLAHE, non-normalized) LAB image. Call this ONCE per image, not
    once per mode.

    Maize leaf veins are elongated ridge structures in the L* channel.
    The Frangi filter responds strongly to these ridges and weakly to the
    interveinal tissue where MSV chlorosis and MLN necrosis actually appear.

    Parameters tuned for maize leaf morphology:
      sigmas=FRANGI_SIGMAS  — captures fine veins (0.5) through the midrib
                              (10.0, ≈10-30 px wide at typical capture
                              resolution). Widened from (0.5, 1.5, 3.0) in v7
                              because the midrib was under-responding and
                              registering as a false positive on yellow maize.
      black_ridges=False     — veins are bright ridges in L* (light on dark)

    P7 (VEIN-2): also run Frangi on the normalised a* channel.
    Veins on yellow maize appear not only as luminance ridges (high L*) but
    also as chromatic ridges — vascular bundles are slightly greener (lower a*)
    than surrounding yellow tissue due to chlorophyll concentration. Running
    Frangi on a* with black_ridges=True (veins are darker = lower a* in the
    a* image) and taking the pixel-wise maximum of both vesselness maps gives
    a more complete vein map that catches veins invisible in L*.
    """
    L_norm = lab_raw[:, :, 0].astype(np.float32) / 255.0
    a_norm = lab_raw[:, :, 1].astype(np.float32) / 255.0

    # Compute vesselness — responds to elongated ridge structures (veins).
    # v8: sigma range widened via FRANGI_SIGMAS (was hardcoded (0.5, 1.5, 3.0))
    # to add response at the midrib scale (~10-30 px wide), which is the
    # primary source of false positives on yellow maize.
    vesselness_L = frangi(
        L_norm,
        sigmas=FRANGI_SIGMAS,  # fine vein → midrib scale range
        black_ridges=False,  # veins are bright in L*
        mode="reflect",
    ).astype(np.float32)

    vesselness_a = frangi(
        a_norm,
        sigmas=FRANGI_SIGMAS,
        black_ridges=True,  # veins are greener = lower a* = darker in a* image
        mode="reflect",
    ).astype(np.float32)

    return vesselness_L, vesselness_a


def _vein_suppression_from_raw(
    vesselness_L: np.ndarray, vesselness_a: np.ndarray, silhouette: np.ndarray
) -> np.ndarray:
    """Cheap, per-silhouette normalize+combine+suppress step. Identical logic to
    the original _compute_vein_suppression, just split apart so the expensive
    Frangi computation above can be cached across modes. Safe to call once per
    mode (the silhouette differs per mode).

    Returns a float32 suppression weight in [0, 1] where:
      - values near 0 = strong vein response → suppress symptom mask here
      - values near 1 = weak vein response  → keep symptom mask here

    The weight is multiplied into the symptom mask after morphological processing,
    so vein pixels are down-weighted without being hard-excluded (which would
    create holes in severe images where veins and symptoms overlap).
    """
    # Normalise vesselness to [0, 1] within the leaf silhouette
    vL = vesselness_L
    leaf_vals = vL[silhouette == 1]
    if leaf_vals.max() > 1e-6:
        vL = vL / leaf_vals.max()

    # Normalise a*-Frangi within the leaf silhouette (same approach as L*-Frangi)
    vA = vesselness_a
    leaf_vals_a = vA[silhouette == 1]
    if leaf_vals_a.max() > 1e-6:
        vA = vA / leaf_vals_a.max()

    # Combine: take pixel-wise maximum before soft suppression formula
    vesselness = np.maximum(vL, vA)

    # Convert to suppression weight: high vesselness → low weight
    # Use soft suppression (1 - v^0.5) rather than hard threshold so
    # partial-vein pixels are smoothly reduced, not binary-excluded.
    suppression_weight = 1.0 - np.sqrt(np.clip(vesselness, 0.0, 1.0))
    suppression_weight = np.clip(suppression_weight, 0.0, 1.0)

    # Outside silhouette: weight = 0 (already excluded by silhouette &)
    suppression_weight *= silhouette.astype(np.float32)

    return suppression_weight


def _get_vein_suppression(
    lab_raw: np.ndarray, silhouette: np.ndarray, vein_cache: dict | None = None
) -> np.ndarray:
    """Drop-in replacement for _compute_vein_suppression(img_rgb, silhouette).

    Pass a per-image {} dict as vein_cache to memoize the expensive Frangi
    computation across the 4-mode loop in process_single_image_cpu, since
    frangi()'s output depends only on lab_raw (i.e. on img_rgb), not on the
    per-mode silhouette.
    """
    if vein_cache is not None and "raw" in vein_cache:
        raw_l, raw_a = vein_cache["raw"]
    else:
        raw_l, raw_a = _compute_raw_vesselness(lab_raw)
        if vein_cache is not None:
            vein_cache["raw"] = (raw_l, raw_a)
    return _vein_suppression_from_raw(raw_l, raw_a, silhouette)


def _rotated_rect_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Create a binary line structuring element of `length` rotated to `angle_deg`.

    Draws a 1-pixel-wide horizontal line of the given length on a
    (length × length) canvas, then rotates it about the canvas center via
    cv2.getRotationMatrix2D / cv2.warpAffine. 0 degrees = horizontal,
    90 degrees = vertical.

    Used to build a directional morphological opening *bank* (Priority 2):
    MSV streaks run parallel to the leaf midrib, which can sit at any
    rotational angle depending on how the leaf is oriented in-frame. A
    single fixed (1, N) vertical kernel only preserves streaks aligned near
    90 degrees, so a bank of rotated kernels is unioned instead.
    """
    canvas = np.zeros((length, length), dtype=np.uint8)
    center_row = length // 2
    canvas[center_row, :] = 1

    rot_mat = cv2.getRotationMatrix2D((length / 2.0, length / 2.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        canvas,
        rot_mat,
        (length, length),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    kernel = (rotated > 0).astype(np.uint8)
    if not kernel.any():
        # Degenerate rotation (shouldn't happen with INTER_NEAREST, but guard
        # against an all-zero structuring element crashing morphologyEx).
        kernel = canvas.copy()
    return kernel


def _multiscale_symptom_union(
    lab_mask: np.ndarray, category: str, sym_area_frac: float = 0.0
) -> np.ndarray:
    """Multi-scale directional morphological opening and union.

    Runs directional opening at two kernel sizes and combines them:
      - Small kernel (length 7):  preserves early-stage flecks and short streaks
      - Large kernel (length 15): targets late-stage long streaks

    Union strategy:
      - Always include small-kernel result (catches early flecks)
      - Include large-kernel result only when symptomatic area is substantial
        (> MULTISCALE_LARGE_KERNEL_THRESHOLD of leaf area), so late-stage
        images get both scales while early-stage images avoid over-smoothing.

    MSV: opening uses an *angled bank* (Priority 2) — one kernel per angle in
    MORPH_OPEN_ANGLES, unioned together — instead of a single fixed vertical
    (1, N) kernel, so streaks survive regardless of leaf rotation in-frame.

    MLN: applies a 5×5 elliptical closing before opening to bridge fragmented
    necrotic patches (unchanged from existing logic), then uses the original
    single vertical kernel — MLN necrosis is blob-like, not streak-like, so
    the angled bank does not apply here.

    HEALTHY: unchanged single vertical kernel (rarely fires in practice).
    """
    if not lab_mask.any():
        return lab_mask

    # ── MLN: close fragmented necrotic patches first ─────────────────────────
    if category == "MLN":
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lab_mask = cv2.morphologyEx(lab_mask, cv2.MORPH_CLOSE, kernel_close)

    if category == "MSV":
        # Angled directional opening bank: union the opening result across
        # every angle in MORPH_OPEN_ANGLES so streaks at any orientation
        # survive, not just near-vertical ones.
        opened_small = np.zeros_like(lab_mask)
        for angle in MORPH_OPEN_ANGLES:
            kernel = _rotated_rect_kernel(7, angle)
            opened_small = cv2.bitwise_or(
                opened_small, cv2.morphologyEx(lab_mask, cv2.MORPH_OPEN, kernel)
            )

        # Only add the large-kernel bank when symptom area is already
        # substantial. This avoids destroying early flecks on low-severity
        # images.
        if sym_area_frac > MULTISCALE_LARGE_KERNEL_THRESHOLD:
            opened_large = np.zeros_like(lab_mask)
            for angle in MORPH_OPEN_ANGLES:
                kernel = _rotated_rect_kernel(15, angle)
                opened_large = cv2.bitwise_or(
                    opened_large, cv2.morphologyEx(lab_mask, cv2.MORPH_OPEN, kernel)
                )
            result = cv2.bitwise_or(opened_small, opened_large)
        else:
            result = opened_small
    else:
        # MLN / HEALTHY: unchanged single fixed vertical-kernel opening.
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

        opened_small = cv2.morphologyEx(lab_mask, cv2.MORPH_OPEN, kernel_small)

        if sym_area_frac > MULTISCALE_LARGE_KERNEL_THRESHOLD:
            opened_large = cv2.morphologyEx(lab_mask, cv2.MORPH_OPEN, kernel_large)
            result = cv2.bitwise_or(opened_small, opened_large)
        else:
            result = opened_small

    return result


def _otsu_2d(
    vals_a: np.ndarray, vals_b: np.ndarray, n_bins: int
) -> tuple[float, float]:
    """2D joint Otsu threshold on the (a*, b*) histogram.

    Independent 1D Otsu on a* and b* separately misses cases where neither
    channel alone is strongly elevated but their *combination* is
    distinctive of early chlorotic tissue. This generalizes Otsu to two
    dimensions: it finds the split point (ta, tb) that partitions the joint
    (a*, b*) histogram into a "background" quadrant (a < ta AND b < tb) and
    everything else (a >= ta OR b >= tb) — mirroring the OR-style gate
    already used downstream (`(a >= a_min) | (b >= b_min)`) — such that the
    between-class variance of the two groups' mean (a*, b*) vectors is
    maximized.

    Implemented with vectorized numpy cumulative sums over the 2D histogram
    rather than a Python double-loop over threshold pairs, since n_bins is
    typically 64 (4096 candidate splits).

    Returns (best_ta, best_tb) in the original [0, 255] LAB scale. Falls
    back to the neutral midpoint (128.0, 128.0) on degenerate input (e.g.
    all pixels in one bin).
    """
    H, a_edges, b_edges = np.histogram2d(
        vals_a.astype(np.float64),
        vals_b.astype(np.float64),
        bins=n_bins,
        range=[[0, 255], [0, 255]],
    )
    total = H.sum()
    if total <= 0:
        return 128.0, 128.0

    P = H / total  # joint probability mass per (a_bin, b_bin)

    # Representative value (bin center) per bin, used for weighted means.
    a_centers = (a_edges[:-1] + a_edges[1:]) / 2.0  # (n_bins,)
    b_centers = (b_edges[:-1] + b_edges[1:]) / 2.0  # (n_bins,)

    PA = P * a_centers[:, None]  # a*-weighted mass per bin
    PB = P * b_centers[None, :]  # b*-weighted mass per bin

    # 2D cumulative sums: cumP[i, j] = mass with a_bin <= i AND b_bin <= j
    cumP = np.cumsum(np.cumsum(P, axis=0), axis=1)
    cumA = np.cumsum(np.cumsum(PA, axis=0), axis=1)
    cumB = np.cumsum(np.cumsum(PB, axis=0), axis=1)

    total_a = cumA[-1, -1]
    total_b = cumB[-1, -1]

    w0 = cumP  # background-quadrant mass (a < ta, b < tb) for each split
    w1 = 1.0 - w0  # everything else (a >= ta OR b >= tb)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_a0 = cumA / w0
        mean_b0 = cumB / w0
        mean_a1 = (total_a - cumA) / w1
        mean_b1 = (total_b - cumB) / w1

        between = w0 * w1 * ((mean_a0 - mean_a1) ** 2 + (mean_b0 - mean_b1) ** 2)

    between = np.nan_to_num(between, nan=-1.0, posinf=-1.0, neginf=-1.0)

    # Exclude degenerate splits where either class would be empty.
    valid = (w0 > 1e-9) & (w1 > 1e-9)
    if not valid.any():
        return 128.0, 128.0
    between[~valid] = -1.0

    i, j = np.unravel_index(np.argmax(between), between.shape)

    best_ta = float(a_edges[i + 1])  # upper edge of bin i = split point
    best_tb = float(b_edges[j + 1])
    return best_ta, best_tb


def compute_lab_hard_mask(
    img_rgb: np.ndarray,
    silhouette: np.ndarray,
    category: str,
    lab_raw: np.ndarray | None = None,
    vein_cache: dict | None = None,
) -> np.ndarray:
    """LAB hard binary symptom mask with Frangi vein suppression + multi-scale opening (v8).

    Changes vs v6 (v6 -> v7):
      ALL   — Frangi vesselness filter (skimage.filters.frangi) applied to L*
               channel after morphological opening to suppress vein-ridge pixels.
               Veins score high in LAB a*/b* space on yellow maize and are the
               primary source of false detections. Suppression is soft
               (1 - sqrt(vesselness)) so partial-vein pixels are smoothly reduced.
               HEALTHY uses the most aggressive threshold (0.6) since any surviving
               symptom there is almost certainly a false positive.
               MSV: threshold 0.5. MLN: threshold 0.35 (necrosis crosses veins).

      ALL   — Single directional opening replaced with _multiscale_symptom_union():
               small kernel (1×7) always runs; large kernel (1×15) added only when
               symptomatic area fraction exceeds 8% of leaf area. This preserves
               early-stage flecks on low-severity images while still capturing
               late-stage long streaks.

      MLN   — The 5×5 elliptical closing (bridges fragmented necrotic patches) is
               now handled inside _multiscale_symptom_union() so the order is
               always: close → open_small → union_large (if area > threshold)
               → vein suppression.

    Changes v7 -> v8:
      P1    — Frangi sigma range widened to FRANGI_SIGMAS (adds 6.0, 10.0) so
               vesselness responds to the midrib, not just fine/mid veins.
               (Lives in _compute_raw_vesselness(), called from here via
               _get_vein_suppression().)

      P2    — MSV opening in _multiscale_symptom_union() now unions an angled
               kernel bank (MORPH_OPEN_ANGLES) instead of a single fixed
               vertical kernel, so streaks survive at any leaf rotation.

      P3    — MSV Gabor response now combines grayscale texture with a*-channel
               texture (GABOR_A_CHANNEL_WEIGHT) via _compute_gabor_dual_channel_raw(),
               since the chlorotic streak's chromatic texture in a* can be more
               discriminative than luminance texture under overcast lighting.

      P4    — MSV adaptive threshold now uses a 2D joint Otsu on the (a*, b*)
               histogram (_otsu_2d(), OTSU_2D_BIN_COUNT bins/axis) instead of
               independent 1D Otsu per channel, catching cases where neither
               channel alone is elevated but their combination is. Falls back
               to the v7 independent 1D Otsu when leaf pixel count < 200.

      P5    — _clahe_a() applied immediately after _clahe_L() to enhance local
               chromatic contrast of early MSV streaks in low-contrast (overcast)
               captures (CLAHE_A_CLIP_LIMIT=1.0, lower than L* clip of 2.0).

      P6    — Convex-hull margin erosion mask (_margin_erosion_mask()) applied
               to MLN branch only (after silhouette &), eroding the leaf perimeter
               by MARGIN_EROSION_FRAC × sqrt(leaf_area) pixels to suppress tip-burn
               false positives (distinct physiological response, not MLN).
    """
    if lab_raw is None:
        lab_raw = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab = _normalize_L(lab_raw)
    lab = _clahe_L(lab, silhouette)  # NEW-2: targeted L* CLAHE within leaf mask
    lab = _clahe_a(
        lab, silhouette
    )  # P5 (CLAHE-1): chromatic contrast on a* within leaf mask
    L_norm = lab[:, :, 0].astype(np.int32)  # 0-255 after normalization
    a = lab[:, :, 1].astype(np.int32)  # 0-255, neutral=128
    b = lab[:, :, 2].astype(np.int32)  # 0-255, neutral=128

    green_excl = a < LAB_GREEN_A_MAX

    if category == "MSV":
        # v8 Priority 4: 2D joint Otsu on the (a*, b*) histogram within the
        # leaf mask. Independent 1D Otsu on a* and b* separately misses
        # cases where neither channel alone is strongly elevated but their
        # *combination* is distinctive of early chlorotic tissue. Falls
        # back to the v7 independent 1D Otsu when there isn't enough leaf
        # signal for a stable 2D histogram.
        leaf_a = a[silhouette == 1].astype(np.uint8)
        leaf_b = b[silhouette == 1].astype(np.uint8)

        if len(leaf_a) >= 200:
            ta_2d, tb_2d = _otsu_2d(leaf_a, leaf_b, OTSU_2D_BIN_COUNT)
            # Same floor+ceiling clamp as the v7 1D Otsu: never go below the
            # fixed floor (LAB_MSV_A_MIN / LAB_MSV_B_MIN), never exceed the
            # ceiling, so the joint threshold cannot drift into healthy
            # tissue range on heavily diseased leaves.
            a_min_eff = int(min(max(ta_2d, LAB_MSV_A_MIN), 148))
            b_min_eff = int(min(max(tb_2d, LAB_MSV_B_MIN), 150))
        else:
            # Fallback: v7 independent 1D Otsu (insufficient pixels for a
            # stable 2D histogram).
            if len(leaf_a) > 100:
                otsu_thresh_a, _ = cv2.threshold(
                    leaf_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                a_min_eff = int(min(max(otsu_thresh_a, LAB_MSV_A_MIN), 148))
            else:
                a_min_eff = LAB_MSV_A_MIN
            if len(leaf_b) > 100:
                otsu_thresh_b, _ = cv2.threshold(
                    leaf_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                b_min_eff = int(min(max(otsu_thresh_b, LAB_MSV_B_MIN), 150))
            else:
                b_min_eff = LAB_MSV_B_MIN

        # LAB colour gate using adaptive thresholds
        lab_mask = ((a >= a_min_eff) | (b >= b_min_eff)).astype(np.uint8)
        lab_mask &= (~green_excl).astype(np.uint8)
        lab_mask &= silhouette

        # Gabor as a soft weight rather than a binary AND gate.
        # v8 Priority 3: combine grayscale Gabor with a*-channel Gabor —
        # the chlorotic streak has a distinctive chromatic texture in a*
        # (oscillation between chlorotic and healthy tissue) that can be
        # more discriminative than luminance texture under overcast
        # lighting where luminance contrast is low.
        gabor_combined = _compute_gabor_dual_channel_raw(img_rgb, lab)

        # Multiply LAB mask by raw Gabor response; threshold product at 0.4.
        # Strong LAB hits survive even when Gabor is borderline.
        weighted = lab_mask.astype(np.float32) * gabor_combined
        symptom = (weighted >= 0.4).astype(np.uint8)

        # ── Multi-scale directional opening (NEW) ────────────────────────────
        # Replaces the single (1,7) opening with a two-scale union:
        #   (1,7)  — preserves early-stage flecks and short broken streaks
        #   (1,15) — added only when symptomatic area is already substantial
        #            (> 8% leaf area), targeting late-stage long streaks.
        # sym_area_frac computed from pre-opening symptom to decide scale.
        leaf_px = max(silhouette.sum(), 1)
        sym_area_frac = symptom.sum() / leaf_px
        symptom = _multiscale_symptom_union(symptom, "MSV", sym_area_frac)

        # ── Frangi vein suppression (NEW) ────────────────────────────────────
        # Down-weight pixels where the Frangi vesselness filter detected vein
        # ridges. Uses soft suppression (1 - sqrt(vesselness)) so partial-vein
        # pixels are smoothly reduced rather than hard-excluded.
        # Applied after morphological opening so vein suppression acts on the
        # cleaned mask, not on raw noisy detections.
        if symptom.any():
            vein_weight = _get_vein_suppression(lab_raw, silhouette, vein_cache)
            symptom_f = symptom.astype(np.float32) * vein_weight
            symptom = (symptom_f >= 0.5).astype(np.uint8)

    elif category == "MLN":
        # NEW-1: Otsu-guided adaptive threshold for MLN channels
        # FIX: clamp Otsu with both floor AND ceiling for MLN branch.
        leaf_a = a[silhouette == 1].astype(np.uint8)
        if len(leaf_a) > 100:
            otsu_a = cv2.threshold(leaf_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
                0
            ]
            a_min_eff = int(min(max(otsu_a, LAB_MLN_A_MIN), 146))
        else:
            a_min_eff = LAB_MLN_A_MIN
        leaf_b = b[silhouette == 1].astype(np.uint8)
        if len(leaf_b) > 100:
            otsu_b = cv2.threshold(leaf_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
                0
            ]
            b_min_eff = int(min(max(otsu_b, LAB_MLN_B_MIN), 150))
        else:
            b_min_eff = LAB_MLN_B_MIN

        # Primary yellowing / red-shift condition
        yellowing = (a >= a_min_eff) | (b >= b_min_eff)

        # NEW: dark necrosis condition — low L* (dark patches) with slight
        # red-shift capture necrotic tissue that has low b* and would be missed
        # by the yellowing gate.
        dark_necrosis = (L_norm < 110) & (a >= 125)

        symptom = (yellowing | dark_necrosis).astype(np.uint8)
        symptom &= (~green_excl).astype(np.uint8)
        symptom &= silhouette

        # P6 (MLN-3): exclude leaf margins to suppress tip-burn false positives.
        # Convex-hull erosion removes the outer MARGIN_EROSION_FRAC × sqrt(area)
        # pixel ring. Applied to MLN branch only — MSV and HEALTHY unaffected.
        margin_mask = _margin_erosion_mask(silhouette)
        symptom &= margin_mask

        # ── Multi-scale directional opening with MLN closing (NEW) ───────────
        # _multiscale_symptom_union handles the 5×5 elliptical closing
        # (to bridge fragmented necrotic patches) before the two-scale opening.
        # Large kernel (1×15) is added when symptomatic area > 8% leaf area.
        leaf_px = max(silhouette.sum(), 1)
        sym_area_frac = symptom.sum() / leaf_px
        symptom = _multiscale_symptom_union(symptom, "MLN", sym_area_frac)

        # ── Frangi vein suppression (NEW) ────────────────────────────────────
        # MLN necrotic patches tend to cross veins more than MSV streaks do,
        # so suppression is softer here — threshold lowered to 0.35 so that
        # necrotic tissue overlapping vein pixels is not over-suppressed.
        if symptom.any():
            vein_weight = _get_vein_suppression(lab_raw, silhouette, vein_cache)
            symptom_f = symptom.astype(np.float32) * vein_weight
            symptom = (symptom_f >= 0.35).astype(np.uint8)

    else:
        # HEALTHY: use stricter own thresholds (a ≥ 140, b ≥ 145) so normal
        # warm-yellow leaf coloration does not trigger a non-zero mask before
        # the area check in process_single_image_cpu().
        symptom = ((a >= 140) | (b >= 145)).astype(np.uint8)
        symptom &= (~green_excl).astype(np.uint8)
        symptom &= silhouette

        # ── Multi-scale opening (NEW) ─────────────────────────────────────────
        # HEALTHY images rarely exceed the 8% threshold so in practice only
        # the small (1×7) kernel fires — this is intentional and conservative.
        leaf_px = max(silhouette.sum(), 1)
        sym_area_frac = symptom.sum() / leaf_px
        symptom = _multiscale_symptom_union(symptom, "HEALTHY", sym_area_frac)

        # ── Frangi vein suppression (NEW) ────────────────────────────────────
        # Most beneficial for HEALTHY: vein ridges on yellow maize score highly
        # in LAB a*/b* space and are the primary source of false-positive
        # symptom detections. Suppressing them is the single biggest improvement
        # for HEALTHY noise gate accuracy.
        # Higher threshold (0.6) than MSV/MLN — be aggressive at suppression
        # since any surviving symptom on HEALTHY is almost certainly a false positive.
        if symptom.any():
            vein_weight = _get_vein_suppression(lab_raw, silhouette, vein_cache)
            symptom_f = symptom.astype(np.float32) * vein_weight
            symptom = (symptom_f >= 0.6).astype(np.uint8)

    return (symptom * 255).astype(np.uint8)


def compute_lab_soft_confidence(
    img_rgb: np.ndarray,
    silhouette: np.ndarray,
    category: str,
    lab_raw: np.ndarray | None = None,
    vein_cache: dict | None = None,
) -> np.ndarray:
    """LAB soft confidence map (analogous to compute_hsv_soft_confidence, for mode_d).

    v7 -> v8: MSV branch now mirrors compute_lab_hard_mask's adaptive
    thresholding — 2D joint Otsu on (a*, b*) with 1D-Otsu fallback (P4) —
    and uses the dual-channel (grayscale + a*) Gabor mask (P3) instead of
    the grayscale-only mask, so the soft and hard mask paths stay
    consistent.

    v7 -> v8 (Part 2):
      P5 (CLAHE-1) — _clahe_a() applied immediately after _clahe_L() to
                     enhance local chromatic contrast of early MSV streaks.
      P6 (MLN-3)   — Convex-hull margin erosion mask applied to MLN branch
                     only to suppress tip-burn false positives.

    v8 -> v9 (bug fixes — mode_d / mode_b/c cross-mode parity):
      FIX-1 (MLN)  — MLN branch now uses Otsu-adaptive thresholding (1D,
                     clamped to floor LAB_MLN_A_MIN/B_MIN and ceiling 146/150)
                     matching compute_lab_hard_mask()'s MLN branch exactly.
                     Previously used fixed LAB_MLN_A_MIN/B_MIN — modes b/c
                     and mode_d saw different adaptive thresholds on the same
                     image under variable lighting, corrupting the cross-mode
                     ablation study.
      FIX-2 (MLN)  — Added dark necrosis soft confidence channel to match
                     compute_lab_hard_mask()'s `(L_norm < 110) & (a >= 125)`
                     gate. Dark necrotic patches scored near-zero confidence
                     in mode_d but non-zero in modes b/c — same ablation
                     corruption. Soft version ramps linearly: darker → higher
                     confidence (capped at 1.0).
    """
    if lab_raw is None:
        lab_raw = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab = _normalize_L(lab_raw)
    lab = _clahe_L(lab, silhouette)  # L* CLAHE within leaf mask (mirrors hard mask)
    lab = _clahe_a(
        lab, silhouette
    )  # P5 (CLAHE-1): chromatic contrast on a* within leaf mask
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    green_excl = a < float(LAB_GREEN_A_MAX)

    # BUG-1 FIX: HEALTHY now uses its own stricter thresholds (was using MLN)
    if category == "MSV":
        # v8 Priority 4: same 2D joint Otsu (with 1D fallback) used in
        # compute_lab_hard_mask, so the soft confidence map's adaptive
        # threshold matches the hard mask's behavior.
        leaf_a = a[silhouette == 1].astype(np.uint8)
        leaf_b = b[silhouette == 1].astype(np.uint8)

        if len(leaf_a) >= 200:
            ta_2d, tb_2d = _otsu_2d(leaf_a, leaf_b, OTSU_2D_BIN_COUNT)
            a_min = float(min(max(ta_2d, LAB_MSV_A_MIN), 148))
            b_min = float(min(max(tb_2d, LAB_MSV_B_MIN), 150))
        else:
            if len(leaf_a) > 100:
                otsu_thresh_a, _ = cv2.threshold(
                    leaf_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                a_min = float(min(max(otsu_thresh_a, LAB_MSV_A_MIN), 148))
            else:
                a_min = float(LAB_MSV_A_MIN)
            if len(leaf_b) > 100:
                otsu_thresh_b, _ = cv2.threshold(
                    leaf_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                b_min = float(min(max(otsu_thresh_b, LAB_MSV_B_MIN), 150))
            else:
                b_min = float(LAB_MSV_B_MIN)
    elif category == "MLN":
        # BUG FIX (v8→v9): MLN branch now uses the same Otsu-adaptive thresholding
        # as compute_lab_hard_mask() so mode_d soft confidence and modes b/c hard
        # mask see the same adaptive threshold. Previously this was fixed at
        # LAB_MLN_A_MIN / LAB_MLN_B_MIN, causing the soft and hard mask paths to
        # diverge under variable lighting — the exact problem Otsu adaptation solves.
        leaf_a_u8 = a[silhouette == 1].astype(np.uint8)
        leaf_b_u8 = b[silhouette == 1].astype(np.uint8)
        if len(leaf_a_u8) > 100:
            otsu_a, _ = cv2.threshold(
                leaf_a_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            a_min = float(min(max(otsu_a, LAB_MLN_A_MIN), 146))
        else:
            a_min = float(LAB_MLN_A_MIN)
        if len(leaf_b_u8) > 100:
            otsu_b, _ = cv2.threshold(
                leaf_b_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            b_min = float(min(max(otsu_b, LAB_MLN_B_MIN), 150))
        else:
            b_min = float(LAB_MLN_B_MIN)
    else:  # HEALTHY — stricter thresholds match compute_lab_hard_mask
        a_min, b_min = 140.0, 145.0

    # Soft confidence: linear ramp from threshold to channel ceiling (255)
    a_conf = np.clip((a - a_min) / max(255.0 - a_min, 1.0), 0.0, 1.0)
    b_conf = np.clip((b - b_min) / max(255.0 - b_min, 1.0), 0.0, 1.0)
    conf_map = np.maximum(a_conf, b_conf)

    # BUG FIX (v8→v9): add dark necrosis soft confidence channel for MLN to
    # match compute_lab_hard_mask()'s `dark_necrosis = (L_norm < 110) & (a >= 125)`
    # gate. Without this, dark necrotic MLN patches (low L*, low a*, low b*) score
    # near zero confidence in mode_d while getting a non-zero hard mask in modes
    # b/c — corrupting the cross-mode ablation study.
    if category == "MLN":
        L_norm_f = lab[:, :, 0].astype(np.float32)
        dark_necrosis_conf = np.where(
            (L_norm_f < 110.0) & (a >= 125.0),
            np.clip((110.0 - L_norm_f) / 110.0, 0.0, 1.0),  # ramp: darker = higher conf
            0.0,
        ).astype(np.float32)
        conf_map = np.maximum(conf_map, dark_necrosis_conf)

    conf_map *= (~green_excl).astype(np.float32)
    conf_map *= silhouette.astype(np.float32)

    # P6 (MLN-3): apply convex-hull margin erosion mask to suppress tip-burn
    # false positives. Applied to MLN branch only — MSV and HEALTHY unaffected.
    if category == "MLN":
        margin_mask = _margin_erosion_mask(silhouette)
        conf_map *= margin_mask.astype(np.float32)

    if category == "MSV":
        # v8 Priority 3: dual-channel (grayscale + a*) Gabor mask in place
        # of the grayscale-only _compute_gabor_combined_mask().
        conf_map *= _compute_gabor_combined_mask_dual(img_rgb, lab).astype(np.float32)

    # BUG-2 FIX: use class-specific directional kernel sizes matching hard mask.
    # NEW: use _multiscale_symptom_union for consistency with hard mask.
    # Vein suppression applied as a soft multiply on the confidence map.
    if conf_map.max() > 0:
        binary_hint = (conf_map > 0).astype(np.uint8)
        sym_area_frac = binary_hint.sum() / max(silhouette.sum(), 1)
        binary_hint = _multiscale_symptom_union(
            binary_hint, category, float(sym_area_frac)
        )
        conf_map *= binary_hint.astype(np.float32)

        # Frangi vein suppression — same logic as hard mask but applied as a
        # soft multiply directly on the confidence values (no threshold).
        vein_weight = _get_vein_suppression(lab_raw, silhouette, vein_cache)
        conf_map *= vein_weight

    return np.clip(conf_map, 0.0, 1.0)


# ─── Legacy HSV detection (kept as fallback / comparison baseline) ────────────
def compute_hsv_hard_mask(img_rgb, silhouette, category):
    img_hsv = to_hsv(img_rgb)
    h, s, v = (
        img_hsv[:, :, 0].astype(np.int32),
        img_hsv[:, :, 1].astype(np.int32),
        img_hsv[:, :, 2].astype(np.int32),
    )
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
    if category == "MSV":
        symptom &= _compute_gabor_combined_mask(img_rgb)
    return (symptom * 255).astype(np.uint8)


def compute_hsv_soft_confidence(img_rgb, silhouette, category):
    img_hsv = to_hsv(img_rgb)
    h, s, v = (
        img_hsv[:, :, 0].astype(np.float32),
        img_hsv[:, :, 1].astype(np.float32),
        img_hsv[:, :, 2].astype(np.float32),
    )
    green_excl = _green_exclusion(
        h.astype(np.int32), s.astype(np.int32), v.astype(np.int32)
    )
    ranges = HSV_MSV_RANGES if category == "MSV" else HSV_MLN_RANGES

    conf_accum = np.zeros(img_rgb.shape[:2], dtype=np.float32)
    hit_count = np.zeros(
        img_rgb.shape[:2], dtype=np.float32
    )  # FIX: track per-pixel hit count
    for i, rng in enumerate(ranges):
        h_lo, h_hi = rng["h"]
        s_lo, s_hi = rng["s"]
        v_lo, v_hi = rng["v"]
        hit = _in_range(h.astype(np.int32), s.astype(np.int32), v.astype(np.int32), rng)
        if category == "MSV" and i == 2:
            hit = _apply_r3_area_filter(hit.astype(np.uint8)).astype(bool)

        h_center, s_center, v_center = (
            (h_lo + h_hi) / 2.0,
            (s_lo + s_hi) / 2.0,
            (v_lo + v_hi) / 2.0,
        )
        h_half, s_half, v_half = (
            max((h_hi - h_lo) / 2.0, 1.0),
            max((s_hi - s_lo) / 2.0, 1.0),
            max((v_hi - v_lo) / 2.0, 1.0),
        )

        d = np.clip(
            (
                1.0
                - np.abs(h - h_center) / h_half
                + 1.0
                - np.abs(s - s_center) / s_half
                + 1.0
                - np.abs(v - v_center) / v_half
            )
            / 3.0,
            0.0,
            1.0,
        )
        hit_f = hit.astype(np.float32)
        conf_accum += hit_f * d
        hit_count += hit_f  # FIX: count how many ranges fired per pixel

    # FIX: normalize by actual hit count (max 1 where any range fired) instead of total
    # len(ranges), which was capping MLN pixels at 0.20 (1/5) and preventing the 0.3 threshold
    safe_count = np.maximum(hit_count, 1.0)
    conf_map = (
        (conf_accum / safe_count)
        * (~green_excl).astype(np.float32)
        * silhouette.astype(np.float32)
    )
    if category == "MSV":
        conf_map *= _compute_gabor_combined_mask(img_rgb).astype(np.float32)
    return np.clip(conf_map, 0.0, 1.0)


def process_single_image_cpu(args):
    """Runs entirely on CPU threads to unblock the GPU pipeline"""
    (
        img_path,
        stem,
        category,
        is_tier1,
        orig_h,
        orig_w,
        img_rgb,
        soft_prob,
        mode_dirs,
    ) = args

    # mode_a: Otsu hard threshold on grayscale
    # mode_b: morphology-refined binary from teacher soft_prob  (close + open)
    # mode_c: raw soft_prob thresholded without morphology cleanup
    # mode_d: same sil as mode_b, but soft confidence symptom map
    binary_sil = refine_silhouette(soft_prob)  # morph-refined  (mode_b, mode_d)

    raw_thresh_sil = (soft_prob >= FACTORY_SILHOUETTE_THRESHOLD).astype(
        np.uint8
    )  # no morph (mode_c)
    otsu_sil = (
        cv2.threshold(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        > 0
    ).astype(np.uint8)

    # Coverage and weight are derived from the refined silhouette (mode_b baseline)
    coverage = float(binary_sil.sum()) / (orig_h * orig_w)
    weight = get_reliability_weight(coverage)

    result = {
        "filename": stem,
        "category": category,
        "coverage": round(coverage, 4),
        "weight": weight if weight is not None else -1,
    }
    modes_data = {}

    SIL_MAP = {
        "mode_a": otsu_sil,
        "mode_b": binary_sil,
        "mode_c": raw_thresh_sil,  # FIX: was binary_sil — mode_c must differ from mode_b
        "mode_d": binary_sil,
    }

    # Performance: compute the raw LAB conversion once per image and share it
    # across all 4 modes (instead of each mode re-running cv2.cvtColor). The
    # vein_cache dict memoizes the expensive Frangi vesselness computation
    # (_compute_raw_vesselness) across modes too, since frangi()'s output is
    # a pure function of img_rgb and does not depend on the per-mode
    # silhouette — only the cheap normalize+combine step does.
    lab_raw = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    vein_cache = {}

    for mode in FACTORY_MODES:
        mode = mode.strip()  # Safeguard against OCR spaces in config
        sil = SIL_MAP[mode]

        # Fix 1 (v8 LAB): use LAB-based detection for all modes (HSV kept as
        # fallback). mode_d -> soft confidence map; all others -> hard mask.
        #
        # v9 (Phase 3b): when the Symptom Teacher is deployed (module-level
        # _SYMPTOM_MODEL / _AE_MODEL populated by load_symptom_models() in
        # main()), it REPLACES the LAB pipeline as the symptom source for
        # ALL modes, not just mode_d — predict_symptom_mask() returns a
        # float32 [0,1] probability map at original resolution, the exact
        # same contract as compute_lab_soft_confidence(), so mode_d uses it
        # directly and the hard-mask modes (a/b/c) threshold it at 0.5.
        # This was the structural fix for the under/over-masking that eight
        # rounds of LAB threshold tuning (v1->v8) could not resolve: a model
        # trained on ~400-500 human-verified masks learns the chlorosis/
        # necrosis decision boundary directly instead of approximating it
        # with fixed color-space cutoffs. LAB remains as automatic fallback
        # if checkpoints are missing, and as an explicit comparison source
        # when FACTORY_SYMPTOM_COMPARE_LAB is True.
        if _SYMPTOM_MODEL is not None and _AE_MODEL is not None:
            from train_symptom_model import predict_symptom_mask
            symptom_prob = predict_symptom_mask(_SYMPTOM_MODEL, _AE_MODEL, img_rgb)
            symptom_prob = symptom_prob * sil.astype(np.float32)  # restrict to leaf silhouette
            symptom = (
                symptom_prob
                if mode == "mode_d"
                else (symptom_prob >= 0.5).astype(np.uint8) * 255
            )
        else:
            symptom = (
                compute_lab_soft_confidence(
                    img_rgb, sil, category, lab_raw=lab_raw, vein_cache=vein_cache
                )
                if mode == "mode_d"
                else compute_lab_hard_mask(
                    img_rgb, sil, category, lab_raw=lab_raw, vein_cache=vein_cache
                )
            )

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
                        symptom = symptom.astype(np.float32) * 0.3
                    else:
                        symptom = np.clip(
                            (symptom.astype(np.float32) * 0.3).astype(np.uint8),
                            0,
                            255,
                        )
                # ≥ 4 % → preserved at full strength (genuine early-stage signal)
            else:
                symptom = np.zeros_like(symptom)

        sym_binary = (
            (symptom >= 0.3).astype(np.uint8)
            if mode == "mode_d"
            else (symptom > 0).astype(np.uint8)
        )

        leaf_px = float(sil.sum())
        sev = (
            (float(sym_binary.sum()) / leaf_px) * 100.0
            if leaf_px >= 1 and weight is not None
            else -1.0
        )

        # Silhouette saved to disk: mode_c and mode_d store the continuous soft_prob
        # so the student can learn from the full probability map, not just the binary mask
        disk_sil = soft_prob if mode in ("mode_c", "mode_d") else sil.astype(np.float32)
        modes_data[mode] = {"silhouette": disk_sil, "symptom": symptom, "severity": sev}

    # Async Disk I/O
    for mode, data in modes_data.items():
        mode_dir = mode_dirs[mode]
        sil_out = cv2.resize(
            data["silhouette"].astype(np.float32),
            (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        np.save(str(mode_dir / f"{stem}_silhouette.npy"), sil_out)

        if mode == "mode_d":
            sym_out = cv2.resize(
                data["symptom"].astype(np.float32),
                (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            np.save(str(mode_dir / f"{stem}_symptom.npy"), sym_out)
        else:
            sym_out = cv2.resize(
                data["symptom"],
                (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imwrite(str(mode_dir / f"{stem}_symptom.png"), sym_out)

        (mode_dir / f"{stem}_sev.txt").write_text(str(round(data["severity"], 4)))
        (mode_dir / f"{stem}_weight.txt").write_text(str(result["weight"]))
        # NEW-3: CIMMYT agronomic grade alongside continuous severity
        grade = _sev_to_cimmyt_grade(data["severity"], category)
        (mode_dir / f"{stem}_grade.txt").write_text(str(grade))

    result["status"] = "processed"
    result.update({f"{m}_sev": round(modes_data[m]["severity"], 4) for m in modes_data})
    result.update(
        {
            f"{m}_grade": _sev_to_cimmyt_grade(modes_data[m]["severity"], category)
            for m in modes_data
        }
    )
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
            w.writerow([status, cnt, round(100 * cnt / max(total, 1), 2)])
    print(f"  Filter breakdown → {breakdown_path}")


def generate_factory_summary(report_rows):
    proc = [r for r in report_rows if r.get("status") == "processed"]
    if not proc:
        return

    df = pd.DataFrame(proc)
    summary_rows = []

    for mode in FACTORY_MODES:
        mode = mode.strip()
        sev_col = f"{mode}_sev"
        if sev_col not in df.columns:
            continue

        for cls in df["category"].unique():
            cls_df = df[df["category"] == cls]
            valid_sev = cls_df[sev_col].replace(-1.0, np.nan).dropna()

            # FIX: warn when mode_a + HEALTHY combination may produce inflated
            # false-positive rates. Otsu silhouette on yellow maize frequently
            # captures background, inflating leaf_px and deflating sym_area_frac,
            # allowing HEALTHY false positives to slip the three-band noise gate.
            if mode == "mode_a" and cls == "HEALTHY":
                pct_sym = (
                    round(100 * (valid_sev > 0).mean(), 1) if len(valid_sev) else 0.0
                )
                if pct_sym > 15.0:
                    print(
                        f"  [WARN] mode_a / HEALTHY: {pct_sym:.1f}% of images have non-zero "
                        f"severity. Otsu silhouette likely inflating leaf_px on yellow maize, "
                        f"suppressing the noise gate. Use mode_b silhouette as reference."
                    )

            summary_rows.append(
                {
                    "mode": mode,
                    "class": cls,
                    "n_total": len(cls_df),
                    "n_processed": len(cls_df[cls_df["status"] == "processed"]),
                    "pct_processed": round(
                        100
                        * len(cls_df[cls_df["status"] == "processed"])
                        / max(len(cls_df), 1),
                        1,
                    ),
                    "mean_severity": round(valid_sev.mean(), 2)
                    if len(valid_sev)
                    else "N/A",
                    "std_severity": round(valid_sev.std(), 2)
                    if len(valid_sev)
                    else "N/A",
                    "median_severity": round(valid_sev.median(), 2)
                    if len(valid_sev)
                    else "N/A",
                    "pct_symptomatic": round(100 * (valid_sev > 0).mean(), 1)
                    if len(valid_sev)
                    else 0.0,
                    "n_excluded": len(cls_df[cls_df["weight"] == -1]),
                    "pct_excluded": round(
                        100 * len(cls_df[cls_df["weight"] == -1]) / max(len(cls_df), 1),
                        1,
                    ),
                }
            )

    if summary_rows:
        summary_path = REPORTS_DIR / "factory_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=summary_rows[0].keys(), extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(summary_rows)
        print(f"  Factory summary → {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(
        "=" * 72,
        "\n  Yellow MAIze | Phase 4: Factory Pseudo-Label Generation\n" + "=" * 72,
    )

    if not GLOBAL_MANIFEST.exists() or not TIER1_MANIFEST.exists():
        print(
            "[FATAL] Missing manifest files. Run partition_dataset.py and sample_15000.py first."
        )
        return

    global_df = pd.read_csv(GLOBAL_MANIFEST)
    trainval = global_df[global_df["split"] != "test"].reset_index(drop=True)
    test_fnames = set(global_df[global_df["split"] == "test"]["filename"].tolist())
    tier1_fnames = set(
        Path(p).name for p in pd.read_csv(TIER1_MANIFEST)["source_path"].tolist()
    )

    mode_dirs = {m.strip(): PSEUDO_DIR / m.strip() for m in FACTORY_MODES}
    for d in mode_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  Loading models to RTX 5060...")
    bouncer_model, bouncer_thresh = load_bouncer()
    teacher_model = load_teacher()

    # v9 (Phase 3b): load Symptom Teacher + HealthyAE into module-level
    # singletons so process_single_image_cpu() (run via ThreadPoolExecutor
    # below) can read them without re-loading checkpoints per call. Threads
    # share the parent process's memory, unlike ProcessPoolExecutor workers,
    # so this is the cheapest correct way to make the models visible there.
    global _SYMPTOM_MODEL, _AE_MODEL
    _SYMPTOM_MODEL, _AE_MODEL = load_symptom_models()
    if _SYMPTOM_MODEL is None:
        print("  Symptom source: legacy LAB/HSV pipeline (v8)")
    else:
        print("  Symptom source: Symptom Teacher (v9, human-supervised)")
        if FACTORY_SYMPTOM_COMPARE_LAB:
            print("  FACTORY_SYMPTOM_COMPARE_LAB=True — LAB will also be computed "
                  "for comparison logging (slower).")

    dataset = FactoryDataset(trainval, tier1_fnames, test_fnames)
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        collate_fn=factory_collate,
        prefetch_factor=3,
    )

    report_rows = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        for i, batch in enumerate(loader):
            if batch is None:
                continue

            # 1. Batched Bouncer Inference (GPU)
            b_tensors = batch["b_tensor"].to(DEVICE, non_blocking=True)
            with torch.no_grad():
                b_probs = torch.sigmoid(bouncer_model(b_tensors)).view(-1)
                b_passed = b_probs >= bouncer_thresh

            passed_indices = torch.where(b_passed)[0].cpu().numpy()
            passed_set = set(
                passed_indices.tolist()
            )  # FIX: O(1) lookup instead of O(n) numpy scan

            for idx in range(len(b_passed)):
                if idx not in passed_set:
                    report_rows.append(
                        {
                            "filename": Path(batch["img_path"][idx]).stem,
                            "status": "filtered_bouncer",
                            "category": batch["category"][idx],
                        }
                    )

            if len(passed_indices) == 0:
                continue

            # 2. Batched Teacher Inference (GPU)
            t_tensors = torch.stack(
                [batch["t_tensor"][idx] for idx in passed_indices]
            ).to(DEVICE, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher_model(t_tensors)
                t_probs_batch = torch.sigmoid(t_logits).squeeze(1).cpu().numpy()

            # 3. Dispatch to CPU ThreadPool
            futures = []
            for i_rel, idx in enumerate(passed_indices):
                orig_h = int(batch["orig_h"][idx])
                orig_w = int(batch["orig_w"][idx])
                is_tier1 = bool(batch["is_tier1"][idx])

                # FIX: explicitly squeeze to 2D before resize.
                # Teacher output after sigmoid+squeeze(1) should be (H,W) but a
                # stray channel dim from the model or autocast produces (1,H,W)
                # or (H,W,1), both of which crash cv2.resize silently or corrupt.
                prob = t_probs_batch[i_rel]
                prob = np.squeeze(prob)  # removes all size-1 dimensions
                assert prob.ndim == 2, (
                    f"Teacher prob map has unexpected shape {prob.shape} "
                    f"after squeeze for image {batch['img_path'][idx]}"
                )

                if is_tier1 and batch["tier1_mask"][idx] is not None:
                    tier1_raw = np.squeeze(batch["tier1_mask"][idx])
                    soft_prob = cv2.resize(
                        tier1_raw.astype(np.float32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    soft_prob = cv2.resize(
                        prob.astype(np.float32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                args = (
                    batch["img_path"][idx],
                    batch["stem"][idx],
                    batch["category"][idx],
                    is_tier1,
                    orig_h,
                    orig_w,
                    batch["img_rgb"][idx],
                    soft_prob,
                    mode_dirs,
                )
                futures.append(executor.submit(process_single_image_cpu, args))

            for future in futures:
                report_rows.append(future.result())

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-3)
                eta = (len(loader) - i - 1) / rate
                print(
                    f"  [{i + 1:>4}/{len(loader)} Batches] | ETA: {eta / 60:.1f} min | Processed: {len(report_rows)}"
                )

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
