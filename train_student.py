"""
================================================================================
 train_student.py — Phase 5: Student Model Training
================================================================================
 PURPOSE:
   Train and compare Student models via two-stage ablation:

   Stage 1 (--stage 1): All 5 encoder variants trained on Factory Mode B.
                         Find the best encoder by composite test score.
   Stage 2 (--stage 2): Best encoder trained on all 4 Factory modes.
                         Find the best pseudo-label strategy.

 VARIANTS (Stage 1 — all on Mode B):
   V1: mobilenet_v2          — primary arch per thesis title; no-attention baseline
   V2: mobilenet_v2_cbam     — CBAM at skip junctions (expected winner)
   V3: mobilenet_v3_small    — ultra-compact; coord. attention
   V4: efficientnet_b0       — compound scaling; no-attention baseline for B0
   V6: efficientnet_b0_cbam  — tests if CBAM generalizes beyond MobileNetV2

   NOTE: mobilevit_xxs removed — TFLite self-attention einsum incompatibility
   makes it unable to participate in mobile composite scoring or deployment
   selection. Cite published MobileViT benchmarks in related work instead.

 ARCHITECTURE (all variants):
   - Shared encoder (variant-dependent) → MobileNetV2 / CBAM / etc.
   - Segmentation head: UNet decoder → float32 silhouette + symptom mask
   - Classification head: GAP → Dropout → Linear(→3) → HEALTHY/MSV/MLN
   - Severity head: GAP → Dropout → Linear(→1) → ReLU → clamp(0,1) → ×100

 KEY FEATURES:
   - Soft segmentation targets (float .npy, no binarization)
   - Asymmetric label smoothing (pathology-informed prior matrix)
   - Homoscedastic uncertainty loss (Kendall et al. 2018) — auto-weights tasks
   - Reliability-weighted sample loss (curriculum learning)
   - Composite checkpoint criterion: 0.5×mIoU + 0.35×MSV_F1 + 0.15×(1−NormMAE)
   - Full held-out test evaluation → student_test_metrics_{variant}_{mode}.csv
   - Per-class classification report + confusion matrix

 USAGE:
   python train_student.py --stage 1
   python train_student.py --stage 2 --encoder mobilenet_v2_cbam
================================================================================
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pandas as pd
import segmentation_models_pytorch as smp
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, roc_auc_score, cohen_kappa_score,
)
from scipy.stats import pearsonr as _pearsonr

from image_utils import load_image_rgb   # EXIF correction, no CLAHE for training
from scripts.safe_collate import safe_collate, reset_skip_counter

from config import (
    SEED,
    GLOBAL_MANIFEST, TIER1_MANIFEST,
    PSEUDO_DIR, STUDENT_CKPT_DIR, LOGS_DIR, REPORTS_DIR,
    CLASSES, CLASS_TO_IDX,
    STUDENT_IMG_SIZE, STUDENT_BATCH_SIZE, STUDENT_NUM_WORKERS,
    STUDENT_PHASE1_EPOCHS, STUDENT_PHASE1_LR, STUDENT_PHASE1_PATIENCE,
    STUDENT_PHASE2_EPOCHS, STUDENT_PHASE2_LR, STUDENT_PHASE2_LR_MIN,
    STUDENT_PHASE2_PATIENCE,
    STUDENT_WEIGHT_DECAY, STUDENT_DROPOUT,
    STUDENT_CKPT_W_MIOU, STUDENT_CKPT_W_MSV_F1, STUDENT_CKPT_W_MLN_F1, STUDENT_CKPT_W_MAE,
    STUDENT_CKPT_W_MSV_F1_EARLY, STUDENT_CKPT_W_MSV_F1_MID, STUDENT_CKPT_W_MSV_F1_SEVERE,
    STUDENT_MAX_SEVERITY,
    STUDENT_LABEL_SMOOTHING, ASYMMETRIC_PRIOR,
    STUDENT_VARIANTS, STUDENT_BEST_VARIANT, STUDENT_FACTORY_MODE,
    FACTORY_MODES,
    CBAM_SPATIAL_KERNEL,
)


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

def set_seeds(seed: int) -> None:
    """
    Set all random seeds for full reproducibility.
    cudnn.deterministic=True, benchmark=False is a deliberate choice —
    benchmark=True would be ~15% faster but non-deterministic.
    See config.py CUDNN_DETERMINISTIC / CUDNN_BENCHMARK.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # reproducibility over speed
    torch.backends.cudnn.benchmark     = False  # see config.CUDNN_BENCHMARK


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# CBAM MODULE
# ══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 1), channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap_out = self.mlp(self.gap(x))
        gmp_out = self.mlp(self.gmp(x))
        scale   = self.sigmoid(gap_out + gmp_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = CBAM_SPATIAL_KERNEL):
        super().__init__()
        pad = kernel_size // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        cat     = torch.cat([avg_out, max_out], dim=1)
        scale   = self.sigmoid(self.conv(cat))
        return x * scale


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al. 2018).
    Applied at UNet skip connections to focus on MSV streak features.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channel  = ChannelAttention(channels)
        self.spatial  = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class StudentModel(nn.Module):
    """
    Multi-task Student model:
      - Shared encoder (configurable)
      - UNet decoder → leaf silhouette + symptom mask (2-channel seg output)
      - Classification head → HEALTHY/MSV/MLN
      - Severity head → continuous 0–1 (×100 at inference)
    """

    def __init__(self, encoder_name: str, use_cbam: bool = False):
        super().__init__()
        self.use_cbam    = use_cbam
        self.encoder_name = encoder_name

        # Clean encoder name for smp
        smp_encoder = encoder_name.replace("_cbam", "")

        self.unet = smp.Unet(
            encoder_name=smp_encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=2,          # Ch0: leaf silhouette | Ch1: symptom mask
            activation=None,
        )

        # CBAM modules for skip connections (one per decoder stage)
        if use_cbam:
            enc_channels = self.unet.encoder.out_channels
            # enc_channels[0] is the raw input channel count (3), not a
            # real feature map.  _apply_cbam_to_features keeps features[0]
            # as-is and iterates features[1:], so block[i] is applied to
            # features[i+1].  We must therefore start from enc_channels[1]
            # (first real encoder stage) and stop before the bottleneck [-1].
            skip_channels = [c for c in enc_channels[1:-1] if c > 0]
            self.cbam_blocks = nn.ModuleList([
                CBAMBlock(c) for c in skip_channels
            ])
            
        # Get encoder output channels for classification/severity heads
        enc_out_ch = self.unet.encoder.out_channels[-1]

        # Classification head
        self.cls_gap  = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(STUDENT_DROPOUT),
            nn.Linear(enc_out_ch, len(CLASSES)),
        )

        # Severity head — ReLU + clamp (NOT Sigmoid, avoids ceiling issue)
        self.sev_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(STUDENT_DROPOUT),
            nn.Linear(enc_out_ch, 1),
            nn.ReLU(),
        )

    def _apply_cbam_to_features(self, features: list) -> list:
        """
        Apply CBAM blocks to encoder skip features before decoder upsampling.
        Called explicitly in forward() — avoids monkey-patching which breaks
        torch.save / checkpoint serialisation.

        features[0] = encoder bottleneck (deepest)
        features[1:] = skip connections (shallowest to deepest)
        """
        if not self.use_cbam:
            return features
        result = [features[0]]
        for i, feat in enumerate(features[1:]):
            if feat is not None and i < len(self.cbam_blocks):
                result.append(self.cbam_blocks[i](feat))
            else:
                result.append(feat)
        return result

    def forward(self, x: torch.Tensor) -> tuple:
        # Encoder — returns list of feature maps (bottleneck last)
        features  = self.unet.encoder(x)
        deep_feat = features[-1]      # bottleneck — shared for cls + sev heads

        # Apply CBAM to skip connections before decoder (safe, no monkey-patch)
        features_cbam = self._apply_cbam_to_features(features)

        # Segmentation decoder — pass as list (safe across all smp encoders)
        decoder_out = self.unet.decoder(features_cbam)
        seg_logits  = self.unet.segmentation_head(decoder_out)

        # Classification head
        pooled  = self.cls_gap(deep_feat)
        cls_out = self.cls_head(pooled)

        # Severity head — ReLU + clamp, no Sigmoid ceiling
        sev_out = self.sev_head(pooled).squeeze(1).clamp(0.0, 1.0)

        return seg_logits, cls_out, sev_out


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class AsymmetricLabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with pathology-informed asymmetric smoothing.
    Prior matrix rows = true class, cols = [HEALTHY, MSV, MLN].
    (Grounded in Cruz et al. 2024, Mushayi et al. 2025)
    """
    def __init__(self, prior: list[list[float]]):
        super().__init__()
        self.register_buffer(
            "prior",
            torch.tensor(prior, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        log_probs    = F.log_softmax(logits, dim=1)
        smooth_targs = self.prior[targets]         # (B, C)
        loss         = -(smooth_targs * log_probs).sum(dim=1)
        return loss.mean()


class HomoscedasticUncertaintyLoss(nn.Module):
    """
    Multi-task loss with learnable homoscedastic uncertainty weights.
    Kendall et al. (2018) — "Multi-Task Learning Using Uncertainty to
    Weigh Losses for Scene Geometry and Semantics"

    L = exp(−s1)·L_seg + s1 + exp(−s2)·L_cls + s2 + exp(−s3)·L_sev + s3
    where s1, s2, s3 are learned log-variance parameters.
    """
    def __init__(self):
        super().__init__()
        # Initialize log-vars near 0 (equal initial weighting)
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.log_var_sev = nn.Parameter(torch.tensor(0.0))

    def forward(self, l_seg: torch.Tensor,
                l_cls: torch.Tensor,
                l_sev: torch.Tensor) -> torch.Tensor:
        w_seg = torch.exp(-self.log_var_seg)
        w_cls = torch.exp(-self.log_var_cls)
        w_sev = torch.exp(-self.log_var_sev)

        loss = (w_seg * l_seg + self.log_var_seg +
                w_cls * l_cls + self.log_var_cls +
                w_sev * l_sev + self.log_var_sev)
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class StudentDataset(Dataset):
    """
    Multi-task dataset for Student training.
    Loads:
      - Image
      - Soft silhouette .npy (float32) — segmentation target Ch0
      - Symptom mask .npy or .png     — segmentation target Ch1
      - Class label (int)
      - Severity (float in [0,1])
      - Sample weight (float)
    """

    def __init__(self, samples: list[dict], mode: str, transform=None):
        self.samples   = samples
        self.mode      = mode
        self.transform = transform
        self.mode_dir  = PSEUDO_DIR / mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row      = self.samples[idx]
        stem     = row["stem"]
        category = row["category"]
        weight   = float(row.get("weight", 1.0))

        # ── Image — EXIF-corrected load with corrupt guard ────────────────
        img_path = Path(row["source_path"])
        if not img_path.exists():
            return None   # source file missing — safe_collate skips
        img_rgb = load_image_rgb(img_path)   # EXIF correction + truncation guard
        if img_rgb is None:
            return None   # corrupt/truncated — safe_collate skips
        h, w     = img_rgb.shape[:2]

        # ── Silhouette (soft float .npy) ───────────────────────────────────
        sil_path = self.mode_dir / f"{stem}_silhouette.npy"
        if sil_path.exists():
            sil = np.load(str(sil_path)).astype(np.float32)
        else:
            sil = np.zeros((h, w), dtype=np.float32)

        sil = cv2.resize(sil, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Symptom mask ────────────────────────────────────────────────────
        if self.mode == "mode_d":
            sym_path = self.mode_dir / f"{stem}_symptom.npy"
            sym = np.load(str(sym_path)).astype(np.float32) if sym_path.exists() \
                  else np.zeros((h, w), dtype=np.float32)
        else:
            sym_path = self.mode_dir / f"{stem}_symptom.png"
            if sym_path.exists():
                sym_img = cv2.imread(str(sym_path), cv2.IMREAD_GRAYSCALE)
                sym = (sym_img / 255.0).astype(np.float32) \
                    if sym_img is not None \
                    else np.zeros((h, w), dtype=np.float32)
            else:
                sym = np.zeros((h, w), dtype=np.float32)

        sym = cv2.resize(sym, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Severity ────────────────────────────────────────────────────────
        sev_path = self.mode_dir / f"{stem}_sev.txt"
        if sev_path.exists():
            sev_raw = float(sev_path.read_text().strip())
        else:
            sev_raw = -1.0
        sev_norm = max(0.0, min(1.0, sev_raw / 100.0)) if sev_raw >= 0 else -1.0

        # ── CIMMYT agronomic grade (written by factory_master._sev_to_cimmyt_grade) ──
        # MSV: 1/3/5/7/9 ; MLN: 1-5 ; HEALTHY: 0 ; -1 = excluded/missing.
        grade_path = self.mode_dir / f"{stem}_grade.txt"
        if grade_path.exists():
            try:
                grade = int(grade_path.read_text().strip())
            except Exception:
                grade = -1
        else:
            grade = -1

        # ── Class label ─────────────────────────────────────────────────────
        cls_idx = CLASS_TO_IDX[category]

        # ── Augmentation ───────────────────────────────────────────────────
        # Stack sil and sym as extra targets
        extra = np.stack([sil, sym], axis=-1)   # H×W×2
        if self.transform:
            aug    = self.transform(image=img_rgb, mask=extra)
            img_t  = aug["image"]                      # C×H×W tensor
            extra_t= aug["mask"]                       # H×W×2 tensor
            sil_t  = extra_t[:, :, 0]
            sym_t  = extra_t[:, :, 1]
        else:
            img_t  = torch.tensor(img_rgb.transpose(2,0,1), dtype=torch.float32)/255.
            sil_t  = torch.tensor(sil, dtype=torch.float32)
            sym_t  = torch.tensor(sym, dtype=torch.float32)

        # Stack segmentation targets: 2×H×W
        seg_target = torch.stack([
            sil_t if isinstance(sil_t, torch.Tensor)
            else torch.tensor(sil_t, dtype=torch.float32),
            sym_t if isinstance(sym_t, torch.Tensor)
            else torch.tensor(sym_t, dtype=torch.float32),
        ], dim=0)

        return {
            "image":     img_t,
            "seg":       seg_target,
            "cls":       torch.tensor(cls_idx, dtype=torch.long),
            "sev":       torch.tensor(max(sev_norm, 0.0), dtype=torch.float32),
            "weight":    torch.tensor(weight, dtype=torch.float32),
            "valid_sev": torch.tensor(sev_norm >= 0.0, dtype=torch.bool),
            "grade":     torch.tensor(grade, dtype=torch.long),
        }


def make_student_transforms(img_size: int, is_train: bool):
    if is_train:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=20,
                val_shift_limit=10, p=0.2),
            A.RandomShadow(p=0.2),          # tropical domain adaptation
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # lighting gamma robustness
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})


def build_sample_list(split: str, mode: str) -> list[dict]:
    """Build sample list from global manifest filtered by split and mode."""
    global_df = pd.read_csv(GLOBAL_MANIFEST)
    split_df  = global_df[global_df["split"] == split].copy()
    mode_dir  = PSEUDO_DIR / mode

    samples = []
    for _, row in split_df.iterrows():
        src   = Path(row["source_path"])
        stem  = src.stem
        cat   = row["category"]

        # Check severity file (skip sentinel -1 only for excluded bracket)
        sev_path = mode_dir / f"{stem}_sev.txt"
        if sev_path.exists():
            sev_raw = float(sev_path.read_text().strip())
            if sev_raw == -1.0:
                continue   # excluded (coverage < 15%)
        else:
            continue

        # Get reliability weight (written by factory_master.py, 0.0–1.0)
        wt_path = mode_dir / f"{stem}_weight.txt"
        weight  = float(wt_path.read_text().strip()) if wt_path.exists() else 1.0

        # Guard: also skip CIMMYT grade == -1 if grade files are ever introduced
        # into curriculum training. The -1 sentinel means the same as sev == -1.0
        # (excluded / insufficient coverage). Filtering here ensures the sample
        # list never contains grade=-1 even if a future curriculum loader reads it.
        grade_path = mode_dir / f"{stem}_grade.txt"
        if grade_path.exists():
            try:
                if int(grade_path.read_text().strip()) == -1:
                    continue   # excluded — same reason as sev == -1.0
            except ValueError:
                pass   # malformed grade file — include anyway

        samples.append({
            "source_path":       str(src),
            "stem":              stem,
            "category":          cat,
            "split":             split,
            "weight":            weight,
            "reliability_weight": weight,   # alias used by WeightedRandomSampler
        })

    return samples


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def msv_grade_to_tier(grade: int) -> str | None:
    """
    Map a CIMMYT MSV grade (1/3/5/7/9) to its severity tier, aligned with
    CIMMYT_MSV_BRACKETS in config.py:
      grade 1-3 (<25% area)  → "early"  (clinically critical — early detection)
      grade 3-5 (25-50%)     → "mid"
      grade 5-9 (>50%)       → "severe"
    Grade 3 and 5 sit on tier boundaries; assigned to the lower (earlier,
    higher-priority) tier per CIMMYT_MSV_BRACKETS upper-exclusive convention.
    Returns None for grade <= 0 (HEALTHY / excluded) — not an MSV sample.
    """
    if grade <= 0:
        return None
    if grade <= 3:
        return "early"
    if grade <= 5:
        return "mid"
    return "severe"


def compute_msv_f1_by_tier(targets: list[int], preds: list[int],
                            grades: list[int], msv_idx: int) -> dict:
    """
    Grade-stratified MSV F1 (CIMMYT-aligned), consuming STUDENT_CKPT_W_MSV_F1_EARLY/
    MID/SEVERE from config.py. Surfaces early-stage MSV detection — the
    clinically critical case — instead of letting it get averaged away in the
    aggregate MSV F1 (see config.py STUDENT_CKPT_W_MSV_F1 comment, v8 P8).

    A sample contributes to tier T's binary "is this early/mid/severe MSV"
    task only if it's a genuine member of tier T (ground-truth MSV graded
    into T — true positive/false negative side) or a true negative/false
    positive for that task (i.e. not MSV at all, or MSV but in a *different*
    tier AND mis-predicted as MSV — a real confusion). Samples that are MSV
    in another tier and correctly predicted as MSV are excluded from this
    tier's set entirely: they're neither a tier-T detection nor a tier-T
    confusion, so counting them as a false positive here would wrongly
    penalize the model for correctly spotting severe/mid MSV while scoring
    the early tier.
    """
    tier_f1 = {}
    for tier in ("early", "mid", "severe"):
        y_true, y_pred = [], []
        for t, p, g in zip(targets, preds, grades):
            is_tier_member = (t == msv_idx) and (msv_grade_to_tier(g) == tier)
            is_other_tier_msv_correct = (
                t == msv_idx and msv_grade_to_tier(g) != tier and p == msv_idx
            )
            if is_other_tier_msv_correct:
                continue  # not relevant to this tier's binary task
            y_true.append(1 if is_tier_member else 0)
            y_pred.append(1 if p == msv_idx else 0)
        if any(y_true):
            tier_f1[tier] = f1_score(y_true, y_pred, zero_division=0)
        else:
            tier_f1[tier] = 0.0
    return tier_f1


def compute_composite_score(miou: float, msv_f1_early: float, msv_f1_mid: float,
                             msv_f1_severe: float, mln_f1: float,
                             norm_mae: float) -> float:
    """
    TRAINING composite checkpoint criterion (used during training for checkpoint saving).
    Formula: 0.40×mIoU + 0.20×MSV_F1_early + 0.10×MSV_F1_mid + 0.05×MSV_F1_severe
             + 0.15×MLN_F1 + 0.10×(1−NormMAE)

    Grade-stratified MSV terms (v8 P8) replace the flat STUDENT_CKPT_W_MSV_F1
    (now 0.00 — dead weight, see config.py), so early-stage MSV detection is
    rewarded ~4x more per-tier than the severe tier, since early detection is
    the clinically actionable case and severe MSV is visually unambiguous
    even to a weak model. Sum of MSV tier weights + flat weight == 0.35,
    preserving the original checkpoint-weight budget split with mIoU/MLN/MAE.

    NOTE: This is the QUALITY composite — optimised for segmentation + classification
    accuracy during training. The MOBILE composite (which adds speed, ROC-AUC, and
    model size) is computed post-training in select_best_pipeline.py:

        mobile_composite = MSV_F1×0.32 + MSV_ROC_AUC×0.18 + mIoU×0.18
                         + speed×0.16 + MLN_F1×0.08 + sev×0.05 + size×0.03
    """
    return (STUDENT_CKPT_W_MIOU         * miou
          + STUDENT_CKPT_W_MSV_F1_EARLY * msv_f1_early
          + STUDENT_CKPT_W_MSV_F1_MID   * msv_f1_mid
          + STUDENT_CKPT_W_MSV_F1_SEVERE * msv_f1_severe
          + STUDENT_CKPT_W_MSV_F1       * 0.0  # retained at 0.0 — see config.py note
          + STUDENT_CKPT_W_MLN_F1       * mln_f1
          + STUDENT_CKPT_W_MAE          * (1.0 - norm_mae))


def seg_stats_from_logits(logits: torch.Tensor,
                           targets: torch.Tensor,
                           ch: int = 0) -> dict:
    """
    Return raw TP/FP/FN/TN counts for one segmentation channel.
    Accumulate these globally across batches, then compute metrics once
    at epoch end — this gives TRUE global mIoU, not mean-of-batch-IoUs.
    """
    prob = torch.sigmoid(logits[:, ch])
    pred = (prob >= 0.5).long()
    tgt  = (targets[:, ch] >= 0.5).long()
    return {
        "tp": ((pred==1)&(tgt==1)).sum().item(),
        "fp": ((pred==1)&(tgt==0)).sum().item(),
        "fn": ((pred==0)&(tgt==1)).sum().item(),
        "tn": ((pred==0)&(tgt==0)).sum().item(),
    }


def seg_metrics_from_stats(tp: float, fp: float,
                            fn: float, tn: float) -> dict:
    """
    Compute all segmentation metrics from globally accumulated TP/FP/FN/TN.
    Called once per epoch after accumulation — produces TRUE global metrics.
    """
    eps  = 1e-7
    iou  = tp / (tp + fp + fn + eps)
    dice = 2*tp / (2*tp + fp + fn + eps)
    rec  = tp / (tp + fn + eps)           # Sensitivity / Recall
    prec = tp / (tp + fp + eps)           # Precision
    spec = tn / (tn + fp + eps)           # Specificity
    return {
        "iou": iou, "dice": dice,
        "recall": rec, "precision": prec, "specificity": spec,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer,
                    seg_criterion, cls_criterion,
                    unc_loss, device) -> float:
    # NOTE: sev_criterion removed — MSE for severity is computed inline
    # via F.mse_loss inside this function. Do not re-add as a parameter.
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        imgs     = batch["image"].to(device)
        seg_tgt  = batch["seg"].to(device)
        cls_tgt  = batch["cls"].to(device)
        sev_tgt  = batch["sev"].to(device)
        weights  = batch["weight"].to(device)
        valid_sv = batch["valid_sev"].to(device)

        optimizer.zero_grad()
        n_samples += imgs.size(0)
        seg_logits, cls_out, sev_out = model(imgs)

        # Segmentation loss (Dice on both channels, averaged)
        l_seg0 = seg_criterion(seg_logits[:, :1], seg_tgt[:, :1])
        l_seg1 = seg_criterion(seg_logits[:, 1:], seg_tgt[:, 1:])
        l_seg  = (l_seg0 + l_seg1) / 2.0

        # Classification loss (asymmetric smoothing)
        l_cls = cls_criterion(cls_out, cls_tgt)

        # Severity loss (reliability-weighted MSE, valid samples only)
        if valid_sv.any():
            sev_pred_v = sev_out[valid_sv]
            sev_tgt_v  = sev_tgt[valid_sv]
            wt_v       = weights[valid_sv]
            l_sev = (wt_v * F.mse_loss(sev_pred_v, sev_tgt_v,
                                        reduction="none")).mean()
        else:
            l_sev = torch.tensor(0.0, device=device)

        # Homoscedastic uncertainty weighting
        loss = unc_loss(l_seg, l_cls, l_sev)
        loss.backward()
        # Gradient clipping — prevents log_var spikes in uncertainty loss
        # and encoder instability at Phase 2 unfreeze
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(unc_loss.parameters()),
            max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def validate(model, loader, seg_criterion, cls_criterion,
             unc_loss, device) -> dict:
    # NOTE: sev_criterion removed — MSE for severity computed inline via F.mse_loss.
    model.eval()

    total_loss  = 0.0
    # Global seg accumulators — Ch0 (silhouette) and Ch1 (symptom)
    s0 = {"tp":0.,"fp":0.,"fn":0.,"tn":0.}
    s1 = {"tp":0.,"fp":0.,"fn":0.,"tn":0.}
    # Severity accumulators
    sev_abs_err  = 0.0
    sev_sq_err   = 0.0
    sev_tgt_sum  = 0.0
    sev_tgt_sq   = 0.0
    sev_mape_num = 0.0   # sum of |pred-true|/max(|true|,eps) for MAPE
    sev_pred_list = []   # for Pearson r
    sev_true_list = []
    n_sev_valid  = 0
    n_samples    = 0
    all_cls_preds   = []
    all_cls_targets = []
    all_grades      = []

    for batch in loader:
        if batch is None:
            continue
        imgs     = batch["image"].to(device)
        seg_tgt  = batch["seg"].to(device)
        cls_tgt  = batch["cls"].to(device)
        sev_tgt  = batch["sev"].to(device)
        weights  = batch["weight"].to(device)
        valid_sv = batch["valid_sev"].to(device)
        grades   = batch["grade"]

        seg_logits, cls_out, sev_out = model(imgs)

        l_seg0 = seg_criterion(seg_logits[:, :1], seg_tgt[:, :1])
        l_seg1 = seg_criterion(seg_logits[:, 1:], seg_tgt[:, 1:])
        l_seg  = (l_seg0 + l_seg1) / 2.0
        l_cls  = cls_criterion(cls_out, cls_tgt)

        if valid_sv.any():
            sev_pred_v = sev_out[valid_sv]
            sev_tgt_v  = sev_tgt[valid_sv]
            wt_v       = weights[valid_sv]
            l_sev = (wt_v * F.mse_loss(sev_pred_v, sev_tgt_v,
                                        reduction="none")).mean()
            diff_pct     = (sev_pred_v - sev_tgt_v) * 100.0
            sev_abs_err += diff_pct.abs().sum().item()
            sev_sq_err  += (diff_pct ** 2).sum().item()
            sev_tgt_sum += (sev_tgt_v * 100.0).sum().item()
            sev_tgt_sq  += ((sev_tgt_v * 100.0) ** 2).sum().item()
            tgt_pct_v    = sev_tgt_v * 100.0
            sev_mape_num += (diff_pct.abs() / (tgt_pct_v.abs() + 1e-6)).sum().item()
            sev_pred_list.extend((sev_out[valid_sv] * 100.0).cpu().tolist())
            sev_true_list.extend(tgt_pct_v.cpu().tolist())
            n_sev_valid += valid_sv.sum().item()
        else:
            l_sev = torch.tensor(0.0, device=device)

        loss = unc_loss(l_seg, l_cls, l_sev)
        total_loss += loss.item() * imgs.size(0)

        # Global seg stat accumulation (Ch0=silhouette, Ch1=symptom)
        st0 = seg_stats_from_logits(seg_logits, seg_tgt, ch=0)
        st1 = seg_stats_from_logits(seg_logits, seg_tgt, ch=1)
        for k in s0: s0[k] += st0[k]
        for k in s1: s1[k] += st1[k]
        n_samples += imgs.size(0)

        # Accumulate classification predictions
        cls_preds = cls_out.argmax(dim=1).cpu().numpy().tolist()
        cls_tgts  = cls_tgt.cpu().numpy().tolist()
        all_cls_preds.extend(cls_preds)
        all_cls_targets.extend(cls_tgts)
        all_grades.extend(grades.numpy().tolist())

    # Classification report — per-class P/R/F1, macro, weighted, MCC
    report  = classification_report(
        all_cls_targets, all_cls_preds,
        target_names=CLASSES, output_dict=True, zero_division=0)
    mcc     = matthews_corrcoef(all_cls_targets, all_cls_preds)
    msv_f1  = report.get("MSV", {}).get("f1-score", 0.0)
    cls_acc = report.get("accuracy", 0.0)
    macro_f1   = report.get("macro avg",    {}).get("f1-score", 0.0)
    weighted_f1= report.get("weighted avg", {}).get("f1-score", 0.0)

    # Global segmentation metrics (TRUE global mIoU, not mean-of-batches)
    seg0 = seg_metrics_from_stats(**s0)
    seg1 = seg_metrics_from_stats(**s1)

    # Severity metrics
    sev_mae   = sev_abs_err / max(n_sev_valid, 1)
    sev_mse   = sev_sq_err  / max(n_sev_valid, 1)
    sev_rmse  = sev_mse ** 0.5
    sev_mape  = (sev_mape_num / max(n_sev_valid, 1)) * 100.0  # as %
    # R²: 1 - SS_res/SS_tot
    ss_tot    = sev_tgt_sq - (sev_tgt_sum ** 2) / max(n_sev_valid, 1)
    sev_r2    = 1.0 - (sev_sq_err / max(ss_tot, 1e-7))
    # Pearson r
    try:
        sev_pearson = float(_pearsonr(sev_true_list, sev_pred_list)[0])                       if len(sev_true_list) > 1 else 0.0
    except Exception:
        sev_pearson = 0.0
    norm_mae  = sev_mae / STUDENT_MAX_SEVERITY

    mln_f1    = report.get("MLN", {}).get("f1-score", 0.0)

    # CIMMYT grade-stratified MSV F1 (early/mid/severe) — see config.py
    # STUDENT_CKPT_W_MSV_F1_EARLY/MID/SEVERE.
    msv_idx = CLASSES.index("MSV")
    msv_tier_f1 = compute_msv_f1_by_tier(
        all_cls_targets, all_cls_preds, all_grades, msv_idx)

    composite = compute_composite_score(
        seg0["iou"],
        msv_tier_f1["early"], msv_tier_f1["mid"], msv_tier_f1["severe"],
        mln_f1, norm_mae,
    )

    return {
        "loss":         total_loss / max(n_samples, 1),
        # Silhouette seg (Ch0)
        "mIoU":         seg0["iou"],   "dice":    seg0["dice"],
        "seg_recall":   seg0["recall"],"seg_prec":seg0["precision"],
        "seg_spec":     seg0["specificity"],
        # Symptom seg (Ch1)
        "sym_iou":      seg1["iou"],   "sym_dice":seg1["dice"],
        "sym_recall":   seg1["recall"],"sym_prec":seg1["precision"],
        "sym_spec":     seg1["specificity"],
        # Classification
        "cls_acc":      cls_acc,  "msv_f1":     msv_f1,  "mln_f1": mln_f1,
        "macro_f1":     macro_f1, "weighted_f1":weighted_f1, "mcc": mcc,
        # CIMMYT grade-stratified MSV F1
        "msv_f1_early":  msv_tier_f1["early"],
        "msv_f1_mid":    msv_tier_f1["mid"],
        "msv_f1_severe": msv_tier_f1["severe"],
        # Severity regression
        "sev_mae":      sev_mae,  "sev_mse":    sev_mse,
        "sev_rmse":     sev_rmse, "sev_mape":   sev_mape,
        "sev_r2":       sev_r2,   "sev_pearson":sev_pearson,
        "composite":    composite,
        "report":    report, "preds": all_cls_preds, "targets": all_cls_targets,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test_split(model, mode: str, variant: str, device) -> dict:
    """
    Held-out test evaluation. Source for ALL thesis performance tables.
    Outputs:
      logs/student_test_metrics_{variant}_{mode}.csv   — all metrics
      logs/student_confusion_{variant}_{mode}.csv      — 3×3 confusion matrix
      logs/student_inference_latency_{variant}.csv     — CPU timing
    """
    import time
    from sklearn.metrics import matthews_corrcoef
    print(f"\n  Running held-out test evaluation ...")
    t_eval_start = time.time()

    test_samples = build_sample_list("test", mode)
    if not test_samples:
        print("  [WARN] No test samples found.")
        return {}

    test_tf  = make_student_transforms(STUDENT_IMG_SIZE, is_train=False)
    test_ds  = StudentDataset(test_samples, mode, test_tf)
    test_loader = DataLoader(
        test_ds, batch_size=STUDENT_BATCH_SIZE,
        shuffle=False, num_workers=STUDENT_NUM_WORKERS, pin_memory=True,
        collate_fn=safe_collate)

    # Global accumulators
    s0 = {"tp":0.,"fp":0.,"fn":0.,"tn":0.}   # silhouette
    s1 = {"tp":0.,"fp":0.,"fn":0.,"tn":0.}   # symptom
    sev_abs_err  = sev_sq_err = 0.0
    sev_tgt_sum  = sev_tgt_sq = 0.0
    sev_mape_num = 0.0
    sev_pred_list = []
    sev_true_list = []
    n_sev_valid  = n_samples = 0
    all_preds    = []
    all_targets  = []
    all_probs    = []   # softmax probabilities for ROC-AUC
    all_grades   = []

    for batch in test_loader:
        if batch is None:
            continue
        imgs     = batch["image"].to(device)
        seg_tgt  = batch["seg"].to(device)
        cls_tgt  = batch["cls"].to(device)
        sev_tgt  = batch["sev"].to(device)
        valid_sv = batch["valid_sev"].to(device)
        grades   = batch["grade"]

        seg_logits, cls_out, sev_out = model(imgs)

        st0 = seg_stats_from_logits(seg_logits, seg_tgt, ch=0)
        st1 = seg_stats_from_logits(seg_logits, seg_tgt, ch=1)
        for k in s0: s0[k] += st0[k]
        for k in s1: s1[k] += st1[k]
        n_samples += imgs.size(0)

        if valid_sv.any():
            diff_pct     = (sev_out[valid_sv] - sev_tgt[valid_sv]) * 100.0
            sev_abs_err  += diff_pct.abs().sum().item()
            sev_sq_err   += (diff_pct**2).sum().item()
            sev_tgt_sum  += (sev_tgt[valid_sv]*100).sum().item()
            sev_tgt_sq   += ((sev_tgt[valid_sv]*100)**2).sum().item()
            tgt_pct_v     = sev_tgt[valid_sv] * 100.0
            sev_mape_num += (diff_pct.abs() / (tgt_pct_v.abs() + 1e-6)).sum().item()
            sev_pred_list.extend((sev_out[valid_sv]*100.0).cpu().tolist())
            sev_true_list.extend(tgt_pct_v.cpu().tolist())
            n_sev_valid  += valid_sv.sum().item()

        probs = torch.softmax(cls_out, dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(cls_out.argmax(1).cpu().numpy().tolist())
        all_targets.extend(cls_tgt.cpu().numpy().tolist())
        all_grades.extend(grades.numpy().tolist())

    t_eval_end = time.time()
    eval_duration_s = round(t_eval_end - t_eval_start, 1)

    # Segmentation metrics (global)
    seg0 = seg_metrics_from_stats(**s0)
    seg1 = seg_metrics_from_stats(**s1)

    # Classification metrics
    report      = classification_report(
        all_targets, all_preds,
        target_names=CLASSES, output_dict=True, zero_division=0)
    cm          = confusion_matrix(all_targets, all_preds)
    mcc         = matthews_corrcoef(all_targets, all_preds)
    macro_f1    = report.get("macro avg",    {}).get("f1-score", 0.0)
    weighted_f1 = report.get("weighted avg", {}).get("f1-score", 0.0)

    # Severity regression
    sev_mae     = sev_abs_err / max(n_sev_valid, 1)
    sev_mse     = sev_sq_err  / max(n_sev_valid, 1)
    sev_rmse    = sev_mse ** 0.5
    sev_mape    = (sev_mape_num / max(n_sev_valid, 1)) * 100.0
    ss_tot      = sev_tgt_sq - (sev_tgt_sum**2) / max(n_sev_valid, 1)
    sev_r2      = 1.0 - (sev_sq_err / max(ss_tot, 1e-7))
    try:
        sev_pearson = float(_pearsonr(sev_true_list, sev_pred_list)[0])                       if len(sev_true_list) > 1 else 0.0
    except Exception:
        sev_pearson = 0.0

    msv_f1_test = report.get("MSV", {}).get("f1-score", 0.0)
    mln_f1_test = report.get("MLN", {}).get("f1-score", 0.0)
    norm_mae_test = sev_mae / STUDENT_MAX_SEVERITY

    # CIMMYT grade-stratified MSV F1 (early/mid/severe)
    msv_idx_t = CLASSES.index("MSV")
    msv_tier_f1_test = compute_msv_f1_by_tier(
        all_targets, all_preds, all_grades, msv_idx_t)

    composite = compute_composite_score(
        seg0["iou"],
        msv_tier_f1_test["early"], msv_tier_f1_test["mid"], msv_tier_f1_test["severe"],
        mln_f1_test, norm_mae_test,
    )

    # ROC-AUC: MSV-vs-rest, per-class, and macro OvR
    probs_arr = np.array(all_probs)
    try:
        msv_idx     = CLASSES.index("MSV")
        healthy_idx = CLASSES.index("HEALTHY")
        mln_idx     = CLASSES.index("MLN")
        msv_roc_auc = round(float(roc_auc_score(
            [1 if t == msv_idx else 0 for t in all_targets],
            probs_arr[:, msv_idx])), 4)
        healthy_roc_auc = round(float(roc_auc_score(
            [1 if t == healthy_idx else 0 for t in all_targets],
            probs_arr[:, healthy_idx])), 4)
        mln_roc_auc = round(float(roc_auc_score(
            [1 if t == mln_idx else 0 for t in all_targets],
            probs_arr[:, mln_idx])), 4)
        macro_roc_auc = round(float(roc_auc_score(
            all_targets, probs_arr, multi_class="ovr",
            average="macro", labels=list(range(len(CLASSES))))), 4)
    except Exception:
        msv_roc_auc = healthy_roc_auc = mln_roc_auc = macro_roc_auc = -1.0

    # Cohen's Kappa
    try:
        cohen_kappa = round(float(cohen_kappa_score(all_targets, all_preds)), 4)
    except Exception:
        cohen_kappa = -1.0

    # ── CPU inference latency (200 images, simulates mobile device) ───────────
    import copy
    model_cpu = copy.deepcopy(model).cpu().eval()
    dummy_batch = torch.zeros(1, 3, STUDENT_IMG_SIZE, STUDENT_IMG_SIZE)
    latencies = []
    with torch.no_grad():
        for _ in range(220):   # 20 warm-up + 200 measured
            t0 = time.perf_counter()
            model_cpu(dummy_batch)
            latencies.append((time.perf_counter() - t0) * 1000)
    lat_mean = round(float(np.mean(latencies[20:])), 2)
    lat_std  = round(float(np.std(latencies[20:])),  2)
    lat_fps  = round(1000.0 / max(lat_mean, 0.001), 1)

    results = {
        "variant": variant, "mode": mode,
        "n_test_samples":   n_samples,
        "eval_duration_s":  eval_duration_s,
        # Silhouette segmentation (Ch0)
        "sil_mIoU":         round(seg0["iou"],         4),
        "sil_dice":         round(seg0["dice"],         4),
        "sil_recall":       round(seg0["recall"],       4),
        "sil_precision":    round(seg0["precision"],    4),
        "sil_specificity":  round(seg0["specificity"],  4),
        # Symptom segmentation (Ch1)
        "sym_mIoU":         round(seg1["iou"],          4),
        "sym_dice":         round(seg1["dice"],          4),
        "sym_recall":       round(seg1["recall"],        4),
        "sym_precision":    round(seg1["precision"],     4),
        "sym_specificity":  round(seg1["specificity"],   4),
        # Classification — overall
        "cls_accuracy":     round(report.get("accuracy", 0), 4),
        "macro_f1":         round(macro_f1,    4),
        "weighted_f1":      round(weighted_f1, 4),
        "mcc":              round(mcc,          4),
        # Classification — per class
        "healthy_prec":     round(report.get("HEALTHY",{}).get("precision",0),4),
        "healthy_rec":      round(report.get("HEALTHY",{}).get("recall",0),   4),
        "healthy_f1":       round(report.get("HEALTHY",{}).get("f1-score",0), 4),
        "msv_prec":         round(report.get("MSV",{}).get("precision",0),    4),
        "msv_rec":          round(report.get("MSV",{}).get("recall",0),       4),
        "msv_f1":           round(report.get("MSV",{}).get("f1-score",0),     4),
        # MSV — CIMMYT grade-stratified F1 (early=grade1-3, mid=3-5, severe=5-9)
        "msv_f1_early":     round(msv_tier_f1_test["early"],  4),
        "msv_f1_mid":       round(msv_tier_f1_test["mid"],    4),
        "msv_f1_severe":    round(msv_tier_f1_test["severe"], 4),
        "mln_prec":         round(report.get("MLN",{}).get("precision",0),    4),
        "mln_rec":          round(report.get("MLN",{}).get("recall",0),       4),
        "mln_f1":           round(report.get("MLN",{}).get("f1-score",0),     4),
        # Severity regression
        "sev_mae_pct":      round(sev_mae,     2),
        "sev_mse_pct":      round(sev_mse,     2),
        "sev_rmse_pct":     round(sev_rmse,    2),
        "sev_mape_pct":     round(sev_mape,    2),
        "sev_r2":           round(sev_r2,      4),
        "sev_pearson":      round(sev_pearson, 4),
        # Composite criterion
        "composite":        round(composite, 4),
        # ROC-AUC (MSV-vs-rest, per-class OvR, macro OvR)
        "msv_roc_auc":      msv_roc_auc,
        "healthy_roc_auc":  healthy_roc_auc,
        "mln_roc_auc":      mln_roc_auc,
        "macro_roc_auc":    macro_roc_auc,
        # Cohen's Kappa
        "cohen_kappa":      cohen_kappa,
        # Inference latency
        "cpu_lat_mean_ms":  lat_mean,
        "cpu_lat_std_ms":   lat_std,
        "cpu_fps":          lat_fps,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Metrics CSV
    metrics_path = LOGS_DIR / f"student_test_metrics_{variant}_{mode}.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    # Confusion matrix CSV
    cm_path = LOGS_DIR / f"student_confusion_{variant}_{mode}.csv"
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + CLASSES)
        for cls, row in zip(CLASSES, cm):
            writer.writerow([cls] + row.tolist())

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"  {'─'*55}")
    print(f"  Segmentation (silhouette):  mIoU {results['sil_mIoU']:.4f}  "
          f"Dice {results['sil_dice']:.4f}  Rec {results['sil_recall']:.4f}")
    print(f"  Segmentation (symptom  ):  mIoU {results['sym_mIoU']:.4f}  "
          f"Dice {results['sym_dice']:.4f}  Rec {results['sym_recall']:.4f}")
    print(f"  Classification:  Acc {results['cls_accuracy']:.4f}  "
          f"MCC {results['mcc']:.4f}  Macro-F1 {results['macro_f1']:.4f}")
    print(f"  MSV (primary):   Prec {results['msv_prec']:.4f}  "
          f"Rec {results['msv_rec']:.4f}  F1 {results['msv_f1']:.4f}")
    print(f"  MSV by grade:    Early {results['msv_f1_early']:.4f}  "
          f"Mid {results['msv_f1_mid']:.4f}  Severe {results['msv_f1_severe']:.4f}")
    print(f"  Severity:        MAE {results['sev_mae_pct']:.2f}%  "
          f"RMSE {results['sev_rmse_pct']:.2f}%  R² {results['sev_r2']:.4f}")
    print(f"  Composite:       {results['composite']:.4f}")
    print(f"  CPU latency:     {lat_mean:.1f} ± {lat_std:.1f} ms  "
          f"({lat_fps:.1f} FPS)")
    print(f"  Eval duration:   {eval_duration_s:.1f}s")
    print(f"  {'─'*55}")
    print(f"  Metrics → {metrics_path}")
    print(f"  Confusion → {cm_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE VARIANT + MODE
# ══════════════════════════════════════════════════════════════════════════════

def train_one(encoder_variant: str, factory_mode: str, stage: int = 1) -> dict:
    # FIX: stage passed as explicit parameter instead of imported from __main__.
    # from __main__ import args is fragile — breaks if module is ever imported
    # rather than run directly (e.g. during testing or ablation scripting).
    set_seeds(SEED)

    use_cbam  = "cbam" in encoder_variant
    ckpt_name = f"{encoder_variant}_{factory_mode}"
    STUDENT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  Encoder : {encoder_variant}")
    print(f"  Mode    : {factory_mode}")
    print(f"{'═' * 60}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_samples = build_sample_list("train", factory_mode)
    val_samples   = build_sample_list("val",   factory_mode)

    if not train_samples:
        print(f"  [FATAL] No training samples for mode {factory_mode}.")
        return {}

    train_tf = make_student_transforms(STUDENT_IMG_SIZE, is_train=True)
    val_tf   = make_student_transforms(STUDENT_IMG_SIZE, is_train=False)

    train_ds = StudentDataset(train_samples, factory_mode, train_tf)
    val_ds   = StudentDataset(val_samples,   factory_mode, val_tf)

    # WeightedRandomSampler — ensures every batch has proportional MSV
    # representation despite class imbalance (HEALTHY/MLN ~38%, MSV ~24%).
    # FIX: multiply class-balance weight by per-sample reliability_weight from
    # factory output so high-coverage, high-confidence samples are proportionally
    # favoured. Samples without a factory reliability_weight default to 1.0.
    class_counts  = {cls: sum(1 for s in train_samples if s["category"] == cls)
                     for cls in CLASSES}
    sample_weights = [
        (1.0 / max(class_counts.get(s["category"], 1), 1))
        * float(s.get("reliability_weight", 1.0))
        for s in train_samples
    ]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=STUDENT_BATCH_SIZE,
        sampler=sampler,                    # replaces shuffle=True
        num_workers=STUDENT_NUM_WORKERS, pin_memory=True, drop_last=True,
        collate_fn=safe_collate)
    val_loader = DataLoader(
        val_ds, batch_size=STUDENT_BATCH_SIZE, shuffle=False,
        num_workers=STUDENT_NUM_WORKERS, pin_memory=True,
        collate_fn=safe_collate)

    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # ── Model + losses ────────────────────────────────────────────────────────
    model = StudentModel(encoder_variant, use_cbam=use_cbam).to(DEVICE)

    seg_criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)
    cls_criterion = AsymmetricLabelSmoothingLoss(ASYMMETRIC_PRIOR).to(DEVICE)
    # sev_criterion removed — MSE computed inline with F.mse_loss in
    # train_one_epoch/validate; nn.MSELoss() was instantiated but never called.
    unc_loss      = HomoscedasticUncertaintyLoss().to(DEVICE)

    # Optimizer includes uncertainty loss parameters
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(unc_loss.parameters()),
        lr=STUDENT_PHASE1_LR,
        weight_decay=STUDENT_WEIGHT_DECAY,
    )

    # ── Phase 1: Frozen encoder ───────────────────────────────────────────────
    print(f"\n  Phase 1 — frozen encoder ({STUDENT_PHASE1_EPOCHS} epochs, "
          f"LR={STUDENT_PHASE1_LR})")
    for param in model.unet.encoder.parameters():
        param.requires_grad = False

    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STUDENT_PHASE1_EPOCHS, eta_min=1e-6)

    # Stage-specific checkpoint subdirectory prevents cross-stage overwrites.
    # Stage 1 (encoder ablation, mode_b fixed) → stage1/
    # Stage 2 (mode ablation, best encoder fixed) → stage2/
    # FIX: use stage parameter directly — no __main__ import needed
    _stage_dir = STUDENT_CKPT_DIR / f"stage{stage}"
    _stage_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path   = _stage_dir / f"student_{ckpt_name}_best.pth"
    metrics_log = LOGS_DIR / f"student_{ckpt_name}_metrics.csv"
    log_rows    = []
    best_composite = 0.0
    no_improve     = 0
    best_val_metrics = {}

    for epoch in range(1, STUDENT_PHASE1_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            seg_criterion, cls_criterion,
            unc_loss, DEVICE)
        val_m = validate(
            model, val_loader, seg_criterion, cls_criterion,
            unc_loss, DEVICE)
        scheduler_p1.step()

        row = {
            "phase": 1, "epoch": epoch,
            "train_loss":   round(train_loss,             5),
            "val_loss":     round(val_m["loss"],           5),
            "sil_mIoU":    round(val_m["mIoU"],           4),
            "sil_dice":    round(val_m["dice"],            4),
            "sil_recall":  round(val_m["seg_recall"],      4),
            "sym_mIoU":    round(val_m["sym_iou"],         4),
            "sym_dice":    round(val_m["sym_dice"],        4),
            "cls_acc":     round(val_m["cls_acc"],         4),
            "msv_f1":      round(val_m["msv_f1"],          4),
            "msv_f1_early": round(val_m.get("msv_f1_early", 0.0), 4),
            "msv_f1_mid":   round(val_m.get("msv_f1_mid", 0.0),   4),
            "msv_f1_severe":round(val_m.get("msv_f1_severe", 0.0),4),
            "mln_f1":      round(val_m.get("mln_f1", 0.0),   4),
            "macro_f1":    round(val_m["macro_f1"],        4),
            "mcc":         round(val_m["mcc"],             4),
            "sev_mae":     round(val_m["sev_mae"],              2),
            "sev_mse":     round(val_m.get("sev_mse", 0.0),      2),
            "sev_rmse":    round(val_m["sev_rmse"],               2),
            "sev_mape":    round(val_m.get("sev_mape", 0.0),      2),
            "sev_r2":      round(val_m["sev_r2"],                 4),
            "sev_pearson": round(val_m.get("sev_pearson", 0.0),   4),
            "composite":   round(val_m["composite"],       4),
        }
        log_rows.append(row)

        print(f"  P1 Ep {epoch:02d} | loss {train_loss:.4f}→{val_m['loss']:.4f} "
              f"| mIoU {val_m['mIoU']:.4f} | MSV_F1 {val_m['msv_f1']:.4f} "
              f"| comp {val_m['composite']:.4f}")

        if val_m["composite"] > best_composite:
            best_composite   = val_m["composite"]
            best_val_metrics = val_m
            no_improve       = 0
            torch.save({
                "epoch": epoch, "phase": 1,
                "model_state":  model.state_dict(),
                "unc_state":    unc_loss.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "composite":    best_composite,
                "val_metrics":  {k: v for k, v in val_m.items()
                                 if k not in ("report","preds","targets")},
                "encoder":      encoder_variant,
                "mode":         factory_mode,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= STUDENT_PHASE1_PATIENCE:
                print(f"  Phase 1 early stop at epoch {epoch}.")
                break

    # ── Phase 2: Unfreeze full model ──────────────────────────────────────────
    print(f"\n  Phase 2 — full fine-tuning ({STUDENT_PHASE2_EPOCHS} epochs, "
          f"LR={STUDENT_PHASE2_LR})")
    for param in model.unet.encoder.parameters():
        param.requires_grad = True

    # Re-initialize optimizer with lower LR for fine-tuning
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(unc_loss.parameters()),
        lr=STUDENT_PHASE2_LR,
        weight_decay=STUDENT_WEIGHT_DECAY,
    )
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STUDENT_PHASE2_EPOCHS,
        eta_min=STUDENT_PHASE2_LR_MIN)

    # Reset patience counter for Phase 2
    no_improve = 0

    for epoch in range(1, STUDENT_PHASE2_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            seg_criterion, cls_criterion,
            unc_loss, DEVICE)
        val_m = validate(
            model, val_loader, seg_criterion, cls_criterion,
            unc_loss, DEVICE)
        scheduler_p2.step()

        row = {
            "phase": 2, "epoch": epoch,
            "train_loss":   round(train_loss,             5),
            "val_loss":     round(val_m["loss"],           5),
            "sil_mIoU":    round(val_m["mIoU"],           4),
            "sil_dice":    round(val_m["dice"],            4),
            "sil_recall":  round(val_m["seg_recall"],      4),
            "sym_mIoU":    round(val_m["sym_iou"],         4),
            "sym_dice":    round(val_m["sym_dice"],        4),
            "cls_acc":     round(val_m["cls_acc"],         4),
            "msv_f1":      round(val_m["msv_f1"],          4),
            "msv_f1_early": round(val_m.get("msv_f1_early", 0.0), 4),
            "msv_f1_mid":   round(val_m.get("msv_f1_mid", 0.0),   4),
            "msv_f1_severe":round(val_m.get("msv_f1_severe", 0.0),4),
            "mln_f1":      round(val_m.get("mln_f1", 0.0),   4),
            "macro_f1":    round(val_m["macro_f1"],        4),
            "mcc":         round(val_m["mcc"],             4),
            "sev_mae":     round(val_m["sev_mae"],              2),
            "sev_mse":     round(val_m.get("sev_mse", 0.0),      2),
            "sev_rmse":    round(val_m["sev_rmse"],               2),
            "sev_mape":    round(val_m.get("sev_mape", 0.0),      2),
            "sev_r2":      round(val_m["sev_r2"],                 4),
            "sev_pearson": round(val_m.get("sev_pearson", 0.0),   4),
            "composite":   round(val_m["composite"],       4),
        }
        log_rows.append(row)

        print(f"  P2 Ep {epoch:02d} | loss {train_loss:.4f}→{val_m['loss']:.4f} "
              f"| mIoU {val_m['mIoU']:.4f} | MSV_F1 {val_m['msv_f1']:.4f} "
              f"| comp {val_m['composite']:.4f}")

        if val_m["composite"] > best_composite:
            best_composite   = val_m["composite"]
            best_val_metrics = val_m
            no_improve       = 0
            torch.save({
                "epoch": epoch, "phase": 2,
                "model_state":  model.state_dict(),
                "unc_state":    unc_loss.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "composite":    best_composite,
                "val_metrics":  {k: v for k, v in val_m.items()
                                 if k not in ("report","preds","targets")},
                "encoder":      encoder_variant,
                "mode":         factory_mode,
            }, ckpt_path)
            print(f"  ✓ New best composite: {best_composite:.4f}")
        else:
            no_improve += 1
            if no_improve >= STUDENT_PHASE2_PATIENCE:
                print(f"  Phase 2 early stop at epoch {epoch}.")
                break

    # ── Save training log ─────────────────────────────────────────────────────
    with open(metrics_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_results = evaluate_test_split(model, factory_mode,
                                       encoder_variant, DEVICE)

    return {
        "encoder":       encoder_variant,
        "mode":          factory_mode,
        "best_composite":round(best_composite, 4),
        **{f"test_{k}": v for k, v in test_results.items()
           if k.startswith("test_")},
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                        help="1=encoder ablation on mode_b  |  2=mode ablation on best encoder")
    parser.add_argument("--encoder", type=str, default=STUDENT_BEST_VARIANT,
                        help="Best encoder variant for stage 2")
    args = parser.parse_args()

    set_seeds(SEED)
    STUDENT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  Yellow MAIze | Phase 5: Student Training — Stage {args.stage}")
    print("=" * 72)

    all_results = []

    if args.stage == 1:
        # ── Stage 1: All 5 encoder variants on Mode B ─────────────────────────
        # V1: mobilenet_v2 | V2: mobilenet_v2_cbam | V3: mobilenet_v3_small
        # V4: efficientnet_b0 | V6: efficientnet_b0_cbam
        print(f"  Variants  : {STUDENT_VARIANTS}")
        print(f"  Mode      : mode_b (fixed for encoder ablation)")
        for variant in STUDENT_VARIANTS:
            result = train_one(variant, "mode_b", stage=1)
            if result:
                all_results.append(result)

        # Find best encoder
        if all_results:
            best = max(all_results,
                       key=lambda r: r.get("test_composite", 0))
            print(f"\n  Best encoder: {best['encoder']}")
            print(f"  Update STUDENT_BEST_VARIANT in config.py to: "
                  f"'{best['encoder']}'")

    elif args.stage == 2:
        # ── Stage 2: Best encoder on all 4 modes ──────────────────────────────
        print(f"  Encoder   : {args.encoder}")
        print(f"  Modes     : {FACTORY_MODES}")
        for mode in FACTORY_MODES:
            result = train_one(args.encoder, mode, stage=2)
            if result:
                all_results.append(result)

        if all_results:
            best = max(all_results,
                       key=lambda r: r.get("test_composite", 0))
            print(f"\n  Best mode: {best['mode']}")
            print(f"  Best composite: {best.get('best_composite','N/A')}")

    # ── Save comparison CSV ───────────────────────────────────────────────────
    if all_results:
        comp_path = LOGS_DIR / f"student_comparison_stage{args.stage}.csv"
        keys      = list(all_results[0].keys())
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Comparison: {comp_path}")

    _duration = round(time.time() - _t_start, 1)
    print(f"\n  Total Student training duration: {_duration}s ({_duration/3600:.2f}h)")
    print(f"\n  NEXT STEP: python select_best_pipeline.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
