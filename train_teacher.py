"""
================================================================================
 train_teacher.py — Phase 3: Teacher Model Training
================================================================================
 PURPOSE:
   Train and compare 4 Teacher segmentation model variants on the 15,000
   SAM2-masked Tier 1 images. The Teacher's sole purpose is generating
   leaf silhouette pseudo-masks for the ~215k Tier 2 images via factory_master.py.

 KEY DESIGN DECISIONS:
   - Soft targets: SAM2 float32 probability maps used as segmentation targets
     (NO binarization). BCEWithLogitsLoss accepts float targets natively.
   - Manifest filter: Only Tier 1 images NOT in the global test split are used.
   - 4 variants compared: ResNet-50 UNet, EfficientNet-B2 UNet, SegFormer-B2,
     DeepLabV3+-EB2 (true decoder architecture comparison vs UNet)
   - Best variant by val Dice → teacher_model_best.pth (used by Factory)

 VARIANTS:
   resnet50          — deep CNN baseline (He et al. 2016)
   efficientnet-b2   — recommended (Tan & Le 2019, best quality/VRAM ratio)
   mit_b2            — SegFormer-B2 via timm (Xie et al. 2021)
   deeplabv3plus-eb2 — DeepLabV3+ + EfficientNet-B2 (true decoder comparison)

 OUTPUTS:
   checkpoints/teacher/teacher_{variant}_best.pth
   checkpoints/teacher/teacher_model_best.pth   ← symlink/copy of winner
   logs/teacher_{variant}_metrics.csv
   logs/teacher_comparison.csv
================================================================================
"""

import csv
import random
import shutil
import sys
import time
from image_utils import load_image_rgb
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pandas as pd
import segmentation_models_pytorch as smp

from image_utils import load_image_rgb   # EXIF correction + corrupt guard (NOT clahe)
from scripts.safe_collate import safe_collate, reset_skip_counter

from config import (
    SEED,
    TIER1_RAW_DIR, TIER1_MASKS_DIR, TIER1_MANIFEST,
    GLOBAL_MANIFEST,
    TEACHER_CKPT_DIR, LOGS_DIR, REPORTS_DIR,
    TEACHER_IMG_SIZE, TEACHER_BATCH_SIZE, TEACHER_BATCH_SIZE_OVERRIDES,
    TEACHER_EPOCHS,
    TEACHER_LR, TEACHER_WEIGHT_DECAY, TEACHER_VAL_SPLIT,
    TEACHER_PATIENCE, TEACHER_LR_FACTOR, TEACHER_LR_PATIENCE,
    TEACHER_VARIANTS, TEACHER_DEPLOYED_VARIANT,
    TEACHER_GRAD_ACCUM_STEPS,
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

# RTX 5060 8 GB: reduce fragmentation from UNet skip-connection allocs.
# Set before launching: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# or it will be set automatically here if not already in the environment.
import os as _os
if "PYTORCH_CUDA_ALLOC_CONF" not in _os.environ:
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ══════════════════════════════════════════════════════════════════════════════
# BOUNDARY-AWARE COMBINED LOSS
# ══════════════════════════════════════════════════════════════════════════════

class BoundaryAwareLoss(nn.Module):
    """
    Combined Dice + boundary-weighted BCE loss.

    Motivation: Teacher was systematically undersegmenting leaf margins —
    the boundary band in SAM2 soft targets (values 0.3–0.7) caused the model
    to learn conservative, shrunk predictions. Standard DiceLoss treats all
    pixels equally and cannot penalise missed edges specifically.

    This loss adds a BCE term where boundary pixels are upweighted by
    BOUNDARY_WEIGHT (default 5.0), forcing the model to focus on the exact
    region where it was failing.

    Soft target sharpening: SAM2 probability maps are amplified slightly
    (×SHARPEN_FACTOR, clipped to [0,1]) before computing loss. This pulls
    boundary-band values (0.3–0.7) toward 1.0 without fully binarizing,
    reducing the amount the model is rewarded for predicting low-confidence
    at edges.

    Loss formula:
        boundary = max_pool(target) − avg_pool(target)   ← edge detector
        weight   = 1.0 + BOUNDARY_WEIGHT × boundary
        L        = DiceLoss(pred, target_sharp)
                 + BCE_weighted(pred, target_sharp, weight)
    """

    BOUNDARY_WEIGHT = 5.0    # upweight factor for boundary pixels
    SHARPEN_FACTOR  = 1.3    # amplify soft targets before loss (clip to 1.0)
    DICE_WEIGHT     = 1.0    # relative weight of Dice term
    BCE_WEIGHT      = 0.5    # relative weight of boundary BCE term

    def __init__(self):
        super().__init__()
        self._dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Sharpen soft targets slightly — reduces ambiguity at leaf edges
        targets_sharp = (targets * self.SHARPEN_FACTOR).clamp(0.0, 1.0)

        # Dice loss on sharpened targets
        l_dice = self._dice(logits, targets_sharp)

        # Boundary detection via max_pool − avg_pool (pure PyTorch, no kornia)
        # Works on [B, 1, H, W] targets
        t = targets_sharp
        boundary = (
            F.max_pool2d(t, kernel_size=3, stride=1, padding=1)
            - F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)
        ).clamp(0.0, 1.0)

        # Pixel weights: 1.0 everywhere + extra at boundary pixels
        weight = 1.0 + self.BOUNDARY_WEIGHT * boundary

        # Weighted BCE
        l_bce = F.binary_cross_entropy_with_logits(
            logits, targets_sharp, weight=weight
        )

        return self.DICE_WEIGHT * l_dice + self.BCE_WEIGHT * l_bce


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def make_train_transforms(img_size: int):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0, fill=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=20,
                             val_shift_limit=10, p=0.2),
        # ElasticTransform: deforms leaf boundaries slightly during training,
        # preventing the model from memorising exact SAM2 boundary shapes and
        # encouraging it to learn robust edge features instead.
        A.ElasticTransform(alpha=60, sigma=6, p=0.3),
        # CoarseDropout: randomly masks small regions, forces the model to
        # infer leaf extent from context rather than local texture — reduces
        # the tendency to shrink predictions where texture is ambiguous.
        A.CoarseDropout(num_holes_range=(1, 6),
                        hole_height_range=(8, 32),
                        hole_width_range=(8, 32),
                        fill=0, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def make_val_transforms(img_size: int):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0, fill=0),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TeacherDataset(Dataset):
    """
    Tier 1 leaf silhouette segmentation dataset.
    Loads SAM2 float32 .npy probability maps as SOFT segmentation targets.
    No binarization — preserves boundary uncertainty signal.
    """

    def __init__(self, samples: list[dict], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row      = self.samples[idx]
        img_path = Path(row["img_path"])
        npy_path = Path(row["npy_path"])

        # EXIF-corrected load with truncation/corrupt guard.
        # NOTE: deliberately load_image_rgb (NOT load_image_clahe).
        # CLAHE is correct for Bouncer and Factory inputs, but Teacher training
        # must use the same image appearance that SAM2 saw when generating the
        # soft probability maps — applying CLAHE here would create an
        # image/target mismatch at the leaf boundary region.
        if not img_path.exists():
            return None
        img_rgb = load_image_rgb(img_path)
        if img_rgb is None:
            return None   # corrupt/truncated — safe_collate skips

        # Load soft probability map (float32, range [0,1])
        # NO cv2.threshold — preserve soft values for boundary uncertainty
        prob_map = np.load(str(npy_path)).astype(np.float32)

        # Resize prob_map to match image if needed (SAM2 may output diff size)
        if prob_map.shape != (img_rgb.shape[0], img_rgb.shape[1]):
            prob_map = cv2.resize(
                prob_map,
                (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        if self.transform:
            augmented = self.transform(image=img_rgb, mask=prob_map)
            img_rgb   = augmented["image"]   # tensor C×H×W
            prob_map  = augmented["mask"]    # tensor H×W

        # Add channel dim to mask: H×W → 1×H×W
        if isinstance(prob_map, torch.Tensor):
            prob_map = prob_map.unsqueeze(0)
        else:
            prob_map = torch.tensor(prob_map, dtype=torch.float32).unsqueeze(0)

        return img_rgb, prob_map


def build_sample_list() -> list[dict]:
    """
    Build list of (img_path, npy_path) dicts from Tier 1 manifest,
    excluding any images that are in the global test split.
    """
    if not TIER1_MANIFEST.exists():
        raise FileNotFoundError(
            f"{TIER1_MANIFEST} not found. Run sample_15000.py first.")
    if not GLOBAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"{GLOBAL_MANIFEST} not found. Run partition_dataset.py first.")

    tier1_df  = pd.read_csv(TIER1_MANIFEST)
    global_df = pd.read_csv(GLOBAL_MANIFEST)

    # Build set of test-split filenames
    test_fnames = set(
        global_df[global_df["split"] == "test"]["filename"].tolist()
    )

    samples = []
    skipped = 0
    for _, row in tier1_df.iterrows():
        dest_fname = row["dest_filename"]
        orig_fname = Path(row["source_path"]).name

        # Exclude test-split images
        if orig_fname in test_fnames:
            skipped += 1
            continue

        img_path = TIER1_RAW_DIR  / dest_fname
        stem     = Path(dest_fname).stem
        npy_path = TIER1_MASKS_DIR / f"{stem}_softmask.npy"

        if img_path.exists() and npy_path.exists():
            samples.append({
                "img_path": str(img_path),
                "npy_path": str(npy_path),
                "category": row["category"],
                "split":    row["split"],
            })

    print(f"  Samples loaded : {len(samples):,}  (skipped {skipped:,} test-split)")

    # ── Verify #7: data distribution check ───────────────────────────────────
    # If this number is significantly below ~12,000 it indicates a pipeline
    # issue (missing .npy files, wrong paths, etc.) — investigate before training.
    by_class = {}
    for s in samples:
        by_class.setdefault(s["category"], 0)
        by_class[s["category"]] += 1
    print(f"  Distribution  : " + " | ".join(
        f"{c}: {by_class.get(c, 0):,}" for c in ["HEALTHY", "MSV", "MLN"]))
    expected_min = 10_000
    if len(samples) < expected_min:
        print(f"  [WARN] Only {len(samples):,} samples — expected ≥ {expected_min:,}.")
        print(f"         Check that tier1_leaf_masks/ has .npy files for all Tier 1 images.")

    return samples


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def build_teacher_model(variant: str) -> nn.Module:
    """
    Build Teacher segmentation model.

    Variants:
      resnet50        — smp.Unet, ResNet-50 encoder    (CNN baseline)
      efficientnet-b2 — smp.Unet, EfficientNet-B2       (recommended)
      mit_b2          — smp.Unet, SegFormer-B2 via timm (ViT encoder)
      deeplabv3plus-eb2 — smp.DeepLabV3Plus, EfficientNet-B2
                          True decoder comparison: ASPP vs UNet skip connections.
                          ASPP handles multi-scale leaf variation better.

    Dropout: applied to the penultimate encoder feature map via a wrapper hook.
    Targeting only the last encoder block avoids disturbing shallow skip
    connections while still regularising the high-level representations that
    tend to overfit first (as seen: val Dice peaked ep 3 and diverged thereafter).
    """
    ENCODER_DROPOUT = 0.3   # drop 30% of encoder output channels during training

    unet_encoders = {
        "resnet50":        "resnet50",
        "efficientnet-b2": "efficientnet-b2",
        "mit_b2":          "mit_b2",
    }

    if variant in unet_encoders:
        model = smp.Unet(
            encoder_name=unet_encoders[variant],
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,        # binary leaf silhouette
            activation=None,  # raw logits — loss applies sigmoid internally
        )
    elif variant == "deeplabv3plus-eb2":
        # DeepLabV3+ with EfficientNet-B2 encoder.
        # ASPP (Atrous Spatial Pyramid Pooling) captures multi-scale context
        # — well-suited for maize leaves photographed at variable distances.
        # Most-cited architecture in plant leaf segmentation literature.
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    else:
        raise ValueError(f"Unknown Teacher variant: {variant}")

    # Register a forward hook that applies Dropout2d to the final encoder
    # feature map during training. This is the standard way to add encoder
    # dropout to smp models without modifying the frozen encoder weights or
    # patching internal smp code.
    _dropout_layer = nn.Dropout2d(p=ENCODER_DROPOUT)

    def _encoder_dropout_hook(module, input, output):
        # output is a list of feature maps from each encoder stage.
        # Apply dropout only to the last (deepest) feature map.
        if isinstance(output, (list, tuple)) and _dropout_layer.training:
            out_list = list(output)
            out_list[-1] = _dropout_layer(out_list[-1])
            return type(output)(out_list)
        return output

    model.encoder.register_forward_hook(_encoder_dropout_hook)

    return model


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_seg_metrics(preds_logits: torch.Tensor,
                        targets: torch.Tensor,
                        threshold: float = 0.5) -> dict:
    """
    Compute Dice, IoU, Recall (Sensitivity), Precision, and Specificity
    from logits vs soft targets. Targets binarized at 0.5 for metric
    computation only — soft values used only for loss.
    """
    preds_prob = torch.sigmoid(preds_logits)
    preds_bin  = (preds_prob >= threshold).long()
    targs_bin  = (targets >= threshold).long()

    tp = (preds_bin & targs_bin).sum().float()
    fp = (preds_bin & ~targs_bin.bool()).sum().float()
    fn = (~preds_bin.bool() & targs_bin).sum().float()
    tn = (~preds_bin.bool() & ~targs_bin.bool()).sum().float()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou  = tp / (tp + fp + fn + 1e-7)
    rec  = tp / (tp + fn + 1e-7)           # Sensitivity / Recall
    prec = tp / (tp + fp + 1e-7)           # Precision
    spec = tn / (tn + fp + 1e-7)           # Specificity
    return {
        "dice": dice.item(), "iou": iou.item(),
        "recall": rec.item(), "precision": prec.item(),
        "specificity": spec.item(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """
    Train one epoch with gradient accumulation.
    TEACHER_GRAD_ACCUM_STEPS micro-batches are accumulated before each
    optimizer step, giving an effective batch size of
    TEACHER_BATCH_SIZE × TEACHER_GRAD_ACCUM_STEPS without the extra VRAM
    cost of a larger physical batch.
    """
    model.train()
    total_loss = 0.0
    n_samples  = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if batch is None:
            continue
        imgs, masks = batch
        imgs, masks = imgs.to(device), masks.to(device)

        logits = model(imgs)
        # Divide loss by accum steps so gradients are averaged, not summed,
        # keeping the effective LR identical to a single large-batch step.
        loss = criterion(logits, masks) / TEACHER_GRAD_ACCUM_STEPS
        loss.backward()

        total_loss += loss.item() * TEACHER_GRAD_ACCUM_STEPS * imgs.size(0)
        n_samples  += imgs.size(0)

        if (step + 1) % TEACHER_GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

    # Flush any remaining accumulated gradients at epoch end
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    dice_sum   = 0.0
    iou_sum    = 0.0
    rec_sum    = 0.0
    prec_sum   = 0.0
    spec_sum   = 0.0
    n_batches  = 0

    for batch in loader:
        if batch is None:
            continue
        imgs, masks = batch
        imgs, masks = imgs.to(device), masks.to(device)
        logits      = model(imgs)
        loss        = criterion(logits, masks)
        total_loss += loss.item() * imgs.size(0)

        m = compute_seg_metrics(logits, masks)
        dice_sum  += m["dice"]
        iou_sum   += m["iou"]
        rec_sum   += m["recall"]
        prec_sum  += m["precision"]
        spec_sum  += m["specificity"]
        n_batches += 1

    return {
        "loss":        total_loss / len(loader.dataset),
        "dice":        dice_sum  / max(n_batches, 1),
        "iou":         iou_sum   / max(n_batches, 1),
        "recall":      rec_sum   / max(n_batches, 1),
        "precision":   prec_sum  / max(n_batches, 1),
        "specificity": spec_sum  / max(n_batches, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN VARIANT
# ══════════════════════════════════════════════════════════════════════════════

def train_variant(variant: str, all_samples: list[dict]) -> dict:
    set_seeds(SEED)
    TEACHER_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  Teacher variant: {variant}")
    print(f"{'─' * 60}")

    # ── Split samples using internal 80/20 (on non-test images only) ──────────
    indices  = list(range(len(all_samples)))
    random.shuffle(indices)
    n_val    = int(len(indices) * TEACHER_VAL_SPLIT)
    val_idx  = indices[:n_val]
    train_idx= indices[n_val:]

    train_tf = make_train_transforms(TEACHER_IMG_SIZE)
    val_tf   = make_val_transforms(TEACHER_IMG_SIZE)

    train_ds = TeacherDataset([all_samples[i] for i in train_idx], train_tf)
    val_ds   = TeacherDataset([all_samples[i] for i in val_idx],   val_tf)

    batch_size = TEACHER_BATCH_SIZE_OVERRIDES.get(variant, TEACHER_BATCH_SIZE)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=safe_collate)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=safe_collate)

    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_teacher_model(variant).to(DEVICE)
    # BoundaryAwareLoss: Dice + boundary-weighted BCE on sharpened soft targets.
    # Replaces plain DiceLoss to specifically penalise missed leaf margin pixels
    # — the failure mode observed in Teacher overlays (IoU 0.72 vs SAM2 0.94).
    criterion = BoundaryAwareLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TEACHER_LR,
        weight_decay=TEACHER_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=TEACHER_LR_FACTOR,
        patience=TEACHER_LR_PATIENCE,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    ckpt_path   = TEACHER_CKPT_DIR / f"teacher_{variant}_best.pth"
    metrics_log = LOGS_DIR / f"teacher_{variant}_metrics.csv"

    best_dice  = 0.0
    no_improve = 0
    log_rows   = []

    for epoch in range(1, TEACHER_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_m = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_m["dice"])

        row = {
            "epoch":           epoch,
            "train_loss":      round(train_loss,            5),
            "val_loss":        round(val_m["loss"],         5),
            "val_dice":        round(val_m["dice"],         4),
            "val_iou":         round(val_m["iou"],          4),
            "val_recall":      round(val_m["recall"],       4),
            "val_precision":   round(val_m["precision"],    4),
            "val_specificity": round(val_m["specificity"],  4),
            "lr":              round(optimizer.param_groups[0]["lr"], 8),
        }
        log_rows.append(row)

        print(f"  Ep {epoch:02d}/{TEACHER_EPOCHS} | "
              f"loss {train_loss:.4f}→{val_m['loss']:.4f} | "
              f"dice {val_m['dice']:.4f} | iou {val_m['iou']:.4f} | "
              f"rec {val_m['recall']:.4f} | spec {val_m['specificity']:.4f}")

        if val_m["dice"] > best_dice:
            best_dice  = val_m["dice"]
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "metrics":     row,
                "variant":     variant,
            }, ckpt_path)
            print(f"  ✓ New best Dice: {best_dice:.4f} → saved")
        else:
            no_improve += 1
            if no_improve >= TEACHER_PATIENCE:
                print(f"  Early stop at epoch {epoch}.")
                break

    # ── Metrics CSV ───────────────────────────────────────────────────────────
    with open(metrics_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # Run quick timed inference (100 images, CPU) for comparison table
    model_cpu  = build_teacher_model(variant).eval()
    ckpt_data  = torch.load(ckpt_path, map_location="cpu")
    model_cpu.load_state_dict(ckpt_data["model_state"])
    import time
    dummy = torch.zeros(1, 3, TEACHER_IMG_SIZE, TEACHER_IMG_SIZE)
    latencies = []
    with torch.no_grad():
        for _ in range(20):   # warm-up + measure
            t0 = time.perf_counter()
            model_cpu(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)
    avg_lat = round(float(__import__("numpy").mean(latencies[5:])), 1)
    print(f"  Inference latency (CPU, 512px): {avg_lat} ms/image")

    return {
        "variant":     variant,
        "best_dice":   round(best_dice, 4),
        "lat_cpu_ms":  avg_lat,
        "ckpt":        str(ckpt_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# TEACHER TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_teacher_test(variant: str, all_samples: list[dict]) -> dict:
    """
    Evaluate best Teacher checkpoint on Tier 1 test-split images.
    Generates: logs/teacher_test_metrics.csv + reports/teacher_overlays/
    """
    import pandas as pd, time as _t
    if not GLOBAL_MANIFEST.exists():
        print("  [WARN] Global manifest not found — skip Teacher test eval.")
        return {}

    global_df  = pd.read_csv(GLOBAL_MANIFEST)
    test_fnames = set(global_df[global_df["split"] == "test"]["filename"].tolist())

    # Filter Tier 1 samples to test split
    test_samples = [s for s in all_samples
                    if Path(s["img_path"]).name in test_fnames]
    if not test_samples:
        print("  [WARN] No Tier 1 test-split images found.")
        return {}

    print(f"  Teacher test evaluation on {len(test_samples)} Tier 1 test images ...")

    ckpt_path = TEACHER_CKPT_DIR / f"teacher_{variant}_best.pth"
    if not ckpt_path.exists():
        print(f"  [WARN] Teacher checkpoint not found: {ckpt_path}")
        return {}

    model     = build_teacher_model(variant).to(DEVICE)
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = BoundaryAwareLoss()
    val_tf    = make_val_transforms(TEACHER_IMG_SIZE)
    test_ds   = TeacherDataset(test_samples, val_tf)
    batch_size = TEACHER_BATCH_SIZE_OVERRIDES.get(variant, TEACHER_BATCH_SIZE)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=safe_collate)

    # Global stat accumulators
    stats     = {"tp":0.,"fp":0.,"fn":0.,"tn":0.}
    total_loss = 0.
    n_samples  = 0
    t0 = _t.time()

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            imgs, masks = batch
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits      = model(imgs)
            loss        = criterion(logits, masks)
            total_loss += loss.item() * imgs.size(0)
            n_samples  += imgs.size(0)

            preds_bin = (torch.sigmoid(logits) >= 0.5).long()
            targs_bin = (masks >= 0.5).long()
            stats["tp"] += ((preds_bin==1)&(targs_bin==1)).sum().item()
            stats["fp"] += ((preds_bin==1)&(targs_bin==0)).sum().item()
            stats["fn"] += ((preds_bin==0)&(targs_bin==1)).sum().item()
            stats["tn"] += ((preds_bin==0)&(targs_bin==0)).sum().item()

    eval_dur = round(_t.time() - t0, 1)
    eps      = 1e-7
    tp,fp,fn,tn = stats["tp"],stats["fp"],stats["fn"],stats["tn"]
    dice     = 2*tp/(2*tp+fp+fn+eps)
    iou      = tp/(tp+fp+fn+eps)
    recall   = tp/(tp+fn+eps)
    prec     = tp/(tp+fp+eps)
    spec     = tn/(tn+fp+eps)

    result = {
        "variant":        variant,
        "n_test_images":  n_samples,
        "eval_duration_s":eval_dur,
        "test_loss":      round(total_loss/max(n_samples,1), 5),
        "test_dice":      round(dice,   4),
        "test_iou":       round(iou,    4),
        "test_recall":    round(recall, 4),
        "test_precision": round(prec,   4),
        "test_specificity":round(spec,  4),
    }

    print(f"  Dice {dice:.4f} | IoU {iou:.4f} | "
          f"Recall {recall:.4f} | Spec {spec:.4f} ({eval_dur}s)")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    test_path = LOGS_DIR / "teacher_test_metrics.csv"
    import csv as _csv
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=result.keys())
        w.writeheader()
        w.writerow(result)
    print(f"  Teacher test metrics → {test_path}")

    # Qualitative overlays (5 per class)
    overlay_dir = REPORTS_DIR / "teacher_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rng = __import__("random").Random(42)
    by_cls = {}
    for s in test_samples:
        c = s["category"]
        by_cls.setdefault(c, []).append(s)

    for cls, slist in by_cls.items():
        sel = rng.sample(slist, min(5, len(slist)))
        for s in sel:
            stem     = Path(s["img_path"]).stem
            npy_path = Path(s["npy_path"])
            if not npy_path.exists():
                continue
            img_rgb  = load_image_rgb(Path(s["img_path"]))
            if img_rgb is None:
                continue
            prob_map = np.load(str(npy_path))
            binary   = (prob_map >= 0.5).astype(np.uint8)
            overlay  = img_rgb.copy()
            contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                contours, -1, (0,200,0), 2)
            out_img = img_rgb.copy()
            mask3   = np.stack([binary*0, binary*200, binary*0], axis=-1)
            out_img = np.clip(out_img.astype(int) + mask3, 0, 255).astype(np.uint8)
            cv2.imwrite(
                str(overlay_dir / f"{stem}_{cls}_teacher_overlay.jpg"),
                cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    print(f"  Overlays → {overlay_dir}")
    return result



def main() -> None:
    _t_start = time.time()
    set_seeds(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Phase 3: Teacher Model Training")
    print("=" * 72)

    all_samples = build_sample_list()
    if not all_samples:
        print("[FATAL] No valid Tier 1 samples found.")
        return

    comparison_rows = []
    for variant in TEACHER_VARIANTS:
        result = train_variant(variant, all_samples)
        comparison_rows.append(result)

    # ── Find best Teacher ─────────────────────────────────────────────────────
    best = max(comparison_rows, key=lambda r: r["best_dice"])

    # Copy best checkpoint to canonical name
    best_src  = Path(best["ckpt"])
    best_dest = TEACHER_CKPT_DIR / "teacher_model_best.pth"
    shutil.copy2(best_src, best_dest)

    # ── Write comparison CSV ──────────────────────────────────────────────────
    comp_path = LOGS_DIR / "teacher_comparison.csv"
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  Teacher comparison summary:")
    print(f"  {'Variant':<20} {'Best Dice':>10}")
    print(f"  {'─'*20} {'─'*10}")
    for r in comparison_rows:
        marker = " ← BEST" if r["variant"] == best["variant"] else ""
        print(f"  {r['variant']:<20} {r['best_dice']:>10.4f}{marker}")

    print(f"\n  Best Teacher : {best['variant']}  (Dice {best['best_dice']:.4f})")
    print(f"  Saved to     : {best_dest}")
    print(f"  Comparison   : {comp_path}")
    print(f"\n  NEXT STEP: python factory_master.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
