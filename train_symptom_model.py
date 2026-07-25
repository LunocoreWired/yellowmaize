"""
================================================================================
 train_symptom_model.py — Phase 3b: Symptom Teacher (Human-Supervised)
================================================================================
 PURPOSE:
   Replace the hand-tuned LAB/HSV color-threshold symptom detection in
   factory_master.py (compute_lab_hard_mask / compute_lab_soft_confidence)
   with a learned segmentation model trained on a small set of human-verified
   symptom masks. Static color thresholds cannot separate early-stage
   chlorosis from healthy yellow-maize tissue under variable field lighting
   — eight rounds of threshold tuning (v1->v8 in factory_master.py) confirm
   this is a structural ceiling, not a tuning problem. A model trained on
   ~400-500 human masks learns the decision boundary directly.

 TWO MODELS TRAINED HERE:
   1. Healthy-only autoencoder (HealthyAE) — trained exclusively on HEALTHY
      images. At inference, per-pixel reconstruction error is high wherever
      input deviates from "what a healthy leaf looks like" — lighting-
      invariant in a way raw RGB/LAB thresholds are not, because the AE
      learns the manifold of healthy appearance rather than a fixed cutoff.
   2. Symptom Teacher (EfficientNet-B2 UNet) — trained on human polygon/brush
      masks. Input is RGB + AE reconstruction-error map (4 channels). The AE
      channel supplies a lighting-invariant anomaly prior; RGB supplies the
      chromatic cues the LAB pipeline already proved informative. Ground
      truth is always the human mask — the AE is an input feature, not a
      label source, so it cannot inherit LAB's systematic biases.

 ANNOTATION REQUIREMENTS:
   All 501 gold-standard images annotated in CVAT using three labels:
     maize-leaf   (MAIZE_LEAF_LABEL_NAME)  — leaf silhouette, all 3 classes
     msv-symptom  (MSV_SYMPTOM_LABEL_NAME) — chlorotic streaks, MSV images only
     mln-symptom  (MLN_SYMPTOM_LABEL_NAME) — necrotic patches, MLN images only

   HEALTHY images need only a maize-leaf annotation (no symptom regions).
   They are included in training with all-zero symptom masks for both channels,
   teaching the model to output nothing on a healthy leaf.

   Export from CVAT:
     Task menu → Export dataset → COCO 1.0 → extract zip
     Place instances_default.json at:
       data/gold_standard/annotations/symptom_annotations.json

   Target: SYMPTOM_MIN_ANNOTATIONS (400) MSV+MLN images with symptom regions,
   stratified across severity levels so early/faint symptoms are represented.
   Install pycocotools for reliable RLE (brush mask) decoding:
     pip install pycocotools

 WORKFLOW:
   1. Train HealthyAE on HEALTHY images from the global manifest (no
      annotation needed — uses the class label only).
   2. Parse symptom_annotations.json -> binary masks.
   3. Train Symptom Teacher on (RGB + AE-error) -> human mask pairs.
   4. Validate on held-out split; report mean IoU.
   5. [Optional] --compare-lab: run the legacy LAB pipeline on the same
      held-out images and report IoU(LAB, human) vs IoU(SymptomTeacher,
      human) side by side for the thesis comparison figure.

 OUTPUTS:
   checkpoints/healthy_ae/healthy_ae_best.pth
   checkpoints/symptom/symptom_teacher_best.pth   <- read by factory_master.py
   logs/symptom_teacher_metrics.csv               <- per-epoch loss/Dice/IoU
   reports/symptom_vs_lab_comparison.csv          <- only with --compare-lab
================================================================================
"""

import argparse
import json
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from image_utils import load_image_rgb
from config import (
    SEED, CLASSES, GLOBAL_MANIFEST,
    GOLD_IMAGES_DIR, SYMPTOM_ANNOTATION_FILE,
    MAIZE_LEAF_LABEL_NAME, MSV_SYMPTOM_LABEL_NAME, MLN_SYMPTOM_LABEL_NAME,
    SYMPTOM_MIN_ANNOTATIONS, SYMPTOM_IMG_SIZE, SYMPTOM_VAL_SPLIT,
    SYMPTOM_BATCH_SIZE, SYMPTOM_EPOCHS, SYMPTOM_LR, SYMPTOM_WEIGHT_DECAY,
    SYMPTOM_PATIENCE, SYMPTOM_ENCODER,
    SYMPTOM_DICE_WEIGHT, SYMPTOM_FOCAL_WEIGHT,
    SYMPTOM_FOCAL_GAMMA, SYMPTOM_FOCAL_POS_WEIGHT,
    SYMPTOM_IOU_TARGET_MEAN,
    HEALTHY_AE_IMG_SIZE, HEALTHY_AE_LATENT_DIM, HEALTHY_AE_BATCH_SIZE,
    HEALTHY_AE_EPOCHS, HEALTHY_AE_LR, HEALTHY_AE_VAL_SPLIT, HEALTHY_AE_PATIENCE,
    SYMPTOM_CKPT_DIR, HEALTHY_AE_CKPT_DIR, LOGS_DIR, REPORTS_DIR,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — HEALTHY-ONLY AUTOENCODER
# ══════════════════════════════════════════════════════════════════════════════

class HealthyAE(nn.Module):
    """
    Lightweight convolutional autoencoder with a true bottleneck (no skip
    connections — unlike a UNet, this MUST compress through the latent
    vector so it cannot passthrough disease pixels it has never seen).
    Trained on HEALTHY images only. Reconstruction error at inference time
    is high wherever input deviates from the learned healthy-leaf manifold.
    """

    def __init__(self, latent_dim: int = HEALTHY_AE_LATENT_DIM):
        super().__init__()
        # Encoder: 256 -> 128 -> 64 -> 32 -> 16
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.to_latent   = nn.Conv2d(256, latent_dim, 1)
        self.from_latent = nn.Conv2d(latent_dim, 256, 1)
        # Decoder: 16 -> 32 -> 64 -> 128 -> 256
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.to_latent(self.enc(x))
        return self.dec(self.from_latent(z))


class HealthyDataset(Dataset):
    """HEALTHY-class images only, drawn from the global train+val manifest."""

    def __init__(self, paths: list[Path], img_size: int):
        self.paths = paths
        self.tf = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image_rgb(self.paths[idx])
        if img is None:
            return None
        img = self.tf(image=img)["image"].float() / 255.0
        return img


def _ae_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.stack(batch)


def train_healthy_ae() -> Path:
    """Train the HealthyAE on HEALTHY images from the global manifest."""
    print(f"\n{'─' * 72}")
    print("  Step 1 — Training Healthy-Only Autoencoder")
    print(f"{'─' * 72}")

    if not GLOBAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"{GLOBAL_MANIFEST} not found. Run partition_dataset.py first.")

    manifest = pd.read_csv(GLOBAL_MANIFEST)
    healthy = manifest[
        (manifest["category"] == "HEALTHY") & (manifest["split"] != "test")
    ]
    paths = [Path(p) for p in healthy["source_path"].tolist()]
    random.shuffle(paths)

    if len(paths) < 50:
        raise RuntimeError(
            f"Only {len(paths)} HEALTHY images available — too few to train "
            f"a reconstruction autoencoder.")

    n_val   = max(1, int(len(paths) * HEALTHY_AE_VAL_SPLIT))
    val_paths   = paths[:n_val]
    train_paths = paths[n_val:]
    print(f"  HEALTHY images: {len(train_paths):,} train / {len(val_paths):,} val")

    train_ds = HealthyDataset(train_paths, HEALTHY_AE_IMG_SIZE)
    val_ds   = HealthyDataset(val_paths, HEALTHY_AE_IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=HEALTHY_AE_BATCH_SIZE,
                              shuffle=True, num_workers=2, collate_fn=_ae_collate)
    val_loader   = DataLoader(val_ds, batch_size=HEALTHY_AE_BATCH_SIZE,
                              shuffle=False, num_workers=2, collate_fn=_ae_collate)

    model = HealthyAE().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=HEALTHY_AE_LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                        factor=0.5, patience=4)

    HEALTHY_AE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = HEALTHY_AE_CKPT_DIR / "healthy_ae_best.pth"
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, HEALTHY_AE_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            if batch is None:
                continue
            batch = batch.to(DEVICE)
            opt.zero_grad()
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches  += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                batch = batch.to(DEVICE)
                recon = model(batch)
                val_loss += F.mse_loss(recon, batch).item()
                n_val_batches += 1
        val_loss /= max(n_val_batches, 1)
        sched.step(val_loss)

        print(f"  Epoch {epoch:>3}/{HEALTHY_AE_EPOCHS}  "
              f"train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(),
                       "val_loss": val_loss,
                       "latent_dim": HEALTHY_AE_LATENT_DIM}, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= HEALTHY_AE_PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {HEALTHY_AE_PATIENCE} epochs)")
                break

    print(f"  Best HealthyAE val_mse: {best_val_loss:.5f}")
    print(f"  Saved: {best_path}")
    return best_path


@torch.no_grad()
def compute_ae_error_map(model: nn.Module, img_rgb: np.ndarray,
                         img_size: int = HEALTHY_AE_IMG_SIZE) -> np.ndarray:
    """
    Run HealthyAE on an RGB image and return a per-pixel reconstruction
    error map, resized back to the original image resolution and min-max
    normalized to [0, 1]. Used as the 4th input channel for the Symptom
    Teacher, and at Factory inference time.
    """
    orig_h, orig_w = img_rgb.shape[:2]
    tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2(),
    ])
    tensor = tf(image=img_rgb)["image"].float().unsqueeze(0).to(DEVICE) / 255.0
    recon  = model(tensor)
    error  = (tensor - recon).pow(2).mean(dim=1).squeeze(0).cpu().numpy()  # H×W

    error = cv2.resize(error.astype(np.float32), (orig_w, orig_h),
                       interpolation=cv2.INTER_LINEAR)
    lo, hi = error.min(), error.max()
    if hi - lo > 1e-8:
        error = (error - lo) / (hi - lo)
    else:
        error = np.zeros_like(error)
    return error.astype(np.float32)


def load_healthy_ae(ckpt_path: Path | None = None) -> nn.Module | None:
    """Load a trained HealthyAE for inference. Returns None if missing."""
    ckpt_path = ckpt_path or (HEALTHY_AE_CKPT_DIR / "healthy_ae_best.pth")
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = HealthyAE(latent_dim=ckpt.get("latent_dim", HEALTHY_AE_LATENT_DIM))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — CVAT COCO-FORMAT ANNOTATION PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_symptom_annotations(json_path: Path, images_dir: Path) -> list[dict]:
    """
    Parse a CVAT COCO 1.0 JSON export containing all three annotation labels:
      - MAIZE_LEAF_LABEL_NAME  ("maize-leaf")  — leaf silhouette, all 3 classes
      - MSV_SYMPTOM_LABEL_NAME ("msv-symptom") — chlorotic streaks, MSV images only
      - MLN_SYMPTOM_LABEL_NAME ("mln-symptom") — necrotic patches, MLN images only

    Returns one record per image that has a maize-leaf annotation, including
    HEALTHY images (which receive all-zero symptom masks for both channels).
    This teaches the Symptom Teacher to output nothing on a healthy leaf,
    ensuring graceful degradation when the Student classification head
    misclassifies a HEALTHY image as MSV or MLN.

    Each record: {
        "img_path" : Path,
        "width"    : int,
        "height"   : int,
        "msv_mask" : np.uint8 H×W  (zeros for HEALTHY and MLN images)
        "mln_mask" : np.uint8 H×W  (zeros for HEALTHY and MSV images)
        "category" : str           (inferred from filename prefix CLASS_*)
    }

    Export from CVAT:
      Task menu → Export dataset → COCO 1.0 → extract zip
      Place instances_default.json at SYMPTOM_ANNOTATION_FILE
      (data/gold_standard/annotations/symptom_annotations.json)
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Symptom annotation file not found: {json_path}\n"
            f"Export from CVAT as 'COCO 1.0' and place instances_default.json "
            f"at that path."
        )

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # ── Build lookup tables ───────────────────────────────────────────────────
    id_to_image: dict[int, dict] = {
        img["id"]: img for img in data.get("images", [])
    }
    id_to_cat: dict[int, str] = {
        cat["id"]: cat["name"] for cat in data.get("categories", [])
    }

    def _find_cat_ids(label_name: str) -> set[int]:
        """Return category IDs whose name matches label_name (case-insensitive)."""
        ids = {cid for cid, name in id_to_cat.items()
               if name.lower() == label_name.lower()}
        if not ids:
            print(f"  [WARN] No category '{label_name}' in COCO JSON. "
                  f"Available: {list(id_to_cat.values())}")
        return ids

    leaf_cat_ids = _find_cat_ids(MAIZE_LEAF_LABEL_NAME)
    msv_cat_ids  = _find_cat_ids(MSV_SYMPTOM_LABEL_NAME)
    mln_cat_ids  = _find_cat_ids(MLN_SYMPTOM_LABEL_NAME)

    # Group annotations by image_id per label type
    from collections import defaultdict
    leaf_anns: dict[int, list] = defaultdict(list)
    msv_anns:  dict[int, list] = defaultdict(list)
    mln_anns:  dict[int, list] = defaultdict(list)

    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id")
        img_id = ann.get("image_id")
        if cat_id in leaf_cat_ids:
            leaf_anns[img_id].append(ann)
        elif cat_id in msv_cat_ids:
            msv_anns[img_id].append(ann)
        elif cat_id in mln_cat_ids:
            mln_anns[img_id].append(ann)

    # ── Build records ─────────────────────────────────────────────────────────
    records   = []
    n_skipped = 0
    n_no_leaf = 0

    for img_id, img_info in id_to_image.items():
        # Skip images with no leaf annotation — cannot determine valid region
        if img_id not in leaf_anns:
            n_no_leaf += 1
            continue

        file_name = Path(img_info["file_name"]).name
        img_path  = images_dir / file_name

        if not img_path.exists():
            candidates = [c for c in images_dir.iterdir()
                          if c.name.endswith(file_name)]
            if candidates:
                img_path = candidates[0]
            else:
                n_skipped += 1
                continue

        orig_w = img_info.get("width")
        orig_h = img_info.get("height")
        if not orig_w or not orig_h:
            probe = load_image_rgb(img_path)
            if probe is None:
                n_skipped += 1
                continue
            orig_h, orig_w = probe.shape[:2]

        # Infer category from filename prefix (e.g. "MSV_img001.jpg")
        stem = img_path.stem.upper()
        if stem.startswith("MSV"):
            category = "MSV"
        elif stem.startswith("MLN"):
            category = "MLN"
        else:
            category = "HEALTHY"

        # Build symptom masks — zeros for HEALTHY (intentional training signal)
        msv_mask = _build_mask(msv_anns.get(img_id, []), orig_h, orig_w)
        mln_mask = _build_mask(mln_anns.get(img_id, []), orig_h, orig_w)

        records.append({
            "img_path": img_path,
            "width":    orig_w,
            "height":   orig_h,
            "msv_mask": msv_mask,
            "mln_mask": mln_mask,
            "category": category,
        })

    n_msv     = sum(1 for r in records if r["category"] == "MSV" and r["msv_mask"].sum() > 0)
    n_mln     = sum(1 for r in records if r["category"] == "MLN" and r["mln_mask"].sum() > 0)
    n_healthy = sum(1 for r in records if r["category"] == "HEALTHY")
    print(f"  Parsed {len(records):,} images "
          f"({n_msv} MSV annotated, {n_mln} MLN annotated, "
          f"{n_healthy} HEALTHY with zero masks, "
          f"{n_skipped} missing files, {n_no_leaf} no leaf annotation)")
    return records


def _build_mask(anns: list, height: int, width: int) -> np.ndarray:
    """
    Rasterize a list of COCO annotations (polygon or RLE) into a binary H×W mask.
    Returns an all-zeros mask if anns is empty.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        seg = ann.get("segmentation")
        if seg is None:
            continue
        if isinstance(seg, list):
            for poly_flat in seg:
                if len(poly_flat) < 6:
                    continue
                pts = np.array(poly_flat, dtype=np.float32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        elif isinstance(seg, dict):
            decoded = _decode_coco_rle(seg, height, width)
            if decoded is not None:
                mask = np.clip(mask + decoded, 0, 1).astype(np.uint8)
    return mask


def _decode_coco_rle(seg: dict, height: int, width: int) -> np.ndarray | None:
    """
    Decode a COCO RLE segmentation dict into a binary H×W uint8 mask.
    Tries pycocotools first, then falls back to manual uncompressed decode.
    Install pycocotools for reliable brush mask decoding: pip install pycocotools
    """
    try:
        from pycocotools import mask as coco_mask
        decoded = coco_mask.decode({"counts": seg["counts"], "size": seg["size"]})
        return (decoded > 0).astype(np.uint8)
    except Exception:
        pass

    counts = seg.get("counts")
    size   = seg.get("size", [height, width])
    if isinstance(counts, list):
        try:
            flat = np.zeros(size[0] * size[1], dtype=np.uint8)
            pos, val = 0, 0
            for run in counts:
                flat[pos: pos + run] = val
                pos += run
                val  = 1 - val
            return flat.reshape(size[0], size[1], order="F").astype(np.uint8)
        except Exception:
            pass

    print("  [WARN] Could not decode RLE mask — skipping one brush annotation.")
    return None
# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — SYMPTOM TEACHER MODEL + DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SymptomTeacher(nn.Module):
    """
    EfficientNet-B2 UNet adapted to 4-channel input (RGB + AE error map).
    Two-channel output:
      Ch0 = MSV symptom probability (chlorotic streaks)
      Ch1 = MLN symptom probability (necrotic patches)

    HEALTHY images are included in training with all-zero targets for both
    channels, teaching the model to output nothing on a healthy leaf.
    This ensures graceful degradation when the Student classification head
    misclassifies a HEALTHY image as MSV or MLN.

    Same backbone family as train_teacher.py leaf-silhouette Teacher.
    """

    def __init__(self, encoder_name: str = SYMPTOM_ENCODER):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=4,          # RGB + AE error map
            classes=2,              # Ch0=MSV symptom, Ch1=MLN symptom
            activation=None,
        )

    def forward(self, x):
        return self.net(x)  # [B, 2, H, W] raw logits


class SymptomDataset(Dataset):
    """
    (RGB + AE-error 4-channel input, human symptom mask) pairs for training
    the Symptom Teacher. AE error maps are computed once at dataset-build
    time (not per-epoch) since HealthyAE weights are frozen during this
    training run.
    """

    def __init__(self, records: list[dict], ae_model: nn.Module,
                img_size: int, augment: bool):
        self.records  = records
        self.ae_model = ae_model
        self.img_size = img_size
        if augment:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RandomGamma(p=0.3),
            ], additional_targets={"err": "image", "mask": "mask"})
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
            ], additional_targets={"err": "image", "mask": "mask"})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_rgb = load_image_rgb(rec["img_path"])
        if img_rgb is None:
            return None

        err_map = compute_ae_error_map(self.ae_model, img_rgb)
        err_3ch = np.stack([err_map] * 3, axis=-1)  # fake 3ch for albumentations

        # Resize masks to match image if needed
        h, w = img_rgb.shape[:2]
        msv_mask = rec["msv_mask"]
        mln_mask = rec["mln_mask"]
        if msv_mask.shape != (h, w):
            msv_mask = cv2.resize(msv_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if mln_mask.shape != (h, w):
            mln_mask = cv2.resize(mln_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Stack masks as H×W×2 so albumentations applies identical
        # geometric transforms to both channels simultaneously
        mask_2ch = np.stack([msv_mask, mln_mask], axis=-1).astype(np.uint8)

        out = self.tf(image=img_rgb, err=err_3ch, mask=mask_2ch)
        img_t    = torch.from_numpy(out["image"]).permute(2, 0, 1).float() / 255.0
        err_t    = torch.from_numpy(out["err"][:, :, 0:1]).permute(2, 0, 1).float()
        # mask_2ch after transform: H×W×2 → 2×H×W
        mask_t   = torch.from_numpy(out["mask"]).permute(2, 0, 1).float()  # [2, H, W]

        x = torch.cat([img_t, err_t], dim=0)  # [4, H, W]
        return x, mask_t                        # targets: [2, H, W] float32


def _symptom_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def dice_focal_loss(logits: torch.Tensor, target: torch.Tensor,
                    dice_weight: float = SYMPTOM_DICE_WEIGHT,
                    focal_weight: float = SYMPTOM_FOCAL_WEIGHT,
                    gamma: float = SYMPTOM_FOCAL_GAMMA,
                    pos_weight_factor: float = SYMPTOM_FOCAL_POS_WEIGHT) -> torch.Tensor:
    """
    Combined Dice + Focal loss for small-region segmentation.

    Focal loss (Lin et al. 2017) down-weights easy background pixels by
    (1 - p)^gamma, forcing the model to focus on hard symptom pixels.
    This is the correct fix for the class imbalance problem in symptom
    segmentation where symptom regions are a small fraction of the leaf area.

    pos_weight_factor upweights foreground (symptom) pixels in the focal
    term — further compensates for the foreground/background imbalance.

    Applied independently to each output channel (Ch0=MSV, Ch1=MLN).
    """
    probs  = torch.sigmoid(logits)
    smooth = 1.0

    # ── Dice component ───────────────────────────────────────────────────────
    inter = (probs * target).sum(dim=(-2, -1))        # [B, C]
    union = (probs + target).sum(dim=(-2, -1))         # [B, C]
    dice  = 1.0 - ((2 * inter + smooth) / (union + smooth)).mean()

    # ── Focal component ──────────────────────────────────────────────────────
    # pos_weight balances foreground vs background per-channel
    pos_weight = torch.ones_like(logits) * pos_weight_factor
    bce_per_pixel = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    p_t = probs * target + (1 - probs) * (1 - target)   # probability of true class
    focal = ((1 - p_t) ** gamma * bce_per_pixel).mean()

    return dice_weight * dice + focal_weight * focal


def symptom_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Combined Dice + Focal loss actually used to train the Symptom Teacher,
    with weights read from config.py (SYMPTOM_DICE_WEIGHT / SYMPTOM_FOCAL_WEIGHT /
    SYMPTOM_FOCAL_GAMMA / SYMPTOM_FOCAL_POS_WEIGHT) so they're tunable without
    editing code, and so this function's config-driven behavior is transparent
    rather than hardcoded.

    FIX (previously dice_bce_loss): this used to accept a dice_weight argument
    from SYMPTOM_DICE_BCE_WEIGHT and silently discard it, always calling
    dice_focal_loss() with its hardcoded defaults regardless of config — the
    "0.5 Dice + 0.5 BCE" the old name/config comment implied was never actually
    running; the loss has always been Dice+Focal. That's kept (Focal is the
    better fit for this task's foreground/background imbalance — see
    dice_focal_loss()'s docstring), but the config weights now genuinely apply.
    """
    return dice_focal_loss(
        logits, target,
        dice_weight=SYMPTOM_DICE_WEIGHT,
        focal_weight=SYMPTOM_FOCAL_WEIGHT,
        gamma=SYMPTOM_FOCAL_GAMMA,
        pos_weight_factor=SYMPTOM_FOCAL_POS_WEIGHT,
    )


# Old name kept as a working alias so any external callers don't break.
def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return symptom_loss(logits, target)


def compute_iou(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    inter = int(np.logical_and(pred_binary, gt_binary).sum())
    union = int(np.logical_or(pred_binary, gt_binary).sum())
    return inter / max(union, 1)


def train_symptom_teacher(ae_model: nn.Module, records: list[dict]) -> Path:
    print(f"\n{'─' * 72}")
    print("  Step 3 — Training Symptom Teacher")
    print(f"{'─' * 72}")

    random.shuffle(records)
    n_val = max(1, int(len(records) * SYMPTOM_VAL_SPLIT))
    val_records   = records[:n_val]
    train_records = records[n_val:]
    print(f"  Annotated images: {len(train_records):,} train / {len(val_records):,} val")

    train_ds = SymptomDataset(train_records, ae_model, SYMPTOM_IMG_SIZE, augment=True)
    val_ds   = SymptomDataset(val_records, ae_model, SYMPTOM_IMG_SIZE, augment=False)

    # WeightedRandomSampler: oversample MSV+MLN, undersample HEALTHY
    # so the model sees proportionally more symptom examples per batch.
    # HEALTHY images are still included (for the suppression signal) but at
    # lower frequency — weight 1.0 for symptom images, 0.3 for HEALTHY.
    from torch.utils.data import WeightedRandomSampler
    sample_weights = [
        1.0 if r["category"] in ("MSV", "MLN") else 0.3
        for r in train_records
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_records),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=SYMPTOM_BATCH_SIZE,
                              sampler=sampler, num_workers=0,
                              collate_fn=_symptom_collate)
    val_loader   = DataLoader(val_ds, batch_size=SYMPTOM_BATCH_SIZE,
                              shuffle=False, num_workers=0, collate_fn=_symptom_collate)

    model = SymptomTeacher().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=SYMPTOM_LR,
                              weight_decay=SYMPTOM_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                        factor=0.5, patience=6,
                                                        min_lr=1e-6)

    SYMPTOM_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = SYMPTOM_CKPT_DIR / "symptom_teacher_best.pth"
    metrics_path = LOGS_DIR / "symptom_teacher_metrics.csv"

    best_val_iou = -1.0
    epochs_no_improve = 0
    metric_rows = []

    for epoch in range(1, SYMPTOM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = symptom_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            train_loss += loss.item()
            n_batches  += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_ious_msv     = []   # Ch0 — MSV images only
        val_ious_mln     = []   # Ch1 — MLN images only
        val_ious_healthy = []   # suppression check — should stay near 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch is None:
                    continue
                x, y = batch
                x = x.to(DEVICE)
                logits = model(x)
                probs  = torch.sigmoid(logits).cpu().numpy()
                gt     = y.numpy()
                # Recover categories for this batch from val_records
                batch_start = batch_idx * SYMPTOM_BATCH_SIZE
                batch_recs  = val_records[batch_start: batch_start + len(probs)]
                for p, g, rec in zip(probs, gt, batch_recs):
                    cat = rec.get("category", "MSV")
                    if cat == "MSV":
                        iou = compute_iou((p[0] >= 0.5).astype(np.uint8),
                                          (g[0] >= 0.5).astype(np.uint8))
                        val_ious_msv.append(iou)
                    elif cat == "MLN":
                        iou = compute_iou((p[1] >= 0.5).astype(np.uint8),
                                          (g[1] >= 0.5).astype(np.uint8))
                        val_ious_mln.append(iou)
                    else:  # HEALTHY — check suppression (pred should be near zero)
                        pred_any = float((p >= 0.5).mean())
                        val_ious_healthy.append(pred_any)

        # Primary metric: mean IoU on symptom-annotated images only
        all_symptom_ious = val_ious_msv + val_ious_mln
        val_iou     = float(np.mean(all_symptom_ious)) if all_symptom_ious else 0.0
        msv_iou     = float(np.mean(val_ious_msv))     if val_ious_msv     else 0.0
        mln_iou     = float(np.mean(val_ious_mln))     if val_ious_mln     else 0.0
        healthy_act = float(np.mean(val_ious_healthy))  if val_ious_healthy else 0.0
        sched.step(val_iou)

        print(f"  Epoch {epoch:>3}/{SYMPTOM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_IoU={val_iou:.4f}  "
              f"(MSV={msv_iou:.3f}  MLN={mln_iou:.3f}  "
              f"HEALTHY_act={healthy_act:.3f})")
        metric_rows.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 5),
            "val_iou":     round(val_iou, 5),
            "msv_iou":     round(msv_iou, 5),
            "mln_iou":     round(mln_iou, 5),
            "healthy_act": round(healthy_act, 5),
        })

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(),
                       "val_iou": val_iou,
                       "encoder": SYMPTOM_ENCODER}, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= SYMPTOM_PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {SYMPTOM_PATIENCE} epochs)")
                break

    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print(f"\n  Best Symptom Teacher val_IoU: {best_val_iou:.4f}")
    print(f"  Saved: {best_path}")
    print(f"  Metrics: {metrics_path}")

    if best_val_iou < SYMPTOM_IOU_TARGET_MEAN:
        print(f"\n  [WARN] val_IoU {best_val_iou:.3f} is below target "
              f"{SYMPTOM_IOU_TARGET_MEAN:.2f}.")
        print("         Consider annotating more images, especially early-stage")
        print("         / low-severity examples which are hardest to learn from few samples.")
    else:
        print(f"\n  [OK] val_IoU {best_val_iou:.3f} meets target "
              f"{SYMPTOM_IOU_TARGET_MEAN:.2f}.")

    return best_path


def load_symptom_teacher(ckpt_path: Path | None = None) -> nn.Module | None:
    """Load a trained Symptom Teacher for inference. Returns None if missing."""
    ckpt_path = ckpt_path or (SYMPTOM_CKPT_DIR / "symptom_teacher_best.pth")
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = SymptomTeacher(encoder_name=ckpt.get("encoder", SYMPTOM_ENCODER))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def predict_symptom_mask(symptom_model: nn.Module, ae_model: nn.Module,
                         img_rgb: np.ndarray,
                         category: str = "MSV",
                         img_size: int = SYMPTOM_IMG_SIZE) -> np.ndarray:
    """
    Run the Symptom Teacher on a single RGB image.

    Returns a float32 probability map at the ORIGINAL image resolution
    (sigmoid output, not binarized) for the channel corresponding to
    the image category:
      category="MSV"     → Ch0 (chlorotic streak probability)
      category="MLN"     → Ch1 (necrotic patch probability)
      category="HEALTHY" → all-zeros (no symptom signal expected)

    Mirrors the contract of compute_lab_soft_confidence() so factory_master.py
    can swap it in with minimal changes — it still receives a single H×W float32
    map per image.
    """
    orig_h, orig_w = img_rgb.shape[:2]

    # HEALTHY: skip inference entirely — return zeros
    if category == "HEALTHY":
        return np.zeros((orig_h, orig_w), dtype=np.float32)

    err_map = compute_ae_error_map(ae_model, img_rgb)

    img_t = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
    ])(image=img_rgb)["image"]
    err_resized = cv2.resize(err_map, (img_t.shape[1], img_t.shape[0]),
                             interpolation=cv2.INTER_LINEAR)

    img_tensor = torch.from_numpy(img_t).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    err_tensor = torch.from_numpy(err_resized).unsqueeze(0).unsqueeze(0).float()
    x = torch.cat([img_tensor, err_tensor], dim=1).to(DEVICE)

    logits = symptom_model(x)          # [1, 2, H, W]
    probs  = torch.sigmoid(logits)     # [1, 2, H, W]

    # Select channel by category
    ch = 0 if category == "MSV" else 1   # Ch0=MSV, Ch1=MLN
    prob = probs[0, ch].cpu().numpy()    # H×W

    prob = cv2.resize(prob.astype(np.float32), (orig_w, orig_h),
                      interpolation=cv2.INTER_LINEAR)
    return prob


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — OPTIONAL: LAB COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_against_lab(symptom_model: nn.Module, ae_model: nn.Module,
                        val_records: list[dict]) -> None:
    """
    For the held-out validation split, run BOTH the legacy LAB pipeline
    (factory_master.compute_lab_hard_mask) and the new Symptom Teacher,
    and report IoU(LAB, human) vs IoU(SymptomTeacher, human) side by side.
    Uses a simple Otsu silhouette as a stand-in leaf mask for this
    comparison, since Teacher leaf masks may not be available for every
    gold-standard image at this stage of the pipeline.
    """
    try:
        from factory_master import compute_lab_hard_mask
    except ImportError as e:
        print(f"  [SKIP] Could not import factory_master for comparison: {e}")
        return

    rows = []
    for rec in val_records:
        img_rgb = load_image_rgb(rec["img_path"])
        if img_rgb is None:
            continue
        # FIX: rec["mask"] doesn't exist in this record's schema (only
        # "msv_mask" / "mln_mask" do — see parse_symptom_annotations()'s
        # docstring) — this line raised a KeyError on the first iteration,
        # every time, which is why symptom_vs_lab_comparison.csv never got
        # produced. The correct gt_mask is selected by category a few lines
        # below anyway, so this line was both wrong and redundant.

        # Simple Otsu silhouette as a stand-in leaf mask for this comparison
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        otsu_sil = (cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0).astype(np.uint8)

        # Infer category from filename prefix (CATEGORY_filename.jpg convention)
        category = rec.get("category", next((c for c in CLASSES if rec["img_path"].name.upper().startswith(c)), "MSV"))

        lab_mask = (compute_lab_hard_mask(img_rgb, otsu_sil, category) > 0).astype(np.uint8)
        symptom_prob = predict_symptom_mask(symptom_model, ae_model, img_rgb, category=category)
        symptom_mask = (symptom_prob >= 0.5).astype(np.uint8)

        # Select the relevant ground-truth mask channel for this category
        if category == "MSV":
            gt_mask = rec["msv_mask"]
        elif category == "MLN":
            gt_mask = rec["mln_mask"]
        else:
            # HEALTHY: both channels should be zero — use MSV channel as reference
            gt_mask = rec["msv_mask"]

        if gt_mask.shape != img_rgb.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (img_rgb.shape[1], img_rgb.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        rows.append({
            "image":               rec["img_path"].name,
            "category":            category,
            "iou_lab":             round(compute_iou(lab_mask, gt_mask), 4),
            "iou_symptom_teacher": round(compute_iou(symptom_mask, gt_mask), 4),
        })

    if not rows:
        print("  [SKIP] No comparison data collected.")
        return

    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "symptom_vs_lab_comparison.csv"
    df.to_csv(out_path, index=False)

    print(f"\n  LAB vs Symptom Teacher comparison ({len(df)} held-out images):")
    print(f"    Mean IoU (LAB pipeline)     : {df['iou_lab'].mean():.4f}")
    print(f"    Mean IoU (Symptom Teacher)  : {df['iou_symptom_teacher'].mean():.4f}")
    for cls in CLASSES:
        cls_df = df[df["category"] == cls]
        if cls_df.empty:
            continue
        print(f"    [{cls}] LAB={cls_df['iou_lab'].mean():.3f}  "
              f"SymptomTeacher={cls_df['iou_symptom_teacher'].mean():.3f}  "
              f"(n={len(cls_df)})")
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the human-supervised Symptom Teacher.")
    parser.add_argument("--skip-ae", action="store_true",
                       help="Skip HealthyAE training; load existing checkpoint.")
    parser.add_argument("--compare-lab", action="store_true",
                       help="After training, compare against the legacy LAB pipeline "
                            "on the held-out split and write a comparison CSV.")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 72)
    print("  Yellow MAIze | Phase 3b: Symptom Teacher Training")
    print("=" * 72)

    # ── Step 1: HealthyAE ─────────────────────────────────────────────────────
    if args.skip_ae:
        print("\n  --skip-ae set: loading existing HealthyAE checkpoint.")
        ae_model = load_healthy_ae()
        if ae_model is None:
            print("[FATAL] No HealthyAE checkpoint found. Run without --skip-ae first.")
            return
    else:
        train_healthy_ae()
        ae_model = load_healthy_ae()

    # ── Step 2: Parse symptom annotations ────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  Step 2 — Parsing symptom annotations")
    print(f"{'─' * 72}")

    if not SYMPTOM_ANNOTATION_FILE.exists():
        print(f"[FATAL] {SYMPTOM_ANNOTATION_FILE} not found.")
        print("  Export from CVAT: Task menu → Export dataset → COCO 1.0")
        print("  Extract zip → place instances_default.json at that path.")
        print(f"  Annotate using labels: {MAIZE_LEAF_LABEL_NAME}, "
              f"{MSV_SYMPTOM_LABEL_NAME}, {MLN_SYMPTOM_LABEL_NAME}")
        return

    records = parse_symptom_annotations(SYMPTOM_ANNOTATION_FILE, GOLD_IMAGES_DIR)

    n_symptom_annotated = sum(
        1 for r in records
        if r["category"] in ("MSV", "MLN")
        and (r["msv_mask"].sum() > 0 or r["mln_mask"].sum() > 0)
    )
    if n_symptom_annotated < SYMPTOM_MIN_ANNOTATIONS:
        print(f"\n  [WARN] Only {n_symptom_annotated} symptom-annotated MSV/MLN images "
              f"(target {SYMPTOM_MIN_ANNOTATIONS}). HEALTHY images with zero masks "
              f"are included in training regardless.")
        answer = input("  Proceed with training anyway? [y/N]: ")
        if answer.strip().lower() != "y":
            print("  Aborted. Annotate more images and re-run.")
            return

    if len(records) < 20:
        print(f"[FATAL] Only {len(records)} usable records. Check annotation export.")
        print("  Ensure 'maize-leaf' annotations exist — images without a leaf")
        print("  annotation are excluded entirely from training.")
        return

    # ── Step 3: Train Symptom Teacher ─────────────────────────────────────────
    random.shuffle(records)
    n_val = max(1, int(len(records) * SYMPTOM_VAL_SPLIT))
    held_out_for_comparison = records[:n_val]  # same split train_symptom_teacher uses

    train_symptom_teacher(ae_model, records)
    symptom_model = load_symptom_teacher()

    # ── Step 4: Optional LAB comparison ───────────────────────────────────────
    if args.compare_lab and symptom_model is not None:
        print(f"\n{'─' * 72}")
        print("  Step 4 — Comparing against legacy LAB pipeline")
        print(f"{'─' * 72}")
        compare_against_lab(symptom_model, ae_model, held_out_for_comparison)

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'═' * 72}")
    print(f"  Done in {elapsed:.1f} min")
    print(f"  HealthyAE       : {HEALTHY_AE_CKPT_DIR / 'healthy_ae_best.pth'}")
    print(f"  Symptom Teacher : {SYMPTOM_CKPT_DIR / 'symptom_teacher_best.pth'}")
    print(f"\n  Set SYMPTOM_TEACHER_DEPLOYED = True in config.py (default) so")
    print(f"  factory_master.py uses this model instead of the LAB pipeline.")
    print(f"\n  NEXT STEP: python factory_master.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
