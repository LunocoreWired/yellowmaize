"""
================================================================================
 train_bouncer.py — Phase 0b: Bouncer Training
================================================================================
 PURPOSE:
   Train and compare Bouncer variants for the maize-vs-not-maize gate:
     - gabor_lbp         : Traditional CV baseline (no training — direct eval)
     - mobilenet_v2      : Lightweight CNN (thesis primary architecture)
     - mobilenet_v3_large: Deployed neural gate (primary)
     - edgevit_xxs       : Hybrid ViT comparison

   NOTE: patchcore removed — anomaly detection is architecturally mismatched
   for supervised binary classification and has no TFLite deployment path.
   gabor_lbp is sufficient as the single non-neural baseline.

   Deployed model: MobileNetV3-Large binary classifier
   Two-stage gate at inference: OpenCV heuristic → neural classifier

   Each trained variant is evaluated on the val split.
   Threshold selection: maximize geometric mean of recall×specificity
   subject to maize recall ≥ 95%.
   Best checkpoint saved to checkpoints/bouncer/.

 OUTPUTS:
   checkpoints/bouncer/bouncer_{variant}_best.pth
   logs/bouncer_{variant}_metrics.csv
   logs/bouncer_comparison.csv
================================================================================
"""

import csv
import math
import random
import sys
import time
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from config import (
    BOUNCER_BATCH_SIZE,
    BOUNCER_CKPT_DIR,
    BOUNCER_DATASET_DIR,
    BOUNCER_DEPLOYED_VARIANT,
    BOUNCER_EPOCHS,
    BOUNCER_IMG_SIZE,
    BOUNCER_LR,
    BOUNCER_MIN_MAIZE_RECALL,
    BOUNCER_PATIENCE,
    BOUNCER_THRESHOLD,
    BOUNCER_VAL_SPLIT,
    BOUNCER_WEIGHT_DECAY,
    GLOBAL_MANIFEST,
    LOGS_DIR,
    SEED,
    VALID_EXTENSIONS,
)
from image_utils import load_image_clahe  # EXIF correction + CLAHE
from scripts.safe_collate import get_skip_count, reset_skip_counter, safe_collate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import models
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
)

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════


def set_seeds(seed: int) -> None:
    """
    Set all random seeds for full reproducibility.
    cudnn.deterministic=True, benchmark=False is deliberate —
    benchmark=True gives ~15% speed but non-deterministic results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # reproducibility over speed
    torch.backends.cudnn.benchmark = False  # see config.CUDNN_BENCHMARK


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════


def make_transforms(img_size: int, is_train: bool):
    """
    Letterbox padding (aspect-ratio preserving) + augmentations.
    CLAHE is applied in load_image_clahe() before this transform pipeline.
    Non-maize class does NOT get domain augmentations (they're already diverse).
    """
    if is_train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=10, p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


class BouncerDataset(Dataset):
    """Binary dataset: maize (label=1) vs not_maize (label=0)."""

    def __init__(self, root: Path, transform=None):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform

        maize_dir = root / "maize"
        nonmaize_dir = root / "not_maize"

        for ext in VALID_EXTENSIONS:
            for p in maize_dir.rglob(f"*{ext}"):
                self.samples.append((p, 1))
            for p in nonmaize_dir.rglob(f"*{ext}"):
                self.samples.append((p, 0))

        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        # load_image_clahe: EXIF correction + CLAHE + corrupt guard
        img = load_image_clahe(path)
        if img is None:
            return None  # safe_collate will skip this sample
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════


def build_mobilenet_v2() -> nn.Module:
    """
    MobileNetV2 binary classifier.
    Added as B3 — thesis title explicitly names MobileNetV2 as the primary
    architecture; it must appear in every ablation including the Bouncer.
    """
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    # Replace classifier head: 1280 → 1 (binary logit)
    model.classifier[-1] = nn.Linear(1280, 1)
    return model


def build_mobilenet_v3_large() -> nn.Module:
    """MobileNetV3-Large binary classifier."""
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    # Replace classifier head: 960 → 1 (binary logit)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def build_edgevit_xxs() -> nn.Module:
    """
    EdgeViT-XXS binary classifier via timm.
    Pan et al. 2022 — Local-Global-Local self-attention.
    Falls back to MobileNetV2 if timm/edgevit not available.
    """
    try:
        import timm

        model = timm.create_model("edgevit_xxs", pretrained=True, num_classes=1)
        return model
    except Exception as e:
        print(
            f"  [WARN] EdgeViT-XXS not available ({e}). "
            f"Using MobileNetV2 as placeholder."
        )
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(1280, 1)
        return model


VARIANT_BUILDERS = {
    "mobilenet_v2": build_mobilenet_v2,
    "mobilenet_v3_large": build_mobilenet_v3_large,
    "edgevit_xxs": build_edgevit_xxs,
}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        if batch is None:
            continue
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping — prevents spikes from destabilising training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.50):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)
    preds = (all_probs >= threshold).astype(int)

    report = classification_report(
        all_labels,
        preds,
        target_names=["not_maize", "maize"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": report["accuracy"],
        "maize_prec": report["maize"]["precision"],
        "maize_rec": report["maize"]["recall"],
        "maize_f1": report["maize"]["f1-score"],
        "specificity": report["not_maize"]["recall"],  # TN rate = specificity
        "all_probs": all_probs,
        "all_labels": all_labels,
    }


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    min_maize_recall: float = BOUNCER_MIN_MAIZE_RECALL,
) -> tuple[float, dict]:
    """
    Find sigmoid threshold that maximizes the geometric mean of recall
    and specificity, subject to maize recall >= min_maize_recall.

    FIX (v2): Previously maximized specificity alone, which caused the
    algorithm to select a near-1.0 threshold on high-performing models
    (ROC AUC ~1.0) — resulting in 88%+ filter rate at inference.
    Now maximizes geometric mean (recall * specificity)^0.5 instead,
    which balances both metrics and produces a usable production threshold.
    Hard cap at 0.70 as an additional safeguard.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
    best_thresh = 0.50
    best_score = 0.0

    for thresh, tpr_val, fpr_val in zip(thresholds, tpr, fpr):
        recall = tpr_val
        spec = 1.0 - fpr_val
        if recall >= min_maize_recall:
            # Geometric mean balances recall and specificity equally
            score = (recall * spec) ** 0.5
            if score > best_score:
                best_score = score
                best_thresh = float(thresh)

    # Hard cap — never save a threshold above 0.70 for production use.
    # A model with ROC AUC ~1.0 needs no threshold above 0.70 to maintain
    # perfect discrimination; anything higher just over-filters at inference.
    best_thresh = min(best_thresh, 0.70)

    preds = (probs >= best_thresh).astype(int)
    preds_at_best = (probs >= best_thresh).astype(int)
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds_at_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec_at_best = tn / max(tn + fp, 1)
    return best_thresh, {
        "threshold": best_thresh,
        "specificity": spec_at_best,
        "roc_auc": float(auc),
        "maize_recall": float(np.sum((preds_at_best == 1) & (labels == 1)))
        / max(float(np.sum(labels == 1)), 1),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GABOR + LBP BASELINE (no training)
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_gabor_lbp(val_maize: list[Path], val_nonmaize: list[Path]) -> dict:
    """
    Gabor filter + LBP texture-based maize/non-maize classifier.
    No training data required — pure signal processing baseline.
    """
    try:
        import cv2
        from skimage.color import rgb2gray
        from skimage.feature import local_binary_pattern
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
    except ImportError:
        print(
            "  [WARN] cv2/skimage/sklearn not available. Skipping Gabor+LBP evaluation."
        )
        return {"error": "dependencies missing"}

    IMG_SIZE = BOUNCER_IMG_SIZE

    def extract_features(path: Path) -> np.ndarray | None:
        try:
            img = cv2.imread(str(path))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            gray = rgb2gray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Gabor features (4 frequencies × 4 orientations = 16 filters)
            gabor_feats = []
            for freq in [0.1, 0.2, 0.3, 0.4]:
                for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                    kern = cv2.getGaborKernel((21, 21), 4.0, theta, 1.0 / freq, 0.5, 0)
                    filtered = cv2.filter2D(gray.astype(np.float32), -1, kern)
                    gabor_feats.extend([filtered.mean(), filtered.std()])

            # LBP features (radius=3, 24 points)
            lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
            return np.concatenate([gabor_feats, hist])
        except Exception:
            return None

    print("    Extracting Gabor+LBP features ...")
    X, y = [], []
    for p in val_maize[:500]:
        f = extract_features(p)
        if f is not None:
            X.append(f)
            y.append(1)
    for p in val_nonmaize[:500]:
        f = extract_features(p)
        if f is not None:
            X.append(f)
            y.append(0)

    if len(X) < 50:
        return {"error": "insufficient features extracted"}

    X, y = np.array(X), np.array(y)
    # Proper stratified train/test split (not a mid-split of the same pool)
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.40, random_state=SEED, stratify=y
    )
    clf = Pipeline([("scaler", StandardScaler()), ("svm", LinearSVC(max_iter=2000))])
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    report = classification_report(
        y_te,
        preds,
        target_names=["not_maize", "maize"],
        output_dict=True,
        zero_division=0,
    )

    # ── CPU inference latency ────────────────────────────────────────────────
    # Full per-image pipeline cost: Gabor+LBP feature extraction + SVM predict.
    # Not directly comparable to the neural variants' forward-pass-only latency
    # (this includes feature engineering the neural models don't need), but
    # reported in the same lat_cpu_ms column for a complete efficiency picture
    # in the comparison table.
    latencies = []
    for p in (val_maize[:5] + val_nonmaize[:5]):
        f = extract_features(p)
        if f is None:
            continue
        t0 = time.perf_counter()
        extract_features(p)
        clf.predict(f.reshape(1, -1))
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_lat = round(float(np.mean(latencies)), 2) if latencies else "N/A"

    return {
        "accuracy": report["accuracy"],
        "maize_prec": report["maize"]["precision"],
        "maize_rec": report["maize"]["recall"],
        "maize_f1": report["maize"]["f1-score"],
        "specificity": report["not_maize"]["recall"],
        "lat_cpu_ms": avg_lat,
        "note": "Gabor+LBP+LinearSVC — traditional CV baseline",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def train_variant(variant_name: str) -> dict:
    """Train one neural Bouncer variant. Returns best val metrics."""
    set_seeds(SEED)
    BOUNCER_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  Variant: {variant_name}")
    print(f"{'─' * 60}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_tf = make_transforms(BOUNCER_IMG_SIZE, is_train=True)
    val_tf = make_transforms(BOUNCER_IMG_SIZE, is_train=False)

    full_train_ds = BouncerDataset(BOUNCER_DATASET_DIR, transform=train_tf)
    full_val_ds = BouncerDataset(BOUNCER_DATASET_DIR, transform=val_tf)

    n_total = len(full_train_ds)
    n_val = int(n_total * BOUNCER_VAL_SPLIT)
    n_train = n_total - n_val

    # Same indices for both (different transforms — separate instances)
    generator = torch.Generator().manual_seed(SEED)
    train_idx, val_idx = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    from torch.utils.data import Subset

    train_ds = Subset(full_train_ds, train_idx.indices)
    val_ds = Subset(full_val_ds, val_idx.indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BOUNCER_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BOUNCER_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=safe_collate,
    )

    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VARIANT_BUILDERS[variant_name]().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BOUNCER_LR,
        weight_decay=BOUNCER_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=BOUNCER_EPOCHS, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    ckpt_path = BOUNCER_CKPT_DIR / f"bouncer_{variant_name}_best.pth"
    metrics_log = LOGS_DIR / f"bouncer_{variant_name}_metrics.csv"

    best_f1 = 0.0
    no_improve = 0
    log_rows = []

    for epoch in range(1, BOUNCER_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_m = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        f1 = val_m["maize_f1"]
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_m["loss"], 5),
            "accuracy": round(val_m["accuracy"], 4),
            "maize_prec": round(val_m["maize_prec"], 4),
            "maize_rec": round(val_m["maize_rec"], 4),
            "maize_f1": round(f1, 4),
            "specificity": round(val_m["specificity"], 4),
            "lr": round(scheduler.get_last_lr()[0], 7),
        }
        log_rows.append(row)

        print(
            f"  Ep {epoch:02d}/{BOUNCER_EPOCHS} | "
            f"loss {train_loss:.4f} → {val_m['loss']:.4f} | "
            f"F1 {f1:.4f} | spec {val_m['specificity']:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": row,
                    "variant": variant_name,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= BOUNCER_PATIENCE:
                print(f"  Early stop at epoch {epoch}.")
                break

    # ── Write metrics CSV ──────────────────────────────────────────────────────
    with open(metrics_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # ── Empirical threshold selection ──────────────────────────────────────────
    print(f"  Loading best checkpoint for threshold tuning ...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    val_m = evaluate(model, val_loader, DEVICE, threshold=0.50)

    best_thresh, thresh_info = find_best_threshold(
        val_m["all_probs"], val_m["all_labels"]
    )

    print(f"  Empirical threshold: {best_thresh:.4f}")
    print(f"  Specificity        : {thresh_info['specificity']:.4f}")
    print(f"  Maize recall       : {thresh_info['maize_recall']:.4f}")

    # Save threshold back into checkpoint
    ckpt["threshold"] = best_thresh
    ckpt["threshold_metrics"] = thresh_info
    torch.save(ckpt, ckpt_path)

    # ── CPU inference latency ────────────────────────────────────────────────
    # Measured on CPU (not the training device) since the Bouncer's deployment
    # target is mobile CPU, not a training GPU. Same protocol as train_teacher.py:
    # 20 forward passes on a dummy input, first 5 discarded as warm-up, mean of
    # the remaining 15 reported. This is what justifies choosing among
    # classification-tied neural variants (see find_best_threshold's docstring
    # and the deployment-selection discussion in Chapter 4) on efficiency rather
    # than accuracy — without this, that reasoning was architectural but unmeasured.
    model_cpu = VARIANT_BUILDERS[variant_name]().eval()
    ckpt_cpu = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cpu.load_state_dict(ckpt_cpu["model_state"])

    dummy = torch.zeros(1, 3, BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE)
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            model_cpu(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)
    avg_lat = round(float(np.mean(latencies[5:])), 2)
    print(f"  Inference latency (CPU, {BOUNCER_IMG_SIZE}px): {avg_lat} ms/image")

    del model_cpu, ckpt_cpu

    return {
        "variant": variant_name,
        "best_f1": round(best_f1, 4),
        "threshold": round(best_thresh, 4),
        "specificity": round(thresh_info["specificity"], 4),
        "maize_recall": round(thresh_info["maize_recall"], 4),
        "roc_auc": round(thresh_info["roc_auc"], 4),
        "TP": thresh_info.get("TP", ""),
        "FP": thresh_info.get("FP", ""),
        "TN": thresh_info.get("TN", ""),
        "FN": thresh_info.get("FN", ""),
        "lat_cpu_ms": avg_lat,
        "ckpt": str(ckpt_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TWO-STAGE INFERENCE HELPERS
# Imported from scripts/bouncer_inference.py — single source of truth shared
# with factory_master.py. No SAM2/Teacher dependencies pulled in.
# ══════════════════════════════════════════════════════════════════════════════

from scripts.bouncer_inference import heuristic_prefilter, neural_bouncer


# ══════════════════════════════════════════════════════════════════════════════
# END-TO-END ADMISSION RATE
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_admission_rate(variant: str) -> dict:
    """
    Run the deployed Bouncer on ALL global test-split maize images.
    Reports: total tested, passed, rejected, admission rate, false rejection %.
    This adjusts the Student's effective end-to-end recall.
    """
    import pandas as pd

    if not GLOBAL_MANIFEST.exists():
        print("  [WARN] Global manifest not found — skip admission rate eval.")
        return {}

    manifest = pd.read_csv(GLOBAL_MANIFEST)
    test_maize = manifest[
        (manifest["split"] == "test")
        & (manifest["category"].isin(["HEALTHY", "MSV", "MLN"]))
    ]

    ckpt_path = BOUNCER_CKPT_DIR / f"bouncer_{variant}_best.pth"
    if not ckpt_path.exists():
        print(f"  [WARN] Checkpoint not found for admission rate: {ckpt_path}")
        return {}

    model = VARIANT_BUILDERS[variant]().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    threshold = ckpt.get("threshold", BOUNCER_THRESHOLD)
    model.eval()

    n_total = n_passed = n_heuristic_fail = 0
    val_tf = make_transforms(BOUNCER_IMG_SIZE, is_train=False)

    for _, row in test_maize.iterrows():
        img_path = Path(row["source_path"])
        img = load_image_clahe(img_path)
        if img is None:
            continue
        n_total += 1

        if not heuristic_prefilter(img):
            n_heuristic_fail += 1
            continue

        passed = neural_bouncer(img, model, threshold)
        if passed:
            n_passed += 1

    admission_rate = n_passed / max(n_total, 1)
    false_rejection = 1.0 - admission_rate
    heuristic_rej_rate = n_heuristic_fail / max(n_total, 1)

    result = {
        "variant": variant,
        "n_test_maize": n_total,
        "n_passed": n_passed,
        "n_rejected": n_total - n_passed,
        "n_heuristic_reject": n_heuristic_fail,
        "admission_rate": round(admission_rate, 4),
        "false_rejection_rate": round(false_rejection, 4),
        "heuristic_rej_rate": round(heuristic_rej_rate, 4),
    }

    print(f"  Admission rate  : {admission_rate * 100:.1f}%")
    print(
        f"  False rejection : {false_rejection * 100:.1f}%  "
        f"(heuristic: {heuristic_rej_rate * 100:.1f}%)"
    )

    # Save
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    adm_path = LOGS_DIR / f"bouncer_admission_rate_{variant}.csv"
    import csv as _csv

    with open(adm_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=result.keys())
        w.writeheader()
        w.writerow(result)
    print(f"  Saved: {adm_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PATCHCORE ANOMALY DETECTOR BASELINE (offline evaluation only)
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_patchcore() -> dict:
    """
    PatchCore anomaly detector baseline (offline evaluation — not deployed).
    Trains on train-split maize images (one-class), evaluates on val split.
    Reports AUROC and FPR95 (FPR at 95% TPR) — standard anomaly detection
    metrics.

    Requires: pip install anomalib
    Returns metrics dict or empty dict if anomalib not installed.
    """
    try:
        import torchvision.transforms as T
        from anomalib.data.utils import read_image
        from anomalib.models import Patchcore
        from anomalib.utils.metrics import AUROC
        from sklearn.metrics import roc_curve

        print("  PatchCore: anomalib found")
    except ImportError:
        print("  [WARN] anomalib not installed — PatchCore skipped.")
        print("         pip install anomalib")
        return {
            "variant": "patchcore",
            "note": "anomalib not installed",
            "auroc": "N/A",
            "fpr95": "N/A",
            "specificity": "N/A",
            "maize_recall": "N/A",
        }

    if not GLOBAL_MANIFEST.exists():
        print("  [WARN] Global manifest not found — PatchCore skipped.")
        return {}

    import pandas as pd

    manifest = pd.read_csv(GLOBAL_MANIFEST)

    # Collect train-split maize images (positive/inlier class)
    train_maize = manifest[
        (manifest["split"] == "train")
        & (manifest["category"].isin(["HEALTHY", "MSV", "MLN"]))
    ]["source_path"].tolist()

    # Collect val-split for evaluation (maize = inlier, test with non-maize)
    val_maize = manifest[
        (manifest["split"] == "val")
        & (manifest["category"].isin(["HEALTHY", "MSV", "MLN"]))
    ]["source_path"].tolist()

    # Collect non-maize images for val evaluation
    nonmaize_dir = BOUNCER_DATASET_DIR / "not_maize"
    val_nonmaize = list(nonmaize_dir.glob("*.jpg"))[: len(val_maize)]

    if not train_maize or not val_maize:
        print("  [WARN] Not enough images for PatchCore evaluation.")
        return {}

    # PatchCore feature extraction setup
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        import torchvision.models as tvm

        # Use ResNet18 backbone for PatchCore feature extraction
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.eval().to(DEVICE)

        def extract_features(paths: list, label: str) -> tuple:
            feats, labels_out = [], []
            from PIL import Image as _PIL

            for p in paths:
                img = load_image_clahe(Path(p))
                if img is None:
                    continue
                t = transform(_PIL.fromarray(img)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    f = backbone(t).mean(dim=[2, 3]).cpu().numpy()
                feats.append(f[0])
                labels_out.append(label)
            return feats, labels_out

        print(
            f"  PatchCore: extracting features from {len(train_maize):,} train images ..."
        )
        train_feats, _ = extract_features(train_maize[:2000], "maize")  # cap for speed

        print(f"  PatchCore: extracting val features ...")
        val_feats_m, val_lbl_m = extract_features(val_maize[:500], "maize")
        val_feats_nm, val_lbl_nm = extract_features(
            [str(p) for p in val_nonmaize[:500]], "nonmaize"
        )

        if not train_feats or not val_feats_m:
            return {"variant": "patchcore", "note": "feature extraction failed"}

        train_arr = np.array(train_feats)
        val_arr = np.array(val_feats_m + val_feats_nm)
        val_labels = [1] * len(val_feats_m) + [0] * len(val_feats_nm)  # 1=maize(inlier)

        # Anomaly score = nearest-neighbor distance to training set
        # Lower score = more similar to training (more likely maize)
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=3, algorithm="ball_tree", metric="euclidean")
        nn.fit(train_arr)
        distances, _ = nn.kneighbors(val_arr)
        anomaly_scores = distances.mean(axis=1)  # high = anomalous = non-maize

        # For anomaly detection: high score = non-maize (label 0 in our convention)
        # Invert for ROC: we want P(maize) so use -anomaly_score
        from sklearn.metrics import roc_auc_score

        auroc = roc_auc_score(val_labels, -anomaly_scores)

        # FPR at 95% TPR
        fpr, tpr, thresholds = roc_curve(val_labels, -anomaly_scores)
        fpr95_idx = np.where(tpr >= 0.95)[0]
        fpr95 = float(fpr[fpr95_idx[0]]) if len(fpr95_idx) > 0 else 1.0

        # Specificity at optimal threshold (same threshold logic as neural Bouncer)
        optimal_thresh_idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[optimal_thresh_idx]
        preds = (-anomaly_scores >= opt_thresh).astype(int)
        from sklearn.metrics import confusion_matrix as _cm

        cm_pc = _cm(val_labels, preds)
        tn, fp, fn, tp = cm_pc.ravel() if cm_pc.size == 4 else (0, 0, 0, 0)
        spec_pc = tn / max(tn + fp, 1)
        recall_pc = tp / max(tp + fn, 1)

        result = {
            "variant": "patchcore",
            "auroc": round(float(auroc), 4),
            "fpr95": round(float(fpr95), 4),
            "specificity": round(float(spec_pc), 4),
            "maize_recall": round(float(recall_pc), 4),
            "note": "Offline eval — nearest-neighbour anomaly detector",
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        }
        print(
            f"  PatchCore AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}  "
            f"Spec: {spec_pc:.4f}  Recall: {recall_pc:.4f}"
        )
        return result

    except Exception as e:
        print(f"  [WARN] PatchCore evaluation error: {e}")
        return {"variant": "patchcore", "note": f"error:{str(e)[:60]}"}


def main() -> None:
    _t_start = time.time()
    set_seeds(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Phase 0b: Bouncer Training")
    print("=" * 72)

    if not BOUNCER_DATASET_DIR.is_dir():
        print("[FATAL] Bouncer dataset not found. Run create_bouncer_dataset.py first.")
        return

    comparison_rows = []

    # ── Train neural variants ─────────────────────────────────────────────────
    for variant in ["mobilenet_v2", "mobilenet_v3_large", "edgevit_xxs"]:
        row = train_variant(variant)
        comparison_rows.append(row)

        # ---TO PREVENT OOM CRASHES ---
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    # ── Gabor+LBP baseline (no training) ─────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Variant: gabor_lbp (traditional CV baseline)")
    print(f"{'─' * 60}")

    maize_dir = BOUNCER_DATASET_DIR / "maize"
    nonmaize_dir = BOUNCER_DATASET_DIR / "not_maize"
    val_maize = list(maize_dir.rglob("*.jpg"))[:1000]
    val_nonmaize = list(nonmaize_dir.rglob("*.jpg"))[:1000]

    gabor_result = evaluate_gabor_lbp(val_maize, val_nonmaize)
    comparison_rows.append(
        {
            "variant": "gabor_lbp",
            "best_f1": gabor_result.get("maize_rec", "N/A"),
            "threshold": "N/A",
            "specificity": gabor_result.get("specificity", "N/A"),
            "maize_recall": gabor_result.get("maize_rec", "N/A"),
            "lat_cpu_ms": gabor_result.get("lat_cpu_ms", "N/A"),
            "ckpt": "N/A",
        }
    )

    # ── Save comparison CSV ───────────────────────────────────────────────────
    comp_path = LOGS_DIR / "bouncer_comparison.csv"
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        # 1. Grab every unique column header across all models
        all_headers = []
        for row in comparison_rows:
            for key in row.keys():
                if key not in all_headers:
                    all_headers.append(key)

        # 2. Write the CSV using the complete header list
        writer = csv.DictWriter(f, fieldnames=all_headers)
        writer.writeheader()
        writer.writerows(comparison_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  Bouncer comparison summary:")
    print(f"  {'Variant':<22} {'F1':>6} {'Spec':>6} {'Recall':>6}")
    print(f"  {'─' * 22} {'─' * 6} {'─' * 6} {'─' * 6}")
    for r in comparison_rows:
        f1 = f"{r['best_f1']:.4f}" if isinstance(r["best_f1"], float) else r["best_f1"]
        sp = (
            f"{r['specificity']:.4f}"
            if isinstance(r["specificity"], float)
            else r["specificity"]
        )
        rec = (
            f"{r['maize_recall']:.4f}"
            if isinstance(r["maize_recall"], float)
            else r["maize_recall"]
        )
        print(f"  {r['variant']:<22} {f1:>6} {sp:>6} {rec:>6}")

    print(f"\n  Comparison CSV: {comp_path}")

    # ── Admission rate on held-out test-split maize images ────────────────────
    # FIX: this function existed but was never called from main() — it was
    # dead code. Running it here, against the deployed variant, is what
    # actually produces the false-rejection-rate figure that Chapter 4's
    # admission-rate table depends on; nothing upstream of this call could
    # have produced it, no matter how many times the comparison above ran.
    print(f"\n{'─' * 72}")
    print(f"  Admission rate — deployed variant ({BOUNCER_DEPLOYED_VARIANT})")
    print(f"{'─' * 72}")
    evaluate_admission_rate(BOUNCER_DEPLOYED_VARIANT)

    print(f"\n  NEXT STEP: python sample_15000.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
