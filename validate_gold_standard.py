"""
================================================================================
 validate_gold_standard.py — Gold Standard IoU Validation
================================================================================
 PURPOSE:
   Validate SAM2 pseudo-masks, Teacher predictions, and Student predictions
   against 300 human-annotated leaf silhouette masks (the gold standard).

   The SAME 300 images are evaluated across all three artifacts, enabling a
   fair chain comparison:

       SAM2 → Teacher → Student
         ↓        ↓         ↓
       IoU      IoU       IoU     (all vs. same human masks)

   This is required for thesis defense. Without it, the committee can
   legitimately challenge the entire pseudo-label foundation.

 METRICS:
   IoU (Intersection over Union) per image — binary segmentation (leaf vs
   background), so IoU and mIoU are equivalent here. Reported as:
     - Per-image IoU
     - Mean IoU ± std per class (HEALTHY / MSV / MLN)
     - Overall mean IoU ± std across all 300 images
     - Per-image flagging when IoU < GOLD_IOU_WARN_THRESHOLD (0.75)

 INPUTS (place these before running):
   data/gold_standard/images/        ← 300 raw images (from sample_gold_standard.py)
   data/gold_standard/annotations/annotations.json  ← Label Studio JSON export

 HOW TO EXPORT FROM LABEL STUDIO:
   Project → Export → JSON → download → rename to annotations.json
   Place at: data/gold_standard/annotations/annotations.json

 OUTPUTS:
   reports/gold_standard_iou_report.csv      ← per-image IoU for all 3 artifacts
   reports/gold_standard_iou_summary.csv     ← mean ± std per class + overall
   reports/gold_standard_overlays/           ← visual overlay PNGs (5 per class)

 RUN ORDER:
   After: generate_tier1_masks.py  (SAM2 masks must exist)
   After: train_teacher.py         (teacher_model_best.pth must exist)
   After: train_student.py         (student best checkpoint must exist)
   Before: export_tflite.py

 USAGE:
   python validate_gold_standard.py              # validate all three artifacts
   python validate_gold_standard.py --sam2-only  # SAM2 only (fastest, run first)
================================================================================
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from image_utils import load_image_rgb
from config import (
    SEED, CLASSES, REPORTS_DIR,
    TIER1_MASKS_DIR,
    TEACHER_CKPT_DIR, TEACHER_IMG_SIZE,
    TEACHER_DEPLOYED_VARIANT,
    STUDENT_CKPT_DIR, STUDENT_IMG_SIZE,
    STUDENT_BEST_VARIANT, STUDENT_FACTORY_MODE,
    STUDENT_DROPOUT, CBAM_SPATIAL_KERNEL,
    GOLD_IMAGES_DIR, GOLD_ANNOTATION_FILE,
    GOLD_IOU_WARN_THRESHOLD, GOLD_IOU_TARGET_MEAN,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OVERLAY_ALPHA    = 0.40
OVERLAYS_PER_CLS = 5   # qualitative overlay figures per class


# ══════════════════════════════════════════════════════════════════════════════
# LABEL STUDIO ANNOTATION PARSER
# ══════════════════════════════════════════════════════════════════════════════

def load_annotations(annotation_file: Path) -> dict[str, np.ndarray]:
    """
    Parse Label Studio JSON export into a dict of {stem: binary_mask}.

    Accepts both:
      - Single project export (list of task dicts at top level)
      - Per-image export (single task dict)

    Each task must have a polygonlabels result type.
    Polygon points are stored as percentage of image dimensions — we
    defer rasterization to get_human_mask() once image size is known.

    Returns dict mapping image stem → list of normalized polygon point lists.
    """
    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}\n"
            "Export from Label Studio: Project → Export → JSON\n"
            f"Place at: {annotation_file}"
        )

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalise to list of tasks
    if isinstance(data, dict):
        data = [data]

    annotations = {}
    n_parsed = 0

    for task in data:
        # Label Studio stores the filename in task["data"]["image"] or
        # task["file_upload"] — handle both formats.
        fname = (
            task.get("data", {}).get("image", "")
            or task.get("file_upload", "")
        )
        # Strip URL prefix if present (Label Studio sometimes includes it)
        fname = Path(fname.split("/")[-1]).stem   # → bare stem

        polygons = []
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") != "polygonlabels":
                    continue
                pts = result["value"].get("points", [])   # [[x%, y%], ...]
                if len(pts) >= 3:
                    polygons.append(pts)

        if polygons:
            annotations[fname] = polygons
            n_parsed += 1

    print(f"  Parsed {n_parsed} annotated images from {annotation_file.name}")
    return annotations


def rasterize_polygons(polygons: list[list], h: int, w: int) -> np.ndarray:
    """
    Convert Label Studio polygon annotations (% coords) to a binary mask.
    Multiple polygons per image are merged (logical OR).

    Args:
        polygons : list of [[x%, y%], ...] polygon point lists
        h, w     : target mask height and width in pixels

    Returns:
        uint8 binary mask of shape (h, w), values 0 or 1.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(
            [[int(p[0] / 100.0 * w), int(p[1] / 100.0 * h)] for p in poly],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 1)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# IoU COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_iou(pred_binary: np.ndarray,
                human_binary: np.ndarray) -> float:
    """
    Binary IoU between predicted mask and human-annotated mask.

    Both inputs are uint8/bool arrays of shape (H, W).
    Returns float in [0, 1]. Returns 0.0 if both masks are empty
    (avoids division by zero — treated as a degenerate case).
    """
    pred_b  = pred_binary.astype(bool)
    human_b = human_binary.astype(bool)

    intersection = np.logical_and(pred_b, human_b).sum()
    union        = np.logical_or(pred_b,  human_b).sum()

    if union == 0:
        # Both masks are entirely empty — degenerate, flag as 0
        return 0.0
    return float(intersection) / float(union)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_teacher(variant: str) -> nn.Module:
    """Load best Teacher checkpoint for inference."""
    ckpt_path = TEACHER_CKPT_DIR / "teacher_model_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {ckpt_path}\n"
            "Run train_teacher.py first."
        )

    unet_encoders = {
        "resnet50":        "resnet50",
        "efficientnet-b2": "efficientnet-b2",
        "mit_b2":          "mit_b2",
    }

    if variant in unet_encoders:
        model = smp.Unet(
            encoder_name=unet_encoders[variant],
            encoder_weights=None,   # weights loaded from checkpoint
            in_channels=3, classes=1, activation=None,
        )
    elif variant == "deeplabv3plus-eb2":
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b2",
            encoder_weights=None,
            in_channels=3, classes=1, activation=None,
        )
    else:
        raise ValueError(f"Unknown teacher variant: {variant}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    print(f"  Teacher loaded: {variant}  ({ckpt_path.name})")
    return model


def load_student(encoder_variant: str) -> nn.Module:
    """
    Load best Student checkpoint for inference.
    Imports StudentModel inline to avoid circular dependencies.
    """
    # Import here — train_student.py defines StudentModel
    try:
        from train_student import StudentModel
    except ImportError:
        raise ImportError(
            "Could not import StudentModel from train_student.py.\n"
            "Ensure train_student.py is in the same directory."
        )

    use_cbam = "cbam" in encoder_variant
    model    = StudentModel(encoder_name=encoder_variant, use_cbam=use_cbam)

    # Find best student checkpoint
    # Prefer: student_{variant}_{mode}_best.pth (from Stage 2)
    # Fallback: student_{variant}_mode_b_best.pth (from Stage 1)
    mode     = STUDENT_FACTORY_MODE
    ckpt_path = STUDENT_CKPT_DIR / f"student_{encoder_variant}_{mode}_best.pth"
    if not ckpt_path.exists():
        # Try Stage 1 checkpoint
        ckpt_path = STUDENT_CKPT_DIR / f"student_{encoder_variant}_mode_b_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Student checkpoint not found for {encoder_variant}.\n"
            "Run train_student.py --stage 1 (and optionally --stage 2) first."
        )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    print(f"  Student loaded: {encoder_variant}  ({ckpt_path.name})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def teacher_predict(model: nn.Module,
                    img_rgb: np.ndarray,
                    img_size: int) -> np.ndarray:
    """
    Run Teacher inference. Returns binary mask at original image resolution.
    """
    tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    h_orig, w_orig = img_rgb.shape[:2]
    img_t = tf(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_t)                          # 1×1×H×W
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Resize back to original resolution
    prob_full = cv2.resize(prob, (w_orig, h_orig),
                           interpolation=cv2.INTER_LINEAR)
    return (prob_full >= 0.5).astype(np.uint8)


def student_predict(model: nn.Module,
                    img_rgb: np.ndarray,
                    img_size: int) -> np.ndarray:
    """
    Run Student inference. Returns binary silhouette mask (channel 0)
    at original image resolution.
    """
    tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    h_orig, w_orig = img_rgb.shape[:2]
    img_t = tf(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg_logits, _, _ = model(img_t)                # 1×2×H×W
        sil_prob = torch.sigmoid(
            seg_logits[:, 0]).squeeze().cpu().numpy()  # channel 0 = silhouette

    sil_full = cv2.resize(sil_prob, (w_orig, h_orig),
                          interpolation=cv2.INTER_LINEAR)
    return (sil_full >= 0.5).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAY GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def draw_comparison_overlay(img_rgb: np.ndarray,
                            human_mask: np.ndarray,
                            pred_mask: np.ndarray,
                            iou: float,
                            label: str) -> np.ndarray:
    """
    Draw a side-by-side comparison overlay:
      Left half  — human mask (green)
      Right half — predicted mask (blue)
      Overlap    — cyan
    IoU and label printed at the top.
    """
    overlay = img_rgb.copy().astype(np.float32)

    # True positive  → cyan  (both agree)
    tp = np.logical_and(human_mask, pred_mask)
    # False negative → green (human has it, pred misses)
    fn = np.logical_and(human_mask, ~pred_mask.astype(bool))
    # False positive → red   (pred has it, human doesn't)
    fp = np.logical_and(pred_mask, ~human_mask.astype(bool))

    green  = np.array([0,   220, 0  ], dtype=np.float32)
    red    = np.array([220, 0,   0  ], dtype=np.float32)
    cyan   = np.array([0,   220, 220], dtype=np.float32)

    overlay[tp] = overlay[tp] * (1 - OVERLAY_ALPHA) + cyan  * OVERLAY_ALPHA
    overlay[fn] = overlay[fn] * (1 - OVERLAY_ALPHA) + green * OVERLAY_ALPHA
    overlay[fp] = overlay[fp] * (1 - OVERLAY_ALPHA) + red   * OVERLAY_ALPHA

    overlay = overlay.clip(0, 255).astype(np.uint8)

    # Draw human mask contour (green) and pred contour (blue)
    for mask, color in [(human_mask, (0, 200, 0)), (pred_mask, (0, 0, 220))]:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    # Label bar at top
    text = f"{label}  |  IoU={iou:.3f}"
    cv2.putText(overlay, text, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, text, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),       1)

    # Legend bottom-left
    legend = "Green=missed | Cyan=correct | Red=extra"
    cv2.putText(overlay, legend, (8, overlay.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(overlay, legend, (8, overlay.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),       1)

    return overlay


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(rows: list[dict],
                    artifact: str) -> dict:
    """
    Compute mean IoU ± std per class and overall for one artifact.
    """
    iou_key = f"{artifact}_iou"
    summary = {"artifact": artifact}

    all_ious = []
    for cls in CLASSES:
        cls_ious = [
            r[iou_key] for r in rows
            if r["category"] == cls and r[iou_key] >= 0
        ]
        summary[f"{cls}_mean_iou"]  = round(float(np.mean(cls_ious)),  4) if cls_ious else -1
        summary[f"{cls}_std_iou"]   = round(float(np.std(cls_ious)),   4) if cls_ious else -1
        summary[f"{cls}_n"]         = len(cls_ious)
        all_ious.extend(cls_ious)

    summary["overall_mean_iou"] = round(float(np.mean(all_ious)),  4) if all_ious else -1
    summary["overall_std_iou"]  = round(float(np.std(all_ious)),   4) if all_ious else -1
    summary["n_total"]          = len(all_ious)
    summary["n_below_warn"]     = sum(
        1 for v in all_ious if v < GOLD_IOU_WARN_THRESHOLD)
    summary["target_met"]       = (
        summary["overall_mean_iou"] >= GOLD_IOU_TARGET_MEAN
    )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.time()
    random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-only", action="store_true",
                        help="Validate SAM2 masks only (no model inference). "
                             "Run this first — Teacher/Student checkpoints not needed.")
    args = parser.parse_args()

    print("=" * 72)
    print("  Yellow MAIze | Gold Standard IoU Validation")
    print("=" * 72)

    # ── 1. Validate input folders ─────────────────────────────────────────────
    if not GOLD_IMAGES_DIR.exists() or not any(GOLD_IMAGES_DIR.iterdir()):
        print(f"\n[FATAL] Gold standard images not found: {GOLD_IMAGES_DIR}")
        print("  Run sample_gold_standard.py first to populate this folder.")
        return

    # ── 2. Load Label Studio annotations ─────────────────────────────────────
    print(f"\n  Loading annotations from: {GOLD_ANNOTATION_FILE}")
    try:
        annotations = load_annotations(GOLD_ANNOTATION_FILE)
    except FileNotFoundError as e:
        print(f"\n[FATAL] {e}")
        return

    # ── 3. Collect gold standard image list ───────────────────────────────────
    valid_exts  = {".jpg", ".jpeg", ".png"}
    gold_images = sorted([
        p for p in GOLD_IMAGES_DIR.iterdir()
        if p.suffix.lower() in valid_exts
    ])
    print(f"  Gold standard images found : {len(gold_images):,}")

    # ── 4. Load models (skip if --sam2-only) ─────────────────────────────────
    teacher_model = None
    student_model = None

    if not args.sam2_only:
        print(f"\n  Loading Teacher ({TEACHER_DEPLOYED_VARIANT}) ...")
        try:
            teacher_model = load_teacher(TEACHER_DEPLOYED_VARIANT)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
            print("  Teacher IoU will be skipped.")

        print(f"\n  Loading Student ({STUDENT_BEST_VARIANT}) ...")
        try:
            student_model = load_student(STUDENT_BEST_VARIANT)
        except (FileNotFoundError, ImportError) as e:
            print(f"  [WARN] {e}")
            print("  Student IoU will be skipped.")

    # ── 5. Output directories ─────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    overlay_dir = REPORTS_DIR / "gold_standard_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Track overlay counts per class per artifact
    overlay_counts = {
        cls: {"sam2": 0, "teacher": 0, "student": 0}
        for cls in CLASSES
    }

    # ── 6. Main per-image loop ────────────────────────────────────────────────
    rows = []
    n_total = n_no_annotation = n_no_sam2 = 0

    print(f"\n  Validating {len(gold_images):,} images ...")
    print(f"  {'─'*68}")

    for i, img_path in enumerate(gold_images):
        stem     = img_path.stem          # e.g. "MSV_image045"
        # Category is encoded as prefix by sample_gold_standard.py
        category = stem.split("_")[0]
        if category not in CLASSES:
            continue

        n_total += 1

        # ── Load image ────────────────────────────────────────────────────────
        img_rgb = load_image_rgb(img_path)
        if img_rgb is None:
            continue
        h, w = img_rgb.shape[:2]

        # ── Human mask ────────────────────────────────────────────────────────
        polygons = annotations.get(stem)
        if not polygons:
            n_no_annotation += 1
            continue
        human_mask = rasterize_polygons(polygons, h, w)

        row = {
            "stem":        stem,
            "category":    category,
            "img_path":    str(img_path),
            "sam2_iou":    -1.0,
            "teacher_iou": -1.0,
            "student_iou": -1.0,
            "sam2_warn":   False,
            "teacher_warn":False,
            "student_warn":False,
        }

        # ── SAM2 IoU ──────────────────────────────────────────────────────────
        npy_path = TIER1_MASKS_DIR / f"{stem}_softmask.npy"
        if npy_path.exists():
            prob_map   = np.load(str(npy_path)).astype(np.float32)
            prob_full  = cv2.resize(prob_map, (w, h),
                                    interpolation=cv2.INTER_LINEAR)
            sam2_mask  = (prob_full >= 0.5).astype(np.uint8)
            sam2_iou   = compute_iou(sam2_mask, human_mask)
            row["sam2_iou"]  = round(sam2_iou,  4)
            row["sam2_warn"] = sam2_iou < GOLD_IOU_WARN_THRESHOLD

            # Save overlay (up to OVERLAYS_PER_CLS per class)
            if overlay_counts[category]["sam2"] < OVERLAYS_PER_CLS:
                ov = draw_comparison_overlay(
                    img_rgb, human_mask, sam2_mask, sam2_iou,
                    f"SAM2 | {category} | {stem}")
                cv2.imwrite(
                    str(overlay_dir / f"{stem}_sam2_overlay.jpg"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                overlay_counts[category]["sam2"] += 1
        else:
            n_no_sam2 += 1

        # ── Teacher IoU ───────────────────────────────────────────────────────
        if teacher_model is not None:
            teacher_mask = teacher_predict(teacher_model, img_rgb,
                                           TEACHER_IMG_SIZE)
            teacher_iou  = compute_iou(teacher_mask, human_mask)
            row["teacher_iou"]  = round(teacher_iou, 4)
            row["teacher_warn"] = teacher_iou < GOLD_IOU_WARN_THRESHOLD

            if overlay_counts[category]["teacher"] < OVERLAYS_PER_CLS:
                ov = draw_comparison_overlay(
                    img_rgb, human_mask, teacher_mask, teacher_iou,
                    f"Teacher | {category} | {stem}")
                cv2.imwrite(
                    str(overlay_dir / f"{stem}_teacher_overlay.jpg"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                overlay_counts[category]["teacher"] += 1

        # ── Student IoU ───────────────────────────────────────────────────────
        if student_model is not None:
            student_mask = student_predict(student_model, img_rgb,
                                           STUDENT_IMG_SIZE)
            student_iou  = compute_iou(student_mask, human_mask)
            row["student_iou"]  = round(student_iou, 4)
            row["student_warn"] = student_iou < GOLD_IOU_WARN_THRESHOLD

            if overlay_counts[category]["student"] < OVERLAYS_PER_CLS:
                ov = draw_comparison_overlay(
                    img_rgb, human_mask, student_mask, student_iou,
                    f"Student | {category} | {stem}")
                cv2.imwrite(
                    str(overlay_dir / f"{stem}_student_overlay.jpg"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                overlay_counts[category]["student"] += 1

        rows.append(row)

        if (i + 1) % 50 == 0 or (i + 1) == len(gold_images):
            print(f"  [{i+1:>4}/{len(gold_images)}]  "
                  f"annotated {len(rows):,}  |  "
                  f"no_annotation {n_no_annotation:,}  |  "
                  f"no_sam2_mask {n_no_sam2:,}")

    # ── 7. Write per-image report ─────────────────────────────────────────────
    report_path = REPORTS_DIR / "gold_standard_iou_report.csv"
    if rows:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Per-image report → {report_path}")

    # ── 8. Compute and write summary ──────────────────────────────────────────
    summary_rows = []
    artifacts    = ["sam2"]
    if teacher_model is not None:
        artifacts.append("teacher")
    if student_model is not None:
        artifacts.append("student")

    print(f"\n{'─' * 72}")
    print(f"  {'Artifact':<12}  {'Class':<10}  {'Mean IoU':>8}  "
          f"{'Std':>6}  {'N':>5}  {'<Warn':>6}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*5}  {'─'*6}")

    for artifact in artifacts:
        s = compute_summary(rows, artifact)
        summary_rows.append(s)

        for cls in CLASSES:
            mean = s[f"{cls}_mean_iou"]
            std  = s[f"{cls}_std_iou"]
            n    = s[f"{cls}_n"]
            warn = sum(
                1 for r in rows
                if r["category"] == cls and r[f"{artifact}_iou"] >= 0
                and r[f"{artifact}_iou"] < GOLD_IOU_WARN_THRESHOLD
            )
            print(f"  {artifact:<12}  {cls:<10}  {mean:>8.4f}  "
                  f"{std:>6.4f}  {n:>5}  {warn:>6}")

        print(f"  {artifact:<12}  {'OVERALL':<10}  "
              f"{s['overall_mean_iou']:>8.4f}  "
              f"{s['overall_std_iou']:>6.4f}  "
              f"{s['n_total']:>5}  "
              f"{s['n_below_warn']:>6}")
        target_str = "✓ TARGET MET" if s["target_met"] else "✗ BELOW TARGET"
        print(f"  {'':12}  {'':10}  {target_str}  "
              f"(target: mean IoU ≥ {GOLD_IOU_TARGET_MEAN})")
        print()

    summary_path = REPORTS_DIR / "gold_standard_iou_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Summary report → {summary_path}")

    # ── 9. IoU chain comparison (thesis-ready) ────────────────────────────────
    if len(artifacts) > 1 and summary_rows:
        print(f"\n  IoU chain (same 300 images across all artifacts):")
        print(f"  {'Artifact':<16}  {'Overall mean IoU':>16}  {'Target met':>10}")
        print(f"  {'─'*16}  {'─'*16}  {'─'*10}")
        for s in summary_rows:
            met = "Yes" if s["target_met"] else "No"
            print(f"  {s['artifact']:<16}  {s['overall_mean_iou']:>16.4f}  {met:>10}")

    # ── 10. Warnings and diagnostics ─────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  Diagnostics:")
    print(f"    Total images attempted : {n_total:,}")
    print(f"    Missing annotations    : {n_no_annotation:,}")
    print(f"    Missing SAM2 masks     : {n_no_sam2:,}")
    print(f"    Successfully validated : {len(rows):,}")
    print(f"    Overlays saved to      : {overlay_dir}")

    if n_no_annotation > 0:
        print(f"\n  [WARN] {n_no_annotation} images have no annotation.")
        print("         Check that image filenames in Label Studio match")
        print(f"         the files in {GOLD_IMAGES_DIR.name}/")

    if n_no_sam2 > 0:
        print(f"\n  [WARN] {n_no_sam2} images have no SAM2 mask.")
        print("         These images may have been rejected by QA filters.")
        print("         SAM2 IoU is only computed for images that passed QA.")

    duration = round(time.time() - t_start, 1)
    print(f"\n  Done in {duration}s")
    print(f"\n  THESIS NOTE:")
    print(f"    Report in Chapter 3: 'SAM2-generated pseudo-masks achieved")
    print(f"    a mean IoU of [sam2_overall] ± [std] against human-verified")
    print(f"    annotations (n={len(rows)}), validating their use as pseudo-labels")
    print(f"    for Teacher model training.'")
    print(f"\n  NEXT STEP: python export_tflite.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
