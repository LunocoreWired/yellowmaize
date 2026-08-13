"""
validate_gold_standard.py — Gold Standard IoU Validation
PURPOSE:
Validate SAM2 pseudo-masks, Teacher predictions, and Student predictions
against human-annotated leaf silhouette masks (the gold standard).
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import (
    CLASSES,
    GOLD_ANNOTATION_FILE,
    GOLD_IMAGES_DIR,
    GOLD_IOU_TARGET_MEAN,
    GOLD_IOU_WARN_THRESHOLD,
    REPORTS_DIR,
    SEED,
    STUDENT_BEST_VARIANT,
    STUDENT_CKPT_DIR,
    STUDENT_FACTORY_MODE,
    STUDENT_IMG_SIZE,
    TEACHER_CKPT_DIR,
    TEACHER_DEPLOYED_VARIANT,
    TEACHER_IMG_SIZE,
    TIER1_MASKS_DIR,
)
from image_utils import load_image_rgb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OVERLAY_ALPHA = 0.40
OVERLAYS_PER_CLS = 5  # qualitative overlay figures per class

# ══════════════════════════════════════════════════════════════════════════════
# LABEL STUDIO ANNOTATION PARSER
# ══════════════════════════════════════════════════════════════════════════════
def load_annotations(annotation_file: Path) -> dict[str, list]:
    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}\n"
            "Export from Label Studio: Project → Export → JSON\n"
            f"Place at: {annotation_file}"
        )

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    annotations = {}
    n_parsed = 0

    for task in data:
        fname = task.get("data", {}).get("image", "") or task.get("file_upload", "")
        fname = Path(fname.split("/")[-1]).stem
        fname = fname.split("-", 1)[-1]  # strip Label Studio hash prefix

        polygons = []
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") != "polygonlabels":
                    continue
                pts = result["value"].get("points", [])
                if len(pts) >= 3:
                    polygons.append(pts)

        if polygons:
            annotations[fname] = polygons
            n_parsed += 1

    print(f"  Parsed {n_parsed} annotated images from {annotation_file.name}")
    return annotations

def rasterize_polygons(polygons: list[list], h: int, w: int) -> np.ndarray:
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
def compute_iou(pred_binary: np.ndarray, human_binary: np.ndarray) -> float:
    pred_b = pred_binary.astype(bool)
    human_b = human_binary.astype(bool)
    intersection = np.logical_and(pred_b, human_b).sum()
    union = np.logical_or(pred_b, human_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_teacher(variant: str = None) -> nn.Module:
    import segmentation_models_pytorch as smp
    
    ckpt_path = TEACHER_CKPT_DIR / "teacher_model_best.pth"
    if not ckpt_path.exists():
        fallback_variant = variant or TEACHER_DEPLOYED_VARIANT
        ckpt_path = TEACHER_CKPT_DIR / f"teacher_{fallback_variant}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    actual_variant = ckpt.get("variant", variant or TEACHER_DEPLOYED_VARIANT)
    print(f"  Detected Teacher variant in checkpoint: {actual_variant}")

    unet_encoders = {"resnet50": "resnet50", "efficientnet-b2": "efficientnet-b2", "mit_b2": "mit_b2"}

    if actual_variant in unet_encoders:
        model = smp.Unet(encoder_name=unet_encoders[actual_variant], encoder_weights=None, in_channels=3, classes=1, activation=None)
    elif actual_variant == "deeplabv3plus-eb2":
        model = smp.DeepLabV3Plus(encoder_name="efficientnet-b2", encoder_weights=None, in_channels=3, classes=1, activation=None)
    else:
        raise ValueError(f"Unknown teacher variant: {actual_variant}")

    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    print(f"  Teacher loaded: {actual_variant}  ({ckpt_path.name})")
    return model

def load_student(encoder_variant: str) -> nn.Module:
    try:
        from train_student import StudentModel
    except ImportError:
        raise ImportError("Could not import StudentModel from train_student.py.")
    
    use_cbam = "cbam" in encoder_variant
    model = StudentModel(encoder_name=encoder_variant, use_cbam=use_cbam)

    mode = STUDENT_FACTORY_MODE
    ckpt_path = STUDENT_CKPT_DIR / f"student_{encoder_variant}_{mode}_best.pth"
    if not ckpt_path.exists():
        ckpt_path = STUDENT_CKPT_DIR / f"student_{encoder_variant}_mode_b_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found for {encoder_variant}.")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    print(f"  Student loaded: {encoder_variant}  ({ckpt_path.name})")
    return model

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS (GEOMETRICALLY PERFECT LETTERBOX REVERSAL)
# ══════════════════════════════════════════════════════════════════════════════
def _predict_letterbox(model: nn.Module, img_rgb: np.ndarray, img_size: int, is_student: bool = False) -> np.ndarray:
    """
    Reverses LongestMaxSize + PadIfNeeded perfectly.
    Prevents the 'squished padding' bug that clips leaf margins.
    """
    h_orig, w_orig = img_rgb.shape[:2]
    
    # 1. Calculate letterbox scale
    scale = img_size / max(h_orig, w_orig)
    new_h, new_w = int(h_orig * scale), int(w_orig * scale)
    
    # 2. Resize keeping aspect ratio
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 3. Center pad to square (matches Albumentations PadIfNeeded default)
    pad_top = (img_size - new_h) // 2
    pad_bottom = img_size - new_h - pad_top
    pad_left = (img_size - new_w) // 2
    pad_right = img_size - new_w - pad_left
    
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    # 4. Normalize (ImageNet) 
    # FIX: explicitly cast mean/std to np.float32 to prevent float64 (double) tensor promotion
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_norm = (img_padded.astype(np.float32) / 255.0 - mean) / std
    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # 5. Inference
    with torch.no_grad():
        if is_student:
            seg_logits, _, _ = model(img_t)
            prob = torch.sigmoid(seg_logits[:, 0]).squeeze().cpu().numpy()
        else:
            logits = model(img_t)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            
    # 6. Reverse Letterbox (Crop padding, then resize)
    y_start, y_end = pad_top, pad_top + new_h
    x_start, x_end = pad_left, pad_left + new_w
    
    prob_cropped = prob[y_start:y_end, x_start:x_end]
    prob_full = cv2.resize(prob_cropped, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    return (prob_full >= 0.5).astype(np.uint8)

def teacher_predict(model: nn.Module, img_rgb: np.ndarray, img_size: int) -> np.ndarray:
    return _predict_letterbox(model, img_rgb, img_size, is_student=False)

def student_predict(model: nn.Module, img_rgb: np.ndarray, img_size: int) -> np.ndarray:
    return _predict_letterbox(model, img_rgb, img_size, is_student=True)

# ══════════════════════════════════════════════════════════════════════════════
# OVERLAY GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def draw_comparison_overlay(img_rgb: np.ndarray, human_mask: np.ndarray, pred_mask: np.ndarray, iou: float, label: str) -> np.ndarray:
    overlay = img_rgb.copy().astype(np.float32)
    tp = np.logical_and(human_mask, pred_mask)
    fn = np.logical_and(human_mask, ~pred_mask.astype(bool))
    fp = np.logical_and(pred_mask, ~human_mask.astype(bool))

    green = np.array([0, 220, 0], dtype=np.float32)
    red = np.array([220, 0, 0], dtype=np.float32)
    cyan = np.array([0, 220, 220], dtype=np.float32)

    overlay[tp] = overlay[tp] * (1 - OVERLAY_ALPHA) + cyan * OVERLAY_ALPHA
    overlay[fn] = overlay[fn] * (1 - OVERLAY_ALPHA) + green * OVERLAY_ALPHA
    overlay[fp] = overlay[fp] * (1 - OVERLAY_ALPHA) + red * OVERLAY_ALPHA
    overlay = overlay.clip(0, 255).astype(np.uint8)

    for mask, color in [(human_mask, (0, 200, 0)), (pred_mask, (0, 0, 220))]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    text = f"{label}  |  IoU={iou:.3f}"
    cv2.putText(overlay, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    legend = "Green=missed | Cyan=correct | Red=extra"
    cv2.putText(overlay, legend, (8, overlay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(overlay, legend, (8, overlay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return overlay

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_summary(rows: list[dict], artifact: str) -> dict:
    iou_key = f"{artifact}_iou"
    summary = {"artifact": artifact}
    all_ious = []
    for cls in CLASSES:
        cls_ious = [r[iou_key] for r in rows if r["category"] == cls and r[iou_key] >= 0]
        summary[f"{cls}_mean_iou"] = round(float(np.mean(cls_ious)), 4) if cls_ious else -1
        summary[f"{cls}_std_iou"] = round(float(np.std(cls_ious)), 4) if cls_ious else -1
        summary[f"{cls}_n"] = len(cls_ious)
        all_ious.extend(cls_ious)

    summary["overall_mean_iou"] = round(float(np.mean(all_ious)), 4) if all_ious else -1
    summary["overall_std_iou"] = round(float(np.std(all_ious)), 4) if all_ious else -1
    summary["n_total"] = len(all_ious)
    summary["n_below_warn"] = sum(1 for v in all_ious if v < GOLD_IOU_WARN_THRESHOLD)
    summary["target_met"] = summary["overall_mean_iou"] >= GOLD_IOU_TARGET_MEAN
    return summary

# ══════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    t_start = time.time()
    random.seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-only", action="store_true", help="Validate SAM2 masks only.")
    args = parser.parse_args()

    print("=" * 72)
    print("  Yellow MAIze | Gold Standard IoU Validation")
    print("=" * 72)

    if not GOLD_IMAGES_DIR.exists() or not any(GOLD_IMAGES_DIR.iterdir()):
        print(f"\n[FATAL] Gold standard images not found: {GOLD_IMAGES_DIR}")
        return

    print(f"\n  Loading annotations from: {GOLD_ANNOTATION_FILE}")
    try:
        annotations = load_annotations(GOLD_ANNOTATION_FILE)
    except FileNotFoundError as e:
        print(f"\n[FATAL] {e}")
        return

    valid_exts = {".jpg", ".jpeg", ".png"}
    gold_images = sorted([p for p in GOLD_IMAGES_DIR.iterdir() if p.suffix.lower() in valid_exts])
    print(f"  Gold standard images found : {len(gold_images):,}")

    teacher_model = None
    student_model = None

    if not args.sam2_only:
        print(f"\n  Loading Teacher ... ")
        try:
            teacher_model = load_teacher(TEACHER_DEPLOYED_VARIANT)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")

        print(f"\n  Loading Student ({STUDENT_BEST_VARIANT}) ... ")
        try:
            student_model = load_student(STUDENT_BEST_VARIANT)
        except (FileNotFoundError, ImportError) as e:
            print(f"  [WARN] {e}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    overlay_dir = REPORTS_DIR / "gold_standard_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_counts = {cls: {"sam2": 0, "teacher": 0, "student": 0} for cls in CLASSES}

    rows = []
    n_total = n_no_annotation = n_no_sam2 = 0

    print(f"\n  Validating {len(gold_images):,} images ... ")
    print(f"  {'─' * 68}")

    for i, img_path in enumerate(gold_images):
        stem = img_path.stem
        category = stem.split("_")[0]
        if category not in CLASSES:
            continue

        n_total += 1
        img_rgb = load_image_rgb(img_path)
        if img_rgb is None:
            continue
        h, w = img_rgb.shape[:2]

        polygons = annotations.get(stem)
        if not polygons:
            n_no_annotation += 1
            continue
        human_mask = rasterize_polygons(polygons, h, w)

        row = {
            "stem": stem, "category": category, "img_path": str(img_path),
            "sam2_iou": -1.0, "teacher_iou": -1.0, "student_iou": -1.0,
            "sam2_warn": False, "teacher_warn": False, "student_warn": False,
        }

        npy_path = TIER1_MASKS_DIR / f"{stem}_softmask.npy"
        if npy_path.exists():
            prob_map = np.load(str(npy_path)).astype(np.float32)
            prob_full = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
            sam2_mask = (prob_full >= 0.5).astype(np.uint8)
            sam2_iou = compute_iou(sam2_mask, human_mask)
            row["sam2_iou"] = round(sam2_iou, 4)
            row["sam2_warn"] = sam2_iou < GOLD_IOU_WARN_THRESHOLD

            if overlay_counts[category]["sam2"] < OVERLAYS_PER_CLS:
                ov = draw_comparison_overlay(img_rgb, human_mask, sam2_mask, sam2_iou, f"SAM2 | {category} | {stem}")
                cv2.imwrite(str(overlay_dir / f"{stem}_sam2_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                overlay_counts[category]["sam2"] += 1
        else:
            n_no_sam2 += 1

        if teacher_model is not None:
            teacher_mask = teacher_predict(teacher_model, img_rgb, TEACHER_IMG_SIZE)
            teacher_iou = compute_iou(teacher_mask, human_mask)
            row["teacher_iou"] = round(teacher_iou, 4)
            row["teacher_warn"] = teacher_iou < GOLD_IOU_WARN_THRESHOLD

            if overlay_counts[category]["teacher"] < OVERLAYS_PER_CLS:
                ov = draw_comparison_overlay(img_rgb, human_mask, teacher_mask, teacher_iou, f"Teacher | {category} | {stem}")
                cv2.imwrite(str(overlay_dir / f"{stem}_teacher_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                overlay_counts[category]["teacher"] += 1

        if student_model is not None:
            student_mask = student_predict(student_model, img_rgb, STUDENT_IMG_SIZE)
            student_iou = compute_iou(student_mask, human_mask)
            row["student_iou"] = round(student_iou, 4)
            row["student_warn"] = student_iou < GOLD_IOU_WARN_THRESHOLD

            if overlay_counts[category]["student"] < OVERLAYS_PER_CLS:
                ov = draw_comparison_overlay(img_rgb, human_mask, student_mask, student_iou, f"Student | {category} | {stem}")
                cv2.imwrite(str(overlay_dir / f"{stem}_student_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                overlay_counts[category]["student"] += 1

        rows.append(row)
        if (i + 1) % 50 == 0 or (i + 1) == len(gold_images):
            print(f"  [{i + 1: >4}/{len(gold_images)}]   annotated {len(rows):,}  |   no_annotation {n_no_annotation:,}  |   no_sam2_mask {n_no_sam2:,}")

    report_path = REPORTS_DIR / "gold_standard_iou_report.csv"
    if rows:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Per-image report → {report_path}")

    summary_rows = []
    artifacts = ["sam2"]
    if teacher_model is not None: artifacts.append("teacher")
    if student_model is not None: artifacts.append("student")

    print(f"\n{'─' * 72}")
    print(f"  {'Artifact': <12}  {'Class': <10}  {'Mean IoU': >8}   {'Std': >6}  {'N': >5}  {' <Warn': >6}")
    print(f"  {'─' * 12}  {'─' * 10}  {'─' * 8}  {'─' * 6}  {'─' * 5}  {'─' * 6}")

    for artifact in artifacts:
        s = compute_summary(rows, artifact)
        summary_rows.append(s)
        for cls in CLASSES:
            mean = s[f"{cls}_mean_iou"]
            std = s[f"{cls}_std_iou"]
            n = s[f"{cls}_n"]
            warn = sum(1 for r in rows if r["category"] == cls and r[f"{artifact}_iou"] >= 0 and r[f"{artifact}_iou"] < GOLD_IOU_WARN_THRESHOLD)
            print(f"  {artifact: <12}  {cls: <10}  {mean: >8.4f}   {std: >6.4f}  {n: >5}  {warn: >6}")

        print(f"  {artifact: <12}  {'OVERALL': <10}   {s['overall_mean_iou']: >8.4f}   {s['overall_std_iou']: >6.4f}   {s['n_total']: >5}   {s['n_below_warn']: >6}")
        target_str = "✓ TARGET MET" if s["target_met"] else "✗ BELOW TARGET"
        print(f"  {'':12}  {'':10}  {target_str}   (target: mean IoU ≥ {GOLD_IOU_TARGET_MEAN})\n")

    summary_path = REPORTS_DIR / "gold_standard_iou_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Summary report → {summary_path}")

    if len(artifacts) > 1 and summary_rows:
        print(f"\n  IoU chain (same {len(rows)} images across all artifacts):")
        print(f"  {'Artifact': <16}  {'Overall mean IoU': >16}  {'Target met': >10}")
        print(f"  {'─' * 16}  {'─' * 16}  {'─' * 10}")
        for s in summary_rows:
            met = "Yes" if s["target_met"] else "No"
            print(f"  {s['artifact']: <16}  {s['overall_mean_iou']: >16.4f}  {met: >10}")

    print(f"\n{'─' * 72}")
    print(f"  Diagnostics:")
    print(f"    Total images attempted : {n_total:,}")
    print(f"    Missing annotations    : {n_no_annotation:,}")
    print(f"    Missing SAM2 masks     : {n_no_sam2:,}")
    print(f"    Successfully validated : {len(rows):,}")
    print(f"    Overlays saved to      : {overlay_dir}")
    
    duration = round(time.time() - t_start, 1)
    print(f"\n  Done in {duration}s")
    print("=" * 72)

if __name__ == "__main__":
    main()