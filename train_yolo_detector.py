"""
================================================================================
 train_yolo_detector.py — Phase 1b: YOLOv8n Leaf Detector Training
================================================================================
 PURPOSE:
   Train a YOLOv8n (nano) single-class leaf detector on Label Studio polygon
   annotations. The trained detector is used by generate_tier1_masks.py (v3)
   to supply tight bounding-box prompts for SAM2, replacing the pure
   HSV-centroid approach and eliminating the most common false segmentation
   failure mode (background clutter scored over the leaf).

 WORKFLOW:
   1. Parse Label Studio JSON export — polygon annotations
   2. Convert polygons → axis-aligned bounding boxes (YOLO xyxy → cx cy w h)
   3. Build YOLO dataset layout + dataset.yaml, train/val split
   4. Train YOLOv8n (single class: "leaf") via Ultralytics
   5. Evaluate on val split — print mAP@0.5 / precision / recall
   6. [OPTIONAL] Calibrate SAM2 QA confidence threshold against gold standard:
        run YOLO-guided SAM2 on the 300 annotated images, compute per-image
        IoU(sam2_mask, human_polygon_mask), and find the SAM2 mean_conf
        threshold T such that images with mean_conf ≥ T achieve
        GOLD_IOU_TARGET_MEAN on average. Writes YOLO_QA_CALIB_FILE.

 ANNOTATION REQUIREMENTS:
   Minimum YOLO_MIN_ANNOTATIONS (400) polygon labels in Label Studio.
   You have 300 from sample_gold_standard.py — annotate ~100–200 more.

 LABEL STUDIO EXPORT:
   Project → Export → JSON → save to:
     data/gold_standard/annotations/annotations.json
   Polygon → bbox conversion is handled here automatically.
   (Label Studio's built-in bbox export also works but is less accurate
   than converting your own polygons, since you annotated with polygons.)

 OUTPUTS:
   checkpoints/yolo/best.pt           ← YOLO weights (read by generate_tier1_masks.py v3)
   data/yolo_dataset/                 ← YOLO-format image + label dataset
     images/train/  images/val/
     labels/train/  labels/val/
     dataset.yaml
   logs/yolo_training_metrics.csv     ← per-epoch mAP / loss
   logs/yolo_qa_calibration.csv       ← SAM2 confidence threshold calibration
                                         (read by generate_tier1_masks.py v3)
================================================================================
"""

import csv
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import (
    SEED,
    YOLO_ANNOTATION_FILE, YOLO_IMAGES_DIR, YOLO_ANNOTATIONS_DIR,
    GOLD_ANNOTATION_FILE, GOLD_IMAGES_DIR, GOLD_ANNOTATIONS_DIR,
    GOLD_IOU_TARGET_MEAN,
    YOLO_DATASET_DIR, YOLO_WEIGHTS_DIR, YOLO_WEIGHTS_BEST,
    YOLO_IMG_SIZE, YOLO_EPOCHS, YOLO_BATCH_SIZE, YOLO_LR0,
    YOLO_PATIENCE, YOLO_CONF_THRESHOLD, YOLO_IOU_NMS,
    YOLO_MIN_ANNOTATIONS, YOLO_VAL_SPLIT, YOLO_CLASS_NAME,
    YOLO_WORKERS, YOLO_QA_CALIB_FILE,
    SAM2_CHECKPOINT, SAM2_CONFIG,
    LOGS_DIR,
)
from image_utils import load_image_rgb

random.seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LABEL STUDIO JSON PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_label_studio_json(json_path: Path, images_dir: Path) -> list[dict]:
    """
    Parse a Label Studio JSON export and return per-image annotation records.

    Label Studio polygon format:
      result[i].type == "polygonlabels"
      result[i].value.points — list of [x, y] in PERCENTAGE coordinates (0–100)
      result[i].original_width / original_height — image pixel dimensions

    Returns a list of dicts:
      {
        "img_path":  Path,
        "width":     int,        # original image width in pixels
        "height":    int,        # original image height in pixels
        "boxes_norm": [(cx, cy, w, h), ...]   # YOLO-normalized [0,1]
      }

    Images with no valid polygon annotations are skipped with a warning.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Label Studio JSON not found: {json_path}\n"
            f"Export: Label Studio → Export → JSON → save to that path."
        )

    with open(json_path, encoding="utf-8") as f:
        tasks = json.load(f)

    records    = []
    n_skipped  = 0
    n_no_boxes = 0

    for task in tasks:
        # ── Resolve image filename ─────────────────────────────────────────────
        data_val = task.get("data", {})
        img_field = data_val.get("image", data_val.get("img", ""))
        # Label Studio stores either a URL or a relative path; extract filename
        img_name  = Path(img_field.replace("\\", "/").split("/")[-1].split("?d=")[-1])
        img_path  = images_dir / img_name

        if not img_path.exists():
            # Try without class prefix (fallback for re-exported sets)
            candidates = list(images_dir.glob(f"*{img_name.suffix}"))
            matches    = [c for c in candidates if c.name.endswith(img_name.name)]
            if matches:
                img_path = matches[0]
            else:
                n_skipped += 1
                continue

        # ── Parse polygon annotations ──────────────────────────────────────────
        annotations = task.get("annotations", [])
        if not annotations:
            n_no_boxes += 1
            continue

        # Use the first completed annotation
        result_items = annotations[0].get("result", [])

        boxes_norm = []
        for item in result_items:
            if item.get("type") not in ("polygonlabels", "polygon"):
                continue

            value  = item.get("value", {})
            points = value.get("points", [])   # list of [x%, y%]
            if len(points) < 3:
                continue

            orig_w = item.get("original_width",  1)
            orig_h = item.get("original_height", 1)

            # Convert percentage → pixel
            xs = [pt[0] / 100.0 * orig_w for pt in points]
            ys = [pt[1] / 100.0 * orig_h for pt in points]

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            # Clip to image bounds
            x1 = max(0.0, x1);  y1 = max(0.0, y1)
            x2 = min(float(orig_w), x2);  y2 = min(float(orig_h), y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # YOLO normalized format: cx, cy, w, h  ∈ [0,1]
            cx = (x1 + x2) / 2.0 / orig_w
            cy = (y1 + y2) / 2.0 / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h

            boxes_norm.append((cx, cy, bw, bh))

        if not boxes_norm:
            n_no_boxes += 1
            continue

        records.append({
            "img_path":   img_path,
            "width":      orig_w,
            "height":     orig_h,
            "boxes_norm": boxes_norm,
        })

    print(f"  Parsed {len(records):,} annotated images "
          f"({n_skipped} missing files, {n_no_boxes} no valid polygons)")
    return records


def polygon_to_sam2_mask(points_pct: list, width: int, height: int) -> np.ndarray:
    """
    Render a Label Studio polygon (percentage coords) as a binary uint8 mask.
    Used during calibration to compute IoU against SAM2 outputs.

    Returns: H×W uint8 mask (0 or 255).
    """
    pts_px = np.array(
        [[pt[0] / 100.0 * width, pt[1] / 100.0 * height] for pt in points_pct],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_px], 255)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — YOLO DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_yolo_dataset(
    records: list[dict],
    out_dir: Path,
    val_split: float = YOLO_VAL_SPLIT,
) -> Path:
    """
    Write YOLO-format dataset from annotation records and return path to dataset.yaml.

    Layout:
      out_dir/
        images/train/  images/val/
        labels/train/  labels/val/
        dataset.yaml
    """
    random.shuffle(records)
    n_val   = max(1, int(len(records) * val_split))
    val_set = records[:n_val]
    trn_set = records[n_val:]

    for split_name, split_records in [("train", trn_set), ("val", val_set)]:
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for rec in split_records:
            # Copy image
            dest_img = img_dir / rec["img_path"].name
            if not dest_img.exists():
                shutil.copy2(rec["img_path"], dest_img)

            # Write YOLO label file (one line per box: class cx cy w h)
            lbl_path = lbl_dir / (rec["img_path"].stem + ".txt")
            with open(lbl_path, "w") as f:
                for cx, cy, bw, bh in rec["boxes_norm"]:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # Write dataset.yaml
    yaml_path = out_dir / "dataset.yaml"
    yaml_content = (
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: 1\n"
        f"names: ['{YOLO_CLASS_NAME}']\n"
    )
    yaml_path.write_text(yaml_content)

    print(f"  Dataset: {len(trn_set):,} train  |  {len(val_set):,} val")
    print(f"  YAML   : {yaml_path}")
    return yaml_path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — YOLOV8N TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_yolo(yaml_path: Path) -> None:
    """
    Train YOLOv8n (nano) via the Ultralytics API.
    Saves best weights to YOLO_WEIGHTS_DIR/best.pt automatically.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics not installed. Run:\n"
            "  pip install ultralytics>=8.0.0"
        )

    # ── GPU preflight check ───────────────────────────────────────────────────
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU detected : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
        device = "0"
    else:
        print()
        print("  [WARNING] CUDA is NOT available — training will run on CPU.")
        print("            This will be extremely slow and will likely exhaust")
        print("            your 16 GB RAM.  Possible causes:")
        print("              1. PyTorch was installed without CUDA support.")
        print("                 Run: pip install torch torchvision --index-url")
        print("                      https://download.pytorch.org/whl/cu128")
        print("              2. NVIDIA drivers not loaded — check: nvidia-smi")
        print("              3. RTX 5060 (Blackwell) may need PyTorch nightly.")
        print("                 See: https://pytorch.org/get-started/locally/")
        print()
        answer = input("  Continue on CPU anyway? [y/N]: ")
        if answer.strip().lower() != "y":
            print("  Aborted. Fix CUDA first and re-run.")
            return
        device = "cpu"

    YOLO_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")    # downloads pretrained weights on first run
    print(f"  Training YOLOv8n for up to {YOLO_EPOCHS} epochs "
          f"(patience {YOLO_PATIENCE}) ...")

    model.train(
        data      = str(yaml_path),
        epochs    = YOLO_EPOCHS,
        imgsz     = YOLO_IMG_SIZE,
        batch     = YOLO_BATCH_SIZE,
        lr0       = YOLO_LR0,
        patience  = YOLO_PATIENCE,
        workers   = YOLO_WORKERS,   # cap dataloader threads to limit RAM usage
        cache     = False,          # do NOT cache images in RAM (prevents OOM on 16 GB)
        project   = str(YOLO_WEIGHTS_DIR.parent),
        name      = "yolo",
        exist_ok  = True,
        seed      = SEED,
        device    = device,
        verbose   = False,
    )

    # Ultralytics saves to <project>/<name>/weights/best.pt
    trained_best = YOLO_WEIGHTS_DIR.parent / "yolo" / "weights" / "best.pt"
    if trained_best.exists() and trained_best != YOLO_WEIGHTS_BEST:
        YOLO_WEIGHTS_BEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained_best, YOLO_WEIGHTS_BEST)

    print(f"  Best weights saved: {YOLO_WEIGHTS_BEST}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — VALIDATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_yolo(yaml_path: Path) -> dict:
    """
    Run validation on the val split and return metrics dict.
    Prints mAP@0.5, precision, recall.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return {}

    if not YOLO_WEIGHTS_BEST.exists():
        print("  [WARN] best.pt not found — skipping evaluation.")
        return {}

    model   = YOLO(str(YOLO_WEIGHTS_BEST))
    results = model.val(data=str(yaml_path), verbose=False)
    metrics = {
        "map50":     float(results.box.map50),
        "map50_95":  float(results.box.map),
        "precision": float(results.box.p.mean()),
        "recall":    float(results.box.r.mean()),
    }

    print(f"\n  YOLOv8n validation results:")
    print(f"    mAP@0.5       : {metrics['map50']:.4f}")
    print(f"    mAP@0.5:0.95  : {metrics['map50_95']:.4f}")
    print(f"    Precision     : {metrics['precision']:.4f}")
    print(f"    Recall        : {metrics['recall']:.4f}")

    # Save to logs
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        LOGS_DIR / "yolo_val_metrics.csv", index=False)

    if metrics["map50"] < 0.70:
        print(f"\n  [WARN] mAP@0.5 = {metrics['map50']:.3f} is below 0.70.")
        print("         Consider annotating more images before using YOLO prompts.")
        print(f"         Current annotations may be below {YOLO_MIN_ANNOTATIONS}.")
    else:
        print(f"\n  [OK] mAP@0.5 = {metrics['map50']:.3f} — acceptable for SAM2 prompting.")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAM2 QA CONFIDENCE CALIBRATION  (optional — requires SAM2 + GPU)
# ══════════════════════════════════════════════════════════════════════════════

def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute binary IoU between two uint8 masks (values 0 or 1)."""
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return inter / max(union, 1)


def calibrate_sam2_confidence(
    json_path: Path,
    images_dir: Path,
) -> float | None:
    """
    Calibrate the SAM2 QA confidence threshold against gold-standard IoU.

    For each gold-standard image:
      1. Run YOLO → get leaf bounding box
      2. Constrain HSV centroid search to pixels inside the box
      3. Run SAM2 with box prompt + constrained centroid points
      4. Compute IoU(sam2_mask, human_polygon_mask)
      5. Record (sam2_mean_conf, iou)

    Then sweep candidate thresholds and find the minimum T such that
    images with mean_conf ≥ T achieve mean_IoU ≥ GOLD_IOU_TARGET_MEAN.
    Writes calibration curve + chosen threshold to YOLO_QA_CALIB_FILE.

    Returns the calibrated threshold, or None if calibration cannot run.
    """
    # ── Dependency checks ─────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO as _YOLO
    except ImportError:
        print("  [SKIP] Ultralytics not available — skipping calibration.")
        return None

    if not YOLO_WEIGHTS_BEST.exists():
        print("  [SKIP] YOLO weights not found — skipping calibration.")
        return None

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _sam2 = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device="cuda")
        sam2_pred = SAM2ImagePredictor(_sam2)
    except Exception as e:
        print(f"  [SKIP] SAM2 unavailable ({e}) — skipping calibration.")
        return None

    yolo_model = _YOLO(str(YOLO_WEIGHTS_BEST))

    # ── Parse gold standard annotations ──────────────────────────────────────
    if not json_path.exists():
        print(f"  [SKIP] Annotations not found at {json_path}.")
        return None

    with open(json_path, encoding="utf-8") as f:
        tasks = json.load(f)

    # ── Run pipeline per image ────────────────────────────────────────────────
    calibration_rows = []
    n_processed = 0

    from image_utils import to_hsv   # inline import — only needed here

    for task in tasks:
        data_val  = task.get("data", {})
        img_field = data_val.get("image", data_val.get("img", ""))
        img_name  = Path(img_field.replace("\\", "/").split("/")[-1].split("?d=")[-1])
        img_path  = images_dir / img_name

        if not img_path.exists():
            continue

        img_rgb = load_image_rgb(img_path)
        if img_rgb is None:
            continue

        h_img, w_img = img_rgb.shape[:2]

        # ── Gold standard mask from first polygon ─────────────────────────────
        annotations  = task.get("annotations", [])
        if not annotations:
            continue

        result_items = annotations[0].get("result", [])
        poly_item    = next(
            (r for r in result_items
             if r.get("type") in ("polygonlabels", "polygon")), None)
        if poly_item is None:
            continue

        poly_pts = poly_item.get("value", {}).get("points", [])
        if len(poly_pts) < 3:
            continue

        orig_w = poly_item.get("original_width",  w_img)
        orig_h = poly_item.get("original_height", h_img)
        gold_mask = polygon_to_sam2_mask(poly_pts, orig_w, orig_h)

        # Resize gold mask to loaded image size if needed
        if gold_mask.shape != (h_img, w_img):
            gold_mask = cv2.resize(gold_mask, (w_img, h_img),
                                   interpolation=cv2.INTER_NEAREST)
        gold_binary = (gold_mask > 127).astype(np.uint8)

        # ── YOLO detection ────────────────────────────────────────────────────
        det = yolo_model(img_rgb,
                         conf=YOLO_CONF_THRESHOLD,
                         iou=YOLO_IOU_NMS,
                         verbose=False)
        boxes_xyxy = det[0].boxes.xyxy.cpu().numpy() if det[0].boxes else np.empty((0, 4))
        confs_det  = det[0].boxes.conf.cpu().numpy() if det[0].boxes else np.empty(0)

        if len(boxes_xyxy) == 0:
            continue    # no detection → skip (don't contaminate calibration)

        best_idx = int(np.argmax(confs_det))
        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[best_idx]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # ── HSV centroid constrained to YOLO box ──────────────────────────────
        roi_rgb = img_rgb[y1:y2, x1:x2]
        if roi_rgb.size == 0:
            continue

        centroid_roi, _ = _centroid_in_roi(roi_rgb)
        # Translate centroid back to full-image coordinates
        cx_full = x1 + centroid_roi[0]
        cy_full = y1 + centroid_roi[1]

        # Build 3 foreground points along vertical leaf axis (clamped to box)
        upper_y = (y1 + cy_full) // 2
        lower_y = (cy_full + y2) // 2
        fg_points = np.array([
            [cx_full, cy_full],
            [cx_full, upper_y],
            [cx_full, lower_y],
        ], dtype=np.float32)
        fg_labels = np.ones(3, dtype=np.int32)

        # ── SAM2 inference ────────────────────────────────────────────────────
        try:
            sam2_pred.set_image(img_rgb)
            masks, scores, logits = sam2_pred.predict(
                point_coords=fg_points,
                point_labels=fg_labels,
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                multimask_output=True,
            )
            best_mask_idx = int(np.argmax(scores))
            prob_map = (1.0 / (1.0 + np.exp(-logits[best_mask_idx]))).squeeze()
        except Exception:
            continue

        # Resize prob_map to image size if needed
        if prob_map.shape != (h_img, w_img):
            prob_map = cv2.resize(prob_map.astype(np.float32),
                                  (w_img, h_img),
                                  interpolation=cv2.INTER_LINEAR)

        sam2_binary  = (prob_map >= 0.5).astype(np.uint8)
        fg_pixels    = sam2_binary.sum()
        mean_conf    = float(prob_map[sam2_binary == 1].mean()) if fg_pixels > 0 else 0.0
        iou          = compute_iou(sam2_binary, gold_binary)

        calibration_rows.append({
            "image":     img_path.name,
            "mean_conf": round(mean_conf, 4),
            "iou":       round(iou, 4),
        })
        n_processed += 1

        if n_processed % 50 == 0:
            print(f"    Calibration: {n_processed} images processed ...")

    if not calibration_rows:
        print("  [SKIP] No calibration data collected.")
        return None

    calib_df = pd.DataFrame(calibration_rows).sort_values("mean_conf").reset_index(drop=True)

    # ── Sweep thresholds to find calibrated T ─────────────────────────────────
    thresholds    = np.arange(0.50, 0.95, 0.01)
    curve_rows    = []
    chosen_thresh = 0.65   # fallback if target is never reached

    for T in thresholds:
        subset     = calib_df[calib_df["mean_conf"] >= T]
        n_sub      = len(subset)
        mean_iou   = float(subset["iou"].mean()) if n_sub > 0 else 0.0
        coverage   = n_sub / max(len(calib_df), 1)  # fraction of images retained
        curve_rows.append({
            "conf_threshold": round(float(T), 2),
            "mean_iou":       round(mean_iou, 4),
            "n_retained":     n_sub,
            "coverage":       round(coverage, 4),
        })
        # Accept the LOWEST threshold that reaches the IoU target
        # (maximize data retained while meeting quality bar)
        if mean_iou >= GOLD_IOU_TARGET_MEAN and T < chosen_thresh:
            chosen_thresh = float(T)

    curve_df = pd.DataFrame(curve_rows)
    # Mark the chosen threshold row
    curve_df["chosen"] = curve_df["conf_threshold"] == round(chosen_thresh, 2)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(YOLO_QA_CALIB_FILE, index=False)
    calib_df.to_csv(LOGS_DIR / "yolo_calib_raw.csv", index=False)

    print(f"\n  Calibration results ({n_processed} gold-standard images):")
    print(f"    Mean IoU  (all)               : {calib_df['iou'].mean():.4f}")
    print(f"    IoU target                    : {GOLD_IOU_TARGET_MEAN:.2f}")
    print(f"    Chosen conf_threshold         : {chosen_thresh:.2f}")

    retained = curve_df[curve_df["conf_threshold"] == round(chosen_thresh, 2)]
    if not retained.empty:
        row = retained.iloc[0]
        print(f"    At T={chosen_thresh:.2f}: mean_IoU={row['mean_iou']:.4f}  "
              f"coverage={row['coverage']*100:.1f}%")

    print(f"  Calibration curve saved: {YOLO_QA_CALIB_FILE}")
    print(f"  generate_tier1_masks.py v3 will use conf_threshold = {chosen_thresh:.2f}")

    return chosen_thresh


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _centroid_in_roi(roi_rgb: np.ndarray) -> tuple[tuple[int, int], str]:
    """
    Compute HSV-based leaf centroid restricted to a cropped ROI.
    Used during calibration to mirror the v3 generate_tier1_masks logic.
    Returns ((cx, cy), strategy) in ROI-local coordinates.
    """
    from config import (
        SAM2_GREEN_H_MIN, SAM2_GREEN_H_MAX,
        SAM2_GREEN_S_MIN, SAM2_GREEN_V_MIN,
    )
    from image_utils import to_hsv

    _YELLOW_H_MIN = 15;  _YELLOW_H_MAX = 45
    _YELLOW_S_MIN = 40;  _YELLOW_V_MIN = 60
    _MIN_TISSUE   = 0.05

    hsv   = to_hsv(roi_rgb)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total   = roi_rgb.shape[0] * roi_rgb.shape[1]

    green  = ((h >= SAM2_GREEN_H_MIN) & (h <= SAM2_GREEN_H_MAX) &
              (s >= SAM2_GREEN_S_MIN) & (v >= SAM2_GREEN_V_MIN)).astype(np.uint8)
    yellow = ((h >= _YELLOW_H_MIN) & (h <= _YELLOW_H_MAX) &
              (s >= _YELLOW_S_MIN) & (v >= _YELLOW_V_MIN)).astype(np.uint8)

    has_g = green.sum()  / total >= _MIN_TISSUE
    has_y = yellow.sum() / total >= _MIN_TISSUE

    if has_g and has_y:
        tissue = np.clip(green + yellow, 0, 1).astype(np.uint8)
        strat  = "combined_centroid"
    elif has_g:
        tissue = green;  strat = "green_centroid"
    elif has_y:
        tissue = yellow; strat = "yellow_centroid"
    else:
        cy = roi_rgb.shape[0] // 2
        cx = roi_rgb.shape[1] // 2
        return (cx, cy), "center_fallback"

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(tissue, 8)
    if n_labels < 2:
        return (roi_rgb.shape[1] // 2, roi_rgb.shape[0] // 2), "center_fallback"

    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (int(centroids[best][0]), int(centroids[best][1])), strat


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.time()

    print("=" * 72)
    print("  Yellow MAIze | Phase 1b: YOLOv8n Leaf Detector Training")
    print("=" * 72)

    # ── Preflight: annotation count check ────────────────────────────────────
    if not YOLO_ANNOTATION_FILE.exists():
        print(f"[FATAL] Annotation file not found: {YOLO_ANNOTATION_FILE}")
        print("  Export from Label Studio → Export → JSON → place at that path.")
        return

    with open(YOLO_ANNOTATION_FILE, encoding="utf-8") as f:
        tasks = json.load(f)

    n_tasks = len(tasks)
    print(f"\n  Annotation tasks found: {n_tasks:,}")
    if n_tasks < YOLO_MIN_ANNOTATIONS:
        print(f"\n  [WARN] Only {n_tasks} annotated images found. "
              f"Target is {YOLO_MIN_ANNOTATIONS}.")
        print("         YOLOv8n typically needs 400–500 annotations for "
              "reliable leaf detection.")
        answer = input("  Proceed with training anyway? [y/N]: ")
        if answer.strip().lower() != "y":
            print("  Aborted. Annotate more images in Label Studio and re-run.")
            return

    # ── Step 1: Parse Label Studio JSON ──────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  Step 1 — Parsing Label Studio annotations")
    records = parse_label_studio_json(YOLO_ANNOTATION_FILE, YOLO_IMAGES_DIR)

    if len(records) < 20:
        print(f"[FATAL] Only {len(records)} usable records after parsing. Check paths.")
        return

    # ── Step 2: Build YOLO dataset ────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  Step 2 — Building YOLO dataset")
    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)
    YOLO_DATASET_DIR.mkdir(parents=True)

    yaml_path = build_yolo_dataset(records, YOLO_DATASET_DIR)

    # ── Step 3: Train YOLOv8n ─────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  Step 3 — Training YOLOv8n")
    train_yolo(yaml_path)

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  Step 4 — Evaluating on validation split")
    metrics = evaluate_yolo(yaml_path)

    # ── Step 5: SAM2 QA calibration (optional) ───────────────────────────────
    print(f"\n{'─' * 40}")
    print("  Step 5 — SAM2 QA confidence calibration (optional)")
    print("           Requires SAM2 + GPU. Skips gracefully if unavailable.")
    chosen_thresh = calibrate_sam2_confidence(
        GOLD_ANNOTATION_FILE, GOLD_IMAGES_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 60
    print(f"\n{'═' * 72}")
    print(f"  Done in {elapsed:.1f} min")
    print(f"  YOLO weights   : {YOLO_WEIGHTS_BEST}")
    if chosen_thresh is not None:
        print(f"  Calibrated QA  : {chosen_thresh:.2f}  (written to {YOLO_QA_CALIB_FILE})")
    else:
        print(f"  Calibration    : skipped — generate_tier1_masks.py v3 will use "
              f"default _QA_MIN_CONFIDENCE = 0.65")
    print(f"\n  NEXT STEP: python generate_tier1_masks.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
