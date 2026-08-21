"""
================================================================================
 validate_symptom.py — Symptom Teacher Qualitative + LAB Comparison Validation
================================================================================
 PURPOSE:
   Produces the visual outputs Chapter 4 (§4.1.6) needs and that previously
   had no code behind them at all:

     Figure 4.16 — HealthyAE reconstruction and anomaly map
     Figure 4.17 — Human symptom mask vs. Symptom Teacher predicted mask
     Figure 4.18 — Symptom Teacher vs. legacy LAB/HSV comparison
     Figure 4.19 — Human symptom mask vs. STUDENT predicted mask, with
                    classification (HEALTHY/MSV/MLN) and severity % agreement
     Table 4.13  — Symptom Teacher vs. legacy LAB, numeric IoU comparison
     Table 4.14  — Human vs. Student: IoU, classification accuracy,
                    severity % mean absolute error (all in the same CSV as
                    Table 4.13 — see OUTPUTS below)

   This script does NOT reimplement inference logic. It reuses
   load_healthy_ae(), load_symptom_teacher(), predict_symptom_mask(),
   parse_symptom_annotations(), and compute_iou() directly from
   train_symptom_model.py, so behavior here always matches training exactly
   — the only new code is the visualization/scoring layer on top. Student
   loading and letterbox inference mirror validate_gold_standard.py's
   load_student()/_predict_letterbox() exactly.

   NOTE ON STUDENT SCOPE: the Student (train_student.py) predicts a single
   combined symptom channel (not separate MSV/MLN masks like the Symptom
   Teacher) plus a separate classification head (HEALTHY/MSV/MLN) and a
   separate severity-% regression head. Figure 4.19/Table 4.14 therefore
   validate three things simultaneously: symptom mask IoU (against whichever
   human mask matches the image's true category), classification accuracy,
   and severity % accuracy (human severity % computed as human symptom
   pixels / human leaf pixels, via the same CVAT export's leaf polygons).

 PREREQUISITE:
   SYMPTOM_ANNOTATION_FILE (data/gold_standard/annotations/symptom_annotations.json)
   must already exist — export from CVAT as COCO 1.0 (see
   parse_symptom_annotations()'s docstring in train_symptom_model.py). No
   script can produce Figures 4.16-4.18 without this human annotation pass
   having already happened. Figure 4.19/Table 4.14 additionally require a
   trained Student checkpoint (STUDENT_CKPT_DIR) — if none is found, that
   section is skipped with a warning rather than failing the whole run.

 OUTPUTS:
   reports/symptom_vs_lab_comparison.csv               — Tables 4.13 + 4.14
   reports/symptom_overlays/{stem}_ae_anomaly.jpg        — Figure 4.16
   reports/symptom_overlays/{stem}_human_vs_pred.jpg     — Figure 4.17
   reports/symptom_overlays/{stem}_lab_vs_teacher.jpg    — Figure 4.18
   reports/symptom_overlays/{stem}_human_vs_student.jpg  — Figure 4.19

 USAGE:
   python validate_symptom.py                  # 5 qualitative samples/class + full numeric tables
   python validate_symptom.py --n 10            # more qualitative samples per class
   python validate_symptom.py --skip-lab        # no LAB comparison (Figure 4.18/Table 4.13 cols)
   python validate_symptom.py --skip-student     # no Student comparison (Figure 4.19/Table 4.14 cols)
   python validate_symptom.py --student-variant mobilenet_v3_small_cbam

 CONSUMED BY (once the outputs above exist):
   generate_charts.py --symptom  → symptom_vs_lab_bar.png (from the CSV above)
   generate_report.py            → Section 4, overlay galleries read directly
                                    from reports/symptom_overlays/
================================================================================
"""

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch

from config import (
    CLASSES,
    GOLD_IMAGES_DIR,
    SYMPTOM_EXTRA_IMAGES_DIR,
    REPORTS_DIR,
    SEED,
    SYMPTOM_ANNOTATION_FILE,
    HEALTHY_AE_IMG_SIZE,
    STUDENT_BEST_VARIANT,
    STUDENT_CKPT_DIR,
    STUDENT_FACTORY_MODE,
    STUDENT_IMG_SIZE,
    STUDENT_MAX_SEVERITY,
)
from image_utils import load_image_rgb
from train_symptom_model import (
    DEVICE,
    compute_iou,
    load_healthy_ae,
    load_symptom_teacher,
    parse_symptom_annotations,
    predict_symptom_mask,
    _build_mask,
)

OVERLAY_ALPHA = 0.40
N_PER_CLASS_DEFAULT = 5
OVERLAY_DIR = REPORTS_DIR / "symptom_overlays"


# ══════════════════════════════════════════════════════════════════════════════
# HEALTHYAE — RECONSTRUCTION + ERROR MAP (Figure 4.16)
# ══════════════════════════════════════════════════════════════════════════════

def get_ae_reconstruction(model, img_rgb: np.ndarray,
                          img_size: int = HEALTHY_AE_IMG_SIZE):
    """
    Mirrors compute_ae_error_map()'s preprocessing exactly (train_symptom_model.py),
    but returns the reconstructed image, the error map, AND recon_std from a
    single forward pass. compute_ae_error_map() only returns the error map —
    Figure 4.16 also needs to show the reconstruction itself, so this is a
    thin wrapper rather than a duplicate implementation of the model logic.

    recon_std is the same diagnostic train_healthy_ae() logs per-epoch during
    training (near zero = collapsed decoder, predicting a near-constant
    output regardless of input). Returning it here lets this script's own
    summary flag the same failure mode at validation time, not just during
    training — useful if validate_symptom.py is run against a checkpoint
    that predates the collapse-detection fix.
    """
    orig_h, orig_w = img_rgb.shape[:2]
    tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
    ])
    img_padded = tf(image=img_rgb)["image"]
    tensor = (torch.from_numpy(img_padded).permute(2, 0, 1).float()
             .unsqueeze(0).to(DEVICE) / 255.0)

    with torch.no_grad():
        recon = model(tensor)

    recon_std = recon.std().item()

    recon_np = (recon.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
               ).clip(0, 255).astype(np.uint8)
    recon_resized = cv2.resize(recon_np, (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)

    error = (tensor - recon).pow(2).mean(dim=1).squeeze(0).cpu().numpy()
    error = cv2.resize(error.astype(np.float32), (orig_w, orig_h),
                       interpolation=cv2.INTER_LINEAR)
    lo, hi = error.min(), error.max()
    error = (error - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(error)

    return recon_resized, error.astype(np.float32), recon_std


def draw_ae_anomaly_panel(img_rgb, recon_rgb, error_map, category, stem):
    """Figure 4.16: original | AE reconstruction | reconstruction-error heatmap."""
    h, w = img_rgb.shape[:2]
    heatmap = cv2.applyColorMap((error_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    panel = np.hstack([img_rgb, recon_rgb, heatmap_rgb]).copy()
    for i, label in enumerate(["Original", "AE Reconstruction", "Reconstruction Error"]):
        x = i * w + 8
        cv2.putText(panel, label, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(panel, label, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    footer = f"{category} | {stem}  (higher error = more unlike a healthy leaf)"
    cv2.putText(panel, footer, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(panel, footer, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# HUMAN MASK VS PREDICTED MASK (Figure 4.17)
# ══════════════════════════════════════════════════════════════════════════════

def draw_human_vs_pred_panel(img_rgb, human_mask, pred_prob, iou, category, stem):
    """
    Figure 4.17. Same TP/FP/FN color convention as validate_gold_standard.py's
    draw_comparison_overlay(), so Chapter 4's overlay figures look consistent
    with each other across the leaf-Teacher and Symptom-Teacher sections.
    """
    pred_mask = pred_prob >= 0.5
    human_bool = human_mask.astype(bool)

    overlay = img_rgb.copy().astype(np.float32)
    tp = np.logical_and(human_bool, pred_mask)
    fn = np.logical_and(human_bool, ~pred_mask)
    fp = np.logical_and(pred_mask, ~human_bool)

    cyan  = np.array([0, 220, 220], dtype=np.float32)
    green = np.array([0, 220, 0],   dtype=np.float32)
    red   = np.array([220, 0, 0],   dtype=np.float32)

    overlay[tp] = overlay[tp] * (1 - OVERLAY_ALPHA) + cyan * OVERLAY_ALPHA
    overlay[fn] = overlay[fn] * (1 - OVERLAY_ALPHA) + green * OVERLAY_ALPHA
    overlay[fp] = overlay[fp] * (1 - OVERLAY_ALPHA) + red * OVERLAY_ALPHA
    overlay = overlay.clip(0, 255).astype(np.uint8)

    for mask, color in [(human_mask.astype(np.uint8), (0, 200, 0)),
                        (pred_mask.astype(np.uint8), (0, 0, 220))]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    text = f"{category} | {stem}  |  IoU={iou:.3f}"
    cv2.putText(overlay, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    legend = "Green=missed (human only) | Cyan=correct | Red=extra (predicted only)"
    cv2.putText(overlay, legend, (8, overlay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(overlay, legend, (8, overlay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return overlay


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT — HUMAN SYMPTOM VS STUDENT SYMPTOM (Figure 4.19, Table 4.14)
# ══════════════════════════════════════════════════════════════════════════════
# NOTE ON SCOPE: the Student (train_student.py) outputs a single combined
# symptom channel (seg_logits Ch1) rather than separate MSV/MLN channels like
# the Symptom Teacher — class identity comes from the classification head
# (cls_out) and severity from a dedicated regression head (sev_out), not from
# per-class masks. This section therefore validates three things the Symptom
# Teacher comparison above does not: (1) symptom mask IoU using whichever
# human mask (MSV or MLN) matches the image's true category, (2) HEALTHY/
# MSV/MLN classification accuracy, (3) severity % regression accuracy against
# a human-derived severity % (human symptom pixels / human leaf pixels).

def resolve_best_student() -> tuple[str, str]:
    """
    Re-derive the actual Stage 2 winning (encoder, mode) from
    logs/student_comparison_stage2.csv, the same way evaluate_xai.py does,
    instead of trusting config.py's STUDENT_BEST_VARIANT/STUDENT_FACTORY_MODE
    directly (those are only auto-updated by select_best_pipeline.py).
    Falls back to config.py's values if the CSV isn't available yet.
    """
    from config import LOGS_DIR
    stage2_csv = LOGS_DIR / "student_comparison_stage2.csv"
    if stage2_csv.exists():
        df = pd.read_csv(stage2_csv)
        if not df.empty and "best_composite" in df.columns:
            best_row = df.loc[df["best_composite"].astype(float).idxmax()]
            return str(best_row["encoder"]), str(best_row["mode"])
    return STUDENT_BEST_VARIANT, STUDENT_FACTORY_MODE


def load_student(encoder_variant: str = STUDENT_BEST_VARIANT, mode: str = None):
    """
    Mirrors validate_gold_standard.py's load_student() exactly, so behavior
    (checkpoint resolution, CBAM detection) matches that script.
    """
    try:
        from train_student import StudentModel
    except ImportError as e:
        raise ImportError(f"Could not import StudentModel from train_student.py: {e}")

    use_cbam = "cbam" in encoder_variant
    model = StudentModel(encoder_name=encoder_variant, use_cbam=use_cbam)

    if mode is None:
        mode = STUDENT_FACTORY_MODE

    # Checkpoints live under stage1/ or stage2/, never flat in STUDENT_CKPT_DIR
    # (train_student.py writes BOTH stages into stage-specific subfolders).
    # mode_b is reused from Stage 1 (never re-saved in stage2/), so check
    # stage1/ first for mode_b and stage2/ first for every other mode.
    ckpt_filename = f"student_{encoder_variant}_{mode}_best.pth"
    search_order = ["stage1", "stage2"] if mode == "mode_b" else ["stage2", "stage1"]

    ckpt_path = None
    for subdir in search_order:
        candidate = STUDENT_CKPT_DIR / subdir / ckpt_filename
        if candidate.exists():
            ckpt_path = candidate
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            f"Student checkpoint not found for {encoder_variant}/{mode}. "
            f"Checked: {STUDENT_CKPT_DIR / search_order[0] / ckpt_filename}  "
            f"and: {STUDENT_CKPT_DIR / search_order[1] / ckpt_filename}"
        )

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    print(f"  Student loaded: {encoder_variant} / {mode}  ({ckpt_path})")
    return model


def _load_leaf_masks(json_path: Path, records: list) -> dict:
    """
    Re-reads SYMPTOM_ANNOTATION_FILE (already parsed once by
    parse_symptom_annotations for the msv/mln masks) to also rasterize the
    'maize-leaf' polygons, so human severity % can be computed as
    (human symptom pixels / human leaf pixels) — the same definition the
    CIMMYT brackets in config.py use downstream. This is a second, cheap
    pass over the same JSON rather than a change to
    parse_symptom_annotations() itself, to avoid touching training code.

    Returns {filename: leaf_mask (np.uint8 H×W)}, keyed by image filename —
    only for images that also appear in `records` (avoids wasted rasterization).
    """
    from config import MAIZE_LEAF_LABEL_NAME
    import json as _json
    from collections import defaultdict

    wanted_names = {r["img_path"].name for r in records}

    with open(json_path, encoding="utf-8") as f:
        data = _json.load(f)

    id_to_image = {img["id"]: img for img in data.get("images", [])}
    id_to_cat = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    leaf_cat_ids = {cid for cid, name in id_to_cat.items()
                    if name.lower() == MAIZE_LEAF_LABEL_NAME.lower()}

    leaf_anns = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("category_id") in leaf_cat_ids:
            leaf_anns[ann.get("image_id")].append(ann)

    leaf_masks = {}
    for img_id, img_info in id_to_image.items():
        fname = Path(img_info["file_name"]).name
        if fname not in wanted_names:
            continue
        h, w = img_info.get("height"), img_info.get("width")
        if not h or not w:
            continue
        leaf_masks[fname] = _build_mask(leaf_anns.get(img_id, []), h, w)

    return leaf_masks


def _student_predict_full(model, img_rgb: np.ndarray, img_size: int = STUDENT_IMG_SIZE):
    """
    Single forward pass returning everything Table 4.14 needs:
      symptom_prob : H×W float32, Ch1 (symptom) sigmoid probs, letterbox-reversed
      cls_pred     : str, argmax over CLASSES
      sev_pred_pct : float, severity 0-100 (sev_out is 0-1, scaled by
                     STUDENT_MAX_SEVERITY same as inference elsewhere)

    Letterbox handling matches validate_gold_standard.py's _predict_letterbox
    exactly (LongestMaxSize + centered PadIfNeeded, ImageNet normalization)
    so Student inputs here are preprocessed identically to training/other
    validation — this is not a reimplementation with different behavior.
    """
    h_orig, w_orig = img_rgb.shape[:2]
    scale = img_size / max(h_orig, w_orig)
    new_h, new_w = int(h_orig * scale), int(w_orig * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (img_size - new_h) // 2
    pad_bottom = img_size - new_h - pad_top
    pad_left = (img_size - new_w) // 2
    pad_right = img_size - new_w - pad_left
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0])

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_norm = (img_padded.astype(np.float32) / 255.0 - mean) / std
    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg_logits, cls_out, sev_out = model(img_t)

    if seg_logits.shape[1] != 2:
        raise RuntimeError(
            f"Expected StudentModel seg_logits with 2 channels "
            f"(Ch0=leaf, Ch1=symptom), got shape {tuple(seg_logits.shape)}. "
            f"StudentModel's architecture may have changed since this "
            f"validation code was written — check train_student.py's "
            f"StudentModel.forward() before trusting these numbers."
        )

    symptom_prob = torch.sigmoid(seg_logits[:, 1]).squeeze().cpu().numpy()
    y0, y1 = pad_top, pad_top + new_h
    x0, x1 = pad_left, pad_left + new_w
    symptom_prob = cv2.resize(symptom_prob[y0:y1, x0:x1], (w_orig, h_orig),
                              interpolation=cv2.INTER_LINEAR)

    cls_idx = int(cls_out.argmax(dim=1).item())
    cls_pred = CLASSES[cls_idx] if cls_idx < len(CLASSES) else f"UNKNOWN_{cls_idx}"
    sev_pred_pct = float(sev_out.item()) * STUDENT_MAX_SEVERITY

    return symptom_prob, cls_pred, sev_pred_pct


def draw_human_vs_student_panel(img_rgb, human_mask, student_prob, iou_student,
                                cls_true, cls_pred, sev_human, sev_student,
                                category, stem):
    """
    Figure 4.19. Same TP/FP/FN convention as draw_human_vs_pred_panel (Figure
    4.17), plus a footer reporting classification and severity agreement,
    since those are Student-only outputs with no Symptom Teacher equivalent.
    """
    panel = draw_human_vs_pred_panel(img_rgb, human_mask, student_prob,
                                     iou_student, category, stem)
    cls_ok = "OK" if cls_pred == cls_true else "MISMATCH"
    text2 = (f"cls: true={cls_true} pred={cls_pred} [{cls_ok}]  |  "
             f"severity: human={sev_human:.1f}%  student={sev_student:.1f}%")
    cv2.putText(panel, text2, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(panel, text2, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# LAB VS SYMPTOM TEACHER (Figure 4.18)
# ══════════════════════════════════════════════════════════════════════════════

def draw_lab_vs_teacher_panel(img_rgb, lab_mask, teacher_prob, human_mask,
                              iou_lab, iou_teacher, category, stem):
    """Figure 4.18: Original | Legacy LAB | Symptom Teacher | Human ground truth."""
    h, w = img_rgb.shape[:2]
    teacher_mask = (teacher_prob >= 0.5).astype(np.uint8) * 255

    def mask_panel(mask_uint8, label, iou_val=None):
        vis = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
        text = label if iou_val is None else f"{label}  (IoU={iou_val:.3f})"
        cv2.putText(vis, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return vis

    orig_vis = img_rgb.copy()
    cv2.putText(orig_vis, "Original", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(orig_vis, "Original", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    lab_vis     = mask_panel(lab_mask.astype(np.uint8) * 255, "Legacy LAB", iou_lab)
    teacher_vis = mask_panel(teacher_mask, "Symptom Teacher", iou_teacher)
    human_vis   = mask_panel(human_mask.astype(np.uint8) * 255, "Human ground truth")

    panel = np.hstack([orig_vis, lab_vis, teacher_vis, human_vis]).copy()
    footer = f"{category} | {stem}"
    cv2.putText(panel, footer, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(panel, footer, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PER_CLASS_DEFAULT,
                        help="Qualitative overlay figures per class (default 5)")
    parser.add_argument("--skip-lab", action="store_true",
                        help="Skip Figure 4.18 and Table 4.13 (LAB comparison)")
    parser.add_argument("--skip-student", action="store_true",
                        help="Skip Figure 4.19 and Table 4.14 (Student comparison)")
    parser.add_argument("--student-variant", type=str, default=None,
                        help="Student encoder variant to load "
                             "(default: auto-resolved from "
                             "logs/student_comparison_stage2.csv, "
                             "falling back to STUDENT_BEST_VARIANT)")
    args = parser.parse_args()

    random.seed(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Symptom Teacher Validation")
    print("=" * 72)

    ae_model = load_healthy_ae()
    if ae_model is None:
        print("[FATAL] No HealthyAE checkpoint found. Run train_symptom_model.py first.")
        return
    symptom_model = load_symptom_teacher()
    if symptom_model is None:
        print("[FATAL] No Symptom Teacher checkpoint found. Run train_symptom_model.py first.")
        return

    if not Path(SYMPTOM_ANNOTATION_FILE).exists():
        print(f"[FATAL] {SYMPTOM_ANNOTATION_FILE} not found.")
        print("        Export symptom annotations from CVAT (COCO 1.0) first — see")
        print("        parse_symptom_annotations()'s docstring in train_symptom_model.py.")
        return

    records = parse_symptom_annotations(
        Path(SYMPTOM_ANNOTATION_FILE), [GOLD_IMAGES_DIR, SYMPTOM_EXTRA_IMAGES_DIR])
    if not records:
        print("[FATAL] No annotated records parsed.")
        return

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    lab_import_ok = False
    if not args.skip_lab:
        try:
            from factory_master import compute_lab_hard_mask
            lab_import_ok = True
        except ImportError as e:
            print(f"  [WARN] Could not import factory_master ({e}) — "
                  "skipping LAB comparison (Figure 4.18 / Table 4.13).")

    student_model = None
    leaf_masks_for_severity = {}
    if not args.skip_student:
        try:
            resolved_variant, resolved_mode = resolve_best_student()
            variant = args.student_variant or resolved_variant
            # If the user explicitly picked a different variant than the
            # resolved one, we can't assume the resolved mode still applies —
            # fall back to STUDENT_FACTORY_MODE in that case.
            mode = resolved_mode if variant == resolved_variant else STUDENT_FACTORY_MODE
            print(f"  Resolved Student: {variant} / {mode}")
            student_model = load_student(variant, mode)
            leaf_masks_for_severity = _load_leaf_masks(Path(SYMPTOM_ANNOTATION_FILE), records)
        except (FileNotFoundError, ImportError) as e:
            print(f"  [WARN] Could not load Student ({e}) — "
                  "skipping Student comparison (Figure 4.19 / Table 4.14).")

    by_class = {c: [r for r in records if r["category"] == c] for c in CLASSES}
    comparison_rows = []

    # ── Qualitative samples: Figures 4.16, 4.17, 4.18, 4.19 ──────────────────
    recon_std_values = []

    for cls in CLASSES:
        pool = by_class.get(cls, [])
        if not pool:
            print(f"  [WARN] No records for {cls}. Skipping.")
            continue
        sample = random.sample(pool, min(args.n, len(pool)))
        print(f"\n  [{cls}] generating {len(sample)} qualitative figures ...")

        for rec in sample:
            img_rgb = load_image_rgb(rec["img_path"])
            if img_rgb is None:
                continue
            stem = rec["img_path"].stem

            # Figure 4.16 — every class, including HEALTHY (this is where the
            # suppression behavior is most visible: error should stay low and
            # unstructured on genuinely healthy leaves).
            recon_rgb, error_map, recon_std = get_ae_reconstruction(ae_model, img_rgb)
            recon_std_values.append(recon_std)
            ae_panel = draw_ae_anomaly_panel(img_rgb, recon_rgb, error_map, cls, stem)
            cv2.imwrite(str(OVERLAY_DIR / f"{stem}_ae_anomaly.jpg"),
                       cv2.cvtColor(ae_panel, cv2.COLOR_RGB2BGR))

            if cls == "HEALTHY":
                continue  # Figures 4.17/4.18 are about symptom masks — N/A for HEALTHY

            human_mask = rec["msv_mask"] if cls == "MSV" else rec["mln_mask"]
            if human_mask.shape != img_rgb.shape[:2]:
                human_mask = cv2.resize(human_mask, (img_rgb.shape[1], img_rgb.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

            pred_prob = predict_symptom_mask(symptom_model, ae_model, img_rgb, category=cls)
            pred_mask = (pred_prob >= 0.5).astype(np.uint8)
            iou_teacher = compute_iou(pred_mask, human_mask)

            hp_panel = draw_human_vs_pred_panel(img_rgb, human_mask, pred_prob,
                                                iou_teacher, cls, stem)
            cv2.imwrite(str(OVERLAY_DIR / f"{stem}_human_vs_pred.jpg"),
                       cv2.cvtColor(hp_panel, cv2.COLOR_RGB2BGR))

            row = {"image": rec["img_path"].name, "category": cls,
                  "iou_symptom_teacher": round(iou_teacher, 4)}

            if lab_import_ok:
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                otsu_sil = (cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
                           ).astype(np.uint8)
                lab_mask = (compute_lab_hard_mask(img_rgb, otsu_sil, cls) > 0).astype(np.uint8)
                iou_lab = compute_iou(lab_mask, human_mask)

                lab_panel = draw_lab_vs_teacher_panel(
                    img_rgb, lab_mask, pred_prob, human_mask,
                    iou_lab, iou_teacher, cls, stem)
                cv2.imwrite(str(OVERLAY_DIR / f"{stem}_lab_vs_teacher.jpg"),
                           cv2.cvtColor(lab_panel, cv2.COLOR_RGB2BGR))
                row["iou_lab"] = round(iou_lab, 4)

            if student_model is not None:
                leaf_mask = leaf_masks_for_severity.get(rec["img_path"].name)
                sev_human = (100.0 * human_mask.sum() / max(leaf_mask.sum(), 1)
                            if leaf_mask is not None else float("nan"))
                student_prob, cls_pred, sev_student = _student_predict_full(
                    student_model, img_rgb)
                student_mask = (student_prob >= 0.5).astype(np.uint8)
                iou_student = compute_iou(student_mask, human_mask)

                stu_panel = draw_human_vs_student_panel(
                    img_rgb, human_mask, student_prob, iou_student,
                    cls, cls_pred, sev_human, sev_student, cls, stem)
                cv2.imwrite(str(OVERLAY_DIR / f"{stem}_human_vs_student.jpg"),
                           cv2.cvtColor(stu_panel, cv2.COLOR_RGB2BGR))

                row["iou_student"]       = round(iou_student, 4)
                row["cls_true"]          = cls
                row["cls_pred"]          = cls_pred
                row["cls_correct"]       = (cls_pred == cls)
                row["severity_human"]    = round(sev_human, 2)
                row["severity_student"]  = round(sev_student, 2)
                row["severity_abs_error"] = round(abs(sev_student - sev_human), 2) \
                                            if not np.isnan(sev_human) else float("nan")

            if lab_import_ok or student_model is not None:
                comparison_rows.append(row)

    # ── Table 4.13/4.14: numeric comparison over EVERY annotated image ───────
    # The qualitative loop above is capped at --n per class for figure
    # generation. The full tables should reflect the full annotated set, not
    # just the sampled subset, so score everything else here (numbers only,
    # no image writes — avoids re-doing the ones already scored above).
    if lab_import_ok or student_model is not None:
        already_scored = {r["image"] for r in comparison_rows}
        remaining = [r for r in records
                    if r["category"] != "HEALTHY" and r["img_path"].name not in already_scored]
        print(f"\n  Scoring remaining {len(remaining)} annotated images "
              f"for the full Table 4.13/4.14 comparison ...")

        for rec in remaining:
            img_rgb = load_image_rgb(rec["img_path"])
            if img_rgb is None:
                continue
            cls = rec["category"]
            human_mask = rec["msv_mask"] if cls == "MSV" else rec["mln_mask"]
            if human_mask.shape != img_rgb.shape[:2]:
                human_mask = cv2.resize(human_mask, (img_rgb.shape[1], img_rgb.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

            pred_prob = predict_symptom_mask(symptom_model, ae_model, img_rgb, category=cls)
            pred_mask = (pred_prob >= 0.5).astype(np.uint8)
            row = {"image": rec["img_path"].name, "category": cls,
                  "iou_symptom_teacher": round(compute_iou(pred_mask, human_mask), 4)}

            if lab_import_ok:
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                otsu_sil = (cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
                           ).astype(np.uint8)
                lab_mask = (compute_lab_hard_mask(img_rgb, otsu_sil, cls) > 0).astype(np.uint8)
                row["iou_lab"] = round(compute_iou(lab_mask, human_mask), 4)

            if student_model is not None:
                leaf_mask = leaf_masks_for_severity.get(rec["img_path"].name)
                sev_human = (100.0 * human_mask.sum() / max(leaf_mask.sum(), 1)
                            if leaf_mask is not None else float("nan"))
                student_prob, cls_pred, sev_student = _student_predict_full(
                    student_model, img_rgb)
                student_mask = (student_prob >= 0.5).astype(np.uint8)

                row["iou_student"]       = round(compute_iou(student_mask, human_mask), 4)
                row["cls_true"]          = cls
                row["cls_pred"]          = cls_pred
                row["cls_correct"]       = (cls_pred == cls)
                row["severity_human"]    = round(sev_human, 2)
                row["severity_student"]  = round(sev_student, 2)
                row["severity_abs_error"] = round(abs(sev_student - sev_human), 2) \
                                            if not np.isnan(sev_human) else float("nan")

            comparison_rows.append(row)

        if comparison_rows:
            df = pd.DataFrame(comparison_rows)
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = REPORTS_DIR / "symptom_vs_lab_comparison.csv"
            df.to_csv(out_path, index=False)

            print(f"\n{'─' * 72}")
            if lab_import_ok:
                print(f"  LAB vs Symptom Teacher comparison ({len(df)} images):")
                print(f"    Mean IoU (LAB pipeline)    : {df['iou_lab'].mean():.4f}")
                print(f"    Mean IoU (Symptom Teacher) : {df['iou_symptom_teacher'].mean():.4f}")
                for cls in CLASSES:
                    cls_df = df[df["category"] == cls]
                    if cls_df.empty:
                        continue
                    print(f"    [{cls}] LAB={cls_df['iou_lab'].mean():.3f}  "
                          f"SymptomTeacher={cls_df['iou_symptom_teacher'].mean():.3f}  "
                          f"(n={len(cls_df)})")

            if student_model is not None:
                print(f"\n  Human vs Student comparison ({len(df)} images):")
                print(f"    Mean IoU (Student symptom mask) : {df['iou_student'].mean():.4f}")
                print(f"    Classification accuracy         : "
                      f"{100 * df['cls_correct'].mean():.1f}%  "
                      f"({df['cls_correct'].sum()}/{len(df)})")
                sev_valid = df.dropna(subset=["severity_abs_error"])
                if not sev_valid.empty:
                    print(f"    Severity MAE (percentage points) : "
                          f"{sev_valid['severity_abs_error'].mean():.2f}")
                for cls in CLASSES:
                    if cls == "HEALTHY":
                        continue
                    cls_df = df[df["category"] == cls]
                    if cls_df.empty:
                        continue
                    print(f"    [{cls}] IoU={cls_df['iou_student'].mean():.3f}  "
                          f"cls_acc={100*cls_df['cls_correct'].mean():.1f}%  "
                          f"(n={len(cls_df)})")

            print(f"  Saved: {out_path}")
        else:
            print("  [SKIP] No comparison data collected.")

    print(f"\n{'─' * 72}")
    if recon_std_values:
        mean_recon_std = sum(recon_std_values) / len(recon_std_values)
        n_collapsed = sum(1 for v in recon_std_values if v < 0.01)
        print(f"  HealthyAE reconstruction check: mean recon_std = "
              f"{mean_recon_std:.5f}  ({n_collapsed}/{len(recon_std_values)} "
              f"samples below 0.01)")
        if mean_recon_std < 0.01:
            print(f"  [WARN] Mean recon_std is below the collapse threshold — "
                  f"the AE decoder is likely producing a near-constant "
                  f"reconstruction regardless of input (see *_ae_anomaly.jpg, "
                  f"the 'AE Reconstruction' panel should show real leaf "
                  f"structure, not a flat block). This is the same check "
                  f"train_healthy_ae() runs during training — see that "
                  f"function's docstring for the diagnosis and fix.")
    print(f"  Overlay figures saved to: {OVERLAY_DIR}")
    print(f"    *_ae_anomaly.jpg        → Figure 4.16")
    print(f"    *_human_vs_pred.jpg     → Figure 4.17")
    if lab_import_ok:
        print(f"    *_lab_vs_teacher.jpg    → Figure 4.18")
    if student_model is not None:
        print(f"    *_human_vs_student.jpg  → Figure 4.19")
    print("=" * 72)


if __name__ == "__main__":
    main()
