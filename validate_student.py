"""
================================================================================
 validate_student.py — Student Qualitative Mask Output Visualization
================================================================================
 PURPOSE:
   Produces qualitative prediction-preview images for the deployed Student
   model, the actual model that ships in the app. This was previously the
   one component in the pipeline with no qualitative visualization at all —
   Teacher has validate_gold_standard.py, Symptom Teacher has
   validate_symptom.py, but Student had nothing beyond numeric test-metric
   CSVs (see build_student_best() in generate_report.py, which reads
   student_test_metrics_*.csv only, no images).

   For each sampled image, this produces a single panel showing:
     Original | Predicted leaf silhouette | Predicted symptom mask
   with a text header reporting the predicted class, continuous severity
   (%), and the corresponding CIMMYT agronomic grade (1-5, via
   factory_master.py's sev_to_cimmyt_grade() — the same function used
   during Factory pseudo-labeling), so a reader can see what the deployed
   model actually outputs on real images, not just its aggregate accuracy
   numbers.

 MODEL LOADED:
   The deployed Student checkpoint, resolved from STUDENT_BEST_VARIANT and
   STUDENT_FACTORY_MODE in config.py (both written automatically by
   select_best_pipeline.py — see that script for how the winner is chosen).
   Checkpoint path: checkpoints/student/stage2/student_{variant}_{mode}_best.pth

 SAMPLE SOURCE:
   The same 501-image gold-standard set used by validate_gold_standard.py
   and validate_symptom.py (GOLD_IMAGES_DIR), for consistency across all
   three qualitative validation scripts and because ground-truth category
   is already encoded in each filename ({CATEGORY}_{original_name}).

 OUTPUTS:
   reports/student_overlays/{stem}_student_pred.jpg

 USAGE:
   python validate_student.py            # 5 samples per class (default)
   python validate_student.py --n 10      # more samples per class
================================================================================
"""

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from config import (
    CLASSES,
    GOLD_IMAGES_DIR,
    REPORTS_DIR,
    SEED,
    STUDENT_BEST_VARIANT,
    STUDENT_CKPT_DIR,
    STUDENT_FACTORY_MODE,
    STUDENT_IMG_SIZE,
)
from factory_master import sev_to_cimmyt_grade
from train_student import StudentModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OVERLAY_DIR = REPORTS_DIR / "student_overlays"
N_PER_CLASS_DEFAULT = 5

CLASS_COLORS = {
    "MSV": (0, 200, 0),      # green — chlorotic streaks
    "MLN": (220, 0, 0),      # red — necrotic patches
    "HEALTHY": (0, 160, 220),  # blue, mostly unused (no symptom expected)
}

_INFER_TF = A.Compose([
    A.LongestMaxSize(max_size=STUDENT_IMG_SIZE),
    A.PadIfNeeded(STUDENT_IMG_SIZE, STUDENT_IMG_SIZE,
                  border_mode=cv2.BORDER_CONSTANT, fill=0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def load_deployed_student() -> torch.nn.Module | None:
    ckpt_name = f"{STUDENT_BEST_VARIANT}_{STUDENT_FACTORY_MODE}"
    ckpt_path = STUDENT_CKPT_DIR / "stage2" / f"student_{ckpt_name}_best.pth"
    if not ckpt_path.exists():
        print(f"[FATAL] {ckpt_path} not found.")
        print("        Run train_student.py (both stages) first, or check "
              "that STUDENT_BEST_VARIANT / STUDENT_FACTORY_MODE in config.py "
              "match a checkpoint that actually exists.")
        return None

    use_cbam = "cbam" in STUDENT_BEST_VARIANT
    model = StudentModel(STUDENT_BEST_VARIANT, use_cbam=use_cbam).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded deployed Student: {STUDENT_BEST_VARIANT} / "
          f"{STUDENT_FACTORY_MODE}  ({ckpt_path.name})")
    return model


@torch.no_grad()
def predict(model, img_rgb: np.ndarray):
    """
    Runs the Student on a single image, returns:
      sil_mask (H,W bool), sym_mask (H,W bool), pred_class (str),
      severity (float 0-100), grade (int, CIMMYT agronomic grade)
    at the ORIGINAL image resolution (resized back up from STUDENT_IMG_SIZE).

    grade is computed via sev_to_cimmyt_grade() (factory_master.py) — the
    SAME function used during Factory pseudo-labeling to compute
    grade-stratified training weights. Reusing it here, rather than a
    second bracket-lookup implementation, is what keeps the grade shown in
    this validation output consistent with the grade the training pipeline
    itself used, even if the CIMMYT_MSV_BRACKETS / CIMMYT_MLN_BRACKETS
    tables in config.py are ever revised later — there is exactly one
    place that needs updating, not two.
    """
    orig_h, orig_w = img_rgb.shape[:2]
    tensor = _INFER_TF(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    seg_logits, cls_out, sev_out = model(tensor)
    sil_prob = torch.sigmoid(seg_logits[0, 0]).cpu().numpy()
    sym_prob = torch.sigmoid(seg_logits[0, 1]).cpu().numpy()

    sil_prob = cv2.resize(sil_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    sym_prob = cv2.resize(sym_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    pred_class = CLASSES[cls_out.argmax(dim=1).item()]
    severity = float(sev_out.item()) * 100.0
    grade = sev_to_cimmyt_grade(severity, pred_class)

    return sil_prob >= 0.5, sym_prob >= 0.5, pred_class, severity, grade


def draw_prediction_panel(img_rgb, sil_mask, sym_mask, pred_class, severity,
                          grade, gt_category, stem):
    """Original | Predicted silhouette | Predicted symptom mask, with a text header."""
    h, w = img_rgb.shape[:2]
    sym_color = CLASS_COLORS.get(pred_class, (200, 200, 0))

    # Silhouette panel: green fill + contour
    sil_panel = img_rgb.copy().astype(np.float32)
    sil_panel[sil_mask] = sil_panel[sil_mask] * 0.55 + np.array([0, 200, 0]) * 0.45
    sil_panel = sil_panel.clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(sil_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sil_panel, contours, -1, (255, 255, 255), 1)

    # Symptom panel: colored by predicted class
    sym_panel = img_rgb.copy().astype(np.float32)
    sym_panel[sym_mask] = sym_panel[sym_mask] * 0.5 + np.array(sym_color) * 0.5
    sym_panel = sym_panel.clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(sym_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sym_panel, contours, -1, sym_color, 1)

    orig_panel = img_rgb.copy()

    for panel, label in [(orig_panel, "Original"),
                         (sil_panel, "Predicted Silhouette"),
                         (sym_panel, "Predicted Symptom Mask")]:
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    combined = np.hstack([orig_panel, sil_panel, sym_panel]).copy()
    correct = pred_class == gt_category
    status = "match" if correct else "mismatch"
    footer = (f"GT: {gt_category}  |  Predicted: {pred_class} ({status})  |  "
             f"Severity: {severity:.1f}%  (CIMMYT Grade {grade})  |  {stem}")
    cv2.putText(combined, footer, (8, combined.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(combined, footer, (8, combined.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PER_CLASS_DEFAULT,
                        help="Samples per class (default 5)")
    args = parser.parse_args()

    random.seed(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Student Qualitative Mask Output Validation")
    print("=" * 72)

    model = load_deployed_student()
    if model is None:
        return

    if not GOLD_IMAGES_DIR.exists():
        print(f"[FATAL] {GOLD_IMAGES_DIR} not found.")
        return

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    n_correct, n_total = 0, 0

    for cls in CLASSES:
        pool = sorted(GOLD_IMAGES_DIR.glob(f"{cls}_*"))
        if not pool:
            print(f"  [WARN] No gold-standard images found for {cls}. Skipping.")
            continue
        sample = random.sample(pool, min(args.n, len(pool)))
        print(f"\n  [{cls}] generating {len(sample)} prediction panels ...")

        for img_path in sample:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            sil_mask, sym_mask, pred_class, severity, grade = predict(model, img_rgb)
            panel = draw_prediction_panel(
                img_rgb, sil_mask, sym_mask, pred_class, severity, grade,
                gt_category=cls, stem=img_path.stem)

            out_path = OVERLAY_DIR / f"{img_path.stem}_student_pred.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            n_total += 1
            n_correct += int(pred_class == cls)

    if n_total:
        print(f"\n{'─' * 72}")
        print(f"  Qualitative sample accuracy: {n_correct}/{n_total} "
              f"({100 * n_correct / n_total:.1f}%)")
        print(f"  NOTE: this is a small illustrative sample, not the formal "
              f"test-set accuracy — see student_test_metrics_*.csv for that.")
    print(f"  Overlay panels saved to: {OVERLAY_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
