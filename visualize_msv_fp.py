"""
visualize_msv_fp.py
──────────────────────────────────────────────────────────────────────────
Diagnoses WHERE MSV false positives are happening: a thin halo/boundary
fuzziness right around real streaks (expected, low-concern — even human
annotators disagree on exact streak edges), vs scattered false positives
elsewhere on the leaf (a real problem — genuine confusion with healthy
tissue or veins, not just boundary imprecision).

Outputs one comparison image per worst-IoU MSV validation sample:
  Original | Ground truth (white) | Prediction (white) | Overlay
  Overlay legend: green=ground truth only (missed), cyan=correct (TP),
                  red=prediction only (false positive)

Usage:
    python visualize_msv_fp.py [--n 10]
"""
import argparse
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_symptom_model_experiment import (
    DEVICE, GOLD_IMAGES_DIR, REPORTS_DIR, SYMPTOM_ANNOTATION_FILE,
    SYMPTOM_BATCH_SIZE, SYMPTOM_CKPT_DIR, SYMPTOM_EXTRA_IMAGES_DIR,
    SYMPTOM_IMG_SIZE, SYMPTOM_VAL_SPLIT, HealthyAE, HEALTHY_AE_CKPT_DIR,
    SymptomDataset, SymptomTeacher, _symptom_collate, compute_iou,
    parse_symptom_annotations,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="Number of worst-IoU MSV samples to visualize")
    args = parser.parse_args()

    print("=" * 72)
    print("  MSV False-Positive Spatial Pattern Check")
    print("=" * 72)

    ae_ckpt_path = HEALTHY_AE_CKPT_DIR / "healthy_ae_best.pth"
    ae_model = HealthyAE().to(DEVICE)
    ae_ckpt = torch.load(ae_ckpt_path, map_location=DEVICE)
    ae_model.load_state_dict(ae_ckpt["model_state"])
    ae_model.eval()

    teacher_ckpt_path = SYMPTOM_CKPT_DIR / "symptom_teacher_best.pth"
    ckpt = torch.load(teacher_ckpt_path, map_location=DEVICE)
    model = SymptomTeacher(encoder_name=ckpt.get("encoder"))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded checkpoint (reported val_IoU: {ckpt.get('val_iou', '?')})\n")

    records = parse_symptom_annotations(
        SYMPTOM_ANNOTATION_FILE, [GOLD_IMAGES_DIR, SYMPTOM_EXTRA_IMAGES_DIR])
    random.shuffle(records)   # deterministic given SEED=42, matches training split
    n_val = max(1, int(len(records) * SYMPTOM_VAL_SPLIT))
    val_records = records[:n_val]

    val_ds = SymptomDataset(val_records, ae_model, SYMPTOM_IMG_SIZE, augment=False)
    val_loader = DataLoader(val_ds, batch_size=SYMPTOM_BATCH_SIZE,
                            shuffle=False, num_workers=0, collate_fn=_symptom_collate)

    results = []  # (iou, orig_img_uint8, gt_binary, pred_binary, stem)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch is None:
                continue
            x, y = batch
            x_dev = x.to(DEVICE)
            logits = model(x_dev)
            probs = torch.sigmoid(logits).cpu().numpy()
            gt = y.numpy()
            batch_start = batch_idx * SYMPTOM_BATCH_SIZE
            batch_recs = val_records[batch_start: batch_start + len(probs)]
            for i, (p, g, rec) in enumerate(zip(probs, gt, batch_recs)):
                if rec.get("category") != "MSV":
                    continue
                pred_bin = (p[0] >= 0.5).astype(np.uint8)
                gt_bin = (g[0] >= 0.5).astype(np.uint8)
                iou = compute_iou(pred_bin, gt_bin)
                # Recover the (padded/resized) RGB image for visualization —
                # first 3 channels of the model input, denormalized
                img_chw = x[i].numpy()
                img_rgb = (img_chw[:3].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                stem = rec["img_path"].stem
                results.append((iou, img_rgb, gt_bin, pred_bin, stem))

    results.sort(key=lambda r: r[0])  # worst IoU first
    n = min(args.n, len(results))
    print(f"  {len(results)} MSV validation samples found. Visualizing {n} worst-IoU cases.\n")

    out_dir = REPORTS_DIR / "msv_fp_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, (iou, img_rgb, gt_bin, pred_bin, stem) in enumerate(results[:n], start=1):
        h, w = gt_bin.shape
        overlay = img_rgb.copy()
        tp = np.logical_and(pred_bin, gt_bin)
        fn = np.logical_and(np.logical_not(pred_bin), gt_bin)
        fp = np.logical_and(pred_bin, np.logical_not(gt_bin))
        overlay[tp] = [0, 255, 255]     # cyan = correct
        overlay[fn] = [0, 255, 0]       # green = missed (ground truth only)
        overlay[fp] = [255, 0, 0]       # red   = false positive (prediction only)

        gt_vis   = cv2.cvtColor(gt_bin * 255, cv2.COLOR_GRAY2RGB)
        pred_vis = cv2.cvtColor(pred_bin * 255, cv2.COLOR_GRAY2RGB)
        panel = np.concatenate([img_rgb, gt_vis, pred_vis, overlay], axis=1)

        out_path = out_dir / f"{rank:02d}_{stem}_iou{iou:.3f}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"  [{rank}] {stem}  IoU={iou:.3f}  -> {out_path}")

    print(f"\n  Saved {n} diagnostic panels to: {out_dir}")
    print("  Legend: green=missed (FN) | cyan=correct (TP) | red=false positive (FP)")
    print("\n  READ: if red pixels form a thin halo hugging real streak edges,")
    print("  that's boundary fuzziness (low concern — even annotators disagree here).")
    print("  If red pixels are scattered away from any real streak (on clean leaf")
    print("  tissue, veins, etc.), that's genuine confusion — a real problem worth")
    print("  addressing (e.g. boundary-aware loss, or annotation review).")


if __name__ == "__main__":
    main()
