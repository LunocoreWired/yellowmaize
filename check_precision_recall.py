"""
check_precision_recall.py
──────────────────────────────────────────────────────────────────────────
Loads the trained Symptom Teacher checkpoint and reports precision, recall,
and false-positive rate separately for MSV and MLN, at the same 0.5
threshold used during training's IoU computation.

WHY: Focal Tversky Loss was configured with beta=0.7 for MSV (penalizes
false negatives harder than false positives). This deliberately trades
toward recall. IoU alone can't tell you whether that gain came from the
model actually resolving thin streak structure better, or just predicting
larger/looser blobs around real streaks (high recall, mediocre precision).
This script separates those two explanations.

CORRECTION: SEED=42 is set at import (config.py), applied via random.seed()
before any shuffling happens. Since --skip-ae skips the only OTHER
random.shuffle() call in the script (HealthyAE's data loading), the
random.shuffle(records) that builds the train/val split here is the first
random call each run — making it fully deterministic given the same input
JSON. This script's split should match the split used by every prior
--skip-ae training run, including the checkpoint being loaded here.

Usage:
    python check_precision_recall.py
"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_symptom_model_experiment import (
    DEVICE, GOLD_IMAGES_DIR, SYMPTOM_ANNOTATION_FILE, SYMPTOM_BATCH_SIZE,
    SYMPTOM_CKPT_DIR, SYMPTOM_EXTRA_IMAGES_DIR, SYMPTOM_IMG_SIZE,
    SYMPTOM_VAL_SPLIT, HealthyAE, HEALTHY_AE_CKPT_DIR, SymptomDataset,
    SymptomTeacher, _symptom_collate, parse_symptom_annotations,
)


def precision_recall_fprate(pred_binary: np.ndarray, gt_binary: np.ndarray):
    tp = int(np.logical_and(pred_binary, gt_binary).sum())
    fp = int(np.logical_and(pred_binary, np.logical_not(gt_binary)).sum())
    fn = int(np.logical_and(np.logical_not(pred_binary), gt_binary).sum())
    tn = int(np.logical_and(np.logical_not(pred_binary), np.logical_not(gt_binary)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    fp_rate   = fp / max(fp + tn, 1)   # fraction of true-background pixels wrongly flagged
    return precision, recall, fp_rate


def main():
    print("=" * 72)
    print("  Precision / Recall / FP-rate check — Symptom Teacher")
    print("=" * 72)
    print("  NOTE: val split is freshly reconstructed (no fixed seed in the")
    print("  original training script) — representative, not an exact audit")
    print("  of the checkpoint's reported val_IoU run.\n")

    # ── Load HealthyAE (needed to build the 4th input channel) ─────────────
    ae_ckpt_path = HEALTHY_AE_CKPT_DIR / "healthy_ae_best.pth"
    if not ae_ckpt_path.exists():
        print(f"[FATAL] HealthyAE checkpoint not found at {ae_ckpt_path}")
        return
    ae_model = HealthyAE().to(DEVICE)
    ae_ckpt = torch.load(ae_ckpt_path, map_location=DEVICE)
    ae_model.load_state_dict(ae_ckpt["model_state"])
    ae_model.eval()

    # ── Load Symptom Teacher checkpoint ─────────────────────────────────────
    teacher_ckpt_path = SYMPTOM_CKPT_DIR / "symptom_teacher_best.pth"
    if not teacher_ckpt_path.exists():
        print(f"[FATAL] Symptom Teacher checkpoint not found at {teacher_ckpt_path}")
        return
    ckpt = torch.load(teacher_ckpt_path, map_location=DEVICE)
    model = SymptomTeacher(encoder_name=ckpt.get("encoder"))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded checkpoint (reported val_IoU: {ckpt.get('val_iou', '?')})\n")

    # ── Rebuild an (approximate) val split ──────────────────────────────────
    records = parse_symptom_annotations(
        SYMPTOM_ANNOTATION_FILE, [GOLD_IMAGES_DIR, SYMPTOM_EXTRA_IMAGES_DIR])
    random.shuffle(records)
    n_val = max(1, int(len(records) * SYMPTOM_VAL_SPLIT))
    val_records = records[:n_val]

    val_ds = SymptomDataset(val_records, ae_model, SYMPTOM_IMG_SIZE, augment=False)
    val_loader = DataLoader(val_ds, batch_size=SYMPTOM_BATCH_SIZE,
                            shuffle=False, num_workers=0, collate_fn=_symptom_collate)

    msv_p, msv_r, msv_fp = [], [], []
    mln_p, mln_r, mln_fp = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch is None:
                continue
            x, y = batch
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            gt = y.numpy()
            batch_start = batch_idx * SYMPTOM_BATCH_SIZE
            batch_recs = val_records[batch_start: batch_start + len(probs)]
            for p, g, rec in zip(probs, gt, batch_recs):
                cat = rec.get("category", "MSV")
                if cat == "MSV":
                    pr, rc, fpr = precision_recall_fprate(
                        (p[0] >= 0.5).astype(np.uint8), (g[0] >= 0.5).astype(np.uint8))
                    msv_p.append(pr); msv_r.append(rc); msv_fp.append(fpr)
                elif cat == "MLN":
                    pr, rc, fpr = precision_recall_fprate(
                        (p[1] >= 0.5).astype(np.uint8), (g[1] >= 0.5).astype(np.uint8))
                    mln_p.append(pr); mln_r.append(rc); mln_fp.append(fpr)

    def summarize(name, p_list, r_list, fp_list):
        if not p_list:
            print(f"  {name}: no validation images in this split")
            return
        print(f"  {name} (n={len(p_list)}):")
        print(f"    precision = {np.mean(p_list):.3f}")
        print(f"    recall    = {np.mean(r_list):.3f}")
        print(f"    FP rate   = {np.mean(fp_list):.4f}  "
              f"(fraction of true-background pixels wrongly flagged as {name})")

    print("─" * 72)
    summarize("MSV", msv_p, msv_r, msv_fp)
    print()
    summarize("MLN", mln_p, mln_r, mln_fp)
    print("─" * 72)
    print("\n  READ: if recall >> precision for MSV, the model is over-predicting")
    print("  (loose blobs around real streaks) rather than precisely resolving")
    print("  thin structure. If precision and recall are closer together, the")
    print("  IoU gain reflects genuinely tighter segmentation, not just recall bias.")


if __name__ == "__main__":
    main()
