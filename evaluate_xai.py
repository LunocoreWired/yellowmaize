"""
================================================================================
 evaluate_xai.py — Phase 6: XAI Three-Way Comparison
================================================================================
 PURPOSE:
   Apply and compare three XAI methods on the best Student model:
     - Grad-CAM        (Selvaraju et al. 2017) — historical baseline
     - Grad-CAM++      (Chattopadhay et al. 2018) — deployed method
     - Score-CAM       (Wang et al. 2020) — gradient-free comparison

   All three are applied to the last convolutional block of the shared
   encoder — reflecting shared disease-relevant features driving both
   classification and segmentation simultaneously.

   Target layer is defined per encoder variant in config.py (XAI_TARGET_LAYERS).

 TWO DISTINCT OUTPUTS (shown separately in the Android app):
   1. UNet segmentation boundary (crisp green contour — pixel-level localization)
   2. Grad-CAM++ heatmap (amber overlay — diagnostic explanation / attention)

 OUTPUTS:
   reports/xai/          — per-image overlays (3 methods × 5 images per class)
   logs/xai_comparison.csv — quantitative comparison (pointing game accuracy,
                              insertion AUC, deletion AUC per method per class)
================================================================================
"""

import csv
import time as _time
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from image_utils import load_image_rgb
import pandas as pd

from config import (
    SEED, LOGS_DIR, REPORTS_DIR,
    GLOBAL_MANIFEST, PSEUDO_DIR,
    STUDENT_CKPT_DIR,
    STUDENT_IMG_SIZE, STUDENT_BATCH_SIZE, STUDENT_NUM_WORKERS,
    CLASSES, CLASS_TO_IDX,
    XAI_METHODS, XAI_DEPLOYED_METHOD, XAI_TARGET_LAYERS,
    XAI_N_SAMPLES_CLASS, XAI_INSERTION_STEPS,
    STUDENT_BEST_VARIANT, STUDENT_FACTORY_MODE,
    FACTORY_MODES,
)
from train_student import StudentModel, StudentDataset, make_student_transforms, build_sample_list


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XAI_OUTPUT_DIR   = REPORTS_DIR / "xai"
N_SAMPLES_CLASS  = XAI_N_SAMPLES_CLASS   # from config — default 30 per class
OVERLAY_ALPHA    = 0.50  # heatmap overlay transparency


# ══════════════════════════════════════════════════════════════════════════════
# TARGET LAYER RESOLVER
# ══════════════════════════════════════════════════════════════════════════════

def get_target_layer(model: nn.Module, encoder_variant: str) -> nn.Module:
    """
    Resolve the target layer for XAI methods from the encoder variant.
    Returns the actual nn.Module corresponding to the last conv block.
    """
    layer_str = XAI_TARGET_LAYERS.get(encoder_variant)
    if layer_str is None:
        raise ValueError(f"No XAI target layer defined for: {encoder_variant}")

    # Resolve the layer string to an actual module
    # Pattern: "encoder.features[-1][0]" or "encoder.blocks[-1][-1]" etc.
    obj = model
    for part in layer_str.split("."):
        if "[" in part:
            # Handle index access e.g. features[-1] or blocks[-1][-1]
            attr = part[:part.index("[")]
            obj  = getattr(obj, attr)
            # Extract all indices
            indices = part[part.index("["):]
            import re
            for idx_str in re.findall(r'\[(-?\d+)\]', indices):
                obj = obj[int(idx_str)]
        else:
            obj = getattr(obj, part)

    return obj


# ══════════════════════════════════════════════════════════════════════════════
# XAI WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

class GradCAMWrapper:
    """
    Grad-CAM (Selvaraju et al. 2017).
    Weighted sum of gradients × feature maps, ReLU applied.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 method: str = "gradcam"):
        try:
            from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            cam_cls = {
                "gradcam":        GradCAM,
                "gradcamplusplus":GradCAMPlusPlus,
                "scorecam":       ScoreCAM,
            }[method]

            self.cam    = cam_cls(model=model, target_layers=[target_layer])
            self.method = method
            self.ClassifierOutputTarget = ClassifierOutputTarget
            self.available = True

        except ImportError:
            print(f"  [WARN] pytorch-grad-cam not installed. "
                  f"Install: pip install grad-cam")
            self.available = False

    def __call__(self, input_tensor: torch.Tensor,
                 target_class: int) -> np.ndarray | None:
        """
        Returns a float32 heatmap in [0,1] of shape (H, W).
        """
        if not self.available:
            return None
        targets = [self.ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
        )
        return grayscale_cam[0]   # (H, W) float32


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def heatmap_overlay(img_rgb: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """Overlay a [0,1] heatmap as amber colormap on an RGB image."""
    h, w     = img_rgb.shape[:2]
    heatmap  = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap8 = (heatmap * 255).astype(np.uint8)
    colored  = cv2.applyColorMap(heatmap8, cv2.COLORMAP_JET)
    colored  = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended  = (img_rgb * (1 - alpha) + colored * alpha).astype(np.uint8)
    return blended


def seg_boundary_overlay(img_rgb: np.ndarray,
                          seg_logits: torch.Tensor,
                          channel: int = 0) -> np.ndarray:
    """
    Draw crisp green contour from UNet segmentation head.
    This is the pixel-level localization output (separate from Grad-CAM).
    """
    h, w    = img_rgb.shape[:2]
    prob    = torch.sigmoid(seg_logits[0, channel]).cpu().numpy()
    prob    = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    binary  = (prob >= 0.5).astype(np.uint8)

    overlay   = img_rgb.copy()
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 0), 2)
    return overlay


def side_by_side(img_rgb: np.ndarray,
                 seg_overlay: np.ndarray,
                 xai_overlay: np.ndarray,
                 method: str,
                 category: str,
                 pred_class: str) -> np.ndarray:
    """
    Combine original + segmentation boundary + XAI heatmap side by side.
    Adds text labels for the app distinction.
    """
    h, w = img_rgb.shape[:2]
    panel = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    panel[:, :w]             = img_rgb
    panel[:, w+10:2*w+10]   = seg_overlay
    panel[:, 2*w+20:]        = xai_overlay

    # Labels
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    col   = (255, 255, 255)
    cv2.putText(panel, "Original",              (8,    18), font, scale, col, 1)
    cv2.putText(panel, "Symptom boundary",      (w+18, 18), font, scale, (0, 255, 0), 1)
    cv2.putText(panel, f"Diagnostic attn ({method})",
                (2*w+28, 18), font, scale, (255, 200, 0), 1)
    cv2.putText(panel, f"{category} → pred:{pred_class}",
                (8, h - 8), font, scale, (200, 200, 200), 1)
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def pointing_game_accuracy(heatmap: np.ndarray,
                            seg_mask: np.ndarray,
                            threshold: float = 0.80) -> float:
    """
    Pointing game: fraction of top-activation pixels that fall within
    the segmentation mask (ground truth region of interest).
    Threshold: pixels above (max * threshold) are considered "pointing."
    """
    if heatmap is None or seg_mask is None:
        return float("nan")
    h, w    = seg_mask.shape
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    cutoff  = heatmap.max() * threshold
    top_pts = heatmap >= cutoff
    if top_pts.sum() == 0:
        return 0.0
    hits = (top_pts & (seg_mask > 0)).sum()
    return float(hits) / float(top_pts.sum())


def insertion_deletion_auc(model: nn.Module,
                            input_tensor: torch.Tensor,
                            heatmap: np.ndarray,
                            target_class: int,
                            n_steps: int = 6) -> tuple[float, float]:
    # n_steps reduced to 6 (was 10) for speed; still gives meaningful AUC
    """
    Simplified insertion/deletion AUC.
    Insertion: progressively reveal pixels by importance → score should rise.
    Deletion:  progressively remove pixels by importance → score should drop.
    Returns (insertion_auc, deletion_auc).
    """
    if heatmap is None:
        return float("nan"), float("nan")

    model.eval()
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    hm   = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    flat_order = np.argsort(hm.ravel())[::-1]   # most important first

    total_px = h * w
    img_flat = input_tensor[0].cpu().numpy().reshape(3, -1)

    ins_scores = []
    del_scores = []

    baseline = np.zeros_like(img_flat)   # black baseline for insertion

    for step in range(n_steps + 1):
        n_reveal = int((step / n_steps) * total_px)
        n_mask   = n_reveal

        # Insertion: reveal top n_reveal pixels from black baseline
        ins_img  = baseline.copy()
        if n_reveal > 0:
            ins_img[:, flat_order[:n_reveal]] = img_flat[:, flat_order[:n_reveal]]
        ins_t = torch.tensor(
            ins_img.reshape(3, h, w), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        # Deletion: mask out top n_mask pixels from original
        del_img = img_flat.copy()
        if n_mask > 0:
            del_img[:, flat_order[:n_mask]] = 0
        del_t = torch.tensor(
            del_img.reshape(3, h, w), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            ins_t_d = ins_t.to(DEVICE)
            del_t_d = del_t.to(DEVICE)
            _, cls_ins, _ = model(ins_t_d)
            _, cls_del, _ = model(del_t_d)
            p_ins = torch.softmax(cls_ins, dim=1)[0, target_class].item()
            p_del = torch.softmax(cls_del, dim=1)[0, target_class].item()

        ins_scores.append(p_ins)
        del_scores.append(p_del)

    ins_auc = float(np.trapz(ins_scores)) / n_steps
    del_auc = float(np.trapz(del_scores)) / n_steps
    return ins_auc, del_auc


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _t_start = _time.time()
    set_seeds(SEED)

    print("=" * 72)
    print("  Yellow MAIze | Phase 6: XAI Three-Way Comparison")
    print("=" * 72)

    XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Find best student checkpoint ──────────────────────────────────────────
    # Look for best mode result in logs
    best_ckpt = None
    best_variant = STUDENT_BEST_VARIANT

    # Try to find the best mode checkpoint from Stage 2 comparison
    stage2_comp = LOGS_DIR / "student_comparison_stage2.csv"
    if stage2_comp.exists():
        df   = pd.read_csv(stage2_comp)
        best_row  = df.loc[df["best_composite"].idxmax()]
        best_mode = best_row["mode"]
        best_variant = best_row["encoder"]
        ckpt_path = STUDENT_CKPT_DIR / f"student_{best_variant}_{best_mode}_best.pth"
    else:
        # Fall back to mode_b with best variant
        best_mode = "mode_b"
        ckpt_path = STUDENT_CKPT_DIR / \
            f"student_{best_variant}_{best_mode}_best.pth"

    if not ckpt_path.exists():
        print(f"[FATAL] Student checkpoint not found: {ckpt_path}")
        print("        Run train_student.py --stage 1 and --stage 2 first.")
        return

    print(f"  Encoder  : {best_variant}")
    print(f"  Mode     : {best_mode}")
    print(f"  Checkpoint: {ckpt_path}")

    # ── Load model ─────────────────────────────────────────────────────────────
    use_cbam = "cbam" in best_variant
    model    = StudentModel(best_variant, use_cbam=use_cbam).to(DEVICE)
    ckpt     = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Resolve target layer ──────────────────────────────────────────────────
    try:
        target_layer = get_target_layer(model, best_variant)
        print(f"  XAI target layer: {XAI_TARGET_LAYERS[best_variant]}")
    except Exception as e:
        print(f"[FATAL] Could not resolve target layer: {e}")
        return

    # ── Build XAI wrappers ────────────────────────────────────────────────────
    cam_wrappers = {}
    for method in XAI_METHODS:
        cam_wrappers[method] = GradCAMWrapper(model, target_layer, method)

    # ── Build test dataset ────────────────────────────────────────────────────
    test_samples = build_sample_list("test", best_mode)
    if not test_samples:
        print("[FATAL] No test samples found.")
        return

    val_tf  = make_student_transforms(STUDENT_IMG_SIZE, is_train=False)
    test_ds = StudentDataset(test_samples, best_mode, val_tf)

    # ── Sample images per class for qualitative overlays ──────────────────────
    by_class = {cls: [] for cls in CLASSES}
    for i, s in enumerate(test_samples):
        by_class[s["category"]].append(i)

    # Use a local RNG instance so sample is reproducible regardless of
    # what torch/numpy operations ran before this point (global RNG state)
    _rng = random.Random(SEED)
    selected_indices = []
    for cls in CLASSES:
        idxs = _rng.sample(
            by_class[cls], min(N_SAMPLES_CLASS, len(by_class[cls])))
        selected_indices.extend(idxs)

    # ── XAI evaluation loop ───────────────────────────────────────────────────
    quant_rows = []

    for sample_idx in selected_indices:
        sample   = test_samples[sample_idx]
        category = sample["category"]
        stem     = sample["stem"]
        true_cls = CLASS_TO_IDX[category]

        # Load raw image for visualization
        img_rgb  = load_image_rgb(sample["source_path"]) or \
            np.array(Image.open(sample["source_path"]).convert("RGB"))

        # Get dataset item (normalized tensor)
        item     = test_ds[sample_idx]
        inp_t    = item["image"].unsqueeze(0).to(DEVICE)
        seg_tgt  = item["seg"]

        # Model forward
        with torch.no_grad():
            seg_logits, cls_out, sev_out = model(inp_t)
        pred_cls  = cls_out.argmax(dim=1).item()
        pred_name = CLASSES[pred_cls]
        pred_sev  = round(sev_out.item() * 100, 1)

        # CIMMYT grade from predicted severity
        def _cimmyt_grade(sev_pct: float, cat: str) -> str:
            if cat == "HEALTHY": return "0 (none)"
            if cat == "MSV":
                for lo, hi, g in [(0,5,1),(5,25,3),(25,50,5),(50,75,7),(75,101,9)]:
                    if lo <= sev_pct < hi: return str(g)
            for lo, hi, g in [(0,10,1),(10,25,2),(25,50,3),(50,75,4),(75,101,5)]:
                if lo <= sev_pct < hi: return str(g)
            return "?"
        pred_grade = _cimmyt_grade(pred_sev, pred_name)

        # Leaf-mask-constrained GradCAM++ severity
        # Uses Ch0 sigmoid as leaf mask to restrict activation to leaf region.
        # More robust than pixel ratio — tied to what the model actually learned.
        leaf_prob  = torch.sigmoid(seg_logits[0, 0]).cpu().numpy()  # Ch0
        leaf_mask  = (leaf_prob >= 0.5)

        # Segmentation boundary overlay (same for all XAI methods)
        seg_ov = seg_boundary_overlay(img_rgb, seg_logits, channel=0)

        # Per-method XAI
        for method, cam in cam_wrappers.items():
            heatmap = cam(inp_t, true_cls)

            # Leaf-constrained GradCAM++ severity (within leaf mask only)
            if heatmap is not None and leaf_mask.sum() > 0:
                heatmap_np = heatmap[0] if heatmap.ndim == 3 else heatmap
                # Resize heatmap to match leaf_mask if needed
                if heatmap_np.shape != leaf_mask.shape:
                    import cv2 as _cv2
                    heatmap_np = _cv2.resize(heatmap_np,
                        (leaf_mask.shape[1], leaf_mask.shape[0]),
                        interpolation=_cv2.INTER_LINEAR)
                gradcam_sev = round(
                    float(heatmap_np[leaf_mask].mean()) * 100, 1)
            else:
                gradcam_sev = -1.0

            # Qualitative overlay
            if heatmap is not None:
                xai_ov = heatmap_overlay(img_rgb, heatmap)
                panel  = side_by_side(
                    img_rgb, seg_ov, xai_ov, method, category, pred_name)
                method_dir = XAI_OUTPUT_DIR / method
                method_dir.mkdir(parents=True, exist_ok=True)
                out_path = method_dir / f"{stem}_{method}.jpg"
                cv2.imwrite(str(out_path),
                            cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            # Quantitative metrics
            seg_mask_np = (seg_tgt[0].numpy() >= 0.5).astype(np.uint8)
            pg_acc = pointing_game_accuracy(heatmap, seg_mask_np)

            ins_auc, del_auc = insertion_deletion_auc(
                model, inp_t, heatmap, true_cls, n_steps=XAI_INSERTION_STEPS)

            quant_rows.append({
                "stem":              stem,
                "category":          category,
                "pred_cls":          pred_name,
                "correct":           category == pred_name,
                "pred_sev":          pred_sev,
                "method":            method,
                "pointing_game_acc": round(pg_acc, 4) if not np.isnan(pg_acc) else "nan",
                "insertion_auc":     round(ins_auc, 4) if not np.isnan(ins_auc) else "nan",
                "deletion_auc":      round(del_auc, 4) if not np.isnan(del_auc) else "nan",
            })

        print(f"  [{category}] {stem[:40]:<40} "
              f"pred={pred_name} sev={pred_sev:.1f}%")

    # ── Write quantitative comparison CSV ────────────────────────────────────
    if quant_rows:
        comp_path = LOGS_DIR / "xai_comparison.csv"
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=quant_rows[0].keys())
            writer.writeheader()
            writer.writerows(quant_rows)

        # ── Per-method overall summary ────────────────────────────────────
        print(f"\n{'─' * 72}")
        print("  XAI quantitative summary (mean across all samples):")
        print(f"  {'Method':<20} {'Pointing Game':>14} {'Ins AUC':>8} {'Del AUC':>8}")
        print(f"  {'─'*20} {'─'*14} {'─'*8} {'─'*8}")

        df = pd.DataFrame(quant_rows)
        method_scores = {}
        for method in XAI_METHODS:
            mdf = df[df["method"] == method]
            pg  = pd.to_numeric(mdf["pointing_game_acc"], errors="coerce")
            ins = pd.to_numeric(mdf["insertion_auc"],     errors="coerce")
            dl  = pd.to_numeric(mdf["deletion_auc"],      errors="coerce")
            print(f"  {method:<20} {pg.mean():>14.4f} "
                  f"{ins.mean():>8.4f} {dl.mean():>8.4f}")
            method_scores[method] = {
                "pg_mean":  pg.mean(),
                "ins_mean": ins.mean(),
                "del_mean": dl.mean(),
                # MSV-specific pointing game (thesis primary focus)
                "msv_pg": pd.to_numeric(
                    mdf[mdf["category"] == "MSV"]["pointing_game_acc"],
                    errors="coerce").mean(),
            }

        # ── Per-class breakdown for each method ──────────────────────────────
        print(f"\n  Per-class breakdown:")
        for method in XAI_METHODS:
            mdf = df[df["method"] == method]
            print(f"  [{method}]")
            print(f"    {'Class':<12} {'Pointing Game':>14} {'Ins AUC':>8} {'Del AUC':>8} {'Correct%':>9}")
            print(f"    {'─'*12} {'─'*14} {'─'*8} {'─'*8} {'─'*9}")
            for cls in CLASSES:
                cdf = mdf[mdf["category"] == cls]
                if cdf.empty:
                    continue
                pg  = pd.to_numeric(cdf["pointing_game_acc"], errors="coerce").mean()
                ins = pd.to_numeric(cdf["insertion_auc"],     errors="coerce").mean()
                dl  = pd.to_numeric(cdf["deletion_auc"],      errors="coerce").mean()
                acc = cdf["correct"].mean() * 100 if "correct" in cdf.columns else float("nan")
                print(f"    {cls:<12} {pg:>14.4f} {ins:>8.4f} {dl:>8.4f} {acc:>8.1f}%")

        # ── Auto-select best XAI method ───────────────────────────────────────
        # Selection criterion: highest MSV pointing game accuracy (primary thesis focus).
        # Tiebreaker: insertion AUC (measures faithfulness of top activations).
        valid_methods = {m: s for m, s in method_scores.items()
                         if not np.isnan(s["msv_pg"])}
        if valid_methods:
            best_method = max(
                valid_methods,
                key=lambda m: (valid_methods[m]["msv_pg"],
                               valid_methods[m]["ins_mean"])
            )
            best_pg  = valid_methods[best_method]["msv_pg"]
            best_ins = valid_methods[best_method]["ins_mean"]

            print(f"\n{'─' * 72}")
            print(f"  XAI AUTO-SELECTION:")
            print(f"    Best method : {best_method}")
            print(f"    MSV pointing game : {best_pg:.4f}")
            print(f"    Insertion AUC     : {best_ins:.4f}")
            print(f"    Criterion: highest MSV pointing game accuracy")
            print(f"    Tiebreaker: insertion AUC")

            # Write best method to selection log
            selection_path = LOGS_DIR / "xai_method_selection.csv"
            import csv as _csv
            with open(selection_path, "w", newline="", encoding="utf-8") as f_sel:
                _w = _csv.DictWriter(f_sel, fieldnames=[
                    "selected_method","msv_pg","ins_auc","del_auc",
                    "gradcam_pg","gradcamplusplus_pg","scorecam_pg"])
                _w.writeheader()
                _w.writerow({
                    "selected_method":   best_method,
                    "msv_pg":            round(best_pg, 4),
                    "ins_auc":           round(best_ins, 4),
                    "del_auc":           round(valid_methods[best_method]["del_mean"], 4),
                    "gradcam_pg":        round(valid_methods.get("gradcam",        {}).get("msv_pg", float("nan")), 4),
                    "gradcamplusplus_pg":round(valid_methods.get("gradcamplusplus",{}).get("msv_pg", float("nan")), 4),
                    "scorecam_pg":       round(valid_methods.get("scorecam",       {}).get("msv_pg", float("nan")), 4),
                })
            print(f"    Selection log: {selection_path}")

            # Update config.py XAI_DEPLOYED_METHOD in-place
            config_path = Path(__file__).parent / "config.py"
            if config_path.exists():
                cfg_text = config_path.read_text(encoding="utf-8")
                import re as _re
                new_cfg = _re.sub(
                    r'XAI_DEPLOYED_METHOD\s*=\s*"[^"]*"',
                    f'XAI_DEPLOYED_METHOD = "{best_method}"',
                    cfg_text)
                if new_cfg != cfg_text:
                    config_path.write_text(new_cfg, encoding="utf-8")
                    print(f"    config.py updated: XAI_DEPLOYED_METHOD = "{best_method}"")
                else:
                    print(f"    config.py unchanged (already {best_method})")
        else:
            best_method = XAI_DEPLOYED_METHOD
            print(f"  [WARN] Could not auto-select XAI method — using config default: {best_method}")

        print(f"\n  Comparison saved : {comp_path}")
        print(f"  Overlays saved   : {XAI_OUTPUT_DIR}")

    print(f"\n  Note: In the Android app, two distinct outputs are shown:")
    print(f"    (1) Green contour overlay  — UNet segmentation head")
    print(f"        'Symptom boundary' — pixel-level localization")
    print(f"    (2) Amber heatmap overlay  — {XAI_DEPLOYED_METHOD} on encoder")
    print(f"        'Diagnostic attention' — explainability / justification")
    print(f"    Deployed XAI method: {XAI_DEPLOYED_METHOD} (auto-selected or config default)")

    print(f"\n  NEXT STEP: python evaluate_severity.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
