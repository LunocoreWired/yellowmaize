"""
factory_master.py — Phase 4: Pseudo-Label Generation (Optimized & Fixed)
Hardware Target: Ryzen 5 3600X (12T) / RTX 5060 (8GB) / 16GB RAM / WSL2
"""
import os
# CRITICAL: Prevent OpenCV/NumPy from hijacking all CPU cores and choking DataLoader workers
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import csv
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchvision import models

cv2.setNumThreads(2)  # Force OpenCV to respect thread limits

from image_utils import load_image_clahe, to_hsv
from config import (
    SEED, GLOBAL_MANIFEST, TIER1_MANIFEST, TIER1_MASKS_DIR, PSEUDO_DIR, REPORTS_DIR,
    TEACHER_CKPT_DIR, BOUNCER_CKPT_DIR, BOUNCER_DEPLOYED_VARIANT,
    TEACHER_DEPLOYED_VARIANT, TEACHER_IMG_SIZE, STUDENT_IMG_SIZE,
    FACTORY_MODES, FACTORY_SILHOUETTE_THRESHOLD, FACTORY_MIN_LEAF_COVERAGE,
    FACTORY_WEIGHT_BRACKETS, FACTORY_R3_MIN_AREA_PX, FACTORY_MORPH_KERNEL_SIZE, HSV_GREEN_EXCL,
    HSV_MSV_RANGES, HSV_MLN_RANGES, BOUNCER_THRESHOLD, VALID_EXTENSIONS, CLASSES,
    GABOR_KERNEL_SIZE, GABOR_SIGMA, GABOR_GAMMA, GABOR_PSI, GABOR_NORMS,
    GABOR_THETAS, GABOR_THRESHOLD, BOUNCER_IMG_SIZE
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
TEACHER_INFER_TF = A.Compose([
    A.LongestMaxSize(max_size=TEACHER_IMG_SIZE),
    A.PadIfNeeded(TEACHER_IMG_SIZE, TEACHER_IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

BOUNCER_INFER_TF = A.Compose([
    A.Resize(BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
def _find_ckpt(base_dir: Path, variant: str, prefix: str) -> Path:
    """Check final/ first, then fallback to variant-specific dir."""
    final_path = base_dir.parent / "final" / f"{prefix}_best.pth"
    if final_path.exists():
        return final_path
    var_path = base_dir / f"{prefix}_{variant}_best.pth"
    if var_path.exists():
        return var_path
    raise FileNotFoundError(f"Checkpoint not found for {prefix} (variant: {variant})")

def load_bouncer() -> tuple[nn.Module, float]:
    ckpt_path = _find_ckpt(BOUNCER_CKPT_DIR, BOUNCER_DEPLOYED_VARIANT, "bouncer")
    model = models.mobilenet_v3_large(weights=None)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 1)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model, BOUNCER_THRESHOLD

def load_teacher() -> nn.Module:
    ckpt_path = _find_ckpt(TEACHER_CKPT_DIR, TEACHER_DEPLOYED_VARIANT, "teacher")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    variant = ckpt.get("variant", TEACHER_DEPLOYED_VARIANT)
    
    if variant == "deeplabv3plus-eb2":
        model = smp.DeepLabV3Plus(encoder_name="efficientnet-b2", encoder_weights=None, in_channels=3, classes=1, activation=None)
    else:
        encoder_map = {"resnet50": "resnet50", "efficientnet-b2": "efficientnet-b2", "mit_b2": "mit_b2"}
        model = smp.Unet(encoder_name=encoder_map.get(variant, "efficientnet-b2"), encoder_weights=None, in_channels=3, classes=1, activation=None)
        
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET & COLLATE (Fixes Varying-Shape RuntimeError)
# ═══════════════════════════════════════════════════════════════════════════════
class FactoryDataset(Dataset):
    def __init__(self, df, tier1_fnames, test_fnames):
        # Exclude Tier 1 images that belong to the test split (Blueprint 10.1)
        self.df = df[~df["source_path"].apply(lambda p: Path(p).name in test_fnames)].reset_index(drop=True)
        self.tier1_fnames = tier1_fnames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["source_path"])
        category = row["category"]
        is_tier1 = img_path.name in self.tier1_fnames
        
        img_rgb = load_image_clahe(img_path)
        if img_rgb is None:
            return None
            
        orig_h, orig_w = img_rgb.shape[:2]
        
        b_tensor = BOUNCER_INFER_TF(image=img_rgb)["image"]
        t_tensor = TEACHER_INFER_TF(image=img_rgb)["image"]
        
        tier1_mask = None
        if is_tier1:
            npy_candidates = list(TIER1_MASKS_DIR.glob(f"*{img_path.stem}*softmask.npy"))
            if npy_candidates:
                tier1_mask = np.load(str(npy_candidates[0])).astype(np.float32)
                
        return {
            "img_path": str(img_path), "stem": img_path.stem, "category": category,
            "is_tier1": is_tier1, "orig_h": orig_h, "orig_w": orig_w,
            "img_rgb": img_rgb, "b_tensor": b_tensor, "t_tensor": t_tensor, "tier1_mask": tier1_mask
        }

def safe_collate(batch):
    """Custom collate to handle mixed tensors and variable-shape numpy arrays."""
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    collated = {}
    elem = batch[0]
    for key in elem:
        if key in ("img_rgb", "tier1_mask"):
            # Keep variable-shape arrays as Python lists
            collated[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch])
        elif isinstance(elem[key], (int, float, bool, str)):
            collated[key] = [d[key] for d in batch]
        else:
            collated[key] = [d[key] for d in batch]
    return collated

# ═══════════════════════════════════════════════════════════════════════════════
# CPU & I/O HEAVY FUNCTIONS (Run in ThreadPool)
# ═══════════════════════════════════════════════════════════════════════════════
def refine_silhouette(prob_map: np.ndarray) -> np.ndarray:
    binary = (prob_map >= FACTORY_SILHOUETTE_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FACTORY_MORPH_KERNEL_SIZE, FACTORY_MORPH_KERNEL_SIZE))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    return binary

def get_reliability_weight(leaf_coverage: float) -> float:
    if leaf_coverage < FACTORY_MIN_LEAF_COVERAGE:
        return None  # Explicitly gate on config constant rather than relying on bracket sentinel
    for lo, hi, weight in FACTORY_WEIGHT_BRACKETS:
        if lo <= leaf_coverage < hi: return weight
    return FACTORY_WEIGHT_BRACKETS[-1][2]

def _in_range(h, s, v, rng: dict) -> np.ndarray:
    return (h >= rng["h"][0]) & (h <= rng["h"][1]) & (s >= rng["s"][0]) & (s <= rng["s"][1]) & (v >= rng["v"][0]) & (v <= rng["v"][1])

def _green_exclusion(h, s, v) -> np.ndarray:
    ge = HSV_GREEN_EXCL
    return (h >= ge["h_min"]) & (h <= ge["h_max"]) & (s >= ge["s_min"]) & (v >= ge["v_min"]) & (v <= ge["v_max"])

def _apply_r3_area_filter(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= FACTORY_R3_MIN_AREA_PX:
            filtered[labels == lbl] = 1
    return filtered

def _compute_gabor_combined_mask(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = img_rgb.shape[:2]
    combined = np.zeros((h, w), dtype=np.float32)
    
    for theta in GABOR_THETAS:
        for norm_freq in GABOR_NORMS:
            lambd = norm_freq * min(h, w) * 0.1
            # FIX: cv2.CV_32F prevents memory corruption on 1-channel grayscale
            kernel = cv2.getGaborKernel((GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE), GABOR_SIGMA, theta, lambd, GABOR_GAMMA, GABOR_PSI, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            filtered = np.abs(filtered)
            if filtered.max() > 0:
                filtered = 255 * (filtered / filtered.max())
            combined += filtered.astype(np.float32) / 255.0

    combined /= (len(GABOR_THETAS) * len(GABOR_NORMS))
    return (combined >= GABOR_THRESHOLD).astype(np.uint8)

def compute_hsv_hard_mask(img_rgb, silhouette, category):
    img_hsv = to_hsv(img_rgb)
    h, s, v = img_hsv[:,:,0].astype(np.int32), img_hsv[:,:,1].astype(np.int32), img_hsv[:,:,2].astype(np.int32)
    green_excl = _green_exclusion(h, s, v)
    ranges = HSV_MSV_RANGES if category == "MSV" else HSV_MLN_RANGES
    
    symptom = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for i, rng in enumerate(ranges):
        hit = _in_range(h, s, v, rng)
        if category == "MSV" and i == 2: 
            hit = _apply_r3_area_filter(hit.astype(np.uint8)).astype(bool)
        symptom |= hit.astype(np.uint8)
        
    symptom &= (~green_excl).astype(np.uint8)
    symptom &= silhouette
    if category == "MSV": symptom &= _compute_gabor_combined_mask(img_rgb)
    return (symptom * 255).astype(np.uint8)

def compute_hsv_soft_confidence(img_rgb, silhouette, category):
    img_hsv = to_hsv(img_rgb)
    h, s, v = img_hsv[:,:,0].astype(np.float32), img_hsv[:,:,1].astype(np.float32), img_hsv[:,:,2].astype(np.float32)
    green_excl = _green_exclusion(h.astype(np.int32), s.astype(np.int32), v.astype(np.int32))
    ranges = HSV_MSV_RANGES if category == "MSV" else HSV_MLN_RANGES
    
    conf_accum = np.zeros(img_rgb.shape[:2], dtype=np.float32)
    hit_count  = np.zeros(img_rgb.shape[:2], dtype=np.float32)  # FIX: track per-pixel hit count
    for i, rng in enumerate(ranges):
        h_lo, h_hi = rng["h"]; s_lo, s_hi = rng["s"]; v_lo, v_hi = rng["v"]
        hit = _in_range(h.astype(np.int32), s.astype(np.int32), v.astype(np.int32), rng)
        if category == "MSV" and i == 2: hit = _apply_r3_area_filter(hit.astype(np.uint8)).astype(bool)
        
        h_center, s_center, v_center = (h_lo+h_hi)/2.0, (s_lo+s_hi)/2.0, (v_lo+v_hi)/2.0
        h_half, s_half, v_half = max((h_hi-h_lo)/2.0, 1.0), max((s_hi-s_lo)/2.0, 1.0), max((v_hi-v_lo)/2.0, 1.0)
        
        d = np.clip((1.0 - np.abs(h-h_center)/h_half + 1.0 - np.abs(s-s_center)/s_half + 1.0 - np.abs(v-v_center)/v_half)/3.0, 0.0, 1.0)
        hit_f = hit.astype(np.float32)
        conf_accum += hit_f * d
        hit_count  += hit_f  # FIX: count how many ranges fired per pixel
        
    # FIX: normalize by actual hit count (max 1 where any range fired) instead of total
    # len(ranges), which was capping MLN pixels at 0.20 (1/5) and preventing the 0.3 threshold
    safe_count = np.maximum(hit_count, 1.0)
    conf_map = (conf_accum / safe_count) * (~green_excl).astype(np.float32) * silhouette.astype(np.float32)
    if category == "MSV": conf_map *= _compute_gabor_combined_mask(img_rgb).astype(np.float32)
    return np.clip(conf_map, 0.0, 1.0)

def process_single_image_cpu(args):
    """Runs entirely on CPU threads to unblock the GPU pipeline"""
    img_path, stem, category, is_tier1, orig_h, orig_w, img_rgb, soft_prob, mode_dirs = args
    
    # mode_a: Otsu hard threshold on grayscale
    # mode_b: morphology-refined binary from teacher soft_prob  (close + open)
    # mode_c: raw soft_prob thresholded without morphology cleanup
    # mode_d: same sil as mode_b, but soft confidence symptom map
    binary_sil    = refine_silhouette(soft_prob)                          # morph-refined  (mode_b, mode_d)
    raw_thresh_sil = (soft_prob >= FACTORY_SILHOUETTE_THRESHOLD).astype(np.uint8)  # no morph (mode_c)
    otsu_sil      = (cv2.threshold(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0).astype(np.uint8)
    
    # Coverage and weight are derived from the refined silhouette (mode_b baseline)
    coverage = float(binary_sil.sum()) / (orig_h * orig_w)
    weight = get_reliability_weight(coverage)
    
    result = {"filename": stem, "category": category, "coverage": round(coverage, 4), "weight": weight if weight is not None else -1}
    modes_data = {}
    
    SIL_MAP = {
        "mode_a": otsu_sil,
        "mode_b": binary_sil,
        "mode_c": raw_thresh_sil,   # FIX: was binary_sil — mode_c must differ from mode_b
        "mode_d": binary_sil,
    }
    
    for mode in FACTORY_MODES:
        mode = mode.strip()  # Safeguard against OCR spaces in config
        sil = SIL_MAP[mode]
        symptom = compute_hsv_soft_confidence(img_rgb, sil, category) if mode == "mode_d" else compute_hsv_hard_mask(img_rgb, sil, category)
        sym_binary = (symptom >= 0.3).astype(np.uint8) if mode == "mode_d" else (symptom > 0).astype(np.uint8)
        
        leaf_px = float(sil.sum())
        sev = (float(sym_binary.sum()) / leaf_px) * 100.0 if leaf_px >= 1 and weight is not None else -1.0
        
        # Silhouette saved to disk: mode_c and mode_d store the continuous soft_prob
        # so the student can learn from the full probability map, not just the binary mask
        disk_sil = soft_prob if mode in ("mode_c", "mode_d") else sil.astype(np.float32)
        modes_data[mode] = {"silhouette": disk_sil, "symptom": symptom, "severity": sev}

    # Async Disk I/O
    for mode, data in modes_data.items():
        mode_dir = mode_dirs[mode]
        sil_out = cv2.resize(data["silhouette"].astype(np.float32), (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        np.save(str(mode_dir / f"{stem}_silhouette.npy"), sil_out)
        
        if mode == "mode_d":
            sym_out = cv2.resize(data["symptom"].astype(np.float32), (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            np.save(str(mode_dir / f"{stem}_symptom.npy"), sym_out)
        else:
            sym_out = cv2.resize(data["symptom"], (STUDENT_IMG_SIZE, STUDENT_IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(mode_dir / f"{stem}_symptom.png"), sym_out)
            
        (mode_dir / f"{stem}_sev.txt").write_text(str(round(data["severity"], 4)))
        (mode_dir / f"{stem}_weight.txt").write_text(str(result["weight"]))
        
    result["status"] = "processed"
    result.update({f"{m}_sev": round(modes_data[m]["severity"], 4) for m in modes_data})
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATORS (Blueprint 10.7)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_filter_breakdown(report_rows):
    from collections import Counter
    status_counts = Counter(r.get("status", "unknown") for r in report_rows)
    total = sum(status_counts.values())
    
    breakdown_path = REPORTS_DIR / "factory_filter_breakdown.csv"
    with open(breakdown_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["status", "count", "pct"])
        for status, cnt in status_counts.most_common():
            w.writerow([status, cnt, round(100*cnt/max(total,1), 2)])
    print(f"  Filter breakdown → {breakdown_path}")

def generate_factory_summary(report_rows):
    proc = [r for r in report_rows if r.get("status") == "processed"]
    if not proc: return
    
    df = pd.DataFrame(proc)
    summary_rows = []
    
    for mode in FACTORY_MODES:
        mode = mode.strip()
        sev_col = f"{mode}_sev"
        if sev_col not in df.columns: continue
        
        for cls in df["category"].unique():
            cls_df = df[df["category"] == cls]
            valid_sev = cls_df[sev_col].replace(-1.0, np.nan).dropna()
            
            summary_rows.append({
                "mode": mode, "class": cls,
                "n_total": len(cls_df), "n_processed": len(cls_df[cls_df["status"]=="processed"]),
                "pct_processed": round(100*len(cls_df[cls_df["status"]=="processed"])/max(len(cls_df),1), 1),
                "mean_severity": round(valid_sev.mean(), 2) if len(valid_sev) else "N/A",
                "std_severity": round(valid_sev.std(), 2) if len(valid_sev) else "N/A",
                "median_severity": round(valid_sev.median(), 2) if len(valid_sev) else "N/A",
                "pct_symptomatic": round(100*(valid_sev > 0).mean(), 1) if len(valid_sev) else 0.0,
                "n_excluded": len(cls_df[cls_df["weight"] == -1]),
                "pct_excluded": round(100*len(cls_df[cls_df["weight"] == -1])/max(len(cls_df),1), 1)
            })
            
    if summary_rows:
        summary_path = REPORTS_DIR / "factory_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys(), extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"  Factory summary → {summary_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 72, "\n  Yellow MAIze | Phase 4: Factory Pseudo-Label Generation\n" + "=" * 72)
    
    if not GLOBAL_MANIFEST.exists() or not TIER1_MANIFEST.exists():
        print("[FATAL] Missing manifest files. Run partition_dataset.py and sample_15000.py first.")
        return
        
    global_df = pd.read_csv(GLOBAL_MANIFEST)
    trainval = global_df[global_df["split"] != "test"].reset_index(drop=True)
    test_fnames = set(global_df[global_df["split"] == "test"]["filename"].tolist())
    tier1_fnames = set(Path(p).name for p in pd.read_csv(TIER1_MANIFEST)["source_path"].tolist())
    
    mode_dirs = {m.strip(): PSEUDO_DIR / m.strip() for m in FACTORY_MODES}
    for d in mode_dirs.values(): d.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n  Loading models to RTX 5060...")
    bouncer_model, bouncer_thresh = load_bouncer()
    teacher_model = load_teacher()
    
    dataset = FactoryDataset(trainval, tier1_fnames, test_fnames)
    loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True, collate_fn=safe_collate, prefetch_factor=2)
    
    report_rows = []
    t_start = time.time()
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        for i, batch in enumerate(loader):
            if batch is None: continue
            
            # 1. Batched Bouncer Inference (GPU)
            b_tensors = batch["b_tensor"].to(DEVICE, non_blocking=True)
            with torch.no_grad():
                b_probs = torch.sigmoid(bouncer_model(b_tensors)).view(-1)
                b_passed = b_probs >= bouncer_thresh
            
            passed_indices = torch.where(b_passed)[0].cpu().numpy()
            passed_set = set(passed_indices.tolist())  # FIX: O(1) lookup instead of O(n) numpy scan
            
            for idx in range(len(b_passed)):
                if idx not in passed_set:
                    report_rows.append({"filename": Path(batch["img_path"][idx]).stem, "status": "filtered_bouncer", "category": batch["category"][idx]})
            
            if len(passed_indices) == 0: continue
                
            # 2. Batched Teacher Inference (GPU)
            t_tensors = torch.stack([batch["t_tensor"][idx] for idx in passed_indices]).to(DEVICE, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher_model(t_tensors)
                t_probs_batch = torch.sigmoid(t_logits).squeeze(1).cpu().numpy()
                
            # 3. Dispatch to CPU ThreadPool
            futures = []
            for i_rel, idx in enumerate(passed_indices):
                orig_h = int(batch["orig_h"][idx])
                orig_w = int(batch["orig_w"][idx])
                is_tier1 = bool(batch["is_tier1"][idx])
                prob = t_probs_batch[i_rel]
                
                if is_tier1 and batch["tier1_mask"][idx] is not None:
                    soft_prob = cv2.resize(batch["tier1_mask"][idx], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                else:
                    soft_prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    
                args = (batch["img_path"][idx], batch["stem"][idx], batch["category"][idx], 
                        is_tier1, orig_h, orig_w, batch["img_rgb"][idx], soft_prob, mode_dirs)
                futures.append(executor.submit(process_single_image_cpu, args))
                
            for future in futures:
                report_rows.append(future.result())
                
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-3)
                eta = (len(loader) - i - 1) / rate
                print(f"  [{i+1:>4}/{len(loader)} Batches] | ETA: {eta/60:.1f} min | Processed: {len(report_rows)}")

    print("\n  Writing reports...")
    # Write one unified per-image report (all mode severities are already columns in report_rows)
    if report_rows:
        report_path = REPORTS_DIR / "phase4_report.csv"
        keys = list(report_rows[0].keys())
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"  Per-image report → {report_path}")
                
    generate_filter_breakdown(report_rows)
    generate_factory_summary(report_rows)
    
    print(f"\n  Total duration: {round(time.time() - t_start, 1)}s")
    print(f"  NEXT STEP: python train_student.py --stage 1")
    print("=" * 72)

if __name__ == "__main__":
    main()