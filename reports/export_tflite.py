"""
================================================================================
 export_tflite.py — Phase 7: TFLite Export for Android Deployment
================================================================================
 PURPOSE:
   Convert the best Student model checkpoint to TFLite format for
   deployment in the MAIze Android application.

 PIPELINE:
   1. Load best Student checkpoint (identified from Stage 2 comparison)
   2. Export to ONNX (intermediate format)
   3. Convert ONNX → TFLite via onnx2tf
   4. Validate the TFLite model on a small batch of test images
   5. Report model size, latency estimate, and inference correctness

 NOTE ON MobileViT-XXS (Variant 5):
   This variant is excluded from TFLite export due to self-attention
   einsum operations that are incompatible with TFLite converter.
   Deployed model is selected from Variants 1–4 only.

 OUTPUTS:
   exports/tflite/student_model.tflite   ← deploy in MAIze Android app
   exports/tflite/export_report.csv      ← size, latency, validation accuracy
================================================================================
"""

import csv
import time
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from PIL import Image

from config import (
    SEED,
    STUDENT_CKPT_DIR, EXPORTS_DIR, LOGS_DIR,
    STUDENT_IMG_SIZE, STUDENT_BATCH_SIZE,
    STUDENT_BEST_VARIANT, CLASSES,
    GLOBAL_MANIFEST, PSEUDO_DIR,
)
from train_student import StudentModel, StudentDataset, make_student_transforms, build_sample_list


ONNX_PATH   = EXPORTS_DIR / "student_model.onnx"
TFLITE_PATH = EXPORTS_DIR / "student_model.tflite"
REPORT_PATH = EXPORTS_DIR / "export_report.csv"

# Variants that cannot convert to TFLite
TFLITE_INCOMPATIBLE = {"mobilevit_xxs"}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_best_checkpoint() -> tuple[Path, str, str]:
    """
    Find the best Student checkpoint.
    Priority: checkpoints/final/student_best.pth (set by select_best_pipeline.py)
    Fallback: read student_comparison_stage2.csv or stage1 CSV.
    Returns (ckpt_path, encoder_variant, factory_mode).
    """
    from config import CHECKPOINTS_DIR
    final_ckpt = CHECKPOINTS_DIR / "final" / "student_best.pth"
    if final_ckpt.exists():
        # Load metadata from checkpoint
        ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=False)
        variant = ckpt.get("encoder", STUDENT_BEST_VARIANT)
        mode    = ckpt.get("mode",    "mode_b")
        print(f"  Using canonical best checkpoint: {final_ckpt.name}")
        return final_ckpt, variant, mode

    # Fallback: read comparison CSVs
    stage2_csv = LOGS_DIR / "student_comparison_stage2.csv"
    if stage2_csv.exists():
        df      = pd.read_csv(stage2_csv)
        df["best_composite"] = pd.to_numeric(df["best_composite"], errors="coerce")
        best_row = df.loc[df["best_composite"].idxmax()]
        variant  = str(best_row["encoder"])
        mode     = str(best_row["mode"])
    else:
        stage1_csv = LOGS_DIR / "student_comparison_stage1.csv"
        if stage1_csv.exists():
            df      = pd.read_csv(stage1_csv)
            df["best_composite"] = pd.to_numeric(df["best_composite"], errors="coerce")
            best_row = df.loc[df["best_composite"].idxmax()]
            variant  = str(best_row["encoder"])
        else:
            variant = STUDENT_BEST_VARIANT
        mode = "mode_b"

    # Check TFLite compatibility
    if variant in TFLITE_INCOMPATIBLE:
        print(f"  [WARN] Best variant '{variant}' is TFLite-incompatible.")
        stage1_csv = LOGS_DIR / "student_comparison_stage1.csv"
        if stage1_csv.exists():
            df     = pd.read_csv(stage1_csv)
            df["best_composite"] = pd.to_numeric(df["best_composite"], errors="coerce")
            compat = df[~df["encoder"].isin(TFLITE_INCOMPATIBLE)]
            if not compat.empty:
                variant = str(compat.loc[compat["best_composite"].idxmax(), "encoder"])
                print(f"         Using: {variant}")

    # Try stage1 subdir first, then root
    ckpt_path = STUDENT_CKPT_DIR / f"student_{variant}_{mode}_best.pth"
    if not ckpt_path.exists():
        ckpt_path = STUDENT_CKPT_DIR / "stage1" / f"student_{variant}_{mode}_best.pth"
    return ckpt_path, variant, mode


# ══════════════════════════════════════════════════════════════════════════════
# ONNX EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_to_onnx(model: torch.nn.Module,
                   onnx_path: Path,
                   img_size: int) -> bool:
    """Export PyTorch model to ONNX format."""
    try:
        import onnx

        dummy = torch.zeros(1, 3, img_size, img_size)
        model.eval()

        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["seg_logits", "cls_logits", "sev_out"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "seg_logits": {0: "batch_size"},
                "cls_logits": {0: "batch_size"},
                "sev_out":    {0: "batch_size"},
            },
        )

        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ONNX export verified: {onnx_path}")
        return True

    except ImportError:
        print("  [WARN] onnx not installed. Install: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"  [ERROR] ONNX export failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TFLITE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def convert_onnx_to_tflite(onnx_path: Path, tflite_path: Path) -> tuple[bool, str]:
    """
    Convert ONNX → TFLite using onnx2tf (the maintained replacement for the
    deprecated onnx-tf, which is incompatible with ONNX >= 1.16 due to the
    removal of `onnx.mapping`).

    onnx2tf converts directly from ONNX to a set of TFLite artifacts
    (typically FP32, FP16, and — if enabled — integer-quantized variants)
    in a single step, without an intermediate SavedModel export.

    Returns (success, quantization_label) where quantization_label
    describes which generated variant was selected (e.g. "FP16", "FP32").
    """
    try:
        import onnx2tf

        # onnx2tf writes its outputs (several .tflite variants plus a
        # SavedModel) into an output folder rather than a single file path.
        output_dir = tflite_path.parent / "onnx2tf_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("  Converting ONNX → TFLite via onnx2tf ...")
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(output_dir),
            output_signaturedefs=True,
            non_verbose=True,
        )

        # onnx2tf typically emits files like:
        #   model_float32.tflite, model_float16.tflite,
        #   model_dynamic_range_quant.tflite, model_integer_quant.tflite, ...
        # Prefer FP16 (smaller, still broadly accurate), then FP32.
        fp16_candidates = sorted(output_dir.glob("*float16*.tflite"))
        fp32_candidates = sorted(output_dir.glob("*float32*.tflite"))
        any_candidates  = sorted(output_dir.glob("*.tflite"))

        if fp16_candidates:
            src, quant_label = fp16_candidates[0], "FP16"
        elif fp32_candidates:
            src, quant_label = fp32_candidates[0], "FP32"
        elif any_candidates:
            src, quant_label = any_candidates[0], "UNKNOWN"
        else:
            print(f"  [ERROR] onnx2tf did not produce any .tflite file in {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)
            return False, "N/A"

        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tflite_path)

        size_mb = tflite_path.stat().st_size / (1024 * 1024)
        print(f"  TFLite model saved: {tflite_path}")
        print(f"  Model size        : {size_mb:.2f} MB")
        print(f"  Quantization      : {quant_label}  (source: {src.name})")

        # Clean up onnx2tf's intermediate output folder
        shutil.rmtree(output_dir, ignore_errors=True)
        return True, quant_label

    except ImportError as e:
        print(f"  [WARN] TFLite conversion dependencies missing: {e}")
        print("         Install: pip install onnx2tf tensorflow")
        return False, "N/A"
    except Exception as e:
        print(f"  [ERROR] TFLite conversion failed: {e}")
        return False, "N/A"


# ══════════════════════════════════════════════════════════════════════════════
# TFLITE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_tflite(tflite_path: Path,
                    model_pytorch: torch.nn.Module,
                    test_samples: list[dict],
                    mode: str,
                    n_validate: int = 50) -> dict:
    """
    Validate TFLite model outputs against PyTorch model outputs on n_validate samples.
    Checks that classification predictions agree and segmentation maps are similar.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("  [WARN] TensorFlow not available for TFLite validation.")
        return {}

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inp_det  = interpreter.get_input_details()
    out_det  = interpreter.get_output_details()

    val_tf  = make_student_transforms(STUDENT_IMG_SIZE, is_train=False)
    val_ds  = StudentDataset(test_samples[:n_validate], mode, val_tf)

    n_agree   = 0
    n_total   = 0
    seg_diffs = []
    latencies = []

    model_pytorch.eval()

    for i in range(min(n_validate, len(val_ds))):
        item    = val_ds[i]
        inp_t   = item["image"].unsqueeze(0)
        inp_np  = inp_t.numpy()

        # PyTorch inference
        with torch.no_grad():
            seg_pt, cls_pt, sev_pt = model_pytorch(inp_t)
        cls_pred_pt = cls_pt.argmax(dim=1).item()

        # TFLite inference
        interpreter.set_tensor(inp_det[0]["index"], inp_np)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)   # ms

        # Get outputs by name (preserved from ONNX export)
        # Names: seg_logits, cls_logits, sev_out
        cls_tflite = None
        seg_tflite = None
        for det in out_det:
            out  = interpreter.get_tensor(det["index"])
            name = det.get("name", "")
            if "cls" in name:
                cls_tflite = out
            elif "seg" in name:
                seg_tflite = out
            # fallback to shape heuristic if names stripped by converter
        if cls_tflite is None and seg_tflite is None:
            for det in out_det:
                out = interpreter.get_tensor(det["index"])
                if out.shape[-1] == len(CLASSES):
                    cls_tflite = out
                elif len(out.shape) == 4:
                    seg_tflite = out

        if cls_tflite is not None:
            cls_pred_tfl = np.argmax(cls_tflite, axis=-1)[0]
            if cls_pred_tfl == cls_pred_pt:
                n_agree += 1
            n_total += 1

        if seg_tflite is not None and seg_pt is not None:
            seg_pt_np = torch.sigmoid(seg_pt[:, 0]).squeeze().numpy()
            # TFLite output is NHWC: [1, H, W, 2] — channel 0 = silhouette
            seg_tfl = seg_tflite.squeeze()   # [H, W, 2]
            if seg_tfl.ndim == 3:
                seg_tfl = seg_tfl[:, :, 0]  # NHWC: take channel 0
            if seg_pt_np.shape == seg_tfl.shape:
                seg_diffs.append(np.abs(seg_pt_np - seg_tfl).mean())

    agreement = n_agree / max(n_total, 1)
    avg_lat   = np.mean(latencies) if latencies else float("nan")
    avg_seg   = np.mean(seg_diffs) if seg_diffs else float("nan")

    print(f"\n  TFLite validation ({n_total} samples):")
    print(f"    Classification agreement : {agreement*100:.1f}%")
    print(f"    Mean seg output diff     : {avg_seg:.5f}")
    print(f"    Mean inference latency   : {avg_lat:.1f} ms")

    return {
        "n_validated":     n_total,
        "cls_agreement":   round(agreement, 4),
        "mean_seg_diff":   round(float(avg_seg), 5) if not np.isnan(avg_seg) else "nan",
        "mean_latency_ms": round(float(avg_lat), 2) if not np.isnan(avg_lat) else "nan",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import time as _time
    set_seeds(SEED)
    _t_start = _time.time()

    print("=" * 72)
    print("  Yellow MAIze | Phase 7: TFLite Export")
    print("=" * 72)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Find best checkpoint ──────────────────────────────────────────────────
    ckpt_path, variant, mode = find_best_checkpoint()

    print(f"\n  Best model : {variant}  |  Mode: {mode}")
    print(f"  Checkpoint : {ckpt_path}")

    if not ckpt_path.exists():
        print(f"[FATAL] Checkpoint not found: {ckpt_path}")
        print("        Run train_student.py --stage 1 and --stage 2 first.")
        return

    if variant in TFLITE_INCOMPATIBLE:
        print(f"[FATAL] {variant} cannot be converted to TFLite.")
        print("        Select a compatible variant from Variants 1–4.")
        return

    # ── Load model ─────────────────────────────────────────────────────────────
    use_cbam = "cbam" in variant
    model    = StudentModel(variant, use_cbam=use_cbam)
    ckpt     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Model loaded. Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ── ONNX export ───────────────────────────────────────────────────────────
    print(f"\n  Step 1: Export to ONNX ...")
    onnx_ok = export_to_onnx(model, ONNX_PATH, STUDENT_IMG_SIZE)

    # ── TFLite conversion ─────────────────────────────────────────────────────
    tflite_ok   = False
    quant_label = "N/A"
    if onnx_ok:
        print(f"\n  Step 2: Convert to TFLite ...")
        tflite_ok, quant_label = convert_onnx_to_tflite(ONNX_PATH, TFLITE_PATH)

    # ── Validate ──────────────────────────────────────────────────────────────
    val_results = {}
    if tflite_ok:
        print(f"\n  Step 3: Validate TFLite model ...")
        test_samples = build_sample_list("test", mode)
        val_results  = validate_tflite(
            TFLITE_PATH, model, test_samples, mode, n_validate=50)

    # ── Export report ─────────────────────────────────────────────────────────
    model_size_mb = (TFLITE_PATH.stat().st_size / (1024*1024)
                     if tflite_ok and TFLITE_PATH.exists() else 0)
    onnx_size_mb  = (ONNX_PATH.stat().st_size   / (1024*1024)
                     if onnx_ok  and ONNX_PATH.exists()   else 0)

    report = {
        "encoder_variant":    variant,
        "factory_mode":       mode,
        "n_parameters":       sum(p.numel() for p in model.parameters()),
        "onnx_export":        "success" if onnx_ok  else "failed",
        "tflite_conversion":  "success" if tflite_ok else "failed",
        "onnx_size_mb":       round(onnx_size_mb,   2),
        "tflite_size_mb":     round(model_size_mb,  2),
        "input_size":         f"{STUDENT_IMG_SIZE}×{STUDENT_IMG_SIZE}",
        "quantization":       quant_label if tflite_ok else "N/A",
        **val_results,
    }

    report["export_duration_s"] = round(_time.time() - _t_start, 1)
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report.keys())
        writer.writeheader()
        writer.writerow(report)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  Export summary:")
    print(f"    Encoder       : {variant}")
    print(f"    ONNX          : {'✓' if onnx_ok  else '✗'}  ({onnx_size_mb:.2f} MB)")
    print(f"    TFLite        : {'✓' if tflite_ok else '✗'}  ({model_size_mb:.2f} MB)")

    if val_results:
        print(f"    Cls agreement : {val_results.get('cls_agreement','N/A')}")
        print(f"    Latency (CPU) : {val_results.get('mean_latency_ms','N/A')} ms")

    if tflite_ok:
        print(f"\n  TFLite model: {TFLITE_PATH}")
        print(f"  → Copy this file into the MAIze Android app's assets/ folder.")
        print(f"  → Input: float32 [{STUDENT_IMG_SIZE}×{STUDENT_IMG_SIZE}×3]  NHWC (channels last, ImageNet normalized)")
        print(f"  → Outputs: seg_logits [H×W×2], cls_logits [3], sev_out [1]")
        print(f"  → Apply sigmoid to seg_logits and sev_out at inference.")
        print(f"  → Classification: argmax(softmax(cls_logits)) → 0=HEALTHY, 1=MSV, 2=MLN")
    else:
        print(f"\n  [NOTE] TFLite conversion requires:")
        print(f"         pip install onnx onnxruntime onnx2tf tensorflow")
        print(f"         ONNX model is available at: {ONNX_PATH}")

    print(f"\n  Report: {REPORT_PATH}")
    print(f"\n  Pipeline complete. All outputs in:")
    print(f"    checkpoints/student/ — model weights")
    print(f"    logs/               — training metrics and test results")
    print(f"    reports/            — QA, XAI overlays, severity analysis")
    print(f"    exports/tflite/     — Android deployment model")
    print("=" * 72)


if __name__ == "__main__":
    main()
