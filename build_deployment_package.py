"""
================================================================================
 build_deployment_package.py — Phase 8: Deployment Package Builder
================================================================================
 PURPOSE:
   Builds a complete, self-contained deployment package in exports/deploy/.
   Everything an Android developer needs to integrate the MAIze models
   in one folder.

 STEPS:
   1. Export Bouncer (MobileNetV3-Large) to TFLite
   2. Copy Student TFLite (already built by export_tflite.py)
   3. Write model_metadata.json — input/output specs, normalization
      parameters, class names, thresholds, all post-processing details
   4. Write DEPLOYMENT_README.md — step-by-step Android integration guide
   5. Validate both TFLite models with a dummy inference pass
   6. Write deployment_report.csv — sizes, latencies, validation results

 OUTPUT STRUCTURE:
   exports/deploy/
     bouncer_model.tflite      ← Gate model (runs first on every frame)
     student_model.tflite      ← Disease model (runs if bouncer passes)
     model_metadata.json       ← All specs an Android dev needs
     DEPLOYMENT_README.md      ← Integration guide
     deployment_report.csv     ← Sizes + latency summary

 RUN AFTER: python export_tflite.py

 ANDROID INTEGRATION OVERVIEW:
   Frame → Bouncer (224×224) → if maize → Student (224×224) → 3 outputs:
     seg_logits [2×224×224] → sigmoid → boundary mask
     cls_logits [3]         → softmax → argmax → class
     sev_out    [1]         → clamp(0,1) → ×100 → severity %
================================================================================
"""

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import (
    SEED,
    EXPORTS_DIR, LOGS_DIR, REPORTS_DIR, CHECKPOINTS_DIR,
    BOUNCER_CKPT_DIR, STUDENT_CKPT_DIR,
    BOUNCER_IMG_SIZE, STUDENT_IMG_SIZE,
    BOUNCER_DEPLOYED_VARIANT, STUDENT_BEST_VARIANT, STUDENT_FACTORY_MODE,
    CLASSES, CLASS_TO_IDX,
)

DEPLOY_DIR   = EXPORTS_DIR.parent / "deploy"
BOUNCER_TFLITE  = DEPLOY_DIR / "bouncer_model.tflite"
STUDENT_TFLITE  = DEPLOY_DIR / "student_model.tflite"
METADATA_JSON   = DEPLOY_DIR / "model_metadata.json"
README_PATH     = DEPLOY_DIR / "DEPLOYMENT_README.md"
REPORT_CSV      = DEPLOY_DIR / "deployment_report.csv"

# ImageNet normalization (same as all training transforms)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
# BOUNCER TFLITE EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_bouncer_tflite() -> tuple[bool, float]:
    """
    Export Bouncer (MobileNetV3-Large binary classifier) to TFLite using onnx2tf.
    Returns (success, size_mb).
    """
    from torchvision import models as tvm

    # Load canonical checkpoint
    final_ckpt = CHECKPOINTS_DIR / "final" / "bouncer_best.pth"
    if not final_ckpt.exists():
        # Fallback to per-variant checkpoint
        variant_ckpt = BOUNCER_CKPT_DIR / f"bouncer_{BOUNCER_DEPLOYED_VARIANT}_best.pth"
        if not variant_ckpt.exists():
            print(f"  [ERROR] Bouncer checkpoint not found.")
            return False, 0.0
        ckpt_path = variant_ckpt
    else:
        ckpt_path = final_ckpt

    print(f"  Loading Bouncer from: {ckpt_path.name}")
    model = tvm.mobilenet_v3_large(weights=None)
    in_f  = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    # ONNX intermediate export (using opset_version=18)
    onnx_path = DEPLOY_DIR / "bouncer_model.onnx"
    dummy = torch.zeros(1, 3, BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE)

    try:
        import onnx
        torch.onnx.export(
            model, dummy, str(onnx_path),
            export_params=True, opset_version=18,  # <--- Updated from 12 to 18
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logit"],
            dynamic_axes={"input": {0: "batch_size"},
                          "logit": {0: "batch_size"}},
        )
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        print(f"  Bouncer ONNX verified: {onnx_path.name}")
    except ImportError:
        print("  [WARN] onnx not installed. pip install onnx onnxruntime")
        return False, 0.0
    except Exception as e:
        print(f"  [ERROR] Bouncer ONNX export failed: {e}")
        return False, 0.0

    # TFLite conversion via onnx2tf
    try:
        import onnx2tf

        out_dir = DEPLOY_DIR / "_bouncer_onnx2tf_out"
        print("  Converting Bouncer ONNX → TFLite via onnx2tf ...")

        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(out_dir),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True,
        )

        tflite_candidates = list(out_dir.glob("*.tflite"))
        if not tflite_candidates:
            print("  [ERROR] No TFLite file generated by onnx2tf for Bouncer.")
            return False, 0.0

        target_file = tflite_candidates[0]
        for f in tflite_candidates:
            if "float16" in f.name or "float32" in f.name:
                target_file = f
                break

        shutil.copy(target_file, BOUNCER_TFLITE)
        shutil.rmtree(out_dir, ignore_errors=True)
        onnx_path.unlink(missing_ok=True)

        size_mb = round(BOUNCER_TFLITE.stat().st_size / (1024 * 1024), 2)
        print(f"  Bouncer TFLite: {BOUNCER_TFLITE.name} ({size_mb} MB)")
        return True, size_mb

    except Exception as e:
        print(f"  [ERROR] Bouncer TFLite conversion failed: {e}")
        return False, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# COPY STUDENT TFLITE
# ══════════════════════════════════════════════════════════════════════════════

def copy_student_tflite() -> tuple[bool, float]:
    """Copy student_model.tflite from exports/tflite/ to exports/deploy/."""
    src = EXPORTS_DIR / "student_model.tflite"
    if not src.exists():
        print(f"  [ERROR] Student TFLite not found: {src}")
        print("         Run export_tflite.py first.")
        return False, 0.0

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, STUDENT_TFLITE)
    size_mb = round(STUDENT_TFLITE.stat().st_size / (1024*1024), 2)
    print(f"  Student TFLite: {STUDENT_TFLITE.name} ({size_mb} MB)")
    return True, size_mb


# ══════════════════════════════════════════════════════════════════════════════
# TFLITE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_tflite_model(tflite_path: Path,
                           model_name: str,
                           input_shape: tuple = None) -> dict:
    """Run dummy inference through a TFLite model. Returns latency stats."""
    if not tflite_path.exists():
        return {"status": "file_not_found"}
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        inp_det = interpreter.get_input_details()
        out_det = interpreter.get_output_details()

        # Dynamically fetch the actual input tensor shape (e.g. NHWC [1, 224, 224, 3])
        expected_shape = tuple(inp_det[0]["shape"])
        dummy = np.random.randn(*expected_shape).astype(np.float32)

        latencies = []
        for _ in range(30):   # 10 warm-up + 20 measured
            interpreter.set_tensor(inp_det[0]["index"], dummy)
            t0 = time.perf_counter()
            interpreter.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)

        lat_mean = round(float(np.mean(latencies[10:])), 2)
        lat_std  = round(float(np.std(latencies[10:])),  2)
        outputs  = [{"index": d["index"], "name": d["name"],
                     "shape": list(d["shape"])} for d in out_det]

        print(f"  {model_name}: {lat_mean:.1f}±{lat_std:.1f} ms | "
              f"outputs: {[o['shape'] for o in outputs]}")
        return {
            "status":        "ok",
            "lat_mean_ms":   lat_mean,
            "lat_std_ms":    lat_std,
            "n_outputs":     len(out_det),
            "output_shapes": str([o["shape"] for o in outputs]),
        }
    except ImportError:
        return {"status": "tensorflow_not_installed"}
    except Exception as e:
        return {"status": f"error:{e}"}


# ══════════════════════════════════════════════════════════════════════════════
# METADATA JSON
# ══════════════════════════════════════════════════════════════════════════════

def write_metadata(bouncer_thresh: float,
                   bouncer_val: dict,
                   student_val: dict) -> None:
    """
    Write model_metadata.json — the complete spec for Android integration.
    Contains everything needed: input format, output format, normalization,
    thresholds, class names, post-processing steps.
    """
    metadata = {
        "project": "Yellow MAIze",
        "version": "1.0",
        "description": f"MAIze: {STUDENT_BEST_VARIANT}-UNet + Grad-CAM for Maize Disease Detection",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_platform": "Android (TFLite)",

        "preprocessing": {
            "description": "Apply to every input image before inference",
            "step1_resize": {
                "method": "letterbox",
                "description": "Resize longest side to target, pad shorter side with zeros",
                "bouncer_target_px": BOUNCER_IMG_SIZE,
                "student_target_px": STUDENT_IMG_SIZE,
            },
            "step2_normalize": {
                "description": "ImageNet normalization: (pixel/255 - mean) / std",
                "mean": IMAGENET_MEAN,
                "std":  IMAGENET_STD,
                "input_range": "float32 [0, 1] after /255, then normalized",
            },
            "step3_layout": "NHWC — shape [1, H, W, 3] (channels last). TFLite always uses NHWC. The ONNX→TF→TFLite conversion automatically transposes from PyTorch NCHW to TFLite NHWC.",
            "clahe_note": "CLAHE is applied in the training pipeline but NOT required at inference — the model has learned to handle lighting variation through augmentation.",
            "exif_note": "Apply EXIF orientation correction before resizing.",
        },

        "bouncer": {
            "file":        "bouncer_model.tflite",
            "architecture":"MobileNetV3-Large binary classifier",
            "input": {
                "name":  "input",
                "shape": [1, BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE, 3],
                "dtype": "float32",
                "layout": "NHWC — channels last (TFLite default)",
            },
            "output": {
                "name":        "logit",
                "shape":       [1, 1],
                "dtype":       "float32",
                "description": "Raw logit — apply sigmoid to get P(maize)",
            },
            "post_processing": {
                "step1": "sigmoid(logit) → probability in [0,1]",
                "step2": f"if probability >= {bouncer_thresh} → PASS to Student",
                "step3": f"if probability < {bouncer_thresh} → REJECT: show 'Not a maize leaf'",
            },
            "threshold":              bouncer_thresh,
            "threshold_note":         "Empirically selected on validation split to maximise specificity subject to maize recall ≥ 95%.",
            "heuristic_prefilter": {
                "description": "Run this BEFORE neural bouncer (microseconds, saves battery)",
                "green_hsv_range": "H:[35,75] S:>50 V:>40",
                "min_green_coverage_pct": 15,
                "min_aspect_ratio": 1.5,
                "max_aspect_ratio": 18.0,
                "action_if_fail": "Reject immediately without calling TFLite model",
            },
            "validation": bouncer_val,
        },

        "student": {
            "file":        "student_model.tflite",
            "architecture":f"{STUDENT_BEST_VARIANT}-UNet multi-task (3 heads)",
            "input": {
                "name":  "input",
                "shape": [1, STUDENT_IMG_SIZE, STUDENT_IMG_SIZE, 3],
                "dtype": "float32",
                "layout": "NHWC — channels last (TFLite default)",
            },
            "outputs": {
                "segmentation": {
                    "index":       0,
                    "shape":       [1, STUDENT_IMG_SIZE, STUDENT_IMG_SIZE, 2],
                    "layout":      "NHWC — [:, :, :, 0]=leaf silhouette, [:, :, :, 1]=symptom mask",
                    "dtype":       "float32 (raw logits)",
                    "channel_0":   "Leaf silhouette — sigmoid → binary boundary mask",
                    "channel_1":   "Symptom mask    — sigmoid → binary symptom region",
                    "post_process":"sigmoid(logits) → binary mask at threshold 0.5",
                },
                "classification": {
                    "index":   1,
                    "shape":   [1, 3],
                    "dtype":   "float32 (raw logits)",
                    "classes": {str(v): k for k, v in CLASS_TO_IDX.items()},
                    "post_process": "softmax(logits) → probabilities; argmax → class index",
                    "class_names": CLASSES,
                },
                "severity": {
                    "index":       2,
                    "shape":       [1, 1],
                    "dtype":       "float32",
                    "range":       "[0.0, 1.0] — multiply by 100 to get severity %",
                    "post_process":"value is already clamped [0,1]; severity_pct = value * 100",
                    "note":        "Severity is a model-learned estimate consistent with HSV-derived pseudo-labels. Not agronomically validated.",
                },
            },
            "grad_cam_note": {
                "description": "Grad-CAM++ heatmap is computed at runtime in the Android app, not stored in TFLite.",
                "target_layer": f"Last convolutional block of the {STUDENT_BEST_VARIANT} encoder",
                "target_layer_note": "For mobilenet_v2: features[-1]; for efficientnet_b2: blocks[-1]; for resnet: layer4. Check train_student.py GradCAMWrapper for exact hook point.",
                "library":      "Use tflite-support or compute gradient manually",
                "app_display":  "Show as semi-transparent amber overlay labelled 'Diagnostic attention'",
                "seg_display":  "Show segmentation channel_0 as green contour labelled 'Symptom boundary'",
            },
            "validation": student_val,
        },

        "inference_pipeline": {
            "step1": "Capture frame from camera",
            "step2": "Apply EXIF orientation correction",
            "step3": "Run heuristic pre-filter (green coverage + aspect ratio)",
            "step4": "If heuristic fails: show 'Point camera at a maize leaf'",
            "step5": "Resize + letterbox pad to 224×224",
            "step6": "Normalize (ImageNet mean/std)",
            "step7": "Run Bouncer TFLite → sigmoid → threshold",
            "step8": "If Bouncer fails: show 'Not detected as maize'",
            "step9": "Run Student TFLite → 3 outputs",
            "step10":"Post-process segmentation: sigmoid → threshold 0.5 → contour",
            "step11":"Post-process classification: softmax → argmax → class name",
            "step12":"Post-process severity: ×100 → display %",
            "step13":"Compute Grad-CAM++ heatmap on classification output",
            "step14":"Display: green boundary contour + amber heatmap + class + severity",
        },

        "android_notes": {
            "min_api_level":  21,
            "tflite_runtime": "org.tensorflow:tensorflow-lite:2.13.0",
            "image_input":    "Use TensorImage with ImageProcessor for resize+normalize",
            "thread_note":    "Run inference on background thread, post results to UI thread",
            "gpu_delegate":   "Optional: use GpuDelegate for faster inference on supported devices",
        },
    }

    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata JSON: {METADATA_JSON.name}")


# ══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT README
# ══════════════════════════════════════════════════════════════════════════════

def write_readme(bouncer_ok: bool, student_ok: bool,
                 bouncer_thresh: float) -> None:
    readme = f"""# MAIze Android Deployment Guide
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Files in this package

| File | Size | Purpose |
|---|---|---|
| `bouncer_model.tflite` | See report | Gate: maize vs non-maize |
| `student_model.tflite` | See report | Disease detection (3 heads) |
| `model_metadata.json`  | —          | Complete input/output specs |
| `deployment_report.csv`| —          | Size + latency summary |

Status: Bouncer {'✓ ready' if bouncer_ok else '⚠ PyTorch fallback only'} | Student {'✓ ready' if student_ok else '✗ not found'}

---

## Android Studio Integration

### 1. Add TFLite dependency (build.gradle)
```groovy
dependencies {{
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'  // optional
}}
```

### 2. Copy model files
```
app/src/main/assets/
    bouncer_model.tflite
    student_model.tflite
```

### 3. Preprocessing (Java/Kotlin)
```kotlin
// Letterbox resize to 224×224 (longest side → 224, pad shorter side with zeros)
val imageProcessor = ImageProcessor.Builder()
    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
    .add(NormalizeOp(
        floatArrayOf(0.485f, 0.456f, 0.406f),  // mean (ImageNet)
        floatArrayOf(0.229f, 0.224f, 0.225f)    // std  (ImageNet)
    ))
    .build()

// Apply EXIF orientation first
val bitmap = ExifInterface(imagePath).let {{
    BitmapFactory.decodeFile(imagePath)
        .rotateBitmap(it.getAttributeInt(ExifInterface.TAG_ORIENTATION, 1))
}}

val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
```

### 4. Heuristic pre-filter (before Bouncer)
```kotlin
// Fast green coverage check — reject non-plant images immediately
fun isLikelyMaize(bitmap: Bitmap): Boolean {{
    val hsv = FloatArray(3)
    var greenPixels = 0
    val total = bitmap.width * bitmap.height
    for (x in 0 until bitmap.width step 4) {{
        for (y in 0 until bitmap.height step 4) {{
            Color.colorToHSV(bitmap.getPixel(x, y), hsv)
            val h = hsv[0]; val s = hsv[1] * 255; val v = hsv[2] * 255
            if (h in 35f..75f && s > 50 && v > 40) greenPixels++
        }}
    }}
    return (greenPixels.toFloat() / (total / 16)) > 0.15f
}}
```

### 5. Run Bouncer
```kotlin
val bouncerInterpreter = Interpreter(
    FileUtil.loadMappedFile(context, "bouncer_model.tflite"))

val bouncerOutput = Array(1) {{ FloatArray(1) }}
bouncerInterpreter.run(tensorImage.buffer, bouncerOutput)

val sigmoid = 1f / (1f + Math.exp(-bouncerOutput[0][0].toDouble())).toFloat()
if (sigmoid < {bouncer_thresh}f) {{
    showMessage("Not detected as a maize leaf")
    return
}}
```

### 6. Run Student
```kotlin
val studentInterpreter = Interpreter(
    FileUtil.loadMappedFile(context, "student_model.tflite"))

// Output buffers — TFLite outputs in NHWC layout
val segOutput = Array(1) {{ Array(224) {{ Array(224) {{ FloatArray(2) }} }} }}  // [1,H,W,2]
val clsOutput = Array(1) {{ FloatArray(3) }}                                     // [1,3]
val sevOutput = Array(1) {{ FloatArray(1) }}                                     // [1,1]

val outputs = mapOf(0 to segOutput, 1 to clsOutput, 2 to sevOutput)
studentInterpreter.runForMultipleInputsOutputs(
    arrayOf(tensorImage.buffer), outputs)
```

### 7. Post-processing
```kotlin
// Classification
val classProbs = softmax(clsOutput[0])
val classIndex = classProbs.indexOfMax()
val className  = arrayOf("HEALTHY", "MSV", "MLN")[classIndex]
val confidence = classProbs[classIndex] * 100f

// Severity
val severityPct = sevOutput[0][0] * 100f

// Segmentation — NHWC layout: segOutput[0][y][x][channel]
// channel 0 = leaf silhouette, channel 1 = symptom mask
val silhouette = Array(224) {{ y -> FloatArray(224) {{ x -> segOutput[0][y][x][0] }} }}
val symptoms   = Array(224) {{ y -> FloatArray(224) {{ x -> segOutput[0][y][x][1] }} }}
// Apply sigmoid and threshold at 0.5 to get binary masks
// Draw green contour overlay from silhouette mask
// Draw symptom region overlay from symptoms mask

// UI labels
displayResult(
    className, confidence, severityPct,
    silhouetteMask = binarize(sigmoid(silhouette), 0.5f),
    symptomMask    = binarize(sigmoid(symptoms),   0.5f)
)
```

### 8. Display outputs
- **Green contour** from `silhouette` channel → label: "Symptom boundary"
- **Amber heatmap** from Grad-CAM++ on classification output → label: "Diagnostic attention"
- **Class badge**: HEALTHY / MSV / MLN with confidence %
- **Severity gauge**: 0–100% bar

---

## Important notes

1. **Input normalization** — must use ImageNet mean/std exactly as above. Wrong normalization produces random outputs with no error.
2. **Channel order** — TFLite uses NHWC (channels last): shape `[1, 224, 224, 3]`. The ONNX→TF→TFLite conversion handles the PyTorch NCHW→NHWC transpose automatically. Pass your Android `TensorImage` directly — do NOT manually transpose to NCHW.
3. **Sigmoid vs softmax** — segmentation and severity outputs need sigmoid; classification needs softmax. Applying the wrong function produces incorrect results silently.
4. **Severity disclaimer** — severity % is learned from HSV-derived pseudo-labels, not expert agronomic ratings. Display as an estimate, not a diagnosis.
5. **Bouncer threshold** — the value `{bouncer_thresh}` was empirically selected on the validation split. Do not hardcode a different value.

---

## Full model_metadata.json
See `model_metadata.json` for complete technical specifications including all
input/output tensor shapes, normalization parameters, and post-processing steps.
"""

    README_PATH.write_text(readme, encoding="utf-8")
    print(f"  README: {README_PATH.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.time()
    print("=" * 65)
    print("  Yellow MAIze | Phase 8: Deployment Package Builder")
    print("=" * 65)

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    # ── Read best pipeline summary for accurate model selection ───────────────
    summary_csv = REPORTS_DIR / "best_pipeline_summary.csv"
    if not summary_csv.exists():
        print("  [WARN] best_pipeline_summary.csv not found.")
        print("         Run select_best_pipeline.py first for accurate model selection.")
        print("         Proceeding with config.py defaults.")
    else:
        import pandas as _pd
        summary_df = _pd.read_csv(summary_csv)
        # Override config defaults with actual best values
        student_rows = summary_df[summary_df.get("component", _pd.Series()) == "Student"]
        if not student_rows.empty:
            row = student_rows.iloc[0]
            if "winner_encoder" in row.index:
                import config as _cfg
                _cfg.STUDENT_BEST_VARIANT = str(row["winner_encoder"])
            if "winner_mode" in row.index:
                import config as _cfg
                _cfg.STUDENT_FACTORY_MODE = str(row["winner_mode"])
        print(f"  Best Student encoder: {STUDENT_BEST_VARIANT}")
        print(f"  Best Factory mode   : {STUDENT_FACTORY_MODE}")

    # Get Bouncer threshold from checkpoint
    bouncer_thresh = 0.65   # default
    final_bouncer  = CHECKPOINTS_DIR / "final" / "bouncer_best.pth"
    fb_ckpt        = BOUNCER_CKPT_DIR / f"bouncer_{BOUNCER_DEPLOYED_VARIANT}_best.pth"
    for bp in [final_bouncer, fb_ckpt]:
        if bp.exists():
            try:
                ckpt = torch.load(bp, map_location="cpu", weights_only=False)
                bouncer_thresh = float(ckpt.get("threshold", bouncer_thresh))
            except Exception:
                pass
            break

    # Step 1: Export Bouncer TFLite
    print("\n  Step 1: Exporting Bouncer to TFLite ...")
    bouncer_ok, bouncer_mb = export_bouncer_tflite()

    # Step 2: Copy Student TFLite
    print("\n  Step 2: Copying Student TFLite ...")
    student_ok, student_mb = copy_student_tflite()

    # Step 3: Validate TFLite models
    print("\n  Step 3: Validating TFLite models ...")
    bouncer_val = validate_tflite_model(
        BOUNCER_TFLITE, "Bouncer",
        (1, 3, BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE))
    student_val = validate_tflite_model(
        STUDENT_TFLITE, "Student",
        (1, 3, STUDENT_IMG_SIZE, STUDENT_IMG_SIZE))

    # Step 4: Write metadata JSON
    print("\n  Step 4: Writing model_metadata.json ...")
    write_metadata(bouncer_thresh, bouncer_val, student_val)

    # Step 5: Write deployment README
    print("\n  Step 5: Writing DEPLOYMENT_README.md ...")
    write_readme(bouncer_ok, student_ok, bouncer_thresh)

    # Step 6: Write deployment report
    print("\n  Step 6: Writing deployment_report.csv ...")
    duration = round(time.time() - t_start, 1)
    report_rows = [
        {
            "model":          "bouncer",
            "file":           "bouncer_model.tflite",
            "tflite_ok":      bouncer_ok,
            "size_mb":        bouncer_mb,
            "input_size_px":  BOUNCER_IMG_SIZE,
            "lat_mean_ms":    bouncer_val.get("lat_mean_ms", "N/A"),
            "lat_std_ms":     bouncer_val.get("lat_std_ms",  "N/A"),
            "val_status":     bouncer_val.get("status", "N/A"),
        },
        {
            "model":          "student",
            "file":           "student_model.tflite",
            "tflite_ok":      student_ok,
            "size_mb":        student_mb,
            "input_size_px":  STUDENT_IMG_SIZE,
            "lat_mean_ms":    student_val.get("lat_mean_ms", "N/A"),
            "lat_std_ms":     student_val.get("lat_std_ms",  "N/A"),
            "val_status":     student_val.get("status", "N/A"),
        },
    ]
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    # Summary
    print(f"\n{'─' * 65}")
    print(f"  Deployment package: {DEPLOY_DIR}")
    print(f"  Bouncer TFLite    : {'✓' if bouncer_ok else '⚠ fallback'}"
          f"  {bouncer_mb} MB  {bouncer_val.get('lat_mean_ms','?')} ms")
    print(f"  Student TFLite    : {'✓' if student_ok else '✗'}"
          f"  {student_mb} MB  {student_val.get('lat_mean_ms','?')} ms")
    print(f"  Total latency est.: ~{sum(filter(lambda x: isinstance(x,(int,float)), [bouncer_val.get('lat_mean_ms',0), student_val.get('lat_mean_ms',0)]))} ms per image")
    print(f"  Duration          : {duration}s")
    print(f"\n  Contents:")
    for f in sorted(DEPLOY_DIR.iterdir()):
        sz = f"{f.stat().st_size/1024:.1f} KB" if f.is_file() else ""
        print(f"    {f.name:<35} {sz}")
    print(f"\n  NEXT STEP: python generate_report.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
