# Yellow MAIze — Complete Project Summary
## Context document for continuing work in a new chat

---

## 1. What This Project Is

**Thesis title:** MAIze: A U-Net with MobileNetV2 and Explainable AI Framework for Maize Streak Virus Identification and Symptom Segmentation in Zea mays L.

**Institution:** Angeles University Foundation, College of Computer Studies, BSCS 3-A

**Team:** Altes, Zylah Klein · Davis, Dominic · Tayer, Catherine P. · Ursua, Walter Vince

**Clinical problem:** MSV (Maize Streak Virus) was first confirmed in the Philippines in 2023 (Bukidnon, South Cotabato — Cruz et al. 2024). Early symptoms are visually indistinguishable from nutrient deficiencies. The system targets agriculture students and smallholder farmers who need a quick, explainable diagnosis on a mobile phone.

**Deployment target:** Android (TFLite). Two models run in sequence on-device: a Bouncer gate that rejects non-maize images, then a Student that classifies disease and segments the affected leaf area.

**Dataset:** Maize Imagery Dataset — Tanzania (Mduma 2023, Mendeley) + Zenodo healthy/MSV samples. ~252,000 images across three classes: HEALTHY (~38%), MLN (~38%), MSV (~24%). Tanzania origin with Philippine deployment is a stated geographic limitation.

---

## 2. Hardware and Software

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 5060, 8 GB VRAM |
| CPU | AMD Ryzen 5 3600X |
| RAM | 16 GB DDR4 |
| OS | Windows 11 + WSL2 Ubuntu — all Python runs in WSL2 |
| Workers | 4 (WSL2 /dev/shm constraint) |
| Seed | 42 — enforced in every script |
| cuDNN | deterministic=True, benchmark=False |

Core libraries: PyTorch ≥ 2.1.0, torchvision ≥ 0.16.0, segmentation-models-pytorch ≥ 0.3.3, albumentations ≥ 1.3.1, timm ≥ 0.9.12, opencv-python ≥ 4.8.0, Pillow ≥ 10.0.0, grad-cam ≥ 1.4.8, imagehash ≥ 4.3.1, scikit-learn ≥ 1.3.0, ultralytics ≥ 8.0.0, SAM2 (from GitHub), TensorFlow ≥ 2.13.0 (TFLite conversion only).

---

## 3. Pipeline Map

| Phase | Script | Primary output |
|---|---|---|
| 0-pre | `partition_dataset.py` | `global_split_manifest.csv` — 70/15/15 stratified split |
| 0a | `create_bouncer_dataset.py` | `data/bouncer_dataset/` — 50k balanced binary dataset |
| 0b | `train_bouncer.py` | `checkpoints/bouncer/bouncer_{variant}_best.pth` |
| 1 | `sample_15000.py` | `data/tier1_raw/` + `tier1_manifest.csv` |
| 1b | `sample_gold_standard.py` | `data/gold_standard/images/` — 501 images, 167/class |
| 1c | `train_yolo_detector.py` | `checkpoints/yolo/best.pt` + `logs/yolo_qa_calibration.csv` |
| 2 | `generate_tier1_masks.py` | `data/tier1_leaf_masks/` — SAM2 float .npy + `tier1_qa_report.csv` |
| 2b | `validate_masks.py` | QA stats + `reports/tier1_overlays/` |
| 2c | `validate_gold_standard.py --sam2-only` | SAM2 IoU vs human masks |
| 3 | `train_teacher.py` | `checkpoints/teacher/teacher_{variant}_best.pth` |
| 2d | `validate_gold_standard.py` | SAM2 + Teacher IoU chain |
| 3b | `train_symptom_model.py` | `checkpoints/healthy_ae/` + `checkpoints/symptom/` |
| 4 | `factory_master.py` | `data/pseudo_masks/{mode}/` — pseudo-labels for ~215k Tier 2 images |
| 4b | `validate_factory.py` | `reports/validate_factory_{mode}.html` — visual QA of pseudo-masks (optional, pre-training) |
| 5a | `train_student.py --stage 1` | 5 encoder variants × Mode B → `checkpoints/student/stage1/` |
| 5b | `train_student.py --stage 2` | Best encoder × 4 modes → `checkpoints/student/` |
| 5c | `validate_gold_standard.py` | Full SAM2 → Teacher → Student chain (final thesis table) |
| 6 | `select_best_pipeline.py` | `checkpoints/final/` + auto-updates `config.py` |
| 7 | `evaluate_xai.py` | 3-way XAI results + overlays |
| 8 | `evaluate_severity.py` | Inter-rater reliability results |
| 9 | `export_tflite.py` | `exports/tflite/student_model.tflite` |
| 10 | `build_deployment_package.py` | `exports/deploy/` — full Android bundle |
| 11 | `generate_report.py` | `reports/evaluation_report.html` |
| any | `generate_charts.py` | `reports/charts/*.png` |

---

## 4. Component Specifications

### 4.1 Preprocessing — `partition_dataset.py`

Run once before everything else. Never re-run after training begins.

Ten validation steps applied to every image:

1. **Zero-byte / tiny file** — stat < 100 bytes → reject
2. **Magic bytes** — JPEG must start with FF D8 FF, PNG with 89 PNG → mismatch → reject
3. **Truncation** — PIL `.load()` forces full pixel decode; exception → reject
4. **Minimum resolution** — min(w,h) < 64px → reject
5. **Maximum resolution** — max(w,h) > 4096px → reject
6. **Extreme aspect ratio** — max(w,h) / min(w,h) > 8.0 → reject
7. **Colour mode** — 1-bit binary → reject; grayscale/palette → flag and keep
8. **Near-uniform** — RGB pixel std < 5.0 → reject
9. **MD5 exact duplicates** — same hash across classes → reject both; within class → reject duplicate
10. **pHash near-duplicates** — Hamming ≤ 2 within class → reject; Hamming ≤ 2 across classes → reject both

Additionally: green-content flag (< 5% green pixels → flagged, kept).

Applies stratified 70/15/15 split per class. Writes `global_split_manifest.csv`, `reports/preprocessing_report.csv`, `reports/preprocessing_flagged.csv`, `reports/preprocessing_summary.txt`.

### 4.2 Image Utilities — `image_utils.py`

Single source of truth for all image loading. No script may call `cv2.imread()` or `PIL.Image.open()` directly.

- `load_image_rgb(path)` — PIL open → `ImageOps.exif_transpose()` → `.load()` (full decode, truncation guard) → uint8 RGB numpy or None
- `load_image_clahe(path)` — `load_image_rgb()` + CLAHE on L channel in LAB space (clipLimit=2.0, tileGridSize=(8,8)). Used for Bouncer and Factory inputs; not for Teacher or Student training.
- `apply_clahe(img_rgb)` — standalone CLAHE transform, returns uint8 RGB
- `to_hsv(img_rgb)` — guaranteed RGB→HSV
- `rgb_to_bgr(img_rgb)` — for `cv2.imwrite()` only

### 4.3 Shared Infrastructure — `scripts/`

**`scripts/safe_collate.py`** — Every DataLoader uses `collate_fn=safe_collate`. Filters None returns from `__getitem__` (corrupt images) before collating. Returns None if the entire batch is invalid; training loop skips with `continue`. Module-level skip counter: `reset_skip_counter()` at epoch start, `get_skip_count()` at epoch end.

**`scripts/bouncer_inference.py`** — Single source of truth for Bouncer inference, shared between `train_bouncer.py` and `factory_master.py`. Contains `BOUNCER_INFER_TF` (LongestMaxSize + PadIfNeeded + ImageNet normalize, Albumentations 2.0 API with `fill`/`fill_mask`), `heuristic_prefilter()` (always True — passthrough), and `neural_bouncer()` (`@torch.no_grad()`, returns `sigmoid(logit) >= threshold`). No heavy imports.

### 4.4 Bouncer — `train_bouncer.py` (Phase 0b)

Binary maize-vs-not-maize gate. Runs first at inference to reject non-maize images.

**Dataset:** 25,000 maize positives (train+val only, test excluded) + 25,000 non-maize negatives. Internal 80/20 split, seeded. Image loading: `load_image_clahe()`.

**4 variants:**

| Variant | Architecture | Deployed? |
|---|---|---|
| gabor_lbp | Gabor filters + LBP + LinearSVC | No — traditional CV baseline |
| mobilenet_v2 | MobileNetV2 head: Linear(1280→1) | No — neural comparison |
| **mobilenet_v3_large** | MobileNetV3-Large head: Linear(960→1) | **Yes** |
| edgevit_xxs | EdgeViT-XXS binary classifier | No — hybrid ViT candidate |

PatchCore (ResNet18 anomaly detector) available as optional offline baseline only — no TFLite path.

**Neural config:** BCEWithLogitsLoss · AdamW(lr=1e-4, wd=1e-4) · CosineAnnealingLR(T_max=15, eta_min=1e-6) · 15 epochs · batch 64 · patience=5 on val F1 · grad clip max_norm=5.0 · safe_collate.

**Threshold selection:** Geometric mean maximiser of recall × specificity, constrained to maize recall ≥ 0.95, hard cap at 0.70. Specificity-alone maximisation was abandoned — it produced near-1.0 thresholds and >88% rejection rates.

### 4.5 Tier 1 Sampling — `sample_15000.py` (Phase 1)

Draws 15,000 images from global train+val split only. Test-split images never included.

| Class | Count | Method |
|---|---|---|
| HEALTHY | 3,000 | `random.sample()` seeded at 42 |
| MSV | 7,500 | Evenly-spaced indices across sorted filenames |
| MLN | 4,500 | Evenly-spaced indices |

SHA-256 of `global_split_manifest.csv` locked to `data/tier1_raw/_manifest_hash.txt` on first run. Aborts if manifest changes.

### 4.6 Gold Standard Export — `sample_gold_standard.py` (Phase 1b)

Exports **501 images (167 per class)** from Tier 1 for human polygon annotation. Files renamed `{CLASS}_{original}.jpg`. SHA-256 of `tier1_manifest.csv` locked on first run.

**This single script and its single `annotations.json` serve three downstream consumers:**
1. `train_yolo_detector.py` — reads polygons, derives bounding boxes automatically (min/max of polygon x,y coordinates), trains YOLOv8n leaf detector
2. `validate_gold_standard.py` — reads polygons, rasterizes to binary masks via `cv2.fillPoly()`, computes IoU chain
3. `train_yolo_detector.py` (calibration step) — runs YOLO+SAM2 on gold images to calibrate SAM2 QA confidence threshold

**Label Studio annotation workflow (two passes, same 501 images):**

Pass 1 — Leaf silhouette: polygonlabels task, trace full leaf outline. Export JSON → `data/gold_standard/annotations/annotations.json`. Do this before running `train_yolo_detector.py`.

Pass 2 — Symptom regions (Phase 3b): return to same project, trace disease areas only (chlorotic streaks for MSV, necrotic patches for MLN), label = `"symptom"`. HEALTHY images: skip. Target ≥ 400 MSV+MLN images. Export JSON → `data/gold_standard/annotations/symptom_annotations.json`. Do this before running `train_symptom_model.py`.

> **Note:** `sample_yolo_annotations.py` is deprecated — there is no longer a separate YOLO annotation export step in the execution order, even if the file itself is still present on disk. `YOLO_IMAGES_DIR`, `YOLO_ANNOTATIONS_DIR`, and `YOLO_ANNOTATION_FILE` in `config.py` all now point to `data/gold_standard/` paths.

### 4.7 YOLO Leaf Detector — `train_yolo_detector.py` (Phase 1c)

Trains YOLOv8n (nano, single-class) by converting the gold standard polygon annotations to bounding boxes. Must complete before `generate_tier1_masks.py`.

Pure HSV prompting drifts onto background on diseased images — MSV yellow and MLN brown share hue ranges with tropical soil. A YOLO bounding box constrains SAM2's segmentation region, eliminating this dominant v2 failure mode.

Target: mAP@0.5 ≥ 0.70. Warns if annotation count < 400. After training, calibrates the SAM2 QA confidence threshold against gold-standard IoU (target mean IoU ≥ 0.85). Calibration written to `logs/yolo_qa_calibration.csv`.

### 4.8 SAM2 Masking — `generate_tier1_masks.py` (Phase 2, v3)

**Prompting strategy (v3):** Run YOLOv8n → tight leaf bounding box → restrict HSV tissue search to box interior → 3 foreground points along vertical leaf axis clamped to box → pass box to SAM2 as hard spatial constraint. HSV fallback if YOLO absent or fails. Four image corners always provided as background prompts (label=0). Select highest-scoring mask → sigmoid → float32 .npy.

**QA filters (actual runtime values):** min coverage 2% · high-coverage gate (≥ 97% coverage requires ≥ 0.80 confidence) · min mean confidence 0.65 default (replaced by calibrated value from `yolo_qa_calibration.csv`) · aspect ratio ≥ 1.00 (effectively disabled). Target < 8% rejection rate.

### 4.9 Teacher — `train_teacher.py` (Phase 3)

Offline segmentation model. Never deployed. Trains on SAM2 float32 probability maps using a custom `BoundaryAwareLoss` that sharpens soft targets (factor 1.3) and up-weights boundary pixels (5.0×) in the BCE term.

**4 variants:** resnet50 (UNet) · **efficientnet-b2** (UNet, deployed) · mit_b2 (UNet) · deeplabv3plus-eb2 (DeepLabV3+). Input 768×768 · AdamW(lr=5e-5, wd=5e-4) · ReduceLROnPlateau · early stop patience=7 on val Dice · grad clip max_norm=5.0 · safe_collate. Test-split Tier 1 images excluded from Teacher training.

### 4.10 Symptom Teacher — `train_symptom_model.py` (Phase 3b)

Replaces the legacy LAB/HSV symptom pipeline. Eight rounds of threshold tuning (v1→v8) hit a structural ceiling — static colour rules cannot separate early chlorosis from healthy yellow-maize tissue.

**HealthyAE:** Convolutional autoencoder with true bottleneck (no skip connections), trained on HEALTHY images only. At inference, per-pixel reconstruction error normalized to [0,1] becomes the 4th input channel for the Symptom Teacher. Input 256×256 · batch 16 · 40 epochs · Adam(lr=1e-3) · MSE loss.

**Symptom Teacher:** `smp.Unet(encoder_name="efficientnet-b2", in_channels=4)`. Input = RGB + AE error map. Ground truth = human polygon masks (AE output is never a label). Input 512×512 · batch 4 · 60 epochs · AdamW(lr=1e-4) · 0.5×Dice + 0.5×BCE loss · target IoU 0.70.

Config toggles: `SYMPTOM_TEACHER_DEPLOYED = True` enables in Factory. `FACTORY_SYMPTOM_COMPARE_LAB = True` logs side-by-side IoU for thesis comparison. Legacy LAB fallback activates automatically if checkpoints are missing.

### 4.11 Factory — `factory_master.py` (Phase 4)

Processes ~215k train+val Tier 2 images, writing pseudo-labels into 4 mode subfolders simultaneously. Per-image: Bouncer gate → leaf silhouette (SAM2 .npy for Tier 1, Teacher inference for Tier 2) → silhouette refinement (threshold 0.35, morphological close+open) → coverage → reliability weight → symptom masking (Symptom Teacher or LAB fallback) → severity % → CIMMYT grade.

**4 modes — silhouette × symptom:**

| Mode | Silhouette | Symptom | Subfolder |
|---|---|---|---|
| A | Otsu | Hard binary | `mode_a/` |
| B | SAM2 binary | Hard binary | `mode_b/` |
| C | SAM2 soft float | Hard binary | `mode_c/` |
| D | SAM2 soft float | Soft confidence | `mode_d/` |

Per-image outputs per mode: `{stem}_silhouette.npy`, `{stem}_symptom.png` or `.npy`, `{stem}_sev.txt`, `{stem}_grade.txt`, `{stem}_weight.txt`. All downscaled to 224×224 at write time.

### 4.11b Factory QA — `validate_factory.py` (Phase 4b)

Optional but recommended visual audit, run right after `factory_master.py` and before
committing to a full `train_student.py` run. Samples N images per class (default 10),
pairs each raw image with its pseudo-mask overlay, and writes a self-contained HTML report
(`reports/validate_factory_{mode}.html`, or `_all_modes.html` with `--all-modes`) with
per-image symptom-coverage stats and an automatic per-class flag: HEALTHY masks should be
near-empty (>~4% fill is suspect), MSV should show narrow vein-parallel streaks, MLN should
show broader diffuse yellowing/necrosis from the leaf margins. The script's docstring also
maps each visual failure pattern to the exact `factory_master.py`/`config.py` parameter to
tune (e.g. `LAB_MSV_B_MIN`, `GABOR_THRESHOLD`). CLI: `--n`, `--mode`, `--all-modes`, `--img`,
`--borderline` (sample near-threshold severity cases), `--out`.

### 4.12 Student — `train_student.py` (Phase 5)

The deployed model. Shared encoder → three heads: segmentation (Ch0=silhouette, Ch1=symptom), classification (HEALTHY/MSV/MLN), severity (ReLU+clamp → ×100).

**5 encoder variants:** V1 mobilenet_v2 · **V2 mobilenet_v2_cbam** (expected winner) · V3 mobilenet_v3_small · V4 efficientnet_b0 · V6 efficientnet_b0_cbam. V5 (MobileViT-XXS) removed — TFLite einsum incompatibility.

**CBAM:** Channel attention (GAP+GMP → shared MLP → sigmoid) + spatial attention (7×7 conv → sigmoid) at each UNet skip connection. Implemented as a proper `smp.UnetDecoder` subclass — not monkey-patching — preserving `torch.save()` serialization.

**Loss:** DiceLoss (float32 soft targets) + asymmetric label-smoothing cross-entropy (HEALTHY→[0.90,0.08,0.02], MSV→[0.05,0.90,0.05], MLN→[0.02,0.05,0.93]) + reliability-weighted MSE for severity + homoscedastic uncertainty loss with 3 learnable log-variance params s1/s2/s3 (Kendall et al. 2018).

**Two-phase training:** Phase 1 frozen encoder (30 epochs, lr=1e-3, patience=10) → Phase 2 full fine-tune (20 epochs, lr=1e-4, patience=5). Optimizer re-initialized at transition. Grad clip max_norm=5.0 throughout. WeightedRandomSampler for class balance.

**Checkpoint criterion (quality composite):** `0.50×mIoU + 0.35×MSV_F1 + 0.15×(1−sev_mae/100)`

**Two-stage ablation:** Stage 1 = 5 encoders × Mode B → `checkpoints/student/stage1/`. Stage 2 = best encoder × 4 modes → `checkpoints/student/` (Mode B reused from Stage 1). Total unique runs: 8.

**Mobile composite (select_best_pipeline.py):** `0.38×MSV_F1 + 0.22×mIoU + 0.22×clamp(150ms/lat,0,1) + 0.10×(1−sev_mae/100) + 0.08×clamp(15MB/size,0,1)`

### 4.13 Gold Standard Validation — `validate_gold_standard.py`

Validates pseudo-label quality against the 501 human-annotated leaf silhouettes on the same images across the entire chain:

```
SAM2 masks → Teacher predictions → Student predictions
     ↓               ↓                    ↓
IoU vs human    IoU vs human        IoU vs human
```

Parser: Label Studio JSON polygonlabels → `cv2.fillPoly()` → binary mask. Multiple polygons merged with logical OR. Warn threshold: IoU < 0.75 per image. Target: mean IoU ≥ 0.85. Run three times: `--sam2-only` after Phase 2, full after Phase 3, full after Phase 5b.

### 4.14 XAI — `evaluate_xai.py`

Three methods (pytorch-grad-cam) on the last encoder conv block: Grad-CAM (baseline), **Grad-CAM++** (deployed — better for multi-region MSV streaks), Score-CAM (gradient-free reference). Metrics: pointing game accuracy, Insertion AUC, Deletion AUC (n_steps=6, GPU).

**Two distinct app outputs — never conflate:**
- Symptom boundary (green contour) — UNet Ch1, pixel-level
- Diagnostic attention (amber heatmap) — Grad-CAM++, coarse ~7×7 upsampled

### 4.15 Best Pipeline Selection — `select_best_pipeline.py`

Selects: Bouncer by max specificity (maize recall ≥ 0.95); Teacher by max val Dice; Student by quality composite during training, mobile composite for deployment. Promotes to `checkpoints/final/`. Auto-updates `STUDENT_BEST_VARIANT` and `STUDENT_FACTORY_MODE` in `config.py` via regex.

### 4.16 Severity Reliability — `evaluate_severity.py`

60 images (20/class), 2 raters, 0–3 scale. Cohen's Kappa (inter-rater) + Spearman ρ (human vs pseudo-label severity). Target ρ ≥ 0.60. Severity MAE measures consistency with pseudo-labels, not expert agronomic ratings — stated limitation in Chapter 4.

### 4.17 Export and Deployment

Path: PyTorch → ONNX (opset 12) → TF SavedModel → TFLite FP16. Student I/O: input `[1,3,224,224]` NCHW ImageNet normalized; output 0 seg `[1,2,224,224]` logits; output 1 cls `[1,3]` logits; output 2 sev `[1,1]` ∈[0,1]. Bouncer I/O: same input; output `[1,1]` logit. Android: API 21+, optional GPU delegate. Bundle at `exports/deploy/`: both TFLite models + `model_metadata.json` + `DEPLOYMENT_README.md` + `deployment_report.csv`.

### 4.18 Charts and Report

**`generate_charts.py`** — Standalone, reads `logs/*.csv` only, safe mid-training, dark theme. Flags: `--bouncer`, `--teacher`, `--student`, or none for all.

**`generate_report.py`** — `reports/evaluation_report.html`, 100% self-contained, 11 sections: Pipeline overview → Preprocessing → Bouncer → Teacher → Student Stage 1 → Student Stage 2 → Best Student → XAI → Severity → Deployment → Training curves.

---

## 5. Complete Metrics

**Bouncer:** Accuracy, Precision, Maize Recall, Maize F1, Specificity (primary), ROC-AUC, TP/FP/TN/FN, threshold, admission rate.

**Teacher:** Val Dice (primary), Val IoU, Val Recall, Val Precision, Val Specificity per epoch. CPU latency ms/image. Test eval on Tier 1 test-split images.

**Symptom Teacher:** Val Dice, Val IoU per epoch. `--compare-lab` produces side-by-side IoU comparison.

**Factory:** Per mode × per class: mean/std/median severity, % symptomatic, exclusion rate, mean silhouette confidence (modes C/D).

**Student (test split):** Seg Ch0+Ch1: mIoU, Dice, Recall, Precision, Specificity. Classification per class: P/R/F1; overall: Accuracy, Macro F1, Weighted F1, MCC. Severity: MAE%, RMSE%, R². Composite. CPU latency mean/std/FPS. All segmentation metrics use global TP/FP/FN accumulation — not per-batch averaging.

**XAI:** Pointing game accuracy, Insertion AUC, Deletion AUC per method per class.

**Severity reliability:** Cohen's Kappa, Spearman ρ.

---

## 6. Key Design Decisions

| Decision | Why |
|---|---|
| "Pseudo-label semi-supervised learning" not "knowledge distillation" | Teacher generates hard mask targets + severity floats. True KD requires soft logit outputs. Wrong term must not appear in the thesis. |
| SAM2 soft probability maps as Teacher targets | Preserves boundary uncertainty at edge pixels (0.3–0.7). Acts as segmentation label smoothing. |
| BoundaryAwareLoss for Teacher | Sharpens soft SAM2 targets (1.3×) and up-weights boundary pixels (5.0×) — forces precise edge learning over interior blob accuracy. |
| Single gold standard export (501 images) for both YOLO and IoU validation | `train_yolo_detector.py` derives bounding boxes from polygons automatically. One annotation task, one `annotations.json`, three consumers. `sample_yolo_annotations.py` is deprecated/unused. |
| Symptom Teacher replaces LAB/HSV | Eight rounds of tuning (v1→v8) confirmed a structural ceiling. Human-supervised model learns the decision boundary directly. |
| HealthyAE as 4th input channel | Lighting-invariant anomaly prior. AE is an input feature, not a label source. |
| Asymmetric label smoothing | Early MSV visually identical to HEALTHY (Cruz et al. 2024). Pathology-informed prior prevents overconfidence. |
| Homoscedastic uncertainty loss | Three learnable log-variance params auto-balance three task losses (Kendall et al. 2018). Replaces manual 0.6/0.2/0.2. |
| ReLU+clamp severity head | Sigmoid never reaches exactly 0.0. HEALTHY leaves always show nonzero severity with sigmoid — biologically incorrect. |
| Global mIoU not mean-of-batches | Per-batch averaging is statistically wrong when batch class distributions vary. |
| CBAM as proper subclass | Monkey-patching breaks `torch.save()` serialization. Explicit subclass with its own `forward()` is fully serializable. |
| Stage 1 checkpoint isolation | Stage 2 Mode B would overwrite the Stage 1 checkpoint for the same encoder+mode. Saving to `stage1/` prevents this. |
| Two-composite selection | Quality composite saves checkpoints during training (TFLite size/latency unknown). Mobile composite picks deployment model post-training. |
| Geometric mean Bouncer threshold | Specificity-alone caused near-1.0 thresholds (>88% rejection). Geometric mean of recall × specificity is balanced. Hard cap at 0.70. |

---

## 7. Terminology Corrections

| Wrong | Correct |
|---|---|
| "Knowledge distillation" | "Pseudo-label semi-supervised learning" or "teacher-guided pseudo-label generation" |
| "Pixel-level segmentation via Grad-CAM" | Grad-CAM++ = coarse ~7×7 class-discriminative heatmap. UNet head = pixel-level segmentation. Entirely distinct outputs. |
| "Severity MAE measures disease severity accuracy" | "Severity MAE measures consistency with Factory's pseudo-labels (Symptom Teacher or HSV-derived), not expert agronomic ratings" |

---

## 8. File Inventory (22 pipeline scripts + config/utils + 2 shared utilities)

```
yellowmaize/
├── config.py                   All hyperparameters — YOLO_* paths now point to gold_standard/
├── image_utils.py
├── requirements.txt
├── partition_dataset.py
├── create_bouncer_dataset.py
├── train_bouncer.py
├── sample_15000.py
├── sample_gold_standard.py     501 images (167/class) — serves YOLO + IoU validation + SAM2 QA
├── train_yolo_detector.py      Polygon→bbox conversion + YOLOv8n training + QA calibration
├── generate_tier1_masks.py
├── validate_masks.py
├── validate_gold_standard.py
├── train_teacher.py
├── train_symptom_model.py
├── factory_master.py
├── validate_factory.py         Visual HTML QA of pseudo-masks (Phase 4b, optional)
├── train_student.py
├── generate_charts.py
├── select_best_pipeline.py
├── evaluate_xai.py
├── evaluate_severity.py
├── export_tflite.py
├── build_deployment_package.py
├── generate_report.py
├── __init__.py
└── scripts/
    ├── __init__.py
    ├── bouncer_inference.py
    └── safe_collate.py
```

> `sample_yolo_annotations.py` is deprecated and unused — not part of the execution order.
> A copy may still be present on disk as legacy cruft (safe to delete). YOLO training reads
> bounding boxes from the same `annotations.json` produced by `sample_gold_standard.py`, and
> the `data/yolo_annotations/` directory is not used. `YOLO_IMAGES_DIR`, `YOLO_ANNOTATIONS_DIR`,
> and `YOLO_ANNOTATION_FILE` in `config.py` all point to `data/gold_standard/` paths.

---

## 9. Datasets to Download

| Dataset | Path | Source |
|---|---|---|
| Maize Tanzania (Mduma 2023) + Zenodo | `maize_dataset/HEALTHY/`, `MSV/`, `MLN/` | Mendeley Data + Zenodo |
| Intel Image Classification | `dataset/raw_kaggle/intel/` | Kaggle: puneet6060 |
| Natural Images | `dataset/raw_kaggle/natural/` | Kaggle: prasunroy |
| PlantVillage (rice, sorghum) | `dataset/crop_neighbors/rice/`, `sorghum/` | Kaggle: emmarex/plantdisease |
| iNaturalist Philippines | `dataset/crop_neighbors/cogon_grass/`, `banana_leaf/`, `sugarcane/` | iNaturalist API |
| Mendeley Maize-Weed | `dataset/crop_neighbors/` | Espejo-Garcia et al. 2020 |
| SAM2 weights | `sam2/sam2_hiera_large.pt` | https://dl.fbaipublicfiles.com/segment_anything_v2/sam2_hiera_large.pt |

---

## 10. Limitations for Chapter 5

1. **Geographic domain gap** — Tanzania data, Philippine deployment. Field validation needed.
2. **Severity ground truth** — Pseudo-labels from Symptom Teacher or HSV, not expert CIMMYT ratings.
3. **Staged ablation** — Not fully crossed encoder × mode factorial (compute budget). Stated in Chapters 3 and 5.
4. **Tier 1 size** — 15k Teacher training images relative to ~215k Factory images.
5. **Android only** — iOS excluded. Stated in scope.
6. **No Philippine field images** — All images from Tanzania + Zenodo.
7. **MobileViT-XXS excluded** — TFLite einsum incompatibility. Published benchmarks cited in related work.
8. **Symptom Teacher annotation burden** — ≥ 400 MSV+MLN symptom annotations required. Non-agronomist annotator quality is a potential bias source.

---

## 11. Key Citations

| Reference | Used for |
|---|---|
| Kendall et al. (2018) NeurIPS | Homoscedastic uncertainty multi-task loss |
| Szegedy et al. (2016) CVPR | Label smoothing |
| Woo et al. (2018) ECCV | CBAM |
| Ke et al. (2020) | Soft pseudo-label segmentation targets |
| Jiang et al. (2018) ICML | MentorNet — reliability-weighted curriculum |
| Cruz et al. (2024) | First confirmed MSV in the Philippines |
| Mushayi et al. (2025) | MSV / HEALTHY visual confusion — asymmetric prior basis |
| Selvaraju et al. (2017) ICCV | Grad-CAM |
| Chattopadhay et al. (2018) WACV | Grad-CAM++ |
| Wang et al. (2020) CVPR | Score-CAM |
| Mduma (2023) Mendeley | Tanzania Maize Imagery Dataset |
| Tan & Le (2019) ICML | EfficientNet |
| Xie et al. (2021) NeurIPS | SegFormer (mit_b2) |
| Pan et al. (2022) ECCV | EdgeViT |
| Fawcett (2006) | ROC threshold selection |
| Zuiderveld (1994) | CLAHE |
