# Yellow MAIze — Complete Project Summary
## For continuing conversation in a new chat

---

## 1. Project Overview

**Title:** MAIze: A U-Net with MobileNetV2 and Explainable AI Framework for Maize Streak Virus Identification and Symptom Segmentation in Zea mays L.

**Institution:** Angeles University Foundation, College of Computer Studies, BSCS 3-A

**Team:** Altes, Zylah Klein · Davis, Dominic · Tayer, Catherine P. · Ursua, Walter Vince

**Goal:** An Android mobile application that detects Maize Streak Virus (MSV) and Maize Lethal Necrosis (MLN) in yellow corn leaves using a multi-task deep learning pipeline, with Explainable AI (Grad-CAM++) for transparent diagnosis.

**Clinical motivation:** MSV was first detected in the Philippines (Bukidnon, South Cotabato) in 2023. Early symptoms are visually indistinguishable from nutrient deficiencies. The tool targets agriculture students and smallholder farmers.

**Dataset:** Maize Imagery Dataset — Tanzania (Mduma, 2023) + Zenodo healthy/MSV samples. ~252k images. Three classes: HEALTHY (~38%), MSV (~24%), MLN (~38%). Geographic mismatch (Tanzania data, Philippine deployment) is a stated limitation.

---

## 2. Hardware & Software

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 (8 GB VRAM) |
| CPU | AMD Ryzen 5 3600X |
| RAM | 16 GB DDR4 |
| OS | Windows 11 + WSL2 (Ubuntu) |
| Runtime | WSL2 — all Python scripts run here |
| NUM_WORKERS | 4 (WSL2 /dev/shm limits) |
| Primary framework | PyTorch + torchvision |
| TFLite export | TensorFlow (conversion only, not training) |
| Global seed | 42 (all scripts) |
| cudnn | deterministic=True, benchmark=False |

---

## 3. Complete Pipeline Architecture

### Phase Overview

```
Phase 0-pre  partition_dataset.py           → global_split_manifest.csv (70/15/15)
Phase 0a     create_bouncer_dataset.py       → 50k binary bouncer dataset
Phase 0b     train_bouncer.py               → bouncer_best.pth
Phase 1      sample_15000.py                → 15k Tier 1 images
Phase 1a     sample_yolo_annotations.py     → 501 images for YOLO bbox annotation
Phase 1b     train_yolo_detector.py         → checkpoints/yolo/best.pt + QA calibration
Phase 1b     sample_gold_standard.py        → 501-image gold standard set for human annotation
             train_yolo_detector.py         → YOLOv8n leaf detector + SAM2 QA calibration
Phase 2      generate_tier1_masks.py        → SAM2 float .npy masks + QA  [v3: YOLO-guided]
Phase 2b     validate_masks.py              → QA report + overlays
Phase 2c     validate_gold_standard.py      → SAM2 IoU vs human masks (--sam2-only)
Phase 3      train_teacher.py               → teacher_model_best.pth
Phase 2e     validate_gold_standard.py      → SAM2 + Teacher IoU vs human masks (full run)
Phase 4      factory_master.py              → pseudo_masks/ (4 modes)
Phase 5a     train_student.py --stage 1     → encoder ablation (5 variants, Mode B)
Phase 5b     train_student.py --stage 2     → mode ablation (best encoder, 4 modes)
             generate_charts.py             → reports/charts/ (run any time after training)
Phase 5c     validate_gold_standard.py      → full chain: SAM2→Teacher→Student IoU (final thesis table)
Phase 6      select_best_pipeline.py        → checkpoints/final/ + summary
Phase 7      evaluate_xai.py                → 3-way XAI comparison
Phase 8      evaluate_severity.py           → inter-rater reliability
Phase 9      export_tflite.py               → student_model.tflite
Phase 10     build_deployment_package.py    → exports/deploy/ (both TFLite models)
Phase 11     generate_report.py             → evaluation_report.html
```

### Execution order (run in this order)

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2

python partition_dataset.py
python create_bouncer_dataset.py
python train_bouncer.py
python generate_charts.py --bouncer      # bouncer comparison + training curves
python sample_15000.py
python sample_yolo_annotations.py       # export 501 images for YOLO bbox annotation
                                         # → import data/yolo_annotations/images/ into Label Studio
                                         # → draw one bbox per image (label: "leaf")
                                         # → export YOLO format → data/yolo_annotations/labels/
python train_yolo_detector.py           # train YOLOv8n + calibrate SAM2 QA threshold
python sample_gold_standard.py       # extract 501-image gold standard set
                                     # → images auto-copied to data/gold_standard/images/
                                     # → upload that folder to Label Studio
                                     # → annotate at least 400 leaf silhouettes (polygonlabels)
                                     # → export JSON → data/gold_standard/annotations/annotations.json
python train_yolo_detector.py        # train YOLOv8n leaf detector + calibrate SAM2 QA threshold
python generate_tier1_masks.py       # SAM2 masking v3 (YOLO-guided prompts)
python validate_masks.py             # review overlays before Teacher training
python validate_gold_standard.py --sam2-only   # SAM2 IoU vs human (before Teacher)
python train_teacher.py
python generate_charts.py --teacher      # teacher comparison + training curves
python validate_gold_standard.py         # SAM2 + Teacher IoU chain vs human
python factory_master.py
python train_student.py --stage 1
python train_student.py --stage 2 --encoder mobilenet_v2_cbam
python generate_charts.py --student      # all student charts
# or regenerate everything at once:
python generate_charts.py
python validate_gold_standard.py         # final chain: SAM2→Teacher→Student (thesis table)
python select_best_pipeline.py       # auto-updates config.py
python evaluate_xai.py
python evaluate_severity.py --sample # fill rater scores, then:
python evaluate_severity.py --analyze
python export_tflite.py
python build_deployment_package.py
python generate_report.py
```

---

## 4. Component Specifications

### 4.1 Preprocessing (partition_dataset.py)

10-step pipeline run once before any training:

1. Zero-byte / tiny file check (< 100 bytes → reject)
2. Magic bytes / format mismatch (JPEG: FF D8 FF, PNG: 89 PNG)
3. Truncated image detection (PIL `.load()` forces full decode)
4. Minimum resolution check (< 64px → reject)
5. Maximum resolution / aspect ratio (> 4096px or > 8:1 → reject)
6. Colour mode check (1-bit binary → reject; grayscale → flag)
7. Near-uniform / solid colour (pixel std < 5.0 → reject)
8. MD5 exact duplicate removal (across ALL classes simultaneously)
9. pHash near-duplicate removal (Hamming ≤ 2, within class)
10. Cross-class pHash duplicate detection (same image, different labels → remove both)

Additionally: low-green-content flag (< 5% green pixels → flagged, kept).
Outputs: `preprocessing_report.csv`, `preprocessing_flagged.csv`, `preprocessing_summary.txt`

**Image loading:** All scripts use `image_utils.py` which applies:
- EXIF orientation correction via `ImageOps.exif_transpose()` (PIL and cv2 see same orientation)
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for Bouncer and Factory inputs
- Channel order safety: all HSV conversions via `to_hsv()`, always pass RGB

### 4.2 Dataset Split

- Global 70/15/15 stratified split via `global_split_manifest.csv`
- Single source of truth — ALL downstream scripts read this manifest
- Stratification preserves class ratios: HEALTHY ~38%, MLN ~38%, MSV ~24%
- Test split (~37,934 images) is held out and never touched until final evaluation
- Bouncer positive class filtered to train+val split only (test images excluded)

### 4.3 Bouncer (Phase 0)

**Purpose:** Binary gate — maize vs non-maize. Runs first on every camera frame.

**Two-stage architecture:**
- Stage 1: OpenCV heuristic pre-filter (green coverage ≥ 15%, aspect ratio 1.5–18.0) — microseconds
- Stage 2: Neural binary classifier

**Training dataset:** 25k maize (train+val split only) + 25k non-maize (50k total)

**Non-maize sources:**
- Intel Image Classification (scenes, objects) — Kaggle
- Natural Images dataset — Kaggle
- Philippine crop neighbors: cogon grass (iNaturalist), banana leaf (iNaturalist), rice, sorghum (PlantVillage), sugarcane (iNaturalist)
- Mendeley Maize-Weed Dataset

**4 comparison variants:**

| Variant | Type | Role |
|---|---|---|
| Gabor + LBP | Traditional CV (LinearSVC) | Baseline A |
| **MobileNetV2** | Binary CNN (thesis primary arch) | Comparison B |
| **MobileNetV3-Large** | Binary CNN | **Deployed** |
| EdgeViT-XXS | Hybrid ViT binary CNN | Comparison C |

> PatchCore (ResNet18 anomaly detector) is available as an optional offline
> evaluation only — excluded from trained variants; no TFLite deployment path.

**Hyperparameters (neural variants):**
- Architecture: MobileNetV3-Large (ImageNet pretrained), head: Linear(960→1)
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=15, eta_min=1e-6)
- Epochs: 15, Batch: 64, Val split: 80/20 (internal)
- Early stopping: patience=5 (by best val F1)
- Threshold: empirically selected from ROC curve (max specificity s.t. maize recall ≥ 95%)

**Primary metric:** Specificity (TN rate — non-maize rejection rate)

**Outputs:**
- `checkpoints/bouncer/bouncer_{variant}_best.pth`
- `logs/bouncer_comparison.csv` (ROC-AUC, specificity, maize recall, TP/FP/TN/FN)
- `logs/bouncer_admission_rate_{variant}.csv` (end-to-end maize admission rate on test split)

### 4.4 Tier 1 Sampling, YOLO Detector & SAM2 Masking (Phases 1–2)

**Tier 1 composition:** 15,000 images total
- HEALTHY: 3,000 (pure random)
- MSV: 7,500 (evenly-spaced across sorted filenames — diversity proxy)
- MLN: 4,500 (evenly-spaced)
- Source: train+val split images only (test excluded)

**Phase 1a: YOLO Annotation Sampler** (`sample_yolo_annotations.py`)

Extracts a reproducible stratified sample of **501 images** (167 per class) from
Tier 1 for YOLO bounding-box annotation. Images are renamed with class prefix (e.g.
`MSV_image045.jpg`) and exported to `data/yolo_annotations/images/`. A SHA-256
hash lock guards against re-sampling if `tier1_manifest.csv` changes post-annotation.

**Phase 1b: Gold Standard Sampling + YOLO Detector** (`sample_gold_standard.py`, `train_yolo_detector.py`)

`sample_gold_standard.py` — Extracts 501 images (167 per class) from Tier 1 for human annotation in Label Studio.
Run immediately after `sample_15000.py`. Annotate at least 400 of the 501 images as polygon labels
before training YOLO.

`train_yolo_detector.py` — Single-class YOLOv8n (nano) trained on Label Studio polygon annotations.
Provides tight bounding-box prompts for SAM2. Polygon → bbox conversion is
automatic. After training, the script runs the full YOLO+SAM2 pipeline on the
gold-standard images and calibrates the QA confidence threshold to achieve
mean IoU ≥ `GOLD_IOU_TARGET_MEAN` (0.85). Written to `logs/yolo_qa_calibration.csv`.

**SAM2 auto-prompting strategy (v3 — YOLO-guided):**
1. Run YOLOv8n → tight leaf bounding box (if `checkpoints/yolo/best.pt` exists)
2. Restrict HSV green+yellow tissue detection to pixels inside the YOLO box
3. Three foreground points along the leaf’s vertical axis (clamped to box)
4. Pass YOLO box as SAM2 `box=` prompt — hard spatial constraint
5. Fallback to v2 full-image HSV centroid if YOLO absent or detection fails

**QA filters (v3 relaxed thresholds):**
1. Coverage range: foreground 3%–90% (v1 was 10%)
2. Mean foreground confidence ≥ calibrated threshold (default 0.65)
3. Mask aspect ratio ≥ 1.01 (v1 was 1.20)
- Target rejection rate: < 8%
- QA report now includes `prompt_mode` and `yolo_box` columns

### 4.5 Teacher Model (Phase 3)

**Purpose:** Offline-only leaf silhouette segmentation model. Never deployed to Android. Sole purpose: generate leaf silhouette pseudo-masks for ~215k Tier 2 images.

**Training:** Soft float32 SAM2 probability maps as targets (NO binarization — preserves boundary uncertainty). BCEWithLogitsLoss accepts float targets natively.

**4 comparison variants:**

| Variant | Type | Params |
|---|---|---|
| ResNet-50 + UNet | Deep CNN baseline | ~32M |
| **EfficientNet-B2 + UNet** | Efficient CNN (recommended) | ~7.7M |
| SegFormer-B2 (mit_b2) via smp | Hierarchical ViT | ~25M |
| EfficientNet-B2 + DeepLabV3+ | Decoder comparison (ASPP) | ~7.7M |

**Hyperparameters:**
- Input: 512×512 (larger than Student for boundary detail)
- Batch: 8, Epochs: 30
- Optimizer: AdamW (lr=5e-5, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5, mode=max)
- Loss: Dice Loss (binary, from_logits=True)
- Early stop: patience=10 (monitors val Dice)
- Checkpoint: best val Dice → `teacher_{variant}_best.pth`
- Best Teacher copied to canonical `teacher_model_best.pth`

**Augmentation (training only):**
Horizontal flip (p=0.5), Vertical flip (p=0.5), RandomRotate90 (p=0.5),
RandomBrightnessContrast (±0.2, p=0.3), HueSaturationValue (H±10, S±20, V±10, p=0.2),
ImageNet normalization

**Metrics:** Dice, IoU, Recall, Precision, Specificity + CPU inference latency

### 4.6 Factory (Phase 4)

**Purpose:** Generate pseudo-labels for all ~215k train+val Tier 2 images across 4 modes simultaneously.

**Processing stages per image:**
1. Bouncer gate — helpers imported from `scripts/bouncer_inference.py` (single source of truth shared with `train_bouncer.py`):
   - `heuristic_prefilter()` — currently a **passthrough** (always True); original green-coverage heuristic removed after false rejections on yellow/bleached MSV leaves
   - `neural_bouncer()` — MobileNetV3-Large, empirical threshold
2. Tier 1 check: if Tier 1 → load SAM2 .npy (skip Teacher); else → Teacher inference
3. Silhouette refinement: threshold at 0.35 + morphological close/open
4. Coverage guard → reliability weight assignment
5. HSV symptom masking (hard or soft depending on mode)
6. Severity computation = (symptom pixels / leaf pixels) × 100%

**4 Factory modes (written to separate subfolders):**

| Mode | Silhouette | Symptom mask | Subfolder |
|---|---|---|---|
| A | Otsu threshold | Hard binary HSV | mode_a/ |
| B | SAM2 hard binary | Hard binary HSV | mode_b/ |
| C | SAM2 soft float | Hard binary HSV | mode_c/ |
| D | SAM2 soft float | Soft HSV confidence map | mode_d/ |

**Reliability weights (replacing hard sentinel -1):**
- Coverage < 15%: exclude (weight = -1)
- 15–25%: weight = 0.30
- 25–50%: weight = 0.70
- > 50%: weight = 1.00

**MSV HSV ranges (4 bands, R3 has area filter ≥ 80px):**
- R1: H 15–38, S 50–255, V 150–255 (bright yellow streaks)
- R2: H 20–45, S 15–70, V 130–255 (pale yellow / early-stage)
- R3: H 0–179, S 0–35, V 210–255 (near-white / bleached — area filter applied)
- R4: H 38–55, S 10–55, V 140–255 (pale yellow-green)

**MLN HSV ranges (5 bands):**
- R1: H 18–40, S 40–255, V 90–255 (chlorotic yellow)
- R2: H 5–20, S 40–255, V 50–220 (orange-amber necrosis)
- R3: H 22–55, S 15–85, V 80–240 (pale yellow-green mosaic)
- R4: H 8–35, S 0–45, V 160–255 (tan / straw tissue)
- R5: H 0–18, S 30–180, V 30–150 (dark brown dead tissue)

**Green exclusion zone:** H 38–85, S 80–255, V 60–230 (removes healthy chlorophyll)

**Mode D soft HSV confidence formula:**
`confidence = (Σ range_hit × normalized_distance_to_center) / N_ranges`

**Output resolution:** All .npy and .png outputs are **downscaled to STUDENT_IMG_SIZE (224×224)** at write time to reduce storage overhead. Silhouette and mode_d symptom use `INTER_LINEAR`; binary symptom PNGs use `INTER_NEAREST` to preserve hard edges.

**Outputs per mode:** `{stem}_silhouette.npy`, `{stem}_symptom.npy/.png`, `{stem}_sev.txt`, `{stem}_weight.txt`

**Factory summary reports:**
- `reports/factory_summary.csv` — per-mode × per-class: mean/std/median severity, % symptomatic, exclusion rate, mean silhouette confidence (modes C/D)
- `reports/factory_filter_breakdown.csv` — counts per status (processed / filtered_bouncer / filtered_heuristic / load_error / etc.)

### 4.7 Student Model (Phase 5)

**The final deployed model.** Single shared encoder with three output heads.

**Architecture:**

```
Input (224×224×3)
    ↓
Shared Encoder (MobileNetV2 or variant)
    ├── UNet Decoder → Segmentation Head (2 channels)
    │     Ch0: leaf silhouette (float32 logits → sigmoid → binary mask)
    │     Ch1: symptom mask   (float32 logits → sigmoid → binary mask)
    ├── GAP → Dropout(0.3) → Linear(→3) → Classification Head
    │     HEALTHY=0 / MSV=1 / MLN=2
    └── GAP → Dropout(0.3) → Linear(→1) → ReLU → clamp(0,1) → Severity Head
          0.0–1.0 (×100 at inference = severity %)
```

**5 encoder variants (Stage 1 ablation on Mode B):**

| Variant | Encoder | Params | TFLite | Notes |
|---|---|---|---|---|
| V1 | MobileNetV2 | 3.4M | Yes | Baseline |
| **V2** | **MobileNetV2 + CBAM** | **~3.5M** | **Yes** | **Expected winner** |
| V3 | MobileNetV3-Small | 2.5M | Yes | Ultra-compact |
| V4 | EfficientNet-B0 | 5.3M | Yes | Compound scaling (no attention) |
| V6 | EfficientNet-B0 + CBAM | ~5.4M | Yes | CBAM generalization test |

> V5 (MobileViT-XXS) removed — `torch.einsum` self-attention causes TFLite subgraph
> errors incompatible with deployment. V6 replaces it.

**CBAM (Convolutional Block Attention Module, Woo et al. 2018):**
Applied at each UNet skip connection junction. Channel attention (GAP+GMP→MLP→sigmoid) + spatial attention (7×7 conv→sigmoid). Implemented as proper subclass of smp UnetDecoder (not monkey-patch — fully serialisable).

**Two-phase transfer learning:**

| Phase | Epochs | LR | Encoder | Patience |
|---|---|---|---|---|
| 1 — Frozen | 30 | 1e-3 | Frozen | 10 |
| 2 — Fine-tuning | 20 | 1e-4 (→1e-6) | Unfrozen | 5 |

Optimizer re-initialized at Phase 1→2. CosineAnnealingLR for both phases. Patience resets at transition.

**Loss functions:**

| Loss | Function | Notes |
|---|---|---|
| Segmentation (both channels) | Dice Loss (binary, from_logits=True) | Float targets — preserves soft boundary uncertainty |
| Classification | Asymmetric label smoothing cross-entropy | Pathology-informed prior matrix |
| Severity | Reliability-weighted MSE | Per-sample weight from Factory coverage |
| Multi-task balancing | Homoscedastic uncertainty loss (Kendall et al. 2018) | 3 learnable log-variance parameters |

**Asymmetric label smoothing prior matrix:**
```
HEALTHY → [0.90, 0.08, 0.02]  # early MSV looks like HEALTHY
MSV     → [0.05, 0.90, 0.05]
MLN     → [0.02, 0.05, 0.93]  # MLN is most visually distinct
```

**WeightedRandomSampler:** Ensures every batch has proportional class representation (MSV is minority at ~24%).

**Checkpoint criterion (composite):**
`0.50 × mIoU + 0.35 × MSV_F1 + 0.15 × (1 − NormMAE)`

**Gradient clipping:** `clip_grad_norm_(max_norm=5.0)` in all training scripts.

**Stage 2 ablation:** Best encoder trained on all 4 Factory modes. Mode B Stage 1 result reused (same seed=42). Stage 1 checkpoints saved to `checkpoints/student/stage1/` to prevent Stage 2 from overwriting.

**Augmentation (training only):**
Letterbox padding, HFlip (p=0.5), VFlip (p=0.5), RandomRotate90 (p=0.5), Rotate ±30° (p=0.5), RandomBrightnessContrast (±0.25, p=0.3), HueSaturationValue (H±15, S±20, V±10, p=0.2), **RandomShadow (p=0.2)** — tropical domain adaptation, ImageNet normalization

### 4.8 XAI (Phase 7)

**3 comparison methods (all via pytorch-grad-cam library):**

| Method | Type | Notes |
|---|---|---|
| Grad-CAM | Gradient-based | Historical baseline (Selvaraju et al. 2017) |
| **Grad-CAM++** | Gradient-based | **Deployed** — better for multi-region MSV streaks (Chattopadhay et al. 2018) |
| Score-CAM | Gradient-free | Stability comparison (Wang et al. 2020) |

**Target layer:** Last convolutional block of shared encoder (before decoder branches). Maps shared disease-relevant features driving both classification and segmentation.

**XAI target layers per encoder:**
- MobileNetV2 / MobileNetV2+CBAM: `encoder.features[-1][0]`
- MobileNetV3-Small: `encoder.features[-1][0]`
- EfficientNet-B0 / EfficientNet-B0+CBAM: `encoder.blocks[-1][-1]`

**Two DISTINCT app outputs (critical — not the same thing):**
1. **Symptom boundary** (green contour): UNet segmentation head → pixel-level localization
2. **Diagnostic attention** (amber heatmap): Grad-CAM++ → class-discriminative explanation

**Quantitative metrics:** Pointing game accuracy (top-20% heatmap pixels inside seg mask), Insertion AUC, Deletion AUC (n_steps=6, GPU).

### 4.9 Severity Reliability (Phase 8)

**Protocol:** 60 images (20 per class), 2 raters, 0–3 scale
- 0=no symptoms, 1=mild (<25%), 2=moderate (25–60%), 3=severe (>60%)
- Cohen's Kappa for inter-rater agreement
- Spearman ρ between human ratings and HSV-derived severity
- Target: ρ ≥ 0.60 = "moderate correlation"

**Important disclaimer:** Severity MAE measures Student consistency with HSV-derived pseudo-labels, NOT expert agronomic ratings. Must be stated explicitly in Chapter 4.

### 4.10 Best Pipeline Selection (Phase 6)

`select_best_pipeline.py`:
- Reads all comparison CSVs
- Selects: best Bouncer (max specificity s.t. recall ≥ 95%), best Teacher (max val Dice), best Student (max composite score)
- Copies winners to `checkpoints/final/bouncer_best.pth`, `teacher_best.pth`, `student_best.pth`
- Auto-updates `STUDENT_BEST_VARIANT` and `STUDENT_FACTORY_MODE` in `config.py`
- Writes `reports/best_pipeline_summary.csv`, `reports/best_pipeline_summary.txt`, `reports/all_variants_ranked.csv`

### 4.11 Deployment (Phases 9–10)

`export_tflite.py`: Student → ONNX → TFLite (FP16 quantized)
`build_deployment_package.py`: Bouncer → TFLite + full deploy bundle

**`exports/deploy/` contents:**
- `bouncer_model.tflite`
- `student_model.tflite`
- `model_metadata.json` (all tensor shapes, normalization, thresholds, class names, post-processing steps)
- `DEPLOYMENT_README.md` (Kotlin/Java code snippets for Android Studio)
- `deployment_report.csv`

**Android inference pipeline:**
```
Frame → EXIF correction → Heuristic pre-filter (green+aspect) →
Bouncer (224×224, sigmoid threshold) → if maize →
Student (224×224, 3 outputs) →
  seg: sigmoid → binary masks → contour overlay
  cls: softmax → argmax → class name
  sev: ×100 → severity % display
  + Grad-CAM++ heatmap overlay
```

### 4.12 HTML Report (Phase 11)

`generate_report.py` → `reports/evaluation_report.html`

Self-contained single HTML file (all charts base64 embedded). Dark-themed. 9 sections:
1. Preprocessing summary (rejection breakdown table)
2. Bouncer comparison (bar charts, confusion matrix, admission rate)
3. Teacher comparison (Dice bars, training curves, qualitative overlays)
4. Student encoder ablation (multi-metric comparison)
5. Student mode ablation (mode A/B/C/D performance + severity distributions)
6. Best Student full test results (all metrics, confusion matrix heatmap, training curves)
7. XAI comparison (pointing game / insertion / deletion charts + overlay images)
8. Severity reliability (Kappa + Spearman ρ)
9. Deployment summary (TFLite sizes, latency, deployment package contents)

### 4.13 Gold Standard Validation (Phases 2c–2e, 5c)

**Purpose:** Validate pseudo-label quality against 501 human-annotated leaf silhouette masks (167 per class). Provides a thesis-defensible chain comparison: SAM2 → Teacher → Student, all measured against the same human ground truth.

**sample_gold_standard.py:**
- Stratified sample: 167 HEALTHY + 167 MSV + 167 MLN from Tier 1 manifest (seed=42) — total 501 images
- Renames files with class prefix (e.g. `MSV_image045.jpg`) for annotator clarity
- Output: `data/gold_standard/images/` — upload directly to Label Studio for polygon annotation
- Target: annotate at least 400 of these 501 images before running `train_yolo_detector.py`
- Also writes `data/gold_standard/gold_manifest.csv` (column: `gold_filename`)
- Hash guard: SHA-256 hash of `tier1_manifest.csv` written to `_manifest_hash.txt` on first run.
  If `sample_15000.py` is re-run after annotation begins, subsequent runs abort with a clear
  error instead of silently producing a mismatched 300-image set.

**validate_gold_standard.py:**
- Parses Label Studio JSON export (polygonlabels, % coordinates → rasterized binary masks)
- Computes binary IoU per image for each of: SAM2 probability maps, Teacher predictions, Student predictions
- Usage: `--sam2-only` flag skips model loading (run before Teacher training)

**Config keys required:**
```
GOLD_IMAGES_DIR         = DATA_DIR / "gold_standard" / "images"
GOLD_MANIFEST           = DATA_DIR / "gold_standard" / "gold_manifest.csv"
GOLD_ANNOTATION_FILE    = DATA_DIR / "gold_standard" / "annotations" / "annotations.json"
GOLD_IOU_WARN_THRESHOLD = 0.75   ← per-image flag threshold
GOLD_IOU_TARGET_MEAN    = 0.85   ← overall validation target (updated from 0.80)
TEACHER_DEPLOYED_VARIANT = "efficientnet-b2"
```

**Run at three points:**
1. After `generate_tier1_masks.py` — `--sam2-only` to validate SAM2 foundation
2. After `train_teacher.py` — full run for SAM2 + Teacher chain
3. After `train_student.py` — full run for complete SAM2 → Teacher → Student chain (thesis table)

**Outputs:**
- `reports/gold_standard_iou_report.csv` — per-image IoU for all 3 artifacts
- `reports/gold_standard_iou_summary.csv` — mean ± std per class + overall
- `reports/gold_standard_overlays/` — 5 comparison overlay PNGs per class per artifact (Green=missed, Cyan=correct, Red=extra)

---

## 5. Evaluation Metrics (Complete List)

### Bouncer
Accuracy, Precision, Maize Recall, Maize F1, Specificity (TN rate — primary), ROC-AUC, TP/FP/TN/FN, empirical threshold, end-to-end admission rate on test split

### Teacher
Val Dice (primary), Val IoU, Val Recall, Val Precision, Val Specificity, loss per epoch, CPU inference latency (ms/image)

### Factory
Per-mode × per-class: mean/std/median severity, % symptomatic, exclusion rate, mean silhouette confidence (modes C/D), filter breakdown by stage

### Student (test split)
**Segmentation (silhouette Ch0):** mIoU, Dice, Recall, Precision, Specificity
**Segmentation (symptom Ch1):** mIoU, Dice, Recall, Precision, Specificity
**Classification (overall):** Accuracy, MCC, Macro F1, Weighted F1
**Classification (per class):** Precision, Recall, F1 for HEALTHY / MSV / MLN
**Severity regression:** MAE%, RMSE%, R²
**Composite score:** 0.50×mIoU + 0.35×MSV_F1 + 0.15×(1−NormMAE)
**Latency:** CPU mean ms, CPU std ms, CPU FPS
**Duration:** Evaluation time in seconds
**CSV outputs:** `student_test_metrics_{variant}_{mode}.csv`, `student_confusion_{variant}_{mode}.csv`

### XAI
Pointing game accuracy, Insertion AUC, Deletion AUC — per method per image

### Severity reliability
Cohen's Kappa (inter-rater), Spearman ρ (HSV vs human)

---

## 6. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Pseudo-label semi-supervised learning (NOT "knowledge distillation") | Teacher generates hard targets (masks + severity floats). True KD requires soft logit targets — not used here. |
| SAM2 soft probability maps as seg targets | Boundary uncertainty signal: pixels near leaf edge get values 0.3–0.7 rather than hard 0/1. Label smoothing for segmentation. |
| Asymmetric label smoothing | Early MSV visually identical to HEALTHY (Cruz et al. 2024). Pathology-informed prior prevents overconfidence on folder labels. |
| Homoscedastic uncertainty loss | Replaces manual 0.6/0.2/0.2 weights. Three learnable log-variance params auto-balance task losses. Kendall et al. 2018. |
| ReLU + clamp severity head | Sigmoid ceiling prevents exactly 0% severity. HEALTHY leaves always show nonzero otherwise. ReLU+clamp allows true 0. |
| Global mIoU (not mean-of-batches) | Per-batch IoU averaging is statistically wrong — batches with different class distributions produce different denominators. Global TP/FP/FN accumulation is standard. |
| Two-channel segmentation output | Ch0=leaf silhouette (where is the leaf?), Ch1=symptom mask (where is the disease?). Separate targets, separate metrics. |
| Composite checkpoint criterion | mIoU alone misses MSV classification quality. Weighted composite (50% IoU, 35% MSV F1, 15% 1−NormMAE) saves checkpoint best for the clinical task. |
| CBAM as proper subclass | Monkey-patching smp decoder breaks torch.save serialisation. Proper subclass with explicit method is fully serialisable. |
| Stage 1 checkpoint isolation | Stage 2 mode_b run would overwrite Stage 1 result for same encoder+mode. Isolation to stage1/ subfolder prevents this. |
| Empirical Bouncer threshold | Arbitrary 95% rule conflates sigmoid probability with "structural similarity". Empirical selection from ROC curve is methodologically correct. |
| Factory Tier 1 flag | Tier 1 images have higher-quality SAM2 silhouettes. Factory skips Teacher inference for them but still runs HSV severity. |

---

## 7. Terminology Corrections (Important for Thesis)

| Wrong term | Correct term |
|---|---|
| "Knowledge distillation" | "Pseudo-label semi-supervised learning" or "teacher-guided pseudo-label generation" |
| "Pixel-level symptom segmentation via Grad-CAM" | Grad-CAM++ provides coarse class-discriminative localization (~7×7 upsampled). UNet head provides pixel-level localization. These are different outputs. |
| "Severity MAE measures disease severity accuracy" | "Severity MAE measures consistency with Factory's HSV-derived proxy, not expert agronomic ratings" |

---

## 8. File Inventory (24 files, ~10,700 lines total)

```
yellowmaize/
├── __init__.py                      Root package marker
├── config.py                        All hyperparameters (409 lines; includes YOLO_* and GOLD_* keys)
├── image_utils.py                   EXIF correction, CLAHE, channel safety
├── requirements.txt                 All dependencies
├── partition_dataset.py             10-step preprocessing + 70/15/15 split
├── create_bouncer_dataset.py        50k bouncer dataset + nonmaize validation
├── train_bouncer.py                 4-variant bouncer + PatchCore + admission rate
├── sample_15000.py                  Stratified 15k Tier 1 sampler
├── sample_yolo_annotations.py       501-image YOLO bbox annotation export (167 per class)
├── generate_tier1_masks.py          SAM2 auto-prompting + QA filters
├── validate_masks.py                QA report + qualitative overlays
├── sample_gold_standard.py          Extract 300-image gold standard set for Label Studio
├── validate_gold_standard.py        IoU validation: SAM2→Teacher→Student vs human masks
├── train_yolo_detector.py           YOLOv8n leaf detector + SAM2 QA threshold calibration
├── train_teacher.py                 4-variant teacher + test evaluation
├── factory_master.py                4-mode pseudo-label factory + summary statistics
├── train_student.py                 5-variant student + all metrics + test eval
├── generate_charts.py               Chart generator — reads logs/ CSVs → reports/charts/ PNGs
├── select_best_pipeline.py          Best model selection + canonical checkpoints
├── evaluate_xai.py                  3-way XAI comparison + overlays
├── evaluate_severity.py             Inter-rater severity reliability
├── export_tflite.py                 Student → TFLite (FP16)
├── build_deployment_package.py      Bouncer TFLite + full deploy bundle
├── generate_report.py               Self-contained HTML evaluation report
└── scripts/
    ├── __init__.py
    ├── bouncer_inference.py         Shared bouncer inference helpers (heuristic + neural gate)
    └── safe_collate.py              DataLoader corruption guard
```

---

## 9. Datasets to Download Manually

| Dataset | Where to put | Source |
|---|---|---|
| Maize Tanzania + Zenodo | `maize_dataset/HEALTHY,MSV,MLN/` | Already have |
| Intel Image Classification | `dataset/raw_kaggle/intel/` | Kaggle: puneet6060/intel-image-classification |
| Natural Images | `dataset/raw_kaggle/natural/` | Kaggle: prasunroy/natural-images |
| PlantVillage (rice, sorghum) | `dataset/crop_neighbors/rice/`, `sorghum/` | Kaggle: emmarex/plantdisease |
| Mendeley Maize-Weed | `dataset/crop_neighbors/` subfolders | Mendeley: Espejo-Garcia et al. 2020 |
| iNaturalist PH (cogon grass, sugarcane, banana) | `dataset/crop_neighbors/cogon_grass/` etc. | iNaturalist export tool, filter Philippines |
| SAM2 weights | `sam2/sam2_hiera_large.pt` | https://dl.fbaipublicfiles.com/segment_anything_v2/sam2_hiera_large.pt |

---

## 10. Remaining Limitations (for Chapter 5)

1. **Geographic domain gap** — Tanzania data, Philippine deployment. Field validation with local images is the necessary next step.
2. **Severity ground truth** — HSV-derived proxy, not expert CIMMYT-protocol ratings. Future: validate against field severity ratings (0–9 scale).
3. **5k → 15k Teacher set** — Still thin relative to ~215k Tier 2 images. Future: expand Tier 1 to 20–25k.
4. **Staged ablation (not fully crossed)** — Compute budget limits encoder × mode full factorial. Stated explicitly in Chapter 3 and Chapter 5.
5. **iOS excluded** — Android only. Stated in scope.
6. **No Philippine field images** — Dataset fully from Tanzania + Zenodo. Augmentation partially compensates for domain gap.
7. **MobileViT-XXS excluded** — `torch.einsum` self-attention produces TFLite subgraph errors. Replaced by V6 (EfficientNet-B0+CBAM) in ablation. MobileViT benchmarks cited from published literature.

---

## 11. Key Citations

| Reference | Used for |
|---|---|
| Kendall et al. (2018) | Homoscedastic uncertainty loss for multi-task learning |
| Szegedy et al. (2016) | Label smoothing |
| Woo et al. (2018) | CBAM attention module |
| Ke et al. (2020) | Soft segmentation pseudo-label targets |
| Jiang et al. (2018) | MentorNet — curriculum/reliability-weighted training |
| Cruz et al. (2024) | First report of MSV in Philippines |
| Mushayi et al. (2025) | MSV confusion with HEALTHY — asymmetric prior basis |
| Selvaraju et al. (2017) | Grad-CAM |
| Chattopadhay et al. (2018) | Grad-CAM++ |
| Wang et al. (2020) | Score-CAM |
| Mduma (2023) | Tanzania Maize Imagery Dataset |
| Tan & Le (2019) | EfficientNet |
| Xie et al. (2021) | SegFormer |
| Pan et al. (2022) | EdgeViT |
| Fawcett (2006) | ROC threshold selection methodology |

---

## 12. Chart Generator (generate_charts.py)

Standalone script — no imports from project modules. Reads `logs/` CSVs, writes
PNGs to `reports/charts/`. Safe to run mid-training (missing CSVs are skipped).

### Usage
```bash
python generate_charts.py              # all charts
python generate_charts.py --bouncer    # bouncer only
python generate_charts.py --teacher    # teacher only
python generate_charts.py --student    # student only
```

### Charts produced

**Bouncer (`--bouncer`):**

| Output file | Description |
|---|---|
| `bouncer_comparison_bar.png` | Grouped bar: F1 / Specificity / Maize Recall / ROC-AUC per variant. Deployed model highlighted. |
| `bouncer_training_curves_{variant}.png` | Per-epoch: Loss · F1 & Accuracy · Specificity/Recall/Precision. One chart per neural variant. |
| `bouncer_confusion_matrix.png` | 2×2 confusion heatmap (all neural variants side by side). Raw counts + normalised %. |

**Teacher (`--teacher`):**

| Output file | Description |
|---|---|
| `teacher_comparison_bar.png` | Two panels: Best Validation Dice · CPU Inference Latency (ms @ 512px). Winner marked ★. |
| `teacher_training_curves_{variant}.png` | Per-epoch: Loss · Dice & IoU · Recall/Precision/Specificity. One chart per variant. |

**Student (`--student`):**

| Output file | Description |
|---|---|
| `student_stage1_comparison_bar.png` | Grouped bar: Sil mIoU / Sym mIoU / MSV F1 / Macro F1 / Composite across encoder variants. |
| `student_stage2_comparison_bar.png` | Same metrics, grouped by Factory mode A/B/C/D. |
| `student_training_curves_{enc}_{mode}.png` | 2×3 grid: Loss · Seg mIoU · Seg Dice · Cls F1 · Sev MAE · Composite. Phase boundary line shown. |
| `student_confusion_{enc}_{mode}.png` | 3×3 confusion matrix: raw counts (left) + row-normalised recall (right). |
| `student_radar_{enc}_{mode}.png` | Radar chart: Sil mIoU / Sym mIoU / MSV F1 / Cls Acc / Composite / Sev R². One per variant. |
| `student_radar_all_overlay.png` | All variants overlaid on one radar chart. |
| `student_metrics_heatmap.png` | Colour heatmap — all variants × all key metrics. Green = better. |

### Design
- Dark theme matching `evaluation_report.html` (navy `#0F172A`, teal accent `#34D399`)
- Consistent colour per variant across all charts (e.g. MobileNetV2 is always `#3B82F6`)
- 150 DPI PNG output, `bbox_inches="tight"`, non-interactive Agg backend
- Dependencies: `matplotlib ≥ 3.7.0`, `pandas`, `numpy` (all already in `requirements.txt`)
