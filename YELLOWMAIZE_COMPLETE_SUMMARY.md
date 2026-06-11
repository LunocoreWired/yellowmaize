# Yellow MAIze — Complete Project Summary
## For continuing a conversation in a new chat

---

## 1. Project Overview

**Thesis title:** MAIze: A U-Net with MobileNetV2 and Explainable AI Framework for Maize Streak Virus Identification and Symptom Segmentation in Zea mays L.

**Institution:** Angeles University Foundation, College of Computer Studies, BSCS 3-A

**Team:** Altes, Zylah Klein · Davis, Dominic · Tayer, Catherine P. · Ursua, Walter Vince

**Deployment target:** Android mobile application (TFLite)

**Primary clinical task:** Detect MSV (Maize Streak Virus) and MLN (Maize Lethal Necrosis) in yellow corn leaves with explainable, quantified diagnosis.

**Clinical motivation:** MSV was first confirmed in the Philippines (Bukidnon, South Cotabato) in 2023 (Cruz et al. 2024). Early symptoms are visually indistinguishable from nutrient deficiencies. The tool targets agriculture students and smallholder farmers.

**Dataset:** Maize Imagery Dataset — Tanzania (Mduma 2023, Mendeley) + Zenodo healthy/MSV samples. ~252,000 images. Three classes: HEALTHY (~38%), MLN (~38%), MSV (~24%). Geographic mismatch (Tanzania data, Philippine deployment) is a stated limitation.

---

## 2. Hardware & Software

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 — 8 GB VRAM |
| CPU | AMD Ryzen 5 3600X |
| RAM | 16 GB DDR4 |
| OS | Windows 11 + WSL2 Ubuntu (all Python runs in WSL2) |
| DataLoader workers | 4 (WSL2 /dev/shm constraint) |
| Primary framework | PyTorch ≥ 2.1.0 + torchvision ≥ 0.16.0 |
| TFLite export | TensorFlow ≥ 2.13.0 (conversion only, not training) |
| Global seed | 42 — all scripts |
| cuDNN | deterministic=True, benchmark=False |

**Key libraries:** segmentation-models-pytorch ≥ 0.3.3, albumentations ≥ 1.3.1, timm ≥ 0.9.12, opencv-python ≥ 4.8.0, Pillow ≥ 10.0.0, grad-cam ≥ 1.4.8, imagehash ≥ 4.3.1, scikit-learn ≥ 1.3.0, ultralytics ≥ 8.0.0, anomalib ≥ 1.0.0, SAM2 (from GitHub).

---

## 3. Pipeline Architecture

### Phase Map

| Phase | Script | Output |
|---|---|---|
| 0-pre | `partition_dataset.py` | `global_split_manifest.csv` (70/15/15 stratified split) |
| 0a | `create_bouncer_dataset.py` | `data/bouncer_dataset/` — 50k balanced binary dataset |
| 0b | `train_bouncer.py` | `checkpoints/bouncer/bouncer_{variant}_best.pth` |
| 1 | `sample_15000.py` | `data/tier1_raw/` + `tier1_manifest.csv` |
| 1a | `sample_yolo_annotations.py` | `data/yolo_annotations/images/` — 501 images for YOLO bbox annotation |
| 1b | `sample_gold_standard.py` | `data/gold_standard/images/` — 501 images for polygon annotation |
| 1c | `train_yolo_detector.py` | `checkpoints/yolo/best.pt` + `logs/yolo_qa_calibration.csv` |
| 2 | `generate_tier1_masks.py` | `data/tier1_leaf_masks/` — SAM2 float .npy masks + QA report |
| 2b | `validate_masks.py` | QA stats + overlays in `reports/tier1_overlays/` |
| 2c | `validate_gold_standard.py --sam2-only` | SAM2 IoU vs human masks |
| 3 | `train_teacher.py` | `checkpoints/teacher/teacher_{variant}_best.pth` |
| 2d | `validate_gold_standard.py` | SAM2 + Teacher IoU chain vs human masks |
| 4 | `factory_master.py` | `data/pseudo_masks/{mode}/` — pseudo-labels for all ~215k Tier 2 images |
| 5a | `train_student.py --stage 1` | Encoder ablation: 5 variants × Mode B → `checkpoints/student/stage1/` |
| 5b | `train_student.py --stage 2` | Mode ablation: best encoder × 4 modes → `checkpoints/student/` |
| 5c | `validate_gold_standard.py` | Complete chain: SAM2 → Teacher → Student vs human (final thesis table) |
| 6 | `select_best_pipeline.py` | `checkpoints/final/` + auto-updates `config.py` |
| 7 | `evaluate_xai.py` | 3-way XAI comparison + overlays |
| 8 | `evaluate_severity.py` | Inter-rater reliability (Cohen's Kappa + Spearman ρ) |
| 9 | `export_tflite.py` | `exports/tflite/student_model.tflite` |
| 10 | `build_deployment_package.py` | `exports/deploy/` — full Android bundle |
| 11 | `generate_report.py` | `reports/evaluation_report.html` (self-contained) |
| any | `generate_charts.py` | `reports/charts/*.png` — safe to run mid-training |

### Execution Order (bash commands)

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2

python partition_dataset.py

python create_bouncer_dataset.py
python train_bouncer.py
python generate_charts.py --bouncer          # optional

python sample_15000.py

python sample_yolo_annotations.py
# → Label Studio (Object Detection): draw ONE bbox per image, label = "leaf"
# → Export YOLO format → data/yolo_annotations/labels/

python sample_gold_standard.py
# → Label Studio (Polygon): annotate ≥ 400 of 501 leaf silhouettes
# → Export JSON → data/gold_standard/annotations/annotations.json

python train_yolo_detector.py                # requires ≥ 400 YOLO bbox annotations

python generate_tier1_masks.py               # SAM2 v3 — YOLO-guided prompts
python validate_masks.py
python validate_gold_standard.py --sam2-only

python train_teacher.py
python generate_charts.py --teacher          # optional
python validate_gold_standard.py             # SAM2 + Teacher chain

python factory_master.py

python train_student.py --stage 1
# → check logs/student_comparison_stage1.csv for best encoder
python train_student.py --stage 2 --encoder mobilenet_v2_cbam
python generate_charts.py --student          # or: python generate_charts.py (all)
python validate_gold_standard.py             # complete chain for thesis table

python select_best_pipeline.py               # auto-updates config.py
python evaluate_xai.py
python evaluate_severity.py --sample         # two raters fill severity_sample.csv
python evaluate_severity.py --analyze
python export_tflite.py
python build_deployment_package.py
python generate_report.py
```

---

## 4. Component Specifications

### 4.1 Preprocessing (partition_dataset.py)

Ten-step validation pipeline, run **once** before any other script. Never re-run after training begins.

1. Zero-byte / tiny file — stat < 100 bytes → reject
2. Magic bytes / format mismatch — JPEG: FF D8 FF, PNG: 89 PNG → mismatch → reject
3. Truncated image — PIL `.load()` (forces full decode, no header-only) → exception → reject
4. Minimum resolution — min(w,h) < 64px → reject
5. Maximum resolution / aspect ratio — max(w,h) > 4096px or max/min > 8.0 → reject
6. Colour mode — 1-bit binary → reject; grayscale → flag (keep); palette → flag (keep)
7. Near-uniform / solid colour — RGB std < 5.0 → reject
8. MD5 exact duplicate removal — cross-class simultaneous; cross-class same hash → reject both
9. pHash near-duplicate removal — Hamming ≤ 2 within class → reject duplicate
10. Cross-class pHash — Hamming ≤ 2 between different classes → reject both

Additionally: low-green-content flag (< 5% green pixels → flagged, kept).

**Outputs:** `preprocessing_report.csv`, `preprocessing_flagged.csv`, `preprocessing_summary.txt`, `global_split_manifest.csv`

**Dataset split:** 70% train / 15% val / 15% test, stratified per class. Test split is held out permanently and never used for any training decision.

### 4.2 Image Utilities (image_utils.py)

Single source of truth for all image loading. Guarantees consistent EXIF orientation and channel order throughout the pipeline.

- `load_image_rgb(path)` — PIL open + `ImageOps.exif_transpose()` + truncation guard → uint8 RGB numpy or None
- `load_image_clahe(path)` — `load_image_rgb()` + CLAHE enhancement → uint8 RGB or None
- `apply_clahe(img_rgb)` — RGB → LAB → CLAHE on L channel → RGB. clipLimit=2.0, tileGridSize=(8,8)
- `to_hsv(img_rgb)` — guaranteed RGB→HSV (never BGR→HSV)
- `rgb_to_bgr(img_rgb)` — for `cv2.imwrite()` calls only

CLAHE is applied to Bouncer inputs and Factory inputs. It is NOT applied to Student/Teacher training (augmentation handles contrast variation there).

### 4.3 Shared Infrastructure (scripts/)

**`scripts/safe_collate.py`** — Every DataLoader uses `collate_fn=safe_collate`. If `__getitem__` returns `None` (corrupt image), it is filtered before collating. If the entire batch is `None`, returns `None` — the training loop skips with `continue`. Module-level skip counter reset per epoch via `reset_skip_counter()`.

**`scripts/bouncer_inference.py`** — Single source of truth for Bouncer inference, shared by `factory_master.py` and `train_bouncer.py`. Contains:
- `BOUNCER_INFER_TF` — letterbox 224×224, ImageNet normalize, ToTensorV2
- `heuristic_prefilter(img_rgb) → bool` — currently a **passthrough** (always True). The original OpenCV green-coverage heuristic was removed after false rejections on yellow/bleached MSV leaves under variable tropical lighting.
- `neural_bouncer(img_rgb, model, threshold) → bool` — applies BOUNCER_INFER_TF, runs model, returns `sigmoid(logit) >= threshold`

Dependencies: `torch`, `albumentations`, `config.BOUNCER_IMG_SIZE` only. No SAM2, Teacher, or other heavy imports.

### 4.4 Bouncer (Phase 0)

**Purpose:** Binary gate — maize vs non-maize. Runs first on every camera frame. Rejects non-maize images before disease analysis.

**Dataset:** 25,000 maize (train+val only, test excluded) + 25,000 non-maize = 50,000 balanced. Internal 80/20 split (Bouncer-internal, not from global manifest).

**Non-maize sources:** Intel Image Classification, Natural Images, PlantVillage (rice/sorghum), iNaturalist Philippines (cogon grass / banana leaf / sugarcane), Mendeley Maize-Weed. Non-maize images pass a 6-step preprocessing validation (Steps 1–6 from `partition_dataset.py`).

**Image loading:** `load_image_clahe()` — EXIF correction + CLAHE + truncation guard.

**4 comparison variants:**

| Variant | Architecture | Deployed? |
|---|---|---|
| gabor_lbp | Gabor filters + LBP + LinearSVC | No (baseline) |
| mobilenet_v2 | MobileNetV2, head: Linear(1280→1) | No (comparison) |
| **mobilenet_v3_large** | MobileNetV3-Large, head: Linear(960→1) | **Yes** |
| edgevit_xxs | EdgeViT-XXS binary classifier | Candidate |

> PatchCore (ResNet18 nearest-neighbour anomaly detector) is available as an optional offline evaluation via `evaluate_patchcore()` in `train_bouncer.py`, but is excluded from `BOUNCER_VARIANTS` — no TFLite deployment path.

**Hyperparameters (neural variants):**
- Loss: BCEWithLogitsLoss | Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=15, eta_min=1e-6) | Epochs: 15 | Batch: 64
- Early stop: patience=5 (monitors val F1) | Gradient clipping: max_norm=5.0

**Training augmentation:** LongestMaxSize(224) + PadIfNeeded(224,224) + HorizontalFlip(0.5) + RandomRotate90(0.3) + ColorJitter(0.2, 0.2) + HueSaturationValue(H±15, S±20, V±10) + ImageNet normalize.

**Threshold selection:** After training, run ROC on val split. Select threshold = argmax(geometric mean of recall × specificity) subject to maize recall ≥ 0.95. Hard cap at 0.70. (Geometric mean balances both metrics; previously using specificity-only caused near-1.0 thresholds on high-AUC models.)

**Primary metric:** Specificity (TN rate — non-maize rejection rate).

**Outputs:** `checkpoints/bouncer/bouncer_{variant}_best.pth`, `logs/bouncer_comparison.csv`, `logs/bouncer_admission_rate_{variant}.csv`

### 4.5 Tier 1 Sampling (sample_15000.py)

Stratified sample of **15,000 images** from train+val split only. Test-split images are never included.

| Class | Count | Method |
|---|---|---|
| HEALTHY | 3,000 | Pure random (seeded) |
| MSV | 7,500 | Evenly-spaced indices across sorted filenames — maximises diversity |
| MLN | 4,500 | Evenly-spaced indices |

**Hash guard:** On first run, SHA-256 of `global_split_manifest.csv` is written to `data/tier1_raw/_manifest_hash.txt`. If `partition_dataset.py` is re-run after masking begins, subsequent runs abort with a clear error instead of silently producing a different set (which would invalidate any SAM2 masks or Label Studio annotations already done).

**Output:** `data/tier1_raw/{CLASS}_{original_filename}`, `tier1_manifest.csv`

### 4.6 YOLO Annotation Sampler (sample_yolo_annotations.py)

Exports **501 images** (167 per class) from Tier 1 for YOLO bounding-box annotation in Label Studio. Files are renamed with class prefix (`{CLASS}_{original}.jpg`). SHA-256 hash guard prevents re-sampling if `tier1_manifest.csv` changes post-annotation.

**Annotation instructions:** Object Detection task type, label = "leaf", ONE tight bounding box around the primary leaf only. Ignore background leaves and stems. Export YOLO format → `data/yolo_annotations/labels/`.

### 4.7 Gold Standard Sampling (sample_gold_standard.py)

Exports **501 images** (167 per class) from Tier 1 for human polygon annotation. Files renamed with class prefix. SHA-256 hash guard. Must run before `train_yolo_detector.py`.

**Annotation instructions:** Label Studio polygonlabels task type. Annotate leaf silhouette as a polygon. Annotate at least 400 of 501 images. Export JSON → `data/gold_standard/annotations/annotations.json`.

**Output:** `data/gold_standard/images/`, `data/gold_standard/gold_manifest.csv`

### 4.8 YOLO Leaf Detector (train_yolo_detector.py)

Trains a YOLOv8n (nano) single-class detector from Label Studio polygon annotations. Provides tight bounding-box prompts for SAM2 in `generate_tier1_masks.py` v3. Must run before `generate_tier1_masks.py`.

**Why YOLO before SAM2:** Pure HSV prompting drifts onto background for diseased images — MSV yellow and MLN brown share hue ranges with tropical soil. A YOLO bounding box constrains both the centroid search and SAM2 segmentation region.

**Target:** mAP@0.5 ≥ 0.70 before using YOLO prompts. Script warns if annotation count < `YOLO_MIN_ANNOTATIONS` (400).

After training, the script runs YOLO+SAM2 on gold-standard images and calibrates the SAM2 QA confidence threshold to achieve mean IoU ≥ `GOLD_IOU_TARGET_MEAN` (0.85). Writes calibration to `logs/yolo_qa_calibration.csv`.

**Outputs:** `checkpoints/yolo/best.pt`, `logs/yolo_qa_calibration.csv`, `logs/yolo_training_metrics.csv`, `data/yolo_dataset/`

### 4.9 SAM2 Masking (generate_tier1_masks.py) [v3: YOLO-guided]

**Auto-prompting strategy (v3):**
1. Run YOLOv8n → tight leaf bounding box (x1,y1,x2,y2) if `checkpoints/yolo/best.pt` exists
2. Restrict HSV tissue search to pixels inside the YOLO box (no centroid drift)
3. Build 3 foreground points along the leaf's vertical axis, clamped to the box
4. Pass the YOLO box as SAM2 `box=` prompt — hard spatial constraint
5. Fallback to full-image HSV-only centroid if YOLO absent or detection fails
6. Four image corners (10px inset) → background point prompts (label=0)
7. Select mask with highest SAM2 score → sigmoid → float32 probability map

**HSV tissue detection:** Green tissue: H∈[35,85], S>40, V>40. Yellow/diseased: H∈[15,45], S>40, V>60.

**QA filters (v3 relaxed):**
1. Coverage: foreground pixels 3%–90% (was 10% minimum in v1)
2. Mean foreground confidence ≥ calibrated threshold (default 0.65 if calibration absent)
3. Mask aspect ratio ≥ 1.01 (was 1.20 in v1 — overhead/square-frame leaves now pass)

Target rejection rate: < 8%. QA report includes `prompt_mode` (yolo | hsv_fallback) and `yolo_box` columns.

**Outputs per image:**
- `data/tier1_leaf_masks/{stem}_softmask.npy` — float32 [0,1] probability map (primary target for Teacher)
- `data/tier1_leaf_masks/{stem}_mask.png` — uint8 binary visualization

**Outputs overall:** `tier1_qa_report.csv`

### 4.10 Teacher Model (train_teacher.py)

**Purpose:** Offline segmentation model — never deployed to Android. Sole purpose: generate leaf silhouette pseudo-masks for ~215k Tier 2 images via Factory.

**Training targets:** SAM2 float32 probability maps — NO binarization. Preserves boundary uncertainty (pixels near edge: 0.3–0.7; interior: ~1.0; exterior: ~0.0). DiceLoss accepts float targets natively.

**4 comparison variants:**

| Variant | Encoder | Decoder | Params |
|---|---|---|---|
| resnet50 | ResNet-50 | UNet | ~32M |
| **efficientnet-b2** | EfficientNet-B2 | UNet | ~7.7M — recommended |
| mit_b2 | SegFormer-B2 (Mix Transformer) | UNet | ~25M |
| deeplabv3plus-eb2 | EfficientNet-B2 | DeepLabV3+ | ~7.7M — decoder comparison |

All via `segmentation_models_pytorch`. SegFormer-B2 uses `encoder_name="mit_b2"` via timm. `deeplabv3plus-eb2` tests the same encoder with a different decoder (ASPP vs standard UNet).

**Hyperparameters:** Input 512×512 | Batch 8 | Epochs 30 | AdamW (lr=5e-5, wd=1e-4) | ReduceLROnPlateau (factor=0.5, patience=5, mode=max) | Early stop patience=10 (monitors val Dice) | Gradient clipping max_norm=5.0 | `safe_collate`.

**Augmentation (training):** LongestMaxSize(512) + PadIfNeeded(512,512) + HFlip(0.5) + VFlip(0.5) + RandomRotate90(0.5) + RandomBrightnessContrast(±0.2, 0.3) + HueSaturationValue(H±10, S±20, V±10, 0.2) + ImageNet normalize.

**Manifest filtering:** Loads `tier1_manifest.csv` and `global_split_manifest.csv`. Excludes any Tier 1 images whose source file belongs to the test split. Teacher never sees test-split images even as Tier 1 training data.

**Best Teacher → `checkpoints/teacher/teacher_model_best.pth`** (canonical, used by Factory).

**Metrics:** Dice (primary), IoU, Recall, Precision, Specificity per epoch; CPU latency ms/image; test eval on Tier 1 test-split images.

### 4.11 Factory (factory_master.py)

**Purpose:** Process all ~215k train+val Tier 2 images. Generate pseudo-labels across 4 modes simultaneously into separate subfolders.

**Per-image processing stages:**

1. Bouncer gate — imports from `scripts/bouncer_inference.py`:
   - `heuristic_prefilter()` — currently a passthrough (always True)
   - `neural_bouncer()` — MobileNetV3-Large at empirical threshold
   - Rejected → log status, skip
2. Leaf silhouette source:
   - Tier 1 images → load pre-existing SAM2 .npy (skip Teacher inference)
   - Tier 2 images → Teacher inference at 512×512 → resize to original dimensions
3. Silhouette refinement: threshold at 0.35 (lower than 0.5 to catch dark leaves) + morphological close/open (kernel=5, elliptical)
4. Coverage guard → reliability weight:
   - < 15%: weight=−1 (exclude, severity sentinel)
   - 15–25%: weight=0.30
   - 25–50%: weight=0.70
   - > 50%: weight=1.00
5. HSV symptom masking (mode-dependent — see below)
6. Severity = (symptom pixels / leaf pixels) × 100%

**4 Factory modes:**

| Mode | Silhouette source | Symptom mask | Subfolder |
|---|---|---|---|
| A | Otsu threshold | Hard binary HSV | `mode_a/` |
| B | SAM2 hard binary | Hard binary HSV | `mode_b/` |
| C | SAM2 soft float | Hard binary HSV | `mode_c/` |
| D | SAM2 soft float | Soft HSV confidence map | `mode_d/` |

**Green exclusion zone (applied first):** H 38–85, S 80–255, V 60–230 — removes healthy chlorophyll.

**MSV HSV ranges (4 bands):**
- R1: H 15–38, S 50–255, V 150–255 (bright yellow streaks)
- R2: H 20–45, S 15–70, V 130–255 (pale yellow / early-stage)
- R3: H 0–179, S 0–35, V 210–255 (near-white / bleached — area filter ≥ 80px)
- R4: H 38–55, S 10–55, V 140–255 (pale yellow-green)

**MLN HSV ranges (5 bands):**
- R1: H 18–40, S 40–255, V 90–255 (chlorotic yellow)
- R2: H 5–20, S 40–255, V 50–220 (orange-amber necrosis)
- R3: H 22–55, S 15–85, V 80–240 (pale yellow-green mosaic)
- R4: H 8–35, S 0–45, V 160–255 (tan / straw tissue)
- R5: H 0–18, S 30–180, V 30–150 (dark brown dead tissue)

**Mode D soft HSV confidence:** `confidence(p) = Σ(range_hit × normalized_distance_to_center) / N_ranges`, clipped [0,1].

**Output resolution:** All .npy and .png outputs downscaled to STUDENT_IMG_SIZE (224×224) at write time. Silhouettes and Mode D symptom: INTER_LINEAR. Binary symptom PNGs: INTER_NEAREST (preserves hard edges).

**Outputs per image per mode:** `{stem}_silhouette.npy`, `{stem}_symptom.npy` (Mode D) or `{stem}_symptom.png` (A/B/C), `{stem}_sev.txt`, `{stem}_weight.txt`.

**Summary reports:** `reports/factory_summary.csv` (per-mode × per-class stats), `reports/factory_filter_breakdown.csv`.

### 4.12 Student Model (train_student.py)

**The final deployed model.** Single shared encoder with three output heads.

**Architecture:**
```
Input [B, 3, 224, 224] (ImageNet normalized)
    ↓
Shared Encoder (MobileNetV2 or variant)
    ├── UNet Decoder (+ CBAM at skip connections for V2, V6)
    │       ↓
    │   Segmentation Head → [B, 2, 224, 224] raw logits
    │       Ch0: leaf silhouette  (sigmoid → binary mask)
    │       Ch1: symptom mask     (sigmoid → binary mask)
    │
    ├── GAP → Dropout(0.3) → Linear(→3) → Classification Head
    │       [B, 3] raw logits → softmax → argmax → HEALTHY=0 / MSV=1 / MLN=2
    │
    └── GAP → Dropout(0.3) → Linear(→1) → ReLU → clamp(0,1) → Severity Head
            [B, 1] → ×100 at inference = severity %
```

**Why ReLU+clamp instead of sigmoid for severity:** Sigmoid never reaches exactly 0.0. HEALTHY leaves would always show nonzero severity. ReLU+clamp allows true 0%.

**5 encoder variants (Stage 1 ablation on Mode B):**

| Variant | Encoder | Params | TFLite | Notes |
|---|---|---|---|---|
| V1 | mobilenet_v2 | 3.4M | Yes | Baseline |
| **V2** | **mobilenet_v2_cbam** | **~3.5M** | **Yes** | **Expected winner** |
| V3 | mobilenet_v3_small | 2.5M | Yes | Ultra-compact |
| V4 | efficientnet_b0 | 5.3M | Yes | Compound scaling (no attention) |
| V6 | efficientnet_b0_cbam | ~5.4M | Yes | CBAM generalization test |

> V5 (MobileViT-XXS) was removed — `torch.einsum` self-attention causes TFLite subgraph errors incompatible with deployment. V6 replaces it.

**CBAM (Woo et al. 2018):** Applied at each UNet skip connection. Channel attention (GAP+GMP → shared MLP → sigmoid) + spatial attention (7×7 conv → sigmoid). Implemented as a proper subclass of `smp.decoders.unet.decoder.UnetDecoder` — NOT monkey-patching. Fully serialisable with `torch.save()`.

**Loss functions:**

| Task | Loss | Notes |
|---|---|---|
| Segmentation (both channels averaged) | DiceLoss (binary, from_logits=True) | Float targets — preserves soft boundary uncertainty |
| Classification | Asymmetric label smoothing cross-entropy | Pathology-informed prior matrix |
| Severity | Reliability-weighted MSE | Per-sample weight from Factory coverage bracket |
| Multi-task balancing | Homoscedastic uncertainty (Kendall et al. 2018) | 3 learnable log-variance parameters (s1, s2, s3) |

**Asymmetric label smoothing prior matrix:**
```
HEALTHY → [0.90, 0.08, 0.02]   # early MSV looks like HEALTHY
MSV     → [0.05, 0.90, 0.05]
MLN     → [0.02, 0.05, 0.93]   # MLN is most visually distinct
```

**Homoscedastic uncertainty loss:** `L_total = exp(−s1)·L_seg + s1 + exp(−s2)·L_cls + s2 + exp(−s3)·L_sev + s3`. s1/s2/s3 learned via backprop. Replaces manual fixed weights (0.6/0.2/0.2).

**Two-phase transfer learning:**

| Phase | Epochs | LR | Encoder | Patience |
|---|---|---|---|---|
| 1 — Frozen | 30 | 1e-3 (cosine) | Frozen | 10 |
| 2 — Fine-tuning | 20 | 1e-4 (cosine) | Unfrozen | 5 |

Optimizer re-initialized at Phase 1→2 transition. Patience resets to 0. Gradient clipping max_norm=5.0 throughout.

**WeightedRandomSampler:** Per-class inverse-frequency weights. Ensures proportional class representation per batch (MSV ~24% otherwise underrepresented).

**Checkpoint criterion (quality composite — used during training):** `0.50×mIoU + 0.35×MSV_F1 + 0.15×(1−NormMAE)`

**Augmentation (training):** Letterbox padding + HFlip(0.5) + VFlip(0.5) + RandomRotate90(0.5) + Rotate(±30°, 0.5) + RandomBrightnessContrast(±0.25, 0.3) + HueSaturationValue(H±15, S±20, V±10, 0.2) + **RandomShadow(0.2)** (tropical domain adaptation) + ImageNet normalize.

**Two-stage ablation:**
- Stage 1: Fix mode=mode_b. Train all 5 encoder variants. Save to `checkpoints/student/stage1/` (prevents Stage 2 overwrite).
- Stage 2: Fix encoder=best from Stage 1. Train on all 4 Factory modes. Mode B result from Stage 1 reused (same seed=42 → identical run).
- Total unique training runs: 5 + 3 = 8.

**Mobile composite (post-training, used by select_best_pipeline.py for deployment selection):**
```
mobile_composite = 0.38 × msv_f1
                 + 0.22 × sil_mIoU
                 + 0.22 × clamp(150ms / cpu_lat_ms, 0, 1)
                 + 0.10 × (1 − norm_mae)
                 + 0.08 × clamp(15MB / tflite_size_mb, 0, 1)
```
TFLite-incompatible variants excluded from deployment selection.

### 4.13 XAI (evaluate_xai.py)

**3 comparison methods (pytorch-grad-cam library):**

| Method | Type | Role |
|---|---|---|
| Grad-CAM | Gradient-based | Historical baseline (Selvaraju et al. 2017) |
| **Grad-CAM++** | Gradient-based | **Deployed** — better for multi-region MSV streaks (Chattopadhay et al. 2018) |
| Score-CAM | Gradient-free | Stability reference (Wang et al. 2020) |

**Target layer:** Last convolutional block of the shared encoder (before decoder branches) — reflects features driving both classification and segmentation simultaneously.

| Encoder | Target layer |
|---|---|
| mobilenet_v2 / mobilenet_v2_cbam | `encoder.features[-1][0]` |
| mobilenet_v3_small | `encoder.features[-1][0]` |
| efficientnet_b0 / efficientnet_b0_cbam | `encoder.blocks[-1][-1]` |

**Two DISTINCT app outputs (not the same thing — critical for thesis):**
1. **Symptom boundary** (green contour) — UNet segmentation head Ch1 → pixel-level localization
2. **Diagnostic attention** (amber heatmap) — Grad-CAM++ → coarse class-discriminative explanation (~7×7 upsampled)

**Quantitative metrics:** Pointing game accuracy (top 20% heatmap activation fraction inside seg mask), Insertion AUC (n_steps=6), Deletion AUC (n_steps=6). All computed on GPU.

### 4.14 Severity Reliability (evaluate_severity.py)

60 images (20 per class), 2 raters, 0–3 scale: 0=none, 1=mild (<25%), 2=moderate (25–60%), 3=severe (>60%).

- Cohen's Kappa — inter-rater agreement
- Spearman ρ — human ratings vs HSV-derived severity
- Target: ρ ≥ 0.60 = "moderate correlation"

**Important disclaimer for Chapter 4:** Severity MAE measures Student consistency with Factory's HSV-derived pseudo-labels, NOT expert agronomic ratings. This is a stated limitation.

### 4.15 Best Pipeline Selection (select_best_pipeline.py)

Selection criteria:
- **Bouncer:** max specificity subject to maize_recall ≥ 0.95 (neural variants only)
- **Teacher:** max val Dice
- **Student (training):** max quality composite (0.50×mIoU + 0.35×MSV_F1 + 0.15×(1−NormMAE))
- **Student (deployment):** max mobile composite (see §4.12)

Promotes winners to `checkpoints/final/`. Auto-updates `STUDENT_BEST_VARIANT` and `STUDENT_FACTORY_MODE` in `config.py` via regex (no manual editing needed).

**Outputs:** `checkpoints/final/bouncer_best.pth`, `teacher_best.pth`, `student_best.pth`; `reports/best_pipeline_summary.csv`, `reports/best_pipeline_summary.txt`, `reports/all_variants_ranked.csv`, `reports/student_mobile_ranking.csv`.

### 4.16 Gold Standard Validation (validate_gold_standard.py)

**Purpose:** Validate pseudo-label quality against human-annotated leaf silhouette masks. Provides a thesis-defensible chain comparison on the same 501 images:

```
SAM2 pseudo-mask → Teacher prediction → Student prediction
       ↓                   ↓                    ↓
  IoU vs human        IoU vs human         IoU vs human
```

**Label Studio parser:** Accepts JSON export (polygonlabels). Polygon points stored as % of image dimensions. Rasterized to binary mask via `cv2.fillPoly()`. Multiple polygons merged via logical OR.

**Run three times:**
1. After `generate_tier1_masks.py` — `--sam2-only` flag (before Teacher training)
2. After `train_teacher.py` — full run (SAM2 + Teacher chain)
3. After `train_student.py` — full run (complete chain for final thesis table)

**Config keys:** `GOLD_IOU_WARN_THRESHOLD = 0.75` (per-image flag), `GOLD_IOU_TARGET_MEAN = 0.85` (overall validation target), `TEACHER_DEPLOYED_VARIANT = "efficientnet-b2"`.

**Outputs:** `reports/gold_standard_iou_report.csv` (per-image IoU), `reports/gold_standard_iou_summary.csv` (mean ± std per class + overall), `reports/gold_standard_overlays/` (5 overlay PNGs per class per artifact — Green=missed, Cyan=correct, Red=extra).

### 4.17 Deployment (export_tflite.py + build_deployment_package.py)

**Export path:** PyTorch → ONNX (opset 12) → TF SavedModel → TFLite (FP16 quantized).

**Student TFLite I/O:**
- Input: `[1, 3, 224, 224]` float32, NCHW, ImageNet normalized
- Output 0 (seg): `[1, 2, 224, 224]` raw logits → sigmoid → binary at 0.5
- Output 1 (cls): `[1, 3]` raw logits → softmax → argmax (0=HEALTHY, 1=MSV, 2=MLN)
- Output 2 (sev): `[1, 1]` float32 ∈ [0,1] → ×100 = severity %

**Bouncer TFLite I/O:**
- Input: `[1, 3, 224, 224]` float32, NCHW, same normalization
- Output: `[1, 1]` raw logit → sigmoid → if ≥ threshold → pass to Student

**Android runtime:** TFLite API 21+, `tensorflow-lite:2.13.0`, optional GPU delegate. Image pipeline: EXIF correction → letterbox 224×224 → ImageNet normalize → NCHW → Bouncer → (if maize) → Student → post-process → display.

**Deploy bundle (`exports/deploy/`):** `bouncer_model.tflite`, `student_model.tflite`, `model_metadata.json`, `DEPLOYMENT_README.md` (Kotlin/Java snippets), `deployment_report.csv`.

### 4.18 Chart Generator (generate_charts.py)

Standalone script — no imports from project modules. Reads `logs/` CSVs, writes PNGs to `reports/charts/`. Safe to run mid-training (missing CSVs silently skipped). Dark theme matching `evaluation_report.html`.

```bash
python generate_charts.py              # all charts
python generate_charts.py --bouncer    # bouncer only
python generate_charts.py --teacher    # teacher only
python generate_charts.py --student    # student only
```

**Charts produced:**
- Bouncer: `bouncer_comparison_bar.png`, `bouncer_training_curves_{variant}.png`, `bouncer_confusion_matrix.png`
- Teacher: `teacher_comparison_bar.png`, `teacher_training_curves_{variant}.png`
- Student: `student_stage1_comparison_bar.png`, `student_stage2_comparison_bar.png`, `student_training_curves_{enc}_{mode}.png`, `student_confusion_{enc}_{mode}.png`, `student_radar_{enc}_{mode}.png`, `student_radar_all_overlay.png`, `student_metrics_heatmap.png`

### 4.19 HTML Report (generate_report.py)

Produces `reports/evaluation_report.html` — 100% self-contained (all charts base64 embedded), dark theme, no external dependencies. Nine sections: Preprocessing → Bouncer → Teacher → Student encoder ablation → Student mode ablation → Best Student full results → XAI → Severity reliability → Deployment summary.

---

## 5. Evaluation Metrics (Complete List)

### Bouncer
Accuracy, Precision, Maize Recall, Maize F1, Specificity (primary), ROC-AUC, TP/FP/TN/FN, empirical threshold, end-to-end admission rate on test split.

### Teacher
Val Dice (primary), Val IoU, Val Recall, Val Precision, Val Specificity, loss per epoch, CPU inference latency ms/image.

### Factory
Per-mode × per-class: mean/std/median severity, % symptomatic, exclusion rate, mean silhouette confidence (Modes C/D), filter breakdown by stage.

### Student (test split)
- Segmentation Ch0 (silhouette): mIoU, Dice, Recall, Precision, Specificity
- Segmentation Ch1 (symptom): mIoU, Dice, Recall, Precision, Specificity
- Classification overall: Accuracy, MCC, Macro F1, Weighted F1
- Classification per class: Precision, Recall, F1 for HEALTHY / MSV / MLN
- Severity: MAE%, RMSE%, R²
- Composite: `0.50×mIoU + 0.35×MSV_F1 + 0.15×(1−NormMAE)`
- Latency: CPU mean ms, CPU std ms, CPU FPS

Note: mIoU is computed via global TP/FP/FN accumulation, not mean-of-batches (per-batch mean is statistically incorrect when batch class distributions vary).

### XAI
Pointing game accuracy, Insertion AUC, Deletion AUC — per method per image.

### Severity reliability
Cohen's Kappa (inter-rater), Spearman ρ (HSV vs human).

---

## 6. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| "Pseudo-label semi-supervised learning" — not "knowledge distillation" | Teacher generates hard mask targets + severity floats. True KD requires soft logit targets. Wrong term must not appear in thesis. |
| SAM2 soft probability maps as Teacher targets | Boundary uncertainty preserved: edge pixels get 0.3–0.7 rather than hard 0/1. Segmentation label smoothing. |
| Asymmetric label smoothing | Early MSV visually identical to HEALTHY (Cruz et al. 2024). Pathology-informed prior prevents overconfidence on folder labels. |
| Homoscedastic uncertainty loss | Replaces manual 0.6/0.2/0.2 weights. Three learnable log-variance parameters auto-balance task losses (Kendall et al. 2018). |
| ReLU+clamp severity head | Sigmoid ceiling prevents true 0% severity. HEALTHY leaves always show nonzero with sigmoid. |
| Global mIoU (not mean-of-batches) | Per-batch IoU averaging is statistically wrong with variable class distributions per batch. |
| Two-channel segmentation | Ch0=leaf silhouette (spatial extent); Ch1=symptom mask (disease location). Separate targets, separate metrics. |
| CBAM as proper subclass | Monkey-patching smp UnetDecoder breaks `torch.save()` serialisation. Proper subclass with explicit forward is fully serialisable. |
| Stage 1 checkpoint isolation | Stage 2 Mode B run would overwrite Stage 1 result for same encoder+mode combination. Isolation to `stage1/` subfolder prevents this. |
| Two-composite selection (quality vs mobile) | During training, TFLite size and CPU latency are unknown. Quality composite saves checkpoints. After all runs complete, mobile composite picks the deployment model. Methodologically clean and fully documentable. |
| Empirical Bouncer threshold | Geometric mean of recall × specificity produces a usable production threshold. Specificity-alone maximisation caused near-1.0 thresholds with high-AUC models (>88% rejection rate). |
| heuristic_prefilter passthrough | Original green-coverage heuristic caused false rejections on yellow/bleached MSV leaves. Neural classifier is sufficient and fast enough at 224×224. |

---

## 7. Terminology Corrections (Important for Thesis)

| Wrong term | Correct term |
|---|---|
| "Knowledge distillation" | "Pseudo-label semi-supervised learning" or "teacher-guided pseudo-label generation" |
| "Pixel-level segmentation via Grad-CAM" | Grad-CAM++ provides coarse class-discriminative localization (~7×7 upsampled). UNet head provides pixel-level localization. These are entirely different outputs. |
| "Severity MAE measures disease severity accuracy" | "Severity MAE measures consistency with Factory's HSV-derived proxy, not expert agronomic ratings" |

---

## 8. File Inventory (24 scripts + shared utilities)

```
yellowmaize/
├── __init__.py                          Root package marker
├── config.py                            All hyperparameters (includes YOLO_* and GOLD_* keys)
├── image_utils.py                       EXIF correction, CLAHE, channel safety
├── requirements.txt
├── partition_dataset.py                 10-step preprocessing + 70/15/15 global split
├── create_bouncer_dataset.py            50k bouncer dataset + nonmaize validation
├── train_bouncer.py                     4-variant bouncer + PatchCore + admission rate eval
├── sample_15000.py                      Stratified 15k Tier 1 sampler
├── sample_yolo_annotations.py          501-image YOLO bbox annotation export (167 per class)
├── sample_gold_standard.py             501-image gold standard polygon annotation export (167 per class)
├── train_yolo_detector.py              YOLOv8n leaf detector + SAM2 QA threshold calibration
├── generate_tier1_masks.py             SAM2 auto-prompting v3 (YOLO-guided) + QA filters
├── validate_masks.py                    QA report + 5 qualitative overlays per class
├── validate_gold_standard.py           IoU chain: SAM2 → Teacher → Student vs human masks
├── train_teacher.py                     4-variant teacher + test evaluation
├── factory_master.py                    4-mode pseudo-label factory (all ~215k Tier 2 images)
├── train_student.py                     5-variant student (Stage 1 + Stage 2) + all metrics
├── generate_charts.py                   Standalone chart generator (reads logs/ → reports/charts/)
├── select_best_pipeline.py             Best model selection + canonical checkpoints + config update
├── evaluate_xai.py                      3-way XAI comparison (Grad-CAM / Grad-CAM++ / Score-CAM)
├── evaluate_severity.py                 Inter-rater severity reliability
├── export_tflite.py                     Student → ONNX → TFLite (FP16)
├── build_deployment_package.py         Bouncer TFLite + full Android deployment bundle
├── generate_report.py                   Self-contained HTML evaluation report
└── scripts/
    ├── __init__.py
    ├── bouncer_inference.py             Shared bouncer inference (heuristic + neural gate)
    └── safe_collate.py                  DataLoader corruption guard
```

---

## 9. Datasets to Download Manually

| Dataset | Location | Source |
|---|---|---|
| Maize Tanzania (Mduma 2023) + Zenodo | `maize_dataset/HEALTHY/`, `MSV/`, `MLN/` | Mendeley Data + Zenodo |
| Intel Image Classification | `dataset/raw_kaggle/intel/` | Kaggle: puneet6060/intel-image-classification |
| Natural Images | `dataset/raw_kaggle/natural/` | Kaggle: prasunroy/natural-images |
| PlantVillage (rice, sorghum) | `dataset/crop_neighbors/rice/`, `sorghum/` | Kaggle: emmarex/plantdisease |
| iNaturalist Philippines (cogon grass, banana, sugarcane) | `dataset/crop_neighbors/cogon_grass/`, `banana_leaf/`, `sugarcane/` | iNaturalist API, filter Philippines |
| Mendeley Maize-Weed | `dataset/crop_neighbors/` subfolders | Espejo-Garcia et al. 2020 |
| SAM2 weights | `sam2/sam2_hiera_large.pt` | https://dl.fbaipublicfiles.com/segment_anything_v2/sam2_hiera_large.pt |

---

## 10. Remaining Limitations (for Chapter 5)

1. **Geographic domain gap** — Tanzania dataset, Philippine deployment. Field validation with local images is the necessary next step.
2. **Severity ground truth** — HSV-derived proxy, not expert CIMMYT-protocol ratings. Future: validate against field severity ratings (0–9 scale).
3. **Tier 1 size** — 15k images for Teacher training relative to ~215k Tier 2 images processed by Factory. Future: expand Tier 1 to 20–25k.
4. **Staged ablation (not fully crossed)** — Compute budget limits a full encoder × mode factorial design. Stated explicitly in Chapter 3 and Chapter 5.
5. **iOS excluded** — Android only. Stated in scope.
6. **No Philippine field images** — All data from Tanzania + Zenodo. Augmentation (RandomShadow, HSV jitter) partially compensates for domain gap.
7. **MobileViT-XXS excluded** — `torch.einsum` self-attention produces TFLite subgraph errors. Replaced by V6 (EfficientNet-B0+CBAM) in ablation. MobileViT benchmarks cited from published literature in related work.

---

## 11. Key Citations

| Reference | Used for |
|---|---|
| Kendall et al. (2018) NeurIPS | Homoscedastic uncertainty loss for multi-task balancing |
| Szegedy et al. (2016) CVPR | Label smoothing |
| Woo et al. (2018) ECCV | CBAM convolutional block attention module |
| Ke et al. (2020) | Soft segmentation pseudo-label targets |
| Jiang et al. (2018) ICML | MentorNet — curriculum / reliability-weighted training |
| Cruz et al. (2024) | First confirmed report of MSV in the Philippines |
| Mushayi et al. (2025) | MSV confusion with HEALTHY — asymmetric prior matrix basis |
| Selvaraju et al. (2017) ICCV | Grad-CAM |
| Chattopadhay et al. (2018) WACV | Grad-CAM++ |
| Wang et al. (2020) CVPR | Score-CAM |
| Mduma (2023) Mendeley | Tanzania Maize Imagery Dataset |
| Tan & Le (2019) ICML | EfficientNet |
| Xie et al. (2021) NeurIPS | SegFormer |
| Pan et al. (2022) ECCV | EdgeViT |
| Fawcett (2006) | ROC threshold selection methodology |
| Zuiderveld (1994) | CLAHE |
