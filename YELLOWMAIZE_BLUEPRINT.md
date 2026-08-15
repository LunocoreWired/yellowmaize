# Yellow MAIze — Detailed Technical Blueprint
## Complete reference for code, pipeline, models, data, and architecture

---

## PART 1 — PROJECT CONTEXT

### 1.1 Thesis Identity

| Field | Value |
|---|---|
| Title | MAIze: A U-Net with MobileNetV2 and Explainable AI Framework for Maize Streak Virus Identification and Symptom Segmentation in Zea mays L. |
| Course | BSCS 3-A, Angeles University Foundation, College of Computer Studies |
| Team | Altes, Zylah Klein · Davis, Dominic · Tayer, Catherine P. · Ursua, Walter Vince |
| Deployment | Android mobile application (TFLite) |
| Clinical task | Detect MSV and MLN in yellow corn leaves with explainable, quantified diagnosis |
| Clinical context | MSV first confirmed in the Philippines (Bukidnon, South Cotabato) in 2023 — Cruz et al. 2024. Early symptoms visually indistinguishable from nutrient deficiencies. Targets agriculture students and smallholder farmers. |

### 1.2 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5060, 8 GB VRAM |
| CPU | AMD Ryzen 5 3600X |
| RAM | 16 GB DDR4 |
| OS | Windows 11 + WSL2 Ubuntu — all Python runs in WSL2 |
| DataLoader workers | 4 (WSL2 /dev/shm constraint) |

### 1.3 Software Stack

| Library | Version | Role |
|---|---|---|
| PyTorch + torchvision | ≥ 2.1.0 | Primary training framework |
| segmentation-models-pytorch | ≥ 0.3.3 | UNet and DeepLabV3+ models |
| albumentations | ≥ 1.3.1 | Augmentation pipeline (2.0 API used throughout) |
| timm | ≥ 0.9.12 | Pretrained encoder variants |
| opencv-python | ≥ 4.8.0 | HSV masking, morphological ops |
| Pillow | ≥ 10.0.0 | Image loading + EXIF handling |
| grad-cam | ≥ 1.4.8 | Grad-CAM, Grad-CAM++, Score-CAM |
| imagehash | ≥ 4.3.1 | pHash deduplication |
| scikit-learn | ≥ 1.3.0 | Metrics, LinearSVC for Gabor baseline |
| scipy | ≥ 1.11.0 | Spearman correlation |
| ultralytics | ≥ 8.0.0 | YOLOv8n leaf detector |
| anomalib | ≥ 1.0.0 | PatchCore baseline (optional, offline only) |
| onnx + onnxruntime | ≥ 1.14.0 | ONNX export |
| tensorflow | ≥ 2.13.0 | TFLite conversion only |
| matplotlib | ≥ 3.7.0 | Chart generation |
| SAM2 | from GitHub | Tier 1 leaf silhouette masking |
| pycocotools | latest | RLE brush-mask decoding for symptom annotations |
| pandas, numpy, tqdm | latest | Data handling, progress bars |

### 1.4 Global Settings
```python
SEED                = 42      # All scripts: random, numpy, torch, cuda
CUDNN_DETERMINISTIC = True    # Reproducibility over speed
CUDNN_BENCHMARK     = False
NUM_WORKERS         = 4       # WSL2 /dev/shm constraint
```

### 1.5 Albumentations 2.0 API Changes
All scripts use the 2.0 API. Key changes from 1.x:
```python
# PadIfNeeded: value → fill + fill_mask
A.PadIfNeeded(H, W, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0)

# CoarseDropout: max_holes/max_height/max_width/min_holes/fill_value →
A.CoarseDropout(num_holes_range=(1, 6),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0, p=0.2)
```

---

## PART 2 — DATASET

### 2.1 Maize Dataset

| Source | Content | Location |
|---|---|---|
| Maize Imagery Dataset — Tanzania (Mduma 2023, Mendeley) | HEALTHY, MSV, MLN leaf images | `maize_dataset/HEALTHY/`, `MSV/`, `MLN/` |
| Zenodo healthy + MSV samples | Additional maize images | Merge into above |

Post-preprocessing distribution: HEALTHY ~96k (38%), MLN ~96k (38%), MSV ~60k (24%), total ~252k.

### 2.2 Non-Maize Dataset

| Source | Location |
|---|---|
| Intel Image Classification | `dataset/raw_kaggle/intel/` |
| Natural Images | `dataset/raw_kaggle/natural/` |
| PlantVillage (rice, sorghum) | `dataset/crop_neighbors/rice/`, `sorghum/` |
| iNaturalist Philippines (cogon grass, banana leaf, sugarcane) | `dataset/crop_neighbors/cogon_grass/`, `banana_leaf/`, `sugarcane/` |
| Mendeley Maize-Weed | `dataset/crop_neighbors/` |

Target: 25,000 maize + 25,000 non-maize = 50,000 balanced Bouncer dataset.

### 2.3 Global Split
```
Strategy : Single 70/15/15 stratified split per class
Source   : global_split_manifest.csv (columns: source_path, category, split)

train 70%  — weight updates, augmentation, backpropagation
val   15%  — early stopping, LR scheduling, checkpoint selection
test  15%  — final evaluation ONLY, accessed once

Test-split images NEVER appear in:
  Tier 1 sampling · Bouncer positive class · Teacher training · Factory processing
```

### 2.4 Annotation Tasks

One Label Studio / CVAT project, two passes, same 501 images:

| Pass | Tool | Labels | Count | Export | Consumers |
|---|---|---|---|---|---|
| 1 — Leaf silhouette | Label Studio polygonlabels | `leaf` | All 501 | `annotations.json` | `train_yolo_detector.py`, `validate_gold_standard.py` |
| 2 — Symptom regions | CVAT COCO 1.0 | `maize-leaf`, `msv-symptom`, `mln-symptom` | ≥ 400 MSV+MLN | `symptom_annotations.json` | `train_symptom_model.py`, `validate_symptom.py` |

> `sample_yolo_annotations.py` is deprecated. `YOLO_IMAGES_DIR`, `YOLO_ANNOTATIONS_DIR`, `YOLO_ANNOTATION_FILE` in `config.py` all point to `data/gold_standard/` paths. `train_yolo_detector.py` derives bboxes from `annotations.json` automatically.

> If Pass 2 falls short of the ≥ 400 MSV+MLN target within the 501-image set, extra images can be annotated into `data/symptom_extra/images/` (`SYMPTOM_EXTRA_IMAGES_DIR`) — same CVAT task, same `symptom_annotations.json` export. `train_symptom_model.py` / `validate_symptom.py` search `[GOLD_IMAGES_DIR, SYMPTOM_EXTRA_IMAGES_DIR]` and skip whichever doesn't exist.

---

## PART 3 — IMAGE UTILITIES (image_utils.py)

All image loading passes through this module. No other script calls `cv2.imread()` or `PIL.Image.open()` directly.

```
load_image_rgb(path)
  PIL.Image.open() → .load() (full decode, catches truncation)
  ImageOps.exif_transpose() → np.array(img.convert("RGB"), uint8)
  Returns None on any failure.

load_image_clahe(path)
  load_image_rgb() + apply_clahe()
  Used by: Bouncer inputs, Factory inputs
  NOT used by: Teacher training, Student training

apply_clahe(img_rgb)
  RGB → LAB → CLAHE on L channel (clipLimit=2.0, tileGridSize=(8,8)) → RGB
  Citation: Zuiderveld (1994)

to_hsv(img_rgb)    → cv2.COLOR_RGB2HSV   (guaranteed correct direction)
to_gray(img_rgb)   → cv2.COLOR_RGB2GRAY
rgb_to_bgr(img)    → cv2.COLOR_RGB2BGR   (for cv2.imwrite() ONLY)
bgr_to_rgb(img)    → cv2.COLOR_BGR2RGB
```

---

## PART 4 — PREPROCESSING (partition_dataset.py)

Run once. Never re-run after training begins.

### Ten-Step Validation Pipeline
```
Step 1  Zero-byte / tiny file      stat().st_size < 100 bytes → reject
Step 2  Magic bytes                JPEG: FF D8 FF · PNG: 89 PNG · mismatch → reject
Step 3  Truncation                 PIL.load() forces full decode · exception → reject
Step 4  Minimum resolution         min(w,h) < 64px → reject
Step 5  Maximum resolution         max(w,h) > 4096px → reject
Step 6  Extreme aspect ratio       max(w,h)/min(w,h) > 8.0 → reject
Step 7  Colour mode                mode=="1" → reject · "L"/"P" → flag, keep
Step 8  Near-uniform               RGB std < 5.0 → reject
Step 9  MD5 exact duplicates       same hash across classes → reject both
                                   same hash within class → reject duplicate
Step 10 pHash near-duplicates      Hamming ≤ 2 within class → reject duplicate
                                   Hamming ≤ 2 across classes → reject both
```

Additionally: green-content flag (< 5% green pixels → flagged, kept).

Outputs: `global_split_manifest.csv`, `preprocessing_report.csv`, `preprocessing_flagged.csv`, `preprocessing_summary.txt`. Rejected files are moved (never deleted) to `quarantine/<reason>/<class>/` at project root, via `quarantine_file()`.

---

## PART 5 — BOUNCER (Phase 0)

### 5.1 Purpose
Binary maize-vs-not-maize gate. Runs first at inference to reject non-maize images.

### 5.2 Dataset
25,000 maize (train+val only, test excluded) + 25,000 non-maize (validated through Steps 1–6 of preprocessing). Internal 80/20 split, seeded. Image loading: `load_image_clahe()`.

### 5.3 4 Variants

| Variant | Architecture | Deployed? |
|---|---|---|
| gabor_lbp | Gabor + LBP + LinearSVC | No — traditional CV baseline |
| mobilenet_v2 | MobileNetV2 head: Linear(1280→1) | No — neural comparison |
| **mobilenet_v3_large** | MobileNetV3-Large head: Linear(960→1) | **Yes** |
| edgevit_xxs | EdgeViT-XXS binary classifier | No — hybrid ViT |

### 5.4 Neural Hyperparameters
```
Loss               : BCEWithLogitsLoss
Optimizer          : AdamW(lr=1e-4, weight_decay=1e-4)
Scheduler          : CosineAnnealingLR(T_max=15, eta_min=1e-6)
Epochs             : 15
Batch size         : 64
Val split          : 80/20 internal (seed=42)
Early stop         : patience=5, monitors val F1
Gradient clipping  : clip_grad_norm_(max_norm=5.0)
safe_collate       : yes
```

### 5.5 Training Augmentation
```python
A.LongestMaxSize(224)
A.PadIfNeeded(224, 224, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0)
A.HorizontalFlip(p=0.5)
A.RandomRotate90(p=0.3)
A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5)
A.HueSaturationValue(H±15, S±20, V±10, p=0.3)
A.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
ToTensorV2()
```

### 5.6 Threshold Selection
ROC on val split → `argmax(sqrt(recall × specificity))` subject to maize_recall ≥ 0.95, hard cap at 0.70. Specificity-alone maximisation was abandoned — it produced near-1.0 thresholds and >88% rejection rates.

### 5.7 Shared Bouncer Inference (scripts/bouncer_inference.py)
```python
BOUNCER_INFER_TF = A.Compose([
    A.LongestMaxSize(BOUNCER_IMG_SIZE),
    A.PadIfNeeded(BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE,
                  border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
    A.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ToTensorV2(),
])

heuristic_prefilter(img_rgb) → bool   # always True (passthrough)
neural_bouncer(img_rgb, model, threshold) → bool   # @torch.no_grad()
```
No heavy imports — shared safely between `train_bouncer.py` and `factory_master.py`.

---

## PART 6 — TIER 1 SAMPLING (sample_15000.py)

```
HEALTHY : 3,000  pure random (random.seed(42))
MSV     : 7,500  evenly-spaced indices across sorted filenames
MLN     : 4,500  evenly-spaced indices
TOTAL   : 15,000  — train+val only, test excluded

Hash guard: SHA-256 of global_split_manifest.csv → data/tier1_raw/_manifest_hash.txt
            Aborts if manifest changes after first run.
```

---

## PART 7 — GOLD STANDARD EXPORT (sample_gold_standard.py)

501 images (167/class) from Tier 1, renamed `{CLASS}_{original}.jpg`. SHA-256 guard on `tier1_manifest.csv`.

**Three downstream consumers of `annotations.json`:**
1. `train_yolo_detector.py` — polygon→bbox (min/max of polygon x,y) → YOLOv8n training
2. `validate_gold_standard.py` — polygon→`cv2.fillPoly()`→binary mask → IoU chain
3. `train_yolo_detector.py` (calibration) — YOLO+SAM2 on gold images → calibrate SAM2 QA threshold

---

## PART 8 — YOLO LEAF DETECTOR (train_yolo_detector.py)

### 8.1 Purpose
Trains YOLOv8n (single-class "leaf") from polygon annotations. Provides tight bounding-box prompts for SAM2 v3. Pure HSV prompting fails on diseased images — MSV yellow / MLN brown share hue ranges with tropical soil.

### 8.2 Key Config
```python
YOLO_IMG_SIZE       = 640
YOLO_EPOCHS         = 100
YOLO_BATCH_SIZE     = 8
YOLO_LR0            = 0.01
YOLO_PATIENCE       = 20
YOLO_MIN_ANNOTATIONS = 400     # warns if below this
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_NMS        = 0.45
```

### 8.3 Validation Metrics
`evaluate_yolo()` runs on the val split → mAP@0.5, mAP@0.5:0.95, precision, recall (Ultralytics `model.val()`). Written to `logs/yolo_val_metrics.csv`. Warns if mAP@0.5 < 0.70 (annotation count may be below `YOLO_MIN_ANNOTATIONS`).

### 8.4 SAM2 QA Calibration
After YOLO training, runs YOLO+SAM2 on gold-standard images → sweeps confidence thresholds 0.50–0.95 in 0.01 steps → for each threshold, computes mean IoU(SAM2 mask, human polygon mask) over images with mean_conf ≥ threshold → picks the **lowest** threshold that reaches `GOLD_IOU_TARGET_MEAN` (0.85), maximizing data retained while meeting the quality bar (fallback 0.65 if the target is never reached). Threshold sweep curve → `logs/yolo_qa_calibration.csv` (read by `generate_tier1_masks.py` at runtime); raw per-image (mean_conf, IoU) points → `logs/yolo_calib_raw.csv`.

---

## PART 9 — SAM2 MASKING (generate_tier1_masks.py) [v3]

### 9.1 Prompting Strategy
```
1. Run YOLOv8n → tight leaf bbox (x1, y1, x2, y2)
2. Restrict HSV tissue search to pixels inside bbox
3. Build 3 foreground points along vertical leaf axis, clamped to bbox
4. Pass bbox as SAM2 box= prompt (hard spatial constraint)
5. Fallback to full-image HSV if YOLO absent or fails
6. Four image corners (10px inset) → background prompts (label=0)
7. Select mask with highest SAM2 score → sigmoid → float32 probability map
```

### 9.2 QA Filters (runtime values)
```python
SAM2_QA_MIN_COVERAGE     = 0.03    # reject if fg < 3% of image
SAM2_QA_MAX_COVERAGE     = 0.90    # reject if fg > 90%
# High-coverage gate: if coverage >= 0.97, require mean_conf >= 0.80
SAM2_QA_MIN_CONFIDENCE   = 0.65    # replaced by calibrated value from yolo_qa_calibration.csv
SAM2_QA_MIN_ASPECT_RATIO = 1.01    # effectively disabled (was 1.20 in v1)
SAM2_QA_MAX_REJECT_RATE  = 0.08    # warn if > 8% rejected
```

### 9.3 Outputs
```
data/tier1_leaf_masks/{stem}_softmask.npy   float32 [0,1] probability map
data/tier1_leaf_masks/{stem}_mask.png       uint8 binary visualization
tier1_qa_report.csv   columns: filename, category, status, reason,
                                prompt_mode (yolo|hsv_fallback), yolo_box,
                                coverage, mean_conf
```

---

## PART 10 — TEACHER (train_teacher.py)

### 10.1 Purpose
Offline segmentation model. Never deployed. Trains on SAM2 float32 probability maps. Generates leaf silhouette pseudo-masks for ~215k Tier 2 images via Factory.

### 10.2 BoundaryAwareLoss
```python
class BoundaryAwareLoss(nn.Module):
    BOUNDARY_WEIGHT = 5.0    # upweight boundary pixels in BCE term
    SHARPEN_FACTOR  = 1.3    # amplify soft SAM2 targets before loss (clip to 1.0)
    DICE_WEIGHT     = 1.0
    BCE_WEIGHT      = 0.5

    def forward(self, logits, targets):
        targets_sharp = (targets * self.SHARPEN_FACTOR).clamp(0.0, 1.0)
        l_dice = DiceLoss(logits, targets_sharp)

        # Boundary detection: max_pool − avg_pool (pure PyTorch, no kornia)
        boundary = (F.max_pool2d(targets_sharp, 3, 1, 1)
                    - F.avg_pool2d(targets_sharp, 3, 1, 1)).clamp(0, 1)
        weight = 1.0 + self.BOUNDARY_WEIGHT * boundary

        l_bce = F.binary_cross_entropy_with_logits(logits, targets_sharp, weight=weight)
        return self.DICE_WEIGHT * l_dice + self.BCE_WEIGHT * l_bce
```
Addresses systematic undersegmentation at leaf margins observed in gold-standard chain validation (SAM2 0.9425 → Teacher 0.7177).

**Important context:** The Teacher val Dice of ~0.97 (on SAM2-mask validation) vs gold-standard IoU of 0.72 is NOT a convergence failure. The gap reflects SAM2 mask style vs human polygon style — the Teacher learned SAM2 faithfully. More epochs do not close this gap.

### 10.3 4 Variants

| Variant | Encoder | Decoder | Notes |
|---|---|---|---|
| resnet50 | ResNet-50 | UNet | Deep CNN baseline |
| **efficientnet-b2** | EfficientNet-B2 | UNet | **Deployed** — best Dice/VRAM ratio |
| mit_b2 | SegFormer-B2 | UNet | Hierarchical transformer encoder |
| deeplabv3plus-eb2 | EfficientNet-B2 | DeepLabV3+ | Decoder architecture comparison |

### 10.4 Hyperparameters
```python
TEACHER_IMG_SIZE              = 768     # increased from 512 for boundary detail
TEACHER_EPOCHS                = 30      # model converges by ep 18–25; more is wasteful
TEACHER_LR                    = 5e-5
TEACHER_WEIGHT_DECAY          = 5e-4
TEACHER_PATIENCE              = 7       # val Dice early stop
TEACHER_GRAD_ACCUM_STEPS      = 2
TEACHER_BATCH_SIZE_OVERRIDES  = {
    "resnet50":          6,
    "efficientnet-b2":   4,
    "mit_b2":            2,
    "deeplabv3plus-eb2": 3,
}
```

All models: `decoder_dropout=0.2` (combats overfitting on soft targets), `decoder_use_batchnorm=True`.

### 10.5 Training Augmentation (Albumentations 2.0)
```python
A.LongestMaxSize(768)
A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0)
A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)
A.RandomBrightnessContrast(0.2, 0.2, p=0.3)
A.HueSaturationValue(H±10, S±20, V±10, p=0.2)
A.ElasticTransform(alpha=60, sigma=6, p=0.3)     # deforms boundary → robust edge features
A.CoarseDropout(num_holes_range=(1,6), hole_height_range=(16,32),
                hole_width_range=(16,32), fill=0, p=0.2)
A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
ToTensorV2()
```

### 10.6 CLAHE Guard
Teacher DataLoader uses `load_image_rgb()` — NOT `load_image_clahe()`. SAM2 generated the soft targets from EXIF-corrected but non-CLAHE images. Applying CLAHE to Teacher inputs would create an image/target mismatch at boundaries.

### 10.7 Data Distribution Check
`build_sample_list()` prints per-class counts and warns if total < 10,000 before training starts. Catches pipeline issues before a long run.

### 10.8 Outputs
```
logs/teacher_{variant}_metrics.csv   per-epoch loss/Dice/IoU/Recall curves
logs/teacher_comparison.csv          best_dice per variant (select_best_pipeline.py primary metric)
logs/teacher_test_metrics.csv        held-out test Dice/IoU/Recall/Precision/Specificity
reports/teacher_overlays/            5 qualitative overlays/class from the test split
```
Note: `reports/teacher_overlays/` is a separate, earlier qualitative check on the test split. The overlays actually embedded in `evaluation_report.html` (Section 4) come from `validate_gold_standard.py`'s `reports/gold_standard_overlays/` — SAM2/Teacher/Student predictions vs. human ground truth, which is the comparison that matters for the thesis chain-validation claim.

---

## PART 11 — SYMPTOM TEACHER (train_symptom_model.py)

### 11.1 Motivation
Eight rounds of LAB/HSV colour-threshold tuning (v1→v8 in `factory_master.py`) confirmed a structural ceiling. Static colour rules cannot separate early chlorosis from healthy yellow-maize tissue, or distinguish tip-burn from MLN necrosis, regardless of how many morphological exceptions are added.

### 11.1b Extra Annotation Source
If the 501 gold-standard images don't reach `SYMPTOM_MIN_ANNOTATIONS` (400) MSV+MLN symptom regions, additional images can be annotated and dropped in `data/symptom_extra/images/` (`SYMPTOM_EXTRA_IMAGES_DIR`) — same CVAT task, same `symptom_annotations.json` export. `parse_symptom_annotations()` accepts `images_dir` as a single `Path` or a `list[Path]` and searches `[GOLD_IMAGES_DIR, SYMPTOM_EXTRA_IMAGES_DIR]` in order, silently dropping any directory that doesn't exist yet — this is an optional, additive source, not a requirement.

### 11.2 HealthyAE
```
Purpose    : Trains on HEALTHY images only. Per-pixel reconstruction error
             normalized to [0,1] becomes 4th input channel for Symptom Teacher.
             Lighting-invariant anomaly prior — AE learns the manifold of
             healthy appearance. AE output is an INPUT FEATURE, not a label.
Architecture: Conv autoencoder with true bottleneck (no skip connections)
Input size : 256×256 (HEALTHY_AE_IMG_SIZE)
Latent dim : 128 (HEALTHY_AE_LATENT_DIM)
Batch      : 16
Epochs     : 40
Optimizer  : Adam(lr=1e-3)
Loss       : MSELoss
Patience   : 8
Checkpoint : checkpoints/healthy_ae/healthy_ae_best.pth
```

### 11.3 Symptom Teacher
```
Architecture: smp.Unet(encoder_name=SYMPTOM_ENCODER, in_channels=4)
              SYMPTOM_ENCODER derived from TEACHER_DEPLOYED_VARIANT in config.py
              (default: "efficientnet-b2"). Never hardcoded.
Input      : 4 channels — RGB (3) + AE reconstruction error map (1)
Ground truth: human polygon/brush masks (NOT AE output — never label source)
Input size : 512×512 (SYMPTOM_IMG_SIZE)
Batch      : 4
Epochs     : 60
Optimizer  : AdamW(lr=1e-4, wd=1e-4)
Loss       : 0.6×DiceLoss + 0.4×FocalLoss(gamma=2.0, pos_weight=5.0)
             Focal loss upweights sparse foreground pixels (thin streaks)
Patience   : 10
Target IoU : ≥ 0.70 (SYMPTOM_IOU_TARGET_MEAN)
Checkpoints: checkpoints/healthy_ae/healthy_ae_best.pth
             checkpoints/symptom/symptom_teacher_best.pth
```

### 11.4 Annotation Format (CVAT COCO 1.0)
```
Labels (case-insensitive):
  maize-leaf   (MAIZE_LEAF_LABEL_NAME)  — leaf silhouette, all 3 classes
  msv-symptom  (MSV_SYMPTOM_LABEL_NAME) — chlorotic streaks, MSV only
  mln-symptom  (MLN_SYMPTOM_LABEL_NAME) — necrotic patches, MLN only

HEALTHY images: annotate maize-leaf only (no symptom regions needed)
               → training uses all-zero symptom masks for HEALTHY
               → teaches model to output nothing on a healthy leaf

Export: CVAT → Task menu → Export dataset → COCO 1.0 → extract zip
Place: data/gold_standard/annotations/symptom_annotations.json
pip install pycocotools  (for RLE brush-mask decoding)
```

### 11.5 Factory Integration
```python
# config.py
SYMPTOM_TEACHER_DEPLOYED    = True   # uses Symptom Teacher in Factory
FACTORY_SYMPTOM_COMPARE_LAB = False  # set True for thesis comparison figures

# factory_master.py behavior:
# if SYMPTOM_TEACHER_DEPLOYED and checkpoints exist → use Symptom Teacher
# else → legacy LAB fallback (automatically)
```

---

## PART 12 — FACTORY (factory_master.py)

### 12.1 Purpose
Process ~215k train+val Tier 2 images. Generate pseudo-labels across 4 modes simultaneously into separate subfolders.

### 12.2 Per-Image Pipeline
```
1. Bouncer gate (scripts/bouncer_inference.py):
   heuristic_prefilter() → True (passthrough)
   neural_bouncer() → True/False
   Rejected → log status, skip

2. Leaf silhouette source:
   Tier 1 → load pre-existing SAM2 .npy
   Tier 2 → Teacher inference at TEACHER_IMG_SIZE (768×768) → resize to original

3. Silhouette refinement:
   Threshold at 0.35 (catches dark leaves)
   Morphological close then open (kernel=5, elliptical)

4. Coverage guard → reliability weight:
   < 15%  : weight = None (exclude, sev = -1)
   15–25% : weight = 0.30
   25–50% : weight = 0.70
   > 50%  : weight = 1.00

5. Symptom masking:
   If SYMPTOM_TEACHER_DEPLOYED and checkpoints exist:
     → run HealthyAE → get AE error map
     → run Symptom Teacher on [RGB + AE error]
   Else:
     → legacy LAB/HSV fallback

6. Severity = (symptom pixels / leaf pixels) × 100

7. CIMMYT/agronomic grade (per class), via sev_to_cimmyt_grade():
   MSV scale (1-5, Soto et al. 1982; validated Sime et al. 2021):
     1=≤10% chlorotic area · 2=11-25% · 3=26-50% · 4=51-75% · 5=≥75%
   MLN scale (1-5, Beyene et al. 2017; Gowda et al. 2015):
     1=<10% · 2=10-25% · 3=25-50% · 4=50-75% · 5=>75%
   HEALTHY: grade = 0. Excluded images (severity < 0): grade = -1.
   NOTE: a published CIMMYT 1-9 odd-numbered scale also exists for MSV, but
   it is used for whole-plant visual resistance scoring by breeders, not
   for leaf-area percentage mapping — CIMMYT_MSV_BRACKETS in config.py
   explicitly uses the 0-5 Soto/IITA scale instead, matching MLN's scale
   structure. Do not confuse the two.
```

### 12.3 4 Factory Modes

| Mode | Silhouette | Symptom | Subfolder |
|---|---|---|---|
| A | Otsu binary | Hard binary | `mode_a/` |
| B | SAM2 hard binary | Hard binary | `mode_b/` |
| C | SAM2 soft float | Hard binary | `mode_c/` |
| D | SAM2 soft float | Soft HSV confidence | `mode_d/` |

### 12.4 Legacy Symptom Fallback — LAB, not HSV (used when Symptom Teacher absent)

**Correction:** `HSV_MSV_RANGES` / `HSV_MLN_RANGES` / `HSV_GREEN_EXCL` (below) are defined in `config.py` and implemented as `compute_hsv_hard_mask()` / `compute_hsv_soft_confidence()` in `factory_master.py`, but **neither function is called anywhere in the pipeline** — they are orphaned code from an earlier (pre v4→v5) iteration. The function actually used as the Symptom-Teacher fallback is the **LAB-based** `compute_lab_hard_mask()` / `compute_lab_soft_confidence()` (v8), which is considerably more elaborate than simple HSV banding:

```
Green exclusion : a* < LAB_GREEN_A_MAX (121)  — greener pixels treated as healthy tissue
Frangi vesselness: skimage.filters.frangi on L*, sigmas FRANGI_SIGMAS=(0.5,1.5,3.0,6.0,10.0),
                   soft-suppresses vein-ridge false positives — (1 − sqrt(vesselness)).
                   Threshold: HEALTHY 0.6 (most aggressive) · MSV 0.5 · MLN 0.35 (necrosis
                   crosses veins, so suppression must be gentler) — hardcoded per-class.
Morphology       : _multiscale_symptom_union() — small (1×7) opening always runs; large
                   (1×15) angled-kernel union (MORPH_OPEN_ANGLES) added only when
                   symptomatic area > 8% of leaf, so early flecks survive.
MSV threshold    : 2D joint Otsu over the (a*, b*) histogram within the leaf mask
                   (OTSU_2D_BIN_COUNT=64 bins/axis; falls back to independent 1D Otsu
                   per channel when leaf pixel count < 200) — catches cases where
                   neither channel alone is elevated but their combination is.
MSV texture      : Gabor filter combines grayscale texture with a*-channel texture
                   (GABOR_A_CHANNEL_WEIGHT=0.4) — chromatic texture of chlorotic
                   streaks is more discriminative than luminance under overcast light.
MLN dark necrosis: additional gate (L* < 110) & (a* ≥ 125) for dark necrotic patches.
MLN margin guard : convex-hull margin erosion (MARGIN_EROSION_FRAC=0.08 × sqrt(leaf
                   area) px) suppresses tip-burn false positives (physiologically
                   distinct from MLN) — applied to MLN branch only, not MSV/HEALTHY.
CLAHE            : L* CLAHE (clipLimit 2.0) then a* CLAHE (CLAHE_A_CLIP_LIMIT=1.0,
                   lower to avoid chromatic noise amplification) within the leaf mask.
```
This LAB pipeline is itself the legacy path (superseded by the Symptom Teacher, active when `SYMPTOM_TEACHER_DEPLOYED=True` and checkpoints exist) — but it is the *real* fallback and comparison baseline, not the orphaned HSV bands below.

**Orphaned HSV bands (dead code, kept for reference only — not called):**

Green exclusion zone: H 38–85, S 80–255, V 60–230

MSV (4 bands): R1 H15–38/S50–255/V150–255 (bright streaks) · R2 H20–45/S15–70/V130–255 (pale/early) · R3 H0–179/S0–35/V210–255 (near-white, ≥80px area filter) · R4 H38–55/S10–55/V140–255 (pale yellow-green)

MLN (5 bands): R1 H18–40/S40–255/V90–255 (chlorotic yellow) · R2 H5–20/S40–255/V50–220 (orange-amber necrosis) · R3 H22–55/S15–85/V80–240 (pale mosaic) · R4 H8–35/S0–45/V160–255 (tan/straw) · R5 H0–18/S30–180/V30–150 (dark brown dead tissue)

### 12.5 Output Files per Image per Mode
```
{stem}_silhouette.npy    float32 [0,1] leaf silhouette probability
{stem}_symptom.npy       float32 confidence map (mode_d) / Symptom Teacher output
{stem}_symptom.png       uint8 binary mask (modes a, b, c)
{stem}_sev.txt           severity % (float) or -1 (excluded)
{stem}_grade.txt         CIMMYT grade (int) or -1
{stem}_weight.txt        reliability weight (0.30/0.70/1.00) or -1

All outputs downscaled to STUDENT_IMG_SIZE (224×224) at write time.
Silhouettes: INTER_LINEAR. Binary symptom PNGs: INTER_NEAREST.
```

Also written once per full run (not per mode):
```
reports/phase4_report.csv          per-image unified report — all 4 modes' severity/
                                    grade/weight columns for every processed image
reports/factory_summary.csv        pass/reject counts + rates
reports/factory_filter_breakdown.csv  rejection reason breakdown
```

### 12.6 Factory QA — validate_factory.py
Run after `factory_master.py`, before `train_student.py`:
```bash
python validate_factory.py --all-modes   # HTML report all 4 modes
python validate_factory.py --mode mode_b # single mode
python validate_factory.py --borderline  # near-threshold severity cases
```
Expected patterns: HEALTHY masks near-empty (<4% fill), MSV narrow vein-parallel streaks, MLN wider diffuse yellowing/necrosis from margins. Failure patterns map to specific `config.py` parameters in the script's docstring.

---

## PART 13 — STUDENT (train_student.py)

### 13.1 Architecture
```
Input: [B, 3, 224, 224] float32 (ImageNet normalized)
    ↓
Shared Encoder (MobileNetV2 or variant)
    ├── UNet Decoder (+ CBAM at skip connections for V2, V6)
    │       ↓
    │   Segmentation Head → [B, 2, 224, 224] raw logits
    │       Ch0: leaf silhouette  (sigmoid → binary at 0.5)
    │       Ch1: symptom mask     (sigmoid → binary at 0.5)
    │
    ├── GAP → Dropout(0.3) → Linear(→3)
    │       [B, 3] raw logits → softmax → argmax
    │       0=HEALTHY / 1=MSV / 2=MLN
    │
    └── GAP → Dropout(0.3) → Linear(→1) → ReLU → clamp(0,1)
            [B, 1] → ×100 at inference = severity %
```

**Why ReLU+clamp:** Sigmoid never reaches 0.0. HEALTHY leaves would always show nonzero severity (biologically incorrect). ReLU allows exactly 0.0.

### 13.2 5 Encoder Variants

| V# | Encoder | CBAM | TFLite |
|---|---|---|---|
| V1 | mobilenet_v2 | No | Yes |
| **V2** | **mobilenet_v2_cbam** | **Yes** | **Yes** |
| V3 | mobilenet_v3_small | No | Yes |
| V4 | efficientnet_b0 | No | Yes |
| V6 | efficientnet_b0_cbam | Yes | Yes |

V5 (MobileViT-XXS) removed — `torch.einsum` self-attention causes TFLite subgraph errors.

### 13.3 CBAM Implementation
```python
class CBAMBlock(nn.Module):
    # Channel attention: GAP+GMP → shared MLP → sigmoid → scale x
    # Spatial attention: [avg_pool, max_pool along channels] → concat → 7×7 conv → sigmoid → scale x

# CBAMUnetDecoder subclasses smp.decoders.unet.decoder.UnetDecoder
# _apply_cbam_to_features() applied to skip connections in forward()
# NOT monkey-patching — preserves torch.save() / state_dict() serialization
# Citation: Woo et al. (2018) ECCV
```

### 13.4 Loss Functions

**A. Segmentation — DiceLoss (float targets)**
```python
smp.losses.DiceLoss(mode="binary", from_logits=True)
# Accepts float32 [0,1] — soft boundary uncertainty preserved
l_seg = (DiceLoss(logits[:,0:1], tgt_sil) + DiceLoss(logits[:,1:2], tgt_sym)) / 2
```

**B. Classification — Asymmetric Label Smoothing**
```python
ASYMMETRIC_PRIOR = [
    [0.90, 0.08, 0.02],   # HEALTHY → most confusion with MSV (Cruz et al. 2024)
    [0.05, 0.90, 0.05],   # MSV
    [0.02, 0.05, 0.93],   # MLN — most visually distinct
]
l_cls = −Σ ASYMMETRIC_PRIOR[true_class] × log(softmax(logits))
```

**C. Severity — Reliability-Weighted MSE**
```python
# weight from Factory coverage bracket (-1 excluded entirely)
l_sev = (sample_weights × MSE(sev_pred, sev_target)).mean()
# Citation: Jiang et al. (2018) ICML — MentorNet
```

**D. Multi-task Balancing — Homoscedastic Uncertainty**
```python
# 3 learnable log-variance params: s1 (seg), s2 (cls), s3 (sev)
L_total = exp(−s1)·L_seg + s1
        + exp(−s2)·L_cls + s2
        + exp(−s3)·L_sev + s3
# Citation: Kendall et al. (2018) NeurIPS
```

### 13.5 Checkpoint Criterion (Quality Composite)
```python
# CIMMYT-stratified MSV F1 replaces aggregate MSV_F1
# Grade tiers aligned with CIMMYT_MSV_BRACKETS in config.py
composite = (
    STUDENT_CKPT_W_MIOU          * mIoU         +   # 0.40
    STUDENT_CKPT_W_MLN_F1        * MLN_F1        +   # 0.15
    STUDENT_CKPT_W_MAE           * (1−NormMAE)   +   # 0.10
    STUDENT_CKPT_W_MSV_F1_EARLY  * MSV_F1_early  +   # 0.20 (grade 1-3, <25%)
    STUDENT_CKPT_W_MSV_F1_MID    * MSV_F1_mid    +   # 0.10 (grade 3-5, 25-50%)
    STUDENT_CKPT_W_MSV_F1_SEVERE * MSV_F1_severe     # 0.05 (grade 5-9, >50%)
)
# Early-stage MSV (grade 1-3) has 4× weight of severe-stage
# because it is the clinically critical detection case
```

### 13.6 Mobile Composite (select_best_pipeline.py)
```python
# Module-level weight constants W_MSV_F1..W_SIZE in select_best_pipeline.py,
# consumed directly by mobile_composite_score(). Sum = 1.00.
mobile = (0.32 × MSV_F1                                # W_MSV_F1  — primary clinical metric
        + 0.18 × MSV_ROC_AUC                           # W_ROC_AUC — threshold-independent detection
        + 0.18 × sil_mIoU                               # W_MIOU    — segmentation/XAI overlay quality
        + 0.16 × clamp(150ms / cpu_lat_mean_ms, 0, 1)   # W_SPEED   — speed_score
        + 0.08 × MLN_F1                                 # W_MLN_F1  — prevents degenerate MSV-only model
        + 0.05 × (1 − sev_mae_pct / 100)                # W_SEV     — severity calibration
        + 0.03 × clamp(15MB / tflite_size_mb, 0, 1))    # W_SIZE    — size_score
# size_score defaults to 1.0 (optimistic) if TFLite size is not yet known —
# noted in the report rather than arbitrarily penalising unexported variants.
# msv_roc falls back to msv_f1 if ROC-AUC hasn't been computed yet.
# TFLite-incompatible variants (TFLITE_VARIANTS_INCOMPATIBLE = {"mobilevit_xxs"})
# excluded from ranking. Targets: TARGET_LATENCY_MS=150, TARGET_SIZE_MB=15.
```

### 13.7 Two-Phase Transfer Learning
```
Phase 1 — Frozen encoder (30 epochs):
  encoder: requires_grad = False
  Adam(decoder + heads + log_vars, lr=1e-3, wd=1e-4)
  CosineAnnealingLR(T_max=30, eta_min=1e-6)
  Early stop patience=10 on quality composite

Phase 2 — Full fine-tune (20 epochs):
  Unfreeze encoder
  Re-initialize optimizer (lr=1e-4 for all params + log_vars)
  CosineAnnealingLR(T_max=20, eta_min=1e-6)
  Early stop patience=5

Gradient clipping: clip_grad_norm_(all params including log_vars, max_norm=5.0)
WeightedRandomSampler: inverse class frequency weights
```

### 13.8 Two-Stage Ablation
```
Stage 1: Fix mode=mode_b, train all 5 encoders → checkpoints/student/stage1/
Stage 2: Fix encoder=best from Stage 1, train all 4 modes → checkpoints/student/
         Mode B result REUSED from Stage 1 (same seed=42 → identical run)
Total unique training runs: 8 (5 + 3)

Stage 1 isolation to stage1/ prevents Stage 2 Mode B from overwriting
the Stage 1 checkpoint for the same encoder+mode name.
```

### 13.9 Validate Student (validate_student.py)
```
Loads: checkpoints/student/stage2/student_{STUDENT_BEST_VARIANT}_{STUDENT_FACTORY_MODE}_best.pth
Source: data/gold_standard/images/ (same 501-image set)
Output: reports/student_overlays/{stem}_student_pred.jpg
Panel: Original | Predicted Silhouette | Predicted Symptom Mask
Footer: GT class / Predicted class / match-mismatch / Severity %
Usage: python validate_student.py --n 5  (default: 5 per class)
Note: qualitative only — not a substitute for student_test_metrics_*.csv
```

---

## PART 14 — GOLD STANDARD VALIDATION (validate_gold_standard.py)

### 14.1 Purpose
Validates pseudo-label quality against 501 human-annotated leaf silhouettes. Required for thesis defense — provides chain comparison on the same images:
```
SAM2 → Teacher → Student
  ↓         ↓         ↓
IoU vs human  IoU vs human  IoU vs human
```

### 14.2 Run Three Times
```bash
# Step 6c — after generate_tier1_masks.py, before train_teacher.py
python validate_gold_standard.py --sam2-only

# Step 6d — after train_teacher.py
python validate_gold_standard.py

# Step 10b — after train_student.py (both stages)
python validate_gold_standard.py
```

### 14.3 Key Config
```python
GOLD_IOU_WARN_THRESHOLD = 0.75    # per-image flag
GOLD_IOU_TARGET_MEAN    = 0.85    # overall validation target
TEACHER_DEPLOYED_VARIANT = "efficientnet-b2"
```

### 14.4 Results Context
```
SAM2    0.9425 mean IoU  ✓ target met
Teacher 0.7177 mean IoU  ✗ below target

The Teacher gap is expected — not a convergence failure. Teacher learned SAM2's
mask style faithfully (val Dice ~0.97). The 0.72 IoU gap to human annotations
reflects SAM2 vs human polygon style differences. BoundaryAwareLoss + 768px
training addresses this for the next run.
```

### 14.5 Outputs
```
reports/gold_standard_iou_report.csv    per-image IoU for all artifacts
reports/gold_standard_iou_summary.csv   mean ± std per class + overall
reports/gold_standard_overlays/         5 overlay PNGs per class per artifact
  {stem}_{artifact}_overlay.jpg  Green=missed, Cyan=correct, Red=extra
```

---

## PART 15 — XAI (evaluate_xai.py)

### 15.1 Three Methods
| Method | Type | Role |
|---|---|---|
| Grad-CAM | Gradient-based | Historical baseline |
| **Grad-CAM++** | Gradient-based | **Deployed** — better for multi-region MSV streaks |
| Score-CAM | Gradient-free | Stability reference |

### 15.2 Target Layers
```python
XAI_TARGET_LAYERS = {
    "mobilenet_v2":          "unet.encoder.features[-1][0]",
    "mobilenet_v2_cbam":     "unet.encoder.features[-1][0]",
    "mobilenet_v3_small":    "unet.encoder.model.blocks[-1][-1]",
    "efficientnet_b0":       "unet.encoder._blocks[-1]",
    "efficientnet_b0_cbam":  "unet.encoder._blocks[-1]",
}
```

### 15.3 Two Distinct App Outputs
```
Output 1 — Symptom boundary (green contour):
  Source: UNet segmentation head Ch1
  Nature: pixel-level localization
  
Output 2 — Diagnostic attention (amber heatmap):
  Source: Grad-CAM++ on last encoder conv block
  Nature: coarse ~7×7 class-discriminative heatmap upsampled to 224×224
  
⚠ These are NOT the same thing. Never conflate in thesis.
```

### 15.4 Metrics
```python
XAI_INSERTION_STEPS = 25    # progressively reveal pixels in importance order
# Pointing game accuracy, Insertion AUC, Deletion AUC — all on GPU
```

### 15.5 Auto-Selection
After computing per-method metrics, auto-selects the empirically best method: highest MSV pointing-game accuracy, tiebreak on Insertion AUC. Written to `logs/xai_method_selection.csv` (selected_method + msv_pg + ins_auc + del_auc + per-method pointing-game scores). This is an evidence-based check against the `XAI_DEPLOYED_METHOD` hardcoded in `config.py` — the two should agree, but the CSV records what the data actually says.

---

## PART 16 — BEST PIPELINE SELECTION (select_best_pipeline.py)

```
Bouncer : max specificity subject to maize_recall ≥ 0.95 (neural only)
Teacher : max val Dice
Student training → quality composite (checkpoint saving)
Student deploy   → mobile composite (post-training, deployment selection)
TFLite-incompatible variants excluded from deployment ranking.

Outputs:
  checkpoints/final/bouncer_best.pth
  checkpoints/final/teacher_best.pth
  checkpoints/final/student_best.pth
  reports/best_pipeline_summary.csv + .txt
  reports/student_mobile_ranking.csv
  reports/all_variants_ranked.csv

Auto-updates config.py via regex:
  STUDENT_BEST_VARIANT = "{winner_encoder}"
  STUDENT_FACTORY_MODE = "{winner_mode}"
```

---

## PART 17 — SEVERITY RELIABILITY (evaluate_severity.py)

### 17.1 Purpose
Grounds the severity MAE disclaimer with empirical evidence: how well do human raters agree with each other, and how well does the pipeline's severity estimate track human judgment.

### 17.2 Protocol
```
1. Sample SEVERITY_EVAL_N_IMAGES (60 = 20/class) from the test split.
2. Two independent raters assign 0–SEVERITY_EVAL_SCALE_MAX (0–3) severity scores:
     0 = No visible symptoms
     1 = Mild     — < 25% of visible leaf area shows symptoms
     2 = Moderate — 25–60% of visible leaf area shows symptoms
     3 = Severe   — > 60% of visible leaf area shows symptoms
3. Cohen's Kappa — inter-rater agreement.
4. Spearman ρ — human ratings vs. pipeline severity (the deployed pseudo-label
   severity, read from data/pseudo_masks/{STUDENT_FACTORY_MODE}/{stem}_sev.txt
   for the current STUDENT_BEST_VARIANT/STUDENT_FACTORY_MODE; the CSV column is
   still named "hsv_severity" for historical reasons even though it reflects
   the Symptom Teacher pseudo-label when SYMPTOM_TEACHER_DEPLOYED=True).

Target: Spearman ρ ≥ 0.60 supports "moderate correlation with expert assessment."
If ρ < 0.50, the severity disclaimer must be made stronger in Chapter 4.
```

### 17.3 Usage
```bash
python evaluate_severity.py --sample     # Step 1: writes reports/severity_sample.csv
                                          # + reports/severity_rating_guide.txt
# → two raters independently fill in the blank score columns
python evaluate_severity.py --analyze    # Step 2: kappa + spearman
```

### 17.4 Outputs
```
reports/severity_sample.csv       60 images for raters to fill in
reports/severity_rating_guide.txt rater instructions (0–3 scale definitions)
reports/severity_analysis.csv     per-image human vs. pipeline severity + correlation
logs/severity_reliability.csv     kappa + spearman for thesis table
```

---

## PART 18 — DEPLOYMENT (export_tflite.py + build_deployment_package.py)

### 18.1 Export Path
```
Student: PyTorch → ONNX (opset 12) → TF SavedModel → TFLite (FP16)
Bouncer: same path
```

### 18.2 Student TFLite I/O
```
Input:   [1, 3, 224, 224] float32, NCHW, ImageNet normalized
Output 0 (seg):  [1, 2, 224, 224] raw logits → sigmoid → binary at 0.5
Output 1 (cls):  [1, 3] raw logits → softmax → argmax (0=HEALTHY, 1=MSV, 2=MLN)
Output 2 (sev):  [1, 1] float32 ∈ [0,1] → ×100 = severity %
```

### 18.3 Bouncer TFLite I/O
```
Input:  [1, 3, 224, 224] float32, NCHW
Output: [1, 1] raw logit → sigmoid → ≥ threshold → PASS
```

### 18.4 Android Runtime
```
Min API     : 21
TFLite      : org.tensorflow:tensorflow-lite:2.13.0
Support     : org.tensorflow:tensorflow-lite-support:0.4.4
GPU delegate: optional

Pipeline:
  EXIF correction → letterbox 224×224 → ImageNet normalize → NCHW
  → Bouncer → (pass) → Student → 3 outputs
  → display: green contour + amber heatmap + class badge + severity gauge
```

### 18.5 Deploy Bundle
```
exports/deploy/
  bouncer_model.tflite
  student_model.tflite
  model_metadata.json      shapes, normalization, thresholds, class names, CIMMYT grades
  DEPLOYMENT_README.md     Kotlin/Java integration guide
  deployment_report.csv    sizes, latencies, validation status
```

---

## PART 19 — CHART GENERATOR (generate_charts.py)

Standalone — no project module imports. Reads `logs/*.csv`. Safe mid-training; missing CSVs silently skipped. Dark theme (navy #0F172A, teal #34D399). Agg backend. 150 DPI PNG.

```bash
python generate_charts.py              # all
python generate_charts.py --bouncer
python generate_charts.py --teacher
python generate_charts.py --student
```

**Colour palette:**
```
gabor_lbp / resnet50 / mode_a    #6B7280  grey
mobilenet_v2 / mode_b             #3B82F6  blue
mobilenet_v3_large / eb2 / mode_c #10B981  emerald
edgevit_xxs / mit_b2 / mode_d    #F59E0B  amber
deeplabv3plus-eb2                 #EF4444  red
mobilenet_v2_cbam                 #8B5CF6  violet
mobilenet_v3_small                #EC4899  pink
efficientnet_b0                   #F97316  orange
efficientnet_b0_cbam              #14B8A6  teal
```

---

## PART 20 — HTML REPORT (generate_report.py)

`reports/evaluation_report.html` — 100% self-contained (charts base64-embedded), dark theme, no external dependencies.

**11 sections:**
1. Pipeline overview + dataset statistics
2. Preprocessing — rejection counts + reasons breakdown
3. Bouncer — comparison bar, ROC table, confusion matrix, admission rate
4. Teacher — Dice/latency bar, training curves, test metrics, overlays
5. Symptom Teacher — HealthyAE anomaly maps, LAB vs Teacher comparison (Table 4.13)
6. Student Stage 1 (encoder ablation) — comparison table + grouped bar
7. Student Stage 2 (mode ablation) — mode A/B/C/D + severity histograms
8. Best Student full results — all metrics, per-class P/R/F1, confusion matrix
9. XAI — pointing game/insertion/deletion bar, overlays, two-output distinction
10. Severity reliability — Kappa + Spearman ρ with quality badges
11. Deployment — TFLite sizes, latency cards, bundle contents

---

## PART 21 — INFRASTRUCTURE

### 21.1 safe_collate
Every DataLoader: `collate_fn=safe_collate`. Filters None `__getitem__` returns. All-None batch returns None; training loop skips with `continue`. `reset_skip_counter()` at epoch start, `get_skip_count()` at epoch end.

### 21.2 Gradient Clipping
All training scripts: `clip_grad_norm_(model.parameters(), max_norm=5.0)`. Student also clips uncertainty log_vars (s1/s2/s3). Protects against log_var spikes and encoder-unfreeze gradient surges.

### 21.3 Reproducibility
Every script calls `set_seeds(42)`: random, numpy, torch, cuda, cudnn.deterministic=True, benchmark=False. Stage 1 Mode B reused in Stage 2: same seed + same manifest + same hyperparameters = identical run.

### 21.4 Timing
Wall-clock duration logged in every script. Student: 220 CPU passes (20 warmup + 200 measured) → mean ms, std ms, FPS. Teacher: 20 CPU passes per variant.

---

## PART 22 — EXECUTION REFERENCE

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2

python partition_dataset.py                          # Step 1 — run once only

python create_bouncer_dataset.py                     # Step 2
python train_bouncer.py                              # Step 3
python generate_charts.py --bouncer                  # optional

python sample_15000.py                               # Step 4

python sample_gold_standard.py                       # Step 4a
# Label Studio: polygonlabels, ALL 501 images → annotations.json

python train_yolo_detector.py                        # Step 4b

python generate_tier1_masks.py                       # Step 5
python validate_masks.py                             # Step 6
python validate_gold_standard.py --sam2-only         # Step 6c

python train_teacher.py                              # Step 7
python generate_charts.py --teacher                  # optional
python validate_gold_standard.py                     # Step 6d

# Step 7b — CVAT symptom annotation (same 501 images, second pass)
# Labels: maize-leaf / msv-symptom / mln-symptom
# Export COCO 1.0 → symptom_annotations.json
python train_symptom_model.py                        # Step 7b
python validate_symptom.py                           # optional

python factory_master.py                             # Step 8
python validate_factory.py --all-modes               # Step 8b (optional, recommended)

python train_student.py --stage 1                    # Step 9
python train_student.py --stage 2 --encoder mobilenet_v2_cbam   # Step 10
python generate_charts.py --student                  # optional
python validate_student.py                           # optional
python validate_gold_standard.py                     # Step 10b

python select_best_pipeline.py                       # Step 11
python evaluate_xai.py                               # Step 12
python evaluate_severity.py --sample                 # Step 13a
python evaluate_severity.py --analyze                # Step 13b
python export_tflite.py                              # Step 14
python build_deployment_package.py                   # Step 15
python generate_report.py                            # Step 16
```

---

## PART 23 — KEY TERMINOLOGY

| Wrong | Correct |
|---|---|
| "Knowledge distillation" | "Pseudo-label semi-supervised learning" or "teacher-guided pseudo-label generation" |
| "Pixel-level segmentation via Grad-CAM" | Grad-CAM++ = coarse ~7×7 class-discriminative heatmap. UNet head = pixel-level segmentation. Entirely distinct outputs. |
| "Severity MAE measures disease severity accuracy" | "Severity MAE measures Student consistency with Factory's pseudo-labels (Symptom Teacher), not expert agronomic ratings" |
| "HSV symptom pipeline" | Legacy LAB/HSV fallback — now replaced by Symptom Teacher. Remains in factory_master.py for comparison only. |

---

## PART 24 — KEY CITATIONS

| Reference | Used for |
|---|---|
| Kendall et al. (2018) NeurIPS | Homoscedastic uncertainty multi-task loss |
| Szegedy et al. (2016) CVPR | Label smoothing |
| Woo et al. (2018) ECCV | CBAM |
| Ke et al. (2020) | Soft pseudo-label segmentation targets |
| Jiang et al. (2018) ICML | MentorNet — reliability-weighted curriculum |
| Lin et al. (2017) ICCV | Focal loss (Symptom Teacher) |
| Cruz et al. (2024) | First confirmed MSV in the Philippines |
| Mushayi et al. (2025) | MSV / HEALTHY confusion — asymmetric prior basis |
| Soto et al. (1982); Sime et al. (2021) Agriculture 11(2):130 | MSV 1-5 severity scale (CIMMYT_MSV_BRACKETS) |
| Beyene et al. (2017) Euphytica 213:224; Gowda et al. (2015); Eunice et al. (2021) | MLN 1-5 severity scale (CIMMYT_MLN_BRACKETS) |
| Selvaraju et al. (2017) ICCV | Grad-CAM |
| Chattopadhay et al. (2018) WACV | Grad-CAM++ |
| Wang et al. (2020) CVPR | Score-CAM |
| Mduma (2023) Mendeley | Tanzania Maize Imagery Dataset |
| Tan & Le (2019) ICML | EfficientNet |
| Xie et al. (2021) NeurIPS | SegFormer (mit_b2) |
| Pan et al. (2022) ECCV | EdgeViT |
| Fawcett (2006) | ROC threshold selection |
| Zuiderveld (1994) | CLAHE |
