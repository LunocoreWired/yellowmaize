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
| Clinical context | MSV first confirmed in the Philippines (Bukidnon, South Cotabato) in 2023 — Cruz et al. 2024. Early symptoms are visually indistinguishable from nutrient deficiencies. Tool targets agriculture students and smallholder farmers. |

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
| albumentations | ≥ 1.3.1 | Augmentation pipeline |
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
| pandas, numpy, tqdm | latest | Data handling, progress bars |

### 1.4 Global Settings
```python
SEED                = 42
CUDNN_DETERMINISTIC = True
CUDNN_BENCHMARK     = False
NUM_WORKERS         = 4
```

---

## PART 2 — DATASET

### 2.1 Maize Dataset

| Source | Content | Location |
|---|---|---|
| Maize Imagery Dataset — Tanzania (Mduma 2023, Mendeley) | HEALTHY, MSV, MLN leaf images | `maize_dataset/HEALTHY/`, `MSV/`, `MLN/` |
| Zenodo healthy + MSV samples | Additional maize images | Merge into above |

Post-preprocessing approximate distribution:

| Class | Images | % |
|---|---|---|
| HEALTHY | ~96,000 | 38% |
| MLN | ~96,000 | 38% |
| MSV | ~60,000 | 24% |
| **TOTAL** | **~252,000** | 100% |

### 2.2 Non-Maize Dataset (Bouncer Negatives)

| Source | Content | Location |
|---|---|---|
| Intel Image Classification | Buildings, forest, glacier, mountain, sea, street | `dataset/raw_kaggle/intel/` |
| Natural Images | Airplane, car, cat, dog, flower, fruit, motorbike, person | `dataset/raw_kaggle/natural/` |
| PlantVillage | Rice and sorghum leaves | `dataset/crop_neighbors/rice/`, `sorghum/` |
| iNaturalist Philippines | Cogon grass, banana leaf, sugarcane | `dataset/crop_neighbors/` |
| Mendeley Maize-Weed | Field weeds photographed in maize | `dataset/crop_neighbors/` |

Target: 25,000 maize + 25,000 non-maize = 50,000 balanced.

### 2.3 Global Split
```
Strategy : Single 70/15/15 stratified split per class
Source   : global_split_manifest.csv (columns: source_path, category, split)

Split    Ratio    Purpose
train    70%      Weight updates, augmentation, backpropagation
val      15%      Early stopping, LR scheduling, checkpoint selection
test     15%      Final evaluation ONLY — accessed once, never in any training decision

Test-split images NEVER appear in:
  - Tier 1 sampling          (sample_15000.py filters by manifest)
  - Bouncer positive class   (create_bouncer_dataset.py filters by manifest)
  - Teacher training         (train_teacher.py filters by manifest)
  - Factory processing       (factory_master.py skips test-split images)
```

### 2.4 Annotation Tasks

One annotation export, two passes, three downstream consumers:

| Task | Script | Count | Per class | Export destination |
|---|---|---|---|---|
| Leaf silhouette polygons (Pass 1) | `sample_gold_standard.py` | 501 | 167 | `data/gold_standard/annotations/annotations.json` |
| Symptom region polygons (Pass 2) | *(same 501 images, Phase 3b)* | ≥ 400 MSV+MLN | — | `data/gold_standard/annotations/symptom_annotations.json` |

**Consumers of `annotations.json`:**
1. `train_yolo_detector.py` — reads polygons → derives bboxes (min/max of x,y) → trains YOLOv8n
2. `validate_gold_standard.py` — reads polygons → `cv2.fillPoly()` → binary IoU masks
3. `train_yolo_detector.py` (calibration) — YOLO+SAM2 on gold images → calibrate QA threshold

> `sample_yolo_annotations.py` is deprecated and unused — it is not part of the execution
> order. `YOLO_IMAGES_DIR`, `YOLO_ANNOTATIONS_DIR`, and `YOLO_ANNOTATION_FILE` in `config.py`
> all point to `data/gold_standard/` paths, so any file still present on disk is legacy
> cruft safe to delete.

---

## PART 3 — IMAGE UTILITIES (image_utils.py)

All image loading throughout the pipeline passes through this module. No other script calls `cv2.imread()` or `PIL.Image.open()` directly.

### 3.1 EXIF Orientation Correction
`cv2.imread()` ignores EXIF rotation. PIL corrects it via `ImageOps.exif_transpose()`. Without this, training DataLoaders and Factory HSV masking would see different spatial orientations for the same image.

`load_image_rgb(path)` — PIL open → `ImageOps.exif_transpose()` → `.load()` (forces full decode, catches truncation) → `np.array(img.convert("RGB"), dtype=np.uint8)`. Returns None on any failure.

### 3.2 CLAHE
Applied to Bouncer inputs and Factory inputs. Not applied to Teacher or Student training.

`apply_clahe(img_rgb)` — RGB → LAB → CLAHE on L channel (clipLimit=2.0, tileGridSize=(8,8)) → RGB. Preserves colour (a, b unchanged). Citation: Zuiderveld (1994).

`load_image_clahe(path)` — `load_image_rgb()` + `apply_clahe()`.

### 3.3 Channel Order Helpers
```
to_hsv(img_rgb)    → cv2.COLOR_RGB2HSV   (guaranteed correct direction)
to_gray(img_rgb)   → cv2.COLOR_RGB2GRAY
rgb_to_bgr(img)    → cv2.COLOR_RGB2BGR   (for cv2.imwrite() ONLY)
bgr_to_rgb(img)    → cv2.COLOR_BGR2RGB
```

---

## PART 4 — PREPROCESSING (partition_dataset.py)

Run once. Never re-run after training begins.

### 4.1 Ten-Step Validation Pipeline
```
Step 1 — Zero-byte / tiny file
  stat().st_size < 100 bytes → reject

Step 2 — Magic bytes
  JPEG: FF D8 FF · PNG: 89 PNG · mismatch → reject

Step 3 — Truncation
  PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False
  PIL.Image.open(path).load() — forces full decode
  Exception → reject

Step 4 — Minimum resolution
  min(w, h) < 64 px → reject

Step 5 — Maximum resolution
  max(w, h) > 4096 px → reject

Step 6 — Extreme aspect ratio
  max(w, h) / min(w, h) > 8.0 → reject

Step 7 — Colour mode
  mode == "1" → reject
  mode == "L" or "P" → flag, keep (converted to RGB)

Step 8 — Near-uniform
  np.array(img.convert("RGB")).std() < 5.0 → reject

Step 9 — MD5 exact duplicates
  Same hash, same class     → reject duplicate
  Same hash, different class → reject BOTH

Step 10 — pHash near-duplicates
  Within class: Hamming ≤ 2 → reject duplicate
  Across classes: Hamming ≤ 2 → reject BOTH

Additional: green content flag
  Green pixels (H∈[35,85], S>30, V>30) / total < 0.05 → flag, keep
```

### 4.2 Outputs
```
global_split_manifest.csv           columns: source_path, category, split
reports/preprocessing_report.csv    per-rejected-image log
reports/preprocessing_flagged.csv   flagged-but-kept images
reports/preprocessing_summary.txt   thesis-ready summary
quarantine/{reason}/{class}/        rejected images moved here (not deleted)
```

---

## PART 5 — BOUNCER (Phase 0)

### 5.1 Purpose
Binary maize-vs-not-maize gate at inference. Runs before any disease analysis.

### 5.2 Dataset Construction (create_bouncer_dataset.py)
```
Positives (maize)    : 25,000 from global train+val — test excluded
Negatives (not_maize): 25,000 from NON_MAIZE_SOURCES, validated through Steps 1–6
Total                : 50,000 balanced
Internal split       : 80% train / 20% val, seed=42, Bouncer-internal only
```

### 5.3 Inference Architecture (scripts/bouncer_inference.py)

Single source of truth shared by `train_bouncer.py` and `factory_master.py`.

**Stage 1 — Heuristic pre-filter:** Always returns True (passthrough). Original green-coverage heuristic removed — caused false rejections on yellow/bleached MSV leaves.

**Inference transform (Albumentations 2.0 API):**
```
A.LongestMaxSize(BOUNCER_IMG_SIZE)
A.PadIfNeeded(BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE, border_mode=BORDER_CONSTANT,
              fill=0, fill_mask=0)   ← 2.0 API: fill/fill_mask not value
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ToTensorV2()
```

**Stage 2 — Neural gate:**
```python
@torch.no_grad()
def neural_bouncer(img_rgb, model, threshold):
    tensor = BOUNCER_INFER_TF(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    return torch.sigmoid(model(tensor).squeeze()).item() >= threshold
```

### 5.4 Four Variants

| Variant | Architecture | Head | Deployed? |
|---|---|---|---|
| gabor_lbp | Gabor (4freq×4orient) + LBP r=3 P=24 + LinearSVC | — | No — traditional CV baseline |
| mobilenet_v2 | MobileNetV2 pretrained | Linear(1280→1) | No — neural comparison |
| **mobilenet_v3_large** | MobileNetV3-Large IMAGENET1K_V2 | Linear(960→1) | **Yes** |
| edgevit_xxs | EdgeViT-XXS | binary head | No — hybrid ViT candidate |

PatchCore: optional offline baseline only, no TFLite path.

### 5.5 Neural Training Configuration
```
Loss              : BCEWithLogitsLoss
Optimizer         : AdamW(lr=1e-4, weight_decay=1e-4)
Scheduler         : CosineAnnealingLR(T_max=15, eta_min=1e-6)
Epochs            : 15
Batch size        : 64
Early stopping    : patience=5, monitors val F1
Gradient clipping : clip_grad_norm_(max_norm=5.0)
safe_collate      : yes
Image loading     : load_image_clahe()
```

**Training augmentation:**
```
LongestMaxSize(224) + PadIfNeeded(224, 224)
HorizontalFlip(p=0.5)
RandomRotate90(p=0.3)
ColorJitter(brightness=0.2, contrast=0.2, p=0.5)
HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=10, p=0.3)
Normalize + ToTensorV2()
```

### 5.6 Threshold Selection
Threshold = argmax(√(recall × specificity)) subject to maize recall ≥ 0.95, hard cap at 0.70. Specificity-alone maximisation abandoned — produced near-1.0 thresholds and >88% rejection rates.

### 5.7 Gabor + LBP Baseline
```
Features : 4×4 Gabor filters (mean+std per filter = 32) + LBP 26-bin histogram = 58 total
Classifier: Pipeline(StandardScaler + LinearSVC(max_iter=2000))
Split     : stratified 60/40, random_state=42
```

### 5.8 Admission Rate Evaluation
Run deployed Bouncer on all test-split maize images. Log: n_total, n_passed, n_rejected, admission_rate, false_rejection_rate. Output: `logs/bouncer_admission_rate_{variant}.csv`.

---

## PART 6 — TIER 1 SAMPLING (sample_15000.py)

### 6.1 Composition
```
HEALTHY : 3,000  random.sample(), seed=42
MSV     : 7,500  evenly-spaced: step=len/n; indices=[int(i*step) for i in range(n)]
MLN     : 4,500  evenly-spaced (same formula, deterministic)
TOTAL   : 15,000
Source  : train+val only — test excluded
```

### 6.2 Hash Guard
SHA-256 of `global_split_manifest.csv` → `data/tier1_raw/_manifest_hash.txt` on first run. Aborts on manifest change rather than silently re-sampling. Delete `_manifest_hash.txt` (and masks + annotations) to restart from scratch.

### 6.3 Outputs
```
data/tier1_raw/{CLASS}_{original_filename}
tier1_manifest.csv   columns: dest_filename, source_path, category, split, tier
```

---

## PART 7 — GOLD STANDARD EXPORT (sample_gold_standard.py)

### 7.1 What It Does
Exports **501 images (167 per class)** from Tier 1. Files renamed `{CLASS}_{original}.jpg`. SHA-256 of `tier1_manifest.csv` locked to `data/gold_standard/images/_manifest_hash.txt` on first run — aborts if manifest changes post-annotation.

This is the **only annotation export script that matters in the pipeline.** An older script, `sample_yolo_annotations.py`, once exported a separate 501-image set for YOLO bbox annotation only — it is now deprecated and unused. `train_yolo_detector.py` derives its bounding boxes directly from the same `annotations.json` produced here, so one annotation task now serves all three consumers and `data/yolo_annotations/` is not used.

### 7.2 Single annotations.json — Three Consumers

```
data/gold_standard/annotations/annotations.json
         │
         ├── train_yolo_detector.py  (Phase 1c)
         │     Reads polygon points → derives tight bbox per image:
         │       x1 = min(xs), x2 = max(xs), y1 = min(ys), y2 = max(ys)
         │       Converts to YOLO format: cx cy w h (normalized)
         │       Trains YOLOv8n on these derived boxes
         │
         ├── validate_gold_standard.py  (Phases 2c / 2d / 10b)
         │     Reads polygon points → cv2.fillPoly() → binary mask
         │     Multiple polygons merged with logical OR
         │     Computes binary IoU vs SAM2 / Teacher / Student predictions
         │
         └── train_yolo_detector.py  (calibration step)
               Runs YOLO+SAM2 pipeline on gold images
               Finds min confidence threshold achieving mean IoU ≥ 0.85
               Writes calibrated value to logs/yolo_qa_calibration.csv
```

### 7.3 Label Studio Annotation Workflow

**Pass 1 — Leaf silhouette (before Phase 1c):**
- Task type: polygonlabels
- Trace the complete leaf outline as a polygon for all 501 images
- Export JSON → `data/gold_standard/annotations/annotations.json`

**Pass 2 — Symptom regions (Phase 3b, same 501 images, after Pass 1):**
- Return to same Label Studio project
- Trace disease regions only: chlorotic streaks (MSV), necrotic patches (MLN)
- Label name: `"symptom"`. HEALTHY images: skip.
- Target ≥ 400 MSV+MLN images (`SYMPTOM_MIN_ANNOTATIONS = 400`)
- Export JSON → `data/gold_standard/annotations/symptom_annotations.json`

### 7.4 config.py Paths
```python
# All YOLO paths now alias to gold_standard — no separate yolo_annotations/ directory
YOLO_IMAGES_DIR      = GOLD_IMAGES_DIR          # data/gold_standard/images/
YOLO_ANNOTATIONS_DIR = GOLD_ANNOTATIONS_DIR      # data/gold_standard/annotations/
YOLO_ANNOTATION_FILE = GOLD_ANNOTATION_FILE       # .../annotations/annotations.json
```

---

## PART 8 — YOLO LEAF DETECTOR (train_yolo_detector.py)

### 8.1 Purpose
Trains YOLOv8n (nano, single class = `"leaf"`) by deriving bounding boxes from the gold standard polygon annotations. Must complete before `generate_tier1_masks.py`.

Pure HSV centroid search drifts onto background on heavily diseased images — MSV yellow and MLN brown share hue ranges with tropical soil. A tight YOLO bounding box constrains SAM2's segmentation region, eliminating this dominant failure mode.

### 8.2 Polygon → Bbox Conversion
```python
# For each polygon in annotations.json:
xs = [pt[0] for pt in polygon_points]   # points stored as % of image dimensions
ys = [pt[1] for pt in polygon_points]
x1, x2 = min(xs), max(xs)
y1, y2 = min(ys), max(ys)
# Convert to YOLO: cx cy w h (normalized 0–1)
cx = (x1 + x2) / 2 / img_w
cy = (y1 + y2) / 2 / img_h
w  = (x2 - x1) / img_w
h  = (y2 - y1) / img_h
```

No separate Label Studio bounding-box annotation is needed. The boxes are derived automatically from the leaf-silhouette polygons.

### 8.3 Training Configuration
```
Model       : YOLOv8n (nano) — smallest ultralytics model
Classes     : 1 ("leaf")
Epochs      : 100
Batch       : 8
Image size  : 640
lr0         : 0.01
Patience    : 20
Conf thresh : 0.25 (inference NMS)
IoU NMS     : 0.45
Val split   : 15% of annotated images
Min annots  : 400 (YOLO_MIN_ANNOTATIONS) — warns if fewer
mAP target  : ≥ 0.70 at IoU=0.5
```

### 8.4 SAM2 QA Confidence Calibration
After training, the script runs the full YOLO+SAM2 pipeline on the 501 gold-standard images and sweeps confidence thresholds to find the minimum that achieves mean IoU ≥ `GOLD_IOU_TARGET_MEAN` (0.85) against the human polygon masks. Result written to `logs/yolo_qa_calibration.csv`. Loaded at runtime by `generate_tier1_masks.py`.

### 8.5 Outputs
```
checkpoints/yolo/best.pt            loaded by generate_tier1_masks.py
logs/yolo_qa_calibration.csv        calibrated threshold + IoU curve
logs/yolo_training_metrics.csv      per-epoch mAP + loss
data/yolo_dataset/                  derived YOLO-format dataset (train/val split)
```

---

## PART 9 — SAM2 MASKING (generate_tier1_masks.py) [v3]

### 9.1 Prompting Strategy
```
For each Tier 1 image:

1. YOLO path (when checkpoints/yolo/best.pt exists):
   a. Run YOLOv8n → bounding box (x1, y1, x2, y2)
   b. Restrict HSV tissue search to pixels inside box
   c. 3 foreground points along vertical leaf axis, clamped to box
   d. Pass box to SAM2 as box= prompt (hard spatial constraint)
   e. QA report: prompt_mode="yolo", yolo_box=[x1,y1,x2,y2]

2. HSV fallback (YOLO absent, failed, or low-confidence):
   a. Full-image HSV search:
      Green tissue:     H∈[35,85], S>40, V>40
      Yellow/diseased:  H∈[15,45], S>40, V>60
      Combined → largest connected component centroid
   b. 3 foreground points from centroid along vertical axis
   c. If no tissue found: image centre fallback (no rejection at prompt stage)
   d. QA report: prompt_mode="hsv_fallback"

3. Always: 4 image corners (10px inset) as background prompts (label=0)

4. SAM2.predict(point_coords, point_labels, box=yolo_box_or_None)
   → select highest-confidence mask
   → sigmoid(logits) → float32 [0,1] probability map
   → store .npy (raw) + .png (binarized at 0.5)
```

### 9.2 QA Filters (actual runtime values)
```
Min coverage             : 2%    (3% was rejecting valid YOLO-confirmed leaves)
Max coverage             : none  (full-frame close-ups are valid)
High-coverage gate       : coverage ≥ 97% requires mean confidence ≥ 0.80
Min mean confidence      : 0.65 default; overridden by yolo_qa_calibration.csv
Min aspect ratio         : 1.00 (disabled — overhead shots at 1.000–1.008 were valid)

Target rejection rate    : < 8%
```

### 9.3 Outputs
```
data/tier1_leaf_masks/{stem}_softmask.npy  float32 [0,1] probability map (primary)
data/tier1_leaf_masks/{stem}_mask.png      uint8 binary visualization
tier1_qa_report.csv   per-image: filename, category, status, reason,
                                 prompt_mode, yolo_box, coverage, mean_conf
```

---

## PART 10 — TEACHER MODEL (train_teacher.py)

### 10.1 Purpose
Offline segmentation model. Never deployed. Generates leaf silhouette pseudo-masks for ~215k Tier 2 images via Factory. Higher quality than Otsu; far faster than SAM2 at scale.

### 10.2 Loss Function — BoundaryAwareLoss
```python
sharpened = clamp(target * SHARPEN_FACTOR, 0, 1)   # SHARPEN_FACTOR = 1.3

# Boundary map: where probability changes rapidly
boundary = clamp(max_pool2d(target,3,1,1) - avg_pool2d(target,3,1,1), 0, 1)

L = DICE_WEIGHT * DiceLoss(logits, sharpened)
  + BCE_WEIGHT  * BCEWithLogitsLoss(logits, sharpened,
                    weight = 1 + BOUNDARY_WEIGHT * boundary)

# Constants: BOUNDARY_WEIGHT=5.0, SHARPEN_FACTOR=1.3, DICE_WEIGHT=1.0, BCE_WEIGHT=0.5
```

Sharpening pushes soft SAM2 targets toward harder boundaries. Boundary up-weighting (5.0×) forces the model to learn precise leaf edges over interior blob accuracy.

### 10.3 Four Variants

| Variant | Encoder | Decoder | Params | Notes |
|---|---|---|---|---|
| resnet50 | ResNet-50 | UNet | ~32M | Deep CNN baseline |
| **efficientnet-b2** | EfficientNet-B2 | UNet | ~7.7M | **Deployed** |
| mit_b2 | SegFormer-B2 | UNet | ~25M | Hierarchical ViT comparison |
| deeplabv3plus-eb2 | EfficientNet-B2 | DeepLabV3+ (ASPP) | ~7.7M | Decoder architecture comparison |

All via `segmentation_models_pytorch`. `mit_b2` uses `encoder_name="mit_b2"` via timm. `deeplabv3plus-eb2` isolates the decoder variable (same EfficientNet-B2 encoder, ASPP vs UNet decoder).

### 10.4 Configuration
```
Input size          : 768×768
Batch size          : 2 default; overrides: resnet50=6, efficientnet-b2=4,
                      mit_b2=2, deeplabv3plus-eb2=3
Gradient accum      : 2 steps
Epochs              : 30
LR                  : 5e-5
Weight decay        : 5e-4
Optimizer           : AdamW
Scheduler           : ReduceLROnPlateau(factor=0.5, patience=5, mode=max)
Early stopping      : patience=7, monitors val Dice
Checkpoint          : best val Dice → teacher_{variant}_best.pth
                      Best variant → teacher_model_best.pth (canonical for Factory)
Gradient clipping   : clip_grad_norm_(max_norm=5.0)
safe_collate        : yes
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True (768px under 8GB VRAM)
```

**Training augmentation:**
```
LongestMaxSize(768) + PadIfNeeded(768, 768)
HorizontalFlip(p=0.5)
VerticalFlip(p=0.5)
RandomRotate90(p=0.5)
RandomBrightnessContrast(limit=0.2, p=0.3)
HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2)
Normalize + ToTensorV2()
```

### 10.5 Manifest Filtering
Any Tier 1 image whose `source_path` appears in the test split is excluded from Teacher training. An internal 80/20 split is applied to remaining images for Teacher's own train/val cycle.

---

## PART 11 — SYMPTOM TEACHER (train_symptom_model.py) — Phase 3b

### 11.1 Why It Replaces LAB/HSV
`factory_master.py` went through eight versions (v1→v8) of LAB and HSV threshold tuning. The structural problem: static colour rules cannot separate early-stage chlorosis from healthy yellow-maize tissue regardless of how many morphological exceptions are added. A model trained on ~400–500 human-verified masks learns the decision boundary directly.

### 11.2 HealthyAE
```
Purpose      : Produce a lighting-invariant anomaly prior as the 4th input channel
Training     : HEALTHY images from global train+val only (no annotation needed)
Architecture :
  Encoder: Conv2d(3→32) → Conv2d(32→64) → Conv2d(64→128) → Conv2d(128→256)
           all stride=2, BN+ReLU, spatial dims: 256→128→64→32→16
  Bottleneck: Conv2d(256→128, 1×1) → Conv2d(128→256, 1×1)
  Decoder: ConvTranspose2d ×4 (reverse), BN+ReLU, Sigmoid output
  NO skip connections — forces compression; decoder cannot pass disease pixels
At inference:
  error_map = |input − reconstruction|²
  Normalized per-image to [0,1] → used as 4th channel for Symptom Teacher
  AE is an INPUT FEATURE, not a label source
Config: input 256×256 · batch 16 · 40 epochs · Adam(lr=1e-3) · val 10% · patience=8 · MSE
Checkpoint: checkpoints/healthy_ae/healthy_ae_best.pth
```

### 11.3 Symptom Teacher
```
Architecture: smp.Unet(encoder_name="efficientnet-b2", in_channels=4, classes=1)
Input       : [RGB (3ch)] + [HealthyAE error map (1ch)] = 4 channels
Ground truth: human polygon/brush masks from symptom_annotations.json
              ALWAYS human masks — model cannot inherit LAB biases
Output      : restricted to leaf silhouette in factory_master.py

Config:
  Input size  : 512×512
  Batch       : 4
  Epochs      : 60
  Optimizer   : AdamW(lr=1e-4, weight_decay=1e-4)
  Loss        : 0.5 × DiceLoss(from_logits=True) + 0.5 × BCEWithLogitsLoss
  Val split   : 15%
  Patience    : 10
  Target IoU  : 0.70
  Checkpoint  : checkpoints/symptom/symptom_teacher_best.pth
```

### 11.4 Factory Integration
```python
# config.py
SYMPTOM_TEACHER_DEPLOYED    = True   # use Symptom Teacher
FACTORY_SYMPTOM_COMPARE_LAB = False  # True → log IoU(LAB) vs IoU(SymptomTeacher)
```
If `SYMPTOM_TEACHER_DEPLOYED = True` and checkpoints exist: `predict_symptom_mask()` runs the Symptom Teacher. If checkpoints are missing: legacy LAB pipeline activates automatically. LAB functions remain in `factory_master.py` unchanged as permanent fallback.

---

## PART 12 — FACTORY (factory_master.py)

### 12.1 Purpose
Process ~215k train+val Tier 2 images, writing pseudo-labels into 4 mode subfolders simultaneously. All 4 modes coexist on disk.

### 12.2 Per-Image Pipeline
```
Stage 1 — Bouncer gate (scripts/bouncer_inference.py)
  heuristic_prefilter() → True (passthrough)
  neural_bouncer()       → True / False
  Rejected: log, skip

Stage 2 — Leaf silhouette
  Tier 1 images: load pre-existing SAM2 .npy
  Tier 2 images: Teacher inference at 768×768 → resize to original

Stage 3 — Silhouette refinement
  threshold at 0.35 (catches darker leaves than 0.5)
  morphological close then open (kernel=5, elliptical)

Stage 4 — Coverage → reliability weight
  < 15%  : weight = −1 (exclude; severity sentinel = −1)
  15–25% : weight = 0.30
  25–50% : weight = 0.70
  > 50%  : weight = 1.00

Stage 5 — Symptom masking
  Primary (SYMPTOM_TEACHER_DEPLOYED=True + checkpoints exist):
    AE error map → normalize [0,1] → Symptom Teacher on [RGB+error] → restrict to silhouette
  Fallback (checkpoints missing):
    Legacy LAB/HSV pipeline (see Part 12.3 below)

Stage 6 — Severity
  severity_pct = symptom_pixels / leaf_pixels × 100

Stage 7 — CIMMYT grading
  MSV  : 1=<5%, 3=5–25%, 5=25–50%, 7=50–75%, 9=>75%
  MLN  : 1=<10%, 2=10–25%, 3=25–50%, 4=50–75%, 5=>75%
  HEALTHY: 0
```

### 12.3 Four Factory Modes

| Mode | Silhouette | Symptom | Subfolder |
|---|---|---|---|
| A | Otsu | Hard binary | `mode_a/` |
| B | SAM2 binary | Hard binary | `mode_b/` |
| C | SAM2 soft float | Hard binary | `mode_c/` |
| D | SAM2 soft float | Soft confidence [0,1] | `mode_d/` |

### 12.4 Legacy HSV Ranges (fallback when Symptom Teacher checkpoints absent)

**Green exclusion (applied first):** H 38–85, S 80–255, V 60–230

**MSV — 4 bands:**
```
R1: H 15–38,  S 50–255, V 150–255   bright yellow streaks
R2: H 20–45,  S 15–70,  V 130–255   pale yellow / early-stage
R3: H 0–179,  S 0–35,   V 210–255   near-white / bleached  [area filter: ≥80px]
R4: H 38–55,  S 10–55,  V 140–255   pale yellow-green
```

**MLN — 5 bands:**
```
R1: H 18–40,  S 40–255, V 90–255    chlorotic yellow
R2: H 5–20,   S 40–255, V 50–220    orange-amber necrosis
R3: H 22–55,  S 15–85,  V 80–240    pale yellow-green mosaic
R4: H 8–35,   S 0–45,   V 160–255   tan / straw tissue
R5: H 0–18,   S 30–180, V 30–150    dark brown dead tissue
```

### 12.5 Output Files per Image per Mode
```
{stem}_silhouette.npy   float32 [0,1]
{stem}_symptom.png      uint8 binary (modes A/B/C)
{stem}_symptom.npy      float32 [0,1] confidence (mode D only)
{stem}_sev.txt          float, or −1 sentinel
{stem}_grade.txt        CIMMYT grade integer
{stem}_weight.txt       0.30 / 0.70 / 1.00 / −1

Output resolution: all downscaled to 224×224 at write time
  Soft maps: INTER_LINEAR · Binary PNGs: INTER_NEAREST
```

### 12.6 Factory QA (validate_factory.py) — Phase 4b

Run after `factory_master.py` to visually audit whether pseudo-masks look correct before
spending a full training run on them. Not a hard dependency of any later script, but the
cheapest place to catch a broken masking pattern.

```
python validate_factory.py                        # 10 images/class, mode_b
python validate_factory.py --n 20                  # 20 images/class
python validate_factory.py --mode mode_c           # validate a specific mode
python validate_factory.py --img path/to/img.jpg   # single image
python validate_factory.py --all-modes             # mode_a/b/c/d side by side
python validate_factory.py --borderline             # sample near-threshold severity cases
```

Output: a self-contained HTML report (`reports/validate_factory_{mode}.html`, or
`_all_modes.html`) pairing each raw image with its pseudo-mask overlay, plus per-image
symptom-coverage stats and an automatic pass/fail flag per class:

| Class | Expected mask behavior | Red flag |
|---|---|---|
| HEALTHY | Mostly empty (near-black); yellow leaf color should not trip symptom thresholds | > ~4% mask fill |
| MSV | Narrow, broken streaks parallel to veins, pale-green→yellow→white | Uniform full-leaf fill, or empty on visibly streaky leaves |
| MLN | Wider, diffuse yellowing + necrotic brown patches from leaf margins inward | Only margins highlighted, or indistinguishable from MSV output |

The docstring in `validate_factory.py` also doubles as a tuning guide: it maps each visual
failure mode to the exact `factory_master.py` / `config.py` parameter to adjust
(e.g. `LAB_MSV_B_MIN`, `GABOR_THRESHOLD`, `dark_necrosis` L*/a* bounds).

---

## PART 13 — STUDENT MODEL (train_student.py)

### 13.1 Architecture
```
Input: [B, 3, 224, 224] float32, ImageNet normalized

Shared Encoder
    │
    ├── UNet Decoder  [+CBAM at skip connections for V2, V6]
    │       └── Segmentation Head: [B, 2, 224, 224] logits
    │               Ch0 = leaf silhouette  → sigmoid → binary at 0.5
    │               Ch1 = symptom mask     → sigmoid → binary at 0.5
    │
    ├── GAP → Dropout(0.3) → Linear(→3)
    │   └── Classification Head: argmax → 0=HEALTHY, 1=MSV, 2=MLN
    │
    └── GAP → Dropout(0.3) → Linear(→1) → ReLU → clamp(0,1)
        └── Severity Head: × 100 at inference = severity %
```

**Why ReLU+clamp:** Sigmoid never reaches exactly 0.0 — HEALTHY leaves always show nonzero severity. ReLU outputs exactly 0.0; clamp(0,1) prevents negative edge cases.

### 13.2 Five Encoder Variants

| ID | Encoder | Params | TFLite | Role |
|---|---|---|---|---|
| V1 | mobilenet_v2 | 3.4M | ✓ | No-attention baseline |
| **V2** | **mobilenet_v2_cbam** | **~3.5M** | **✓** | **Expected winner** |
| V3 | mobilenet_v3_small | 2.5M | ✓ | Ultra-compact |
| V4 | efficientnet_b0 | 5.3M | ✓ | Compound-scaling baseline |
| V6 | efficientnet_b0_cbam | ~5.4M | ✓ | CBAM generalization test |

V5 (MobileViT-XXS) removed: `torch.einsum` self-attention causes TFLite subgraph errors.

### 13.3 CBAM Implementation
```
CBAMBlock(channels):
  ChannelAttention:
    gap = AdaptiveAvgPool2d(1)(x)
    gmp = AdaptiveMaxPool2d(1)(x)
    mlp: Linear(C→C//16) → ReLU → Linear(C//16→C)   [shared]
    x  = x × sigmoid(mlp(gap) + mlp(gmp)).unsqueeze(-1,-1)

  SpatialAttention (kernel=7):
    cat = torch.cat([mean(x,dim=1), max(x,dim=1)], dim=1)
    x   = x × sigmoid(Conv2d(2→1, 7×7, padding=3)(cat))

Insertion: CBAMUnetDecoder subclasses smp.decoders.unet.decoder.UnetDecoder
  forward() calls _apply_cbam_to_features() explicitly
  NOT monkey-patching → torch.save() / state_dict() work correctly

CBAM_SPATIAL_KERNEL = 7  (config.py)
Citation: Woo et al. (2018) ECCV
```

### 13.4 Loss Functions

**A. Segmentation:**
```python
l_seg = (DiceLoss(logits[:,0:1], tgt[:,0:1]) + DiceLoss(logits[:,1:2], tgt[:,1:2])) / 2
# smp.losses.DiceLoss(mode="binary", from_logits=True, smooth_factor=0.0)
# Accepts float32 targets — soft boundary uncertainty preserved
```

**B. Asymmetric label-smoothing classification:**
```
Prior matrix P (rows = true class):
  HEALTHY → [0.90, 0.08, 0.02]
  MSV     → [0.05, 0.90, 0.05]
  MLN     → [0.02, 0.05, 0.93]

L_cls = −Σ P[true_class] × log(softmax(logits_cls))
Citations: Szegedy et al. (2016); Cruz et al. (2024); Mushayi et al. (2025)
```

**C. Reliability-weighted severity MSE:**
```
Valid samples: weight > 0 (sentinel −1 excluded from batch entirely)
l_sev = mean(sample_weights × MSE(sev_pred, sev_target))
Weights: <15%=excluded, 15–25%=0.30, 25–50%=0.70, >50%=1.00
Citation: Jiang et al. (2018) ICML — MentorNet
```

**D. Homoscedastic uncertainty (multi-task balance):**
```
Learnable: s1 (seg), s2 (cls), s3 (sev), initialized to 0.0

L_total = exp(−s1)·L_seg + s1
        + exp(−s2)·L_cls + s2
        + exp(−s3)·L_sev + s3

s1/s2/s3 learned via backprop. Replaces manual 0.6/0.2/0.2 weights.
Citation: Kendall et al. (2018) NeurIPS
```

### 13.5 Two-Phase Training
```
Phase 1 — Frozen encoder
  encoder.requires_grad_(False)
  Optimizer : AdamW(decoder + heads + log_vars, lr=1e-3, wd=1e-4)
  Scheduler : CosineAnnealingLR(T_max=30, eta_min=1e-6)
  Epochs    : up to 30
  Patience  : 10

Transition:
  encoder.requires_grad_(True)
  Optimizer re-initialized: AdamW(all + log_vars, lr=1e-4, wd=1e-4)
  Scheduler re-initialized: CosineAnnealingLR(T_max=20, eta_min=1e-6)
  Patience reset to 0

Phase 2 — Full fine-tuning
  Epochs   : up to 20
  Patience : 5

Throughout: clip_grad_norm_(model + log_vars, max_norm=5.0)
            WeightedRandomSampler (per-class inverse-frequency weights)
```

### 13.6 Training Augmentation
```
LongestMaxSize(224) + PadIfNeeded(224, 224)
HorizontalFlip(p=0.5)
VerticalFlip(p=0.5)
RandomRotate90(p=0.5)
Rotate(limit=30, p=0.5)
RandomBrightnessContrast(limit=0.25, p=0.3)
HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=10, p=0.2)
RandomShadow(p=0.2)    ← tropical domain adaptation
Normalize + ToTensorV2()
```

### 13.7 Checkpoint Criterion
```
quality_composite = 0.50 × mIoU_silhouette
                  + 0.35 × MSV_F1
                  + 0.15 × (1 − sev_mae_pct / 100)
Computed on val set every epoch. Checkpoint saved on improvement.
```

### 13.8 Two-Stage Ablation
```
Stage 1: mode=mode_b (fixed), all 5 encoders
  → checkpoints/student/stage1/{enc}_mode_b_best.pth   [ISOLATED]
  → logs/student_comparison_stage1.csv

Stage 2: encoder=best from Stage 1 (fixed), all 4 modes
  Mode B: reuse Stage 1 checkpoint (same seed=42 → identical)
  Modes A, C, D: new training runs
  → checkpoints/student/{enc}_{mode}_best.pth
  → logs/student_comparison_stage2.csv

Total unique runs: 5 + 3 = 8
Stated limitation: not fully crossed factorial design (Chapters 3 and 5)
```

### 13.9 Mobile Composite (post-training, select_best_pipeline.py)
```
mobile_composite = 0.38 × msv_f1
                 + 0.22 × sil_mIoU
                 + 0.22 × clamp(150ms / cpu_lat_ms, 0, 1)
                 + 0.10 × (1 − sev_mae / 100)
                 + 0.08 × clamp(15MB  / tflite_size_mb, 0, 1)

TFLite-incompatible variants excluded from ranking.
size_score defaults to 0.80 if TFLite file absent.
```

### 13.10 Metrics (Test Split)
```
Segmentation (global TP/FP/FN accumulation — NOT per-batch averaging):
  Ch0 silhouette + Ch1 symptom: mIoU, Dice, Recall, Precision, Specificity each

Classification:
  Per class: Precision, Recall, F1
  Overall: Accuracy, Macro F1, Weighted F1, MCC

Severity:
  MAE%, RMSE%, R²  (vs pseudo-labels, not expert ratings)

Latency:
  220 CPU passes (20 warmup + 200 measured) → mean ms, std ms, FPS

Primary outputs:
  logs/student_test_metrics_{enc}_{mode}.csv   ← thesis tables source
  logs/student_confusion_{enc}_{mode}.csv
```

---

## PART 14 — GOLD STANDARD VALIDATION (validate_gold_standard.py)

### 14.1 Purpose
Validates pseudo-label quality against 501 human-annotated leaf silhouettes. Provides the thesis-defensible chain comparison:

```
SAM2 masks    Teacher predictions    Student predictions
     ↓                ↓                     ↓
IoU vs human     IoU vs human          IoU vs human
```

### 14.2 Annotation Parser
Label Studio JSON polygonlabels → points as % of image dims → `cv2.fillPoly()` → binary mask. Multiple polygons per image: logical OR. Binary IoU = intersection / union (returns 0.0 if both masks empty).

### 14.3 Thresholds
```
GOLD_IOU_WARN_THRESHOLD = 0.75   per-image flag
GOLD_IOU_TARGET_MEAN    = 0.85   overall validation target
```

### 14.4 Run Order
```
After Phase 2:  validate_gold_standard.py --sam2-only   (validates SAM2 foundation)
After Phase 3:  validate_gold_standard.py               (SAM2 + Teacher chain)
After Phase 5b: validate_gold_standard.py               (complete chain — final thesis table)
```

### 14.5 Outputs
```
reports/gold_standard_iou_report.csv     per-image IoU for all 3 artifacts
reports/gold_standard_iou_summary.csv    mean ± std per class + overall per artifact
reports/gold_standard_overlays/          5 PNGs per class per artifact
  {stem}_{artifact}_overlay.jpg          cyan=correct, green=missed, red=extra
```

---

## PART 15 — XAI (evaluate_xai.py)

### 15.1 Three Methods
Applied to the last convolutional block of the shared encoder (before decoder branches — shared features for both classification and segmentation).

| Method | Class | Type | Role |
|---|---|---|---|
| Grad-CAM | GradCAM | Gradient-based | Historical baseline |
| **Grad-CAM++** | GradCAMPlusPlus | Gradient-based | **Deployed** |
| Score-CAM | ScoreCAM | Gradient-free | Stability reference |

**Target layers by encoder:**
```
mobilenet_v2 / mobilenet_v2_cbam        : encoder.features[-1][0]
mobilenet_v3_small                      : encoder.features[-1][0]
efficientnet_b0 / efficientnet_b0_cbam  : encoder.blocks[-1][-1]
```

### 15.2 Two Distinct App Outputs
```
Symptom boundary  : UNet Ch1 → pixel-level → crisp green contour
Diagnostic attention: Grad-CAM++ → coarse ~7×7 → amber heatmap (upsampled to 224×224)

These are NOT the same thing. Thesis must describe them separately.
Grad-CAM++ does NOT provide pixel-level segmentation.
```

### 15.3 Quantitative Metrics
```
Pointing game accuracy : top 20% heatmap pixels inside seg mask
Insertion AUC         : n_steps=6, GPU
Deletion AUC          : n_steps=6, GPU
Samples per class     : XAI_N_SAMPLES_CLASS = 30
Output                : logs/xai_comparison.csv, reports/xai/{method}/
```

---

## PART 16 — BEST PIPELINE SELECTION (select_best_pipeline.py)

```
Bouncer : max specificity, maize recall ≥ 0.95, neural only
Teacher : max val Dice
Student : quality composite during training (checkpoints)
          mobile composite post-training (deployment selection)

Checkpoint promotion:
  checkpoints/final/bouncer_best.pth
  checkpoints/final/teacher_best.pth
  checkpoints/final/student_best.pth   (best by mobile composite)

Config auto-update (regex):
  STUDENT_BEST_VARIANT = "{winner}"
  STUDENT_FACTORY_MODE = "{winner_mode}"

Outputs:
  reports/best_pipeline_summary.csv / .txt
  reports/all_variants_ranked.csv
  reports/student_mobile_ranking.csv
```

---

## PART 17 — DEPLOYMENT (export_tflite.py + build_deployment_package.py)

### 17.1 Export Path
```
PyTorch → ONNX (opset 12) → TF SavedModel → TFLite FP16
tf.lite.Optimize.DEFAULT + target_spec = [tf.float16]
```

### 17.2 Student TFLite I/O
```
Input  : [1, 3, 224, 224] float32, NCHW, ImageNet normalized
Output 0 (seg): [1, 2, 224, 224] logits → sigmoid → binary at 0.5
Output 1 (cls): [1, 3] logits → softmax → argmax (0=HEALTHY, 1=MSV, 2=MLN)
Output 2 (sev): [1, 1] float32 ∈[0,1] → ×100 = severity %
```

### 17.3 Bouncer TFLite I/O
```
Input  : [1, 3, 224, 224] float32, NCHW, same normalization
Output : [1, 1] logit → sigmoid → ≥ threshold → PASS
```

### 17.4 Android Runtime
```
Min API     : 21
TFLite      : org.tensorflow:tensorflow-lite:2.13.0
Support     : org.tensorflow:tensorflow-lite-support:0.4.4
GPU delegate: org.tensorflow:tensorflow-lite-gpu:2.13.0 (optional)

Pipeline:
  EXIF correction → letterbox 224×224 → ImageNet normalize → NCHW
  → Bouncer → (pass) → Student → 3 outputs
  → display: green contour + amber heatmap + class badge + severity gauge
```

### 17.5 Deploy Bundle
```
exports/deploy/
  bouncer_model.tflite
  student_model.tflite
  model_metadata.json      shapes, normalization, thresholds, class names
  DEPLOYMENT_README.md     Kotlin/Java integration guide
  deployment_report.csv    sizes, latencies, validation status
```

---

## PART 18 — CHART GENERATOR (generate_charts.py)

Standalone — no project module imports. Reads `logs/*.csv` only. Safe mid-training; missing CSVs silently skipped. Dark theme (navy #0F172A, teal #34D399). Agg backend (headless WSL2). 150 DPI PNG.

```bash
python generate_charts.py              # all
python generate_charts.py --bouncer
python generate_charts.py --teacher
python generate_charts.py --student
```

**Colour palette (consistent across all charts):**
```
gabor_lbp / resnet50 / mode_a   #6B7280  grey
mobilenet_v2 / mode_b            #3B82F6  blue
mobilenet_v3_large / efficientnet-b2 / mode_c  #10B981  emerald
edgevit_xxs / mit_b2 / mode_d   #F59E0B  amber
deeplabv3plus-eb2                #EF4444  red
mobilenet_v2_cbam                #8B5CF6  violet
mobilenet_v3_small               #EC4899  pink
efficientnet_b0                  #F97316  orange
efficientnet_b0_cbam             #14B8A6  teal
```

**Charts produced:**
- Bouncer: comparison bar (F1/Spec/Recall/AUC, deployed marked ▲), training curves (3 panels), confusion matrix
- Teacher: comparison bar (Dice/latency, best marked ★), training curves (3 panels)
- Student: Stage 1 bar, Stage 2 bar, training curves (2×3 grid, Phase 1/2 boundary dashed), confusion matrix (raw + normalized), radar per variant, all-variants radar overlay, metrics heatmap (column-normalized, MAE/latency inverted)

---

## PART 19 — HTML REPORT (generate_report.py)

`reports/evaluation_report.html` — 100% self-contained (charts base64-embedded), dark theme, no external dependencies.

**11 sections:**
1. Pipeline overview + dataset statistics
2. Preprocessing — rejection counts + reasons breakdown
3. Bouncer — comparison bar, ROC table, confusion matrix, admission rate
4. Teacher — Dice/latency bar, training curves, test metrics, overlays
5. Student Stage 1 (encoder ablation) — comparison table + grouped bar
6. Student Stage 2 (mode ablation) — mode A/B/C/D comparison + severity histograms
7. Best Student full results — all metrics, per-class P/R/F1, confusion matrix, training curves
8. XAI — pointing game/insertion/deletion bar, overlays, two-output distinction note
9. Severity reliability — Kappa + Spearman ρ with quality badges
10. Deployment — TFLite sizes, latency cards, bundle contents
11. Training curves — Phase 1/2 boundary annotated, all key metrics per epoch

---

## PART 20 — INFRASTRUCTURE

### 20.1 safe_collate (scripts/safe_collate.py)
Every DataLoader in every training script uses `collate_fn=safe_collate`. Filters None `__getitem__` returns. Returns None for all-None batches; training loop skips with `continue`. API: `reset_skip_counter()` at epoch start, `get_skip_count()` at epoch end.

### 20.2 Gradient Clipping
All training scripts: `clip_grad_norm_(model.parameters(), max_norm=5.0)`. Student also clips uncertainty log_vars (s1/s2/s3). Protects against log_var spikes and encoder-unfreeze surges.

### 20.3 Reproducibility
Every script calls `set_seeds(42)`: `random`, `numpy`, `torch`, `cuda`, `cudnn.deterministic=True`, `cudnn.benchmark=False`. Stage 1 Mode B reused in Stage 2 because same seed + same manifest + same hyperparameters = identical run.

### 20.4 Timing
Wall-clock duration logged in every script. Student latency: 220 CPU passes (20 warmup + 200 measured) → mean ms, std ms, FPS. Teacher: 20 CPU passes per variant.

---

## PART 21 — EXECUTION REFERENCE

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2

python partition_dataset.py                          # Step 1 — run once only

python create_bouncer_dataset.py                     # Step 2
python train_bouncer.py                              # Step 3
python generate_charts.py --bouncer                  # optional

python sample_15000.py                               # Step 4

python sample_gold_standard.py                       # Step 4a — 501 images, 167/class
# Label Studio: polygonlabels, trace full leaf outline for all 501 images
# Export JSON → data/gold_standard/annotations/annotations.json
# This one file is used by train_yolo_detector.py AND validate_gold_standard.py

python train_yolo_detector.py                        # Step 4b
# Reads annotations.json → derives bboxes → trains YOLOv8n → calibrates SAM2 QA threshold

python generate_tier1_masks.py                       # Step 5
python validate_masks.py                             # Step 6
python validate_gold_standard.py --sam2-only         # Step 6c

python train_teacher.py                              # Step 7
python generate_charts.py --teacher                  # optional
python validate_gold_standard.py                     # Step 6d

# Step 7b — second annotation pass on same 501 images (symptom regions):
# Label Studio: trace chlorotic streaks (MSV) / necrotic patches (MLN), label="symptom"
# HEALTHY: skip. Target ≥ 400 MSV+MLN images.
# Export JSON → data/gold_standard/annotations/symptom_annotations.json
python train_symptom_model.py                        # Step 7b

python factory_master.py                             # Step 8

python validate_factory.py --all-modes               # Step 8b — optional visual QA
# → reports/validate_factory_all_modes.html — check before spending a full Student run
# on pseudo-masks; tuning guide for LAB_*/GABOR_THRESHOLD is in the script docstring

python train_student.py --stage 1                    # Step 9
# check logs/student_comparison_stage1.csv for best encoder
python train_student.py --stage 2 --encoder mobilenet_v2_cbam   # Step 10
python generate_charts.py --student                  # optional
python validate_gold_standard.py                     # Step 10b — full chain

python select_best_pipeline.py                       # Step 11

python evaluate_xai.py                               # Step 12
python evaluate_severity.py --sample                 # Step 13a
python evaluate_severity.py --analyze                # Step 13b
python export_tflite.py                              # Step 14
python build_deployment_package.py                   # Step 15
python generate_report.py                            # Step 16
```

---

## PART 22 — KEY TERMINOLOGY

| Wrong | Correct |
|---|---|
| "Knowledge distillation" | "Pseudo-label semi-supervised learning" or "teacher-guided pseudo-label generation" |
| "Pixel-level segmentation via Grad-CAM" | Grad-CAM++ = coarse ~7×7 class-discriminative heatmap upsampled to 224×224. UNet head = pixel-level segmentation. Entirely distinct outputs. |
| "Severity MAE measures disease severity accuracy" | "Severity MAE measures Student consistency with Factory's pseudo-labels (Symptom Teacher or HSV-derived), not expert agronomic ratings" |

---

## PART 23 — KEY CITATIONS

| Reference | Used for |
|---|---|
| Kendall et al. (2018) NeurIPS | Homoscedastic uncertainty multi-task loss |
| Szegedy et al. (2016) CVPR | Label smoothing |
| Woo et al. (2018) ECCV | CBAM |
| Ke et al. (2020) | Soft pseudo-label targets |
| Jiang et al. (2018) ICML | MentorNet — reliability-weighted curriculum |
| Cruz et al. (2024) | First confirmed MSV in the Philippines |
| Mushayi et al. (2025) | MSV / HEALTHY confusion — asymmetric prior basis |
| Selvaraju et al. (2017) ICCV | Grad-CAM |
| Chattopadhay et al. (2018) WACV | Grad-CAM++ |
| Wang et al. (2020) CVPR | Score-CAM |
| Mduma (2023) Mendeley | Tanzania Maize Imagery Dataset |
| Tan & Le (2019) ICML | EfficientNet |
| Xie et al. (2021) NeurIPS | SegFormer (mit_b2) |
| Pan et al. (2022) ECCV | EdgeViT |
| Fawcett (2006) | ROC threshold selection |
| Zuiderveld (1994) | CLAHE |
