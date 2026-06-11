# Yellow MAIze — Detailed Technical Blueprint
## Complete reference for code, pipeline, models, data, and architecture

---

## PART 1 — PROJECT CONTEXT

### 1.1 Thesis Identity
- **Title:** MAIze: A U-Net with MobileNetV2 and Explainable AI Framework for Maize Streak Virus Identification and Symptom Segmentation in Zea mays L.
- **Course:** BSCS 3-A, Angeles University Foundation, College of Computer Studies
- **Team:** Altes, Zylah Klein · Davis, Dominic · Tayer, Catherine P. · Ursua, Walter Vince
- **Deployment target:** Android mobile application (TFLite)
- **Primary clinical task:** Detect MSV (Maize Streak Virus) and MLN (Maize Lethal Necrosis) in yellow corn leaves with explainable, quantified diagnosis

### 1.2 Hardware
| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 — 8 GB VRAM |
| CPU | AMD Ryzen 5 3600X |
| RAM | 16 GB DDR4 |
| OS | Windows 11 + WSL2 Ubuntu (all Python runs in WSL2) |
| DataLoader workers | 4 (WSL2 /dev/shm limit) |

### 1.3 Software Stack
| Library | Version | Role |
|---|---|---|
| PyTorch + torchvision | ≥ 2.1.0 | Primary training framework |
| segmentation-models-pytorch | ≥ 0.3.3 | UNet encoder-decoder models |
| albumentations | ≥ 1.3.1 | Augmentation pipeline |
| timm | ≥ 0.9.12 | Pretrained encoder variants |
| opencv-python | ≥ 4.8.0 | HSV masking, morphological ops |
| Pillow | ≥ 10.0.0 | Image loading with EXIF handling |
| grad-cam | ≥ 1.4.8 | Grad-CAM, Grad-CAM++, Score-CAM |
| imagehash | ≥ 4.3.1 | pHash deduplication |
| scikit-learn | ≥ 1.3.0 | Metrics, SVM for Gabor baseline |
| scipy | ≥ 1.11.0 | Spearman correlation |
| ultralytics | ≥ 8.0.0 | YOLOv8n leaf detector |
| anomalib | ≥ 1.0.0 | PatchCore baseline (optional) |
| onnx + onnxruntime | ≥ 1.14.0 | ONNX export path |
| tensorflow | ≥ 2.13.0 | TFLite conversion only |
| matplotlib | ≥ 3.7.0 | Chart generation |
| SAM2 | from GitHub | Tier 1 leaf silhouette masking |
| pandas, numpy, tqdm | latest | Data handling, progress |

### 1.4 Global Settings
```python
SEED                = 42      # All scripts: random, numpy, torch, cuda
CUDNN_DETERMINISTIC = True    # Reproducibility over speed
CUDNN_BENCHMARK     = False   # benchmark=True gives ~15% speed but is non-deterministic
NUM_WORKERS         = 4       # WSL2 /dev/shm constraint
```

---

## PART 2 — DATASET

### 2.1 Maize Dataset (Primary)
| Source | Content | Location |
|---|---|---|
| Maize Imagery Dataset — Tanzania (Mduma 2023, Mendeley) | HEALTHY, MSV, MLN labelled leaf images | `maize_dataset/HEALTHY/`, `MSV/`, `MLN/` |
| Zenodo healthy + MSV samples | Additional maize images | Merge into above folders |

**Class distribution after preprocessing (approximate):**
| Class | Images | % |
|---|---|---|
| HEALTHY | ~96,000 | 38% |
| MLN | ~96,000 | 38% |
| MSV | ~60,000 | 24% |
| **TOTAL** | **~252,000** | 100% |

MSV is the primary clinical target but the smallest class.

### 2.2 Non-Maize Dataset (Bouncer Negative Class)
| Source | Content | Location | Citation |
|---|---|---|---|
| Intel Image Classification | Buildings, forest, glacier, mountain, sea, street | `dataset/raw_kaggle/intel/` | Kaggle: puneet6060 |
| Natural Images | Airplane, car, cat, dog, flower, fruit, motorbike, person | `dataset/raw_kaggle/natural/` | Kaggle: prasunroy |
| PlantVillage | Rice and sorghum leaves | `dataset/crop_neighbors/rice/`, `sorghum/` | Hughes & Salathé 2015 |
| iNaturalist Philippines | Cogon grass (Imperata cylindrica) | `dataset/crop_neighbors/cogon_grass/` | iNaturalist API |
| iNaturalist Philippines | Sugarcane (Saccharum officinarum) | `dataset/crop_neighbors/sugarcane/` | iNaturalist API |
| iNaturalist Philippines | Banana leaf (Musa spp.) | `dataset/crop_neighbors/banana_leaf/` | iNaturalist API |
| Mendeley Maize-Weed | Field weeds photographed in maize | `dataset/crop_neighbors/` | Espejo-Garcia et al. 2020 |

Bouncer target: 25,000 maize + 25,000 non-maize = 50,000 balanced.

### 2.3 Dataset Split
```
Strategy: Single global 70/15/15 stratified split via global_split_manifest.csv
Stratification: per-class (preserves HEALTHY:MLN:MSV ratio in each split)

Split    Ratio   Approx count   Purpose
train    70%     ~176,400       Weight updates, augmentation, backpropagation
val      15%     ~37,800        Early stopping, LR scheduling, checkpoint selection
test     15%     ~37,800        Final unbiased evaluation ONLY (accessed once)

RULE: Test split images NEVER appear in:
  - Tier 1 sampling (sample_15000.py filters by manifest)
  - Bouncer positive class (create_bouncer_dataset.py filters by manifest)
  - Teacher training (train_teacher.py filters by manifest)
  - Factory processing (factory_master.py skips test-split images)
```

### 2.4 External Model Weights to Download
| Model | File | URL | Location |
|---|---|---|---|
| SAM2 | sam2_hiera_large.pt | https://dl.fbaipublicfiles.com/segment_anything_v2/sam2_hiera_large.pt | `sam2/sam2_hiera_large.pt` |

---

## PART 3 — FILE STRUCTURE

See `PROJECT_STRUCTURE.md` for the full annotated directory tree. Key auto-generated files:

```
global_split_manifest.csv         ← partition_dataset.py — single split source of truth
tier1_manifest.csv                 ← sample_15000.py
tier1_qa_report.csv                ← generate_tier1_masks.py
```

Key config keys:
```python
# Paths
DATA_DIR              = Path("data")
TIER1_RAW_DIR         = DATA_DIR / "tier1_raw"
TIER1_MASKS_DIR       = DATA_DIR / "tier1_leaf_masks"
BOUNCER_DATASET_DIR   = DATA_DIR / "bouncer_dataset"
PSEUDO_MASKS_DIR      = DATA_DIR / "pseudo_masks"
GOLD_IMAGES_DIR       = DATA_DIR / "gold_standard" / "images"
GOLD_MANIFEST         = DATA_DIR / "gold_standard" / "gold_manifest.csv"
GOLD_ANNOTATION_FILE  = DATA_DIR / "gold_standard" / "annotations" / "annotations.json"
YOLO_IMAGES_DIR       = DATA_DIR / "yolo_annotations" / "images"
YOLO_MANIFEST         = DATA_DIR / "yolo_annotations" / "yolo_manifest.csv"
CHECKPOINTS_DIR       = Path("checkpoints")
LOGS_DIR              = Path("logs")
REPORTS_DIR           = Path("reports")

# Model sizes
BOUNCER_IMG_SIZE      = 224
TEACHER_IMG_SIZE      = 512
STUDENT_IMG_SIZE      = 224

# Validation thresholds
GOLD_IOU_WARN_THRESHOLD  = 0.75
GOLD_IOU_TARGET_MEAN     = 0.85
YOLO_MIN_ANNOTATIONS     = 400

# Student selection
TEACHER_DEPLOYED_VARIANT  = "efficientnet-b2"
STUDENT_BEST_VARIANT      = "mobilenet_v2_cbam"   # auto-updated by select_best_pipeline.py
STUDENT_FACTORY_MODE      = "mode_b"               # auto-updated by select_best_pipeline.py

# Tier 1 composition
TIER1_PER_CLASS = {"HEALTHY": 3000, "MSV": 7500, "MLN": 4500}

# Bouncer dataset
BOUNCER_TARGET_PER_CLASS = 25000
```

---

## PART 4 — IMAGE UTILITIES (image_utils.py)

All image loading throughout the pipeline passes through this module. Never call `cv2.imread()` or `PIL.Image.open()` directly in other scripts.

### 4.1 EXIF Orientation Correction
```
Problem: cv2.imread() ignores EXIF rotation flags.
         PIL.Image.open() respects them automatically via exif_transpose().
         Mismatch causes spatial inconsistency between training (PIL-based
         DataLoaders) and Factory (cv2-based HSV masking).

Solution: load_image_rgb(path):
          PIL.Image.open() → .load() (forces full decode) → ImageOps.exif_transpose()
          → np.array(pil_img.convert("RGB"), dtype=np.uint8) → uint8 RGB or None

          Returns None on truncation, corruption, or any OSError.
          Every cv2 operation downstream receives this corrected array.
```

### 4.2 CLAHE (Contrast Limited Adaptive Histogram Equalization)
```
Applied to: Bouncer inputs, Factory inputs
NOT applied to: Student/Teacher training — augmentation handles contrast variation

Parameters:
  clipLimit    = 2.0       contrast enhancement ceiling (prevents noise amplification)
  tileGridSize = (8, 8)    local region size for histogram equalization

Method: RGB → LAB colour space → CLAHE on L channel only → back to RGB
        Preserves colour (a, b channels unchanged), improves luminance contrast only

Citation: Zuiderveld (1994), "Contrast Limited Adaptive Histogram Equalization"
```

### 4.3 Channel Order Safety
```
Rule: All arrays passed between functions are uint8 RGB.
      cv2-specific operations convert internally via helper functions.

Functions:
  load_image_rgb(path)      → EXIF-corrected, truncation-guarded uint8 RGB or None
  load_image_clahe(path)    → load_image_rgb() + apply_clahe(), uint8 RGB or None
  apply_clahe(img_rgb)      → CLAHE on L channel, returns uint8 RGB
  to_hsv(img_rgb)           → cv2.COLOR_RGB2HSV  (guaranteed correct direction)
  to_gray(img_rgb)          → cv2.COLOR_RGB2GRAY
  rgb_to_bgr(img_rgb)       → cv2.COLOR_RGB2BGR  (for cv2.imwrite() calls ONLY)
  bgr_to_rgb(img_bgr)       → cv2.COLOR_BGR2RGB  (if cv2.imread() used externally)
```

---

## PART 5 — PREPROCESSING PIPELINE (partition_dataset.py)

**Run once, before any other script. Never re-run after training begins.**

### 5.1 Ten-Step Validation Pipeline
Each image passes through all 10 steps. Any failure → reject + log reason.

```
Step 1: Zero-byte / tiny file
  stat().st_size < 100 bytes → reject("zero_or_tiny:{N}b")

Step 2: Magic bytes / format mismatch
  Read first 8 bytes, compare to known headers:
    JPEG: FF D8 FF
    PNG:  89 PNG
  Mismatch → reject("magic_mismatch:{hex}")

Step 3: Truncated image detection
  PIL.Image.open() + .load() (forces full decode — not just header)
  PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False
  Exception on .load() → reject("truncated:{error}")

Step 4: Resolution checks
  min(w,h) < 64px              → reject("too_small:{w}x{h}")
  max(w,h) > 4096px            → reject("too_large:{w}x{h}")
  max(w,h) / min(w,h) > 8.0   → reject("extreme_aspect:{ratio}:{w}x{h}")

Step 5: Colour mode check
  mode == "1" (1-bit binary)   → reject("binary_1bit_image")
  mode == "L" (grayscale)      → flag("grayscale_converted_to_rgb") — KEEP
  mode == "P" (palette)        → flag("palette_mode") — KEEP

Step 6: Near-uniform / solid colour
  Convert to RGB numpy, compute array.std()
  std < 5.0 → reject("near_uniform:std={N}")

Step 7: MD5 exact duplicate removal (cross-class simultaneous)
  MD5 hash of full file bytes
  Same hash, same class     → reject duplicate("exact_duplicate_of:{filename}")
  Same hash, different class → reject BOTH ("cross_class_exact_duplicate:also_in_{class}")

Step 8: pHash near-duplicate removal (within class)
  64-bit perceptual hash, hash_size=8
  Hamming distance ≤ 2 to any already-kept image → reject("phash_near_duplicate")

Step 9: Cross-class pHash duplicate detection
  After within-class dedup, compare hashes across all classes
  Hamming ≤ 2 between different classes → reject both
  Reason: "cross_class_near_duplicate:similar_to_{class}:{filename}"
  Prevents same image with contradictory labels

Step 10: Low green content flag
  Green pixels (H∈[35,85], S>30, V>30) / total pixels < 0.05
  → flag("low_green_content:{pct}") — KEEP (field photos may have less)
```

### 5.2 Output Files
```
global_split_manifest.csv   → columns: source_path, category, split (train/val/test)
reports/preprocessing_report.csv   → per-image: path, status, reason, split
reports/preprocessing_flagged.csv  → flagged-but-kept images only
reports/preprocessing_summary.txt  → thesis-ready summary with counts
```

---

## PART 6 — BOUNCER (train_bouncer.py)

### 6.1 Purpose
Binary gate. Runs first on every camera frame. Rejects non-maize images before disease analysis.

### 6.2 Dataset Construction (create_bouncer_dataset.py)
```
Positive (maize):     25,000 images sampled from global manifest train+val split only
                      Test-split maize images EXCLUDED
Negative (not_maize): 25,000 images from NON_MAIZE_SOURCES
                      Validated through Steps 1–6 of preprocessing before sampling

Total: 50,000 balanced binary dataset
Internal split: 80% train / 20% val (Bouncer-internal, not from global manifest)
```

### 6.3 Stage 1 — Heuristic Pre-filter

**Currently a passthrough (always returns True).** The original OpenCV green-coverage heuristic was removed after causing false rejections on yellow/bleached MSV leaves under variable tropical lighting. The neural classifier is sufficient and fast enough at 224×224.

The passthrough is implemented in `scripts/bouncer_inference.py` as `heuristic_prefilter(img_rgb) → bool`. This is the shared single source of truth used by both `train_bouncer.py` and `factory_master.py`.

### 6.4 Stage 2 — Neural Variants

**4 variants compared:**

| Variant | Architecture | Type | Deployed? |
|---|---|---|---|
| gabor_lbp | Gabor filters + LBP + LinearSVC | Traditional CV | No (baseline) |
| mobilenet_v2 | MobileNetV2, head: Linear(1280→1) | Neural CNN | No (comparison) |
| **mobilenet_v3_large** | MobileNetV3-Large, head: Linear(960→1) | Neural CNN | **Yes** |
| edgevit_xxs | EdgeViT-XXS binary classifier | Hybrid ViT | Candidate |

> PatchCore (ResNet18 nearest-neighbour anomaly detector) is available as an optional offline evaluation via `evaluate_patchcore()` in `train_bouncer.py`, but is excluded from `BOUNCER_VARIANTS` — anomaly detection is architecturally mismatched for supervised binary classification and has no TFLite deployment path.

**MobileNetV3-Large hyperparameters:**
```
Pretrained weights : IMAGENET1K_V2
Head               : Linear(960 → 1)  [replaces ImageNet classifier]
Loss               : BCEWithLogitsLoss
Optimizer          : AdamW(lr=1e-4, weight_decay=1e-4)
Scheduler          : CosineAnnealingLR(T_max=15, eta_min=1e-6)
Epochs             : 15
Batch size         : 64
Val split          : 80/20 internal (seed=42)
Early stop         : patience=5, monitors val F1
Checkpoint         : best val F1 → bouncer_{variant}_best.pth
Gradient clipping  : clip_grad_norm_(max_norm=5.0)
safe_collate       : yes
```

**Image loading:** `load_image_clahe()` — EXIF correction + CLAHE + truncation guard.

**Training augmentation (Albumentations):**
```
LongestMaxSize(224) + PadIfNeeded(224, 224, border_mode=0)
HorizontalFlip(p=0.5)
RandomRotate90(p=0.3)
ColorJitter(brightness=0.2, contrast=0.2, p=0.5)
HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=10, p=0.3)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ToTensorV2()
```

### 6.5 Threshold Selection
```
After training, load best checkpoint.
Run on val split, collect sigmoid scores + true labels.
Plot ROC curve (fpr, tpr, thresholds).
Select threshold = argmax(geometric_mean(recall × specificity) ^ 0.5)
  subject to maize recall ≥ 0.95.
Hard cap: threshold = min(selected_threshold, 0.70).

Rationale: Previously maximised specificity alone, which caused near-1.0 thresholds
on high-performing models (ROC AUC ~1.0), resulting in >88% filter rates at inference.
Geometric mean balances both metrics and produces a usable production threshold.
Hard cap at 0.70 as additional safeguard.

Report: threshold value, specificity, maize recall, ROC-AUC, TP/FP/TN/FN
```

### 6.6 Gabor + LBP Baseline
```
Features:
  Gabor: 4 frequencies × 4 orientations = 16 filters
         mean + std per filter = 32 Gabor features
  LBP:   radius=3, P=24 points, uniform → 26-bin histogram
  Total: 58 features per image

Classifier: Pipeline(StandardScaler + LinearSVC(max_iter=2000))
Train/test:  stratified train_test_split(test_size=0.40, random_state=42)
Metrics:     Accuracy, Precision, Recall, F1, Specificity
```

### 6.7 PatchCore Baseline
```
Backbone: ResNet18 (ImageNet pretrained), remove final 2 layers
Feature extraction: forward pass → mean(dim=[2,3]) per image
Training set: train-split maize images (up to 2000 for speed)
Anomaly score: mean nearest-neighbour distance to training feature set (k=3)
Threshold: optimal from ROC (argmax tpr−fpr)
Metrics: AUROC, FPR@95%TPR, specificity, maize recall
```

### 6.8 Admission Rate Evaluation
```
Run deployed Bouncer on ALL test-split maize images from global manifest.
Both stages: heuristic_prefilter() → neural_bouncer().
Report:
  n_test_maize         : total tested
  n_passed             : passed both stages
  n_rejected           : rejected by either stage
  n_heuristic_reject   : rejected at Stage 1 (currently always 0 — passthrough)
  admission_rate       : n_passed / n_total
  false_rejection_rate : 1 − admission_rate
Output: logs/bouncer_admission_rate_{variant}.csv
```

### 6.9 Shared Bouncer Inference (scripts/bouncer_inference.py)
```
Single source of truth used by BOTH factory_master.py and train_bouncer.py.
Any future changes to threshold logic or transform pipeline are made here only.

BOUNCER_INFER_TF:
  A.LongestMaxSize(BOUNCER_IMG_SIZE)
  A.PadIfNeeded(BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE, border_mode=0, value=0)
  A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  ToTensorV2()

heuristic_prefilter(img_rgb: np.ndarray) → bool
  Currently a passthrough (always True).

neural_bouncer(img_rgb: np.ndarray, model: nn.Module, threshold: float) → bool
  Applies BOUNCER_INFER_TF, runs model, returns sigmoid(logit) >= threshold.
  Decorated @torch.no_grad().

Dependencies: torch, albumentations, config.BOUNCER_IMG_SIZE only.
No SAM2, Teacher, or other heavy imports — lightweight for import anywhere.
```

### 6.10 Metrics Reported
```
Per-epoch log: loss, accuracy, maize_prec, maize_rec, maize_f1, specificity, lr
Comparison CSV (logs/bouncer_comparison.csv):
  variant, best_f1, threshold, specificity, maize_recall, roc_auc,
  fpr95 (patchcore), TP, FP, TN, FN
```

---

## PART 7 — TIER 1 SAMPLING (sample_15000.py)

### 7.1 Composition
```
HEALTHY : 3,000  → pure random sample (seeded — no severity gradient needed)
MSV     : 7,500  → evenly-spaced indices across sorted filenames (diversity proxy)
MLN     : 4,500  → evenly-spaced indices
TOTAL   : 15,000

Source: train+val split images only. Test-split images NEVER included.

Evenly-spaced formula:
  step    = len(all_images) / n_select
  indices = [int(i * step) for i in range(n_select)]
  → deterministic math, no randomness
```

### 7.2 Hash Guard
```
On first run: SHA-256 of global_split_manifest.csv → data/tier1_raw/_manifest_hash.txt
On subsequent runs: re-check hash. If changed (partition_dataset.py was re-run),
  abort with a clear error instead of silently producing a different Tier 1 set.
  A different Tier 1 set invalidates SAM2 masks AND gold standard annotations.

To start fresh: delete _manifest_hash.txt and re-run.
```

### 7.3 Output
```
data/tier1_raw/{CLASS}_{original_filename}   ← images copied with class prefix
tier1_manifest.csv  → columns: dest_filename, source_path, category, split, tier
```

---

## PART 7b — YOLO ANNOTATION SAMPLER (sample_yolo_annotations.py)

### Purpose
Exports **501 images** (167 HEALTHY + 167 MSV + 167 MLN) from Tier 1 for YOLO bounding-box annotation in Label Studio. Must run after `sample_15000.py` and before `train_yolo_detector.py`.

### Why Separate from Gold Standard
`sample_gold_standard.py` yields 501 polygon annotations (silhouette masks). YOLO training needs bounding-box labels. Both samplers use the same 501-per-set size and seed, but their image sets may overlap — both use `pd.DataFrame.sample(random_state=SEED)` on the same manifest, so the overlap is deterministic.

### Annotation Instructions
- Label Studio task type: Object Detection (bounding box)
- Label: `"leaf"` (one label only)
- Draw ONE tight bounding box around the PRIMARY leaf only
- Ignore background leaves, stems, and hands
- Export format: YOLO (txt) → `data/yolo_annotations/labels/`

### Hash Guard
SHA-256 of `tier1_manifest.csv` written to `data/yolo_annotations/images/_manifest_hash.txt` on first run. If `sample_15000.py` is re-run after annotation begins, subsequent runs abort.

### Output
```
data/yolo_annotations/images/   ← 501 renamed images ({CLASS}_{original}.jpg)
data/yolo_annotations/images/_manifest_hash.txt
data/yolo_annotations/yolo_manifest.csv
```

---

## PART 7c — YOLO LEAF DETECTOR (train_yolo_detector.py)

### Purpose
Trains YOLOv8n (nano) single-class detector on Label Studio polygon annotations. Calibrates the SAM2 QA confidence threshold against gold-standard IoU. Must run **after** annotating at least 400 images and **before** `generate_tier1_masks.py`.

### Why YOLO Before SAM2
Pure HSV prompting places the centroid correctly for healthy green leaves but drifts onto background for heavily diseased images — MSV streak yellow and MLN necrotic brown share hue ranges with tropical soil and sand. A tight YOLO bounding box constrains both the centroid search and SAM2 segmentation region, eliminating the dominant v2 failure mode.

### mAP Target
mAP@0.5 ≥ 0.70 before using YOLO box prompts. Script warns and requests confirmation if annotation count < `YOLO_MIN_ANNOTATIONS` (400).

### Polygon → Bbox Conversion
Label Studio polygon annotations (from gold standard JSON) are parsed automatically. Bounding boxes are derived as the tight enclosing rectangle of each polygon. No separate bbox annotation step is required if polygon annotations are already available.

### QA Threshold Calibration
After training, the script runs the full YOLO+SAM2 pipeline on gold-standard images and finds the minimum mean-foreground-confidence threshold that achieves mean IoU ≥ `GOLD_IOU_TARGET_MEAN` (0.85). Written to `logs/yolo_qa_calibration.csv`. Used by `generate_tier1_masks.py` at runtime.

### Outputs
```
checkpoints/yolo/best.pt          ← Weights loaded by generate_tier1_masks.py v3
logs/yolo_qa_calibration.csv      ← Calibrated confidence threshold + IoU curve
logs/yolo_training_metrics.csv    ← Per-epoch mAP + loss
data/yolo_dataset/                ← YOLO-format dataset with train/val split
```

---

## PART 8 — SAM2 MASKING (generate_tier1_masks.py) [v3: YOLO-guided]

### 8.1 Auto-Prompting Strategy (v3)
```
For each Tier 1 image:

1. load_image_rgb(path) → EXIF-corrected uint8 RGB

2. [YOLO path] Run YOLOv8n → tight leaf bounding box (x1, y1, x2, y2)
   - Restrict HSV tissue search to pixels inside that box
   - No centroid drift: diseased yellow/brown tissue is contained within the box
   - Build 3 foreground points along vertical leaf axis, clamped to box
   - Pass box as SAM2 box= prompt (hard spatial constraint)

3. [HSV fallback] If YOLO absent or detection fails:
   Full-image combined green+yellow HSV mask
   - Green:  H∈[35,85],  S>40, V>40  (healthy tissue)
   - Yellow: H∈[15,45],  S>40, V>60  (MSV/MLN diseased tissue)
   - Largest connected component centroid → 3 foreground points along vertical axis
   - Center-of-image fallback if no tissue detected (no rejection at prompt stage)

4. Four image corners (10px inset) → background point prompts (label=0)

5. SAM2.predict(point_coords, point_labels, box=yolo_box_or_None)

6. Select mask with highest SAM2 score
   → convert logits to sigmoid → float32 probability map [0,1]

7. Store raw float32 .npy (NO binarization) + binarized uint8 .png

QA report columns: filename, category, status, reason, prompt_strategy,
                   prompt_mode (yolo|hsv_fallback), yolo_box, coverage, mean_conf
```

### 8.2 QA Filters (v3 — relaxed thresholds)
```
Filter 1 — Coverage range:
  Reject if fg_coverage < 0.03  (v1 was 0.10 — diseased leaves are sparser)
  Reject if fg_coverage > 0.90

Filter 2 — Mean foreground confidence:
  Threshold: calibrated from gold-standard IoU via train_yolo_detector.py
  Default fallback: 0.65 if calibration file absent

Filter 3 — Shape sanity:
  aspect = max(h, w) / min(h, w)
  Reject if aspect < 1.01  (v1 was 1.20 — overhead/square-frame leaves now pass)

Target: < 8% rejection rate
```

### 8.3 Output Files per Image
```
data/tier1_leaf_masks/{stem}_softmask.npy   ← float32 [0,1] probability map (MAIN TARGET)
data/tier1_leaf_masks/{stem}_mask.png       ← uint8 255/0 binary visualization
```

Overall: `tier1_qa_report.csv`

---

## PART 9 — TEACHER MODEL (train_teacher.py)

### 9.1 Purpose
Offline segmentation model. Never deployed to Android. Generates leaf silhouette pseudo-masks for ~215k Tier 2 images via Factory. Higher quality than Otsu; lighter than SAM2 at scale.

### 9.2 Architecture Comparison
| Variant | Encoder | Decoder | Params | Notes |
|---|---|---|---|---|
| resnet50 | ResNet-50 | UNet | ~32M | Deep CNN baseline |
| **efficientnet-b2** | EfficientNet-B2 | UNet | ~7.7M | **Recommended** |
| mit_b2 | SegFormer-B2 (Mix Transformer) | UNet | ~25M | Hierarchical ViT encoder |
| deeplabv3plus-eb2 | EfficientNet-B2 | DeepLabV3+ | ~7.7M | Decoder comparison (ASPP) |

All via `segmentation_models_pytorch`. SegFormer-B2 uses `encoder_name="mit_b2"` via timm. `deeplabv3plus-eb2` uses the same EfficientNet-B2 encoder as the UNet variant with a DeepLabV3+ ASPP decoder — true decoder architecture comparison.

### 9.3 Soft Target Training
```
Targets: SAM2 float32 probability maps in [0,1]
         NO binarization — preserves boundary uncertainty
         Interior pixels: near 1.0; exterior: near 0.0
         Edge pixels: 0.3–0.7 (ambiguous region — boundary smoothing)

Loss: smp.losses.DiceLoss(mode="binary", from_logits=True)
      Accepts float targets natively — no code change needed

Why soft targets: teaches model to output calibrated probabilities at boundaries
                  rather than forcing a binary commitment
Citation: Ke et al. (2020) "Guided Collaborative Training for Semi-Supervised Learning"
```

### 9.4 Hyperparameters
```
Input size       : 512×512 (larger than Student — preserves boundary detail)
Batch size       : 8
Epochs           : 30
LR               : 5e-5
Weight decay     : 1e-4
Optimizer        : AdamW
Scheduler        : ReduceLROnPlateau(factor=0.5, patience=5, mode=max)
Early stop       : patience=10, monitors val Dice
Checkpoint       : best val Dice → teacher_{variant}_best.pth
                   Best variant also copied to teacher_model_best.pth (canonical)
safe_collate     : yes
Gradient clipping: clip_grad_norm_(max_norm=5.0)
```

### 9.5 Augmentation (training only)
```
LongestMaxSize(512) + PadIfNeeded(512, 512)
HorizontalFlip(p=0.5)
VerticalFlip(p=0.5)
RandomRotate90(p=0.5)
RandomBrightnessContrast(limit=0.2, p=0.3)
HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2)
Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
ToTensorV2()
```

### 9.6 Metrics
```
Per epoch: train_loss, val_loss, val_dice, val_iou, val_recall, val_precision,
           val_specificity, lr
Test eval: test_dice, test_iou, test_recall, test_precision, test_specificity
           (Tier 1 test-split images only)
Latency:   20 CPU inference passes per variant → mean ms/image
Comparison: logs/teacher_comparison.csv (best_dice, lat_cpu_ms per variant)
            logs/teacher_test_metrics.csv
Overlays:  reports/teacher_overlays/ (5 per class, green contour on original image)
```

### 9.7 Manifest Filtering
```
Load tier1_manifest.csv → all Tier 1 images
Load global_split_manifest.csv → get test-split source filenames
Exclude Tier 1 images whose source_path is in the test split
Apply internal 80/20 split to remaining images (Teacher-internal)
→ Teacher never sees test-split images even as Tier 1 training data
```

---

## PART 10 — FACTORY (factory_master.py)

### 10.1 Purpose
Process all ~215k train+val Tier 2 images. Generate pseudo-labels across 4 modes simultaneously. Each mode writes to its own subfolder so all 4 coexist on disk.

### 10.2 Per-Image Processing Stages
```
Stage 1 — Bouncer gate (imports from scripts/bouncer_inference.py):
  a. heuristic_prefilter(img_rgb) → currently always True (passthrough)
  b. neural_bouncer(img_rgb, model, threshold) → True/False
  → Rejected: log status ("filtered_heuristic" or "filtered_bouncer"), skip

Stage 2 — Leaf silhouette:
  Tier 1 images: load pre-existing SAM2 .npy (skip Teacher inference)
  Tier 2 images: Teacher inference at 512×512 → resize to original dimensions
  → Result: float32 probability map [0,1]

Stage 3 — Silhouette refinement:
  Threshold at 0.35 (lower than 0.5 to catch darker leaves)
  Morphological close then open (kernel=5, elliptical)
  → Result: cleaned binary uint8 silhouette

Stage 4 — Coverage guard → reliability weight:
  leaf_coverage = binary_sil.sum() / (H × W)
  < 15%  : weight = −1 (exclude, severity sentinel = −1)
  15–25% : weight = 0.30
  25–50% : weight = 0.70
  > 50%  : weight = 1.00

Stage 5 — HSV symptom masking (mode-dependent):
  img_hsv = to_hsv(img_rgb)   [guaranteed RGB→HSV via image_utils]
  Apply green exclusion zone first (remove healthy chlorophyll)
  Apply disease-specific HSV ranges (MSV: 4 bands, MLN: 5 bands)
  Mode A: Otsu silhouette + hard binary symptom
  Mode B: SAM2 hard binary silhouette + hard binary symptom
  Mode C: SAM2 soft float silhouette + hard binary symptom
  Mode D: SAM2 soft float silhouette + soft HSV confidence map

Stage 6 — Severity:
  severity = (symptom_pixels / leaf_pixels) × 100%
  Healthy leaves: severity = 0 (no symptom pixels after green exclusion)
```

### 10.3 HSV Ranges

**Green exclusion zone (applied first to all modes):**
```
H: 38–85, S: 80–255, V: 60–230
Removes healthy chlorophyll from all symptom masks
```

**MSV ranges (4 overlapping bands):**
```
R1: H 15–38,  S 50–255, V 150–255   ← bright yellow streaks
R2: H 20–45,  S 15–70,  V 130–255   ← pale yellow / early-stage
R3: H 0–179,  S 0–35,   V 210–255   ← near-white / bleached
    ⚠ R3 area filter: min 80px connected component (removes specular highlights)
R4: H 38–55,  S 10–55,  V 140–255   ← pale yellow-green
```

**MLN ranges (5 overlapping bands):**
```
R1: H 18–40,  S 40–255, V 90–255    ← chlorotic yellow
R2: H 5–20,   S 40–255, V 50–220    ← orange-amber necrosis
R3: H 22–55,  S 15–85,  V 80–240    ← pale yellow-green mosaic
R4: H 8–35,   S 0–45,   V 160–255   ← tan / straw tissue
R5: H 0–18,   S 30–180, V 30–150    ← dark brown dead tissue
```

### 10.4 Mode D Soft HSV Confidence Formula
```
For each pixel p inside leaf silhouette (after green exclusion):
  For each range r in {R1..R4} (MSV) or {R1..R5} (MLN):
    range_hit(p, r) = 1 if pixel falls within range bounds, else 0
    d(p, r) = 1 − mean(|h − h_center| / h_half,
                        |s − s_center| / s_half,
                        |v − v_center| / v_half)
              clipped to [0, 1]

confidence(p) = Σ(range_hit(p, r) × d(p, r)) / N_ranges
                clipped to [0, 1]

Interpretation:
  1.0 = pixel satisfies all ranges at their centers (highly symptomatic)
  0.0 = pixel satisfies no ranges (healthy or excluded)
```

### 10.5 4 Factory Modes
| Mode | Silhouette source | Symptom mask type | Subfolder |
|---|---|---|---|
| A | Otsu threshold | Hard binary HSV | `mode_a/` |
| B | SAM2 hard binary | Hard binary HSV | `mode_b/` |
| C | SAM2 soft float | Hard binary HSV | `mode_c/` |
| D | SAM2 soft float | Soft HSV confidence [0,1] | `mode_d/` |

### 10.6 Output Files per Image per Mode
```
data/pseudo_masks/{mode}/{stem}_silhouette.npy  ← float32 [0,1] leaf silhouette probability
data/pseudo_masks/{mode}/{stem}_symptom.npy     ← float32 confidence map (mode_d only)
data/pseudo_masks/{mode}/{stem}_symptom.png     ← uint8 binary mask (modes a, b, c)
data/pseudo_masks/{mode}/{stem}_sev.txt         ← severity % (float) or −1 (sentinel)
data/pseudo_masks/{mode}/{stem}_weight.txt      ← reliability weight (0.30 / 0.70 / 1.00 / −1)

Output resolution: All .npy and .png downscaled to STUDENT_IMG_SIZE (224×224) at write time.
  Silhouettes and mode_d symptom: INTER_LINEAR
  Binary symptom PNGs: INTER_NEAREST (preserves hard edges)
```

### 10.7 Factory Summary Statistics
```
Per mode × per class:
  n_total, n_processed, pct_processed
  mean_severity, std_severity, median_severity
  pct_symptomatic (severity > 0)
  n_excluded, pct_excluded (weight == −1)
  mean_sil_confidence (modes c, d only — sampled from 500 images)

Output: reports/factory_summary.csv
Filter breakdown: reports/factory_filter_breakdown.csv
  Columns: status, count  (processed / filtered_bouncer / filtered_heuristic / load_error / etc.)
```

---

## PART 11 — STUDENT MODEL (train_student.py)

### 11.1 Architecture
```
Input: [B, 3, 224, 224] float32 (ImageNet normalized)
    ↓
Shared Encoder (MobileNetV2 or variant)
  Returns: [feat_1, feat_2, ..., feat_n, bottleneck]
    ↓                              ↓
UNet Decoder (+ CBAM at skip     GAP on bottleneck → Dropout(0.3) → Flatten
connections for V2, V6)                ↓                    ↓
    ↓                         Classification Head     Severity Head
Segmentation Head             [B, 3] raw logits      [B, 1] → ReLU → clamp(0,1)
[B, 2, 224, 224] raw logits   → softmax → argmax     → ×100 at inference = severity %
  Ch0: leaf silhouette         HEALTHY=0 / MSV=1 / MLN=2
  Ch1: symptom mask
  → sigmoid → binary masks at 0.5
```

**Why ReLU+clamp instead of sigmoid for severity head:**
- Sigmoid ceiling prevents exactly 0.0 — HEALTHY leaves always show nonzero severity (incorrect)
- ReLU allows exactly 0.0; clamp(0,1) prevents negative outputs from ReLU edge cases
- At inference: multiply by 100 to get severity percentage

### 11.2 Five Encoder Variants
| V# | Encoder | Params | CBAM | TFLite | Role |
|---|---|---|---|---|---|
| V1 | mobilenet_v2 | 3.4M | No | Yes | Baseline |
| **V2** | **mobilenet_v2_cbam** | **~3.5M** | **Yes** | **Yes** | **Expected winner** |
| V3 | mobilenet_v3_small | 2.5M | No | Yes | Ultra-compact |
| V4 | efficientnet_b0 | 5.3M | No | Yes | Compound scaling, no attention |
| V6 | efficientnet_b0_cbam | ~5.4M | Yes | Yes | CBAM generalization test |

> V5 (MobileViT-XXS) was removed — `torch.einsum` self-attention operations generate TFLite subgraph errors that cannot be resolved without architectural surgery. V6 replaces it, testing whether CBAM attention gains generalize to an EfficientNet backbone.

### 11.3 CBAM Implementation
```
Class: CBAMBlock(channels)

  ChannelAttention(channels):
    GAP(1×1) + GMP(1×1)
    → shared MLP(channels → channels//16 → channels)
    → sigmoid
    x = x × sigmoid(gap_out + gmp_out).unsqueeze(−1, −1)

  SpatialAttention(kernel_size=7):
    [avg_pool, max_pool along channel dim] → concat
    → conv(2→1, 7×7) → sigmoid
    x = x × sigmoid(conv_out)

Insertion: CBAMUnetDecoder subclasses smp.decoders.unet.decoder.UnetDecoder
  _apply_cbam_to_features(features):
    for i, feat in enumerate(features[1:]):   # skip features[0] = bottleneck
        features[1+i] = cbam_blocks[i](feat)
    return features

  forward() explicitly calls _apply_cbam_to_features() — NOT monkey-patching.
  Preserves torch.save() / state_dict() serialisation.

Citation: Woo et al. (2018) ECCV "CBAM: Convolutional Block Attention Module"
```

### 11.4 Loss Functions

**A. Segmentation Loss (both channels averaged):**
```
smp.losses.DiceLoss(mode="binary", from_logits=True)
Accepts float32 targets [0,1] — soft boundary uncertainty preserved

l_seg = (DiceLoss(logits[:,0:1], tgt[:,0:1]) + DiceLoss(logits[:,1:2], tgt[:,1:2])) / 2
```

**B. Asymmetric Label Smoothing Loss:**
```
Prior matrix P (rows=true_class, cols=[HEALTHY, MSV, MLN]):
  HEALTHY → [0.90, 0.08, 0.02]   # early MSV looks like HEALTHY (Cruz et al. 2024)
  MSV     → [0.05, 0.90, 0.05]
  MLN     → [0.02, 0.05, 0.93]   # MLN is most visually distinct

L_cls = −Σ P[true_class] × log(softmax(logits))

Citation: Szegedy et al. (2016) "Rethinking Inception Architecture"
          Cruz et al. (2024), Mushayi et al. (2025) — pathological basis
```

**C. Reliability-Weighted Severity Loss:**
```
Valid samples only (weight ≥ 0, i.e. not sentinel −1):
  l_sev = (sample_weights × MSE(sev_pred, sev_target)).mean()

sample_weights from Factory reliability brackets:
  coverage < 15%   : excluded entirely (weight = −1, not in batch)
  15–25%           : weight = 0.30
  25–50%           : weight = 0.70
  > 50%            : weight = 1.00

Citation: Jiang et al. (2018) ICML "MentorNet"
```

**D. Homoscedastic Uncertainty Loss (multi-task balancing):**
```
3 learnable log-variance parameters: s1 (seg), s2 (cls), s3 (sev)
Initialized to 0.0 (equal initial weighting)

L_total = exp(−s1) · L_seg + s1
        + exp(−s2) · L_cls + s2
        + exp(−s3) · L_sev + s3

s1, s2, s3 learned via backprop alongside model parameters.
Replaces manual fixed weights (0.6 / 0.2 / 0.2).
Citation: Kendall et al. (2018) NeurIPS "Multi-Task Learning Using Uncertainty"
```

### 11.5 Two-Phase Transfer Learning
```
Phase 1 — Frozen encoder (up to 30 epochs):
  encoder.parameters(): requires_grad = False
  Optimizer: Adam(decoder + heads + log_vars, lr=1e-3, weight_decay=1e-4)
  Scheduler: CosineAnnealingLR(T_max=30, eta_min=1e-6)
  Early stop: patience=10 (monitors quality composite)

Phase 1→2 transition:
  If early stop patience exceeded → proceed to Phase 2 regardless
  Unfreeze encoder: all parameters.requires_grad = True
  Reinitialize optimizer with lr=1e-4 (all parameters + log_vars)
  Reset patience counter to 0

Phase 2 — Full fine-tuning (up to 20 epochs):
  Optimizer: Adam(all parameters + log_vars, lr=1e-4, weight_decay=1e-4)
  Scheduler: CosineAnnealingLR(T_max=20, eta_min=1e-6)
  Early stop: patience=5 (tighter — catch fast convergence)

Gradient clipping: clip_grad_norm_(model + unc_loss params, max_norm=5.0)
  Applied after both model and unc_loss backward pass.
```

### 11.6 Two-Stage Selection Criterion

**Stage A — Quality composite (used during training for checkpoint saving):**
```
quality_composite = 0.50 × mIoU
                  + 0.35 × MSV_F1
                  + 0.15 × (1 − sev_mae_normalized)

sev_mae_normalized = sev_mae_pct / 100.0

Monitors val set. Saves best checkpoint when quality_composite improves.
Applied at every val epoch end, both phases.

Why quality-only during training: TFLite size and mobile latency are not
known until after training and export. Cannot include them in the training loop.
```

**Stage B — Mobile composite (used POST-training by select_best_pipeline.py):**
```
mobile_composite = 0.38 × msv_f1
                 + 0.22 × sil_mIoU
                 + 0.22 × speed_score
                 + 0.10 × (1 − norm_mae)
                 + 0.08 × size_score

speed_score = clamp(150ms / cpu_lat_mean_ms, 0.0, 1.0)
size_score  = clamp(15MB  / tflite_size_mb,  0.0, 1.0)
  If TFLite size unknown: size_score defaults to 0.80 (conservative)

Weight rationale:
  MSV_F1  0.38 — Primary clinical metric; MSV detection is the thesis claim
  mIoU    0.22 — Segmentation quality; directly visible in app as green overlay
  Speed   0.22 — Mobile usability; total pipeline must feel responsive
  Severity 0.10 — Secondary; severity display improves farmer trust
  Size    0.08 — Minor differentiator; MobileNet variants all score near 1.0

Target values:
  150ms — realistic for Snapdragon 680-class mid-range Android
          total pipeline: Bouncer ~30ms + Student ~120ms ≈ 250ms
  15MB  — comfortable for app store distribution; MobileNet FP16 ≈ 3–7MB

Input: logs/student_test_metrics_{enc}_{mode}.csv (have CPU latency)
Output: reports/student_mobile_ranking.csv
```

### 11.7 WeightedRandomSampler
```
class_counts = {HEALTHY: n1, MSV: n2, MLN: n3}
sample_weight[i] = 1.0 / class_counts[sample[i].category]
WeightedRandomSampler(weights=sample_weights, num_samples=len(samples), replacement=True)

Ensures proportional class representation per batch.
MSV (~24%) is otherwise underrepresented in random batches.
```

### 11.8 Augmentation (training only)
```
LongestMaxSize(224) + PadIfNeeded(224, 224, border_mode=0)
HorizontalFlip(p=0.5)
VerticalFlip(p=0.5)
RandomRotate90(p=0.5)
Rotate(limit=30, p=0.5)
RandomBrightnessContrast(limit=0.25, p=0.3)
HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=10, p=0.2)
RandomShadow(p=0.2)          ← tropical domain adaptation
Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
ToTensorV2()
```

### 11.9 Two-Stage Ablation Protocol
```
Stage 1: Fix mode = mode_b
         Train all 5 encoder variants (V1–V4, V6)
         Evaluate each on global test split
         Select best encoder by quality composite score
         Checkpoints → checkpoints/student/stage1/  ← ISOLATED (prevents Stage 2 overwrite)

Stage 2: Fix encoder = best from Stage 1
         Train on all 4 Factory modes (A, B, C, D)
         Mode B result from Stage 1 REUSED (same seed=42 → identical weights + trajectory)
         Evaluate each mode on global test split
         Select best mode by quality composite score
         Checkpoints → checkpoints/student/

Total unique training runs: 5 (Stage 1) + 3 (Stage 2, Mode B reused) = 8
Stated limitation: not fully crossed factorial — documented in Chapter 3 and Chapter 5.
```

### 11.10 Metrics Tracked

**Segmentation (global TP/FP/FN/TN accumulation — NOT mean-of-batches):**
```
Ch0 (silhouette): mIoU, Dice, Recall, Precision, Specificity
Ch1 (symptom):    mIoU, Dice, Recall, Precision, Specificity

Note: Global accumulation is statistically correct.
Per-batch mIoU averaging is wrong because batch class compositions vary,
producing different denominators → biased estimates.
```

**Classification:**
```
Per class (HEALTHY, MSV, MLN): Precision, Recall, F1
Overall: Accuracy, Macro F1, Weighted F1, MCC (Matthews Correlation Coefficient)
Confusion matrix: 3×3 saved to logs/student_confusion_{enc}_{mode}.csv
```

**Severity regression:**
```
MAE%  — mean absolute error in percentage points
RMSE% — root mean squared error in percentage points
R²    — coefficient of determination
Note: measured against HSV-derived pseudo-labels, not expert ratings
```

**Latency:**
```
CPU timed inference: 220 images (20 warm-up + 200 measured)
Metrics: mean ms/image, std ms/image, FPS
Device: CPU (simulates mobile/edge — GPU not available at deployment)
```

---

## PART 12 — SOFT LABEL STRATEGY

### 12.1 Where Soft Labels Are Used
| Location | Soft target source | Represents | Loss |
|---|---|---|---|
| Teacher training | SAM2 float32 probability map | Boundary uncertainty | DiceLoss (float targets) |
| Student Ch0 (silhouette) | Teacher soft pseudo-silhouette | Propagated boundary uncertainty | DiceLoss (float targets) |
| Student Ch1 (symptom) | HSV confidence map (Mode D) / binary (A/B/C) | Disease confidence | DiceLoss (float or binary) |
| Student classification | Asymmetric prior matrix | Pathological confusion structure | Custom cross-entropy |
| Student severity | Reliability-weighted MSE | Variable HSV reliability | Weighted MSELoss |

### 12.2 Why NOT Sigmoid on Severity Head
```
Sigmoid: output approaches but never reaches 0.0 or 1.0
         HEALTHY leaves always show nonzero severity (biologically incorrect)

Fix: Linear(→1) → ReLU → clamp(0,1) → ×100 at inference
     ReLU allows exactly 0.0 for healthy leaves
     clamp(0,1) prevents negative values from ReLU numerical edge cases
```

---

## PART 13 — XAI (evaluate_xai.py)

### 13.1 Three Methods Compared
| Method | Type | Library class | Strength |
|---|---|---|---|
| Grad-CAM | Gradient-based | GradCAM | Historical baseline |
| **Grad-CAM++** | Gradient-based | GradCAMPlusPlus | Better for multi-region MSV streaks — **deployed** |
| Score-CAM | Gradient-free | ScoreCAM | No gradient noise, stable reference |

**Library:** pytorch-grad-cam (`pip install grad-cam`)

### 13.2 Target Layer Selection
```
Applied to last convolutional block of shared encoder (before decoder branches).
Reflects shared features driving both classification AND segmentation simultaneously.

Encoder → Target layer
mobilenet_v2:            encoder.features[-1][0]
mobilenet_v2_cbam:       encoder.features[-1][0]
mobilenet_v3_small:      encoder.features[-1][0]
efficientnet_b0:         encoder.blocks[-1][-1]
efficientnet_b0_cbam:    encoder.blocks[-1][-1]
```

### 13.3 Two Distinct App Outputs
```
Output 1 — Symptom boundary (green contour):
  Source: UNet segmentation head Ch1
  Nature: Pixel-level localization
  Display: crisp green contour overlay
  Label in app: "Symptom boundary"

Output 2 — Diagnostic attention (amber heatmap):
  Source: Grad-CAM++ on last encoder conv block
  Nature: Class-discriminative explanation (~7×7 upsampled to 224×224)
  Display: semi-transparent amber heatmap overlay
  Label in app: "Diagnostic attention"

⚠ CRITICAL THESIS NOTE:
These are NOT the same thing. Grad-CAM does NOT provide pixel-level segmentation.
Objective 3 must describe both separately and never conflate them.
```

### 13.4 Quantitative Metrics
```
Pointing game accuracy:
  Top 20% of heatmap activation (pixels ≥ heatmap.max() × 0.80)
  Fraction that falls inside the segmentation mask (ground truth ROI)

Insertion AUC (n_steps=6):
  Progressively reveal pixels in importance order (most important first)
  Classifier confidence should increase as more important pixels revealed
  AUC of confidence curve = insertion_auc

Deletion AUC (n_steps=6):
  Progressively remove pixels in importance order
  Classifier confidence should decrease as important pixels removed
  AUC of confidence curve = deletion_auc

All computed on GPU for speed.
```

---

## PART 14 — BEST PIPELINE SELECTION (select_best_pipeline.py)

### 14.1 Selection Criteria
```
Bouncer:  max specificity subject to maize_recall ≥ 0.95
          Neural variants only (gabor_lbp, patchcore excluded from deployment)

Teacher:  max val Dice

Student:  TWO-STAGE selection:
  During training → quality composite (checkpoint saving):
    0.50 × mIoU + 0.35 × MSV_F1 + 0.15 × (1 − NormMAE)

  Post-training → mobile composite (deployment selection):
    MSV_F1 × 0.38 + sil_mIoU × 0.22 + speed × 0.22 + sev × 0.10 + size × 0.08
    speed = clamp(150ms / cpu_lat, 0, 1)
    size  = clamp(15MB / tflite_mb, 0, 1)
    TFLite-incompatible variants excluded from deployment ranking.
```

### 14.2 Canonical Checkpoint Promotion
```
Best Bouncer → checkpoints/final/bouncer_best.pth
Best Teacher → checkpoints/final/teacher_best.pth
Best Student → checkpoints/final/student_best.pth  (best by MOBILE composite)

All downstream scripts check final/ first, then fall back to per-variant paths.
```

### 14.3 Config Auto-Update
```
select_best_pipeline.py edits config.py via regex:
  STUDENT_BEST_VARIANT = "{winner_encoder}"
  STUDENT_FACTORY_MODE = "{winner_mode}"

No manual config.py editing needed after Stage 2.
```

### 14.4 Outputs
```
checkpoints/final/bouncer_best.pth
checkpoints/final/teacher_best.pth
checkpoints/final/student_best.pth
reports/best_pipeline_summary.csv       ← all winners + quality + mobile metrics
reports/best_pipeline_summary.txt       ← thesis-formatted results table (both composites)
reports/all_variants_ranked.csv         ← every variant ranked by primary metric
reports/student_mobile_ranking.csv      ← Student variants ranked by mobile composite
```

---

## PART 15 — DEPLOYMENT (export_tflite.py + build_deployment_package.py)

### 15.1 TFLite Export Path
```
Student: PyTorch → ONNX (opset 12) → TF SavedModel → TFLite (FP16)
Bouncer: PyTorch → ONNX (opset 12) → TF SavedModel → TFLite (FP16)

Both models go through onnx-tf conversion.
FP16 quantization: tf.lite.Optimize.DEFAULT + target_spec=[tf.float16]
```

### 15.2 Student TFLite Input/Output
```
Input:   shape=[1, 3, 224, 224], dtype=float32, layout=NCHW
         ImageNet normalized: (pixel/255 − mean) / std

Output 0 (segmentation): shape=[1, 2, 224, 224], dtype=float32 (raw logits)
  Post: sigmoid(output) → binary mask at threshold 0.5
  Ch0: leaf silhouette
  Ch1: symptom mask

Output 1 (classification): shape=[1, 3], dtype=float32 (raw logits)
  Post: softmax(output) → argmax → class_index
  0=HEALTHY, 1=MSV, 2=MLN

Output 2 (severity): shape=[1, 1], dtype=float32, range=[0, 1]
  Post: value × 100 = severity_percentage
```

### 15.3 Bouncer TFLite Input/Output
```
Input:  shape=[1, 3, 224, 224], dtype=float32, NCHW (same normalization as Student)
Output: shape=[1, 1], dtype=float32 (raw logit)
  Post: sigmoid(logit) → probability
        if probability ≥ threshold → PASS to Student
        if probability < threshold → REJECT (show "not a maize leaf" message)
```

### 15.4 Deployment Package (exports/deploy/)
```
bouncer_model.tflite        ← deploy to Android assets/
student_model.tflite        ← deploy to Android assets/
model_metadata.json         ← complete spec (shapes, normalization, thresholds, class names, post-processing)
DEPLOYMENT_README.md        ← Kotlin/Java code snippets for Android Studio
deployment_report.csv       ← sizes, latencies, validation status
```

### 15.5 Android Runtime
```
Minimum API: 21
TFLite: org.tensorflow:tensorflow-lite:2.13.0
Support: org.tensorflow:tensorflow-lite-support:0.4.4
GPU delegate: org.tensorflow:tensorflow-lite-gpu:2.13.0 (optional)

Image pipeline:
  1. Apply EXIF orientation correction
  2. Letterbox resize to 224×224 (LongestMaxSize + PadIfNeeded equivalent)
  3. Normalize (ImageNet mean/std)
  4. Layout: NCHW [1, 3, 224, 224]
  5. Heuristic pre-filter (green coverage + aspect ratio)  ← currently passthrough
  6. Bouncer inference → sigmoid → threshold
  7. If rejected → display "Not a maize leaf"
  8. If passed → Student inference → 3 outputs
  9. Post-process each output head
  10. Display: green contour (silhouette boundary) + amber heatmap (Grad-CAM++) +
              class badge + severity gauge
```

---

## PART 16 — HTML REPORT (generate_report.py)

### 16.1 Output
```
reports/evaluation_report.html
  - 100% self-contained (all charts base64 embedded, all images inline)
  - Dark theme (navy #0F172A background, teal #34D399 accent)
  - No external dependencies to view — open directly in any browser
```

### 16.2 Nine Sections
```
1. Preprocessing Summary
   - Rejection counts and breakdown table by reason and class

2. Bouncer Gate Comparison
   - Multi-metric bar chart (specificity, recall, F1, AUC)
   - Admission rate cards per variant
   - Bouncer comparison table with TP/FP/TN/FN

3. Teacher Model Comparison
   - Dice bar chart + latency panel
   - Training curves (loss, Dice, IoU, Recall, Specificity over epochs)
   - Qualitative leaf silhouette overlays (15 images)
   - Test evaluation metrics

4. Student Encoder Ablation (Stage 1)
   - Multi-metric comparison table (all 5 variants)
   - Grouped bar chart (composite, mIoU, MSV_F1)

5. Student Mode Ablation (Stage 2)
   - Mode A/B/C/D comparison table
   - Grouped bar chart
   - Severity distribution histograms per mode × class

6. Best Student — Full Test Results
   - Headline metric cards
   - Per-class P/R/F1 table
   - Complete metrics table (segmentation, classification, severity, latency)
   - 3×3 confusion matrix heatmap
   - Training curves (loss, mIoU, MSV_F1, composite over epochs)

7. XAI Comparison
   - Pointing game / Insertion / Deletion bar chart
   - Note distinguishing two distinct app outputs
   - Embedded overlay images (15+ per method)

8. Severity Reliability
   - Cohen's Kappa + Spearman ρ cards with quality badges
   - Interpretation text + rating table

9. Deployment Summary
   - TFLite size + latency cards
   - Deployment package contents list
```

---

## PART 17 — INFRASTRUCTURE

### 17.1 safe_collate (scripts/safe_collate.py)
```
Purpose: Every DataLoader uses collate_fn=safe_collate.
         If __getitem__ returns None (corrupt image), safe_collate filters it out.
         If entire batch is None, returns None — training loop skips with continue.

Without this: one corrupt image crashes the DataLoader worker silently.
With this: training continues; skip counter is logged per epoch.

API:
  safe_collate(batch) → collated batch or None
  reset_skip_counter() → None   (call at epoch start)
  get_skip_count() → int         (call at epoch end to log skipped images)
```

### 17.2 Shared Bouncer Inference (scripts/bouncer_inference.py)
```
Single source of truth for inference helpers used by BOTH factory_master.py
and train_bouncer.py. Changes to transform pipeline or threshold logic are
made here once — not duplicated.

BOUNCER_INFER_TF:
  A.LongestMaxSize(BOUNCER_IMG_SIZE)
  A.PadIfNeeded(BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE, border_mode=0, value=0)
  A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  ToTensorV2()
  Used at inference time (same as val transform — letterbox + normalize, no augmentation)

heuristic_prefilter(img_rgb: np.ndarray) → bool
  Currently a passthrough (always returns True).
  Original OpenCV green-coverage heuristic removed after causing false rejections
  on yellow/bleached MSV leaves under variable tropical lighting.

neural_bouncer(img_rgb: np.ndarray, model: nn.Module, threshold: float) → bool
  Applies BOUNCER_INFER_TF, runs model, returns sigmoid(logit) >= threshold.
  @torch.no_grad() decorated.

DEVICE = torch.device("cuda" if available else "cpu")

Dependencies: torch, albumentations, config.BOUNCER_IMG_SIZE ONLY.
No SAM2, Teacher, or any heavy imports — lightweight for use anywhere.
```

### 17.3 Gradient Clipping
```
Applied in all training scripts (Bouncer, Teacher, Student):
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

In Student: also clips unc_loss (log-variance) parameters.

Rationale:
  - Homoscedastic log_vars can spike early in training
  - Phase 2 encoder unfreeze can cause large initial gradients
  - Multi-task loss interactions can amplify gradient magnitudes
```

### 17.4 Reproducibility
```
set_seeds(seed=42) in every training script:
  random.seed(42)
  numpy.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

Stage 1 Mode B checkpoint is reused for Stage 2 Mode B because:
  - Same seed=42 → identical weight initialization
  - Same global_split_manifest.csv → identical data splits
  - Same hyperparameters → identical training trajectory
  No re-training needed. Just copy the checkpoint.
```

### 17.5 Timing
```
Every script records wall-clock duration:
  _t_start = time.time() at main() entry
  duration = time.time() − _t_start at main() exit
  Duration printed and included in output CSV where applicable.

Latency measurements:
  Teacher: 20 CPU inference passes per variant → mean ms/image
  Student: 220 CPU passes (20 warm-up + 200 measured) → mean ± std ms, FPS
  Bouncer: via deployment validation in build_deployment_package.py → mean ± std ms
```

---

## PART 18 — GOLD STANDARD VALIDATION (validate_gold_standard.py)

### 18.1 Purpose
Validate pseudo-label quality against human-annotated leaf silhouette masks. Required for thesis defense — without it, the committee can challenge whether SAM2 pseudo-masks were accurate enough to serve as Teacher training targets.

Provides a chain comparison on the SAME 501 images:
```
SAM2 pseudo-mask → Teacher prediction → Student prediction
       ↓                   ↓                    ↓
  IoU vs human        IoU vs human         IoU vs human
     mask                mask                 mask
```

### 18.2 Label Studio Annotation Parser
```
Accepts Label Studio JSON export format (polygonlabels task type).
Polygon points stored as percentage of image dimensions.
Rasterized to binary mask at inference time using cv2.fillPoly().
Multiple polygons per image merged via logical OR.
Handles both list-of-tasks and single-task export formats.
```

### 18.3 IoU Computation
```
Binary IoU:
  iou = intersection / union
  Returns 0.0 if both masks are entirely empty (degenerate case).

Thresholds (config.py):
  GOLD_IOU_WARN_THRESHOLD = 0.75    ← per-image flag (below = suspicious)
  GOLD_IOU_TARGET_MEAN    = 0.85    ← overall target (≥ this = foundation valid)
```

### 18.4 Run Order
```
Step 6c: validate_gold_standard.py --sam2-only
  → Run after generate_tier1_masks.py, before train_teacher.py
  → Validates SAM2 foundation only (no model loading)

Step 6d: validate_gold_standard.py
  → Run after train_teacher.py
  → Full chain: SAM2 + Teacher IoU vs human

Step 10b: validate_gold_standard.py
  → Run after train_student.py
  → Complete chain: SAM2 → Teacher → Student IoU (thesis Chapter 3/4 table)
```

### 18.5 Metrics Reported
```
Per image: sam2_iou, teacher_iou, student_iou (−1 if unavailable), *_warn flags

Per class (HEALTHY / MSV / MLN) per artifact:
  mean IoU ± std, n, count below warning threshold

Overall per artifact:
  mean IoU ± std, n_total, n_below_warn, target_met (bool)

Chain comparison table (thesis-ready):
  Artifact | Overall mean IoU | Target met
  SAM2     | x.xxxx           | Yes/No
  Teacher  | x.xxxx           | Yes/No
  Student  | x.xxxx           | Yes/No
```

### 18.6 Outputs
```
reports/gold_standard_iou_report.csv    ← per-image IoU for all 3 artifacts
reports/gold_standard_iou_summary.csv   ← mean ± std per class + overall
reports/gold_standard_overlays/         ← visual comparison PNGs (5 per class per artifact)
  {stem}_sam2_overlay.jpg               ← Green=missed by pred, Cyan=correct, Red=extra
  {stem}_teacher_overlay.jpg
  {stem}_student_overlay.jpg
```

### 18.7 Config Keys Required
```python
GOLD_IMAGES_DIR          = DATA_DIR / "gold_standard" / "images"
GOLD_MANIFEST            = DATA_DIR / "gold_standard" / "gold_manifest.csv"
GOLD_ANNOTATION_FILE     = DATA_DIR / "gold_standard" / "annotations" / "annotations.json"
GOLD_IOU_WARN_THRESHOLD  = 0.75
GOLD_IOU_TARGET_MEAN     = 0.85
TEACHER_DEPLOYED_VARIANT = "efficientnet-b2"
```

---

## PART 19 — CHART GENERATOR (generate_charts.py)

### 19.1 Design Principles
```
- No dependency on project modules (config.py, image_utils.py, etc.)
  Reads ONLY from logs/ — safe to run on any machine with the log files
- Dark theme (navy #0F172A background, teal #34D399 accent)
  Consistent with evaluation_report.html
- Consistent colour palette: same variant always gets the same colour
- Output directory: reports/charts/ (auto-created if missing)
- Non-interactive Agg backend — works headlessly on WSL2 / servers
- 150 DPI PNG output, bbox_inches="tight"
- Missing CSVs are silently skipped — safe to run mid-training
```

### 19.2 Usage
```bash
python generate_charts.py              # all charts
python generate_charts.py --bouncer    # bouncer charts only
python generate_charts.py --teacher    # teacher charts only
python generate_charts.py --student    # student charts only
```

### 19.3 Input CSVs → Output PNGs

**Bouncer (--bouncer):**
```
logs/bouncer_comparison.csv
  → reports/charts/bouncer_comparison_bar.png
     Grouped bar chart: F1 / Specificity / Maize Recall / ROC-AUC per variant.
     Deployed model (mobilenet_v3_large) highlighted with a ▲ marker.

logs/bouncer_{variant}_metrics.csv   (one per neural variant)
  → reports/charts/bouncer_training_curves_{variant}.png
     3 panels: Loss curves · F1 & Accuracy · Specificity / Recall / Precision

logs/bouncer_comparison.csv  (TP/FP/TN/FN columns)
  → reports/charts/bouncer_confusion_matrix.png
     2×2 confusion heatmap for all neural variants side by side.
     Raw counts + normalised percentages shown in each cell.
```

**Teacher (--teacher):**
```
logs/teacher_comparison.csv
  → reports/charts/teacher_comparison_bar.png
     Two panels: Best Validation Dice · CPU Inference Latency (ms @ 512px).
     Best Dice variant highlighted with ★ marker.

logs/teacher_{variant}_metrics.csv   (one per variant)
  → reports/charts/teacher_training_curves_{variant}.png
     3 panels: Loss · Dice & IoU · Recall / Precision / Specificity
```

**Student (--student):**
```
logs/student_comparison_stage1.csv
  → reports/charts/student_stage1_comparison_bar.png
     Grouped bar: Sil mIoU / Sym mIoU / MSV F1 / Macro F1 / Composite.
     Best composite shaded + ★ label.

logs/student_comparison_stage2.csv
  → reports/charts/student_stage2_comparison_bar.png
     Same metrics, grouped by Factory mode (A / B / C / D).

logs/student_{enc}_{mode}_metrics.csv  (one per training run)
  → reports/charts/student_training_curves_{enc}_{mode}.png
     2×3 grid: Loss · Seg mIoU · Seg Dice · Cls F1 · Sev MAE · Composite
     Phase 1 / Phase 2 boundary marked with a vertical dashed line.

logs/student_confusion_{enc}_{mode}.csv
  → reports/charts/student_confusion_{enc}_{mode}.png
     3×3 confusion matrix: raw counts (left) + row-normalised recall (right).

logs/student_test_metrics_{enc}_{mode}.csv
  → reports/charts/student_radar_{enc}_{mode}.png
     Radar chart of 6 test metrics:
       Sil mIoU · Sym mIoU · MSV F1 · Cls Accuracy · Composite · Sev R²

  → reports/charts/student_radar_all_overlay.png
     All variants overlaid on one radar chart for direct comparison.

  → reports/charts/student_metrics_heatmap.png
     Colour heatmap: rows = variant × mode, columns = all key metrics.
     Columns normalised to [0,1] (greener = better).
     MAE% and CPU ms columns inverted before normalising (lower is better).
```

### 19.4 Colour Palette (consistent across all charts)
```
gabor_lbp              #6B7280  grey
mobilenet_v2           #3B82F6  blue       (thesis primary architecture)
mobilenet_v3_large     #10B981  emerald    (deployed Bouncer)
edgevit_xxs            #F59E0B  amber
resnet50               #6B7280  grey
efficientnet-b2        #10B981  emerald    (deployed Teacher)
mit_b2                 #F59E0B  amber
deeplabv3plus-eb2      #EF4444  red
mobilenet_v2_cbam      #8B5CF6  violet     (expected Student winner)
mobilenet_v3_small     #EC4899  pink
efficientnet_b0        #F97316  orange
efficientnet_b0_cbam   #14B8A6  teal
mode_a                 #6B7280  grey
mode_b                 #3B82F6  blue
mode_c                 #10B981  emerald
mode_d                 #F59E0B  amber
```

---

## PART 20 — EXECUTION GUIDE

### Full Step-by-Step Commands
```bash
# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2

# STEP 1: Preprocessing + global split (RUN ONCE — never re-run after training begins)
python partition_dataset.py

# STEP 2: Build Bouncer dataset
python create_bouncer_dataset.py

# STEP 3: Train all Bouncer variants
python train_bouncer.py
python generate_charts.py --bouncer      # optional: generate charts now

# STEP 4: Sample 15k Tier 1 images
python sample_15000.py

# STEP 4a: Export 501 images for YOLO bounding-box annotation
python sample_yolo_annotations.py
# → Import data/yolo_annotations/images/ into Label Studio (Object Detection task)
# → Draw ONE bounding box per image (primary leaf only). Label = "leaf"
# → Export YOLO format → data/yolo_annotations/labels/

# STEP 4b: Export 501 images for gold standard polygon annotation
python sample_gold_standard.py
# → Upload data/gold_standard/images/ to Label Studio (polygonlabels task)
# → Annotate at least 400 leaf silhouettes as polygons
# → Export JSON → data/gold_standard/annotations/annotations.json

# STEP 4c: Train YOLOv8n leaf detector + calibrate SAM2 QA threshold
python train_yolo_detector.py
# → Requires ≥ 400 YOLO bbox annotations
# → Outputs: checkpoints/yolo/best.pt + logs/yolo_qa_calibration.csv

# STEP 5: SAM2 masking [v3: YOLO-guided box prompts]
python generate_tier1_masks.py
# → Review tier1_qa_report.csv; target < 8% rejection rate

# STEP 6: Review QA overlays before Teacher training
python validate_masks.py
# → Review reports/tier1_overlays/ manually

# STEP 6c: Validate SAM2 masks vs human annotations (before Teacher)
python validate_gold_standard.py --sam2-only
# → Target: mean IoU ≥ GOLD_IOU_TARGET_MEAN (0.85)

# STEP 7: Train all Teacher variants
python train_teacher.py
python generate_charts.py --teacher      # optional

# STEP 6d: Full chain validation — SAM2 + Teacher IoU vs human
python validate_gold_standard.py

# STEP 8: Generate pseudo-labels (all 4 modes simultaneously)
python factory_master.py

# STEP 9: Student Stage 1 — encoder ablation (5 variants × Mode B)
python train_student.py --stage 1

# STEP 10: Student Stage 2 — mode ablation (best encoder × 4 modes)
# Check logs/student_comparison_stage1.csv for best encoder first
python train_student.py --stage 2 --encoder mobilenet_v2_cbam
python generate_charts.py --student      # optional, or:
python generate_charts.py               # regenerate all charts at once

# STEP 10b: Final gold standard validation — complete chain for thesis table
python validate_gold_standard.py
# → reports/gold_standard_iou_summary.csv — SAM2 → Teacher → Student

# STEP 11: Select best pipeline, promote canonical checkpoints
python select_best_pipeline.py
# → auto-updates config.py STUDENT_BEST_VARIANT and STUDENT_FACTORY_MODE

# STEP 12: XAI evaluation (3 methods)
python evaluate_xai.py

# STEP 13: Severity inter-rater reliability
python evaluate_severity.py --sample   # two raters fill in reports/severity_sample.csv
python evaluate_severity.py --analyze

# STEP 14: Export Student to TFLite (FP16)
python export_tflite.py

# STEP 15: Build Android deployment bundle
python build_deployment_package.py

# STEP 16: Generate HTML evaluation report
python generate_report.py
# → open reports/evaluation_report.html in any browser
```

### Key Outputs at Pipeline Completion
```
exports/deploy/
  bouncer_model.tflite         ← Bouncer for Android
  student_model.tflite         ← Student for Android
  model_metadata.json          ← All specs for Android developer
  DEPLOYMENT_README.md         ← Kotlin/Java integration guide

reports/
  evaluation_report.html       ← Complete self-contained visual report
  best_pipeline_summary.txt    ← Thesis-ready results table
  gold_standard_iou_summary.csv ← Chain comparison (SAM2→Teacher→Student vs human)
  charts/                      ← All PNG comparison charts

logs/
  student_test_metrics_{enc}_{mode}.csv  ← Source for thesis tables (Chapters 3/4)
  student_confusion_{enc}_{mode}.csv     ← Confusion matrices
  bouncer_comparison.csv                 ← Bouncer results
  teacher_test_metrics.csv               ← Teacher results
  xai_comparison.csv                     ← XAI results
  severity_reliability.csv               ← Kappa + Spearman ρ
```

---

## PART 21 — KEY TERMINOLOGY

| Wrong term | Correct term |
|---|---|
| "Knowledge distillation" | "Pseudo-label semi-supervised learning" or "teacher-guided pseudo-label generation" |
| "Pixel-level segmentation via Grad-CAM" | Grad-CAM++ provides coarse class-discriminative localization (~7×7 upsampled to 224×224). UNet head provides pixel-level localization. These are entirely different outputs. |
| "Severity MAE measures disease severity" | "Severity MAE measures consistency with Factory's HSV-derived pseudo-labels, not expert agronomic ratings" |

---

## PART 22 — KEY CITATIONS

| Citation | Used for |
|---|---|
| Kendall et al. (2018) NeurIPS | Homoscedastic uncertainty loss for multi-task balancing |
| Szegedy et al. (2016) CVPR | Label smoothing / asymmetric prior |
| Woo et al. (2018) ECCV | CBAM convolutional block attention module |
| Ke et al. (2020) | Soft segmentation pseudo-label targets |
| Jiang et al. (2018) ICML | MentorNet — reliability-weighted training curriculum |
| Cruz et al. (2024) | First confirmed report of MSV in the Philippines |
| Mushayi et al. (2025) | MSV confusion with HEALTHY — asymmetric prior matrix basis |
| Selvaraju et al. (2017) ICCV | Grad-CAM |
| Chattopadhay et al. (2018) WACV | Grad-CAM++ |
| Wang et al. (2020) CVPR | Score-CAM |
| Mduma (2023) Mendeley | Tanzania Maize Imagery Dataset |
| Tan & Le (2019) ICML | EfficientNet |
| Xie et al. (2021) NeurIPS | SegFormer (mit_b2 encoder) |
| Pan et al. (2022) ECCV | EdgeViT |
| Fawcett (2006) | ROC threshold selection methodology |
| Zuiderveld (1994) | CLAHE |
