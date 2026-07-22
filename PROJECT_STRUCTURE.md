# Yellow MAIze — Project Structure & Dataset Guide

---

## Directory Layout

```
yellowmaize/
│
├── maize_dataset/                         ← Primary disease dataset (place here manually)
│   ├── HEALTHY/
│   ├── MSV/
│   └── MLN/
│
├── dataset/                               ← Non-maize negatives for Bouncer training
│   ├── raw_kaggle/
│   │   ├── intel/                         ← Intel Image Classification (6 scene categories)
│   │   └── natural/                       ← Natural Images (8 object categories)
│   └── crop_neighbors/                    ← Visually similar Philippine crops
│       ├── cogon_grass/                   ← Imperata cylindrica (iNaturalist Philippines)
│       ├── banana_leaf/                   ← Musa spp. (iNaturalist / PlantVillage)
│       ├── rice/                          ← Oryza sativa (PlantVillage)
│       ├── sorghum/                       ← Sorghum bicolor (PlantVillage)
│       └── sugarcane/                     ← Saccharum officinarum (iNaturalist)
│
├── sam2/                                  ← SAM2 weights (download separately)
│   └── sam2_hiera_large.pt
│
├── data/                                  ← All auto-generated data — do not edit manually
│   ├── bouncer_dataset/                   ← Phase 0a output
│   │   ├── maize/                         ← 25,000 maize positives (train+val only)
│   │   └── not_maize/                     ← 25,000 non-maize negatives
│   │
│   ├── tier1_raw/                         ← Phase 1 output
│   │   ├── HEALTHY_*.jpg / MSV_*.jpg / MLN_*.jpg
│   │   └── _manifest_hash.txt             ← SHA-256 lock — do not delete while masking
│   │
│   ├── tier1_leaf_masks/                  ← Phase 2 output
│   │   ├── {stem}_softmask.npy            ← float32 [0,1] SAM2 probability map
│   │   └── {stem}_mask.png                ← uint8 binary visualization
│   │
│   ├── gold_standard/                     ← Phase 1b output + Label Studio exports
│   │   ├── images/                        ← 501 renamed images (167 per class)
│   │   │   └── _manifest_hash.txt         ← SHA-256 lock — do not delete while annotating
│   │   ├── annotations/
│   │   │   ├── annotations.json           ← Label Studio leaf-silhouette export (place here)
│   │   │   │                                 Used by: train_yolo_detector.py (polygon→bbox)
│   │   │   │                                          validate_gold_standard.py (IoU chain)
│   │   │   │                                          train_yolo_detector.py (SAM2 QA cal.)
│   │   │   └── symptom_annotations.json   ← Label Studio symptom export (Phase 3b)
│   │   └── gold_manifest.csv
│   │
│   ├── yolo_dataset/                      ← Phase 1c output
│   │   ├── images/train/ + val/
│   │   ├── labels/train/ + val/
│   │   └── dataset.yaml
│   │
│   └── pseudo_masks/                      ← Phase 4 output — all 4 modes coexist
│       ├── mode_a/  ← Otsu silhouette + hard symptom
│       │   └── {stem}_silhouette.npy / _symptom.png / _sev.txt / _grade.txt / _weight.txt
│       ├── mode_b/  ← SAM2 binary silhouette + hard symptom
│       │   └── (same pattern)
│       ├── mode_c/  ← SAM2 soft silhouette + hard symptom
│       │   └── (same pattern)
│       └── mode_d/  ← SAM2 soft silhouette + soft symptom confidence
│           └── {stem}_silhouette.npy / _symptom.npy / _sev.txt / _grade.txt / _weight.txt
│
├── checkpoints/                           ← Auto-generated model weights
│   ├── yolo/
│   │   └── best.pt
│   ├── bouncer/
│   │   └── bouncer_{variant}_best.pth
│   ├── teacher/
│   │   ├── teacher_{variant}_best.pth
│   │   └── teacher_model_best.pth         ← canonical best, used by factory_master.py
│   ├── healthy_ae/
│   │   └── healthy_ae_best.pth            ← Phase 3b
│   ├── symptom/
│   │   └── symptom_teacher_best.pth       ← Phase 3b, used by factory_master.py
│   ├── student/
│   │   ├── stage1/
│   │   │   └── student_{enc}_mode_b_best.pth
│   │   └── student_{enc}_{mode}_best.pth
│   └── final/                             ← Canonical deploy targets
│       ├── bouncer_best.pth
│       ├── teacher_best.pth
│       └── student_best.pth
│
├── logs/                                  ← Auto-generated CSVs
│   ├── bouncer_{variant}_metrics.csv
│   ├── bouncer_comparison.csv
│   ├── bouncer_admission_rate_{variant}.csv
│   ├── nonmaize_validation.csv
│   ├── yolo_training_metrics.csv
│   ├── yolo_qa_calibration.csv
│   ├── teacher_{variant}_metrics.csv
│   ├── teacher_comparison.csv
│   ├── teacher_test_metrics.csv
│   ├── symptom_teacher_metrics.csv
│   ├── student_{enc}_{mode}_metrics.csv
│   ├── student_test_metrics_{enc}_{mode}.csv   ← primary source for thesis tables
│   ├── student_confusion_{enc}_{mode}.csv
│   ├── student_comparison_stage1.csv
│   ├── student_comparison_stage2.csv
│   ├── xai_comparison.csv
│   └── severity_reliability.csv
│
├── reports/                               ← Auto-generated reports and overlays
│   ├── preprocessing_report.csv
│   ├── preprocessing_flagged.csv
│   ├── preprocessing_summary.txt
│   ├── factory_summary.csv
│   ├── factory_filter_breakdown.csv
│   ├── validate_factory_{mode}.html        ← Phase 4b QA — visual audit of pseudo-masks
│   ├── validate_factory_all_modes.html     ← --all-modes: side-by-side mode_a/b/c/d
│   ├── symptom_vs_lab_comparison.csv      ← Phase 3b --compare-lab only
│   ├── gold_standard_iou_report.csv
│   ├── gold_standard_iou_summary.csv
│   ├── gold_standard_overlays/
│   ├── tier1_overlays/
│   ├── xai/
│   │   ├── gradcam/
│   │   ├── gradcamplusplus/
│   │   └── scorecam/
│   ├── charts/
│   │   ├── bouncer_comparison_bar.png
│   │   ├── bouncer_training_curves_{variant}.png
│   │   ├── bouncer_confusion_matrix_{variant}.png
│   │   ├── teacher_comparison_bar.png
│   │   ├── teacher_training_curves_{variant}.png
│   │   ├── student_stage1_comparison_bar.png
│   │   ├── student_stage2_comparison_bar.png
│   │   ├── student_training_curves_{enc}_{mode}.png
│   │   ├── student_confusion_{enc}_{mode}.png
│   │   ├── student_radar_{enc}_{mode}.png
│   │   ├── student_radar_all_overlay.png
│   │   └── student_metrics_heatmap.png
│   ├── severity_sample.csv                ← fill in rater scores here
│   ├── severity_analysis.csv
│   ├── severity_rating_guide.txt
│   ├── best_pipeline_summary.csv
│   ├── best_pipeline_summary.txt
│   ├── student_mobile_ranking.csv
│   ├── all_variants_ranked.csv
│   └── evaluation_report.html             ← self-contained, 11 sections
│
├── exports/
│   ├── tflite/
│   │   ├── student_model.tflite
│   │   └── export_report.csv
│   └── deploy/
│       ├── bouncer_model.tflite
│       ├── student_model.tflite
│       ├── model_metadata.json
│       ├── DEPLOYMENT_README.md
│       └── deployment_report.csv
│
├── scripts/                               ← Shared utilities
│   ├── __init__.py
│   ├── bouncer_inference.py               ← single source of truth for Bouncer inference
│   └── safe_collate.py                    ← DataLoader corruption guard
│
├── __init__.py
├── config.py                              ← all hyperparameters — single source of truth
├── image_utils.py                         ← EXIF correction, CLAHE, channel safety
├── requirements.txt
│
├── partition_dataset.py                   ← Step 1
├── create_bouncer_dataset.py              ← Step 2
├── train_bouncer.py                       ← Step 3
├── sample_15000.py                        ← Step 4
├── sample_gold_standard.py               ← Step 4a  (501 images, 167/class)
├── train_yolo_detector.py                ← Step 4b
├── generate_tier1_masks.py               ← Step 5   (SAM2 v3, YOLO-guided)
├── validate_masks.py                      ← Step 6
├── validate_gold_standard.py             ← Steps 6c / 6d / 10b
├── train_teacher.py                       ← Step 7
├── train_symptom_model.py                ← Step 7b  (HealthyAE + Symptom Teacher)
├── factory_master.py                      ← Step 8
├── validate_factory.py                    ← Step 8b  (visual QA of pseudo-masks)
├── train_student.py                       ← Steps 9 + 10
├── generate_charts.py                     ← any time after training
├── select_best_pipeline.py               ← Step 11
├── evaluate_xai.py                        ← Step 12
├── evaluate_severity.py                   ← Step 13
├── export_tflite.py                       ← Step 14
├── build_deployment_package.py           ← Step 15
├── generate_report.py                     ← Step 16
│
├── global_split_manifest.csv             ← auto: partition_dataset.py
├── tier1_manifest.csv                     ← auto: sample_15000.py
└── tier1_qa_report.csv                    ← auto: generate_tier1_masks.py
```

---

## Execution Order

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2

# Step 1 — run once, never re-run after training begins
python partition_dataset.py

# Step 2–3 — Bouncer
python create_bouncer_dataset.py
python train_bouncer.py
python generate_charts.py --bouncer      # optional

# Step 4 — Tier 1 sampling
python sample_15000.py

# Step 4a — Gold standard export (501 images, 167/class)
python sample_gold_standard.py
# → Label Studio: polygonlabels task, trace full leaf outline for all 501 images
# → Export JSON → data/gold_standard/annotations/annotations.json
# This single annotations.json is used by:
#   train_yolo_detector.py  (polygon → bbox for YOLO training)
#   validate_gold_standard.py (polygon → binary mask for IoU chain)
#   train_yolo_detector.py  (SAM2 QA confidence calibration)

# Step 4b — YOLO leaf detector (derives bboxes from gold standard polygons)
python train_yolo_detector.py            # requires annotations.json from Step 4a

# Step 5 — SAM2 masking
python generate_tier1_masks.py

# Step 6 — QA
python validate_masks.py
python validate_gold_standard.py --sam2-only    # target mean IoU ≥ 0.85

# Step 7 — Teacher
python train_teacher.py
python generate_charts.py --teacher      # optional
python validate_gold_standard.py         # SAM2 + Teacher chain

# Step 7b — Symptom Teacher (second annotation pass on same 501 gold standard images)
# → Label Studio: return to same project, trace symptom regions only
#   (chlorotic streaks for MSV, necrotic patches for MLN), label="symptom"
#   HEALTHY images: skip. Target ≥ 400 MSV+MLN images.
# → Export JSON → data/gold_standard/annotations/symptom_annotations.json
python train_symptom_model.py

# Step 8 — Factory pseudo-labels
python factory_master.py

# Step 8b — Visual QA of pseudo-masks (optional but recommended before training Student)
python validate_factory.py --all-modes   # → reports/validate_factory_all_modes.html
# python validate_factory.py --mode mode_b --n 20   # single-mode, more samples
# python validate_factory.py --borderline           # flag borderline severity cases

# Steps 9–10 — Student ablation
python train_student.py --stage 1
# → check logs/student_comparison_stage1.csv for best encoder
python train_student.py --stage 2 --encoder mobilenet_v2_cbam
python generate_charts.py --student      # optional
python validate_gold_standard.py         # full chain for thesis table

# Step 11 — Select best pipeline
python select_best_pipeline.py

# Steps 12–16 — Evaluation and export
python evaluate_xai.py
python evaluate_severity.py --sample     # raters fill in severity_sample.csv
python evaluate_severity.py --analyze
python export_tflite.py
python build_deployment_package.py
python generate_report.py
```

---

## Script Reference

| Script | Phase | Purpose |
|---|---|---|
| `partition_dataset.py` | 0-pre | 10-step image validation + 70/15/15 stratified global split |
| `create_bouncer_dataset.py` | 0a | Build 50k balanced Bouncer binary dataset |
| `train_bouncer.py` | 0b | Train 4 Bouncer variants + admission rate evaluation |
| `sample_15000.py` | 1 | Stratified 15k Tier 1 sampler (train+val only) |
| `sample_gold_standard.py` | 1b | Export 501 images (167/class) for polygon annotation — serves YOLO training, IoU validation, and SAM2 QA calibration |
| `train_yolo_detector.py` | 1c | Polygon → bbox conversion + train YOLOv8n + calibrate SAM2 QA threshold |
| `generate_tier1_masks.py` | 2 | SAM2 YOLO-guided auto-masking + QA filters [v3] |
| `validate_masks.py` | 2b | QA stats + qualitative overlays |
| `validate_gold_standard.py` | 2c / 2d / 10b | IoU chain: SAM2 → Teacher → Student vs human masks |
| `train_teacher.py` | 3 | Train 4 leaf-silhouette segmentation variants on SAM2 soft targets |
| `train_symptom_model.py` | 3b | Train HealthyAE + Symptom Teacher on human symptom masks |
| `factory_master.py` | 4 | Generate pseudo-labels for ~215k Tier 2 images across 4 modes |
| `validate_factory.py` | 4b | Visual HTML audit of pseudo-masks vs raw images, per class, with auto-flagged outliers |
| `train_student.py` | 5a / 5b | Two-stage Student ablation (encoder × mode) |
| `generate_charts.py` | any | Reads logs/ CSVs → reports/charts/ PNGs |
| `select_best_pipeline.py` | 6 | Select best variants, promote to final/, auto-update config.py |
| `evaluate_xai.py` | 7 | 3-way XAI comparison (Grad-CAM / Grad-CAM++ / Score-CAM) |
| `evaluate_severity.py` | 8 | Inter-rater severity reliability (Cohen's Kappa + Spearman ρ) |
| `export_tflite.py` | 9 | Student → ONNX → TFLite (FP16) |
| `build_deployment_package.py` | 10 | Full Android deployment bundle |
| `generate_report.py` | 11 | Self-contained HTML evaluation report (11 sections) |
| `config.py` | — | All hyperparameters — single source of truth |
| `image_utils.py` | — | EXIF correction, CLAHE, channel safety |
| `scripts/bouncer_inference.py` | — | Shared Bouncer inference helpers |
| `scripts/safe_collate.py` | — | DataLoader corruption guard |

---

## Dataset Sources

| Dataset | Purpose | Location | Citation |
|---|---|---|---|
| Maize Imagery Dataset — Tanzania (Mduma 2023) | HEALTHY / MSV / MLN labelled leaves | `maize_dataset/` | Mendeley Data |
| Zenodo healthy + MSV samples | Additional maize images (merge into above) | `maize_dataset/` | Zenodo |
| Intel Image Classification | Non-maize scene negatives | `dataset/raw_kaggle/intel/` | Kaggle: puneet6060 |
| Natural Images | Non-maize object negatives | `dataset/raw_kaggle/natural/` | Kaggle: prasunroy |
| PlantVillage | Rice + sorghum leaf negatives | `dataset/crop_neighbors/rice/`, `sorghum/` | Hughes & Salathé 2015 |
| iNaturalist Philippines | Cogon grass, banana leaf, sugarcane negatives | `dataset/crop_neighbors/` | iNaturalist API |
| Mendeley Maize-Weed | Field weed negatives | `dataset/crop_neighbors/` | Espejo-Garcia et al. 2020 |
| SAM2 weights | Tier 1 silhouette masking | `sam2/sam2_hiera_large.pt` | Meta AI (download separately) |

**Approximate maize class distribution after preprocessing:**

| Class | Images | % |
|---|---|---|
| HEALTHY | ~96,000 | 38% |
| MLN | ~96,000 | 38% |
| MSV | ~60,000 | 24% |
| **TOTAL** | **~252,000** | 100% |

**Global split:**

| Split | Ratio | Purpose |
|---|---|---|
| train | 70% | Weight updates, augmentation, backpropagation |
| val | 15% | Early stopping, LR scheduling, checkpoint selection |
| test | 15% | Final evaluation only — never used in any training decision |

Test-split images are never included in Tier 1 sampling, Bouncer positives, Teacher training, or Factory processing.

---

## Annotation Summary

| Task | Script | # Images | Per class | Destination |
|---|---|---|---|---|
| Leaf silhouette polygons | `sample_gold_standard.py` | 501 | 167 | `data/gold_standard/annotations/annotations.json` |
| Symptom region polygons | *(second pass, same 501 images)* | ≥ 400 MSV+MLN | — | `data/gold_standard/annotations/symptom_annotations.json` |

**One `annotations.json` — three consumers:**
- `train_yolo_detector.py` — reads polygons, derives bounding boxes, trains YOLOv8n
- `validate_gold_standard.py` — reads polygons, rasterizes to binary masks, computes IoU chain
- `train_yolo_detector.py` (calibration step) — runs YOLO+SAM2 on gold images to calibrate QA confidence threshold

---

## Key Rules

- **Never re-run `partition_dataset.py`** after any training has started. It regenerates `global_split_manifest.csv` and silently invalidates all downstream splits, masks, and annotations.
- **Never delete `_manifest_hash.txt`** in `data/tier1_raw/` or `data/gold_standard/images/` while annotation or masking is in progress. These SHA-256 locks abort the sampler instead of silently producing a different image set.
- **Stage 1 student checkpoints go to `checkpoints/student/stage1/`** to prevent the Stage 2 Mode B run from overwriting them (same encoder + mode name collision).
- **All image loading must go through `image_utils.py`** — never call `cv2.imread()` or `PIL.Image.open()` directly. This guarantees consistent EXIF orientation and RGB channel order everywhere.
- **All DataLoaders must use `collate_fn=safe_collate`** — if `__getitem__` returns `None` (corrupt image), the batch is filtered gracefully instead of crashing the worker.
- **`train_symptom_model.py` must run before `factory_master.py`** when `SYMPTOM_TEACHER_DEPLOYED = True` in `config.py`. If the checkpoints are missing, factory falls back to the legacy LAB pipeline automatically.
- **Factory outputs include `_grade.txt`** (CIMMYT severity grade integer) alongside `_sev.txt` (continuous severity %) for every processed image.
- **`sample_yolo_annotations.py` is deprecated/unused.** YOLO training now uses the same `annotations.json` produced by `sample_gold_standard.py`; the `data/yolo_annotations/` directory is no longer used, and `YOLO_IMAGES_DIR`, `YOLO_ANNOTATIONS_DIR`, and `YOLO_ANNOTATION_FILE` in `config.py` all point to `data/gold_standard/` paths. The file itself may still be present on disk as legacy cruft — it is safe to delete and is not part of the execution order above.
- **Run `validate_factory.py` before `train_student.py`.** It is not strictly required by any downstream script, but catching a bad pseudo-mask pattern (e.g. MSV/MLN masks looking identical, or HEALTHY masks with high fill) here is far cheaper than discovering it after a full Student training run.
