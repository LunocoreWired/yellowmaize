# Yellow MAIze — Complete Project Summary
## Context document for continuing work in a new chat

---

## 1. What This Project Is

**Thesis title:** MAIze: A U-Net with MobileNetV2 and Explainable AI Framework for Maize Streak Virus Identification and Symptom Segmentation in Zea mays L.

**Institution:** Angeles University Foundation, College of Computer Studies, BSCS 3-A

**Team:** Altes, Zylah Klein · Davis, Dominic · Tayer, Catherine P. · Ursua, Walter Vince

**Clinical problem:** MSV (Maize Streak Virus) was first confirmed in the Philippines in 2023 (Bukidnon, South Cotabato — Cruz et al. 2024). Early symptoms are visually indistinguishable from nutrient deficiencies. Tool targets agriculture students and smallholder farmers.

**Deployment target:** Android (TFLite). Two models run in sequence on-device: a Bouncer gate that rejects non-maize images, then a Student that classifies disease and segments the affected leaf area.

**Dataset:** Maize Imagery Dataset — Tanzania (Mduma 2023, Mendeley) + Zenodo healthy/MSV samples. ~252,000 images: HEALTHY (~38%), MLN (~38%), MSV (~24%). Geographic mismatch (Tanzania data, Philippine deployment) is a stated limitation.

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

**Albumentations note:** Version 2.0+ API uses `fill` and `fill_mask` instead of `value` in `PadIfNeeded`, and `num_holes_range` / `hole_height_range` / `hole_width_range` instead of `max_holes` etc. in `CoarseDropout`. All scripts have been updated to use the 2.0 API.

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
| 4 | `factory_master.py` | `data/pseudo_masks/{mode}/` — pseudo-labels for ~215k images |
| 4b | `validate_factory.py` | `reports/validate_factory_{mode}.html` — visual QA (optional) |
| 5a | `train_student.py --stage 1` | 5 encoder variants × Mode B → `checkpoints/student/stage1/` |
| 5b | `train_student.py --stage 2` | Best encoder × 4 modes → `checkpoints/student/` |
| 5c | `validate_gold_standard.py` | Full SAM2 → Teacher → Student chain (thesis table) |
| 6 | `select_best_pipeline.py` | `checkpoints/final/` + auto-updates `config.py` |
| 7 | `evaluate_xai.py` | 3-way XAI results + overlays |
| 8 | `evaluate_severity.py` | Inter-rater reliability |
| 9 | `export_tflite.py` | `exports/tflite/student_model.tflite` |
| 10 | `build_deployment_package.py` | `exports/deploy/` — full Android bundle |
| 11 | `generate_report.py` | `reports/evaluation_report.html` |
| any | `generate_charts.py` | `reports/charts/*.png` |
| optional | `validate_symptom.py` | Figures 4.16-4.18 + Table 4.13 |
| optional | `validate_student.py` | Qualitative Student prediction panels |

---

## 4. Component Specifications

### 4.1 Preprocessing — `partition_dataset.py`
Run once before everything else. Never re-run after training begins. Ten validation steps: zero-byte, magic bytes, truncation, min resolution (64px), max resolution (4096px), extreme aspect ratio (>8.0), colour mode, near-uniform (std < 5.0), MD5 exact dedup, pHash near-dedup (Hamming ≤ 2). Applies 70/15/15 stratified split. Outputs: `global_split_manifest.csv`, `preprocessing_report.csv`, `preprocessing_flagged.csv`, `preprocessing_summary.txt`. Rejected images are moved (not deleted) to `quarantine/<reason>/<class>/`.

### 4.2 Image Utilities — `image_utils.py`
Single source of truth for all image loading. No other script calls `cv2.imread()` or `PIL.Image.open()` directly.

- `load_image_rgb(path)` — PIL + `ImageOps.exif_transpose()` + truncation guard → uint8 RGB or None
- `load_image_clahe(path)` — above + CLAHE on L channel in LAB (clipLimit=2.0, tileGridSize=(8,8)). Used by Bouncer and Factory. NOT used by Teacher or Student training.
- `apply_clahe(img_rgb)` — standalone CLAHE; returns uint8 RGB
- `to_hsv(img_rgb)` — guaranteed RGB→HSV
- `rgb_to_bgr(img_rgb)` — for `cv2.imwrite()` only

### 4.3 Shared Infrastructure — `scripts/`

**`scripts/safe_collate.py`** — Every DataLoader uses `collate_fn=safe_collate`. Filters None `__getitem__` returns. Entire-None batch → returns None; training loop skips with `continue`. `reset_skip_counter()` at epoch start, `get_skip_count()` at epoch end.

**`scripts/bouncer_inference.py`** — Single source of truth for Bouncer inference, shared by `train_bouncer.py` and `factory_master.py`. Contains `BOUNCER_INFER_TF` (LongestMaxSize + PadIfNeeded using Albumentations 2.0 `fill`/`fill_mask` API + ImageNet normalize), `heuristic_prefilter()` (always True — passthrough), and `neural_bouncer()` (`@torch.no_grad()`, returns `sigmoid(logit) >= threshold`). No heavy imports.

### 4.4 Bouncer — `train_bouncer.py` (Phase 0b)
Binary maize-vs-not-maize gate. Dataset: 25,000 maize (train+val only) + 25,000 non-maize. Image loading: `load_image_clahe()`.

**4 variants:**

| Variant | Architecture | Deployed? |
|---|---|---|
| gabor_lbp | Gabor + LBP + LinearSVC | No — traditional CV baseline |
| mobilenet_v2 | MobileNetV2 head: Linear(1280→1) | No — neural comparison |
| **mobilenet_v3_large** | MobileNetV3-Large head: Linear(960→1) | **Yes** |
| edgevit_xxs | EdgeViT-XXS binary classifier | No — hybrid ViT |

Neural config: BCEWithLogitsLoss · AdamW(lr=1e-4, wd=1e-4) · CosineAnnealingLR(T_max=15) · 15 epochs · batch 64 · patience=5 on val F1 · grad clip 5.0 · safe_collate.

Threshold: geometric mean maximiser of recall × specificity, maize recall ≥ 0.95, hard cap 0.70.

### 4.5 Tier 1 Sampling — `sample_15000.py` (Phase 1)
15,000 images from train+val only. HEALTHY: 3,000 random; MSV: 7,500 evenly-spaced; MLN: 4,500 evenly-spaced. SHA-256 hash guard on `global_split_manifest.csv`.

### 4.6 Gold Standard Export — `sample_gold_standard.py` (Phase 1b)
Exports **501 images (167/class)** from Tier 1. Files renamed `{CLASS}_{original}.jpg`. SHA-256 guard on `tier1_manifest.csv`. **One export, three consumers:**

1. `train_yolo_detector.py` — polygon→bbox conversion for YOLO training
2. `validate_gold_standard.py` — polygon→binary mask for IoU chain
3. `train_yolo_detector.py` (calibration) — YOLO+SAM2 on gold images to calibrate QA threshold

**Two annotation passes, same 501 images:**
- Pass 1 (Label Studio, polygonlabels): Trace full leaf silhouette → `annotations.json`
- Pass 2 (CVAT, COCO 1.0): Trace symptom regions (`maize-leaf`, `msv-symptom`, `mln-symptom`) → `symptom_annotations.json`. HEALTHY images get only `maize-leaf`. Target ≥ 400 MSV+MLN images.

> `sample_yolo_annotations.py` is deprecated and unused. `YOLO_IMAGES_DIR`, `YOLO_ANNOTATIONS_DIR`, `YOLO_ANNOTATION_FILE` in `config.py` all point to `data/gold_standard/` paths.

### 4.7 YOLO Leaf Detector — `train_yolo_detector.py` (Phase 1c)
Trains YOLOv8n (single-class) by deriving bboxes from gold standard polygon annotations (min/max of polygon x,y). Validation metrics (mAP@0.5, mAP@0.5:0.95, precision, recall) written to `logs/yolo_val_metrics.csv`. After training, calibrates SAM2 QA confidence threshold against gold-standard IoU (target mean IoU ≥ 0.85): sweeps thresholds 0.50–0.95, picks the lowest threshold that reaches the target, writes the sweep curve to `logs/yolo_qa_calibration.csv` and the raw per-image (mean_conf, IoU) points to `logs/yolo_calib_raw.csv`. Must run before `generate_tier1_masks.py`. Warns if annotation count < 400 or if mAP@0.5 < 0.70.

### 4.8 SAM2 Masking — `generate_tier1_masks.py` (Phase 2, v3)
YOLO box → restrict HSV tissue search to box interior → 3 foreground points along vertical axis → SAM2 box prompt. HSV fallback if YOLO absent. Four corner background prompts always. QA filters: min coverage 2%, high-coverage gate (≥ 97% requires ≥ 0.80 confidence), calibrated min confidence (default 0.65), aspect ratio ≥ 1.00. Target < 8% rejection.

### 4.9 Teacher — `train_teacher.py` (Phase 3)
Offline segmentation model, never deployed. Trains on SAM2 float32 probability maps using `BoundaryAwareLoss` (Dice + boundary-weighted BCE with ×5.0 edge upweighting + ×1.3 soft target sharpening). Uses Albumentations 2.0 API (`fill`/`fill_mask`, `num_holes_range`). ElasticTransform + CoarseDropout augmentations added to combat overfitting at margins.

**4 variants:** resnet50 · **efficientnet-b2** (deployed) · mit_b2 · deeplabv3plus-eb2. Input 768×768 · AdamW(lr=5e-5, wd=5e-4) · ReduceLROnPlateau · patience=7 · grad clip 5.0 · safe_collate · decoder_dropout=0.2.

**Key training note:** Val Dice ~0.97 on SAM2-mask validation set, but ~0.72 IoU against human gold standard annotations. The gap reflects SAM2 mask style vs human polygon style — not underfitting. More epochs (beyond 30) do not close this gap.

Also writes `logs/teacher_test_metrics.csv` (held-out test Dice/IoU/Recall/Precision/Specificity) and qualitative overlays to `reports/teacher_overlays/` (5 images/class from the test split). Note: `reports/teacher_overlays/` is a separate, earlier qualitative check — the overlays actually embedded in `evaluation_report.html` come from `validate_gold_standard.py`'s `reports/gold_standard_overlays/` (SAM2/Teacher/Student vs. human ground truth), not from this folder.

### 4.10 Symptom Teacher — `train_symptom_model.py` (Phase 3b)
Replaces the LAB/HSV colour-threshold symptom pipeline. Eight rounds of threshold tuning (v1→v8) confirmed a structural ceiling — static colour rules cannot separate early chlorosis from healthy yellow-maize tissue. A model trained on ~400 human masks learns the decision boundary directly.

**HealthyAE:** Conv autoencoder with true bottleneck (no skip connections), trained on HEALTHY images only. At inference, per-pixel MSE reconstruction error normalized to [0,1] becomes the 4th input channel. Input 256×256 · batch 16 · 40 epochs · Adam(lr=1e-3) · patience=8.

**Symptom Teacher:** `smp.Unet(encoder_name="efficientnet-b2", in_channels=4)`. Input = RGB + AE error map. Ground truth = human polygon/brush masks. Input 512×512 · batch 4 · 60 epochs · AdamW(lr=1e-4) · 0.6×Dice + 0.4×Focal loss · patience=10 · target IoU ≥ 0.70.

`SYMPTOM_ENCODER` is derived from `TEACHER_DEPLOYED_VARIANT` in `config.py` — never hardcoded.

If the 501 gold-standard images don't yield ≥ 400 MSV+MLN symptom annotations, extra images can be annotated and placed in `data/symptom_extra/images/` (`SYMPTOM_EXTRA_IMAGES_DIR`) — annotate them in the same CVAT task and export to the same `symptom_annotations.json`; `parse_symptom_annotations()` searches `[GOLD_IMAGES_DIR, SYMPTOM_EXTRA_IMAGES_DIR]` and silently skips whichever doesn't exist yet.

Config toggles: `SYMPTOM_TEACHER_DEPLOYED = True` enables Symptom Teacher in Factory. `FACTORY_SYMPTOM_COMPARE_LAB = False` (set True for thesis comparison). LAB fallback activates automatically if checkpoints are missing.

### 4.11 Factory — `factory_master.py` (Phase 4)
Processes ~215k train+val Tier 2 images across 4 modes simultaneously. Per-image pipeline: Bouncer gate → leaf silhouette (SAM2 .npy for Tier 1, Teacher inference for Tier 2) → refinement (threshold 0.35, morph close+open) → coverage guard → reliability weight → symptom masking (Symptom Teacher if available, else LAB fallback) → severity % → CIMMYT grade.

**4 modes:**

| Mode | Silhouette | Symptom | Subfolder |
|---|---|---|---|
| A | Otsu | Hard binary | `mode_a/` |
| B | SAM2 binary | Hard binary | `mode_b/` |
| C | SAM2 soft float | Hard binary | `mode_c/` |
| D | SAM2 soft float | Soft HSV confidence | `mode_d/` |

Per-image outputs per mode: `{stem}_silhouette.npy`, `{stem}_symptom.png` or `.npy`, `{stem}_sev.txt`, `{stem}_grade.txt` (CIMMYT grade), `{stem}_weight.txt`. All downscaled to 224×224 at write time. A single unified per-image report across all 4 modes is also written to `reports/phase4_report.csv`, alongside the summary/breakdown CSVs.

CIMMYT grading: MSV uses a 1-5 Soto/IITA leaf-area scale (Soto et al. 1982, validated Sime et al. 2021): 1=≤10%, 2=11-25%, 3=26-50%, 4=51-75%, 5=≥75%. MLN uses a 1-5 scale (Beyene et al. 2017; Gowda et al. 2015): 1=<10%, 2=10-25%, 3=25-50%, 4=50-75%, 5=>75%. (A published CIMMYT 1-9 odd-numbered scale exists for MSV but is for whole-plant breeder resistance scoring, not leaf-area percentage mapping — `sev_to_cimmyt_grade()` in `factory_master.py` deliberately uses the 0-5 Soto/IITA scale instead, structurally identical to the MLN scale.)

### 4.11b Factory QA — `validate_factory.py` (Phase 4b)
Optional but recommended before committing to a full Student run. Samples N images/class, produces HTML report with raw image vs pseudo-mask overlay. CLI: `--n`, `--mode`, `--all-modes`, `--img`, `--borderline`, `--out`. Docstring maps each visual failure pattern to the exact config.py parameter to tune.

### 4.12 Student — `train_student.py` (Phase 5)
The deployed model. Shared encoder → three heads: segmentation (Ch0=silhouette, Ch1=symptom), classification (HEALTHY/MSV/MLN), severity (ReLU+clamp → ×100).

**5 encoder variants:** V1 mobilenet_v2 · **V2 mobilenet_v2_cbam** (expected winner) · V3 mobilenet_v3_small · V4 efficientnet_b0 · V6 efficientnet_b0_cbam. V5 (MobileViT-XXS) removed — TFLite einsum incompatibility.

**CBAM:** Proper `smp.UnetDecoder` subclass — not monkey-patching. Channel attention (GAP+GMP → shared MLP → sigmoid) + spatial attention (7×7 conv → sigmoid) at each UNet skip connection. Fully serializable with `torch.save()`.

**Loss:** DiceLoss (float32 soft targets) + asymmetric label-smoothing cross-entropy (HEALTHY→[0.90,0.08,0.02], MSV→[0.05,0.90,0.05], MLN→[0.02,0.05,0.93]) + reliability-weighted MSE for severity + homoscedastic uncertainty loss with 3 learnable log-variance params s1/s2/s3 (Kendall et al. 2018).

**Checkpoint criterion (quality composite — during training):**
`0.40×mIoU + 0.15×MLN_F1 + 0.10×(1−NormMAE) + 0.20×MSV_F1_early + 0.10×MSV_F1_mid + 0.05×MSV_F1_severe`

The severity-stratified MSV F1 terms (aligned with CIMMYT brackets) replaced the aggregate MSV_F1 to surface early-stage detection — the clinically critical case.

**Two-phase training:** Phase 1 frozen encoder (30 epochs, lr=1e-3, patience=10) → Phase 2 full fine-tune (20 epochs, lr=1e-4, patience=5). Optimizer re-initialized at transition. Grad clip 5.0 throughout. WeightedRandomSampler for class balance.

**Two-stage ablation:** Stage 1 = 5 encoders × Mode B → `checkpoints/student/stage1/`. Stage 2 = best encoder × 4 modes → `checkpoints/student/`. Stage 1 Mode B reused for Stage 2. Total unique runs: 8.

**Mobile composite (select_best_pipeline.py):** `0.32×MSV_F1 + 0.18×MSV_ROC_AUC + 0.18×sil_mIoU + 0.16×clamp(150ms/lat,0,1) + 0.08×MLN_F1 + 0.05×(1−sev_mae/100) + 0.03×clamp(15MB/size,0,1)`. Size score defaults to 1.0 (optimistic) if TFLite size is not yet known.

### 4.13 Qualitative Validation Scripts

**`validate_student.py`** — Loads deployed Student checkpoint, runs on gold-standard images (5/class by default), produces 3-panel overlays: Original | Predicted silhouette | Predicted symptom mask. Reports qualitative sample accuracy. Not a substitute for formal test metrics in `student_test_metrics_*.csv`.

**`validate_symptom.py`** — Produces Figures 4.16-4.18 and Table 4.13: HealthyAE reconstruction/anomaly map, human vs predicted symptom mask, Symptom Teacher vs legacy LAB comparison. Reuses inference functions from `train_symptom_model.py` directly.

### 4.14 Gold Standard Validation — `validate_gold_standard.py`
Validates pseudo-label quality against 501 human-annotated leaf silhouettes across the full chain: SAM2 → Teacher → Student IoU vs human. Parser: Label Studio JSON polygonlabels → `cv2.fillPoly()` → binary mask. Warn threshold: IoU < 0.75/image. Target: mean IoU ≥ 0.85. Run three times (--sam2-only after Phase 2, full after Phase 3, full after Phase 5b).

### 4.15 XAI — `evaluate_xai.py`
Three methods (pytorch-grad-cam) on last encoder conv block: Grad-CAM (baseline), **Grad-CAM++** (deployed — multi-region MSV streaks), Score-CAM (gradient-free reference). Metrics: pointing game accuracy, Insertion AUC, Deletion AUC (n_steps=25, GPU). Also auto-selects the empirically best method (highest MSV pointing-game accuracy, tiebreak Insertion AUC) and writes it to `logs/xai_method_selection.csv` — an evidence-based check against the config.py-hardcoded `XAI_DEPLOYED_METHOD`.

**Two distinct app outputs — never conflate:** Symptom boundary (green contour) — UNet Ch1, pixel-level. Diagnostic attention (amber heatmap) — Grad-CAM++, coarse ~7×7 upsampled.

### 4.16 Best Pipeline Selection — `select_best_pipeline.py`
Bouncer: max specificity (maize recall ≥ 0.95). Teacher: max val Dice. Student: quality composite during training, mobile composite for deployment. Promotes to `checkpoints/final/`. Auto-updates `STUDENT_BEST_VARIANT` and `STUDENT_FACTORY_MODE` in `config.py` via regex.

### 4.17 Severity Reliability — `evaluate_severity.py`
60 images (20/class), 2 raters, 0–3 scale. Cohen's Kappa + Spearman ρ (vs pseudo-label severity). Target ρ ≥ 0.60. Severity MAE measures consistency with pseudo-labels, not expert agronomic ratings — stated limitation in Chapter 4. `--sample` writes `reports/severity_sample.csv` (blank rating column for raters to fill) and `reports/severity_rating_guide.txt` (rater instructions); `--analyze` writes `reports/severity_analysis.csv` and `logs/severity_reliability.csv`.

### 4.18 Export and Deployment
PyTorch → ONNX (opset 12) → TF SavedModel → TFLite (FP16). Student I/O: input [1,3,224,224] NCHW; output 0 seg logits [1,2,224,224]; output 1 cls logits [1,3]; output 2 severity [1,1] ∈ [0,1] → ×100. Bouncer I/O: input [1,3,224,224]; output [1,1] logit → sigmoid ≥ threshold → pass.

---

## 5. Design Decisions

| Decision | Rationale |
|---|---|
| Symptom Teacher replaces LAB/HSV | 8 rounds of LAB tuning (v1→v8) confirmed a structural ceiling. Human-supervised model learns the boundary directly. |
| HealthyAE as 4th input channel | Lighting-invariant anomaly prior. AE output is an input feature, not a label source. |
| BoundaryAwareLoss for Teacher | Addresses systematic undersegmentation at leaf margins observed in IoU chain validation (SAM2 0.94 → Teacher 0.72). |
| Single gold standard export | `train_yolo_detector.py` derives bboxes from polygons automatically. One annotation task, one `annotations.json`, three consumers. `sample_yolo_annotations.py` deprecated. |
| CIMMYT severity grading | Maps continuous severity % to published agronomic scale (MSV and MLN both use a 1-5 leaf-area scale — MSV: Soto/IITA per Soto et al. 1982; MLN: Beyene et al. 2017). Provides defensible ground for severity thresholds. |
| Severity-stratified MSV F1 checkpoint | Aggregate MSV F1 sacrificed early-stage detection (grade 1-3). CIMMYT-stratified weights surface the clinically critical case. |
| Asymmetric label smoothing | Early MSV visually identical to HEALTHY (Cruz et al. 2024). Pathology-informed prior prevents overconfidence. |
| Homoscedastic uncertainty loss | Three learnable log-variance params auto-balance three task losses (Kendall et al. 2018). Replaces manual 0.6/0.2/0.2. |
| ReLU+clamp severity head | Sigmoid never reaches exactly 0.0 — HEALTHY leaves always show nonzero severity. ReLU allows true 0.0. |
| Global mIoU not mean-of-batches | Per-batch averaging is wrong when batch class distributions vary. |
| CBAM as proper subclass | Monkey-patching breaks `torch.save()` serialization. |
| Stage 1 checkpoint isolation | Prevents Stage 2 Mode B from overwriting Stage 1 checkpoint for same encoder+mode. |
| Two-composite selection | Quality composite during training (TFLite size/latency unknown). Mobile composite post-training for deployment. |
| Albumentations 2.0 API | `fill`/`fill_mask` replaces `value` in PadIfNeeded. `num_holes_range` etc. replaces `max_holes` in CoarseDropout. All scripts updated. |

---

## 6. Terminology Corrections

| Wrong | Correct |
|---|---|
| "Knowledge distillation" | "Pseudo-label semi-supervised learning" or "teacher-guided pseudo-label generation" |
| "Pixel-level segmentation via Grad-CAM" | Grad-CAM++ = coarse ~7×7 class-discriminative heatmap. UNet head = pixel-level segmentation. Entirely distinct outputs. |
| "Severity MAE measures disease severity accuracy" | "Severity MAE measures consistency with Factory's pseudo-labels (Symptom Teacher), not expert agronomic ratings" |

---

## 7. File Inventory

```
yellowmaize/
├── config.py                     All hyperparameters — YOLO_* paths point to gold_standard/
├── image_utils.py
├── requirements.txt
├── __init__.py
├── partition_dataset.py
├── create_bouncer_dataset.py
├── train_bouncer.py
├── sample_15000.py
├── sample_gold_standard.py       501 images — serves YOLO + IoU + SAM2 QA (3 consumers)
├── train_yolo_detector.py        Polygon→bbox + YOLOv8n + QA calibration
├── generate_tier1_masks.py       SAM2 v3 YOLO-guided
├── validate_masks.py
├── validate_gold_standard.py     Run 3× at Steps 6c / 6d / 10b
├── train_teacher.py              BoundaryAwareLoss + 768px + decoder_dropout
├── train_symptom_model.py        HealthyAE + Symptom Teacher
├── factory_master.py             4 modes + CIMMYT grading + Symptom Teacher integration
├── validate_factory.py           HTML visual QA (Phase 4b, optional)
├── train_student.py              CIMMYT-stratified MSV F1 + updated composite
├── validate_student.py           Qualitative Student prediction panels (optional)
├── validate_symptom.py           Symptom Teacher vs LAB figures (optional)
├── generate_charts.py
├── select_best_pipeline.py
├── evaluate_xai.py
├── evaluate_severity.py
├── export_tflite.py
├── build_deployment_package.py
├── generate_report.py
└── scripts/
    ├── __init__.py
    ├── bouncer_inference.py      Albumentations 2.0 API applied
    └── safe_collate.py
```

> `sample_yolo_annotations.py` is deprecated and unused.

---

## 8. Datasets to Download

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

## 9. Limitations for Chapter 5

1. Geographic domain gap — Tanzania data, Philippine deployment.
2. Severity ground truth — Symptom Teacher pseudo-labels, not expert CIMMYT-protocol ratings.
3. Staged ablation — not fully crossed encoder × mode factorial (compute budget). Stated in Chapters 3 and 5.
4. Tier 1 size — 15k Teacher training images relative to ~215k Factory images.
5. Android only — iOS excluded.
6. No Philippine field images — all from Tanzania + Zenodo.
7. MobileViT-XXS excluded — TFLite einsum incompatibility.
8. Symptom Teacher annotation burden — ≥ 400 MSV+MLN symptom annotations required. Non-agronomist annotator quality is a potential bias source.

---

## 10. Key Citations

| Reference | Used for |
|---|---|
| Kendall et al. (2018) NeurIPS | Homoscedastic uncertainty multi-task loss |
| Szegedy et al. (2016) CVPR | Label smoothing |
| Woo et al. (2018) ECCV | CBAM |
| Ke et al. (2020) | Soft pseudo-label segmentation targets |
| Jiang et al. (2018) ICML | MentorNet — reliability-weighted curriculum |
| Cruz et al. (2024) | First confirmed MSV in the Philippines |
| Mushayi et al. (2025) | MSV / HEALTHY confusion — asymmetric prior matrix basis |
| Soto et al. (1982); Sime et al. (2021) Agriculture 11(2):130 | MSV 1-5 severity scale (CIMMYT_MSV_BRACKETS) |
| Beyene et al. (2017) Euphytica 213:224; Gowda et al. (2015); Eunice et al. (2021) Cogent Food & Agriculture 7(1) | MLN 1-5 severity scale (CIMMYT_MLN_BRACKETS) |
| Selvaraju et al. (2017) ICCV | Grad-CAM |
| Chattopadhay et al. (2018) WACV | Grad-CAM++ |
| Wang et al. (2020) CVPR | Score-CAM |
| Mduma (2023) Mendeley | Tanzania Maize Imagery Dataset |
| Tan & Le (2019) ICML | EfficientNet |
| Xie et al. (2021) NeurIPS | SegFormer (mit_b2) |
| Pan et al. (2022) ECCV | EdgeViT |
| Lin et al. (2017) ICCV | Focal loss (Symptom Teacher) |
| Fawcett (2006) | ROC threshold selection |
| Zuiderveld (1994) | CLAHE |
