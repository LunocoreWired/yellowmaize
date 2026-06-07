"""
================================================================================
 config.py — Yellow MAIze Central Configuration
================================================================================
 Single source of truth for ALL hyperparameters, paths, and settings.
 Every training script imports from here. Change a value once, it propagates
 everywhere. Do NOT hardcode values in individual scripts.
================================================================================
"""

import os
from pathlib import Path
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL SEED — enforced in every training script for reproducibility
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# ROOT PATHS
# ══════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).parent
MAIZE_DIR = ROOT / "maize_dataset"
DATASET_DIR = ROOT / "dataset"
SAM2_CHECKPOINT = ROOT / "sam2" / "sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"  # relative to sam2 repo

DATA_DIR = ROOT / "data"
TIER1_RAW_DIR = DATA_DIR / "tier1_raw"
TIER1_MASKS_DIR = DATA_DIR / "tier1_leaf_masks"
PSEUDO_DIR = DATA_DIR / "pseudo_masks"
BOUNCER_DATASET_DIR = DATA_DIR / "bouncer_dataset"

CHECKPOINTS_DIR = ROOT / "checkpoints"
BOUNCER_CKPT_DIR = CHECKPOINTS_DIR / "bouncer"
TEACHER_CKPT_DIR = CHECKPOINTS_DIR / "teacher"
STUDENT_CKPT_DIR = CHECKPOINTS_DIR / "student"

LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"
EXPORTS_DIR = ROOT / "exports" / "tflite"

GLOBAL_MANIFEST = ROOT / "global_split_manifest.csv"
TIER1_MANIFEST = ROOT / "tier1_manifest.csv"
TIER1_QA_REPORT = ROOT / "tier1_qa_report.csv"

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
PREPROC_MIN_DIMENSION = 64  # px — images smaller than this are rejected
PREPROC_MAX_DIMENSION = 4096  # px — images larger than this are rejected
PREPROC_MAX_ASPECT_RATIO = 8.0  # max(w,h)/min(w,h) beyond this → rejected
PREPROC_UNIFORM_STD_THRESH = 5.0  # pixel std below this → near-uniform → rejected
PREPROC_GREEN_FLAG_THRESH = 0.05  # < 5% green pixels → flagged for review
PREPROC_PHASH_HAMMING = 2  # Hamming distance ≤ this → near-duplicate
PREPROC_MIN_FILE_BYTES = 100  # files smaller than this → rejected

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
CLASSES = ["HEALTHY", "MSV", "MLN"]
CLASS_TO_IDX = {"HEALTHY": 0, "MSV": 1, "MLN": 2}
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# 70 / 15 / 15 split
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Tier 1 composition
TIER1_PER_CLASS = {"HEALTHY": 3000, "MSV": 7500, "MLN": 4500}  # total 15k

# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH REPRODUCIBILITY (set in every training script via set_seeds())
# ══════════════════════════════════════════════════════════════════════════════
# deterministic=True, benchmark=False chosen deliberately for reproducibility.
# benchmark=True would give ~15% faster training on RTX 5060 but non-deterministic
# results. For a thesis where exact reproducibility matters, keep these as-is.
CUDNN_DETERMINISTIC = True
CUDNN_BENCHMARK = False

# ══════════════════════════════════════════════════════════════════════════════
# BOUNCER
# ══════════════════════════════════════════════════════════════════════════════
BOUNCER_TARGET_PER_CLASS = 25_000  # 25k maize + 25k non-maize = 50k
BOUNCER_IMG_SIZE = 224
BOUNCER_BATCH_SIZE = 40
BOUNCER_EPOCHS = 15
BOUNCER_LR = 1.25e-4
BOUNCER_WEIGHT_DECAY = 1e-4
BOUNCER_VAL_SPLIT = 0.20  # internal 80/20
BOUNCER_PATIENCE = 5

# Heuristic pre-filter thresholds
BOUNCER_MIN_ASPECT_RATIO = 1.5  # reject if < 1.5 (too round)
BOUNCER_MAX_ASPECT_RATIO = 18.0  # reject if > 18.0 (too thin)
BOUNCER_MIN_GREEN_COVERAGE = 0.15  # reject if green < 15% of frame

# Neural gate sigmoid threshold (will be tuned empirically on val ROC)
# This default is a starting point — replaced by empirical value after training
BOUNCER_THRESHOLD = 0.65

# Bouncer variants to train
# NOTE: patchcore removed — anomaly detection is architecturally mismatched
# for supervised binary classification and has no TFLite deployment path.
# gabor_lbp is sufficient as the single non-neural baseline.
BOUNCER_VARIANTS = [
    "gabor_lbp",  # traditional CV baseline (no training)
    "mobilenet_v2",  # added — thesis title names MobileNetV2 as primary arch
    "mobilenet_v3_large",  # deployed model
    "edgevit_xxs",  # hybrid ViT comparison
]
BOUNCER_DEPLOYED_VARIANT = "mobilenet_v3_large"

# Non-maize source subdirectories
NON_MAIZE_SOURCES = [
    DATASET_DIR / "raw_kaggle" / "intel",
    DATASET_DIR / "raw_kaggle" / "natural",
    DATASET_DIR / "crop_neighbors" / "cogon_grass",
    DATASET_DIR / "crop_neighbors" / "banana_leaf",
    DATASET_DIR / "crop_neighbors" / "rice",
    DATASET_DIR / "crop_neighbors" / "sorghum",
    DATASET_DIR / "crop_neighbors" / "sugarcane",
]

# ══════════════════════════════════════════════════════════════════════════════
# CLAHE CONFIGURATION (image_utils.py)
# ══════════════════════════════════════════════════════════════════════════════
# Applied to: Bouncer inputs, Factory inputs
# NOT applied to: Student/Teacher training (handled by albumentations aug)
CLAHE_CLIP_LIMIT = 2.0  # contrast enhancement ceiling
CLAHE_TILE_GRID = (8, 8)  # local region size for histogram equalization

# ══════════════════════════════════════════════════════════════════════════════
# SAM2 / TIER 1
# ══════════════════════════════════════════════════════════════════════════════
# Auto-prompting HSV range for foreground detection.
# GREEN: healthy leaf tissue
SAM2_GREEN_H_MIN = 35   # was 30 — raised to avoid warm yellows
SAM2_GREEN_H_MAX = 75   # was 90 — excludes cogon grass band (H=75–90)
SAM2_GREEN_S_MIN = 50   # was 40 — excludes dull/background greens
SAM2_GREEN_V_MIN = 40   # unchanged

# YELLOW: MSV chlorotic streaks / early-stage yellowing
SAM2_YELLOW_H_MIN = 15
SAM2_YELLOW_H_MAX = 38
SAM2_YELLOW_S_MIN = 40
SAM2_YELLOW_V_MIN = 80

# BROWN: MLN necrotic tissue / dark amber regions
SAM2_BROWN_H_MIN = 5
SAM2_BROWN_H_MAX = 20
SAM2_BROWN_S_MIN = 30
SAM2_BROWN_V_MIN = 50

# QA filter thresholds
SAM2_QA_MIN_COVERAGE = 0.03   # reject if foreground < 3% of image (close-up leaves fill frame)
SAM2_QA_MAX_COVERAGE = 0.99   # was 0.90 — raised after data showed 2,537 valid close-up masks wrongly rejected
SAM2_QA_MIN_CONFIDENCE = 0.50 # was 0.60/0.65 — lowered after data showed diseased leaves cluster at 0.51–0.64
SAM2_QA_MIN_ASPECT_RATIO = 1.01  # was 1.20 — overhead/square-frame leaves are valid
SAM2_QA_MAX_REJECT_RATE = 0.08   # warn if > 8% of images rejected

# ══════════════════════════════════════════════════════════════════════════════
# TEACHER
# ══════════════════════════════════════════════════════════════════════════════
TEACHER_IMG_SIZE = 512
TEACHER_BATCH_SIZE = 8
TEACHER_EPOCHS = 30
TEACHER_LR = 5e-5
TEACHER_WEIGHT_DECAY = 1e-4
TEACHER_VAL_SPLIT = 0.20
TEACHER_PATIENCE = 10
TEACHER_LR_FACTOR = 0.5
TEACHER_LR_PATIENCE = 5

TEACHER_VARIANTS = [
    "resnet50",  # deep CNN baseline
    "efficientnet-b2",  # recommended — best quality/VRAM ratio
    "mit_b2",  # SegFormer-B2 via timm (transformer comparison)
    "deeplabv3plus-eb2",  # DeepLabV3+ decoder — true architecture comparison vs UNet
]
TEACHER_DEPLOYED_VARIANT = "efficientnet-b2"  # best by val Dice

# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════
FACTORY_MODES = ["mode_a", "mode_b", "mode_c", "mode_d"]

# Teacher inference threshold (lowered from 0.5 to catch dark leaves)
FACTORY_SILHOUETTE_THRESHOLD = 0.35

# Coverage guard — below this, severity = -1 (excluded from training)
FACTORY_MIN_LEAF_COVERAGE = 0.15  # 15% minimum (was 25%, relaxed)

# Reliability weights for borderline images (replacing hard sentinel)
FACTORY_WEIGHT_BRACKETS = [
    (0.00, 0.15, None),  # < 15% coverage → exclude (severity = -1)
    (0.15, 0.25, 0.30),  # 15–25% → weight 0.3
    (0.25, 0.50, 0.70),  # 25–50% → weight 0.7
    (0.50, 1.00, 1.00),  # > 50%  → weight 1.0
]

# R3 area filter for MSV near-white range (min connected component size)
FACTORY_R3_MIN_AREA_PX = 80

# Morphological ops for silhouette refinement
FACTORY_MORPH_KERNEL_SIZE = 5

# HSV green exclusion zone (removes healthy leaf tissue from symptom masks)
HSV_GREEN_EXCL = {"h_min": 38, "h_max": 85, "s_min": 80, "v_min": 60, "v_max": 230}

# MSV HSV ranges (4 overlapping bands)
HSV_MSV_RANGES = [
    {
        "name": "R1",
        "h": (15, 38),
        "s": (50, 255),
        "v": (150, 255),
    },  # bright yellow streaks
    {
        "name": "R2",
        "h": (20, 45),
        "s": (15, 70),
        "v": (130, 255),
    },  # pale yellow / early-stage
    {
        "name": "R3",
        "h": (0, 179),
        "s": (0, 35),
        "v": (210, 255),
    },  # near-white / bleached (R3 area filter applied)
    {"name": "R4", "h": (38, 55), "s": (10, 55), "v": (140, 255)},  # pale yellow-green
]

# MLN HSV ranges (5 overlapping bands)
HSV_MLN_RANGES = [
    {"name": "R1", "h": (18, 40), "s": (40, 255), "v": (90, 255)},  # chlorotic yellow
    {
        "name": "R2",
        "h": (5, 20),
        "s": (40, 255),
        "v": (50, 220),
    },  # orange-amber necrosis
    {
        "name": "R3",
        "h": (22, 55),
        "s": (15, 85),
        "v": (80, 240),
    },  # pale yellow-green mosaic
    {"name": "R4", "h": (8, 35), "s": (0, 45), "v": (160, 255)},  # tan / straw tissue
    {
        "name": "R5",
        "h": (0, 18),
        "s": (30, 180),
        "v": (30, 150),
    },  # dark brown dead tissue
]

# ══════════════════════════════════════════════════════════════════════════════
# STUDENT
# ══════════════════════════════════════════════════════════════════════════════
STUDENT_IMG_SIZE = 224
STUDENT_BATCH_SIZE = 32
STUDENT_NUM_WORKERS = 4

# Two-phase training
STUDENT_PHASE1_EPOCHS = 30
STUDENT_PHASE1_LR = 1e-3
STUDENT_PHASE1_PATIENCE = 10
STUDENT_PHASE2_EPOCHS = 20
STUDENT_PHASE2_LR = 1e-4
STUDENT_PHASE2_LR_MIN = 1e-6
STUDENT_PHASE2_PATIENCE = 5

STUDENT_WEIGHT_DECAY = 1e-4
STUDENT_DROPOUT = 0.3

# Composite checkpoint criterion weights
STUDENT_CKPT_W_MIOU   = 0.40   # was 0.50 — reduced to make room for MLN
STUDENT_CKPT_W_MSV_F1 = 0.35   # unchanged — primary thesis claim
STUDENT_CKPT_W_MLN_F1 = 0.15   # NEW — prevents degenerate MSV-only model
STUDENT_CKPT_W_MAE    = 0.10   # was 0.15 — reduced; severity is secondary
STUDENT_MAX_SEVERITY = 100.0  # for normalizing MAE

# Label smoothing
STUDENT_LABEL_SMOOTHING = 0.10

# ════════════════════════════════════════════════════════════════════════════════
# GABOR FILTER PARAMETERS (for MSV texture enhancement)
# ═══════════════════════════════════════════════════════════════════════════════
# Gabor filters for capturing MSV streak directionality (parallel to leaf veins)
GABOR_KERNEL_SIZE = 21          # Kernel size (must be odd)
GABOR_SIGMA      = 4.0          # Standard deviation of Gaussian envelope
GABOR_LAMBDA     = 10.0         # Wavelength of sinusoidal factor
GABOR_GAMMA      = 0.5          # Spatial aspect ratio
GABOR_PSI        = 0            # Phase offset
GABOR_NORMS      = [0.1, 0.2, 0.3, 0.4]  # Normalized frequencies
GABOR_THETAS     = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Orientations (0°, 45°, 90°, 135°)
GABOR_THRESHOLD  = 0.3          # Threshold for Gabor response (0-1)

# Asymmetric label smoothing prior matrix
# Rows = true class [HEALTHY, MSV, MLN]
# Grounded in pathological literature (Cruz et al. 2024, Mushayi et al. 2025)
ASYMMETRIC_PRIOR = [
    [0.90, 0.08, 0.02],  # HEALTHY → most confusion with MSV
    [0.05, 0.90, 0.05],  # MSV     → equal confusion both ways
    [0.02, 0.05, 0.93],  # MLN     → most distinct class
]

# Student encoder variants (Stage 1 ablation — all trained on Mode B)
# NOTE: mobilevit_xxs removed — TFLite self-attention einsum incompatibility
# produces an incomplete comparison row that cannot participate in mobile
# composite scoring or deployment selection. Cite published MobileViT
# benchmarks in related work instead.
STUDENT_VARIANTS = [
    "mobilenet_v2",  # V1 — primary architecture per thesis title; no-attention baseline
    "mobilenet_v2_cbam",  # V2 — CBAM at skip junctions (expected winner, core contribution)
    "mobilenet_v3_small",  # V3 — ultra-compact, coord. attention
    "efficientnet_b0",  # V4 — compound scaling, no-attention baseline for B0
    "efficientnet_b0_cbam",  # V6 — tests if CBAM generalizes beyond MobileNetV2
]
# Set after Stage 1 completes
STUDENT_BEST_VARIANT = "mobilenet_v2_cbam"

# Factory mode for Stage 2 ablation
# NOTE: Do not change STUDENT_FACTORY_MODE manually between runs.
# It is used only as a fallback default. The actual best mode is
# auto-detected from logs/student_comparison_stage2.csv by all
# evaluation scripts. Set this to the winning mode AFTER Stage 2 completes.
STUDENT_FACTORY_MODE = "mode_b"  # update after Stage 2

# Severity head
STUDENT_SEV_ACTIVATION = "relu_clamp"  # ReLU + clamp(0,1), NOT sigmoid

# CBAM kernel size for spatial attention
CBAM_SPATIAL_KERNEL = 7

# ══════════════════════════════════════════════════════════════════════════════
# XAI
# ══════════════════════════════════════════════════════════════════════════════
XAI_METHODS = ["gradcam", "gradcamplusplus", "scorecam"]
XAI_DEPLOYED_METHOD = "gradcamplusplus"

# Target layer per encoder (last conv block before decoder branch)
XAI_TARGET_LAYERS = {
    "mobilenet_v2": "encoder.features[-1][0]",
    "mobilenet_v2_cbam": "encoder.features[-1][0]",
    "mobilenet_v3_small": "encoder.features[-1][0]",
    "efficientnet_b0": "encoder.blocks[-1][-1]",
    "efficientnet_b0_cbam": "encoder.blocks[-1][-1]",
}

# ══════════════════════════════════════════════════════════════════════════════
# GOLD STANDARD HUMAN VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
# 300 images manually annotated in Label Studio (100 per class: HEALTHY/MSV/MLN)
# Used for IoU validation of SAM2 masks, Teacher predictions, and Student
# predictions against human-verified leaf silhouette ground truth.
#
# FOLDER LAYOUT (place files here before running validate_gold_standard.py):
#
#   data/gold_standard/
#   ├── images/                     ← raw images (copied by sample_gold_standard.py)
#   │   ├── HEALTHY_image001.jpg
#   │   ├── MSV_image045.jpg
#   │   └── MLN_image012.jpg
#   └── annotations/                ← Label Studio JSON export (one file per image
#                                      OR a single project export JSON)
#       └── annotations.json        ← export via Label Studio → Export → JSON
#
# HOW TO EXPORT FROM LABEL STUDIO:
#   Project → Export → JSON → download → place at data/gold_standard/annotations/
#   The script accepts both single-file project export and per-image JSON files.

GOLD_DIR = DATA_DIR / "gold_standard"
GOLD_IMAGES_DIR = GOLD_DIR / "images"
GOLD_ANNOTATIONS_DIR = GOLD_DIR / "annotations"
GOLD_ANNOTATION_FILE = GOLD_ANNOTATIONS_DIR / "annotations.json"
GOLD_MANIFEST = GOLD_DIR / "gold_manifest.csv"  # auto-generated on first run

# IoU threshold below which a mask is flagged as a poor match
GOLD_IOU_WARN_THRESHOLD = 0.75  # warn if per-image IoU drops below this
# Expected minimum mean IoU for thesis validation claim (target > 0.85)
GOLD_IOU_TARGET_MEAN = 0.85

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
# Severity inter-rater protocol
SEVERITY_EVAL_N_IMAGES = 60  # 20 per class
SEVERITY_EVAL_SCALE_MAX = 3  # 0=none, 1=mild, 2=moderate, 3=severe

# Bouncer ROC evaluation — minimum maize recall constraint
BOUNCER_MIN_MAIZE_RECALL = 0.95
