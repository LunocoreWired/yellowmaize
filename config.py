"""
config.py — Yellow MAIze Central Configuration
Single source of truth for ALL hyperparameters, paths, and settings.
"""
import os
from pathlib import Path
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL SEED
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# ROOT PATHS
# ══════════════════════════════════════════════════════════════════════════════
ROOT                = Path(__file__).parent
MAIZE_DIR           = ROOT / "maize_dataset"
DATASET_DIR         = ROOT / "dataset"
SAM2_CHECKPOINT     = ROOT / "sam2" / "sam2_hiera_large.pt"
SAM2_CONFIG         = "sam2_hiera_l.yaml"
DATA_DIR            = ROOT / "data"
TIER1_RAW_DIR       = DATA_DIR / "tier1_raw"
TIER1_MASKS_DIR     = DATA_DIR / "tier1_leaf_masks"
PSEUDO_DIR          = DATA_DIR / "pseudo_masks"
BOUNCER_DATASET_DIR = DATA_DIR / "bouncer_dataset"
CHECKPOINTS_DIR     = ROOT / "checkpoints"
BOUNCER_CKPT_DIR    = CHECKPOINTS_DIR / "bouncer"
TEACHER_CKPT_DIR    = CHECKPOINTS_DIR / "teacher"
STUDENT_CKPT_DIR    = CHECKPOINTS_DIR / "student"
LOGS_DIR            = ROOT / "logs"
REPORTS_DIR         = ROOT / "reports"
EXPORTS_DIR         = ROOT / "exports" / "tflite"
GLOBAL_MANIFEST     = ROOT / "global_split_manifest.csv"
TIER1_MANIFEST      = ROOT / "tier1_manifest.csv"
TIER1_QA_REPORT     = ROOT / "tier1_qa_report.csv"

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
PREPROC_MIN_DIMENSION        = 64
PREPROC_MAX_DIMENSION        = 4096
PREPROC_MAX_ASPECT_RATIO     = 8.0
PREPROC_UNIFORM_STD_THRESH   = 5.0
PREPROC_GREEN_FLAG_THRESH    = 0.05
PREPROC_PHASH_HAMMING        = 2
PREPROC_MIN_FILE_BYTES       = 100

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
CLASSES             = ["HEALTHY", "MSV", "MLN"]
CLASS_TO_IDX        = {"HEALTHY": 0, "MSV": 1, "MLN": 2}
VALID_EXTENSIONS    = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
SPLIT_RATIOS        = {"train": 0.70, "val": 0.15, "test": 0.15}
TIER1_PER_CLASS     = {"HEALTHY": 3000, "MSV": 7500, "MLN": 4500}

# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════
CUDNN_DETERMINISTIC = True
CUDNN_BENCHMARK     = False

# ══════════════════════════════════════════════════════════════════════════════
# BOUNCER
# ══════════════════════════════════════════════════════════════════════════════
BOUNCER_TARGET_PER_CLASS = 25_000
BOUNCER_IMG_SIZE         = 224
BOUNCER_BATCH_SIZE       = 64
BOUNCER_EPOCHS           = 15
BOUNCER_LR               = 1e-4
BOUNCER_WEIGHT_DECAY     = 1e-4
BOUNCER_VAL_SPLIT        = 0.20
BOUNCER_PATIENCE         = 5
BOUNCER_MIN_ASPECT_RATIO = 1.5
BOUNCER_MAX_ASPECT_RATIO = 18.0
BOUNCER_MIN_GREEN_COVERAGE = 0.15
BOUNCER_THRESHOLD        = 0.65
BOUNCER_VARIANTS = [
    "gabor_lbp", "mobilenet_v2", "mobilenet_v3_large", "edgevit_xxs",
]
BOUNCER_DEPLOYED_VARIANT = "mobilenet_v3_large"
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
# CLAHE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_TILE_GRID    = (8, 8)

# ══════════════════════════════════════════════════════════════════════════════
# SAM2 / TIER 1
# ══════════════════════════════════════════════════════════════════════════════
SAM2_GREEN_H_MIN    = 30
SAM2_GREEN_H_MAX    = 90
SAM2_GREEN_S_MIN    = 40
SAM2_GREEN_V_MIN    = 40
SAM2_YELLOW_H_MIN   = 15
SAM2_YELLOW_H_MAX   = 38
SAM2_YELLOW_S_MIN   = 40
SAM2_YELLOW_V_MIN   = 80
SAM2_BROWN_H_MIN    = 5
SAM2_BROWN_H_MAX    = 20
SAM2_BROWN_S_MIN    = 30
SAM2_BROWN_V_MIN    = 50
SAM2_QA_MIN_COVERAGE     = 0.03  # v3 relaxed: diseased leaves are sparser
SAM2_QA_MAX_COVERAGE     = 0.90
SAM2_QA_MIN_CONFIDENCE   = 0.65
SAM2_QA_MIN_ASPECT_RATIO = 1.01  # v3 relaxed: overhead/square-frame leaves
SAM2_QA_MAX_REJECT_RATE  = 0.08

# ══════════════════════════════════════════════════════════════════════════════
# TEACHER
# ══════════════════════════════════════════════════════════════════════════════
TEACHER_IMG_SIZE     = 768
TEACHER_BATCH_SIZE   = 2
TEACHER_BATCH_SIZE_OVERRIDES = {
    "resnet50":           6,
    "efficientnet-b2":    4,
    "mit_b2":             2,
    "deeplabv3plus-eb2":  3,
}
TEACHER_GRAD_ACCUM_STEPS = 2
TEACHER_EPOCHS       = 30
TEACHER_LR           = 5e-5
TEACHER_WEIGHT_DECAY = 5e-4
TEACHER_VAL_SPLIT    = 0.20
TEACHER_PATIENCE     = 7
TEACHER_LR_FACTOR    = 0.5
TEACHER_LR_PATIENCE  = 5
TEACHER_VARIANTS = [
    "resnet50", "efficientnet-b2", "mit_b2", "deeplabv3plus-eb2",
]
TEACHER_DEPLOYED_VARIANT = "efficientnet-b2"

# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════
FACTORY_MODES = ["mode_a", "mode_b", "mode_c", "mode_d"]
FACTORY_SILHOUETTE_THRESHOLD = 0.35
FACTORY_MIN_LEAF_COVERAGE    = 0.15
FACTORY_WEIGHT_BRACKETS = [
    # (0.00, 0.15) is handled by get_reliability_weight() early-exit
    # on FACTORY_MIN_LEAF_COVERAGE — bracket below that is dead code.
    (0.15, 0.25, 0.30),
    (0.25, 0.50, 0.70),
    (0.50, 1.00, 1.00),
]
FACTORY_R3_MIN_AREA_PX = 80
FACTORY_MORPH_KERNEL_SIZE = 5
HSV_GREEN_EXCL = {"h_min": 38, "h_max": 85, "s_min": 80, "v_min": 60, "v_max": 230}
HSV_MSV_RANGES = [
    {"name": "R1", "h": (15, 38),   "s": (50, 255),  "v": (150, 255)},
    {"name": "R2", "h": (20, 45),   "s": (15, 70),   "v": (130, 255)},
    {"name": "R3", "h": (0,  179),  "s": (0,  35),   "v": (210, 255)},
    {"name": "R4", "h": (38, 55),   "s": (10, 55),   "v": (140, 255)},
]
HSV_MLN_RANGES = [
    {"name": "R1", "h": (18, 40),   "s": (40, 255),  "v": (90,  255)},
    {"name": "R2", "h": (5,  20),   "s": (40, 255),  "v": (50,  220)},
    {"name": "R3", "h": (22, 55),   "s": (15, 85),   "v": (80,  240)},
    {"name": "R4", "h": (8,  35),   "s": (0,  45),   "v": (160, 255)},
    {"name": "R5", "h": (0,  18),   "s": (30, 180),  "v": (30,  150)},
]

# ══════════════════════════════════════════════════════════════════════════════
# STUDENT
# ══════════════════════════════════════════════════════════════════════════════
STUDENT_IMG_SIZE        = 224
STUDENT_BATCH_SIZE      = 32
STUDENT_NUM_WORKERS     = 4
STUDENT_PHASE1_EPOCHS   = 30
STUDENT_PHASE1_LR       = 1e-3
STUDENT_PHASE1_PATIENCE = 10
STUDENT_PHASE2_EPOCHS   = 20
STUDENT_PHASE2_LR       = 1e-4
STUDENT_PHASE2_LR_MIN   = 1e-6
STUDENT_PHASE2_PATIENCE = 5
STUDENT_WEIGHT_DECAY    = 1e-4
STUDENT_DROPOUT         = 0.3
STUDENT_CKPT_W_MIOU   = 0.40
STUDENT_CKPT_W_MSV_F1 = 0.35
STUDENT_CKPT_W_MLN_F1 = 0.15
STUDENT_CKPT_W_MAE    = 0.10
STUDENT_MAX_SEVERITY = 100.0
STUDENT_LABEL_SMOOTHING = 0.10

# ══════════════════════════════════════════════════════════════════════════════
# GABOR FILTER PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
GABOR_KERNEL_SIZE = 21
GABOR_SIGMA      = 4.0
GABOR_LAMBDA     = 10.0   # authoritative — used directly in factory_master.py
GABOR_GAMMA      = 0.5
GABOR_PSI        = 0
GABOR_NORMS      = [0.1, 0.2, 0.3, 0.4]
GABOR_THETAS     = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GABOR_THRESHOLD  = 0.3
ASYMMETRIC_PRIOR = [
    [0.90, 0.08, 0.02],
    [0.05, 0.90, 0.05],
    [0.02, 0.05, 0.93],
]
STUDENT_VARIANTS = [
    "mobilenet_v2", "mobilenet_v2_cbam", "mobilenet_v3_small", "efficientnet_b0", "efficientnet_b0_cbam",
]
STUDENT_BEST_VARIANT = "mobilenet_v2_cbam"
STUDENT_FACTORY_MODE = "mode_b"
STUDENT_SEV_ACTIVATION = "relu_clamp"
CBAM_SPATIAL_KERNEL = 7

# ══════════════════════════════════════════════════════════════════════════════
# XAI
# ══════════════════════════════════════════════════════════════════════════════
XAI_METHODS = ["gradcam", "gradcamplusplus", "scorecam"]
XAI_DEPLOYED_METHOD = "gradcamplusplus"
XAI_N_SAMPLES_CLASS = 30
XAI_INSERTION_STEPS = 25
XAI_TARGET_LAYERS = {
    "mobilenet_v2":           "encoder.features[-1][0]",
    "mobilenet_v2_cbam":      "encoder.features[-1][0]",
    "mobilenet_v3_small":     "encoder.features[-1][0]",
    "efficientnet_b0":        "encoder.blocks[-1][-1]",
    "efficientnet_b0_cbam":   "encoder.blocks[-1][-1]",
}

# ══════════════════════════════════════════════════════════════════════════════
# YOLO LEAF DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
YOLO_DATASET_DIR       = DATA_DIR / "yolo_dataset"
YOLO_IMAGES_DIR        = DATA_DIR / "yolo_annotations" / "images"
YOLO_ANNOTATIONS_DIR   = DATA_DIR / "yolo_annotations" / "labels"
YOLO_ANNOTATION_FILE   = YOLO_ANNOTATIONS_DIR / "annotations.json"
YOLO_WEIGHTS_DIR       = CHECKPOINTS_DIR / "yolo"
YOLO_WEIGHTS_BEST      = CHECKPOINTS_DIR / "yolo" / "best.pt"
YOLO_IMG_SIZE          = 640
YOLO_EPOCHS            = 100
YOLO_BATCH_SIZE        = 8
YOLO_WORKERS           = 2
YOLO_LR0               = 0.01
YOLO_PATIENCE          = 20
YOLO_CONF_THRESHOLD    = 0.25
YOLO_IOU_NMS           = 0.45
YOLO_MIN_ANNOTATIONS   = 400
YOLO_VAL_SPLIT         = 0.15
YOLO_CLASS_NAME        = "leaf"
YOLO_QA_CALIB_FILE     = LOGS_DIR / "yolo_qa_calibration.csv"

# ══════════════════════════════════════════════════════════════════════════════
# GOLD STANDARD HUMAN VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
GOLD_DIR             = DATA_DIR / "gold_standard"
GOLD_IMAGES_DIR      = GOLD_DIR / "images"
GOLD_ANNOTATIONS_DIR = GOLD_DIR / "annotations"
GOLD_ANNOTATION_FILE = GOLD_ANNOTATIONS_DIR / "annotations.json"
GOLD_MANIFEST        = GOLD_DIR / "gold_manifest.csv"
GOLD_IOU_WARN_THRESHOLD  = 0.75
GOLD_IOU_TARGET_MEAN     = 0.85

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
SEVERITY_EVAL_N_IMAGES   = 60
SEVERITY_EVAL_SCALE_MAX  = 3
BOUNCER_MIN_MAIZE_RECALL = 0.95