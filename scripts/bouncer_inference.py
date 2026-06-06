"""
================================================================================
 scripts/bouncer_inference.py — Shared Bouncer Inference Helpers
================================================================================
 PURPOSE:
   Single source of truth for the two-stage Bouncer inference functions used
   by both factory_master.py and train_bouncer.py.

   Extracting these here avoids copy-pasting between the two scripts and
   ensures any future changes (e.g. threshold logic, transform pipeline)
   are made in one place only.

   factory_master.py  — uses these at scale for every Tier 2 image
   train_bouncer.py   — uses these for evaluate_admission_rate() only

 IMPORTANT:
   This module has NO imports from factory_master.py or train_bouncer.py.
   It depends only on: torch, albumentations, config.BOUNCER_IMG_SIZE.
   This keeps it lightweight — importing it does NOT pull in SAM2, Teacher,
   or any other heavy dependency.
================================================================================
"""

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2

from config import BOUNCER_IMG_SIZE

# ── Inference transform (letterbox, normalize — matches val transform) ────────
BOUNCER_INFER_TF = A.Compose([
    A.LongestMaxSize(max_size=BOUNCER_IMG_SIZE),
    A.PadIfNeeded(BOUNCER_IMG_SIZE, BOUNCER_IMG_SIZE, border_mode=0, value=0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def heuristic_prefilter(img_rgb: np.ndarray) -> bool:
    """
    Lightweight pre-filter before neural Bouncer inference.
    Currently a passthrough (always returns True).

    The original OpenCV green-coverage heuristic was removed after causing
    false rejections on yellow/bleached MSV leaves under variable tropical
    lighting. The neural classifier is sufficient and fast enough at 224×224.
    """
    return True


@torch.no_grad()
def neural_bouncer(img_rgb: np.ndarray,
                   model: nn.Module,
                   threshold: float) -> bool:
    """
    Run the trained neural Bouncer on a single EXIF-corrected RGB image.
    Returns True (maize) if sigmoid(logit) >= threshold, False (not-maize) otherwise.

    Args:
        img_rgb   : uint8 RGB numpy array (EXIF-corrected, from load_image_clahe)
        model     : trained Bouncer nn.Module in eval() mode
        threshold : empirical sigmoid threshold from ROC curve selection
    """
    tensor = BOUNCER_INFER_TF(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    logit  = model(tensor).squeeze()
    prob   = torch.sigmoid(logit).item()
    return prob >= threshold
