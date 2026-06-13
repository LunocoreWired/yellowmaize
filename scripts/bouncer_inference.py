"""
scripts/bouncer_inference.py — Shared Bouncer Inference Helpers
PURPOSE:
Single source of truth for the two-stage Bouncer inference functions used
by both factory_master.py and train_bouncer.py.
"""
import cv2
import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from config import BOUNCER_IMG_SIZE

# FIX: Albumentations 2.0 API uses 'fill' and 'fill_mask' instead of 'value'
BOUNCER_INFER_TF = A.Compose([
    A.LongestMaxSize(max_size=BOUNCER_IMG_SIZE),
    A.PadIfNeeded(
        min_height=BOUNCER_IMG_SIZE, 
        min_width=BOUNCER_IMG_SIZE, 
        border_mode=cv2.BORDER_CONSTANT, 
        fill=0, 
        fill_mask=0
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def heuristic_prefilter(img_rgb: np.ndarray) -> bool:
    """
    Lightweight pre-filter before neural Bouncer inference.
    Currently a passthrough (always returns True).
    """
    return True

@torch.no_grad()
def neural_bouncer(img_rgb: np.ndarray, model: nn.Module, threshold: float) -> bool:
    """
    Run the trained neural Bouncer on a single EXIF-corrected RGB image.
    Returns True (maize) if sigmoid(logit) >= threshold, False otherwise.
    """
    tensor = BOUNCER_INFER_TF(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    logit  = model(tensor).squeeze()
    prob   = torch.sigmoid(logit).item()
    return prob >= threshold