"""
================================================================================
 image_utils.py — Centralised Image Loading Utilities
================================================================================
 PURPOSE:
   Single source of truth for all image loading across the pipeline.
   Ensures consistent:
     1. EXIF orientation correction  — PIL respects EXIF, cv2 ignores it.
        Loading through this module guarantees the same orientation
        regardless of which library is used downstream.
     2. CLAHE enhancement            — Contrast Limited Adaptive Histogram
        Equalization for Bouncer and Factory inputs. Improves local contrast
        under variable tropical lighting without blowing out highlights.
        Referenced in thesis vocabulary as applied to Bouncer/Factory inputs.
     3. RGB channel order guarantee  — all functions return uint8 RGB numpy
        arrays. cv2 BGR ↔ RGB conversion is handled here, never in callers.
     4. Corrupt / truncated guard     — full pixel load attempted; returns
        None on failure so callers can skip gracefully.

 USAGE:
   from image_utils import load_image_rgb, load_image_clahe

   img = load_image_rgb(path)       # Returns H×W×3 uint8 RGB or None
   img = load_image_clahe(path)     # CLAHE-enhanced version
================================================================================
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

# Allow partial reads only during EXIF check — not during actual load
ImageFile.LOAD_TRUNCATED_IMAGES = False

# CLAHE parameters — tuned for tropical field photography
# clipLimit: contrast enhancement ceiling (prevents noise amplification)
# tileGridSize: local region size for histogram equalization
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ══════════════════════════════════════════════════════════════════════════════
# EXIF-CORRECTED IMAGE LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_image_rgb(path: Path | str) -> np.ndarray | None:
    """
    Load an image as a uint8 RGB numpy array with EXIF orientation applied.

    PIL automatically applies EXIF orientation via ImageOps.exif_transpose().
    This ensures the spatial content is identical whether the image is later
    processed by PIL or cv2 — cv2 ignores EXIF orientation by default, which
    would otherwise cause spatial mismatches between the training pipeline
    (PIL-based DataLoaders) and the Factory (cv2-based HSV masking).

    Returns None if the image is truncated, corrupt, or unreadable.
    """
    try:
        path = Path(path)
        pil_img = Image.open(path)
        pil_img.load()                          # forces full decode — catches truncation

        # Apply EXIF orientation (rotate/flip to canonical upright orientation)
        pil_img = ImageOps.exif_transpose(pil_img)

        img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        return img_rgb
    except Exception:
        return None


def load_image_rgb_cv2(path: Path | str) -> np.ndarray | None:
    """
    Load an image via cv2 with EXIF orientation correction applied.
    cv2.imread() ignores EXIF orientation — this wrapper corrects it
    by reading via PIL first (for EXIF), then returning as uint8 RGB.
    Equivalent to load_image_rgb() but explicitly documents that cv2
    would have been wrong without the PIL EXIF correction.
    """
    return load_image_rgb(path)


# ══════════════════════════════════════════════════════════════════════════════
# CLAHE ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    L channel of an LAB colour space image. Returns uint8 RGB.

    Applied to:
      - Bouncer inputs (improves discrimination under variable field lighting)
      - Factory inputs (improves HSV masking accuracy on low-contrast images)

    NOT applied to Student/Teacher training inputs — those use augmentation-
    based brightness/contrast variation instead, which is more diverse.

    Reference: Zuiderveld (1994), "Contrast Limited Adaptive Histogram
    Equalization", Graphics Gems IV.
    """
    # Convert RGB → LAB (perceptually uniform space)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel only — preserves colour (a, b unchanged)
    l_clahe = _CLAHE.apply(l)

    # Merge and convert back to RGB
    lab_clahe = cv2.merge([l_clahe, a, b])
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return img_clahe.astype(np.uint8)


def load_image_clahe(path: Path | str) -> np.ndarray | None:
    """
    Load image with EXIF correction + CLAHE enhancement.
    Returns uint8 RGB numpy array or None on failure.
    """
    img = load_image_rgb(path)
    if img is None:
        return None
    return apply_clahe(img)


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL ORDER SAFETY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB → BGR for cv2 operations (cv2.imwrite, cv2.imshow etc.)."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR → RGB (when reading with cv2.imread for processing)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_hsv(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB → HSV. Always pass RGB — never BGR."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB → Grayscale."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
