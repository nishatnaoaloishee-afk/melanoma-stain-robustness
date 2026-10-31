# ============================================================
# Legacy preprocessing implementation
#
# - Macenko stain normalization (fit once on reference_qupath.png)
# - Median filtering
# - Tissue mask (Otsu + morphology)
# - Optional 2x downsample (40x -> 20x approximation)
# ============================================================

import os
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from skimage import filters, morphology, color as skcolor


# ---------------------------
# Reference image path (same folder as this file)
# ---------------------------

REFERENCE_IMAGE_PATH = "reference_qupath.png"


# ---------------------------
# Macenko utilities
# ---------------------------
def RGB_to_OD(I: np.ndarray) -> np.ndarray:
    I = I.astype(np.float32)
    I[I == 0] = 1
    return -np.log((I + 1) / 255.0)

def OD_to_RGB(OD: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(-OD) * 255.0, 0, 255).astype(np.uint8)

def normalize_rows(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)

def standardize_brightness(I: np.ndarray) -> np.ndarray:
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / (p + 1e-8), 0, 255).astype(np.uint8)

def get_concentrations(I: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    OD = RGB_to_OD(I).reshape((-1, 3))
    return np.dot(OD, np.linalg.pinv(stain_matrix))

def get_stain_matrix(I: np.ndarray, beta: float = 0.05, alpha: float = 0.5) -> np.ndarray:
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[(OD > beta).any(axis=1), :]

    if OD.shape[0] < 500:
        raise ValueError("Insufficient OD variation for stain matrix")

    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, [2, 1]]

    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1

    That = np.dot(OD, V)
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi, maxPhi = np.percentile(phi, [alpha, 100 - alpha])
    if abs(maxPhi - minPhi) < 0.15:
        raise ValueError("Small angular spread")

    v1 = np.dot(V, [np.cos(minPhi), np.sin(minPhi)])
    v2 = np.dot(V, [np.cos(maxPhi), np.sin(maxPhi)])

    HE = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])
    return normalize_rows(HE)


class MacenkoNormalizer:
    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None

    def fit(self, target_img: np.ndarray):
        target = standardize_brightness(target_img)
        self.stain_matrix_target = get_stain_matrix(target)
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)

    def transform(self, I: np.ndarray):
        I = standardize_brightness(I)
        try:
            stain_matrix_source = get_stain_matrix(I)
            vec1, vec2 = stain_matrix_source
            if np.abs(np.dot(vec1, vec2)) > 0.95:
                raise ValueError("Stain vectors too close")

            source_conc = get_concentrations(I, stain_matrix_source)

            maxC_src = np.percentile(source_conc, 99, axis=0).reshape((1, 2))
            maxC_tgt = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
            source_conc *= (maxC_tgt / (maxC_src + 1e-8))

            H, W, _ = I.shape
            OD_norm = np.dot(source_conc, self.stain_matrix_target)
            out = OD_to_RGB(OD_norm.reshape(H, W, 3))
            return out, False

        except (np.linalg.LinAlgError, ValueError):
            # Fallback: standardized brightness only
            return standardize_brightness(I), True


# ---------------------------
# 2x downsample helpers
# ---------------------------
def downsample2x_rgb_average(img_u8: np.ndarray) -> np.ndarray:
    """2× downsample RGB uint8 by 2×2 average pooling."""
    H, W, C = img_u8.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    img = img_u8[:H2, :W2].astype(np.float32)
    out = img.reshape(H2 // 2, 2, W2 // 2, 2, C).mean(axis=(1, 3))
    return np.clip(out, 0, 255).astype(np.uint8)

def downsample2x_mask_max(mask: np.ndarray) -> np.ndarray:
    """2× downsample boolean mask by 2×2 max pooling."""
    H, W = mask.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    m = mask[:H2, :W2].astype(np.uint8)
    out = m.reshape(H2 // 2, 2, W2 // 2, 2).max(axis=(1, 3))
    return out.astype(bool)


# ---------------------------
# Build normalizer once (module import time)
# ---------------------------
def build_normalizer(reference_image_path: str = REFERENCE_IMAGE_PATH) -> MacenkoNormalizer:
    if not os.path.exists(reference_image_path):
        raise FileNotFoundError(
            f"Reference image not found: {reference_image_path}\n"
            f"Place reference_qupath.png next to puma_preprocess_final.py"
        )
    ref_img = np.array(Image.open(reference_image_path).convert("RGB"))
    norm = MacenkoNormalizer()
    norm.fit(ref_img)
    return norm

NORMALIZER = build_normalizer()


# ---------------------------
# Main preprocessing function
# ---------------------------
def preprocess_puma_roi(img_rgb: np.ndarray, to_20x: bool = True):
    """
    img_rgb: uint8 RGB array (H,W,3)
    to_20x : if True, downsample 2× to approximate 20×

    returns:
      img_out (uint8 RGB),
      tissue_mask (bool),
      fallback_used (bool)
    """
    # 1) Macenko + fallback
    img_norm, fallback_used = NORMALIZER.transform(img_rgb)

    # 2) Median filter
    img_med = median_filter(img_norm, size=(3, 3, 1)).astype(np.uint8)

    # 3) Tissue mask
    gray = skcolor.rgb2gray(img_med)
    gray8 = (gray * 255).astype(np.uint8)
    gray8 = median_filter(gray8, size=3)
    otsu = filters.threshold_otsu(gray8)

    mask = gray8 < otsu
    mask = morphology.remove_small_objects(mask, min_size=200)
    mask = morphology.binary_opening(mask, morphology.disk(3))
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = morphology.remove_small_holes(mask, area_threshold=100)
    mask = mask.astype(bool)

    # 4) Optional 40× -> 20×
    if to_20x:
        img_med = downsample2x_rgb_average(img_med)
        mask = downsample2x_mask_max(mask)

    return img_med, mask, bool(fallback_used)
