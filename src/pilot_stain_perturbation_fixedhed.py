from collections import OrderedDict

import numpy as np
from skimage import color


#    
# LOCKED STAIN-PERTURBATION CONDITIONS
#    
#
# alpha_H = multiplicative hematoxylin scaling
# alpha_E = multiplicative eosin scaling
#
# Geometry is never modified.
#
PERTURBATIONS = OrderedDict(
    [
        (
            "clean",
            {
                "alpha_H": 1.00,
                "alpha_E": 1.00,
                "family": "clean",
                "severity": "reference",
                "direction": "none",
                "display_name": "Clean",
            },
        ),

        (
            "H_low_mild",
            {
                "alpha_H": 0.85,
                "alpha_E": 1.00,
                "family": "H",
                "severity": "mild",
                "direction": "under",
                "display_name": "H− mild",
            },
        ),

        (
            "H_low_severe",
            {
                "alpha_H": 0.70,
                "alpha_E": 1.00,
                "family": "H",
                "severity": "severe",
                "direction": "under",
                "display_name": "H− severe",
            },
        ),

        (
            "H_high_mild",
            {
                "alpha_H": 1.15,
                "alpha_E": 1.00,
                "family": "H",
                "severity": "mild",
                "direction": "over",
                "display_name": "H+ mild",
            },
        ),

        (
            "H_high_severe",
            {
                "alpha_H": 1.30,
                "alpha_E": 1.00,
                "family": "H",
                "severity": "severe",
                "direction": "over",
                "display_name": "H+ severe",
            },
        ),

        (
            "E_low_mild",
            {
                "alpha_H": 1.00,
                "alpha_E": 0.85,
                "family": "E",
                "severity": "mild",
                "direction": "under",
                "display_name": "E− mild",
            },
        ),

        (
            "E_low_severe",
            {
                "alpha_H": 1.00,
                "alpha_E": 0.70,
                "family": "E",
                "severity": "severe",
                "direction": "under",
                "display_name": "E− severe",
            },
        ),

        (
            "E_high_mild",
            {
                "alpha_H": 1.00,
                "alpha_E": 1.15,
                "family": "E",
                "severity": "mild",
                "direction": "over",
                "display_name": "E+ mild",
            },
        ),

        (
            "E_high_severe",
            {
                "alpha_H": 1.00,
                "alpha_E": 1.30,
                "family": "E",
                "severity": "severe",
                "direction": "over",
                "display_name": "E+ severe",
            },
        ),

        (
            "HE_low_mild",
            {
                "alpha_H": 0.85,
                "alpha_E": 0.85,
                "family": "HE",
                "severity": "mild",
                "direction": "under",
                "display_name": "HE− mild",
            },
        ),

        (
            "HE_low_severe",
            {
                "alpha_H": 0.70,
                "alpha_E": 0.70,
                "family": "HE",
                "severity": "severe",
                "direction": "under",
                "display_name": "HE− severe",
            },
        ),

        (
            "HE_high_mild",
            {
                "alpha_H": 1.15,
                "alpha_E": 1.15,
                "family": "HE",
                "severity": "mild",
                "direction": "over",
                "display_name": "HE+ mild",
            },
        ),

        (
            "HE_high_severe",
            {
                "alpha_H": 1.30,
                "alpha_E": 1.30,
                "family": "HE",
                "severity": "severe",
                "direction": "over",
                "display_name": "HE+ severe",
            },
        ),
    ]
)


#    
# BASIC VALIDATION
#    

def _validate_rgb_u8(img_rgb):
    if not isinstance(img_rgb, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected HxWx3 RGB image, got shape {img_rgb.shape}."
        )

    if img_rgb.dtype != np.uint8:
        raise TypeError(
            f"Expected uint8 RGB image, got {img_rgb.dtype}."
        )


#    
# H&E PERTURBATION
#    

def perturb_he(
    img_rgb_u8: np.ndarray,
    alpha_H: float = 1.0,
    alpha_E: float = 1.0,
) -> np.ndarray:
    """
    Apply controlled H&E concentration perturbation.

    Steps
    -----
    1. RGB uint8 -> RGB float [0,1]
    2. RGB -> HED stain space
    3. Scale hematoxylin and eosin channels independently
    4. HED -> RGB
    5. Clip to valid RGB range
    6. Return uint8 image

    Geometry is unchanged.

    The third HED channel is intentionally left unchanged.
    """

    _validate_rgb_u8(img_rgb_u8)

    if alpha_H <= 0:
        raise ValueError("alpha_H must be > 0.")

    if alpha_E <= 0:
        raise ValueError("alpha_E must be > 0.")

    rgb = (
        img_rgb_u8.astype(np.float32)
        / 255.0
    )

    hed = color.rgb2hed(rgb)

    hed_mod = hed.copy()

    # H = channel 0
    hed_mod[:, :, 0] *= float(alpha_H)

    # E = channel 1
    hed_mod[:, :, 1] *= float(alpha_E)

    # channel 2 is deliberately unchanged

    rgb_out = color.hed2rgb(hed_mod)

    rgb_out = np.nan_to_num(
        rgb_out,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    rgb_out = np.clip(
        rgb_out,
        0.0,
        1.0,
    )

    out_u8 = np.round(
        rgb_out * 255.0
    ).astype(np.uint8)

    if out_u8.shape != img_rgb_u8.shape:
        raise RuntimeError(
            "Perturbation unexpectedly changed image geometry."
        )

    return out_u8


#    
# CONDITION WRAPPER
#    

def apply_condition(
    img_rgb_u8: np.ndarray,
    condition_name: str,
) -> np.ndarray:

    if condition_name not in PERTURBATIONS:
        raise KeyError(
            f"Unknown perturbation condition: {condition_name}"
        )

    cfg = PERTURBATIONS[condition_name]

    if condition_name == "clean":
        return img_rgb_u8.copy()

    return perturb_he(
        img_rgb_u8,
        alpha_H=cfg["alpha_H"],
        alpha_E=cfg["alpha_E"],
    )


#    
# CONDITION TABLE
#    

def perturbation_manifest_rows():
    rows = []

    for condition, cfg in PERTURBATIONS.items():
        rows.append(
            {
                "condition": condition,
                "display_name": cfg["display_name"],
                "family": cfg["family"],
                "severity": cfg["severity"],
                "direction": cfg["direction"],
                "alpha_H": cfg["alpha_H"],
                "alpha_E": cfg["alpha_E"],
            }
        )

    return rows