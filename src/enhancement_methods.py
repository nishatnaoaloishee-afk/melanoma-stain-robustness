from pathlib import Path
import json

import cv2
import numpy as np
from skimage import color

from adaptive_stain import (
    decompose_he,
    rgb_to_od,
    od_to_rgb,
)


METHODS = [
    "no_correction",
    "macenko",
    "reinhard",
    "clahe",
    "macenko_clahe",
]


DISPLAY_NAMES = {
    "no_correction": "No correction",
    "macenko": "Macenko",
    "reinhard": "Reinhard",
    "clahe": "CLAHE",
    "macenko_clahe": "Macenko + CLAHE",
}


#    
# BASIC CHECK
#    

def validate_rgb_u8(img):
    if not isinstance(img, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Expected HxWx3 RGB image, got {img.shape}."
        )

    if img.dtype != np.uint8:
        raise TypeError(
            f"Expected uint8 image, got {img.dtype}."
        )


#    
# SIMPLE TISSUE SUPPORT
#    

def make_tissue_mask(img_rgb_u8):
    """
    Tissue support estimated from the CURRENT input image.

    This is used only for preprocessing statistics.
    Final segmentation evaluation will later use a fixed
    clean-reference support mask.
    """

    validate_rgb_u8(img_rgb_u8)

    od = rgb_to_od(img_rgb_u8)

    mask = np.any(
        od > 0.05,
        axis=2
    )

    return mask.astype(bool)


#    
# LOAD TRAINING-ONLY TARGET
#    

def load_fold_target(
    target_dir: Path,
    fold: int,
):
    path = (
        Path(target_dir)
        / f"fold{int(fold):02d}_enhancement_target.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with open(
        path,
        "r"
    ) as f:
        cfg = json.load(f)

    return cfg


#    
# NO CORRECTION
#    

def no_correction(img_rgb_u8):
    validate_rgb_u8(img_rgb_u8)

    return img_rgb_u8.copy()


#    
# MACENKO NORMALIZATION
#    

def macenko_normalize(
    img_rgb_u8,
    cfg,
):
    """
    Image-adaptive Macenko normalization using the target
    estimated only from the training ROIs of the corresponding
    fold.
    """

    validate_rgb_u8(img_rgb_u8)

    tissue = make_tissue_mask(
        img_rgb_u8
    )

    if tissue.sum() == 0:
        return img_rgb_u8.copy()

    dec = decompose_he(
        img_rgb_u8
    )

    H = dec["H"]
    E = dec["E"]

    source_H_p99 = float(
        np.percentile(
            H[tissue],
            99
        )
    )

    source_E_p99 = float(
        np.percentile(
            E[tissue],
            99
        )
    )

    target_cfg = cfg["macenko"]

    target_matrix = np.asarray(
        target_cfg["stain_matrix"],
        dtype=np.float32
    )

    target_H_p99 = float(
        target_cfg["target_H_p99"]
    )

    target_E_p99 = float(
        target_cfg["target_E_p99"]
    )

    scale_H = (
        target_H_p99
        / (
            source_H_p99
            + 1e-8
        )
    )

    scale_E = (
        target_E_p99
        / (
            source_E_p99
            + 1e-8
        )
    )

    H_norm = (
        H * scale_H
    )

    E_norm = (
        E * scale_E
    )

    concentrations = np.stack(
        [
            H_norm,
            E_norm,
        ],
        axis=-1
    )

    height, width = H.shape

    od_norm = np.dot(
        concentrations.reshape(
            -1,
            2
        ),
        target_matrix
    ).reshape(
        height,
        width,
        3
    )

    od_norm = np.maximum(
        od_norm,
        0.0
    )

    normalized = od_to_rgb(
        od_norm
    )

    # Preserve non-tissue background from input
    out = img_rgb_u8.copy()
    out[tissue] = normalized[tissue]

    return out


#    
# REINHARD NORMALIZATION
#    

def reinhard_normalize(
    img_rgb_u8,
    cfg,
):
    """
    Reinhard colour normalization in Lab space.

    Target Lab statistics come only from the training portion
    of the corresponding fold.
    """

    validate_rgb_u8(img_rgb_u8)

    tissue = make_tissue_mask(
        img_rgb_u8
    )

    if tissue.sum() == 0:
        return img_rgb_u8.copy()

    rgb = (
        img_rgb_u8.astype(
            np.float32
        )
        / 255.0
    )

    lab = color.rgb2lab(
        rgb
    )

    tissue_lab = lab[
        tissue
    ]

    source_mean = tissue_lab.mean(
        axis=0
    )

    source_std = tissue_lab.std(
        axis=0,
        ddof=1
    )

    target_mean = np.asarray(
        cfg["reinhard"]["mean"],
        dtype=np.float32
    )

    target_std = np.asarray(
        cfg["reinhard"]["std"],
        dtype=np.float32
    )

    lab_norm = lab.copy()

    for ch in range(3):

        lab_norm[
            :,
            :,
            ch
        ] = (
            (
                lab[
                    :,
                    :,
                    ch
                ]
                -
                source_mean[ch]
            )
            /
            (
                source_std[ch]
                +
                1e-8
            )
            *
            target_std[ch]
            +
            target_mean[ch]
        )

    # Safe Lab ranges
    lab_norm[:, :, 0] = np.clip(
        lab_norm[:, :, 0],
        0.0,
        100.0
    )

    lab_norm[:, :, 1] = np.clip(
        lab_norm[:, :, 1],
        -128.0,
        127.0
    )

    lab_norm[:, :, 2] = np.clip(
        lab_norm[:, :, 2],
        -128.0,
        127.0
    )

    rgb_norm = color.lab2rgb(
        lab_norm
    )

    rgb_norm = np.clip(
        rgb_norm,
        0.0,
        1.0
    )

    normalized = np.round(
        rgb_norm * 255.0
    ).astype(np.uint8)

    out = img_rgb_u8.copy()
    out[tissue] = normalized[tissue]

    return out


#    
# CLAHE ON LAB L CHANNEL
#    

def clahe_l(
    img_rgb_u8,
    cfg,
):
    validate_rgb_u8(img_rgb_u8)

    clahe_cfg = cfg["clahe"]

    clip_limit = float(
        clahe_cfg["clip_limit"]
    )

    tile_grid = tuple(
        int(x)
        for x in clahe_cfg[
            "tile_grid_size"
        ]
    )

    lab = cv2.cvtColor(
        img_rgb_u8,
        cv2.COLOR_RGB2LAB
    )

    L, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid
    )

    L2 = clahe.apply(
        L
    )

    lab2 = cv2.merge(
        [
            L2,
            a,
            b
        ]
    )

    rgb2 = cv2.cvtColor(
        lab2,
        cv2.COLOR_LAB2RGB
    )

    return rgb2.astype(
        np.uint8
    )


#    
# METHOD WRAPPER
#    

def apply_enhancement(
    img_rgb_u8,
    method,
    cfg,
):

    if method not in METHODS:
        raise KeyError(
            f"Unknown method: {method}"
        )

    if method == "no_correction":
        return no_correction(
            img_rgb_u8
        )

    if method == "macenko":
        return macenko_normalize(
            img_rgb_u8,
            cfg
        )

    if method == "reinhard":
        return reinhard_normalize(
            img_rgb_u8,
            cfg
        )

    if method == "clahe":
        return clahe_l(
            img_rgb_u8,
            cfg
        )

    if method == "macenko_clahe":

        temp = macenko_normalize(
            img_rgb_u8,
            cfg
        )

        return clahe_l(
            temp,
            cfg
        )

    raise RuntimeError(
        "Unhandled enhancement method."
    )