import numpy as np

from scipy.ndimage import median_filter

from adaptive_stain import rgb_to_od

from enhancement_methods import (
    apply_enhancement,
)


#    
# 2x DOWNSAMPLING
#    

def downsample2x_rgb_average(
    img_u8
):
    """
    40x -> approximately 20x by 2x2 RGB average pooling.
    """

    H, W, C = img_u8.shape

    H2 = (
        H // 2
    ) * 2

    W2 = (
        W // 2
    ) * 2

    img = (
        img_u8[
            :H2,
            :W2
        ]
        .astype(
            np.float32
        )
    )

    out = (
        img
        .reshape(
            H2 // 2,
            2,
            W2 // 2,
            2,
            C
        )
        .mean(
            axis=(
                1,
                3
            )
        )
    )

    return np.clip(
        out,
        0,
        255
    ).astype(
        np.uint8
    )


def downsample2x_mask_max(
    mask
):
    H, W = mask.shape

    H2 = (
        H // 2
    ) * 2

    W2 = (
        W // 2
    ) * 2

    m = (
        mask[
            :H2,
            :W2
        ]
        .astype(
            np.uint8
        )
    )

    out = (
        m
        .reshape(
            H2 // 2,
            2,
            W2 // 2,
            2
        )
        .max(
            axis=(
                1,
                3
            )
        )
    )

    return out.astype(
        bool
    )


#    
# COMMON CLEAN-REFERENCE TISSUE SUPPORT
#    

def clean_reference_tissue_mask_20x(
    raw_rgb_u8
):
    """
    Create tissue support from the CLEAN RAW ROI.

    This same support is used for all five preprocessing
    strategies so evaluation area cannot change by method.
    """

    od = rgb_to_od(
        raw_rgb_u8
    )

    mask_40x = np.any(
        od > 0.05,
        axis=2
    )

    mask_20x = downsample2x_mask_max(
        mask_40x
    )

    return mask_20x


#    
# FINAL CONFERENCE PREPROCESSING
#    

def preprocess_for_method(
    raw_rgb_u8,
    method,
    fold_cfg,
):
    """
    Final common preprocessing order:

    raw RGB
        ->
    selected enhancement / normalization
        ->
    3x3 median filtering
        ->
    2x RGB average downsampling
        ->
    return 20x image

    Tissue support is deliberately derived independently
    from the clean raw ROI.
    """

    enhanced = apply_enhancement(
        raw_rgb_u8,
        method,
        fold_cfg
    )

    filtered = median_filter(
        enhanced,
        size=(
            3,
            3,
            1
        )
    ).astype(
        np.uint8
    )

    img_20x = downsample2x_rgb_average(
        filtered
    )

    tissue_20x = (
        clean_reference_tissue_mask_20x(
            raw_rgb_u8
        )
    )

    if (
        img_20x.shape[:2]
        != tissue_20x.shape
    ):
        raise RuntimeError(
            "20x image/tissue-mask shape mismatch."
        )

    return (
        img_20x,
        tissue_20x
    )