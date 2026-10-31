import numpy as np


#    
# BASIC RGB / OPTICAL-DENSITY CONVERSION
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


def rgb_to_od(img_rgb_u8):
    """
    RGB uint8 -> optical density.

    Uses 256 in denominator so pure white maps to OD=0.
    """

    validate_rgb_u8(img_rgb_u8)

    img = img_rgb_u8.astype(np.float32)

    od = -np.log(
        (img + 1.0) / 256.0
    )

    return od.astype(np.float32)


def od_to_rgb(od):
    """
    Optical density -> RGB uint8.
    """

    rgb = (
        np.exp(-od) * 256.0
        - 1.0
    )

    rgb = np.clip(
        rgb,
        0,
        255
    )

    return np.round(
        rgb
    ).astype(np.uint8)


#    
# BRIGHTNESS STANDARDIZATION
#    

def standardize_brightness(img_rgb_u8):
    """
    Used only while estimating stain vectors.
    """

    p90 = float(
        np.percentile(
            img_rgb_u8,
            90
        )
    )

    if p90 <= 0:
        return img_rgb_u8.copy()

    out = (
        img_rgb_u8.astype(np.float32)
        * 255.0
        / p90
    )

    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


#    
# IMAGE-ADAPTIVE MACENKO STAIN MATRIX
#    

def estimate_stain_matrix(
    img_rgb_u8,
    beta=0.05,
    alpha=0.5,
):
    """
    Estimate an image-specific two-stain matrix.

    Returns a 2x3 matrix:
        row 0 -> estimated hematoxylin direction
        row 1 -> estimated eosin direction
    """

    validate_rgb_u8(img_rgb_u8)

    standardized = standardize_brightness(
        img_rgb_u8
    )

    od = rgb_to_od(
        standardized
    ).reshape(-1, 3)

    # Remove near-background pixels
    valid = np.any(
        od > float(beta),
        axis=1
    )

    od_valid = od[valid]

    if od_valid.shape[0] < 500:
        raise ValueError(
            "Insufficient optical-density variation "
            "for stain estimation."
        )

    # Principal OD plane
    cov = np.cov(
        od_valid,
        rowvar=False
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(cov)
    )

    # Two principal eigenvectors
    V = eigenvectors[:, [2, 1]]

    # Stabilize direction
    if V[0, 0] < 0:
        V[:, 0] *= -1

    if V[0, 1] < 0:
        V[:, 1] *= -1

    projected = np.dot(
        od_valid,
        V
    )

    phi = np.arctan2(
        projected[:, 1],
        projected[:, 0]
    )

    min_phi, max_phi = np.percentile(
        phi,
        [
            float(alpha),
            100.0 - float(alpha),
        ]
    )

    if abs(max_phi - min_phi) < 0.15:
        raise ValueError(
            "Estimated stain directions are too similar."
        )

    v1 = np.dot(
        V,
        [
            np.cos(min_phi),
            np.sin(min_phi),
        ]
    )

    v2 = np.dot(
        V,
        [
            np.cos(max_phi),
            np.sin(max_phi),
        ]
    )

    # Keep ordering consistent with legacy Macenko logic
    if v1[0] > v2[0]:
        W = np.array(
            [v1, v2],
            dtype=np.float32
        )
    else:
        W = np.array(
            [v2, v1],
            dtype=np.float32
        )

    # Unit-normalize each stain vector
    norms = np.linalg.norm(
        W,
        axis=1,
        keepdims=True
    )

    W = W / (
        norms + 1e-8
    )

    return W.astype(np.float32)


#    
# DECOMPOSITION
#    

def decompose_he(img_rgb_u8):
    """
    Image-adaptive H&E decomposition.

    Returns:
        stain_matrix : 2x3
        H             : HxW hematoxylin concentration
        E             : HxW eosin concentration
        residual_od   : HxWx3 residual OD

    The residual is retained so clean reconstruction remains
    extremely close to the original RGB image.
    """

    validate_rgb_u8(img_rgb_u8)

    H_img, W_img = (
        img_rgb_u8.shape[:2]
    )

    stain_matrix = (
        estimate_stain_matrix(
            img_rgb_u8
        )
    )

    od = rgb_to_od(
        img_rgb_u8
    )

    od_flat = od.reshape(
        -1,
        3
    )

    concentrations = np.dot(
        od_flat,
        np.linalg.pinv(
            stain_matrix
        )
    )

    # Negative stain concentrations have no physical meaning
    concentrations = np.maximum(
        concentrations,
        0.0
    ).astype(np.float32)

    reconstructed_stain_od = np.dot(
        concentrations,
        stain_matrix
    )

    # Keep whatever the two-stain model does not explain
    residual = (
        od_flat
        - reconstructed_stain_od
    )

    H_conc = (
        concentrations[:, 0]
        .reshape(
            H_img,
            W_img
        )
    )

    E_conc = (
        concentrations[:, 1]
        .reshape(
            H_img,
            W_img
        )
    )

    residual_od = residual.reshape(
        H_img,
        W_img,
        3
    ).astype(np.float32)

    return {
        "stain_matrix":
            stain_matrix,

        "H":
            H_conc.astype(
                np.float32
            ),

        "E":
            E_conc.astype(
                np.float32
            ),

        "residual_od":
            residual_od,
    }


#    
# RECONSTRUCTION / PERTURBATION
#    

def reconstruct_from_decomposition(
    decomposition,
    alpha_H=1.0,
    alpha_E=1.0,
):
    """
    Reconstruct RGB while optionally scaling H and E
    concentrations.

    alpha_H = 1 and alpha_E = 1 gives clean reconstruction.
    """

    H = decomposition["H"]
    E = decomposition["E"]

    stain_matrix = (
        decomposition[
            "stain_matrix"
        ]
    )

    residual = (
        decomposition[
            "residual_od"
        ]
    )

    height, width = H.shape

    concentrations = np.stack(
        [
            H * float(alpha_H),
            E * float(alpha_E),
        ],
        axis=-1
    )

    stain_od = np.dot(
        concentrations.reshape(
            -1,
            2
        ),
        stain_matrix
    ).reshape(
        height,
        width,
        3
    )

    od_new = (
        stain_od
        + residual
    )

    # Negative OD is not physically meaningful
    od_new = np.maximum(
        od_new,
        0.0
    )

    return od_to_rgb(
        od_new
    )