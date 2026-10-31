from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import polygon

import sys

 
# PATHS
 

ROOT = Path(__file__).resolve().parents[1]

ROI_DIR = (
    ROOT
    / "01_training_dataset_tif_ROIs"
)

GT_DIR = (
    ROOT
    / "01_training_dataset_geojson_nuclei"
)

SPLIT_FILE = (
    ROOT
    / "inputs"
    / "splits"
    / "cv_splits_seed13_folds5.npz"
)

OUTDIR = (
    ROOT
    / "inputs"
    / "training_samples"
)

QC_DIR = (
    ROOT
    / "outputs"
    / "training_sample_qc"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

QC_DIR.mkdir(
    parents=True,
    exist_ok=True
)


sys.path.insert(
    0,
    str(ROOT / "src")
)

from adaptive_stain import (
    rgb_to_od,
)


 
# LOCKED SETTINGS
 

SEED = 13

TOTAL_SAMPLES_PER_ROI = 6000

POSITIVE_SAMPLES_PER_ROI = 3000

NEGATIVE_SAMPLES_PER_ROI = 3000

DOWNSAMPLE_FACTOR = 2


print("=" * 82)
print("PUMA — CREATE COMMON RF TRAINING PIXEL SAMPLES")
print("=" * 82)


 
# LOAD FROZEN COHORT
 

sp = np.load(
    SPLIT_FILE,
    allow_pickle=True
)

roi_names = list(
    sp["roi_names"].tolist()
)

slide_types = list(
    sp["slide_types"].tolist()
)

train_folds = sp["tr"].tolist()
val_folds = sp["va"].tolist()

if len(roi_names) != 205:
    raise RuntimeError(
        f"Expected 205 ROIs, found {len(roi_names)}."
    )


 
# GEOJSON HELPERS
 

def draw_polygon_geometry(
    mask,
    coordinates,
):
    """
    Draw a GeoJSON Polygon.

    First ring = exterior.
    Additional rings = holes.
    """

    if not coordinates:
        return

    # Exterior
    exterior = coordinates[0]

    xy = np.asarray(
        exterior,
        dtype=np.float32
    )

    if (
        xy.ndim != 2
        or xy.shape[1] < 2
        or len(xy) < 3
    ):
        return

    rr, cc = polygon(
        xy[:, 1],
        xy[:, 0],
        shape=mask.shape
    )

    mask[
        rr,
        cc
    ] = True

    # Interior holes
    for hole in coordinates[1:]:

        xy_hole = np.asarray(
            hole,
            dtype=np.float32
        )

        if (
            xy_hole.ndim != 2
            or xy_hole.shape[1] < 2
            or len(xy_hole) < 3
        ):
            continue

        rr, cc = polygon(
            xy_hole[:, 1],
            xy_hole[:, 0],
            shape=mask.shape
        )

        mask[
            rr,
            cc
        ] = False


def draw_geometry(
    mask,
    geometry,
):
    if geometry is None:
        return

    gtype = geometry.get(
        "type",
        None
    )

    coords = geometry.get(
        "coordinates",
        None
    )

    if gtype == "Polygon":

        draw_polygon_geometry(
            mask,
            coords
        )

    elif gtype == "MultiPolygon":

        for poly in coords:

            draw_polygon_geometry(
                mask,
                poly
            )

    elif gtype == "GeometryCollection":

        for geom in geometry.get(
            "geometries",
            []
        ):

            draw_geometry(
                mask,
                geom
            )


def load_nuclei_mask(
    geojson_path,
    image_shape,
):
    """
    Convert every polygon in the PUMA nuclei GeoJSON
    into one binary nuclei mask.
    """

    with open(
        geojson_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    mask = np.zeros(
        image_shape,
        dtype=bool
    )

    # Standard FeatureCollection
    if data.get("type") == "FeatureCollection":

        features = data.get(
            "features",
            []
        )

        for feature in features:

            draw_geometry(
                mask,
                feature.get(
                    "geometry"
                )
            )

    # Single feature
    elif data.get("type") == "Feature":

        draw_geometry(
            mask,
            data.get(
                "geometry"
            )
        )

    # Geometry directly
    else:

        draw_geometry(
            mask,
            data
        )

    return mask


 
# DOWNSAMPLING
 

def downsample_mask_max(
    mask,
):
    """
    2x max-pooling for a binary mask.

    Matches the idea used in the legacy 40x -> 20x pipeline:
    a 20x pixel is positive if any corresponding 40x pixel
    was positive.
    """

    h, w = mask.shape

    h2 = (
        h // 2
    )

    w2 = (
        w // 2
    )

    cropped = mask[
        : h2 * 2,
        : w2 * 2
    ]

    pooled = (
        cropped
        .reshape(
            h2,
            2,
            w2,
            2
        )
        .max(
            axis=(
                1,
                3
            )
        )
    )

    return pooled.astype(
        bool
    )


 
# CLEAN REFERENCE TISSUE MASK
 

def clean_tissue_mask(
    img_rgb_u8
):
    """
    Tissue support generated ONCE from the clean raw ROI.
    """

    od = rgb_to_od(
        img_rgb_u8
    )

    mask = np.any(
        od > 0.05,
        axis=2
    )

    return mask.astype(
        bool
    )


 
# COMMON RNG
 

rng = np.random.default_rng(
    SEED
)


 
# STORAGE
 

sample_rows = []

summary_rows = []

qc_examples = []


 
# PROCESS ALL 205 ROIS
 

for i, (
    roi_name,
    slide_type,
) in enumerate(
    zip(
        roi_names,
        slide_types
    ),
    start=1
):

    print(
        f"[{i:03d}/205] {roi_name}"
    )

    img_path = (
        ROI_DIR
        / f"{roi_name}.tif"
    )

    gt_path = (
        GT_DIR
        / f"{roi_name}_nuclei.geojson"
    )

    if not img_path.exists():
        raise FileNotFoundError(
            img_path
        )

    if not gt_path.exists():
        raise FileNotFoundError(
            gt_path
        )


      
    # LOAD RAW IMAGE
      

    img = np.array(
        Image.open(
            img_path
        ).convert("RGB"),
        dtype=np.uint8
    )

    h, w = img.shape[:2]


      
    # LOAD FULL-RES NUCLEI GT
      

    nuclei_40x = load_nuclei_mask(
        gt_path,
        (
            h,
            w
        )
    )


      
    # CLEAN TISSUE SUPPORT
      

    tissue_40x = clean_tissue_mask(
        img
    )


      
    # DOWNSAMPLE TO THE 20x WORKING GRID
      

    nuclei_20x = downsample_mask_max(
        nuclei_40x
    )

    tissue_20x = downsample_mask_max(
        tissue_40x
    )


    if (
        nuclei_20x.shape
        != tissue_20x.shape
    ):
        raise RuntimeError(
            f"Mask shape mismatch for {roi_name}."
        )


      
    # AVAILABLE TRAINING PIXELS
      

    positive_mask = (
        nuclei_20x
        &
        tissue_20x
    )

    negative_mask = (
        (~nuclei_20x)
        &
        tissue_20x
    )


    pos_y, pos_x = np.where(
        positive_mask
    )

    neg_y, neg_x = np.where(
        negative_mask
    )


    n_pos_available = len(
        pos_y
    )

    n_neg_available = len(
        neg_y
    )


    if (
        n_pos_available
        <
        POSITIVE_SAMPLES_PER_ROI
    ):

        raise RuntimeError(
            f"{roi_name}: only "
            f"{n_pos_available} positive pixels available; "
            f"need {POSITIVE_SAMPLES_PER_ROI}."
        )


    if (
        n_neg_available
        <
        NEGATIVE_SAMPLES_PER_ROI
    ):

        raise RuntimeError(
            f"{roi_name}: only "
            f"{n_neg_available} negative pixels available; "
            f"need {NEGATIVE_SAMPLES_PER_ROI}."
        )


      
    # SAMPLE WITHOUT REPLACEMENT
      

    pos_idx = rng.choice(
        n_pos_available,
        size=POSITIVE_SAMPLES_PER_ROI,
        replace=False
    )

    neg_idx = rng.choice(
        n_neg_available,
        size=NEGATIVE_SAMPLES_PER_ROI,
        replace=False
    )


    sampled_pos_y = pos_y[
        pos_idx
    ]

    sampled_pos_x = pos_x[
        pos_idx
    ]

    sampled_neg_y = neg_y[
        neg_idx
    ]

    sampled_neg_x = neg_x[
        neg_idx
    ]


      
    # RECORD SAMPLES
      

    for y, x in zip(
        sampled_pos_y,
        sampled_pos_x
    ):

        sample_rows.append(
            {
                "roi":
                    roi_name,

                "slide_type":
                    slide_type,

                "y":
                    int(y),

                "x":
                    int(x),

                "label":
                    1,
            }
        )


    for y, x in zip(
        sampled_neg_y,
        sampled_neg_x
    ):

        sample_rows.append(
            {
                "roi":
                    roi_name,

                "slide_type":
                    slide_type,

                "y":
                    int(y),

                "x":
                    int(x),

                "label":
                    0,
            }
        )


    summary_rows.append(
        {
            "roi":
                roi_name,

            "slide_type":
                slide_type,

            "height_20x":
                nuclei_20x.shape[0],

            "width_20x":
                nuclei_20x.shape[1],

            "available_positive_pixels":
                n_pos_available,

            "available_negative_pixels":
                n_neg_available,

            "sampled_positive_pixels":
                POSITIVE_SAMPLES_PER_ROI,

            "sampled_negative_pixels":
                NEGATIVE_SAMPLES_PER_ROI,

            "total_sampled_pixels":
                TOTAL_SAMPLES_PER_ROI,

            "tissue_fraction_20x":
                float(
                    tissue_20x.mean()
                ),

            "nuclei_fraction_within_tissue":
                float(
                    positive_mask.sum()
                    /
                    max(
                        tissue_20x.sum(),
                        1
                    )
                ),
        }
    )


      
    # SAVE TWO QC EXAMPLES
      

    if (
        len(qc_examples) < 2
        and (
            len(qc_examples) == 0
            or slide_type
            != qc_examples[0][
                "slide_type"
            ]
        )
    ):

        # Downsample raw RGB for display
        img_20x = np.array(
            Image.fromarray(
                img
            ).resize(
                (
                    nuclei_20x.shape[1],
                    nuclei_20x.shape[0]
                ),
                resample=Image.Resampling.BILINEAR
            )
        )

        qc_examples.append(
            {
                "roi":
                    roi_name,

                "slide_type":
                    slide_type,

                "img":
                    img_20x,

                "gt":
                    nuclei_20x,

                "pos_y":
                    sampled_pos_y,

                "pos_x":
                    sampled_pos_x,

                "neg_y":
                    sampled_neg_y,

                "neg_x":
                    sampled_neg_x,
            }
        )


 
# SAVE SAMPLE TABLE
 

samples = pd.DataFrame(
    sample_rows
)

summary = pd.DataFrame(
    summary_rows
)


samples.to_csv(
    OUTDIR
    / "common_training_pixels_seed13.csv",
    index=False
)


summary.to_csv(
    OUTDIR
    / "common_training_pixel_summary.csv",
    index=False
)


 
# SAVE COMPACT NPZ
 

np.savez_compressed(
    OUTDIR
    / "common_training_pixels_seed13.npz",

    roi=np.asarray(
        samples["roi"],
        dtype=object
    ),

    y=samples[
        "y"
    ].to_numpy(
        dtype=np.int16
    ),

    x=samples[
        "x"
    ].to_numpy(
        dtype=np.int16
    ),

    label=samples[
        "label"
    ].to_numpy(
        dtype=np.uint8
    ),

    seed=np.asarray(
        [SEED],
        dtype=np.int32
    ),
)


 
# FOLD TRAINING SUMMARY
 

fold_rows = []


for fold in range(
    1,
    6
):

    tr_idx = np.asarray(
        train_folds[
            fold - 1
        ],
        dtype=int
    )

    va_idx = np.asarray(
        val_folds[
            fold - 1
        ],
        dtype=int
    )

    train_names = {
        roi_names[i]
        for i in tr_idx
    }

    fold_samples = samples[
        samples[
            "roi"
        ].isin(
            train_names
        )
    ]

    fold_rows.append(
        {
            "fold":
                fold,

            "training_rois":
                len(tr_idx),

            "validation_rois":
                len(va_idx),

            "training_pixels":
                len(
                    fold_samples
                ),

            "positive_training_pixels":
                int(
                    (
                        fold_samples[
                            "label"
                        ]
                        == 1
                    ).sum()
                ),

            "negative_training_pixels":
                int(
                    (
                        fold_samples[
                            "label"
                        ]
                        == 0
                    ).sum()
                ),
        }
    )


fold_summary = pd.DataFrame(
    fold_rows
)

fold_summary.to_csv(
    OUTDIR
    / "fold_training_pixel_summary.csv",
    index=False
)


 
# QC FIGURES
 

for example in qc_examples:

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            14,
            4.5
        )
    )


    axes[0].imshow(
        example[
            "img"
        ]
    )

    axes[0].set_title(
        "Clean ROI at 20x"
    )


    axes[1].imshow(
        example[
            "gt"
        ],
        cmap="gray"
    )

    axes[1].set_title(
        "Binary nuclei ground truth"
    )


    axes[2].imshow(
        example[
            "img"
        ]
    )


    # Show only a subset so the plot remains readable
    display_pos = slice(
        None,
        None,
        20
    )

    display_neg = slice(
        None,
        None,
        20
    )


    axes[2].scatter(
        example[
            "neg_x"
        ][
            display_neg
        ],
        example[
            "neg_y"
        ][
            display_neg
        ],
        s=5,
        marker="."
    )


    axes[2].scatter(
        example[
            "pos_x"
        ][
            display_pos
        ],
        example[
            "pos_y"
        ][
            display_pos
        ],
        s=5,
        marker="x"
    )


    axes[2].set_title(
        "Common sampled training pixels"
    )


    for ax in axes:
        ax.axis("off")


    fig.suptitle(
        (
            f"{example['slide_type'].capitalize()} — "
            f"{example['roi']}"
        ),
        fontsize=13
    )


    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.92
        ]
    )


    stem = (
        f"{example['slide_type']}_"
        f"training_sample_qc"
    )


    plt.savefig(
        QC_DIR
        / f"{stem}.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.savefig(
        QC_DIR
        / f"{stem}.pdf",
        bbox_inches="tight"
    )


    plt.close()


 
# FINAL INTEGRITY CHECKS
 

expected_rows = (
    len(
        roi_names
    )
    *
    TOTAL_SAMPLES_PER_ROI
)


if len(samples) != expected_rows:

    raise RuntimeError(
        f"Expected {expected_rows} sampled rows, "
        f"found {len(samples)}."
    )


counts = (
    samples
    .groupby(
        [
            "roi",
            "label"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


if not (
    counts[
        0
    ]
    == NEGATIVE_SAMPLES_PER_ROI
).all():

    raise RuntimeError(
        "Negative sample count mismatch."
    )


if not (
    counts[
        1
    ]
    == POSITIVE_SAMPLES_PER_ROI
).all():

    raise RuntimeError(
        "Positive sample count mismatch."
    )


 
# REPORT
 

report = [
    "PUMA COMMON RF TRAINING PIXEL SAMPLING",
    "=" * 65,
    "",

    f"Frozen cohort: {len(roi_names)} ROIs",
    f"Random seed: {SEED}",

    "",

    f"Samples per ROI: {TOTAL_SAMPLES_PER_ROI}",
    f"Nuclei pixels per ROI: {POSITIVE_SAMPLES_PER_ROI}",
    f"Background tissue pixels per ROI: {NEGATIVE_SAMPLES_PER_ROI}",

    "",

    f"Total stored pixel coordinates: {len(samples)}",

    "",

    (
        "The exact same sampled coordinates will be used "
        "for all five preprocessing methods."
    ),

    (
        "Negative pixels are sampled only from clean-reference "
        "tissue support."
    ),

    (
        "Sampling was performed before any segmentation-performance "
        "comparison."
    ),
]


(
    OUTDIR
    / "common_training_pixel_report.txt"
).write_text(
    "\n".join(
        report
    ),
    encoding="utf-8"
)


 
# TERMINAL OUTPUT
 

print(
    "\n"
    + "=" * 82
)

print(
    "COMMON TRAINING PIXEL GENERATION COMPLETE"
)

print(
    "=" * 82
)

print(
    "\nROIs:",
    len(
        roi_names
    )
)

print(
    "Samples per ROI:",
    TOTAL_SAMPLES_PER_ROI
)

print(
    "Total sampled coordinates:",
    len(
        samples
    )
)

print(
    "\nFold summary:"
)

print(
    fold_summary.to_string(
        index=False
    )
)

print(
    "\nPositive-pixel availability:"
)

print(
    "Minimum available:",
    int(
        summary[
            "available_positive_pixels"
        ].min()
    )
)

print(
    "Median available:",
    int(
        summary[
            "available_positive_pixels"
        ].median()
    )
)

print(
    "\nSaved samples to:"
)

print(
    OUTDIR
)

print(
    "\nQC figures saved to:"
)

print(
    QC_DIR
)

print(
    "=" * 82
)