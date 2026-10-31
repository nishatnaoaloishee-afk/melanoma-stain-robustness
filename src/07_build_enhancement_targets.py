from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color



# PATHS


ROOT = Path(__file__).resolve().parents[1]

ROI_DIR = (
    ROOT
    / "01_training_dataset_tif_ROIs"
)

SPLIT_FILE = (
    ROOT
    / "inputs"
    / "splits"
    / "cv_splits_seed13_folds5.npz"
)

OUTDIR = (
    ROOT
    / "outputs"
    / "enhancement_targets"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(
    0,
    str(ROOT / "src")
)

from adaptive_stain import (
    decompose_he,
    rgb_to_od,
)


print("=" * 80)
print("PUMA — BUILD TRAINING-ONLY ENHANCEMENT TARGETS")
print("=" * 80)



# LOAD FROZEN SPLITS


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



# SIMPLE RAW-IMAGE TISSUE SUPPORT

def make_tissue_mask(img_rgb_u8):
    """
    Training-only tissue support for colour-statistics estimation.
    """

    od = rgb_to_od(
        img_rgb_u8
    )

    mask = np.any(
        od > 0.05,
        axis=2
    )

    return mask.astype(bool)



# MEASURE EACH ROI ONCE

rows = []

for i, (roi_name, slide_type) in enumerate(
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

    if not img_path.exists():
        raise FileNotFoundError(
            img_path
        )

    img = np.array(
        Image.open(
            img_path
        ).convert("RGB"),
        dtype=np.uint8
    )

    tissue = make_tissue_mask(
        img
    )

    if tissue.sum() == 0:
        raise RuntimeError(
            f"No tissue pixels found for {roi_name}."
        )

    
    # Adaptive stain statistics for Macenko target


    dec = decompose_he(
        img
    )

    W = dec["stain_matrix"]

    H = dec["H"]
    E = dec["E"]

    H_p99 = float(
        np.percentile(
            H[tissue],
            99
        )
    )

    E_p99 = float(
        np.percentile(
            E[tissue],
            99
        )
    )


    # Lab statistics for Reinhard target


    rgb_float = (
        img.astype(np.float32)
        / 255.0
    )

    lab = color.rgb2lab(
        rgb_float
    )

    tissue_lab = lab[
        tissue
    ]

    lab_mean = (
        tissue_lab.mean(
            axis=0
        )
    )

    lab_std = (
        tissue_lab.std(
            axis=0,
            ddof=1
        )
    )

    rows.append(
        {
            "roi":
                roi_name,

            "slide_type":
                slide_type,

            "H_vector_R":
                float(W[0, 0]),

            "H_vector_G":
                float(W[0, 1]),

            "H_vector_B":
                float(W[0, 2]),

            "E_vector_R":
                float(W[1, 0]),

            "E_vector_G":
                float(W[1, 1]),

            "E_vector_B":
                float(W[1, 2]),

            "H_p99":
                H_p99,

            "E_p99":
                E_p99,

            "Lab_L_mean":
                float(lab_mean[0]),

            "Lab_a_mean":
                float(lab_mean[1]),

            "Lab_b_mean":
                float(lab_mean[2]),

            "Lab_L_std":
                float(lab_std[0]),

            "Lab_a_std":
                float(lab_std[1]),

            "Lab_b_std":
                float(lab_std[2]),
        }
    )


roi_stats = pd.DataFrame(
    rows
)

roi_stats.to_csv(
    OUTDIR
    / "enhancement_stats_per_roi.csv",
    index=False
)



# BUILD ONE TARGET PER FOLD FROM TRAINING ROIS ONLY


target_rows = []

for fold in range(
    1,
    6
):

    tr_idx = np.array(
        train_folds[
            fold - 1
        ],
        dtype=int
    )

    va_idx = np.array(
        val_folds[
            fold - 1
        ],
        dtype=int
    )

    train_names = [
        roi_names[i]
        for i in tr_idx
    ]

    train_df = roi_stats[
        roi_stats["roi"]
        .isin(
            train_names
        )
    ].copy()

    if len(train_df) != len(tr_idx):
        raise RuntimeError(
            f"Fold {fold}: training count mismatch."
        )

   
    # Macenko consensus stain vectors
   

    H_vector = np.array(
        [
            train_df["H_vector_R"].median(),
            train_df["H_vector_G"].median(),
            train_df["H_vector_B"].median(),
        ],
        dtype=np.float32
    )

    E_vector = np.array(
        [
            train_df["E_vector_R"].median(),
            train_df["E_vector_G"].median(),
            train_df["E_vector_B"].median(),
        ],
        dtype=np.float32
    )

    H_vector = (
        H_vector
        / (
            np.linalg.norm(
                H_vector
            )
            + 1e-8
        )
    )

    E_vector = (
        E_vector
        / (
            np.linalg.norm(
                E_vector
            )
            + 1e-8
        )
    )

    target_H_p99 = float(
        train_df[
            "H_p99"
        ].median()
    )

    target_E_p99 = float(
        train_df[
            "E_p99"
        ].median()
    )

  
    # Reinhard consensus Lab target


    target_L_mean = float(
        train_df[
            "Lab_L_mean"
        ].median()
    )

    target_a_mean = float(
        train_df[
            "Lab_a_mean"
        ].median()
    )

    target_b_mean = float(
        train_df[
            "Lab_b_mean"
        ].median()
    )

    target_L_std = float(
        train_df[
            "Lab_L_std"
        ].median()
    )

    target_a_std = float(
        train_df[
            "Lab_a_std"
        ].median()
    )

    target_b_std = float(
        train_df[
            "Lab_b_std"
        ].median()
    )

    target_rows.append(
        {
            "fold":
                fold,

            "n_train":
                len(tr_idx),

            "n_validation":
                len(va_idx),

            "Mac_H_R":
                float(H_vector[0]),

            "Mac_H_G":
                float(H_vector[1]),

            "Mac_H_B":
                float(H_vector[2]),

            "Mac_E_R":
                float(E_vector[0]),

            "Mac_E_G":
                float(E_vector[1]),

            "Mac_E_B":
                float(E_vector[2]),

            "Mac_H_p99":
                target_H_p99,

            "Mac_E_p99":
                target_E_p99,

            "Rein_L_mean":
                target_L_mean,

            "Rein_a_mean":
                target_a_mean,

            "Rein_b_mean":
                target_b_mean,

            "Rein_L_std":
                target_L_std,

            "Rein_a_std":
                target_a_std,

            "Rein_b_std":
                target_b_std,
        }
    )


targets = pd.DataFrame(
    target_rows
)

targets.to_csv(
    OUTDIR
    / "fold_enhancement_targets.csv",
    index=False
)



# SAVE ONE JSON PER FOLD


for _, row in targets.iterrows():

    fold = int(
        row["fold"]
    )

    cfg = {
        "fold":
            fold,

        "n_train":
            int(
                row["n_train"]
            ),

        "macenko": {
            "stain_matrix": [
                [
                    float(
                        row["Mac_H_R"]
                    ),
                    float(
                        row["Mac_H_G"]
                    ),
                    float(
                        row["Mac_H_B"]
                    ),
                ],
                [
                    float(
                        row["Mac_E_R"]
                    ),
                    float(
                        row["Mac_E_G"]
                    ),
                    float(
                        row["Mac_E_B"]
                    ),
                ],
            ],

            "target_H_p99":
                float(
                    row[
                        "Mac_H_p99"
                    ]
                ),

            "target_E_p99":
                float(
                    row[
                        "Mac_E_p99"
                    ]
                ),
        },

        "reinhard": {
            "mean": [
                float(
                    row[
                        "Rein_L_mean"
                    ]
                ),
                float(
                    row[
                        "Rein_a_mean"
                    ]
                ),
                float(
                    row[
                        "Rein_b_mean"
                    ]
                ),
            ],

            "std": [
                float(
                    row[
                        "Rein_L_std"
                    ]
                ),
                float(
                    row[
                        "Rein_a_std"
                    ]
                ),
                float(
                    row[
                        "Rein_b_std"
                    ]
                ),
            ],
        },

        "clahe": {
            "clip_limit":
                0.5,

            "tile_grid_size": [
                8,
                8
            ],
        },
    }

    with open(
        OUTDIR
        / f"fold{fold:02d}_enhancement_target.json",
        "w"
    ) as f:

        json.dump(
            cfg,
            f,
            indent=2
        )



# FOLD-STABILITY SUMMARY


summary_rows = []

columns_to_check = [
    "Mac_H_p99",
    "Mac_E_p99",

    "Rein_L_mean",
    "Rein_a_mean",
    "Rein_b_mean",

    "Rein_L_std",
    "Rein_a_std",
    "Rein_b_std",
]

for col in columns_to_check:

    vals = targets[
        col
    ].astype(float)

    summary_rows.append(
        {
            "parameter":
                col,

            "mean":
                float(
                    vals.mean()
                ),

            "std":
                float(
                    vals.std(
                        ddof=1
                    )
                ),

            "min":
                float(
                    vals.min()
                ),

            "max":
                float(
                    vals.max()
                ),
        }
    )


stability = pd.DataFrame(
    summary_rows
)

stability.to_csv(
    OUTDIR
    / "enhancement_target_stability.csv",
    index=False
)



# REPORT


report = [
    "PUMA TRAINING-ONLY ENHANCEMENT TARGETS",
    "=" * 65,
    "",

    "Frozen cohort: 205 ROIs",
    "Cross-validation folds: 5",
    "",

    "For each fold:",
    "- Macenko target derived from training ROIs only.",
    "- Reinhard Lab target derived from training ROIs only.",
    "- Validation ROIs never contribute target statistics.",
    "",

    "CLAHE settings:",
    "- clip limit = 2.0",
    "- tile grid = 8 x 8",
]

(
    OUTDIR
    / "enhancement_target_report.txt"
).write_text(
    "\n".join(
        report
    ),
    encoding="utf-8"
)



# TERMINAL OUTPUT


print(
    "\n"
    + "=" * 80
)

print(
    "ENHANCEMENT TARGET GENERATION COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nFold targets:"
)

display_cols = [
    "fold",
    "Mac_H_p99",
    "Mac_E_p99",
    "Rein_L_mean",
    "Rein_a_mean",
    "Rein_b_mean",
]

print(
    targets[
        display_cols
    ].to_string(
        index=False
    )
)

print(
    "\nSaved to:"
)

print(
    OUTDIR
)

print(
    "=" * 80
)