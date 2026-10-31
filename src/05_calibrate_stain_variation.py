from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image



# PATHS

ROOT = Path(__file__).resolve().parents[1]

ROI_DIR = ROOT / "01_training_dataset_tif_ROIs"

SPLIT_FILE = (
    ROOT
    / "inputs"
    / "splits"
    / "cv_splits_seed13_folds5.npz"
)

OUTDIR = (
    ROOT
    / "outputs"
    / "stain_calibration"
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
)


print("=" * 78)
print("PUMA — TRAINING-DERIVED H&E STAIN CALIBRATION")
print("=" * 78)



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
        f"Expected frozen cohort of 205 ROIs, found {len(roi_names)}."
    )



# PART 1 — MEASURE EACH ROI ONCE

rows = []

for i, (roi_name, slide_type) in enumerate(
    zip(roi_names, slide_types),
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

    try:

        dec = decompose_he(
            img
        )

        H = dec["H"]
        E = dec["E"]

        # Use robust high-percentile concentration as
        # stain-strength descriptor.
        H_p99 = float(
            np.percentile(
                H,
                99
            )
        )

        E_p99 = float(
            np.percentile(
                E,
                99
            )
        )

        H_p95 = float(
            np.percentile(
                H,
                95
            )
        )

        E_p95 = float(
            np.percentile(
                E,
                95
            )
        )

        W = dec[
            "stain_matrix"
        ]

        rows.append(
            {
                "roi":
                    roi_name,

                "slide_type":
                    slide_type,

                "status":
                    "OK",

                "H_p95":
                    H_p95,

                "H_p99":
                    H_p99,

                "E_p95":
                    E_p95,

                "E_p99":
                    E_p99,

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
            }
        )

    except Exception as e:

        rows.append(
            {
                "roi":
                    roi_name,

                "slide_type":
                    slide_type,

                "status":
                    "FAILED",

                "error":
                    str(e),
            }
        )

        print(
            "   FAILED:",
            str(e)
        )


stats = pd.DataFrame(
    rows
)

stats.to_csv(
    OUTDIR
    / "stain_stats_per_roi.csv",
    index=False
)



# CHECK FAILURES

failed = stats[
    stats["status"]
    != "OK"
]

print(
    "\nSuccessful decompositions:",
    int(
        (
            stats["status"]
            == "OK"
        ).sum()
    )
)

print(
    "Failed decompositions:",
    len(failed)
)

if len(failed) > 0:

    failed.to_csv(
        OUTDIR
        / "failed_rois.csv",
        index=False
    )

    raise RuntimeError(
        "One or more ROIs failed adaptive stain decomposition. "
        "Inspect failed_rois.csv before continuing."
    )



# PART 2 — FOLD-SPECIFIC TRAINING CALIBRATION

fold_rows = []
parameter_rows = []

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

    train_df = stats[
        stats["roi"]
        .isin(
            train_names
        )
    ].copy()

    if len(train_df) != len(tr_idx):
        raise RuntimeError(
            f"Fold {fold}: training ROI count mismatch."
        )

    H_values = (
        train_df["H_p99"]
        .astype(float)
        .to_numpy()
    )

    E_values = (
        train_df["E_p99"]
        .astype(float)
        .to_numpy()
    )


    # Robust training-only distribution landmarks

    H_q10, H_q25, H_q50, H_q75, H_q90 = (
        np.percentile(
            H_values,
            [10, 25, 50, 75, 90]
        )
    )

    E_q10, E_q25, E_q50, E_q75, E_q90 = (
        np.percentile(
            E_values,
            [10, 25, 50, 75, 90]
        )
    )

    fold_rows.append(
        {
            "fold":
                fold,

            "n_train":
                len(tr_idx),

            "n_validation":
                len(va_idx),

            "H_q10":
                H_q10,

            "H_q25":
                H_q25,

            "H_q50":
                H_q50,

            "H_q75":
                H_q75,

            "H_q90":
                H_q90,

            "E_q10":
                E_q10,

            "E_q25":
                E_q25,

            "E_q50":
                E_q50,

            "E_q75":
                E_q75,

            "E_q90":
                E_q90,
        }
    )


    # Convert observed distribution into scaling
    # factors relative to the training median.

    H_low_severe = (
        H_q10
        / H_q50
    )

    H_low_mild = (
        H_q25
        / H_q50
    )

    H_high_mild = (
        H_q75
        / H_q50
    )

    H_high_severe = (
        H_q90
        / H_q50
    )

    E_low_severe = (
        E_q10
        / E_q50
    )

    E_low_mild = (
        E_q25
        / E_q50
    )

    E_high_mild = (
        E_q75
        / E_q50
    )

    E_high_severe = (
        E_q90
        / E_q50
    )

   
    # Sanity checks

    expected_H = (
        H_low_severe
        < H_low_mild
        < 1.0
        < H_high_mild
        < H_high_severe
    )

    expected_E = (
        E_low_severe
        < E_low_mild
        < 1.0
        < E_high_mild
        < E_high_severe
    )

    if not expected_H:
        raise RuntimeError(
            f"Unexpected H calibration ordering in fold {fold}."
        )

    if not expected_E:
        raise RuntimeError(
            f"Unexpected E calibration ordering in fold {fold}."
        )

    parameter_rows.extend(
        [
            {
                "fold": fold,
                "condition": "H_low_mild",
                "alpha_H": H_low_mild,
                "alpha_E": 1.0,
            },

            {
                "fold": fold,
                "condition": "H_low_severe",
                "alpha_H": H_low_severe,
                "alpha_E": 1.0,
            },

            {
                "fold": fold,
                "condition": "H_high_mild",
                "alpha_H": H_high_mild,
                "alpha_E": 1.0,
            },

            {
                "fold": fold,
                "condition": "H_high_severe",
                "alpha_H": H_high_severe,
                "alpha_E": 1.0,
            },

            {
                "fold": fold,
                "condition": "E_low_mild",
                "alpha_H": 1.0,
                "alpha_E": E_low_mild,
            },

            {
                "fold": fold,
                "condition": "E_low_severe",
                "alpha_H": 1.0,
                "alpha_E": E_low_severe,
            },

            {
                "fold": fold,
                "condition": "E_high_mild",
                "alpha_H": 1.0,
                "alpha_E": E_high_mild,
            },

            {
                "fold": fold,
                "condition": "E_high_severe",
                "alpha_H": 1.0,
                "alpha_E": E_high_severe,
            },

            {
                "fold": fold,
                "condition": "HE_low_mild",
                "alpha_H": H_low_mild,
                "alpha_E": E_low_mild,
            },

            {
                "fold": fold,
                "condition": "HE_low_severe",
                "alpha_H": H_low_severe,
                "alpha_E": E_low_severe,
            },

            {
                "fold": fold,
                "condition": "HE_high_mild",
                "alpha_H": H_high_mild,
                "alpha_E": E_high_mild,
            },

            {
                "fold": fold,
                "condition": "HE_high_severe",
                "alpha_H": H_high_severe,
                "alpha_E": E_high_severe,
            },
        ]
    )


fold_stats = pd.DataFrame(
    fold_rows
)

parameters = pd.DataFrame(
    parameter_rows
)

fold_stats.to_csv(
    OUTDIR
    / "fold_stain_statistics.csv",
    index=False
)

parameters.to_csv(
    OUTDIR
    / "fold_perturbation_parameters.csv",
    index=False
)



# PART 3 — SAVE HUMAN-READABLE PARAMETER TABLE

parameter_pivot = (
    parameters
    .pivot(
        index="condition",
        columns="fold",
        values=[
            "alpha_H",
            "alpha_E",
        ]
    )
)

parameter_pivot.to_csv(
    OUTDIR
    / "fold_perturbation_parameters_wide.csv"
)



# PART 4 — DISTRIBUTION FIGURES

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.hist(
    stats["H_p99"],
    bins=25
)

ax.set_xlabel(
    "Hematoxylin concentration strength (P99)"
)

ax.set_ylabel(
    "Number of ROIs"
)

ax.set_title(
    "Distribution of hematoxylin stain strength"
)

plt.tight_layout()

plt.savefig(
    OUTDIR
    / "hematoxylin_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTDIR
    / "hematoxylin_distribution.pdf",
    bbox_inches="tight"
)

plt.close()


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.hist(
    stats["E_p99"],
    bins=25
)

ax.set_xlabel(
    "Eosin concentration strength (P99)"
)

ax.set_ylabel(
    "Number of ROIs"
)

ax.set_title(
    "Distribution of eosin stain strength"
)

plt.tight_layout()

plt.savefig(
    OUTDIR
    / "eosin_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTDIR
    / "eosin_distribution.pdf",
    bbox_inches="tight"
)

plt.close()



# PART 5 — PERTURBATION FACTOR FIGURE


condition_order = [
    "H_low_severe",
    "H_low_mild",
    "H_high_mild",
    "H_high_severe",
    "E_low_severe",
    "E_low_mild",
    "E_high_mild",
    "E_high_severe",
]

plot_rows = []

for condition in condition_order:

    sub = parameters[
        parameters["condition"]
        == condition
    ]

    if condition.startswith("H_"):
        vals = sub["alpha_H"]

    else:
        vals = sub["alpha_E"]

    plot_rows.append(
        {
            "condition":
                condition,

            "mean_factor":
                float(vals.mean()),

            "min_factor":
                float(vals.min()),

            "max_factor":
                float(vals.max()),
        }
    )

plot_df = pd.DataFrame(
    plot_rows
)

x = np.arange(
    len(plot_df)
)

lower = (
    plot_df["mean_factor"]
    -
    plot_df["min_factor"]
)

upper = (
    plot_df["max_factor"]
    -
    plot_df["mean_factor"]
)

fig, ax = plt.subplots(
    figsize=(10, 5.5)
)

ax.errorbar(
    x,
    plot_df["mean_factor"],
    yerr=np.vstack(
        [
            lower,
            upper,
        ]
    ),
    fmt="o",
    capsize=4
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    plot_df["condition"],
    rotation=45,
    ha="right"
)

ax.set_ylabel(
    "Training-derived concentration scaling factor"
)

ax.set_xlabel(
    "Perturbation condition"
)

ax.set_title(
    "Fold-specific stain perturbation calibration"
)

plt.tight_layout()

plt.savefig(
    OUTDIR
    / "perturbation_factor_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTDIR
    / "perturbation_factor_summary.pdf",
    bbox_inches="tight"
)

plt.close()



# REPORT


report = [
    "PUMA TRAINING-DERIVED STAIN CALIBRATION",
    "=" * 65,
    "",
    f"Frozen cohort: {len(roi_names)} ROIs",
    f"Successful adaptive decompositions: {len(stats)}",
    "Cross-validation folds: 5",
    "",
    (
        "Perturbation levels were derived independently "
        "from the training ROIs of each fold."
    ),
    "",
    (
        "Mild low  = training Q25 / median"
    ),
    (
        "Severe low = training Q10 / median"
    ),
    (
        "Mild high = training Q75 / median"
    ),
    (
        "Severe high = training Q90 / median"
    ),
    "",
    (
        "No segmentation performance was used "
        "to select perturbation magnitude."
    ),
]

(
    OUTDIR
    / "calibration_report.txt"
).write_text(
    "\n".join(report),
    encoding="utf-8"
)



# TERMINAL SUMMARY

print("\n" + "=" * 78)
print("STAIN CALIBRATION COMPLETE")
print("=" * 78)

print(
    "\nPer-fold perturbation factors:"
)

for fold in range(
    1,
    6
):

    print(
        f"\nFold {fold}"
    )

    sub = parameters[
        parameters["fold"]
        == fold
    ]

    for _, row in sub.iterrows():

        print(
            f"  {row['condition']:16s} "
            f"H={row['alpha_H']:.4f} "
            f"E={row['alpha_E']:.4f}"
        )


print(
    "\nSaved to:"
)

print(
    OUTDIR
)

print("=" * 78)