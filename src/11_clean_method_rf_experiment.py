from pathlib import Path
import sys
import argparse
import json
import time
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.ensemble import (
    RandomForestClassifier
)


  
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

SAMPLE_FILE = (
    ROOT
    / "inputs"
    / "training_samples"
    / "common_training_pixels_seed13.csv"
)

TARGET_DIR = (
    ROOT
    / "outputs"
    / "enhancement_targets"
)

BASE_OUTDIR = (
    ROOT
    / "outputs"
    / "clean_rf"
)


  
# MAKE PROJECT MODULES IMPORTABLE
  

SRC_DIR = (
    ROOT
    / "src"
)

LEGACY_DIR = (
    ROOT
    / "legacy_code"
)

sys.path.insert(
    0,
    str(SRC_DIR)
)

sys.path.insert(
    0,
    str(LEGACY_DIR)
)


  
# IMPORT FINAL CONFERENCE PREPROCESSING
  

from conference_preprocess import (
    preprocess_for_method,
)


from enhancement_methods import (
    METHODS,
    DISPLAY_NAMES,
    load_fold_target,
)


  
# IMPORT EXACT LEGACY RF FUNCTIONS
  

from puma_nuclei_rf_only_cv5_split import (
    compute_feature_stack,
    binary_to_instances,
    rasterize_geojson_aligned,
    balanced_accuracy_from_counts,
)

from hybridA_utils import (
    compute_pixel_metrics,
    make_overlay,
)


  
# LOCKED RF SETTINGS
  

SEED = 13

N_ESTIMATORS = 500

MAX_DEPTH = None

MIN_SAMPLES_LEAF = 1

PROB_THRESHOLD = 0.40

PREDICT_CHUNK = 250000

MIN_AREA = 5

MAX_AREA = 1200

MIN_PEAK_DISTANCE = 2

DIST_SIGMA = 1.0

N_JOBS = -1

OVERLAYS_PER_FOLD = 2


  
# ARGS
  

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--method",
        required=True,
        choices=METHODS
    )

    ap.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help=(
            "1 = rerun completed folds; "
            "0 = skip completed folds"
        )
    )

    return ap.parse_args()


  
# MAIN
  

def main():

    args = parse_args()

    method = args.method

    method_display = (
        DISPLAY_NAMES[
            method
        ]
    )

    OUTDIR = (
        BASE_OUTDIR
        / method
    )

    FOLD_DIR = (
        OUTDIR
        / "folds"
    )

    OVERLAY_DIR = (
        OUTDIR
        / "overlays"
    )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FOLD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OVERLAY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "=" * 84
    )

    print(
        "PUMA — CLEAN METHOD-MATCHED RF EXPERIMENT"
    )

    print(
        "=" * 84
    )

    print(
        "\nMethod:",
        method_display
    )

    print(
        "RF models will NOT be saved to disk."
    )


    # ========================================================
    # LOAD FROZEN CV SPLIT
    # ========================================================

    sp = np.load(
        SPLIT_FILE,
        allow_pickle=True
    )

    roi_names = list(
        sp[
            "roi_names"
        ].tolist()
    )

    slide_types = list(
        sp[
            "slide_types"
        ].tolist()
    )

    train_folds = (
        sp[
            "tr"
        ].tolist()
    )

    val_folds = (
        sp[
            "va"
        ].tolist()
    )


    if len(
        roi_names
    ) != 205:

        raise RuntimeError(
            f"Expected 205 ROIs, "
            f"found {len(roi_names)}."
        )


    # ========================================================
    # LOAD COMMON TRAINING PIXELS
    # ========================================================

    print(
        "\nLoading common training pixels..."
    )

    sample_df = pd.read_csv(
        SAMPLE_FILE,
        dtype={
            "roi":
                str,

            "slide_type":
                str,

            "y":
                np.int16,

            "x":
                np.int16,

            "label":
                np.uint8,
        }
    )


    if len(
        sample_df
    ) != 1230000:

        raise RuntimeError(
            "Common training sample count is not 1,230,000."
        )


    sample_lookup = {}

    for roi, group in sample_df.groupby(
        "roi",
        sort=False
    ):

        sample_lookup[
            roi
        ] = {
            "y":
                group[
                    "y"
                ].to_numpy(
                    dtype=np.int16
                ),

            "x":
                group[
                    "x"
                ].to_numpy(
                    dtype=np.int16
                ),

            "label":
                group[
                    "label"
                ].to_numpy(
                    dtype=np.uint8
                ),
        }


    if len(
        sample_lookup
    ) != 205:

        raise RuntimeError(
            "Common sample ROI count mismatch."
        )


    print(
        "Common coordinate set loaded:",
        len(sample_df),
        "pixels"
    )


    # ========================================================
    # VERIFY CLAHE IS FROZEN AT 0.5
    # ========================================================

    for fold in range(
        1,
        6
    ):

        cfg = load_fold_target(
            TARGET_DIR,
            fold
        )

        clip_limit = float(
            cfg[
                "clahe"
            ][
                "clip_limit"
            ]
        )

        if abs(
            clip_limit
            - 0.5
        ) > 1e-9:

            raise RuntimeError(
                f"Fold {fold} CLAHE clip limit is "
                f"{clip_limit}; expected frozen value 0.5."
            )


    print(
        "Frozen CLAHE configuration check: PASS"
    )


    # ========================================================
    # RUN CONFIG
    # ========================================================

    run_config = {
        "method":
            method,

        "method_display":
            method_display,

        "cohort_size":
            205,

        "folds":
            5,

        "common_samples_per_roi":
            6000,

        "positive_samples_per_roi":
            3000,

        "negative_samples_per_roi":
            3000,

        "random_seed":
            SEED,

        "n_estimators":
            N_ESTIMATORS,

        "max_depth":
            MAX_DEPTH,

        "min_samples_leaf":
            MIN_SAMPLES_LEAF,

        "class_weight":
            "balanced",

        "probability_threshold":
            PROB_THRESHOLD,

        "predict_chunk":
            PREDICT_CHUNK,

        "min_area":
            MIN_AREA,

        "max_area":
            MAX_AREA,

        "min_peak_distance":
            MIN_PEAK_DISTANCE,

        "distance_sigma":
            DIST_SIGMA,

        "model_files_saved":
            False,

        "clahe_clip_limit":
            0.5,

        "clahe_tile_grid":
            [
                8,
                8
            ],
    }


    with open(
        OUTDIR
        / "run_config.json",
        "w"
    ) as f:

        json.dump(
            run_config,
            f,
            indent=2
        )


    # ========================================================
    # 5 FOLDS
    # ========================================================

    for fold in range(
        1,
        6
    ):

        fold_metrics_path = (
            FOLD_DIR
            / f"fold{fold:02d}_metrics.csv"
        )

        fold_importance_path = (
            FOLD_DIR
            / f"fold{fold:02d}_feature_importance.csv"
        )


        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if (
            fold_metrics_path.exists()
            and
            fold_importance_path.exists()
            and
            not bool(
                args.overwrite
            )
        ):

            print(
                f"\nFold {fold} already complete — SKIPPING."
            )

            continue


        print(
            "\n"
            + "=" * 84
        )

        print(
            f"FOLD {fold}/5 — {method_display}"
        )

        print(
            "=" * 84
        )


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


        print(
            "Train:",
            len(
                tr_idx
            ),
            "| Validation:",
            len(
                va_idx
            )
        )


        cfg = load_fold_target(
            TARGET_DIR,
            fold
        )


        # ====================================================
        # BUILD TRAINING MATRIX
        # ====================================================

        X_list = []

        y_list = []

        feature_names = None


        train_start = time.perf_counter()


        for j, ii in enumerate(
            tr_idx,
            start=1
        ):

            roi = (
                roi_names[
                    ii
                ]
            )

            img_path = (
                ROI_DIR
                / f"{roi}.tif"
            )


            raw = np.array(
                Image.open(
                    img_path
                ).convert(
                    "RGB"
                ),
                dtype=np.uint8
            )


            img_20x, _tissue = (
                preprocess_for_method(
                    raw,
                    method,
                    cfg
                )
            )


            fnames, feature_stack = (
                compute_feature_stack(
                    img_20x
                )
            )


            if feature_names is None:

                feature_names = list(
                    fnames
                )

            elif (
                list(
                    fnames
                )
                != feature_names
            ):

                raise RuntimeError(
                    "Feature-name mismatch."
                )


            samples = (
                sample_lookup[
                    roi
                ]
            )

            yy = samples[
                "y"
            ].astype(
                np.intp
            )

            xx = samples[
                "x"
            ].astype(
                np.intp
            )

            labels = samples[
                "label"
            ]


            # Coordinate integrity check
            if (
                yy.max()
                >= feature_stack.shape[0]
                or
                xx.max()
                >= feature_stack.shape[1]
            ):

                raise RuntimeError(
                    f"Training sample coordinates outside "
                    f"feature map for {roi}."
                )


            X_roi = (
                feature_stack[
                    yy,
                    xx,
                    :
                ]
                .astype(
                    np.float32
                )
            )


            X_list.append(
                X_roi
            )

            y_list.append(
                labels
            )


            if (
                j % 20
                == 0
                or
                j
                == len(
                    tr_idx
                )
            ):

                print(
                    f"  Training features: "
                    f"{j}/{len(tr_idx)} ROIs"
                )


            del (
                raw,
                img_20x,
                feature_stack,
                X_roi
            )


        X_train = np.concatenate(
            X_list,
            axis=0
        )

        y_train = np.concatenate(
            y_list,
            axis=0
        )


        del X_list
        del y_list

        gc.collect()


        print(
            "\nTraining matrix:",
            X_train.shape
        )

        print(
            "Positive:",
            int(
                (
                    y_train
                    == 1
                ).sum()
            ),
            "| Negative:",
            int(
                (
                    y_train
                    == 0
                ).sum()
            )
        )


        if X_train.shape != (
            984000,
            9
        ):

            raise RuntimeError(
                f"Unexpected training matrix shape: "
                f"{X_train.shape}"
            )


        if int(
            (
                y_train
                == 1
            ).sum()
        ) != 492000:

            raise RuntimeError(
                "Unexpected positive training count."
            )


        if int(
            (
                y_train
                == 0
            ).sum()
        ) != 492000:

            raise RuntimeError(
                "Unexpected negative training count."
            )


        feature_time = (
            time.perf_counter()
            - train_start
        )


        # ====================================================
        # TRAIN RF
        # ====================================================

        print(
            "\nTraining Random Forest..."
        )

        fit_start = (
            time.perf_counter()
        )


        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            n_jobs=N_JOBS,
            class_weight="balanced",
            random_state=SEED,
        )


        rf.fit(
            X_train,
            y_train
        )


        fit_time = (
            time.perf_counter()
            - fit_start
        )


        print(
            "RF fit complete."
        )

        print(
            "Feature-building time:",
            f"{feature_time:.1f} s"
        )

        print(
            "RF fit time:",
            f"{fit_time:.1f} s"
        )


        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        importance = pd.DataFrame(
            {
                "fold":
                    fold,

                "method":
                    method,

                "method_display":
                    method_display,

                "feature":
                    feature_names,

                "importance":
                    rf.feature_importances_,
            }
        )


        importance.to_csv(
            fold_importance_path,
            index=False
        )


        # Training matrix no longer needed
        del X_train
        del y_train

        gc.collect()


        # ====================================================
        # CLASS = 1 COLUMN
        # ====================================================

        classes = list(
            rf.classes_
        )

        if 1 not in classes:

            raise RuntimeError(
                "RF does not contain nuclei class 1."
            )


        pos_col = (
            classes.index(
                1
            )
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        validation_rows = []

        overlays_saved = 0


        for k, ii in enumerate(
            va_idx,
            start=1
        ):

            roi = (
                roi_names[
                    ii
                ]
            )

            slide_type = (
                slide_types[
                    ii
                ]
            )


            roi_path = (
                ROI_DIR
                / f"{roi}.tif"
            )

            gt_path = (
                GT_DIR
                / f"{roi}_nuclei.geojson"
            )


            # ------------------------------------------------
            # READ
            # ------------------------------------------------

            t0 = time.perf_counter()


            raw = np.array(
                Image.open(
                    roi_path
                ).convert(
                    "RGB"
                ),
                dtype=np.uint8
            )


            raw_H, raw_W = (
                raw.shape[:2]
            )


            t1 = time.perf_counter()


            # ------------------------------------------------
            # METHOD-MATCHED PREPROCESSING
            # ------------------------------------------------

            img_20x, tissue_mask = (
                preprocess_for_method(
                    raw,
                    method,
                    cfg
                )
            )


            t2 = time.perf_counter()


            # ------------------------------------------------
            # GROUND TRUTH
            # ------------------------------------------------

            gt_bin = (
                rasterize_geojson_aligned(
                    str(
                        gt_path
                    ),
                    raw_H,
                    raw_W,
                    to_20x=True
                )
            )


            if (
                gt_bin.shape
                != tissue_mask.shape
            ):

                raise RuntimeError(
                    f"GT/tissue shape mismatch: {roi}"
                )


            t3 = time.perf_counter()


            # ------------------------------------------------
            # SEGMENTATION
            # ------------------------------------------------

            tseg0 = (
                time.perf_counter()
            )


            fnames, feature_stack = (
                compute_feature_stack(
                    img_20x
                )
            )


            if list(
                fnames
            ) != feature_names:

                raise RuntimeError(
                    "Validation feature-name mismatch."
                )


            H, W, F = (
                feature_stack.shape
            )


            X_all = (
                feature_stack
                .reshape(
                    -1,
                    F
                )
            )


            tissue_flat = (
                tissue_mask
                .astype(
                    bool
                )
                .reshape(
                    -1
                )
            )


            tissue_idx = np.flatnonzero(
                tissue_flat
            )


            probability = np.zeros(
                H * W,
                dtype=np.float32
            )


            for start in range(
                0,
                len(
                    tissue_idx
                ),
                PREDICT_CHUNK
            ):

                subset = (
                    tissue_idx[
                        start:
                        start
                        +
                        PREDICT_CHUNK
                    ]
                )


                p = (
                    rf
                    .predict_proba(
                        X_all[
                            subset
                        ]
                    )[
                        :,
                        pos_col
                    ]
                    .astype(
                        np.float32
                    )
                )


                probability[
                    subset
                ] = p


            probability_map = (
                probability
                .reshape(
                    H,
                    W
                )
            )


            pred_binary = (
                (
                    probability_map
                    >= PROB_THRESHOLD
                )
                &
                tissue_mask
            )


            instances, pred_final = (
                binary_to_instances(
                    pred_binary,
                    tissue_mask,

                    min_area=
                        MIN_AREA,

                    max_area=
                        MAX_AREA,

                    min_peak_distance=
                        MIN_PEAK_DISTANCE,

                    dist_sigma=
                        DIST_SIGMA,
                )
            )


            tseg1 = (
                time.perf_counter()
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            metrics = (
                compute_pixel_metrics(
                    pred_final,
                    gt_bin,
                    tissue_mask
                )
            )


            balanced_accuracy = (
                balanced_accuracy_from_counts(
                    metrics[
                        "tp"
                    ],
                    metrics[
                        "fp"
                    ],
                    metrics[
                        "fn"
                    ],
                    metrics[
                        "tn"
                    ],
                )
            )


            validation_rows.append(
                {
                    "method":
                        method,

                    "method_display":
                        method_display,

                    "fold":
                        fold,

                    "roi":
                        roi,

                    "slide_type":
                        slide_type,

                    "tissue_pixels":
                        int(
                            tissue_mask.sum()
                        ),

                    "gt_pos_pixels":
                        int(
                            gt_bin[
                                tissue_mask
                            ].sum()
                        ),

                    "pred_pos_pixels":
                        int(
                            pred_final[
                                tissue_mask
                            ].sum()
                        ),

                    "n_pred_instances":
                        int(
                            instances.max()
                        ),

                    "tp":
                        int(
                            metrics[
                                "tp"
                            ]
                        ),

                    "fp":
                        int(
                            metrics[
                                "fp"
                            ]
                        ),

                    "fn":
                        int(
                            metrics[
                                "fn"
                            ]
                        ),

                    "tn":
                        int(
                            metrics[
                                "tn"
                            ]
                        ),

                    "precision":
                        float(
                            metrics[
                                "precision"
                            ]
                        ),

                    "recall":
                        float(
                            metrics[
                                "recall"
                            ]
                        ),

                    "f1":
                        float(
                            metrics[
                                "f1"
                            ]
                        ),

                    "dice":
                        float(
                            metrics[
                                "dice"
                            ]
                        ),

                    "iou":
                        float(
                            metrics[
                                "iou"
                            ]
                        ),

                    "accuracy":
                        float(
                            metrics[
                                "accuracy"
                            ]
                        ),

                    "specificity":
                        float(
                            metrics[
                                "specificity"
                            ]
                        ),

                    "balanced_accuracy":
                        float(
                            balanced_accuracy
                        ),

                    "time_read_s":
                        float(
                            t1
                            -
                            t0
                        ),

                    "time_preprocess_s":
                        float(
                            t2
                            -
                            t1
                        ),

                    "time_gt_s":
                        float(
                            t3
                            -
                            t2
                        ),

                    "time_segment_s":
                        float(
                            tseg1
                            -
                            tseg0
                        ),

                    "time_total_s":
                        float(
                            tseg1
                            -
                            t0
                        ),
                }
            )


            # ------------------------------------------------
            # SMALL OVERLAY SAMPLE
            # ------------------------------------------------

            if (
                overlays_saved
                <
                OVERLAYS_PER_FOLD
            ):

                overlay = make_overlay(
                    img_20x,
                    gt_bin,
                    pred_final
                )


                overlay_path = (
                    OVERLAY_DIR
                    /
                    (
                        f"fold{fold:02d}_"
                        f"{roi}_overlay.png"
                    )
                )


                Image.fromarray(
                    overlay
                ).save(
                    overlay_path
                )


                overlays_saved += 1


            print(
                f"  Validation "
                f"{k:02d}/{len(va_idx)} | "
                f"{roi} | "
                f"Dice={metrics['dice']:.4f}"
            )


            del (
                raw,
                img_20x,
                gt_bin,
                feature_stack,
                X_all,
                probability,
                probability_map,
                pred_binary,
                instances,
                pred_final
            )


        # ====================================================
        # SAVE FOLD RESULTS IMMEDIATELY
        # ====================================================

        fold_df = pd.DataFrame(
            validation_rows
        )


        fold_df.to_csv(
            fold_metrics_path,
            index=False
        )


        print(
            "\nFold",
            fold,
            "mean Dice:",
            f"{fold_df['dice'].mean():.6f}"
        )


        fold_runtime = {
            "fold":
                fold,

            "method":
                method,

            "feature_building_seconds":
                float(
                    feature_time
                ),

            "rf_fit_seconds":
                float(
                    fit_time
                ),

            "validation_mean_preprocess_seconds":
                float(
                    fold_df[
                        "time_preprocess_s"
                    ].mean()
                ),

            "validation_mean_segment_seconds":
                float(
                    fold_df[
                        "time_segment_s"
                    ].mean()
                ),

            "validation_mean_total_seconds":
                float(
                    fold_df[
                        "time_total_s"
                    ].mean()
                ),
        }


        with open(
            FOLD_DIR
            /
            f"fold{fold:02d}_runtime.json",
            "w"
        ) as f:

            json.dump(
                fold_runtime,
                f,
                indent=2
            )


        # ====================================================
        # DISCARD HUGE RF
        # ====================================================

        del rf

        gc.collect()


        print(
            "RF discarded from memory."
        )

        print(
            "No model file was written."
        )


    # ========================================================
    # COMBINE ALL COMPLETED FOLDS
    # ========================================================

    fold_metric_files = [
        FOLD_DIR
        / f"fold{fold:02d}_metrics.csv"

        for fold in range(
            1,
            6
        )
    ]


    if not all(
        p.exists()
        for p in fold_metric_files
    ):

        print(
            "\nNot all 5 folds are complete yet."
        )

        print(
            "Run the same command again to resume."
        )

        return


    all_metrics = pd.concat(
        [
            pd.read_csv(
                p
            )
            for p in fold_metric_files
        ],
        ignore_index=True
    )


    if len(
        all_metrics
    ) != 205:

        raise RuntimeError(
            f"Expected 205 validation rows, "
            f"found {len(all_metrics)}."
        )


    all_metrics.to_csv(
        OUTDIR
        / "clean_metrics_per_roi.csv",
        index=False
    )


    # ========================================================
    # FOLD SUMMARY
    # ========================================================

    metrics_to_summarize = [
        "precision",
        "recall",
        "f1",
        "dice",
        "iou",
        "accuracy",
        "specificity",
        "balanced_accuracy",
        "time_preprocess_s",
        "time_segment_s",
        "time_total_s",
    ]


    fold_summary = (
        all_metrics
        .groupby(
            "fold"
        )[
            metrics_to_summarize
        ]
        .agg(
            [
                "mean",
                "std"
            ]
        )
    )


    fold_summary.to_csv(
        OUTDIR
        / "clean_summary_by_fold.csv"
    )


    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    summary_rows = []


    for metric in (
        metrics_to_summarize
    ):

        vals = (
            all_metrics[
                metric
            ]
            .astype(
                float
            )
        )


        summary_rows.append(
            {
                "method":
                    method,

                "method_display":
                    method_display,

                "metric":
                    metric,

                "n":
                    len(
                        vals
                    ),

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

                "median":
                    float(
                        vals.median()
                    ),

                "q25":
                    float(
                        vals.quantile(
                            0.25
                        )
                    ),

                "q75":
                    float(
                        vals.quantile(
                            0.75
                        )
                    ),
            }
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_df.to_csv(
        OUTDIR
        / "clean_summary_all.csv",
        index=False
    )


    # ========================================================
    # COMBINE FEATURE IMPORTANCE
    # ========================================================

    importance_files = [
        FOLD_DIR
        /
        f"fold{fold:02d}_feature_importance.csv"

        for fold in range(
            1,
            6
        )
    ]


    importance_all = pd.concat(
        [
            pd.read_csv(
                p
            )
            for p in importance_files
        ],
        ignore_index=True
    )


    importance_all.to_csv(
        OUTDIR
        / "feature_importance_by_fold.csv",
        index=False
    )


    importance_summary = (
        importance_all
        .groupby(
            "feature"
        )[
            "importance"
        ]
        .agg(
            [
                "mean",
                "std"
            ]
        )
        .reset_index()
        .sort_values(
            "mean",
            ascending=False
        )
    )


    importance_summary.to_csv(
        OUTDIR
        / "feature_importance_summary.csv",
        index=False
    )


    # ========================================================
    # FIGURE — FEATURE IMPORTANCE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            8,
            5
        )
    )


    ax.bar(
        importance_summary[
            "feature"
        ],
        importance_summary[
            "mean"
        ]
    )


    ax.set_ylabel(
        "Mean RF feature importance"
    )

    ax.set_xlabel(
        "Handcrafted feature"
    )

    ax.set_title(
        (
            f"Clean RF feature importance — "
            f"{method_display}"
        )
    )


    ax.tick_params(
        axis="x",
        rotation=45
    )


    plt.tight_layout()


    plt.savefig(
        OUTDIR
        / "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.savefig(
        OUTDIR
        / "feature_importance.pdf",
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # TERMINAL FINAL
    # ========================================================

    dice = (
        all_metrics[
            "dice"
        ]
    )


    print(
        "\n"
        + "=" * 84
    )

    print(
        "CLEAN RF EXPERIMENT COMPLETE"
    )

    print(
        "=" * 84
    )

    print(
        "Method:",
        method_display
    )

    print(
        "Validation ROIs:",
        len(
            all_metrics
        )
    )

    print(
        "Mean Dice:",
        f"{dice.mean():.6f}"
    )

    print(
        "Std Dice:",
        f"{dice.std(ddof=1):.6f}"
    )

    print(
        "Mean Precision:",
        f"{all_metrics['precision'].mean():.6f}"
    )

    print(
        "Mean Recall:",
        f"{all_metrics['recall'].mean():.6f}"
    )

    print(
        "Mean IoU:",
        f"{all_metrics['iou'].mean():.6f}"
    )

    print(
        "Mean preprocessing time:",
        f"{all_metrics['time_preprocess_s'].mean():.4f} s"
    )

    print(
        "Mean segmentation time:",
        f"{all_metrics['time_segment_s'].mean():.4f} s"
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTDIR
    )

    print(
        "=" * 84
    )


if __name__ == "__main__":
    main()