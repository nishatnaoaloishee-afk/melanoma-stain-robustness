from pathlib import Path
import sys
import argparse
import json
import time
import gc

import numpy as np
import pandas as pd

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

PERTURBATION_FILE = (
    ROOT
    / "outputs"
    / "stain_calibration"
    / "fold_perturbation_parameters.csv"
)

TARGET_DIR = (
    ROOT
    / "outputs"
    / "enhancement_targets"
)

CLEAN_BASE_DIR = (
    ROOT
    / "outputs"
    / "clean_rf"
)

BASE_OUTDIR = (
    ROOT
    / "outputs"
    / "stain_robustness"
)


  
# IMPORT PROJECT MODULES
  

SRC_DIR = ROOT / "src"

LEGACY_DIR = ROOT / "legacy_code"

sys.path.insert(
    0,
    str(SRC_DIR)
)

sys.path.insert(
    0,
    str(LEGACY_DIR)
)


from adaptive_stain import (
    decompose_he,
    reconstruct_from_decomposition,
)


from conference_preprocess import (
    preprocess_for_method,
    clean_reference_tissue_mask_20x,
)


from enhancement_methods import (
    METHODS,
    DISPLAY_NAMES,
    load_fold_target,
)


  
# EXACT LEGACY NUCLEI RF FUNCTIONS
  

from puma_nuclei_rf_only_cv5_split import (
    compute_feature_stack,
    binary_to_instances,
    rasterize_geojson_aligned,
    balanced_accuracy_from_counts,
)


from hybridA_utils import (
    compute_pixel_metrics,
)


  
# LOCKED SETTINGS
  

SEED = 13

N_ESTIMATORS = 500

MAX_DEPTH = None

MIN_SAMPLES_LEAF = 1

N_JOBS = -1

PROB_THRESHOLD = 0.40

PREDICT_CHUNK = 250000

MIN_AREA = 5

MAX_AREA = 1200

MIN_PEAK_DISTANCE = 2

DIST_SIGMA = 1.0


CONDITIONS = [
    "H_low_mild",
    "H_low_severe",

    "H_high_mild",
    "H_high_severe",

    "E_low_mild",
    "E_low_severe",

    "E_high_mild",
    "E_high_severe",

    "HE_low_mild",
    "HE_low_severe",

    "HE_high_mild",
    "HE_high_severe",
]


  
# ARGUMENTS
  

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--method",
        required=True,
        choices=METHODS
    )

    ap.add_argument(
        "--fold",
        type=int,
        default=0,
        help=(
            "0 = all folds; "
            "1-5 = run only one fold"
        )
    )

    ap.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help=(
            "0 = resume existing results; "
            "1 = delete and rerun selected fold(s)"
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


    if (
        args.fold < 0
        or
        args.fold > 5
    ):

        raise ValueError(
            "--fold must be 0, 1, 2, 3, 4, or 5."
        )


    OUTDIR = (
        BASE_OUTDIR
        / method
    )

    FOLD_DIR = (
        OUTDIR
        / "folds"
    )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FOLD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "=" * 88
    )

    print(
        "PUMA — STAIN ROBUSTNESS RF EXPERIMENT"
    )

    print(
        "=" * 88
    )

    print(
        "\nMethod:",
        method_display
    )

    print(
        "Perturbation conditions:",
        len(CONDITIONS)
    )

    print(
        "RF models will NOT be saved."
    )


     
    # LOAD CV SPLITS
     

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


     
    # LOAD FROZEN COMMON TRAINING PIXELS
     

    print(
        "\nLoading common training pixels..."
    )


    samples_df = pd.read_csv(
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
        samples_df
    ) != 1230000:

        raise RuntimeError(
            "Expected 1,230,000 saved training pixels."
        )


    sample_lookup = {}


    for roi, group in samples_df.groupby(
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
            "Training sample ROI count mismatch."
        )


    print(
        "Training coordinate set: PASS"
    )


     
    # LOAD PERTURBATION PARAMETERS
     

    perturb_df = pd.read_csv(
        PERTURBATION_FILE
    )


    perturb_lookup = {}


    for _, row in perturb_df.iterrows():

        perturb_lookup[
            (
                int(
                    row[
                        "fold"
                    ]
                ),
                str(
                    row[
                        "condition"
                    ]
                )
            )
        ] = (
            float(
                row[
                    "alpha_H"
                ]
            ),
            float(
                row[
                    "alpha_E"
                ]
            ),
        )


    for fold in range(
        1,
        6
    ):

        for condition in CONDITIONS:

            if (
                fold,
                condition
            ) not in perturb_lookup:

                raise RuntimeError(
                    "Missing perturbation parameter: "
                    f"fold={fold}, "
                    f"condition={condition}"
                )


    print(
        "Perturbation parameter check: PASS"
    )


     
    # LOAD CLEAN METHOD BASELINE
     

    CLEAN_METRICS_FILE = (
        CLEAN_BASE_DIR
        / method
        / "clean_metrics_per_roi.csv"
    )


    if not CLEAN_METRICS_FILE.exists():

        raise FileNotFoundError(
            f"Missing clean baseline:\n"
            f"{CLEAN_METRICS_FILE}"
        )


    clean_metrics = pd.read_csv(
        CLEAN_METRICS_FILE
    )


    if len(
        clean_metrics
    ) != 205:

        raise RuntimeError(
            f"Expected 205 clean metric rows, "
            f"found {len(clean_metrics)}."
        )


    clean_lookup = {}


    for _, row in clean_metrics.iterrows():

        clean_lookup[
            str(
                row[
                    "roi"
                ]
            )
        ] = {
            "fold":
                int(
                    row[
                        "fold"
                    ]
                ),

            "dice":
                float(
                    row[
                        "dice"
                    ]
                ),

            "iou":
                float(
                    row[
                        "iou"
                    ]
                ),

            "precision":
                float(
                    row[
                        "precision"
                    ]
                ),

            "recall":
                float(
                    row[
                        "recall"
                    ]
                ),

            "specificity":
                float(
                    row[
                        "specificity"
                    ]
                ),

            "balanced_accuracy":
                float(
                    row[
                        "balanced_accuracy"
                    ]
                ),
        }


    print(
        "Clean method baseline check: PASS"
    )


     
    # VERIFY FROZEN CLAHE SETTINGS
     

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
                f"Fold {fold} CLAHE clip limit "
                f"is {clip_limit}, not frozen 0.5."
            )


    print(
        "Frozen enhancement settings: PASS"
    )


     
    # SAVE RUN CONFIG
     

    run_config = {
        "method":
            method,

        "method_display":
            method_display,

        "cohort_size":
            205,

        "folds":
            5,

        "perturbation_conditions":
            CONDITIONS,

        "n_perturbations":
            len(
                CONDITIONS
            ),

        "samples_per_training_roi":
            6000,

        "positive_samples":
            3000,

        "negative_samples":
            3000,

        "seed":
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

        "models_saved":
            False,

        "evaluation_support":
            (
                "clean-reference tissue mask"
            ),

        "clean_baseline_source":
            str(
                CLEAN_METRICS_FILE
            ),
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


     
    # FOLDS TO RUN
     

    if args.fold == 0:

        folds_to_run = [
            1,
            2,
            3,
            4,
            5,
        ]

    else:

        folds_to_run = [
            int(
                args.fold
            )
        ]


     
    # PROCESS EACH FOLD
     

    for fold in folds_to_run:

        print(
            "\n"
            + "=" * 88
        )

        print(
            f"FOLD {fold}/5 — {method_display}"
        )

        print(
            "=" * 88
        )


        fold_path = (
            FOLD_DIR
            /
            f"fold{fold:02d}_robustness_metrics.csv"
        )


        runtime_path = (
            FOLD_DIR
            /
            f"fold{fold:02d}_runtime.json"
        )


          
        # OVERWRITE IF REQUESTED
          

        if bool(
            args.overwrite
        ):

            if fold_path.exists():

                fold_path.unlink()


            if runtime_path.exists():

                runtime_path.unlink()


        
        # CHECK EXISTING PROGRESS
    

        existing_df = None

        completed_pairs = set()


        if fold_path.exists():

            existing_df = pd.read_csv(
                fold_path
            )


            for _, row in existing_df.iterrows():

                completed_pairs.add(
                    (
                        str(
                            row[
                                "roi"
                            ]
                        ),
                        str(
                            row[
                                "condition"
                            ]
                        )
                    )
                )


            expected_fold_rows = (
                41
                *
                len(
                    CONDITIONS
                )
            )


            if len(
                completed_pairs
            ) == expected_fold_rows:

                print(
                    "Fold already complete — SKIPPING."
                )

                continue


            print(
                "Resuming fold."
            )

            print(
                "Already completed:",
                len(
                    completed_pairs
                ),
                "/",
                expected_fold_rows
            )


          
        # SPLIT
          

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


        if len(
            tr_idx
        ) != 164:

            raise RuntimeError(
                "Unexpected train count."
            )


        if len(
            va_idx
        ) != 41:

            raise RuntimeError(
                "Unexpected validation count."
            )


        cfg = load_fold_target(
            TARGET_DIR,
            fold
        )


          
        # REBUILD EXACT METHOD-MATCHED CLEAN RF
          

        print(
            "\nBuilding clean training features..."
        )


        X_list = []

        y_list = []

        feature_names = None


        t_feature_start = (
            time.perf_counter()
        )


        for j, ii in enumerate(
            tr_idx,
            start=1
        ):

            roi = (
                roi_names[
                    ii
                ]
            )


            raw = np.array(
                Image.open(
                    ROI_DIR
                    / f"{roi}.tif"
                ).convert(
                    "RGB"
                ),
                dtype=np.uint8
            )


            img_20x, _unused_mask = (
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

            elif list(
                fnames
            ) != feature_names:

                raise RuntimeError(
                    "Training feature-name mismatch."
                )


            sample = (
                sample_lookup[
                    roi
                ]
            )


            yy = (
                sample[
                    "y"
                ]
                .astype(
                    np.intp
                )
            )

            xx = (
                sample[
                    "x"
                ]
                .astype(
                    np.intp
                )
            )

            yy_label = (
                sample[
                    "label"
                ]
            )


            if (
                yy.max()
                >= feature_stack.shape[0]
                or
                xx.max()
                >= feature_stack.shape[1]
            ):

                raise RuntimeError(
                    f"Saved coordinate outside image: {roi}"
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
                yy_label
            )


            if (
                j % 20 == 0
                or
                j == len(
                    tr_idx
                )
            ):

                print(
                    "  Training ROIs:",
                    f"{j}/{len(tr_idx)}"
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


        if X_train.shape != (
            984000,
            9
        ):

            raise RuntimeError(
                f"Unexpected X_train shape: "
                f"{X_train.shape}"
            )


        n_pos = int(
            (
                y_train
                == 1
            ).sum()
        )

        n_neg = int(
            (
                y_train
                == 0
            ).sum()
        )


        if (
            n_pos != 492000
            or
            n_neg != 492000
        ):

            raise RuntimeError(
                "Unexpected class counts."
            )


        feature_build_seconds = (
            time.perf_counter()
            -
            t_feature_start
        )


        print(
            "\nTraining matrix:",
            X_train.shape
        )

        print(
            "Positive:",
            n_pos,
            "| Negative:",
            n_neg
        )


          
        # TRAIN RF
          

        print(
            "\nTraining Random Forest..."
        )


        t_fit = (
            time.perf_counter()
        )


        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=
                MIN_SAMPLES_LEAF,
            n_jobs=N_JOBS,
            class_weight="balanced",
            random_state=SEED,
        )


        rf.fit(
            X_train,
            y_train
        )


        fit_seconds = (
            time.perf_counter()
            -
            t_fit
        )


        del X_train
        del y_train

        gc.collect()


        print(
            "RF fit complete."
        )

        print(
            "Feature-building time:",
            f"{feature_build_seconds:.1f} s"
        )

        print(
            "RF fit time:",
            f"{fit_seconds:.1f} s"
        )


        classes = list(
            rf.classes_
        )


        if 1 not in classes:

            raise RuntimeError(
                "RF does not contain class 1."
            )


        pos_col = (
            classes.index(
                1
            )
        )


          
        # VALIDATE ALL 12 STAIN CONDITIONS
          

        t_fold_eval = (
            time.perf_counter()
        )


        for roi_number, ii in enumerate(
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


            pending_conditions = [
                condition

                for condition
                in CONDITIONS

                if (
                    roi,
                    condition
                )
                not in completed_pairs
            ]


            if len(
                pending_conditions
            ) == 0:

                print(
                    f"ROI {roi_number:02d}/41 | "
                    f"{roi} | already complete"
                )

                continue


              
            # LOAD CLEAN RAW ROI
              

            raw = np.array(
                Image.open(
                    ROI_DIR
                    / f"{roi}.tif"
                ).convert(
                    "RGB"
                ),
                dtype=np.uint8
            )


            raw_H, raw_W = (
                raw.shape[:2]
            )


              
            # COMMON CLEAN REFERENCE SUPPORT
              

            tissue_mask = (
                clean_reference_tissue_mask_20x(
                    raw
                )
            )


              
            # GT ONCE
              

            gt_bin = (
                rasterize_geojson_aligned(
                    str(
                        GT_DIR
                        /
                        f"{roi}_nuclei.geojson"
                    ),
                    raw_H,
                    raw_W,
                    to_20x=True
                )
            )


            if (
                tissue_mask.shape
                != gt_bin.shape
            ):

                raise RuntimeError(
                    f"GT/tissue mismatch: {roi}"
                )


              
            # ADAPTIVE H&E DECOMPOSITION ONCE
              

            decomposition = (
                decompose_he(
                    raw
                )
            )


            clean_row = (
                clean_lookup[
                    roi
                ]
            )


            if int(
                clean_row[
                    "fold"
                ]
            ) != fold:

                raise RuntimeError(
                    f"Clean baseline fold mismatch: {roi}"
                )


            roi_rows = []


              
            # 12 PERTURBATION CONDITIONS
              

            for condition in pending_conditions:

                alpha_H, alpha_E = (
                    perturb_lookup[
                        (
                            fold,
                            condition
                        )
                    ]
                )


                # --------------------------------------------
                # PERTURB
                # --------------------------------------------

                t0 = time.perf_counter()


                perturbed_raw = (
                    reconstruct_from_decomposition(
                        decomposition,
                        alpha_H=alpha_H,
                        alpha_E=alpha_E
                    )
                )


                t1 = time.perf_counter()


                # --------------------------------------------
                # METHOD-SPECIFIC PREPROCESSING
                # --------------------------------------------

                img_20x, _unused_mask = (
                    preprocess_for_method(
                        perturbed_raw,
                        method,
                        cfg
                    )
                )


                t2 = time.perf_counter()


                if (
                    img_20x.shape[:2]
                    != tissue_mask.shape
                ):

                    raise RuntimeError(
                        "Image/tissue shape mismatch."
                    )


                # --------------------------------------------
                # FEATURES + RF + POSTPROCESSING
                # --------------------------------------------

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

                    idx = (
                        tissue_idx[
                            start:
                            start
                            +
                            PREDICT_CHUNK
                        ]
                    )


                    probability[
                        idx
                    ] = (
                        rf
                        .predict_proba(
                            X_all[
                                idx
                            ]
                        )[
                            :,
                            pos_col
                        ]
                        .astype(
                            np.float32
                        )
                    )


                probability_map = (
                    probability.reshape(
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


                # --------------------------------------------
                # METRICS
                # --------------------------------------------

                m = compute_pixel_metrics(
                    pred_final,
                    gt_bin,
                    tissue_mask
                )


                balanced_accuracy = (
                    balanced_accuracy_from_counts(
                        m[
                            "tp"
                        ],
                        m[
                            "fp"
                        ],
                        m[
                            "fn"
                        ],
                        m[
                            "tn"
                        ],
                    )
                )


                perturbed_dice = float(
                    m[
                        "dice"
                    ]
                )


                perturbed_iou = float(
                    m[
                        "iou"
                    ]
                )


                clean_dice = float(
                    clean_row[
                        "dice"
                    ]
                )


                clean_iou = float(
                    clean_row[
                        "iou"
                    ]
                )


                dice_loss = (
                    clean_dice
                    -
                    perturbed_dice
                )


                iou_loss = (
                    clean_iou
                    -
                    perturbed_iou
                )


                dice_retention = (
                    perturbed_dice
                    /
                    (
                        clean_dice
                        +
                        1e-8
                    )
                )


                roi_rows.append(
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

                        "condition":
                            condition,

                        "alpha_H":
                            alpha_H,

                        "alpha_E":
                            alpha_E,

                        "clean_dice":
                            clean_dice,

                        "dice":
                            perturbed_dice,

                        "dice_loss":
                            float(
                                dice_loss
                            ),

                        "dice_retention":
                            float(
                                dice_retention
                            ),

                        "clean_iou":
                            clean_iou,

                        "iou":
                            perturbed_iou,

                        "iou_loss":
                            float(
                                iou_loss
                            ),

                        "clean_precision":
                            float(
                                clean_row[
                                    "precision"
                                ]
                            ),

                        "precision":
                            float(
                                m[
                                    "precision"
                                ]
                            ),

                        "clean_recall":
                            float(
                                clean_row[
                                    "recall"
                                ]
                            ),

                        "recall":
                            float(
                                m[
                                    "recall"
                                ]
                            ),

                        "clean_specificity":
                            float(
                                clean_row[
                                    "specificity"
                                ]
                            ),

                        "specificity":
                            float(
                                m[
                                    "specificity"
                                ]
                            ),

                        "balanced_accuracy":
                            float(
                                balanced_accuracy
                            ),

                        "tp":
                            int(
                                m[
                                    "tp"
                                ]
                            ),

                        "fp":
                            int(
                                m[
                                    "fp"
                                ]
                            ),

                        "fn":
                            int(
                                m[
                                    "fn"
                                ]
                            ),

                        "tn":
                            int(
                                m[
                                    "tn"
                                ]
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

                        "time_perturb_s":
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

                        "time_segment_s":
                            float(
                                tseg1
                                -
                                tseg0
                            ),

                        "time_condition_total_s":
                            float(
                                tseg1
                                -
                                t0
                            ),
                    }
                )


                print(
                    f"  ROI {roi_number:02d}/41 | "
                    f"{condition:16s} | "
                    f"Dice={perturbed_dice:.4f} | "
                    f"Loss={dice_loss:+.4f}"
                )


                del (
                    perturbed_raw,
                    img_20x,
                    feature_stack,
                    X_all,
                    probability,
                    probability_map,
                    pred_binary,
                    instances,
                    pred_final
                )


              
            # SAVE AFTER EVERY ROI
              

            if len(
                roi_rows
            ) > 0:

                new_df = pd.DataFrame(
                    roi_rows
                )


                if fold_path.exists():

                    old_df = pd.read_csv(
                        fold_path
                    )


                    combined = pd.concat(
                        [
                            old_df,
                            new_df
                        ],
                        ignore_index=True
                    )


                    combined = (
                        combined
                        .drop_duplicates(
                            subset=[
                                "roi",
                                "condition"
                            ],
                            keep="last"
                        )
                    )

                else:

                    combined = new_df


                combined.to_csv(
                    fold_path,
                    index=False
                )


                completed_pairs.update(
                    (
                        str(
                            row[
                                "roi"
                            ]
                        ),
                        str(
                            row[
                                "condition"
                            ]
                        )
                    )

                    for _, row
                    in new_df.iterrows()
                )


            del (
                raw,
                tissue_mask,
                gt_bin,
                decomposition
            )

            gc.collect()


          
        # FINALIZE FOLD
          

        fold_df = pd.read_csv(
            fold_path
        )


        fold_df = (
            fold_df
            .drop_duplicates(
                subset=[
                    "roi",
                    "condition"
                ],
                keep="last"
            )
        )


        expected_rows = (
            41
            *
            len(
                CONDITIONS
            )
        )


        if len(
            fold_df
        ) != expected_rows:

            print(
                "\nFold not yet complete."
            )

            print(
                "Rows:",
                len(
                    fold_df
                ),
                "/",
                expected_rows
            )

        else:

            fold_df = fold_df.sort_values(
                [
                    "roi",
                    "condition"
                ]
            )


            fold_df.to_csv(
                fold_path,
                index=False
            )


            fold_eval_seconds = (
                time.perf_counter()
                -
                t_fold_eval
            )


            runtime_info = {
                "fold":
                    fold,

                "method":
                    method,

                "training_feature_seconds":
                    float(
                        feature_build_seconds
                    ),

                "rf_fit_seconds":
                    float(
                        fit_seconds
                    ),

                "robustness_evaluation_seconds":
                    float(
                        fold_eval_seconds
                    ),

                "evaluations":
                    int(
                        len(
                            fold_df
                        )
                    ),

                "mean_condition_seconds":
                    float(
                        fold_df[
                            "time_condition_total_s"
                        ].mean()
                    ),
            }


            with open(
                runtime_path,
                "w"
            ) as f:

                json.dump(
                    runtime_info,
                    f,
                    indent=2
                )


            print(
                "\nFold",
                fold,
                "complete."
            )

            print(
                "Mean perturbed Dice:",
                f"{fold_df['dice'].mean():.6f}"
            )

            print(
                "Mean Dice loss:",
                f"{fold_df['dice_loss'].mean():.6f}"
            )


   
        # DISCARD RF


        del rf

        gc.collect()


        print(
            "RF discarded from memory."
        )


     
    # COMBINE METHOD IF ALL 5 FOLDS EXIST
     

    fold_files = [
        FOLD_DIR
        /
        f"fold{fold:02d}_robustness_metrics.csv"

        for fold
        in range(
            1,
            6
        )
    ]


    all_complete = True


    for p in fold_files:

        if not p.exists():

            all_complete = False

            break


        temp = pd.read_csv(
            p
        )


        if len(
            temp.drop_duplicates(
                subset=[
                    "roi",
                    "condition"
                ]
            )
        ) != (
            41
            *
            len(
                CONDITIONS
            )
        ):

            all_complete = False

            break


    if not all_complete:

        print(
            "\nSelected run finished."
        )

        print(
            "Not all five folds for this method "
            "are complete yet."
        )

        return


     
    # METHOD-WIDE RESULTS
     

    all_metrics = pd.concat(
        [
            pd.read_csv(
                p
            )

            for p in fold_files
        ],
        ignore_index=True
    )


    all_metrics = (
        all_metrics
        .drop_duplicates(
            subset=[
                "roi",
                "condition"
            ],
            keep="last"
        )
    )


    expected_total = (
        205
        *
        len(
            CONDITIONS
        )
    )


    if len(
        all_metrics
    ) != expected_total:

        raise RuntimeError(
            f"Expected {expected_total} robustness rows, "
            f"found {len(all_metrics)}."
        )


    all_metrics.to_csv(
        OUTDIR
        / "robustness_metrics_per_roi_condition.csv",
        index=False
    )


     
    # CONDITION SUMMARY
     

    condition_summary = (
        all_metrics
        .groupby(
            "condition"
        )
        .agg(
            n=(
                "roi",
                "count"
            ),

            mean_clean_dice=(
                "clean_dice",
                "mean"
            ),

            mean_perturbed_dice=(
                "dice",
                "mean"
            ),

            std_perturbed_dice=(
                "dice",
                "std"
            ),

            mean_dice_loss=(
                "dice_loss",
                "mean"
            ),

            std_dice_loss=(
                "dice_loss",
                "std"
            ),

            median_dice_loss=(
                "dice_loss",
                "median"
            ),

            mean_dice_retention=(
                "dice_retention",
                "mean"
            ),

            mean_iou=(
                "iou",
                "mean"
            ),

            mean_iou_loss=(
                "iou_loss",
                "mean"
            ),

            mean_precision=(
                "precision",
                "mean"
            ),

            mean_recall=(
                "recall",
                "mean"
            ),

            mean_specificity=(
                "specificity",
                "mean"
            ),

            mean_condition_seconds=(
                "time_condition_total_s",
                "mean"
            ),
        )
        .reset_index()
    )


    condition_summary.to_csv(
        OUTDIR
        / "robustness_summary_by_condition.csv",
        index=False
    )


     
    # ROI-LEVEL MEAN ROBUSTNESS LOSS
     

    roi_summary = (
        all_metrics
        .groupby(
            [
                "roi",
                "fold",
                "slide_type"
            ]
        )
        .agg(
            clean_dice=(
                "clean_dice",
                "first"
            ),

            mean_perturbed_dice=(
                "dice",
                "mean"
            ),

            mean_dice_loss=(
                "dice_loss",
                "mean"
            ),

            mean_dice_retention=(
                "dice_retention",
                "mean"
            ),

            worst_dice=(
                "dice",
                "min"
            ),

            worst_dice_loss=(
                "dice_loss",
                "max"
            ),
        )
        .reset_index()
    )


    roi_summary.to_csv(
        OUTDIR
        / "robustness_summary_per_roi.csv",
        index=False
    )


     
    # FINAL METHOD SUMMARY
     

    method_summary = pd.DataFrame(
        [
            {
                "method":
                    method,

                "method_display":
                    method_display,

                "n_rois":
                    205,

                "n_conditions":
                    len(
                        CONDITIONS
                    ),

                "total_evaluations":
                    len(
                        all_metrics
                    ),

                "clean_mean_dice":
                    float(
                        clean_metrics[
                            "dice"
                        ].mean()
                    ),

                "mean_perturbed_dice":
                    float(
                        all_metrics[
                            "dice"
                        ].mean()
                    ),

                "mean_dice_loss":
                    float(
                        roi_summary[
                            "mean_dice_loss"
                        ].mean()
                    ),

                "std_roi_mean_dice_loss":
                    float(
                        roi_summary[
                            "mean_dice_loss"
                        ].std(
                            ddof=1
                        )
                    ),

                "mean_dice_retention":
                    float(
                        roi_summary[
                            "mean_dice_retention"
                        ].mean()
                    ),

                "mean_worst_case_dice":
                    float(
                        roi_summary[
                            "worst_dice"
                        ].mean()
                    ),
            }
        ]
    )


    method_summary.to_csv(
        OUTDIR
        / "robustness_method_summary.csv",
        index=False
    )


    print(
        "\n"
        + "=" * 88
    )

    print(
        "FULL METHOD ROBUSTNESS EXPERIMENT COMPLETE"
    )

    print(
        "=" * 88
    )

    print(
        "Method:",
        method_display
    )

    print(
        "ROIs:",
        205
    )

    print(
        "Perturbations:",
        12
    )

    print(
        "Evaluations:",
        len(
            all_metrics
        )
    )

    print(
        "Clean mean Dice:",
        f"{clean_metrics['dice'].mean():.6f}"
    )

    print(
        "Mean perturbed Dice:",
        f"{all_metrics['dice'].mean():.6f}"
    )

    print(
        "Mean Dice loss:",
        f"{roi_summary['mean_dice_loss'].mean():.6f}"
    )

    print(
        "Mean Dice retention:",
        f"{roi_summary['mean_dice_retention'].mean():.6f}"
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTDIR
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()