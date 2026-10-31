# ============================================================
# puma_nuclei_rf_only_cv5_split.py
#
# RF-only nuclei segmentation (pixel-level) with PRE-SAVED CV splits.
#
# - Trains RF on training folds (nuclei pixels vs non-nuclei pixels inside tissue)
# - Predicts prob map on validation folds, thresholds -> binary nuclei mask
# - Converts binary -> instances via watershed (same postprocess as rule-based)
#
# FAIR RUNTIME:
# - time_segment_s includes: feature computation + RF inference + threshold + instance conversion
# - excludes: read, preprocess, GT raster, metrics, overlay saving, disk I/O
#
# OUTPUTS (OUTDIR):
#  - metrics_per_roi.csv
#  - runtime_per_roi.csv
#  - summary_by_fold.csv
#  - summary_all.csv
#  - runtime_summary_all.csv
#  - global_metrics_summary_thesis.csv
#  - confusion_matrix_by_fold.csv
#  - confusion_matrix_global.csv
#  - confusion_matrix_global_counts.csv
#  - confusion_matrix_global_rowpct.csv
#  - confusion_matrix_global_count_and_rowpct.csv
#  - overlays_samples/
#  - models/ (RF model per fold)
#  - run_config.json
# ============================================================

import os, sys, json, glob, argparse, time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import scipy.ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
import joblib

from skimage import morphology, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import color as skcolor
from skimage.filters import sobel


# ------------------------------------------------------------
# Make imports work when script lives in puma_thesis_seg/scripts
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # puma_thesis_seg/
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, THIS_DIR)

from puma_preprocess_final import preprocess_puma_roi, downsample2x_mask_max
from hybridA_utils import compute_pixel_metrics, make_overlay


# ============================================================
# Helpers
# ============================================================
def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def robust_normalize_to_01(arr: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi <= lo:
        return normalize_to_01(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def slide_type_from_base(base: str) -> str:
    b = base.lower()
    if "metastatic" in b:
        return "metastatic"
    if "primary" in b:
        return "primary"
    return "unknown"


def balanced_accuracy_from_counts(tp, fp, fn, tn) -> float:
    tp = float(tp); fp = float(fp); fn = float(fn); tn = float(tn)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    if np.isnan(tpr) and np.isnan(tnr):
        return np.nan
    if np.isnan(tpr):
        return tnr
    if np.isnan(tnr):
        return tpr
    return 0.5 * (tpr + tnr)


def mean_std_str(x: pd.Series, digits=4) -> str:
    m = float(x.mean())
    s = float(x.std(ddof=1))
    return f"{m:.{digits}f} ± {s:.{digits}f}"


# ============================================================
# GeoJSON rasterization (binary nuclei GT)
# ============================================================
def _extract_polygons_from_geometry(geom):
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", None)
    if coords is None:
        return []

    polys = []
    if gtype == "Polygon":
        if len(coords) > 0:
            polys.append(coords[0])
    elif gtype == "MultiPolygon":
        for poly in coords:
            if len(poly) > 0 and len(poly[0]) > 0:
                polys.append(poly[0])

    out = []
    for poly in polys:
        pts = [(float(x), float(y)) for x, y in poly]
        if len(pts) >= 3:
            out.append(pts)
    return out


def rasterize_geojson_binary(gt_path, H, W):
    canvas = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(canvas)
    data = json.load(open(gt_path, "r"))
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        for poly in _extract_polygons_from_geometry(geom):
            draw.polygon(poly, outline=1, fill=1)
    return (np.array(canvas) > 0)


def rasterize_geojson_aligned(gt_path, raw_H, raw_W, to_20x: bool):
    gt0 = rasterize_geojson_binary(gt_path, raw_H, raw_W)
    if to_20x:
        gt0 = downsample2x_mask_max(gt0)
    return gt0.astype(bool)


# ============================================================
# Feature extraction (RF-only)
# ============================================================
def compute_feature_stack(img_rgb_u8: np.ndarray) -> tuple[list[str], np.ndarray]:
    """
    Returns (feature_names, feat_stack) where:
      feat_stack: float32 array (H, W, F)
    Features:
      - Hinv (robust normalized)
      - Gaussian(Hinv) at sigmas [0.7,1.2,1.4]
      - LoG(Hinv) at sigmas [0.7,1.2,1.4]
      - Sobel(Hinv)
      - grayscale
    """
    img = img_rgb_u8.astype(np.float32) / 255.0
    gray = skcolor.rgb2gray(img).astype(np.float32)
    gray01 = robust_normalize_to_01(gray)

    # HED -> hematoxylin channel
    hed = skcolor.rgb2hed(img)
    h = hed[:, :, 0].astype(np.float32)
    # invert hematoxylin (nuclei tend to be darker in H channel)
    hinv = 1.0 - robust_normalize_to_01(h)

    sigmas = [0.7, 1.2, 1.4]

    feats = []
    names = []

    feats.append(hinv); names.append("Hinv")

    for s in sigmas:
        g = ndi.gaussian_filter(hinv, sigma=float(s)).astype(np.float32)
        feats.append(g); names.append(f"Hinv_gauss_s{s}")

    for s in sigmas:
        # gaussian_laplace gives negative/positive; keep as-is but normalize robustly
        lg = ndi.gaussian_laplace(hinv, sigma=float(s)).astype(np.float32)
        lg01 = robust_normalize_to_01(lg)
        feats.append(lg01); names.append(f"Hinv_LoG_s{s}")

    sh = sobel(hinv).astype(np.float32)
    sh01 = robust_normalize_to_01(sh)
    feats.append(sh01); names.append("Hinv_sobel")

    feats.append(gray01); names.append("gray")

    feat_stack = np.stack(feats, axis=-1).astype(np.float32)  # (H,W,F)
    return names, feat_stack


def sample_training_pixels(tissue_mask: np.ndarray, gt_bin: np.ndarray, n_samples: int, rng: np.random.Generator):
    """
    Balanced sampling inside tissue:
      pos: nuclei pixels (gt=1)
      neg: non-nuclei pixels (gt=0)
    """
    tissue = tissue_mask.astype(bool)
    pos_idx = np.flatnonzero(tissue & gt_bin)
    neg_idx = np.flatnonzero(tissue & (~gt_bin))

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None, None

    n_pos = min(len(pos_idx), n_samples // 2)
    n_neg = min(len(neg_idx), n_samples - n_pos)

    pos_s = rng.choice(pos_idx, size=n_pos, replace=(len(pos_idx) < n_pos))
    neg_s = rng.choice(neg_idx, size=n_neg, replace=(len(neg_idx) < n_neg))

    idx = np.concatenate([pos_s, neg_s])
    y = np.concatenate([np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)])
    return idx, y


# ============================================================
# Binary -> instances
# ============================================================
def binary_to_instances(binary, tissue_mask,
                        min_area=5, max_area=1200,
                        min_peak_distance=2, dist_sigma=1.0):
    tissue = tissue_mask.astype(bool)
    b = (binary.astype(bool) & tissue)

    b = morphology.remove_small_objects(b, min_size=min_area)
    b = morphology.binary_opening(b, morphology.disk(1))

    if b.sum() == 0:
        return np.zeros_like(b, dtype=np.int32), b

    dist = ndi.distance_transform_edt(b)
    dist_s = ndi.gaussian_filter(dist, sigma=dist_sigma)

    coords = peak_local_max(
        dist_s,
        labels=b,
        min_distance=int(min_peak_distance),
        footprint=np.ones((3, 3), dtype=bool)
    )
    markers = np.zeros_like(dist_s, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    if markers.max() == 0:
        markers = measure.label(b).astype(np.int32)

    labels = watershed(-dist_s, markers, mask=b).astype(np.int32)

    out = np.zeros_like(labels, dtype=np.int32)
    cur = 1
    for reg in measure.regionprops(labels):
        if not (min_area <= reg.area <= max_area):
            continue
        rr, cc = zip(*reg.coords)
        out[rr, cc] = cur
        cur += 1

    pred_bin = out > 0
    return out, pred_bin


# ============================================================
# Args
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--roi_dir", type=str, default="01_training_dataset_tif_ROIs")
    ap.add_argument("--gt_dir", type=str, default="01_training_dataset_geojson_nuclei")

    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--to_20x", type=int, default=1)

    ap.add_argument("--splits_npz", type=str, required=True)

    # RF training
    ap.add_argument("--samples_per_roi", type=int, default=6000)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_depth", type=int, default=0, help="0 means None")
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=13)

    # inference
    ap.add_argument("--prob_thresh", type=float, default=0.40)
    ap.add_argument("--predict_chunk", type=int, default=250000,
                    help="Predict in chunks to limit RAM (num pixels per chunk).")

    # instance postprocess
    ap.add_argument("--min_area", type=int, default=5)
    ap.add_argument("--max_area", type=int, default=1200)
    ap.add_argument("--min_peak_distance", type=int, default=2)
    ap.add_argument("--dist_sigma", type=float, default=1.0)

    # saving
    ap.add_argument("--save_overlays_per_fold", type=int, default=12)

    # debug
    ap.add_argument("--max_rois", type=int, default=0)

    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    to_20x = bool(args.to_20x)
    rng = np.random.default_rng(int(args.seed))

    os.makedirs(args.outdir, exist_ok=True)
    overlays_dir = os.path.join(args.outdir, "overlays_samples")
    os.makedirs(overlays_dir, exist_ok=True)
    models_dir = os.path.join(args.outdir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # pairs in sorted order (must match splits file)
    roi_files = sorted(glob.glob(os.path.join(args.roi_dir, "*.tif")))
    pairs = []
    for roi_path in roi_files:
        base = os.path.splitext(os.path.basename(roi_path))[0]
        gt_path = os.path.join(args.gt_dir, base + "_nuclei.geojson")
        if os.path.exists(gt_path):
            pairs.append((base, roi_path, gt_path))

    if len(pairs) == 0:
        raise RuntimeError("No (ROI,GT) pairs found.")

    if args.max_rois and args.max_rois > 0:
        pairs = pairs[:int(args.max_rois)]

    names = [p[0] for p in pairs]
    types = [slide_type_from_base(n) for n in names]
    print("Paired ROIs:", len(pairs), "| to_20x:", to_20x)
    print("Type counts:", pd.Series(types).value_counts().to_dict())

    # load splits
    sp = np.load(args.splits_npz, allow_pickle=True)
    roi_names_saved = list(sp["roi_names"].tolist())
    tr_list = sp["tr"].tolist()
    va_list = sp["va"].tolist()

    if list(names) != roi_names_saved[:len(names)]:
        if not (args.max_rois and roi_names_saved[:len(names)] == list(names)):
            raise RuntimeError("ROI order mismatch vs saved split file.")

    n_folds = len(va_list)
    print(f"Loaded splits: {args.splits_npz} | folds={n_folds}")

    all_rows = []
    fold_summary_rows = []
    cm_by_fold_rows = []

    # ============================================================
    # CV folds
    # ============================================================
    for fold in range(1, n_folds + 1):
        tr_idx = np.array(tr_list[fold - 1], dtype=int)
        va_idx = np.array(va_list[fold - 1], dtype=int)

        print(f"\n========== FOLD {fold}/{n_folds} ==========")
        print(f"Train={len(tr_idx)} | Val={len(va_idx)}")

        # ----------------------------
        # Build RF training set
        # ----------------------------
        X_list = []
        y_list = []
        feature_names = None

        for j, ii in enumerate(tr_idx, start=1):
            base, roi_path, gt_path = pairs[ii]

            img0 = np.array(Image.open(roi_path).convert("RGB"), dtype=np.uint8)
            raw_H, raw_W = img0.shape[:2]

            img_pp, tissue_mask, _fb = preprocess_puma_roi(img0, to_20x=to_20x)
            gt_bin = rasterize_geojson_aligned(gt_path, raw_H, raw_W, to_20x=to_20x)

            # feature maps
            fnames, feat_stack = compute_feature_stack(img_pp)
            if feature_names is None:
                feature_names = fnames

            idx, y = sample_training_pixels(tissue_mask, gt_bin, int(args.samples_per_roi), rng)
            if idx is None:
                continue

            H, W, F = feat_stack.shape
            X = feat_stack.reshape(-1, F)[idx]
            X_list.append(X)
            y_list.append(y)

            if j % 25 == 0:
                print(f"  train ROIs processed: {j}/{len(tr_idx)}")

        if len(X_list) == 0:
            raise RuntimeError("No training samples collected. Check GT/tissue masks.")

        X_train = np.concatenate(X_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)

        print(f"Training set: X={X_train.shape}, y_pos={(y_train==1).sum()}, y_neg={(y_train==0).sum()}")

        rf = RandomForestClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=None if int(args.max_depth) == 0 else int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            n_jobs=int(args.n_jobs),
            class_weight="balanced",
            random_state=int(args.seed),
        )
        rf.fit(X_train, y_train)

        model_path = os.path.join(models_dir, f"rf_fold{fold:02d}.joblib")
        joblib.dump({"model": rf, "feature_names": feature_names}, model_path)
        print("Saved model:", model_path)

        # ----------------------------
        # Validate fold
        # ----------------------------
        overlays_saved = 0
        fold_rows = []
        fold_tp = fold_fp = fold_fn = fold_tn = 0

        # proba index for class=1
        cls = list(rf.classes_)
        if 1 not in cls:
            raise RuntimeError("RF classes_ does not contain class 1 (nuclei).")
        pos_col = cls.index(1)

        for k, ii in enumerate(va_idx, start=1):
            base, roi_path, gt_path = pairs[ii]

            # read
            t0 = time.perf_counter()
            img0 = np.array(Image.open(roi_path).convert("RGB"), dtype=np.uint8)
            raw_H, raw_W = img0.shape[:2]
            t1 = time.perf_counter()

            # preprocess
            img_pp, tissue_mask, fb = preprocess_puma_roi(img0, to_20x=to_20x)
            t2 = time.perf_counter()

            # GT
            gt_bin = rasterize_geojson_aligned(gt_path, raw_H, raw_W, to_20x=to_20x)
            t3 = time.perf_counter()

            # ----------------------------
            # SEGMENTATION-ONLY timing
            # features + inference + threshold + instances
            # ----------------------------
            tseg0 = time.perf_counter()

            fnames, feat_stack = compute_feature_stack(img_pp)
            H, W, F = feat_stack.shape
            X_all = feat_stack.reshape(-1, F)

            tissue = tissue_mask.astype(bool).reshape(-1)
            idx_t = np.flatnonzero(tissue)

            prob = np.zeros(H * W, dtype=np.float32)

            # predict in chunks
            chunk = int(args.predict_chunk)
            for s in range(0, len(idx_t), chunk):
                sub = idx_t[s:s + chunk]
                p = rf.predict_proba(X_all[sub])[:, pos_col].astype(np.float32)
                prob[sub] = p

            prob_map = prob.reshape(H, W)
            pred_bin = (prob_map >= float(args.prob_thresh)) & tissue_mask.astype(bool)

            inst, pred_bin2 = binary_to_instances(
                pred_bin, tissue_mask,
                min_area=int(args.min_area),
                max_area=int(args.max_area),
                min_peak_distance=int(args.min_peak_distance),
                dist_sigma=float(args.dist_sigma)
            )

            tseg1 = time.perf_counter()
            time_seg = tseg1 - tseg0

            # metrics (excluded from seg runtime)
            m = compute_pixel_metrics(pred_bin2, gt_bin, tissue_mask)

            fold_tp += int(m["tp"])
            fold_fp += int(m["fp"])
            fold_fn += int(m["fn"])
            fold_tn += int(m["tn"])

            # runtime breakdown
            time_read = t1 - t0
            time_pre  = t2 - t1
            time_gt   = t3 - t2
            time_tot  = (tseg1 - t0)

            bal_acc = balanced_accuracy_from_counts(m["tp"], m["fp"], m["fn"], m["tn"])

            row = {
                "fold": fold,
                "roi": base,
                "slide_type": slide_type_from_base(base),
                "fallback_used": bool(fb),
                "to_20x": bool(to_20x),

                "tissue_pixels": int(tissue_mask.sum()),
                "gt_pos_pixels": int(gt_bin[tissue_mask].sum()),
                "pred_pos_pixels": int(pred_bin2[tissue_mask].sum()),
                "n_pred_instances": int(inst.max()),

                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "fn": int(m["fn"]),
                "tn": int(m["tn"]),

                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1": float(m["f1"]),
                "dice": float(m["dice"]),
                "iou": float(m["iou"]),
                "accuracy": float(m["accuracy"]),
                "specificity": float(m["specificity"]),
                "balanced_accuracy": float(bal_acc) if bal_acc == bal_acc else np.nan,

                "time_read_s": float(time_read),
                "time_preprocess_s": float(time_pre),
                "time_gt_raster_s": float(time_gt),
                "time_segment_s": float(time_seg),
                "time_total_s": float(time_tot),

                "samples_per_roi": int(args.samples_per_roi),
                "n_estimators": int(args.n_estimators),
                "max_depth": None if int(args.max_depth) == 0 else int(args.max_depth),
                "min_samples_leaf": int(args.min_samples_leaf),
                "prob_thresh": float(args.prob_thresh),
            }

            fold_rows.append(row)
            all_rows.append(row)

            # overlays
            if overlays_saved < int(args.save_overlays_per_fold):
                vis = make_overlay(img_pp, gt_bin, pred_bin2)
                out_png = os.path.join(overlays_dir, f"rf_fold{fold:02d}_{base}_overlay.png")
                Image.fromarray(vis).save(out_png)
                overlays_saved += 1

            if k % 10 == 0:
                print(f"  validated ROIs: {k}/{len(va_idx)}")

        # fold summaries
        df_fold = pd.DataFrame(fold_rows)
        metric_cols = ["precision", "recall", "f1", "dice", "iou", "accuracy", "specificity", "balanced_accuracy"]
        fold_mean = {c: float(df_fold[c].mean()) for c in metric_cols + ["time_segment_s"]}
        fold_mean.update({"fold": fold, "n_rois": int(len(df_fold))})
        fold_summary_rows.append(fold_mean)

        cm_by_fold_rows.append({
            "fold": fold,
            "tp": int(fold_tp), "fp": int(fold_fp), "fn": int(fold_fn), "tn": int(fold_tn),
            "balanced_accuracy": float(balanced_accuracy_from_counts(fold_tp, fold_fp, fold_fn, fold_tn))
        })

        print("Fold mean metrics:", {k: round(fold_mean[k], 4) for k in metric_cols})
        print("Fold mean seg runtime (s):", round(fold_mean["time_segment_s"], 4))

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================
    df_all = pd.DataFrame(all_rows)

    metric_cols = ["precision", "recall", "f1", "dice", "iou", "accuracy", "specificity", "balanced_accuracy"]

    # metrics_per_roi
    metrics_keep = [
        "fold", "roi", "slide_type", "fallback_used", "to_20x",
        "tissue_pixels", "gt_pos_pixels", "pred_pos_pixels", "n_pred_instances",
        "tp", "fp", "fn", "tn",
        *metric_cols,
        "time_segment_s",
        "samples_per_roi", "n_estimators", "max_depth", "min_samples_leaf", "prob_thresh",
    ]
    df_metrics = df_all[metrics_keep].copy()
    out_metrics = os.path.join(args.outdir, "metrics_per_roi.csv")
    df_metrics.to_csv(out_metrics, index=False)

    # runtime_per_roi
    runtime_keep = [
        "fold", "roi", "slide_type", "fallback_used", "to_20x",
        "time_read_s", "time_preprocess_s", "time_gt_raster_s",
        "time_segment_s", "time_total_s",
    ]
    df_runtime = df_all[runtime_keep].copy()
    out_runtime = os.path.join(args.outdir, "runtime_per_roi.csv")
    df_runtime.to_csv(out_runtime, index=False)

    # summaries
    summary_all = df_metrics[metric_cols + ["time_segment_s"]].agg(["mean", "std"])
    out_summary_all = os.path.join(args.outdir, "summary_all.csv")
    summary_all.to_csv(out_summary_all)

    df_by_fold = pd.DataFrame(fold_summary_rows)
    out_by_fold = os.path.join(args.outdir, "summary_by_fold.csv")
    df_by_fold.to_csv(out_by_fold, index=False)

    runtime_summary = df_runtime[[
        "time_read_s", "time_preprocess_s", "time_gt_raster_s",
        "time_segment_s", "time_total_s"
    ]].agg(["mean", "std"])
    out_runtime_summary = os.path.join(args.outdir, "runtime_summary_all.csv")
    runtime_summary.to_csv(out_runtime_summary)

    # confusion matrices (global)
    tp = int(df_all["tp"].sum())
    fp = int(df_all["fp"].sum())
    fn = int(df_all["fn"].sum())
    tn = int(df_all["tn"].sum())

    cm_global = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["GT_neg", "GT_pos"],
        columns=["Pred_neg", "Pred_pos"]
    )
    out_cm_global = os.path.join(args.outdir, "confusion_matrix_global.csv")
    cm_global.to_csv(out_cm_global)

    # fold confusion
    cm_by_fold = pd.DataFrame(cm_by_fold_rows)
    out_cm_fold = os.path.join(args.outdir, "confusion_matrix_by_fold.csv")
    cm_by_fold.to_csv(out_cm_fold, index=False)

    # normalized confusion matrices
    cm_counts = cm_global.copy()
    row_sums = cm_counts.sum(axis=1).replace(0, np.nan)
    cm_rowpct = (cm_counts.div(row_sums, axis=0) * 100)
    cm_pretty = cm_counts.astype(np.int64).astype(str) + " (" + cm_rowpct.round(1).astype(str) + "%)"

    out_cm_counts = os.path.join(args.outdir, "confusion_matrix_global_counts.csv")
    out_cm_rowpct = os.path.join(args.outdir, "confusion_matrix_global_rowpct.csv")
    out_cm_pretty = os.path.join(args.outdir, "confusion_matrix_global_count_and_rowpct.csv")
    cm_counts.to_csv(out_cm_counts)
    cm_rowpct.to_csv(out_cm_rowpct)
    cm_pretty.to_csv(out_cm_pretty)

    # thesis paste-ready summary (one row)
    thesis_row = {}
    for c in metric_cols + ["time_segment_s"]:
        thesis_row[f"{c}_mean"] = float(df_metrics[c].mean())
        thesis_row[f"{c}_std"]  = float(df_metrics[c].std(ddof=1))
        thesis_row[f"{c}_mean±std"] = mean_std_str(df_metrics[c], digits=4)

    thesis_row["balanced_accuracy_from_global_cm"] = float(balanced_accuracy_from_counts(tp, fp, fn, tn))
    thesis_row["prob_thresh"] = float(args.prob_thresh)
    thesis_row["samples_per_roi"] = int(args.samples_per_roi)
    thesis_row["n_estimators"] = int(args.n_estimators)

    df_thesis = pd.DataFrame([thesis_row])
    out_thesis = os.path.join(args.outdir, "global_metrics_summary_thesis.csv")
    df_thesis.to_csv(out_thesis, index=False)

    # save config
    cfg = vars(args)
    cfg["to_20x"] = bool(to_20x)
    cfg["feature_set"] = [
        "Hinv", "Hinv_gauss_s0.7", "Hinv_gauss_s1.2", "Hinv_gauss_s1.4",
        "Hinv_LoG_s0.7", "Hinv_LoG_s1.2", "Hinv_LoG_s1.4",
        "Hinv_sobel", "gray"
    ]
    cfg_path = os.path.join(args.outdir, "run_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print("\n✅ Saved to:", args.outdir)
    print(" - metrics_per_roi.csv")
    print(" - runtime_per_roi.csv")
    print(" - summary_all.csv")
    print(" - summary_by_fold.csv")
    print(" - runtime_summary_all.csv")
    print(" - global_metrics_summary_thesis.csv")
    print(" - confusion_matrix_by_fold.csv")
    print(" - confusion_matrix_global.csv")
    print(" - confusion_matrix_global_counts.csv")
    print(" - confusion_matrix_global_rowpct.csv")
    print(" - confusion_matrix_global_count_and_rowpct.csv")
    print(" - overlays_samples/*.png")
    print(" - models/rf_foldXX.joblib")
    print(" - run_config.json")


if __name__ == "__main__":
    main()
