r"""Stage 14: read-only five-ROI feature-invariance pilot, no RF training.

Run from project root:
  .venv\Scripts\python.exe src\14_feature_invariance_pilot.py

Uses existing PUMA folds and calibration; leaves previous experiment outputs intact.
The 5-ROI result is a diagnostic pilot, not a powered statistical test.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import color as skcolor
from skimage.filters import sobel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "legacy_code"))

from adaptive_stain import decompose_he, reconstruct_from_decomposition
from conference_preprocess import preprocess_for_method, clean_reference_tissue_mask_20x
from enhancement_methods import load_fold_target
from puma_nuclei_rf_only_cv5_split import compute_feature_stack

ROI_DIR = ROOT / "01_training_dataset_tif_ROIs"
SPLITS = ROOT / "inputs" / "splits" / "cv_splits_seed13_folds5.npz"
CALIBRATION = ROOT / "outputs" / "stain_calibration" / "fold_perturbation_parameters.csv"
TARGETS = ROOT / "outputs" / "enhancement_targets"
OUT = ROOT / "outputs" / "feature_invariance_pilot"
FOLD = 1
N_ROIS = 5
PIXELS_PER_ROI = 4096
SEED = 13
CONDITIONS = (
    "H_low_severe", "H_high_severe",
    "E_low_severe", "E_high_severe",
    "HE_low_severe", "HE_high_severe",
)
SIGMAS = (0.7, 1.2, 1.4)


def get_stats(arr):
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = (float(x) for x in np.percentile(a, [1.0, 99.0]))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    return (lo, hi)


def norm_using(arr, stats):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = stats
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def raw_channels(img):
    rgb = img.astype(np.float32) / 255.0
    gray = skcolor.rgb2gray(rgb).astype(np.float32)
    hematoxylin = skcolor.rgb2hed(rgb)[:, :, 0].astype(np.float32)
    return hematoxylin, gray


def clean_reference_stats(img):
    """Clean-image percentile anchors for H, gray, LoG, and Sobel."""
    h, gray = raw_channels(img)
    stats = {"h": get_stats(h), "gray": get_stats(gray)}
    hinv = (1.0 - norm_using(h, stats["h"])).astype(np.float32)
    for s in SIGMAS:
        lg = ndi.gaussian_laplace(hinv, sigma=float(s)).astype(np.float32)
        stats[f"log_{s}"] = get_stats(lg)
    stats["sobel"] = get_stats(sobel(hinv).astype(np.float32))
    return stats


def frozen_feature_stack(img, stats):
    """Only normalization reference statistics change; feature formulas remain identical."""
    h, gray = raw_channels(img)
    hinv = (1.0 - norm_using(h, stats["h"])).astype(np.float32)
    features = [hinv]
    names = ["Hinv"]
    for s in SIGMAS:
        features.append(ndi.gaussian_filter(hinv, sigma=float(s)).astype(np.float32))
        names.append(f"Hinv_gauss_s{s}")
    for s in SIGMAS:
        lg = ndi.gaussian_laplace(hinv, sigma=float(s)).astype(np.float32)
        features.append(norm_using(lg, stats[f"log_{s}"]))
        names.append(f"Hinv_LoG_s{s}")
    features.append(norm_using(sobel(hinv).astype(np.float32), stats["sobel"]))
    names.append("Hinv_sobel")
    features.append(norm_using(gray, stats["gray"]))
    names.append("gray")
    return names, np.stack(features, axis=-1).astype(np.float32)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))


def corr(a, b):
    aa = np.asarray(a, np.float64).ravel()
    bb = np.asarray(b, np.float64).ravel()
    if aa.size < 2 or np.std(aa) < 1e-10 or np.std(bb) < 1e-10:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def main():
    for path in (ROI_DIR, SPLITS, CALIBRATION, TARGETS):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    with np.load(SPLITS, allow_pickle=True) as split:
        all_rois = [str(x) for x in split["roi_names"].tolist()]
        val_indices = np.asarray(split["va"].tolist()[FOLD - 1], dtype=int)
    if len(all_rois) != 205 or len(val_indices) != 41:
        raise RuntimeError("Unexpected frozen CV split contents")
    rois = [all_rois[i] for i in val_indices[:N_ROIS]]

    df = pd.read_csv(CALIBRATION)
    if not {"fold", "condition", "alpha_H", "alpha_E"}.issubset(df.columns):
        raise RuntimeError("Calibration CSV missing mandatory columns")
    params = {(int(r.fold), str(r.condition)): (float(r.alpha_H), float(r.alpha_E))
              for r in df.itertuples(index=False)}
    if any((FOLD, cond) not in params for cond in CONDITIONS):
        raise RuntimeError("Missing severe-condition calibration")
    fold_cfg = load_fold_target(TARGETS, FOLD)

    OUT.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    t0 = time.perf_counter()

    print("=" * 72, flush=True)
    print("FEATURE-INVARIANCE PILOT | 5 validation ROIs | 6 severe conditions", flush=True)
    print("Fold 1 only; NO RF training; no Stage 11/12 files modified", flush=True)
    print("=" * 72, flush=True)

    for roi_number, roi in enumerate(rois, 1):
        img_path = ROI_DIR / f"{roi}.tif"
        if not img_path.is_file():
            raise FileNotFoundError(img_path)
        raw = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        tissue = clean_reference_tissue_mask_20x(raw)
        clean20, _ = preprocess_for_method(raw, "no_correction", fold_cfg)
        if tissue.shape != clean20.shape[:2]:
            raise RuntimeError(f"Tissue/feature geometry mismatch: {roi}")

        names, clean_f = compute_feature_stack(clean20)
        stats = clean_reference_stats(clean20)
        frozen_names, reconstructed_clean_f = frozen_feature_stack(clean20, stats)
        if names != frozen_names:
            raise RuntimeError("Feature name/order mismatch")
        max_error = float(np.max(np.abs(clean_f - reconstructed_clean_f)))
        if max_error > 1e-5:
            raise RuntimeError(f"Frozen implementation disagrees with original on CLEAN image: {roi}, {max_error}")
        rng = np.random.default_rng(SEED + FOLD * 100 + roi_number)
        candidate = np.flatnonzero(tissue.reshape(-1))
        if len(candidate) < 100:
            raise RuntimeError(f"Insufficient tissue pixels in {roi}")
        sample = rng.choice(candidate, size=min(PIXELS_PER_ROI, len(candidate)), replace=False)
        yy, xx = np.unravel_index(sample, tissue.shape)
        h_clean, gray_clean = raw_channels(clean20)
        decomposition = decompose_he(raw)

        for cond in CONDITIONS:
            alpha_h, alpha_e = params[(FOLD, cond)]
            perturbed_raw = reconstruct_from_decomposition(
                decomposition, alpha_H=alpha_h, alpha_E=alpha_e)
            pert20, _ = preprocess_for_method(perturbed_raw, "no_correction", fold_cfg)
            if pert20.shape != clean20.shape:
                raise RuntimeError(f"Geometry changed in {roi}/{cond}")
            new_names, adaptive_f = compute_feature_stack(pert20)
            frozen_names, frozen_f = frozen_feature_stack(pert20, stats)
            if names != new_names or names != frozen_names:
                raise RuntimeError("Feature ordering changed")
            h_pert, gray_pert = raw_channels(pert20)
            raw_h_mae = mae(h_clean[yy, xx], h_pert[yy, xx])
            raw_gray_mae = mae(gray_clean[yy, xx], gray_pert[yy, xx])
            rgb_mae = mae(clean20[yy, xx] / 255.0, pert20[yy, xx] / 255.0)
            h_scale = max(stats["h"][1] - stats["h"][0], 1e-8)

            for j, feature in enumerate(names):
                a = clean_f[yy, xx, j]
                b = adaptive_f[yy, xx, j]
                c = frozen_f[yy, xx, j]
                raw_rows.append({
                    "fold": FOLD, "roi": roi, "condition": cond,
                    "feature": feature, "n_pixels": len(yy),
                    "alpha_H": alpha_h, "alpha_E": alpha_e,
                    "rgb_mae_01": rgb_mae,
                    "raw_h_mae": raw_h_mae,
                    "raw_h_mae_scaled_by_clean_p1_p99": raw_h_mae / h_scale,
                    "raw_gray_mae": raw_gray_mae,
                    "adaptive_feature_mae": mae(a, b),
                    "frozen_feature_mae": mae(a, c),
                    "adaptive_feature_pearson": corr(a, b),
                    "frozen_feature_pearson": corr(a, c),
                })
        # ROI checkpoint: only the new pilot output is written.
        pd.DataFrame(raw_rows).to_csv(OUT / "pilot_feature_drift.csv", index=False)
        print(f"ROI {roi_number}/{len(rois)} complete: {roi} | "
              f"rows={len(raw_rows)}/{len(rois)*len(CONDITIONS)*9}", flush=True)

    result = pd.DataFrame(raw_rows)
    group = result.groupby(["condition", "feature"], as_index=False).agg(
        n_rois=("roi", "nunique"),
        mean_rgb_mae_01=("rgb_mae_01", "mean"),
        mean_adaptive_mae=("adaptive_feature_mae", "mean"),
        mean_frozen_mae=("frozen_feature_mae", "mean"),
        mean_adaptive_pearson=("adaptive_feature_pearson", "mean"),
        mean_frozen_pearson=("frozen_feature_pearson", "mean"),
    )
    group.to_csv(OUT / "pilot_summary_by_condition_feature.csv", index=False)
    per_roi = result.groupby(["roi", "condition"], as_index=False).agg(
        rgb_mae_01=("rgb_mae_01", "first"),
        mean_9feature_adaptive_mae=("adaptive_feature_mae", "mean"),
        mean_9feature_frozen_mae=("frozen_feature_mae", "mean"),
    )
    per_roi["adaptive_better_than_frozen"] = (
        per_roi["mean_9feature_adaptive_mae"] < per_roi["mean_9feature_frozen_mae"]
    )
    per_roi.to_csv(OUT / "pilot_roi_condition_summary.csv", index=False)
    info = {
        "status": "PASS", "fold": FOLD, "rois": rois,
        "conditions": list(CONDITIONS), "n_rois": len(rois),
        "n_conditions": len(CONDITIONS), "n_features": len(names),
        "n_rows": len(result), "pixels_per_roi_max": PIXELS_PER_ROI,
        "runtime_seconds": time.perf_counter() - t0,
        "mean_adaptive_mae": float(result["adaptive_feature_mae"].mean()),
        "mean_frozen_mae": float(result["frozen_feature_mae"].mean()),
        "count_roi_conditions_adaptive_lower_drift": int(per_roi["adaptive_better_than_frozen"].sum()),
        "total_roi_conditions": len(per_roi),
        "note": "Diagnostic only; no RF training or segmentation outcomes. Do not claim causation.",
    }
    with (OUT / "pilot_integrity.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print("=" * 72, flush=True)
    print("PASS: " + json.dumps(info, indent=2), flush=True)
    print("Output: " + str(OUT), flush=True)


if __name__ == "__main__":
    main()
