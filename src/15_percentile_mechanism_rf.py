"""Stage 15: controlled adaptive-versus-frozen percentile-normalization RF test.

Uses EXACT Stage-12 no-correction clean training pixels, hyperparameters, features,
perturbations, evaluation masks, and postprocessing. The sole intervention is
replacing image-adaptive feature percentile anchors with CLEAN ROI anchors at
inference time; training and the fitted RF are identical in both branches.

From E:\\PUMA_Stain_Robustness in the activated environment:
  .venv\\Scripts\\python.exe src\\15_percentile_mechanism_rf.py --fold 1 --check-only
  .venv\\Scripts\\python.exe src\\15_percentile_mechanism_rf.py --fold 1

Outputs are separate from all earlier experiments. ROI-level CSV checkpoints
permit resume after interruption, but the RF must be rebuilt on restart.
IMPORTANT: frozen CLEAN validation-ROI statistics are a counterfactual oracle,
not a deployable preprocessing pipeline and NOT a way to train the classifier.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
LEGACY = ROOT / 'legacy_code'
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(LEGACY))

from adaptive_stain import decompose_he, reconstruct_from_decomposition
from conference_preprocess import preprocess_for_method, clean_reference_tissue_mask_20x
from enhancement_methods import load_fold_target
from puma_nuclei_rf_only_cv5_split import (
    compute_feature_stack,
    rasterize_geojson_aligned,
    binary_to_instances,
)
from hybridA_utils import compute_pixel_metrics

ROI_DIR = ROOT / '01_training_dataset_tif_ROIs'
GT_DIR = ROOT / '01_training_dataset_geojson_nuclei'
SPLITS = ROOT / 'inputs' / 'splits' / 'cv_splits_seed13_folds5.npz'
SAMPLES = ROOT / 'inputs' / 'training_samples' / 'common_training_pixels_seed13.csv'
CALIBRATION = ROOT / 'outputs' / 'stain_calibration' / 'fold_perturbation_parameters.csv'
TARGETS = ROOT / 'outputs' / 'enhancement_targets'
CLEAN_DIR = ROOT / 'outputs' / 'clean_rf' / 'no_correction'
STAGE12_DIR = ROOT / 'outputs' / 'stain_robustness' / 'no_correction' / 'folds'
OUTDIR = ROOT / 'outputs' / 'percentile_mechanism'
PILOT_SCRIPT = SRC / '14_feature_invariance_pilot.py'

SEED = 13
N_TREES = 500
THRESHOLD = 0.40
CHUNK = 250000
FEATURE_PIXELS_PER_ROI = 4096
CONDITIONS = (
    'H_low_severe', 'H_high_severe',
    'E_low_severe', 'E_high_severe',
    'HE_low_severe', 'HE_high_severe',
)
# Fail rather than infer a mechanism if the unchanged Stage-12 pipeline differs.
REPRO_TOL_DICE = 1e-6
REPRO_TOL_FEATURE = 1e-5


def log(message: str):
    print(message, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def load_pilot_helpers():
    """Reuse EXACT frozen-feature logic validated in Stage 14."""
    if not PILOT_SCRIPT.is_file():
        raise FileNotFoundError(f'Missing validated Stage 14 script: {PILOT_SCRIPT}')
    spec = importlib.util.spec_from_file_location('puma_feature_invariance_stage14', PILOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load Stage 14 percentile-feature implementation')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.clean_reference_stats, mod.frozen_feature_stack


def expected_pairs(rois):
    return {(r, c) for r in rois for c in CONDITIONS}


def load_inputs(fold: int, *, load_training_pixels: bool):
    for path in [ROI_DIR, GT_DIR, SPLITS, SAMPLES, CALIBRATION, TARGETS, CLEAN_DIR, STAGE12_DIR]:
        if not path.exists():
            raise FileNotFoundError(f'Missing required Stage-11/12 input: {path}')

    with np.load(SPLITS, allow_pickle=True) as split:
        names = list(map(str, split['roi_names'].tolist()))
        slide_types = list(map(str, split['slide_types'].tolist()))
        tr = np.asarray(split['tr'].tolist()[fold - 1], dtype=int)
        va = np.asarray(split['va'].tolist()[fold - 1], dtype=int)
    if len(names) != 205 or len(set(names)) != 205 or len(tr) != 164 or len(va) != 41:
        raise RuntimeError('Frozen folds fail 205/164/41 structural check')
    if len(set(tr) & set(va)) or set(tr) | set(va) != set(range(205)):
        raise RuntimeError('Training/validation split overlaps or lacks ROIs')

    cfg = load_fold_target(TARGETS, fold)
    p = pd.read_csv(CALIBRATION)
    mandatory = {'fold', 'condition', 'alpha_H', 'alpha_E'}
    if not mandatory.issubset(p.columns):
        raise RuntimeError('Perturbation calibration missing required columns')
    if p.duplicated(['fold', 'condition']).any():
        raise RuntimeError('Duplicate fold-condition in calibration')
    alpha = {(int(r.fold), str(r.condition)): (float(r.alpha_H), float(r.alpha_E))
             for r in p.itertuples(index=False)}
    for cond in CONDITIONS:
        if (fold, cond) not in alpha or not np.all(np.isfinite(alpha[(fold, cond)])):
            raise RuntimeError(f'Missing/invalid fold {fold} calibration: {cond}')

    clean_path = CLEAN_DIR / 'clean_metrics_per_roi.csv'
    clean = pd.read_csv(clean_path, dtype={'roi': str})
    if len(clean) != 205 or clean['roi'].duplicated().any():
        raise RuntimeError('Clean Stage-11 No Correction baseline must contain 205 unique ROIs')
    clean_lookup = clean.set_index('roi')

    stage12_path = STAGE12_DIR / f'fold{fold:02d}_robustness_metrics.csv'
    old = pd.read_csv(stage12_path, dtype={'roi': str, 'condition': str})
    expected = expected_pairs([names[i] for i in va])
    actual = set(zip(old['roi'], old['condition']))
    if len(old) != 492 or old.duplicated(['roi', 'condition']).any() or actual != {
        (roi, cond) for roi in [names[i] for i in va]
        for cond in (
            'H_low_mild', 'H_low_severe', 'H_high_mild', 'H_high_severe',
            'E_low_mild', 'E_low_severe', 'E_high_mild', 'E_high_severe',
            'HE_low_mild', 'HE_low_severe', 'HE_high_mild', 'HE_high_severe',
        )
    }:
        raise RuntimeError('Stage-12 No Correction reference fold is not complete/unique')
    if not expected.issubset(actual):
        raise RuntimeError('Stage-12 No Correction reference is missing severe conditions')
    stage12_lookup = old.set_index(['roi', 'condition'])

    samples = None
    if load_training_pixels:
        df = pd.read_csv(SAMPLES, dtype={
            'roi': str, 'slide_type': str, 'y': np.int16,
            'x': np.int16, 'label': np.uint8,
        })
        if len(df) != 1230000:
            raise RuntimeError(f'Expected 1,230,000 common training pixel rows, got {len(df)}')
        samples = {}
        for roi, group in df.groupby('roi', sort=False):
            samples[roi] = {
                'y': group['y'].to_numpy(dtype=np.int16),
                'x': group['x'].to_numpy(dtype=np.int16),
                'label': group['label'].to_numpy(dtype=np.uint8),
            }
        if set(samples) != set(names):
            raise RuntimeError('Frozen common pixel set does not cover the 205 frozen ROIs')
        del df
    return names, slide_types, tr, va, cfg, alpha, clean_lookup, stage12_lookup, samples


def check_feature_equivalence(clean20, reference_stats_fn, frozen_features_fn):
    names, adaptive = compute_feature_stack(clean20)
    stats = reference_stats_fn(clean20)
    frozen_names, frozen = frozen_features_fn(clean20, stats)
    if names != frozen_names or len(names) != 9 or adaptive.shape != frozen.shape:
        raise RuntimeError('Feature names, count or shapes differ from legacy RF')
    maximum = float(np.max(np.abs(adaptive - frozen)))
    if maximum > REPRO_TOL_FEATURE:
        raise RuntimeError(f'Clean adaptive versus frozen feature mismatch: max abs={maximum}')
    return names, stats, adaptive


def prediction_metrics(rf, feature_stack, tissue, gt, positive_col):
    """Unchanged Stage-12 RF inference -> threshold -> watershed -> metric."""
    hh, ww, nf = feature_stack.shape
    if tissue.shape != (hh, ww) or gt.shape != (hh, ww) or nf != 9:
        raise RuntimeError('Tissue/GT/feature shape mismatch')
    matrix = feature_stack.reshape(-1, nf)
    tissue_idx = np.flatnonzero(tissue.reshape(-1))
    probability = np.zeros(hh * ww, dtype=np.float32)
    for first in range(0, len(tissue_idx), CHUNK):
        idx = tissue_idx[first:first + CHUNK]
        probability[idx] = rf.predict_proba(matrix[idx])[:, positive_col].astype(np.float32)
    binary = (probability.reshape(hh, ww) >= THRESHOLD) & tissue
    instances, final = binary_to_instances(
        binary, tissue, min_area=5, max_area=1200,
        min_peak_distance=2, dist_sigma=1.0,
    )
    scores = compute_pixel_metrics(final, gt, tissue)
    result = {key: float(scores[key]) for key in ['dice', 'iou', 'precision', 'recall']}
    result['pred_pos_pixels'] = int(final[tissue].sum())
    result['n_instances'] = int(instances.max())
    return result


def fit_clean_reference_rf(roi_names, training_idx, cfg, samples):
    features = []
    labels = []
    feat_names = None
    start = time.perf_counter()
    log('Building EXACT Stage-12 training features for 164 clean training ROIs...')
    for j, ii in enumerate(training_idx, 1):
        roi = roi_names[ii]
        with Image.open(ROI_DIR / f'{roi}.tif') as im:
            raw = np.asarray(im.convert('RGB'), dtype=np.uint8)
        clean20, _ = preprocess_for_method(raw, 'no_correction', cfg)
        names, fmap = compute_feature_stack(clean20)
        if feat_names is None:
            feat_names = names
        elif names != feat_names:
            raise RuntimeError('Training feature order mismatch')
        coords = samples[roi]
        yy, xx = coords['y'].astype(np.intp), coords['x'].astype(np.intp)
        if len(yy) != 6000 or yy.max() >= fmap.shape[0] or xx.max() >= fmap.shape[1] or yy.min() < 0 or xx.min() < 0:
            raise RuntimeError(f'Invalid frozen training coordinates for {roi}')
        features.append(fmap[yy, xx, :].astype(np.float32))
        labels.append(coords['label'])
        del raw, clean20, fmap
        if j % 20 == 0 or j == len(training_idx):
            log(f'  Training ROIs: {j}/{len(training_idx)}')
    X_train = np.concatenate(features, axis=0)
    y_train = np.concatenate(labels, axis=0)
    del features, labels
    gc.collect()
    if X_train.shape != (984000, 9) or np.count_nonzero(y_train == 1) != 492000 or np.count_nonzero(y_train == 0) != 492000:
        raise RuntimeError('Stage-12 training data shape / class balance mismatch')
    feature_seconds = time.perf_counter() - start
    log('Training ONE 500-tree RF, shared between adaptive and frozen branches...')
    tfit = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        n_jobs=-1, class_weight='balanced', random_state=SEED,
    )
    rf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - tfit
    del X_train, y_train
    gc.collect()
    log(f'RF fit complete. Feature extraction: {feature_seconds:.1f}s; fit: {fit_seconds:.1f}s')
    if 1 not in rf.classes_:
        raise RuntimeError('Trained RF missing positive class 1')
    return rf, list(rf.classes_).index(1), feature_seconds, fit_seconds


def atomic_csv(df: pd.DataFrame, path: Path):
    temp_path = path.with_suffix('.tmp')
    try:
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def check_checkpoint(df, validation_rois):
    columns = {'fold', 'roi', 'condition', 'adaptive_dice', 'frozen_dice',
               'adaptive_stage12_abs_error', 'clean_stage11_abs_error'}
    if not columns.issubset(df.columns):
        raise RuntimeError('Checkpoint lacks required columns; refusing unsafe resume')
    if df.duplicated(['roi', 'condition']).any():
        raise RuntimeError('Duplicate ROI-condition pair in Stage-15 checkpoint')
    actual = set(zip(df.roi.astype(str), df.condition.astype(str)))
    if not actual.issubset(expected_pairs(validation_rois)):
        raise RuntimeError('Checkpoint has unexpected ROI-condition pairs')
    numeric = ['adaptive_dice', 'frozen_dice', 'adaptive_stage12_abs_error', 'clean_stage11_abs_error']
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError('Stage-15 checkpoint includes non-finite metrics')
    if (df['adaptive_stage12_abs_error'].to_numpy(dtype=float) > REPRO_TOL_DICE).any() or (
        df['clean_stage11_abs_error'].to_numpy(dtype=float) > REPRO_TOL_DICE
    ).any():
        raise RuntimeError('Checkpoint contains unresolved Stage-11/12 reproduction error')
    return actual


def save_final_summary(df, fold, rois, start, feature_seconds, fit_seconds):
    pairs = check_checkpoint(df, rois)
    if pairs != expected_pairs(rois) or len(df) != 246:
        raise RuntimeError(f'Not a complete Stage-15 fold: {len(df)}/246 rows')
    by_condition = df.groupby('condition', as_index=False).agg(
        n_rois=('roi', 'nunique'),
        mean_clean_dice=('clean_dice', 'mean'),
        mean_adaptive_dice=('adaptive_dice', 'mean'),
        mean_frozen_dice=('frozen_dice', 'mean'),
        mean_adaptive_loss=('adaptive_dice_loss', 'mean'),
        mean_frozen_loss=('frozen_dice_loss', 'mean'),
        mean_adaptive_advantage=('adaptive_advantage_dice', 'mean'),
        median_adaptive_advantage=('adaptive_advantage_dice', 'median'),
        adaptive_wins=('adaptive_advantage_dice', lambda x: int((x > 0).sum())),
        mean_adaptive_feature_drift=('adaptive_feature_mae', 'mean'),
        mean_frozen_feature_drift=('frozen_feature_mae', 'mean'),
    )
    by_condition['n_rois'] = by_condition['n_rois'].astype(int)
    by_condition.to_csv(OUTDIR / f'fold{fold:02d}_summary_by_condition.csv', index=False)
    overall = {
        'status': 'PASS', 'fold': fold, 'n_rois': len(rois), 'n_conditions': len(CONDITIONS),
        'n_roi_condition_pairs': len(df),
        'mean_clean_dice': float(df.clean_dice.mean()),
        'mean_adaptive_dice': float(df.adaptive_dice.mean()),
        'mean_frozen_dice': float(df.frozen_dice.mean()),
        'mean_adaptive_dice_loss': float(df.adaptive_dice_loss.mean()),
        'mean_frozen_dice_loss': float(df.frozen_dice_loss.mean()),
        'mean_adaptive_advantage_dice': float(df.adaptive_advantage_dice.mean()),
        'mean_adaptive_feature_drift': float(df.adaptive_feature_mae.mean()),
        'mean_frozen_feature_drift': float(df.frozen_feature_mae.mean()),
        'max_adaptive_dice_diff_from_stage12': float(df.adaptive_stage12_abs_error.max()),
        'max_clean_dice_diff_from_stage11': float(df.clean_stage11_abs_error.max()),
        'feature_build_seconds_current_run': feature_seconds,
        'rf_fit_seconds_current_run': fit_seconds,
        'wall_seconds_current_run': time.perf_counter() - start,
        'caution': ('Fold-1 results are a pilot, not independent validation. '
                    'Frozen CLEAN validation-image percentiles are oracle counterfactuals, not deployable. '
                    'Paired ROIs, not 246 independent biological cases. '
                    'A Dice difference alone does not establish full causal mediation.'),
    }
    with (OUTDIR / f'fold{fold:02d}_integrity.json').open('w', encoding='utf-8') as fh:
        json.dump(overall, fh, indent=2)
    log('PASS: all 246 paired condition evaluations complete and Stage-11/12 reproduction verified')
    log(f'Mean adaptive Dice loss: {overall["mean_adaptive_dice_loss"]:.6f}')
    log(f'Mean frozen Dice loss:   {overall["mean_frozen_dice_loss"]:.6f}')
    log(f'Mean adaptive advantage: {overall["mean_adaptive_advantage_dice"]:+.6f}')
    log(f'Output folder: {OUTDIR}')


def run(fold, check_only):
    start = time.perf_counter()
    log('=' * 82)
    log(f'STAGE 15 | PERCENTILE-NORMALIZATION MECHANISM | FOLD {fold}/5')
    log('No Correction RF | SIX severe conditions | adaptive vs clean-frozen anchors')
    log('NO modifications to existing Stage-11, 12, or 14 outputs')
    log('=' * 82)
    reference_stats_fn, frozen_features_fn = load_pilot_helpers()
    names, types, tr, va, cfg, alpha, clean_lookup, stage12, samples = load_inputs(
        fold, load_training_pixels=not check_only,
    )
    rois = [names[i] for i in va]
    for roi in rois:
        row = clean_lookup.loc[roi]
        if int(row.fold) != fold:
            raise RuntimeError(f'Clean baseline fold mismatch for {roi}')

    # Always preflight a representative validation ROI before any RF training.
    first_roi = rois[0]
    with Image.open(ROI_DIR / f'{first_roi}.tif') as im:
        raw = np.asarray(im.convert('RGB'), dtype=np.uint8)
    clean20, tissue = preprocess_for_method(raw, 'no_correction', cfg)
    expected_tissue = clean_reference_tissue_mask_20x(raw)
    if not np.array_equal(tissue, expected_tissue):
        raise RuntimeError('Clean-reference tissue mask disagrees with preprocessing output')
    names_f, stats, clean_f = check_feature_equivalence(clean20, reference_stats_fn, frozen_features_fn)
    dec = decompose_he(raw)
    ah, ae = alpha[(fold, CONDITIONS[0])]
    perturbed = reconstruct_from_decomposition(dec, alpha_H=ah, alpha_E=ae)
    pert20, _ = preprocess_for_method(perturbed, 'no_correction', cfg)
    names_a, adaptive_f = compute_feature_stack(pert20)
    names_b, frozen_f = frozen_features_fn(pert20, stats)
    if names_a != names_f or names_b != names_f or adaptive_f.shape != frozen_f.shape:
        raise RuntimeError('Perturbed adaptive/frozen feature geometry or names mismatch')
    rgb_mae = float(np.mean(np.abs(pert20.astype(np.float32) - clean20.astype(np.float32))) / 255.0)
    if not (rgb_mae > 0.0 and np.isfinite(rgb_mae)):
        raise RuntimeError('Severe perturbation produced no meaningful measurable RGB change')
    log(f'Preflight PASS: fold {fold}, {first_roi}, 9 features, severe RGB MAE {rgb_mae:.6f}')
    del raw, clean20, tissue, clean_f, perturbed, pert20, adaptive_f, frozen_f, dec
    gc.collect()
    if check_only:
        log('CHECK-ONLY PASS. No RF training; no experimental outputs written.')
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / f'fold{fold:02d}_paired_mechanism.csv'
    if csv_path.exists():
        existing = pd.read_csv(csv_path, dtype={'roi': str, 'condition': str})
        done = check_checkpoint(existing, rois)
        if done == expected_pairs(rois):
            log('Existing fold already complete: no retraining needed.')
            save_final_summary(existing, fold, rois, start, None, None)
            return
        log(f'Resuming {len(done)}/246 pairs (will rebuild this fold\'s one shared RF).')
    else:
        existing = pd.DataFrame()
        done = set()

    cfg_path = OUTDIR / f'fold{fold:02d}_run_config.json'
    with cfg_path.open('w', encoding='utf-8') as fh:
        json.dump({
            'fold': fold, 'method': 'no_correction',
            'conditions': list(CONDITIONS), 'train_rois': 164, 'validation_rois': 41,
            'training_pixels': 984000, 'n_estimators': 500, 'max_depth': None,
            'min_samples_leaf': 1, 'n_jobs': -1, 'class_weight': 'balanced',
            'seed': SEED, 'probability_threshold': THRESHOLD, 'predict_chunk': CHUNK,
            'postprocess': {'min_area': 5, 'max_area': 1200, 'min_peak_distance': 2, 'dist_sigma': 1.0},
            'frozen_reference': 'within-ROI CLEAN percentile anchors, including derived LoG/Sobel',
            'feature_mae_pixels_per_roi': FEATURE_PIXELS_PER_ROI,
            'pilot_script_sha256': sha256(PILOT_SCRIPT),
            'no_model_checkpoint': True,
            'legacy_reference': 'Stage-11/12 No Correction matched RF',
        }, fh, indent=2)

    rf, pos_col, feature_seconds, fit_seconds = fit_clean_reference_rf(names, tr, cfg, samples)
    del samples
    gc.collect()

    log(f'Validating 41 ROIs x 6 severe conditions x 2 branches; checkpoint per ROI.')
    new_rows = []
    for index, ii in enumerate(va, start=1):
        roi = names[ii]
        pending = [c for c in CONDITIONS if (roi, c) not in done]
        if not pending:
            log(f'ROI {index:02d}/41: already complete')
            continue
        with Image.open(ROI_DIR / f'{roi}.tif') as im:
            raw = np.asarray(im.convert('RGB'), dtype=np.uint8)
        tissue = clean_reference_tissue_mask_20x(raw)
        original_h, original_w = raw.shape[:2]
        gt = rasterize_geojson_aligned(str(GT_DIR / f'{roi}_nuclei.geojson'), original_h, original_w, to_20x=True)
        clean20, _ = preprocess_for_method(raw, 'no_correction', cfg)
        if gt.shape != tissue.shape or clean20.shape[:2] != tissue.shape:
            raise RuntimeError(f'GT/tissue/20x dimension mismatch: {roi}')
        feat_names, anchors, clean_map = check_feature_equivalence(clean20, reference_stats_fn, frozen_features_fn)
        clean_scores = prediction_metrics(rf, clean_map, tissue, gt, pos_col)
        reference_clean = float(clean_lookup.loc[roi, 'dice'])
        clean_err = abs(clean_scores['dice'] - reference_clean)
        if clean_err > REPRO_TOL_DICE:
            raise RuntimeError(f'STOP: clean RF mismatches Stage 11 for {roi}: '
                               f'{clean_scores["dice"]:.9f} vs {reference_clean:.9f}, diff={clean_err:.3g}')
        rng = np.random.default_rng(SEED + fold * 100 + index)
        tissue_indices = np.flatnonzero(tissue.reshape(-1))
        if len(tissue_indices) < 100:
            raise RuntimeError(f'Insufficient tissue pixels: {roi}')
        samples_idx = rng.choice(tissue_indices, size=min(FEATURE_PIXELS_PER_ROI, len(tissue_indices)), replace=False)
        baseline_features = clean_map.reshape(-1, 9)[samples_idx].copy()
        yy, xx = np.unravel_index(samples_idx, tissue.shape)
        del clean_map
        dec = decompose_he(raw)
        roi_rows = []
        for cond in pending:
            ah, ae = alpha[(fold, cond)]
            tcond = time.perf_counter()
            perturbed = reconstruct_from_decomposition(dec, alpha_H=ah, alpha_E=ae)
            pert20, _ = preprocess_for_method(perturbed, 'no_correction', cfg)
            if pert20.shape != clean20.shape:
                raise RuntimeError(f'Perturbation changed geometry: {roi}/{cond}')
            rgb_mae = float(np.mean(np.abs(pert20[yy, xx].astype(np.float32) - clean20[yy, xx].astype(np.float32))) / 255.0)
            del perturbed
            adaptive_names, adaptive_map = compute_feature_stack(pert20)
            if adaptive_names != feat_names:
                raise RuntimeError(f'Adaptive feature names differ: {roi}/{cond}')
            adaptive_mae = float(np.mean(np.abs(adaptive_map.reshape(-1, 9)[samples_idx] - baseline_features)))
            adaptive_scores = prediction_metrics(rf, adaptive_map, tissue, gt, pos_col)
            stage12_dice = float(stage12.loc[(roi, cond), 'dice'])
            stage12_err = abs(adaptive_scores['dice'] - stage12_dice)
            if stage12_err > REPRO_TOL_DICE:
                raise RuntimeError(f'STOP: adaptive Stage-15 Dice mismatches Stage 12, {roi}/{cond}: '
                                   f'{adaptive_scores["dice"]:.9f} vs {stage12_dice:.9f}, diff={stage12_err:.3g}')
            del adaptive_map
            frozen_names, frozen_map = frozen_features_fn(pert20, anchors)
            if frozen_names != feat_names:
                raise RuntimeError(f'Frozen feature names differ: {roi}/{cond}')
            frozen_mae = float(np.mean(np.abs(frozen_map.reshape(-1, 9)[samples_idx] - baseline_features)))
            frozen_scores = prediction_metrics(rf, frozen_map, tissue, gt, pos_col)
            del frozen_map, pert20
            roi_rows.append({
                'fold': fold, 'roi': roi, 'slide_type': types[ii], 'condition': cond,
                'alpha_H': ah, 'alpha_E': ae,
                'clean_dice': reference_clean,
                'adaptive_dice': adaptive_scores['dice'],
                'frozen_dice': frozen_scores['dice'],
                'adaptive_dice_loss': reference_clean - adaptive_scores['dice'],
                'frozen_dice_loss': reference_clean - frozen_scores['dice'],
                'adaptive_advantage_dice': adaptive_scores['dice'] - frozen_scores['dice'],
                'adaptive_iou': adaptive_scores['iou'],
                'frozen_iou': frozen_scores['iou'],
                'adaptive_precision': adaptive_scores['precision'],
                'frozen_precision': frozen_scores['precision'],
                'adaptive_recall': adaptive_scores['recall'],
                'frozen_recall': frozen_scores['recall'],
                'adaptive_pred_pos_pixels': adaptive_scores['pred_pos_pixels'],
                'frozen_pred_pos_pixels': frozen_scores['pred_pos_pixels'],
                'adaptive_instances': adaptive_scores['n_instances'],
                'frozen_instances': frozen_scores['n_instances'],
                'adaptive_feature_mae': adaptive_mae,
                'frozen_feature_mae': frozen_mae,
                'rgb_mae_01': rgb_mae,
                'adaptive_stage12_abs_error': stage12_err,
                'clean_stage11_abs_error': clean_err,
                'condition_seconds': time.perf_counter() - tcond,
            })
            log(f'  ROI {index:02d}/41 | {cond:16s} | '
                f'adaptive={adaptive_scores["dice"]:.4f}, frozen={frozen_scores["dice"]:.4f}, '
                f'A-F={adaptive_scores["dice"]-frozen_scores["dice"]:+.4f}')
        new_rows.extend(roi_rows)
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        check_checkpoint(combined, rois)
        atomic_csv(combined, csv_path)
        done.update((roi, str(r['condition'])) for r in roi_rows)
        existing = combined
        new_rows = []
        log(f'ROI {index:02d}/41 SAVED | total={len(done)}/246 paired severe-condition comparisons')
        del raw, tissue, gt, clean20, dec, baseline_features
        gc.collect()

    del rf
    gc.collect()
    final = pd.read_csv(csv_path, dtype={'roi': str, 'condition': str})
    save_final_summary(final, fold, rois, start, feature_seconds, fit_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, default=1, choices=range(1, 6), help='CV fold 1 through 5; use 1 for initial study')
    parser.add_argument('--check-only', action='store_true', help='Input/preprocessing/feature checks, without any RF training')
    args = parser.parse_args()
    run(fold=args.fold, check_only=args.check_only)


if __name__ == '__main__':
    main()
