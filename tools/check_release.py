#!/usr/bin/env python3
"""Audit presence of fixed reproducibility artifacts (does not run experiments)."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
methods=('no_correction','clahe','macenko','reinhard')
required=[ROOT/'inputs/splits/cv_splits_seed13_folds5.npz',ROOT/'inputs/training_samples/common_training_pixels_seed13.npz',ROOT/'outputs/stain_calibration/fold_perturbation_parameters.csv']
for fold in range(1,6):
    required += [ROOT/f'outputs/enhancement_targets/fold{fold:02d}_enhancement_target.json',ROOT/f'outputs/percentile_mechanism/fold{fold:02d}_paired_mechanism.csv']
for method in methods:
    required.append(ROOT/f'outputs/clean_rf/{method}/clean_metrics_per_roi.csv')
    for fold in range(1,6): required.append(ROOT/f'outputs/stain_robustness/{method}/folds/fold{fold:02d}_robustness_metrics.csv')
required += [ROOT/'outputs/stage16_statistical_audit/stage16_combined_paired_rows.csv',ROOT/'outputs/stage17_manuscript/stage17_five_exception_rois.csv']
for script in list((ROOT/'src').glob('*.py'))+list((ROOT/'legacy_code').glob('*.py')):
    ast.parse(script.read_text(encoding='utf-8'))
missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file()]
print('Syntax verified for',len(list((ROOT/'src').glob('*.py'))+list((ROOT/'legacy_code').glob('*.py'))),'original scripts')
print('Required fixed artifacts:',len(required),'present:',len(required)-len(missing))
if missing:
    print('MISSING:')
    for p in missing:print(' - '+p)
    raise SystemExit(1)
print('Fixed artifacts present. This does NOT independently reproduce every published statistic.')
