#!/usr/bin/env python3
"""Restore original-order training-sample CSV from its compact original NPZ.
This is a release helper, not a component of the originally executed experiments.
Run after copying the two locked NPZ files. Use only trusted local NPZ files.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'inputs'/'training_samples'/'common_training_pixels_seed13.npz'
DEST = ROOT/'inputs'/'training_samples'/'common_training_pixels_seed13.csv'
SPLITS = ROOT/'inputs'/'splits'/'cv_splits_seed13_folds5.npz'

def main():
    if not SRC.exists() or not SPLITS.exists():
        raise SystemExit('Missing original sample NPZ or locked CV split NPZ; run copy_local_assets.ps1')
    # Original NPZ contains an object array; do not load NPZ files from untrusted sources.
    with np.load(SRC, allow_pickle=True) as d, np.load(SPLITS, allow_pickle=True) as sp:
        roi = list(map(str, d['roi'].tolist()))
        yy = d['y']; xx = d['x']; label = d['label']
        samples_seed = int(d['seed'][0])
        types = dict(zip(map(str, sp['roi_names'].tolist()), map(str, sp['slide_types'].tolist())))
    if samples_seed != 13 or len(roi) != 1_230_000 or not (len(yy)==len(xx)==len(label)==len(roi)):
        raise SystemExit('Unexpected seed or training-sample row count; verify original NPZ')
    if len(types)!=205 or set(roi)!=set(types):
        raise SystemExit('Sample ROI identities do not match locked split')
    data=pd.DataFrame({'roi':roi,'slide_type':[types[r] for r in roi], 'y':yy,'x':xx,'label':label})
    if not set(data.label.unique()).issubset({0,1}):
        raise SystemExit('Unexpected class labels in original samples')
    DEST.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(DEST,index=False)
    print('Restored exact row sequence:', DEST, '| rows:',len(data))
    print('This CSV is ignored by Git because the compact original NPZ is distributed.')

if __name__=='__main__':main()
