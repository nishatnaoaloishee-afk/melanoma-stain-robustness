import os, glob, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def slide_type_from_base(base: str) -> str:
    b = base.lower()
    if "metastatic" in b:
        return "metastatic"
    if "primary" in b:
        return "primary"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_dir", type=str, default="01_training_dataset_tif_ROIs")
    ap.add_argument("--gt_dir", type=str, default="01_training_dataset_geojson_nuclei")
    ap.add_argument("--outdir", type=str, default=os.path.join("outputs", "cv_splits"))
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    roi_files = sorted(glob.glob(os.path.join(args.roi_dir, "*.tif")))
    pairs = []
    for roi_path in roi_files:
        base = os.path.splitext(os.path.basename(roi_path))[0]
        gt_path = os.path.join(args.gt_dir, base + "_nuclei.geojson")
        if os.path.exists(gt_path):
            pairs.append((base, roi_path, gt_path))

    if len(pairs) == 0:
        raise RuntimeError("No (ROI,GT) pairs found.")

    names = [p[0] for p in pairs]
    types = [slide_type_from_base(n) for n in names]
    map_lab = {"primary": 0, "metastatic": 1, "unknown": 2}
    y_strat = np.array([map_lab.get(t, 2) for t in types], dtype=int)

    print("Pairs:", len(pairs))
    print("Type counts:", pd.Series(types).value_counts().to_dict())

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    tr_list, va_list = [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(pairs)), y_strat), start=1):
        tr_list.append(tr_idx)
        va_list.append(va_idx)
        print(f"Fold {fold}: train={len(tr_idx)} val={len(va_idx)}",
              "| val types:", pd.Series([types[i] for i in va_idx]).value_counts().to_dict())

    out_npz = os.path.join(args.outdir, f"cv_splits_seed{args.seed}_folds{args.cv_folds}.npz")
    np.savez_compressed(
        out_npz,
        roi_names=np.array(names, dtype=object),
        roi_paths=np.array([p[1] for p in pairs], dtype=object),
        gt_paths=np.array([p[2] for p in pairs], dtype=object),
        slide_types=np.array(types, dtype=object),
        tr=np.array(tr_list, dtype=object),
        va=np.array(va_list, dtype=object),
        seed=args.seed,
        cv_folds=args.cv_folds,
    )
    print("Saved:", out_npz)

if __name__ == "__main__":
    main()
