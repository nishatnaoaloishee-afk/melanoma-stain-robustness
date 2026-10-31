#!/usr/bin/env python3
r"""Stage 16: read-only 5-fold consolidation of the Stage-15 mechanism experiment.

Run from the project root:
  .\.venv\Scripts\python.exe src\16_fivefold_scientific_audit.py

Requires numpy, pandas, scipy, matplotlib. Inputs stay unchanged.
Intentionally does NOT infer patient/slide independence from ROI names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FOLDS = (1, 2, 3, 4, 5)
CONDITIONS = (
    "H_low_severe", "H_high_severe", "E_low_severe", "E_high_severe",
    "HE_low_severe", "HE_high_severe",
)
N_ROI_PER_FOLD = 41
EPS = 1e-10
REQUIRED = (
    "fold", "roi", "slide_type", "condition", "alpha_H", "alpha_E",
    "clean_dice", "adaptive_dice", "frozen_dice", "adaptive_dice_loss",
    "frozen_dice_loss", "adaptive_advantage_dice", "adaptive_iou",
    "frozen_iou", "adaptive_precision", "frozen_precision",
    "adaptive_recall", "frozen_recall", "adaptive_feature_mae",
    "frozen_feature_mae", "rgb_mae_01", "adaptive_stage12_abs_error",
    "clean_stage11_abs_error",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("STAGE 16 AUDIT FAILED: " + message)


def close(a, b, what, tol=EPS) -> None:
    require(bool(np.allclose(a, b, rtol=0, atol=tol, equal_nan=False)), what)


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            d.update(chunk)
    return d.hexdigest()


def validate_alpha(row: pd.Series) -> bool:
    cond = str(row["condition"])
    h, e = float(row["alpha_H"]), float(row["alpha_E"])
    if not (np.isfinite(h) and np.isfinite(e) and h > 0 and e > 0):
        return False
    if cond.startswith("H_"):
        return bool(np.isclose(e, 1.0, atol=1e-8) and ((h < 1) if "low" in cond else (h > 1)))
    if cond.startswith("E_"):
        return bool(np.isclose(h, 1.0, atol=1e-8) and ((e < 1) if "low" in cond else (e > 1)))
    return bool((h < 1 and e < 1) if "low" in cond else (h > 1 and e > 1))


def collect_and_validate(src: Path):
    frames, integrity_details, inventory = [], [], {}
    roi_folds = {}
    for fold in FOLDS:
        stem = f"fold{fold:02d}"
        files = {
            "paired": src / f"{stem}_paired_mechanism.csv",
            "condition": src / f"{stem}_summary_by_condition.csv",
            "integrity": src / f"{stem}_integrity.json",
        }
        for kind, path in files.items():
            require(path.is_file(), f"Missing {path}")
            inventory[f"{stem}_{kind}"] = {"filename": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        df = pd.read_csv(files["paired"], dtype={"roi": "string", "slide_type": "string", "condition": "string"})
        require(all(col in df.columns for col in REQUIRED), f"{stem} required columns missing: {set(REQUIRED)-set(df.columns)}")
        require(len(df) == N_ROI_PER_FOLD * len(CONDITIONS), f"{stem} row count {len(df)} != 246")
        require(df.fold.nunique() == 1 and int(df.fold.iloc[0]) == fold, f"{stem} wrong fold identifiers")
        require(df.roi.nunique() == N_ROI_PER_FOLD, f"{stem} does not contain 41 unique ROIs")
        require(not df.duplicated(["roi", "condition"]).any(), f"{stem} duplicate ROI/condition")
        require(set(df.condition) == set(CONDITIONS), f"{stem} unexpected condition set")
        require(df.groupby("roi").condition.nunique().eq(6).all(), f"{stem} some ROIs missing conditions")
        require(df.groupby("roi").slide_type.nunique().eq(1).all(), f"{stem} inconsistent slide types")
        require(set(df.slide_type.unique()).issubset({"primary", "metastatic"}), f"{stem} unknown slide type")
        require(df.groupby("roi").clean_dice.agg(lambda s: s.max()-s.min()).max() < EPS, f"{stem} clean Dice changes across conditions")
        require(df.groupby("condition").alpha_H.agg(lambda s: s.max()-s.min()).max() < EPS, f"{stem} H calibration changes across same fold-condition")
        require(df.groupby("condition").alpha_E.agg(lambda s: s.max()-s.min()).max() < EPS, f"{stem} E calibration changes across same fold-condition")
        require(df.apply(validate_alpha, axis=1).all(), f"{stem} invalid perturbation direction or alpha")
        numeric = df.select_dtypes(include=[np.number])
        require(np.isfinite(numeric.to_numpy(dtype=float)).all(), f"{stem} nonfinite numeric values")
        for col in ("clean_dice", "adaptive_dice", "frozen_dice", "adaptive_iou", "frozen_iou", "adaptive_precision", "frozen_precision", "adaptive_recall", "frozen_recall"):
            require(df[col].between(-EPS, 1+EPS).all(), f"{stem} out-of-range {col}")
        for col in ("adaptive_feature_mae", "frozen_feature_mae", "rgb_mae_01", "adaptive_stage12_abs_error", "clean_stage11_abs_error"):
            require((df[col] >= -EPS).all(), f"{stem} negative {col}")
        close(df.clean_dice-df.adaptive_dice, df.adaptive_dice_loss, f"{stem} adaptive loss arithmetic")
        close(df.clean_dice-df.frozen_dice, df.frozen_dice_loss, f"{stem} frozen loss arithmetic")
        close(df.adaptive_dice-df.frozen_dice, df.adaptive_advantage_dice, f"{stem} advantage arithmetic")
        require(df.adaptive_stage12_abs_error.max() < EPS, f"{stem} adaptive does not reproduce Stage12")
        require(df.clean_stage11_abs_error.max() < EPS, f"{stem} clean does not reproduce Stage11")
        for roi in df.roi.unique():
            require(roi not in roi_folds, f"ROI appears in two folds: {roi}")
            roi_folds[roi] = fold
        with files["integrity"].open("r", encoding="utf-8") as f:
            report = json.load(f)
        require(report.get("status") == "PASS", f"{stem} integrity status not PASS")
        for key, expected in {"fold":fold, "n_rois":41, "n_conditions":6, "n_roi_condition_pairs":246}.items():
            require(report.get(key) == expected, f"{stem} integrity {key} mismatch")
        comparisons = {
            "mean_clean_dice": df.clean_dice.mean(),
            "mean_adaptive_dice": df.adaptive_dice.mean(),
            "mean_frozen_dice": df.frozen_dice.mean(),
            "mean_adaptive_dice_loss": df.adaptive_dice_loss.mean(),
            "mean_frozen_dice_loss": df.frozen_dice_loss.mean(),
            "mean_adaptive_advantage_dice": df.adaptive_advantage_dice.mean(),
            "mean_adaptive_feature_drift": df.adaptive_feature_mae.mean(),
            "mean_frozen_feature_drift": df.frozen_feature_mae.mean(),
        }
        for key, observed in comparisons.items():
            require(key in report and abs(float(report[key])-observed) < EPS, f"{stem} integrity mean mismatch for {key}")
        require(float(report["max_adaptive_dice_diff_from_stage12"]) < EPS, f"{stem} integrity Stage12 reproduction issue")
        require(float(report["max_clean_dice_diff_from_stage11"]) < EPS, f"{stem} integrity Stage11 reproduction issue")
        summary = pd.read_csv(files["condition"])
        require(len(summary) == 6 and summary.condition.nunique() == 6, f"{stem} condition-summary coverage")
        for _, saved_row in summary.iterrows():
            cond = str(saved_row.condition)
            require(cond in CONDITIONS, f"{stem} unexpected summary condition {cond}")
            seg = df.loc[df.condition == cond]
            expected = {
                "n_rois":len(seg),
                "mean_clean_dice":seg.clean_dice.mean(),
                "mean_adaptive_dice":seg.adaptive_dice.mean(),
                "mean_frozen_dice":seg.frozen_dice.mean(),
                "mean_adaptive_loss":seg.adaptive_dice_loss.mean(),
                "mean_frozen_loss":seg.frozen_dice_loss.mean(),
                "mean_adaptive_advantage":seg.adaptive_advantage_dice.mean(),
                "median_adaptive_advantage":seg.adaptive_advantage_dice.median(),
                "adaptive_wins":int((seg.adaptive_advantage_dice > 0).sum()),
                "mean_adaptive_feature_drift":seg.adaptive_feature_mae.mean(),
                "mean_frozen_feature_drift":seg.frozen_feature_mae.mean(),
            }
            for key, val in expected.items():
                require(key in summary.columns, f"{stem} missing summary key {key}")
                require(abs(float(saved_row[key]) - float(val)) < EPS, f"{stem} summary mismatch {cond}, {key}")
        frames.append(df)
        integrity_details.append({"fold":fold, "status":"PASS", "n_roi_condition":int(len(df)),
                                  "max_stage12_error":float(df.adaptive_stage12_abs_error.max()),
                                  "max_stage11_error":float(df.clean_stage11_abs_error.max())})
    require(len(roi_folds) == 205, f"Five-fold unique ROI count {len(roi_folds)}, expected 205")
    all_df = pd.concat(frames, ignore_index=True)
    all_df["feature_drift_reduction"] = all_df.frozen_feature_mae-all_df.adaptive_feature_mae
    all_df["prediction_positive_pixel_diff"] = all_df.adaptive_pred_pos_pixels-all_df.frozen_pred_pos_pixels if all(c in all_df for c in ("adaptive_pred_pos_pixels","frozen_pred_pos_pixels")) else np.nan
    return all_df, inventory, integrity_details


def by_roi(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, roi, slide_type), g in df.groupby(["fold", "roi", "slide_type"], sort=True):
        rows.append({
            "fold":int(fold), "roi":str(roi), "slide_type":str(slide_type),
            "clean_dice":float(g.clean_dice.iloc[0]),
            "mean_adaptive_dice":float(g.adaptive_dice.mean()),
            "mean_frozen_dice":float(g.frozen_dice.mean()),
            "mean_adaptive_advantage":float(g.adaptive_advantage_dice.mean()),
            "median_adaptive_advantage":float(g.adaptive_advantage_dice.median()),
            "worst_adaptive_dice":float(g.adaptive_dice.min()),
            "worst_frozen_dice":float(g.frozen_dice.min()),
            "worst_case_advantage":float(g.adaptive_dice.min()-g.frozen_dice.min()),
            "mean_adaptive_feature_drift":float(g.adaptive_feature_mae.mean()),
            "mean_frozen_feature_drift":float(g.frozen_feature_mae.mean()),
            "mean_feature_drift_reduction":float(g.feature_drift_reduction.mean()),
            "mean_rgb_mae_01":float(g.rgb_mae_01.mean()),
            "conditions_adaptive_wins":int((g.adaptive_advantage_dice > 0).sum()),
        })
    result = pd.DataFrame(rows)
    require(len(result) == 205 and result.roi.nunique() == 205, "ROI table is not 205 unique units")
    return result


def bootstrap_stratified(df: pd.DataFrame, roi: pd.DataFrame, iters: int, seed: int):
    # Sample 41 ROI identities WITH replacement within each of five fixed folds.
    rng = np.random.default_rng(seed)
    roi = roi.sort_values(["fold", "roi"]).reset_index(drop=True)
    sampled = np.concatenate([rng.integers(i*41,(i+1)*41,size=(iters,41)) for i in range(5)],axis=1)
    mat = (df.pivot(index="roi",columns="condition", values="adaptive_advantage_dice")
           .reindex(index=roi.roi,columns=CONDITIONS).to_numpy(dtype=float))
    require(mat.shape == (205,6) and np.isfinite(mat).all(), "Bootstrap ROI x condition matrix invalid")
    main = mat.mean(axis=1)
    main_boot = main[sampled].mean(axis=1)
    condition_boot = {cond:mat[:,j][sampled].mean(axis=1) for j,cond in enumerate(CONDITIONS)}
    mean_ci = tuple(float(v) for v in np.quantile(main_boot,[.025,.975]))
    condition_ci = {c:tuple(float(v) for v in np.quantile(bs,[.025,.975])) for c,bs in condition_boot.items()}
    return mean_ci, condition_ci


def sign_test(values: np.ndarray):
    values=np.asarray(values,dtype=float)
    n_pos = int((values>EPS).sum()); n_neg=int((values < -EPS).sum()); n_tie=int(len(values)-n_pos-n_neg)
    n=n_pos+n_neg
    p=float(stats.binomtest(n_pos,n,0.5,alternative="two-sided").pvalue) if n>0 else 1.0
    return n_pos,n_neg,n_tie,p


def holm_adjust(pvalues):
    ps=np.asarray(pvalues,dtype=float)
    order=np.argsort(ps)
    out=np.zeros(len(ps),dtype=float)
    running=0
    m=len(ps)
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*ps[idx])
        out[idx]=min(1.0,running)
    return out


def spearman(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:
        return float("nan"),float("nan")
    res=stats.spearmanr(x,y)
    return float(res.statistic),float(res.pvalue)


def create_tables(df, roi, overall_ci, condition_ci):
    mean_ad=float(df.adaptive_dice.mean()); mean_fr=float(df.frozen_dice.mean())
    wins,losses,ties,p = sign_test(roi.mean_adaptive_advantage.to_numpy())
    overall = pd.DataFrame([{
        "unit_of_inference":"ROI (205 mean-over-six-condition units)",
        "n_rois":len(roi), "n_conditions":6, "n_paired_roi_condition_rows":len(df),
        "mean_clean_dice":float(df.clean_dice.mean()),
        "mean_adaptive_dice":mean_ad,"mean_frozen_dice":mean_fr,
        "mean_adaptive_loss":float(df.adaptive_dice_loss.mean()),
        "mean_frozen_loss":float(df.frozen_dice_loss.mean()),
        "mean_adaptive_advantage":float(roi.mean_adaptive_advantage.mean()),
        "median_roi_mean_adaptive_advantage":float(roi.mean_adaptive_advantage.median()),
        "bootstrap_95ci_low":overall_ci[0],"bootstrap_95ci_high":overall_ci[1],
        "mean_adaptive_feature_mae":float(df.adaptive_feature_mae.mean()),
        "mean_frozen_feature_mae":float(df.frozen_feature_mae.mean()),
        "relative_feature_drift_reduction_pct":float(100*(1-df.adaptive_feature_mae.mean()/df.frozen_feature_mae.mean())),
        "roi_wins":wins,"roi_losses":losses,"roi_ties":ties,"roi_sign_test_p_exploratory":p,
        "roi_worst_case_adaptive_dice_mean":float(roi.worst_adaptive_dice.mean()),
        "roi_worst_case_frozen_dice_mean":float(roi.worst_frozen_dice.mean()),
        "mean_roi_worst_case_advantage":float(roi.worst_case_advantage.mean()),
    }])
    fold_rows=[]
    for fold,g in df.groupby("fold",sort=True):
        rg=roi[roi.fold==fold]
        fold_rows.append({"fold":int(fold),"n_rois":len(rg),"n_rows":len(g),
                          "mean_clean_dice":float(g.clean_dice.mean()),"mean_adaptive_dice":float(g.adaptive_dice.mean()),
                          "mean_frozen_dice":float(g.frozen_dice.mean()),
                          "mean_adaptive_advantage":float(g.adaptive_advantage_dice.mean()),
                          "mean_adaptive_feature_mae":float(g.adaptive_feature_mae.mean()),
                          "mean_frozen_feature_mae":float(g.frozen_feature_mae.mean()),
                          "roi_wins":int((rg.mean_adaptive_advantage>0).sum())})
    fold_tbl=pd.DataFrame(fold_rows)
    cond_rows=[]
    for condition in CONDITIONS:
        g=df[df.condition==condition]
        v=g.adaptive_advantage_dice.to_numpy()
        w,l,t,p=sign_test(v)
        cond_rows.append({
            "condition":condition,"n_rois":g.roi.nunique(),
            "mean_clean_dice":float(g.clean_dice.mean()),
            "mean_adaptive_dice":float(g.adaptive_dice.mean()),"mean_frozen_dice":float(g.frozen_dice.mean()),
            "mean_adaptive_dice_loss":float(g.adaptive_dice_loss.mean()),
            "mean_frozen_dice_loss":float(g.frozen_dice_loss.mean()),
            "mean_adaptive_advantage":float(g.adaptive_advantage_dice.mean()),
            "median_adaptive_advantage":float(g.adaptive_advantage_dice.median()),
            "roi_bootstrap_95ci_low":condition_ci[condition][0],
            "roi_bootstrap_95ci_high":condition_ci[condition][1],
            "roi_wins":w,"roi_losses":l,"roi_ties":t,
            "roi_sign_test_p_exploratory":p,
            "mean_adaptive_feature_mae":float(g.adaptive_feature_mae.mean()),
            "mean_frozen_feature_mae":float(g.frozen_feature_mae.mean()),
            "mean_rgb_mae_01":float(g.rgb_mae_01.mean()),
        })
    conditions=pd.DataFrame(cond_rows)
    conditions["holm_adjusted_roi_sign_test_p_exploratory"]=holm_adjust(conditions.roi_sign_test_p_exploratory)
    subgroup=[]
    for st,g in roi.groupby("slide_type",sort=True):
        vals=g.mean_adaptive_advantage.to_numpy()
        w,l,t,p=sign_test(vals)
        subgroup.append({"slide_type":st,"n_rois":len(g),
                         "mean_roi_adaptive_advantage":float(np.mean(vals)),
                         "median_roi_adaptive_advantage":float(np.median(vals)),
                         "roi_wins":w,"roi_losses":l,"roi_ties":t,
                         "roi_sign_test_p_exploratory":p})
    subgroup=pd.DataFrame(subgroup)
    return overall,fold_tbl,conditions,subgroup


def build_associations(df,roi):
    rows=[]
    def add(label,x,y,unit):
        rho,p=spearman(x,y)
        rows.append({"analysis":label,"unit":unit,"n":len(x),"spearman_rho":rho,
                     "nominal_p_exploratory":p,"note":"Association does NOT prove causal mediation; p assumes independent units."})
    add("ROI mean: reduction in feature MAE vs adaptive Dice advantage",
        roi.mean_feature_drift_reduction,roi.mean_adaptive_advantage,"ROI means")
    add("ROI mean: adaptive feature MAE vs adaptive Dice loss",
        roi.mean_adaptive_feature_drift,
        roi.clean_dice-roi.mean_adaptive_dice,"ROI means")
    add("ROI mean: frozen feature MAE vs frozen Dice loss",
        roi.mean_frozen_feature_drift,
        roi.clean_dice-roi.mean_frozen_dice,"ROI means")
    for condition in CONDITIONS:
        g=df[df.condition==condition]
        add(f"{condition}: drift reduction vs Dice advantage",g.feature_drift_reduction,
            g.adaptive_advantage_dice,"ROI, single condition")
    return pd.DataFrame(rows)


def make_plots(conditions,roi,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered=conditions.set_index("condition").reindex(CONDITIONS)
    labels=[s.replace("_severe","").replace("_"," ") for s in CONDITIONS]
    y=ordered.mean_adaptive_advantage.to_numpy()
    low=y-ordered.roi_bootstrap_95ci_low.to_numpy()
    high=ordered.roi_bootstrap_95ci_high.to_numpy()-y
    fig,ax=plt.subplots(figsize=(8.4,4.6))
    ax.errorbar(np.arange(len(y)),y,yerr=np.array([low,high]),fmt="o",capsize=4)
    ax.axhline(0,linewidth=1,linestyle="--")
    ax.set_xticks(range(len(y)),labels,rotation=25,ha="right")
    ax.set_ylabel("Adaptive − frozen Dice")
    ax.set_title("Stage 15: Paired Dice advantage under severe H&E perturbations")
    fig.tight_layout(); fig.savefig(out/"stage16_condition_effects.png",dpi=230); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    ax.scatter(roi.mean_feature_drift_reduction,roi.mean_adaptive_advantage,alpha=.6,s=22)
    ax.axhline(0,linestyle="--",linewidth=1)
    ax.set_xlabel("Frozen − adaptive feature MAE (ROI mean)")
    ax.set_ylabel("Adaptive − frozen Dice (ROI mean)")
    ax.set_title("Feature stabilization vs Dice advantage (205 ROIs)")
    fig.tight_layout(); fig.savefig(out/"stage16_feature_drift_association.png",dpi=230); plt.close(fig)


def write_report(out,overall,folds,conditions,subgroups,assoc,audit):
    x=overall.iloc[0]
    table=lambda d: d.to_markdown(index=False,floatfmt=".5f") if hasattr(d,"to_markdown") else d.to_string(index=False)
    # Avoid optional tabulate dependency; keep markdown tables simple.
    def mdtable(df, cols):
        def fmt(v):
            if isinstance(v, (float,np.floating)): return f"{v:.5f}" if np.isfinite(v) else "NA"
            return str(v)
        rows=["| " + " | ".join(cols) + " |", "|"+"|".join("---" for _ in cols)+"|"]
        rows += ["| " + " | ".join(fmt(r[c]) for c in cols)+" |" for _,r in df.iterrows()]
        return "\n".join(rows)
    parts=["# Stage 16 — Five-fold statistical consolidation and scientific audit", "",
      "**Audit status: PASS** — five complete folds, 205 unique validation ROIs, 1,230 paired severe-condition observations.", "",
      f"**Mean adaptive Dice:** {x.mean_adaptive_dice:.6f}. **Mean frozen Dice:** {x.mean_frozen_dice:.6f}.",
      f"**Mean paired adaptive advantage:** {x.mean_adaptive_advantage:.6f} Dice (ROI-stratified bootstrap 95% percentile CI {x.bootstrap_95ci_low:.6f} to {x.bootstrap_95ci_high:.6f}).",
      f"**Feature drift:** adaptive {x.mean_adaptive_feature_mae:.6f}, frozen {x.mean_frozen_feature_mae:.6f}, relative reduction {x.relative_feature_drift_reduction_pct:.1f}%.",
      "", "## Fold consistency", "",
      mdtable(folds,["fold","n_rois","mean_adaptive_dice","mean_frozen_dice","mean_adaptive_advantage","roi_wins"]),
      "", "## Six predefined severe perturbations", "",
      mdtable(conditions,["condition","mean_adaptive_dice","mean_frozen_dice","mean_adaptive_advantage","roi_bootstrap_95ci_low","roi_bootstrap_95ci_high","roi_wins","holm_adjusted_roi_sign_test_p_exploratory"]),
      "", "## Subgroups (descriptive)","",
      mdtable(subgroups,["slide_type","n_rois","mean_roi_adaptive_advantage","roi_wins"]),
      "", "## Feature-drift associations", "",
      mdtable(assoc[["analysis","n","spearman_rho"]],["analysis","n","spearman_rho"]),
      "", "## Research interpretation and limitations", "",
      "- The counterfactual *freezes percentiles taken from the clean validation image*. This is an oracle reference, **not** a deployable system or a retrained, appropriately matched comparator.",
      "- The RF was trained with adaptive normalization. Changing only inference normalization directly tests the trained pipeline's **sensitivity to that normalization policy**; it does not prove superiority to every alternative architecture or establish full causal mediation.",
      "- The same ROI appears in six perturbation conditions. The primary unit of analysis is the **ROI-mean effect (n=205)**, not 1,230 independent cases.",
      "- ROI identifiers and 'primary/metastatic' labels alone do not establish distinct patients or slides. Slide/patient-level correlation could make ROI-bootstrap confidence intervals and nominal p values **overconfident**. Treat all p values as exploratory until clustering metadata is obtained.",
      "- This is internally cross-validated evidence on one dataset with **synthetic H&E concentration changes**. It is not validation on independent clinical laboratories or scanners.",
      "- Six severe conditions were selected for the mechanism experiment. The four-method Stage-12 benchmark used 12 conditions; don't confuse their aggregate means.",
      "- Per-condition sign tests are Holm-corrected across six conditions, but correction cannot account for unknown slide/patient clustering.",
      "- Feature MAE is averaged across nine features and does not reveal RF feature importance or establish that drift reduction explains the Dice benefit.",
      "", "## Reproducibility", "",
      f"All five Stage-15 integrity files read PASS. Maximum Stage-12 reproduction error: {audit['maximum_adaptive_stage12_abs_error']:.3e}; Stage-11 reproduction error: {audit['maximum_clean_stage11_abs_error']:.3e}.",
      f"Resampling: {audit['bootstrap']['replicates']} stratified ROI bootstrap samples, seed {audit['bootstrap']['seed']}; 41 ROI draws with replacement from each original CV fold.",
      "", "**Next:** obtain reliable patient/slide identifiers if available, inspect worst-case ROIs, and prepare a focused IEEE results section with appropriately qualified inference.", ""]
    (out/"stage16_scientific_report.md").write_text("\n".join(parts), encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    ap.add_argument("--input-dir", type=Path,default=root/"outputs"/"percentile_mechanism")
    ap.add_argument("--out-dir",type=Path,default=root/"outputs"/"stage16_statistical_audit")
    ap.add_argument("--bootstrap",type=int,default=10000)
    ap.add_argument("--seed",type=int,default=13016)
    ap.add_argument("--no-plots",action="store_true")
    args=ap.parse_args()
    require(100<=args.bootstrap<=100000, "--bootstrap must be 100..100000")
    src=args.input_dir.resolve(); out=args.out_dir.resolve()
    require(src.is_dir(),f"Input directory missing: {src}")
    require(src != out, "Input and output directories must differ")
    print(f"Stage16 INPUT: {src}",flush=True)
    df,inventory,integrity=collect_and_validate(src)
    print("AUDIT PASS: all five folds, 205 unique ROIs, 1230 pairs, Stage11/12 reproduction and fold summaries",flush=True)
    roi=by_roi(df)
    ci,condition_ci=bootstrap_stratified(df,roi,args.bootstrap,args.seed)
    overall,folds,conditions,subgroups=create_tables(df,roi,ci,condition_ci)
    associations=build_associations(df,roi)
    worst=roi.sort_values("mean_adaptive_advantage").head(25)
    out.mkdir(parents=True,exist_ok=True)
    dfs={
       "stage16_combined_paired_rows.csv":df,
       "stage16_roi_level.csv":roi,
       "stage16_overall.csv":overall,
       "stage16_by_fold.csv":folds,
       "stage16_by_condition.csv":conditions,
       "stage16_by_slide_type.csv":subgroups,
       "stage16_feature_dice_associations.csv":associations,
       "stage16_lowest_advantage_rois.csv":worst,
    }
    for filename,table in dfs.items():table.to_csv(out/filename,index=False,float_format="%.16g")
    audit={
        "status":"PASS","stage":16,"n_folds":5,"n_rois":205,
        "n_severe_conditions":6,"n_paired_rows":len(df),
        "input_dir":str(src),"source_sha256":inventory,"fold_checks":integrity,
        "maximum_adaptive_stage12_abs_error":float(df.adaptive_stage12_abs_error.max()),
        "maximum_clean_stage11_abs_error":float(df.clean_stage11_abs_error.max()),
        "bootstrap":{"method":"stratified ROI resampling within CV fold","replicates":args.bootstrap,"seed":args.seed,"confidence_level":0.95},
        "roi_independence_unverified":True,
        "patient_or_slide_id_available":False,
        "counterfactual_is_oracle_clean_validation_percentiles":True,
        "caveats":["ROI bootstrap/p-values exploratory without slide/patient clustering metadata",
                   "Frozen clean-image percentiles are an oracle counterfactual, not a deployable comparator",
                   "Same RF was trained with adaptive normalization; no claim of full causal mediation",
                   "Synthetic stain changes on one dataset; not external generalization"],
        "python_version":platform.python_version(),"numpy_version":np.__version__,"pandas_version":pd.__version__,
    }
    (out/"stage16_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    write_report(out,overall,folds,conditions,subgroups,associations,audit)
    if not args.no_plots:
        make_plots(conditions,roi,out)
    print("OVERALL paired Dice advantage: %.6f (95%% ROI bootstrap %.6f .. %.6f)" %
          (overall.mean_adaptive_advantage.iloc[0],ci[0],ci[1]),flush=True)
    print("ROI wins: %d/205; results: %s" %(overall.roi_wins.iloc[0],out),flush=True)
    print("CAUTION: unknown patient/slide grouping; oracle frozen reference; p values exploratory.",flush=True)

if __name__ == "__main__":
    main()
