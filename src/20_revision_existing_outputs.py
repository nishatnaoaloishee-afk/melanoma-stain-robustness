"""Stage 20: read-only analysis of validated Stage 12, 16, 17 and calibration CSVs."""
from __future__ import annotations
import argparse, json, platform, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

METHODS = ('no_correction', 'clahe', 'macenko', 'reinhard')
CONDITIONS = ('H_low_severe', 'H_high_severe', 'E_low_severe', 'E_high_severe', 'HE_low_severe', 'HE_high_severe')
ALL_CONDITIONS = ('H_low_mild','H_low_severe','H_high_mild','H_high_severe','E_low_mild','E_low_severe','E_high_mild','E_high_severe','HE_low_mild','HE_low_severe','HE_high_mild','HE_high_severe')


def require(test, message):
    if not bool(test):
        raise RuntimeError('STAGE 20 AUDIT FAILED: ' + message)


def atomic_csv(frame, path):
    tmp = path.with_suffix('.tmp')
    frame.to_csv(tmp, index=False, float_format='%.16g')
    tmp.replace(path)


def paired_perm_p(delta, seed=20260, iterations=50000, chunk=2500):
    """Two-sided paired sign-flip Monte Carlo, with +1 correction."""
    x = np.asarray(delta, dtype=np.float64)
    obs = abs(float(x.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, iterations, chunk):
        count = min(chunk, iterations - start)
        signs = rng.integers(0, 2, size=(count,len(x)), dtype=np.int8)*2 - 1
        means = np.abs(signs @ x / len(x))
        exceed += int(np.count_nonzero(means >= obs - 1e-15))
    return (exceed + 1)/(iterations + 1)


def roi_boot_ci(rows, seed, iterations=10000):
    rng = np.random.default_rng(seed)
    groups = [g.delta.to_numpy(dtype=float) for _,g in rows.groupby('fold',sort=True)]
    require([len(g) for g in groups] == [41]*5, 'Expected 41 ROIs per fold')
    samples = np.zeros(iterations, dtype=np.float64)
    for group in groups:
        idx = rng.integers(0,len(group),size=(iterations,len(group)))
        samples += group[idx].mean(axis=1) / 5.0
    return np.quantile(samples,[.025,.975])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args()
    root=args.root.resolve()
    out=root/'outputs'/'revision_stage20';out.mkdir(parents=True,exist_ok=True)
    print('Stage 20: validate Stage 12 four-method metrics...',flush=True)
    frames=[]
    for method in METHODS:
        for fold in range(1,6):
            path=root/'outputs'/'stain_robustness'/method/'folds'/f'fold{fold:02d}_robustness_metrics.csv'
            require(path.is_file(),f'Missing {path}')
            d=pd.read_csv(path,dtype={'roi':str,'condition':str})
            require(len(d)==492 and d.roi.nunique()==41,'Incorrect Stage12 shape: '+str(path))
            require(set(d.condition)==set(ALL_CONDITIONS),'Unexpected conditions: '+str(path))
            require(not d.duplicated(['roi','condition']).any(),'Duplicate Stage12 pairs: '+str(path))
            require({'fold','roi','condition','dice','clean_dice','dice_loss'}.issubset(d.columns),'Missing Stage12 columns')
            require((d.fold==fold).all(),'Fold mismatch '+str(path))
            require(np.allclose(d.clean_dice-d.dice,d.dice_loss,atol=1e-9),'Dice loss arithmetic incorrect: '+str(path))
            d['method']=method
            frames.append(d[['method','fold','roi','condition','dice','clean_dice','dice_loss']])
    all_=pd.concat(frames,ignore_index=True)
    require(len(all_)==9840, 'Expected 4 x 205 x 12 observations')
    bases=all_[all_.method=='no_correction'].set_index(['fold','roi','condition'])
    diffs=[]
    summaries=[]
    for method in METHODS:
        d=all_[all_.method==method].set_index(['fold','roi','condition'])
        require(d.index.is_unique and set(d.index)==set(bases.index),'Unpaired methods: '+method)
        summaries.append(dict(method=method,clean_dice=float(d.clean_dice.mean()),perturbed_dice=float(d.dice.mean()),dice_degradation=float(d.dice_loss.mean())))
        if method=='no_correction':continue
        paired=(d.dice_loss-bases.dice_loss).rename('delta').reset_index().groupby(['fold','roi'],as_index=False).delta.mean()
        require(len(paired)==205,'Expected 205 paired ROI means')
        x=paired.delta.to_numpy(dtype=float)
        ci=roi_boot_ci(paired,seed=20260+len(diffs))
        wil=stats.wilcoxon(x,alternative='two-sided',zero_method='wilcox',method='auto')
        perm=paired_perm_p(x,seed=1800+len(diffs))
        diffs.append(dict(method=method,n_rois=len(x),mean_additional_degradation=float(x.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),wilcoxon_p_two_sided=float(wil.pvalue),permutation_p_two_sided=float(perm),roi_more_loss=int((x>0).sum()),roi_less_loss=int((x<0).sum()),roi_ties=int((x==0).sum())))
        atomic_csv(paired,out/f'paired_roi_deltas_{method}.csv')
    table=pd.DataFrame(diffs)
    # Holm correction for the family of three method comparisons.
    for col in ('wilcoxon_p_two_sided','permutation_p_two_sided'):
        ps=np.asarray(table[col].values,dtype=float)
        order=np.argsort(ps)
        holm=np.zeros(len(ps),dtype=float)
        running=0.0
        for rank,idx in enumerate(order):
            running=max(running,(len(ps)-rank)*float(ps[idx]))
            holm[idx]=min(1.0,running)
        table[col.replace('_p_two_sided','_holm_adjusted_p')]=holm
    atomic_csv(pd.DataFrame(summaries),out/'experiment1_method_summary.csv')
    atomic_csv(table,out/'experiment1_paired_tests.csv')
    print('Stage 20: validate Stage 16 condition results...',flush=True)
    stage16=root/'outputs'/'stage16_statistical_audit'
    paired=pd.read_csv(stage16/'stage16_combined_paired_rows.csv',dtype={'roi':str,'condition':str})
    require(len(paired)==1230 and paired.roi.nunique()==205 and set(paired.condition)==set(CONDITIONS),'Stage16 pairs do not match 205 x 6')
    no=all_[(all_.method=='no_correction') & (all_.condition.isin(CONDITIONS))].set_index(['fold','roi','condition']).sort_index()
    orig=paired.set_index(['fold','roi','condition']).sort_index()
    require(no.index.equals(orig.index),'Stage12 vs Stage16 pairing mismatch')
    require(np.allclose(no.dice,orig.adaptive_dice,atol=1e-8),'Stage12 vs Stage16 adaptive Dice mismatch')
    result=paired.groupby('condition',sort=False).agg(n_rois=('roi','nunique'),clean_dice=('clean_dice','mean'),adaptive_dice=('adaptive_dice','mean'),frozen_dice=('frozen_dice','mean'),adaptive_degradation=('adaptive_dice_loss','mean'),frozen_degradation=('frozen_dice_loss','mean'),paired_advantage=('adaptive_advantage_dice','mean'),adaptive_feature_drift_mae=('adaptive_feature_mae','mean'),frozen_feature_drift_mae=('frozen_feature_mae','mean')).reindex(CONDITIONS).reset_index()
    atomic_csv(result,out/'experiment2_six_condition_results.csv')
    corr=pd.read_csv(stage16/'stage16_feature_dice_associations.csv')
    corr.to_csv(out/'existing_feature_drift_associations.csv',index=False)
    print('Stage 20: validate and summarize calibration...',flush=True)
    cal=pd.read_csv(root/'outputs'/'stain_calibration'/'fold_perturbation_parameters.csv')
    require({'fold','condition','alpha_H','alpha_E'}.issubset(cal.columns),'Calibration fields missing')
    selected=cal[cal.condition.isin(ALL_CONDITIONS)].copy()
    require(len(selected)==60 and not selected.duplicated(['fold','condition']).any(),'Expected all 12 conditions x 5 folds')
    require(selected[['alpha_H','alpha_E']].apply(np.isfinite).all().all(),'Invalid stain factors')
    require(set(selected.fold)==set(range(1,6)),'Calibration folds missing')
    selected.sort_values(['condition','fold']).to_csv(out/'calibration_factors_all_folds.csv',index=False,float_format='%.16g')
    ag=selected.groupby('condition').agg(h_min=('alpha_H','min'),h_max=('alpha_H','max'),e_min=('alpha_E','min'),e_max=('alpha_E','max'),n_folds=('fold','nunique')).reindex(ALL_CONDITIONS).reset_index()
    atomic_csv(ag,out/'calibration_factor_ranges.csv')
    try:
        import sklearn, scipy, skimage, PIL
        versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'scikit-image':skimage.__version__,'Pillow':PIL.__version__}
    except ImportError as exc:
        versions={'environment_collection_error':str(exc)}
    with (out/'revision20_audit.json').open('w',encoding='utf-8') as fh:
        json.dump({'status':'PASS','unit_of_analysis':'205 ROIs averaged over 12 conditions','training_unchanged':True,'paired_tests':'Wilcoxon signed-rank + Monte Carlo sign-flip; exploratory, Holm adjusted','bootstrap':'10,000 fold-stratified ROI resamples; percentile CIs','conditions':list(CONDITIONS),'versions_of_this_revision_run_not_necessarily_original_run':versions},fh,indent=2)
    print('STAGE 20 PASS ->',out,flush=True)

if __name__=='__main__':main()
