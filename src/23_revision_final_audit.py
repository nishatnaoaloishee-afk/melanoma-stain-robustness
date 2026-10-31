"""Stage 23: verify all revision outputs and assemble CSV evidence."""
from __future__ import annotations
import argparse, json, platform, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

CONDITIONS=('H_high_severe','E_low_severe','HE_high_severe')
THRESHOLDS=(0.30,0.35,0.40,0.45,0.50)

def require(test,msg):
    if not bool(test):raise RuntimeError('STAGE 23 AUDIT FAILED: '+msg)

def atom_csv(df,p):
    tmp=p.with_suffix('.tmp');df.to_csv(tmp,index=False,float_format='%.16g');tmp.replace(p)

def bootstrap_ci(data, seed=20261, n=10000):
    """Resample ROI identities within original folds, retain three conditions per ROI."""
    roi=data.groupby(['fold','roi'],as_index=False).paired_advantage.mean()
    require(len(roi)==205 and roi.roi.nunique()==205,'Bootstrap requires 205 independent ROI identities')
    rng=np.random.default_rng(seed);means=np.zeros(n)
    for fold,g in roi.groupby('fold'):
        x=g.paired_advantage.to_numpy()
        require(len(x)==41,'Fold size not 41')
        samples=rng.integers(0,len(x),size=(n,len(x)))
        means += x[samples].mean(axis=1)/5.
    return np.quantile(means,[.025,.975])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args()
    root=a.root.resolve(); out=root/'outputs'/'revision_stage23';out.mkdir(parents=True,exist_ok=True)
    src20=root/'outputs'/'revision_stage20'; src21=root/'outputs'/'revision_stage21'; src22=root/'outputs'/'revision_stage22'
    for fname in ('experiment1_paired_tests.csv','experiment2_six_condition_results.csv','calibration_factor_ranges.csv'):
        require((src20/fname).is_file(),'Missing Stage20 '+fname)
    for fname in ('channel_drift_by_condition.csv','channel_drift_overall.csv','channel_drift_dice_correlations.csv','five_exception_roi_characteristics.csv','revision21_audit.json'):
        require((src21/fname).is_file(),'Missing Stage21 '+fname)
    frames=[]
    for fold in range(1,6):
        f=src22/f'fold{fold:02d}_threshold_sensitivity.csv'
        audit=src22/f'fold{fold:02d}_audit.json'
        require(f.is_file() and audit.is_file(),'Missing completed Stage22 fold '+str(fold))
        d=pd.read_csv(f,dtype={'roi':str,'condition':str})
        require(len(d)==41*3*5 and set(d.condition)==set(CONDITIONS),'Incomplete threshold fold '+str(fold))
        require(d.roi.nunique()==41 and (d.fold==fold).all(),'Fold mismatch in Stage22 '+str(fold))
        require(set(np.round(d.threshold,2))==set(THRESHOLDS),'Threshold grid mismatch')
        require(not d.duplicated(['fold','roi','condition','threshold']).any(),'Duplicate threshold rows')
        require(json.loads(audit.read_text(encoding='utf-8'))['status']=='PASS','Incomplete fold audit')
        frames.append(d)
    data=pd.concat(frames,ignore_index=True)
    require(len(data)==3075 and data.roi.nunique()==205,'Expected 205x3x5 threshold data')
    require(data.groupby(['roi','condition']).threshold.nunique().eq(5).all(),'Missing thresholds per ROI-condition')
    orig=pd.read_csv(root/'outputs'/'stage16_statistical_audit'/'stage16_combined_paired_rows.csv',dtype={'roi':str,'condition':str})
    base=data[np.isclose(data.threshold,.4)].set_index(['fold','roi','condition']).sort_index()
    check=orig[orig.condition.isin(CONDITIONS)].set_index(['fold','roi','condition']).sort_index()
    require(base.index.equals(check.index),'Stage22 baseline index mismatch')
    require(np.allclose(base.adaptive_dice,check.adaptive_dice,atol=1e-6),'Stage22 adaptive baseline does not match Stage16')
    require(np.allclose(base.frozen_dice,check.frozen_dice,atol=1e-6),'Stage22 frozen baseline does not match Stage16')
    summary=[]
    for t,g in data.groupby('threshold'):
        ci=bootstrap_ci(g,seed=20261+int(t*100))
        summary.append({'threshold':t,'n_rois':g.roi.nunique(),'n_roi_conditions':len(g),'adaptive_dice':g.adaptive_dice.mean(),'frozen_dice':g.frozen_dice.mean(),'mean_paired_advantage':g.paired_advantage.mean(),'roi_bootstrap_ci_low':ci[0],'roi_bootstrap_ci_high':ci[1],'roi_wins_mean_over_three_conditions':int((g.groupby('roi').paired_advantage.mean()>0).sum()),'adaptive_pixel_auc_mean':g.adaptive_auc.mean(),'frozen_pixel_auc_mean':g.frozen_auc.mean()})
    table=pd.DataFrame(summary).sort_values('threshold')
    atom_csv(table,out/'threshold_sensitivity_overall.csv')
    bycond=data.groupby(['threshold','condition']).agg(n_rois=('roi','nunique'),adaptive_dice=('adaptive_dice','mean'),frozen_dice=('frozen_dice','mean'),paired_advantage=('paired_advantage','mean'),adaptive_auc=('adaptive_auc','mean'),frozen_auc=('frozen_auc','mean')).reset_index()
    atom_csv(bycond,out/'threshold_sensitivity_by_condition.csv')
    atom_csv(data.sort_values(['fold','roi','condition','threshold']),out/'threshold_sensitivity_all_pairs.csv')
    ci=table[table.threshold==.40].iloc[0]
    require(float(ci.mean_paired_advantage)>0,'Baseline selected-conditions advantage not positive')
    # Deliberately do not invent patient or slide mapping: no verified linkage was supplied.
    with np.load(root/'inputs'/'splits'/'cv_splits_seed13_folds5.npz',allow_pickle=True) as splits:
        used=set(map(str,splits['roi_names'].tolist()))
    images=set(p.stem for p in (root/'01_training_dataset_tif_ROIs').glob('*.tif')) | set(p.stem for p in (root/'01_training_dataset_tif_ROIs').glob('*.tiff'))
    require(len(used)==205 and used.issubset(images),'205-subset is not contained in ROI images')
    excluded=sorted(images-used)
    missing_gt=sorted(r for r in used if not (root/'01_training_dataset_geojson_nuclei'/f'{r}_nuclei.geojson').is_file())
    require(not missing_gt,'Missing annotations for retained ROI: '+str(missing_gt))
    provenance_files = {
        'stage12_source': root/'src'/'12_stain_robustness_rf_experiment.py',
        'stage14_source': root/'src'/'14_feature_invariance_pilot.py',
        'stage15_source': root/'src'/'15_percentile_mechanism_rf.py',
        'cv_splits': root/'inputs'/'splits'/'cv_splits_seed13_folds5.npz',
        'fold_stain_parameters': root/'outputs'/'stain_calibration'/'fold_perturbation_parameters.csv',
        'paired_stage16': root/'outputs'/'stage16_statistical_audit'/'stage16_combined_paired_rows.csv',
    }
    provenance = {}
    for name, path in provenance_files.items():
        require(path.is_file(), 'Missing provenance source: '+str(path))
        h=hashlib.sha256()
        with path.open('rb') as fh:
            for block in iter(lambda:fh.read(1024*1024),b''):h.update(block)
        provenance[name]={'relative_path':str(path.relative_to(root)),'sha256':h.hexdigest()}
    report={
        'status':'PASS', 'original_sources_treated_as_read_only':True, 'source_file_sha256':provenance,
        'n_original_roi_images':len(images),'n_retained_rois':len(used),'excluded_roi_names_from_saved_cv_splits':excluded,
        'excluded_roi_reason':'NOT INFERRED. Verify against authoritative PUMA release notes before asserting reason.',
        'patient_slide_mapping':'NOT VERIFIED. ROI folds alone do not establish slide/patient independence.',
        'threshold_conditions':list(CONDITIONS),'thresholds':list(THRESHOLDS),
        'threshold_baseline_reproduces_saved_stage16':'PASS within 1e-6 for all 615 ROI-condition comparisons, both branches',
        'n_threshold_rows':len(data),'n_channel_feature_rows':205*6*9,
        'mean_advantage_threshold_0_40_subset':float(ci.mean_paired_advantage),
        'bootstrap_unit':'ROI; 3 conditions retained together; 41 ROIs resampled within each fixed fold',
        'confidence_intervals_are_exploratory':True,
        'note':'Threshold-sensitivity is a deliberately selected three-condition subset, not a replacement for the six-condition main result.',
    }
    (out/'revision23_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=[
        '# Internal revision terminology',
        '',f'- Cohort: {len(used)} retained ROIs; extra ROI image(s) found: {excluded}. No exclusion reason assumed.',
        '- Original 205-ROI x six-condition Stage16 findings are NOT replaced by threshold sensitivity on three conditions.',
        '- Stage20 contains paired tests, bootstrap intervals, six-condition summary and stain-calibration factors.',
        '- Stage21 contains all 9 channels x 6 conditions x 205 ROIs and five exceptional ROI descriptors.',
        '- Stage22 contains 5 thresholds x 3 severe conditions x 205 ROIs with exact 0.40 reproductions.',
        '', '## Threshold sensitivity (selected conditions; descriptive, exploratory)',
        '', '| Threshold | Adaptive Dice | Frozen Dice | Paired gain | ROI 95% CI |',
        '|---|---:|---:|---:|---|',
    ]
    for r in table.itertuples():lines.append(f'| {r.threshold:.2f} | {r.adaptive_dice:.4f} | {r.frozen_dice:.4f} | {r.mean_paired_advantage:+.4f} | [{r.roi_bootstrap_ci_low:.4f}, {r.roi_bootstrap_ci_high:.4f}] |')
    lines.extend(['','**Limitations:** ROI clustering by patients/slides is unverified. Differences may reflect adaptive-trained RF distribution mismatch. Analyses are synthetic-stain-only and exploratory.'])
    (out/'revision_evidence_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('STAGE 23 PASS ->',out,flush=True)

if __name__=='__main__':main()
