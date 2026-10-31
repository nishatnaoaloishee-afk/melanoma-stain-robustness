#!/usr/bin/env python3
"""Stage 17: statistical summaries and figures for the PUMA stain-robustness study.

Inputs:
    Validated Stage 16 statistical tables and the corresponding Stage 12
    method-specific fold metrics, when available.

Outputs:
    Analysis figures (PNG/SVG), numerical CSV summaries, a machine-readable
    audit record, and descriptions of generated figures.

This script performs analysis of existing results; it does not train models,
modify input data, or generate synthetic histology images.

Example (from the project root):
    python src/17_manuscript_figures_analysis.py
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=17017
CONDITIONS=('H_low_severe','H_high_severe','E_low_severe','E_high_severe','HE_low_severe','HE_high_severe')
METHODS=('no_correction','clahe','macenko','reinhard')
METHOD_DISPLAY={'no_correction':'No correction','clahe':'CLAHE','macenko':'Macenko','reinhard':'Reinhard'}
FOLD_IDS=(1,2,3,4,5)
PALETTE={'adaptive':'#006E68','frozen':'#C96A30','neutral':'#303A4B','grid':'#CBD3D7','primary':'#3D628C','metastatic':'#9A6A53'}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.1,'axes.titlesize':11,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#68707A','axes.labelcolor':'#263441','text.color':'#263441','xtick.color':'#344454','ytick.color':'#344454','savefig.facecolor':'white','figure.facecolor':'white','axes.grid':False,'svg.fonttype':'none'})

def die(s): raise ValueError('STAGE 17 VALIDATION FAILED: '+str(s))
def check(expr,msg):
    if not bool(expr):die(msg)
def same(a,b,msg,tol=5e-9):
    if not np.allclose(a,b,rtol=0,atol=tol,equal_nan=False):die(msg)
def readable_condition(s):return {'H_low_severe':'H lower','H_high_severe':'H higher','E_low_severe':'E lower','E_high_severe':'E higher','HE_low_severe':'H+E lower','HE_high_severe':'H+E higher'}.get(s,s)
def savefig(fig,path):
    fig.savefig(str(path)+'.png',dpi=300,bbox_inches='tight',pad_inches=0.13)
    fig.savefig(str(path)+'.svg',bbox_inches='tight',pad_inches=0.13)
    plt.close(fig)
def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def boot_mean(x,rs,n=10000):
    a=np.asarray(x,float)
    check(len(a)>1,'bootstrap needs at least two ROI values')
    result=np.empty(n)
    for k in range(n):result[k]=a[rs.integers(0,len(a),size=len(a))].mean()
    return np.quantile(result,[.025,.975]).tolist()
def roi_folds_boot(roi,n=10000,seed=SEED):
    rs=np.random.default_rng(seed)
    groups=[g['mean_adaptive_advantage'].to_numpy() for _,g in roi.groupby('fold',sort=True)]
    out=np.zeros(n)
    for k in range(n):
        out[k]=np.mean([g[rs.integers(0,len(g),size=len(g))].mean() for g in groups])
    return np.quantile(out,[.025,.975]).tolist()
def as_float(v):return float(np.asarray(v).item())

def load_stage16(src):
    names=['stage16_combined_paired_rows.csv','stage16_roi_level.csv','stage16_by_fold.csv','stage16_by_condition.csv','stage16_overall.csv','stage16_by_slide_type.csv','stage16_feature_dice_associations.csv']
    for name in names:check((src/name).is_file(),f'missing {src/name}; run Stage 16 first')
    paths={f.replace('stage16_','').replace('.csv',''):src/f for f in names}
    df=pd.read_csv(paths['combined_paired_rows'],dtype={'roi':'string','slide_type':'string','condition':'string'})
    roi=pd.read_csv(paths['roi_level'],dtype={'roi':'string','slide_type':'string'})
    folds=pd.read_csv(paths['by_fold']);cond=pd.read_csv(paths['by_condition']);overall=pd.read_csv(paths['overall']);type_df=pd.read_csv(paths['by_slide_type']);assoc=pd.read_csv(paths['feature_dice_associations'])
    check(len(df)==1230 and df.roi.nunique()==205 and df.condition.nunique()==6,'paired data must be 205 ROIs x 6 severe conditions')
    check(not df.duplicated(['roi','condition']).any(),'duplicate ROI-condition results')
    check(set(df.condition)==set(CONDITIONS),'unexpected severe condition list')
    check(df.groupby('roi').condition.nunique().eq(6).all(),'missing ROI conditions')
    check(set(df.fold.unique())==set(FOLD_IDS),'missing folds')
    check(df.groupby('roi').fold.nunique().eq(1).all(),'ROI appears in multiple folds')
    check(df[['adaptive_dice','frozen_dice','clean_dice']].apply(lambda s:s.between(0,1).all()).all(),'out-of-range Dice')
    check(np.isfinite(df.select_dtypes(include='number').to_numpy()).all(),'nonfinite data')
    same(df.adaptive_dice-df.frozen_dice,df.adaptive_advantage_dice,'Dice advantage arithmetic')
    check(df.adaptive_stage12_abs_error.max()<1e-9 and df.clean_stage11_abs_error.max()<1e-9,'Stage 11/12 reproduction failure')
    check(len(roi)==205 and roi.roi.nunique()==205,'ROI summary incomplete')
    q=df.groupby('roi').adaptive_advantage_dice.mean()
    same(roi.set_index('roi').loc[q.index,'mean_adaptive_advantage'],q.values,'ROI summary mismatch')
    check(len(folds)==5 and len(cond)==6 and len(overall)==1,'Stage 16 table counts incorrect')
    same(overall.mean_adaptive_advantage.iloc[0],df.adaptive_advantage_dice.mean(),'Stage16 main effect incorrect')
    return df,roi,folds,cond,overall,type_df,assoc,paths

def plot_condition_effects(cond,out):
    d=cond.set_index('condition').loc[list(CONDITIONS)].reset_index()
    n=len(d);ys=np.arange(n)[::-1];x=d.mean_adaptive_advantage.to_numpy()
    lo=d.roi_bootstrap_95ci_low.to_numpy();hi=d.roi_bootstrap_95ci_high.to_numpy()
    fig,ax=plt.subplots(figsize=(8.4,4.1));ax.axvline(0,c='#68707A',lw=1)
    ax.errorbar(x,ys,xerr=np.vstack([x-lo,hi-x]),fmt='o',color=PALETTE['adaptive'],ecolor=PALETTE['adaptive'],markersize=6,capsize=3.5,linewidth=1.6)
    ax.set_yticks(ys,[readable_condition(c) for c in d.condition]);ax.set_xlabel('Mean paired Dice advantage: adaptive − frozen')
    for i,row in enumerate(d.itertuples()):ax.text(.188,ys[i],f'{int(row.roi_wins)}/205 ROIs',va='center',ha='left',fontsize=8,color='#415161')
    ax.set_xlim(-.008,.241);ax.set_ylim(-.7,5.7);ax.set_title('A. Paired Dice advantage across severe stain perturbations',loc='left',fontweight='bold',pad=15)
    ax.grid(axis='x',alpha=.25);fig.subplots_adjust(left=.18,right=.97,top=.86,bottom=.29)
    fig.text(.19,.035,'Whiskers: Stage 16 exploratory ROI-bootstrap 95% intervals  |  H: hematoxylin; E: eosin',fontsize=7.7,color='#66757F');savefig(fig,out/'fig17_01_condition_effects')

def plot_roi_pairs(roi,out):
    fig,ax=plt.subplots(figsize=(5.9,5.3))
    for t,color in [('primary',PALETTE['primary']),('metastatic',PALETTE['metastatic'])]:
        a=roi[roi.slide_type==t];ax.scatter(a.mean_frozen_dice,a.mean_adaptive_dice,s=23,alpha=.73,color=color,edgecolors='white',lw=.24,label=t.capitalize())
    ax.plot([0,1],[0,1],'--',lw=1,c='#54616D');ax.set(xlim=(.32,1.0),ylim=(.32,1.0),xlabel='Frozen percentiles: mean perturbed Dice per ROI',ylabel='Adaptive percentiles: mean perturbed Dice per ROI')
    ax.set_aspect('equal','box');ax.grid(alpha=.16);ax.legend(frameon=False,loc='lower right');ax.set_title('B. ROI-level paired Dice: 200/205 favor adaptive',loc='left',fontweight='bold',pad=13)
    ax.text(.03,.98,'Six severe perturbations averaged within each ROI',transform=ax.transAxes,va='top',fontsize=8,color='#65717B')
    fig.tight_layout();savefig(fig,out/'fig17_02_paired_roi_scatter')

def plot_fold_forest(roi,folds,out):
    rs=np.random.default_rng(SEED+1);rows=[]
    for fold,g in roi.groupby('fold'):
        vals=g.mean_adaptive_advantage.to_numpy();ci=boot_mean(vals,rs,n=4000);rows.append((int(fold),vals.mean(),*ci))
    t=pd.DataFrame(rows,columns=['fold','mean','ci_low','ci_high']);ys=np.arange(5)[::-1]
    fig,ax=plt.subplots(figsize=(7.2,3.5));ax.axvline(0,c='#777777',linestyle='--',lw=.9)
    ax.errorbar(t['mean'],ys,xerr=np.vstack([t['mean']-t.ci_low,t.ci_high-t['mean']]),fmt='o',color=PALETTE['adaptive'],capsize=4,markersize=6)
    ax.set_yticks(ys,[f'Fold {k} (n=41)' for k in t.fold]);ax.set(xlabel='Mean paired Dice advantage (adaptive − frozen)',xlim=(-.012,.16));ax.grid(axis='x',alpha=.21)
    ax.set_title('C. Paired Dice advantage across the five validation folds',loc='left',fontweight='bold',pad=12)
    ax.text(0,-.24,'Whiskers: per-fold exploratory ROI-bootstrap 95% intervals',transform=ax.transAxes,fontsize=7.8,color='#68737D')
    fig.subplots_adjust(left=.22,right=.98,bottom=.28,top=.87);savefig(fig,out/'fig17_03_fold_consistency')
    t.to_csv(out/'stage17_fold_intervals.csv',index=False,float_format='%.10g')

def plot_feature_drift(df,out):
    t=df.groupby('condition').agg(adaptive=('adaptive_feature_mae','mean'),frozen=('frozen_feature_mae','mean')).reindex(list(CONDITIONS))
    inds=np.arange(len(t));w=.35
    fig,ax=plt.subplots(figsize=(8.1,3.7));ax.bar(inds-w/2,t.adaptive,width=w,color=PALETTE['adaptive'],label='Adaptive');ax.bar(inds+w/2,t.frozen,width=w,color=PALETTE['frozen'],label='Frozen')
    ax.set_xticks(inds,[readable_condition(c) for c in t.index]);ax.set_ylabel('Mean absolute change of nine features');ax.set_title('D. Mean feature drift by inference-time scaling policy',loc='left',fontweight='bold',pad=12)
    ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,ncol=2);ax.set_ylim(0,max(t.frozen)*1.21);fig.tight_layout();savefig(fig,out/'fig17_04_feature_drift')

def plot_mechanism_drift_relation(roi,out):
    fig,ax=plt.subplots(figsize=(6.2,4.8))
    for t,color in [('primary',PALETTE['primary']),('metastatic',PALETTE['metastatic'])]:
        g=roi[roi.slide_type==t];ax.scatter(g.mean_feature_drift_reduction,g.mean_adaptive_advantage,s=22,alpha=.65,color=color,label=t.capitalize(),edgecolor='white',linewidth=.2)
    sp=stats.spearmanr(roi.mean_feature_drift_reduction,roi.mean_adaptive_advantage)
    ax.axhline(0,color='#7C868B',linewidth=1,linestyle='--');ax.set(xlabel='Frozen − adaptive feature MAE, averaged within ROI',ylabel='Adaptive − frozen Dice, averaged within ROI');ax.legend(frameon=False)
    ax.text(.03,.98,f'Spearman ρ = {sp.statistic:+.3f}  |  n = 205 ROIs',transform=ax.transAxes,va='top',fontsize=8,backgroundcolor='white')
    ax.set_title('E. ROI-level feature drift reduction versus Dice advantage',loc='left',fontweight='bold',pad=13)
    ax.grid(alpha=.15);fig.tight_layout();savefig(fig,out/'fig17_05_drift_vs_dice')

def plot_exceptions(df,roi,out):
    worst=roi.sort_values('mean_adaptive_advantage').head(5)
    pv=df[df.roi.isin(worst.roi)].pivot(index='roi',columns='condition',values='adaptive_advantage_dice').loc[worst.roi,list(CONDITIONS)]
    fig,ax=plt.subplots(figsize=(9.5,3.4));lim=max(.16,np.abs(pv.values).max());im=ax.imshow(pv.values,aspect='auto',cmap='BrBG',vmin=-lim,vmax=lim)
    for i in range(len(pv)):
        for j in range(6):ax.text(j,i,f'{pv.iloc[i,j]:+.3f}',ha='center',va='center',fontsize=7.5,color='black' if abs(pv.iloc[i,j])<lim*.65 else 'white')
    ax.set_xticks(range(6),[readable_condition(c) for c in CONDITIONS]);ax.set_yticks(range(5),[r.replace('training_set_','').replace('_roi_',' #') for r in pv.index]);ax.set_title('F. Five ROIs with the lowest mean adaptive advantage',loc='left',fontweight='bold',pad=12)
    ax.tick_params(length=0);fig.colorbar(im,ax=ax,fraction=.035,pad=.018,label='Adaptive − frozen Dice');fig.tight_layout();savefig(fig,out/'fig17_06_exceptions')
    out_tbl=worst[['fold','roi','slide_type','clean_dice','mean_adaptive_dice','mean_frozen_dice','mean_adaptive_advantage','conditions_adaptive_wins']]
    out_tbl.to_csv(out/'stage17_five_exception_rois.csv',index=False,float_format='%.10g')
    # Exception tables summarize numeric measurements; image overlays are not created here.
    k=df[df.roi.isin(worst.roi)].copy();k['manual_review_priority']=np.where(k.adaptive_advantage_dice<0,'Adaptive worse','Adaptive better')
    k.sort_values(['roi','adaptive_advantage_dice']).to_csv(out/'stage17_exception_condition_detail.csv',index=False,float_format='%.10g')

def plot_precision_recall(df,out):
    a=df.groupby('condition').agg(adaptive_precision=('adaptive_precision','mean'),frozen_precision=('frozen_precision','mean'),adaptive_recall=('adaptive_recall','mean'),frozen_recall=('frozen_recall','mean')).loc[list(CONDITIONS)]
    fig,axes=plt.subplots(1,2,figsize=(10.1,3.5),sharey=True)
    for ax,label,suff in zip(axes,['Precision','Recall'],['precision','recall']):
        x=np.arange(len(a));ax.plot(x,a[f'adaptive_{suff}'],'o-',label='Adaptive',color=PALETTE['adaptive'],lw=1.75,ms=5)
        ax.plot(x,a[f'frozen_{suff}'],'s--',label='Frozen',color=PALETTE['frozen'],lw=1.75,ms=4)
        ax.set_xticks(x,[readable_condition(c) for c in CONDITIONS],rotation=25,ha='right');ax.set_title(label,loc='left',fontweight='bold');ax.grid(axis='y',alpha=.2);ax.set_ylim(0,1)
    axes[0].set_ylabel('ROI-macro mean');axes[1].legend(frameon=False,loc='lower left');fig.suptitle('G. Precision and recall by perturbation condition',x=.06,ha='left',fontweight='bold',fontsize=11)
    fig.tight_layout();savefig(fig,out/'fig17_07_precision_recall')
    a.reset_index().to_csv(out/'stage17_precision_recall_by_condition.csv',index=False,float_format='%.10g')

def sensitivity(df,roi,out):
    rows=[]
    x=roi.mean_adaptive_advantage.to_numpy();
    rows.append(('All conditions',205,x.mean(),np.median(x),stats.trim_mean(x,.10),int((x>0).sum())))
    for omit in CONDITIONS:
        temp=df[df.condition!=omit].groupby('roi').adaptive_advantage_dice.mean().values
        rows.append(('Exclude '+omit,205,temp.mean(),np.median(temp),stats.trim_mean(temp,.10),int((temp>0).sum())))
    for omit in FOLD_IDS:
        temp=roi[roi.fold!=omit].mean_adaptive_advantage.values
        rows.append((f'Exclude fold {omit}',len(temp),temp.mean(),np.median(temp),stats.trim_mean(temp,.10),int((temp>0).sum())))
    t=pd.DataFrame(rows,columns=['scenario','n_rois','mean_advantage','median_advantage','trimmed_10pct_mean','n_positive_rois'])
    t.to_csv(out/'stage17_sensitivity_analysis.csv',index=False,float_format='%.10g')
    return t

def optional_stage12_benchmark(stage12,base,folds,out):
    names=['H_low_mild','H_low_severe','H_high_mild','H_high_severe','E_low_mild','E_low_severe','E_high_mild','E_high_severe','HE_low_mild','HE_low_severe','HE_high_mild','HE_high_severe']
    expected=[(fold,roi) for fold,roi in zip(base.fold,base.roi)]
    inputs=[stage12/m/'folds'/f'fold{fold:02d}_robustness_metrics.csv' for m in METHODS for fold in FOLD_IDS]
    missing=[str(p) for p in inputs if not p.is_file()]
    if missing:
        return {'status':'SKIPPED','reason':'Not all 20 Stage12 fold metric files were present. No partial four-method comparison was generated.','missing':missing}
    frames=[]
    for method in METHODS:
        chunks=[]
        for f in FOLD_IDS:
            path=stage12/method/'folds'/f'fold{f:02d}_robustness_metrics.csv'
            d=pd.read_csv(path,dtype={'roi':'string','condition':'string'})
            check(len(d)==492 and d.roi.nunique()==41 and d.condition.nunique()==12 and not d.duplicated(['roi','condition']).any(),f'Stage12 invalid fold file: {path}')
            check(d.fold.nunique()==1 and int(d.fold.iloc[0])==f,f'Stage12 wrong fold in {path}')
            check(set(d.condition)==set(names),f'Stage12 condition list differs in {path}')
            check(all(c in d.columns for c in ('clean_dice','dice','dice_loss','fold','roi','condition')),f'Stage12 missing columns {path}')
            same(d.clean_dice-d.dice,d.dice_loss,f'Stage12 loss arithmetic {path}')
            check(d[['clean_dice','dice']].apply(lambda s:s.between(0,1).all()).all(),f'Stage12 out-of-range Dice {path}')
            if method=='no_correction':
                got=d[d.condition.isin(CONDITIONS)].set_index(['fold','roi','condition']).sort_index()['dice']
                expected=base[(base.fold==f)].set_index(['fold','roi','condition']).sort_index()['adaptive_dice']
                check(got.index.equals(expected.index),f'Stage12 vs Stage15 ROI-condition keys differ: {path}')
                same(got.values,expected.values,f'Stage12 no_correction vs Stage15 adaptive severe mismatch: {path}',tol=5e-9)
            chunks.append(d)
        d=pd.concat(chunks,ignore_index=True)
        check(len(d)==2460 and d.roi.nunique()==205,f'Stage12 invalid coverage {method}')
        key=set(zip(d.fold,d.roi,d.condition));check(len(key)==2460,f'Stage12 duplicate {method}')
        check(d.groupby('roi').condition.nunique().eq(12).all(),f'Stage12 incomplete ROI coverage {method}')
        d['method']=method;frames.append(d[['method','fold','roi','condition','clean_dice','dice','dice_loss']])
    d=pd.concat(frames,ignore_index=True)
    # Pair identical held-out ROI-condition observations across pipelines.
    ref=set(zip(frames[0].fold,frames[0].roi,frames[0].condition))
    for g in frames[1:]:check(set(zip(g.fold,g.roi,g.condition))==ref,f'Stage12 nonmatching ROI-condition support: {g.method.iloc[0]}')
    t=d.groupby('method').agg(n=('dice','size'),clean_dice=('clean_dice','mean'),perturbed_dice=('dice','mean'),dice_loss=('dice_loss','mean')).loc[list(METHODS)].reset_index()
    t['method_name']=t.method.map(METHOD_DISPLAY)
    t.to_csv(out/'stage17_stage12_four_method_descriptive.csv',index=False,float_format='%.10g')
    pv=d.pivot(index=['fold','roi','condition'],columns='method',values='dice_loss')
    roi_pv=pv.groupby(level=['fold','roi']).mean()
    lines=[];rs=np.random.default_rng(SEED+2)
    for method in METHODS[1:]:
        delta=(roi_pv[method]-roi_pv['no_correction']).to_numpy()
        ci=boot_mean(delta,rs,n=10000)
        lines.append({'method':method,'n_rois':len(delta),'mean_extra_dice_loss_vs_no_correction':delta.mean(),'ci_low_exploratory':ci[0],'ci_high_exploratory':ci[1],'roi_more_loss':int((delta>0).sum()),'roi_less_loss':int((delta<0).sum())})
    pair=pd.DataFrame(lines);pair.to_csv(out/'stage17_stage12_roi_paired_method_differences.csv',index=False,float_format='%.10g')
    xx=np.arange(4);w=.37;fig,axes=plt.subplots(1,2,figsize=(10.2,3.6))
    axes[0].bar(xx-w/2,t.clean_dice,width=w,color='#9AA3AA',label='Clean');axes[0].bar(xx+w/2,t.perturbed_dice,width=w,color=PALETTE['adaptive'],label='Perturbed')
    axes[0].set_ylim(.70,.84);axes[0].set_ylabel('Mean Dice');axes[0].set_title('Clean and perturbed performance',fontweight='bold');axes[0].legend(frameon=False,fontsize=8)
    axes[1].bar(xx,t.dice_loss,color=['#006E68','#628E89','#C78E58','#945B4F']);axes[1].set_ylabel('Mean Dice degradation');axes[1].set_title('Stain robustness (all 12 conditions)',fontweight='bold')
    for a in axes:a.set_xticks(xx,t.method_name,rotation=16,ha='right');a.grid(axis='y',alpha=.18)
    fig.suptitle('H. Four method-matched RF pipelines (205 ROIs each)',x=.02,ha='left',fontsize=11,fontweight='bold');fig.tight_layout();savefig(fig,out/'fig17_08_four_method_benchmark')
    return {'status':'PASS','n_rows':len(d),'means':t.to_dict(orient='records'),'paired_differences':pair.to_dict(orient='records'),'source_file_sha256':{str(p):hash_file(p) for p in inputs}}

def write_outputs(src, stage12, out, df, roi, folds, cond, overall,
                  subgroups, assoc, inputs, benchmark, sens, ci):
    """Save the analysis audit and descriptions of generated figures.

    Numeric summary tables and figure files are written by the analysis
    functions above. This function does not change any experimental result.
    """
    overall_dice = float(overall.iloc[0].mean_adaptive_advantage)
    lowest_advantage = roi.nsmallest(5, 'mean_adaptive_advantage')
    figure_files = sorted(path.name for path in out.glob('fig17_*.png'))

    audit = {
        'stage': '17',
        'status': 'PASS',
        'n_folds': 5,
        'n_rois': 205,
        'n_roi_condition_rows': 1230,
        'mean_adaptive_dice': float(df.adaptive_dice.mean()),
        'mean_frozen_dice': float(df.frozen_dice.mean()),
        'mean_adaptive_advantage_dice': overall_dice,
        'fold_stratified_roi_bootstrap_95ci': ci,
        'mean_feature_mae_adaptive': float(df.adaptive_feature_mae.mean()),
        'mean_feature_mae_frozen': float(df.frozen_feature_mae.mean()),
        'roi_adaptive_wins': int((roi.mean_adaptive_advantage > 0).sum()),
        'roi_negative_mean_advantage': lowest_advantage[
            lowest_advantage.mean_adaptive_advantage < 0
        ].roi.tolist(),
        'stage12_benchmark': benchmark,
        'figures': figure_files,
        'input_csv_sha256': {
            name: hash_file(path) for name, path in inputs.items()
        },
        'limitations': [
            'Single-cohort evaluation with synthetic H&E perturbations.',
            'Frozen scaling uses clean held-out ROI percentile anchors.',
            'The RF was trained with adaptive scaling; the frozen branch '
            'introduces a training-inference feature mismatch.',
            'CV splits are ROI-disjoint; patient-disjointness was not '
            'independently verified.',
            'ROI-bootstrap intervals are exploratory.',
            'Subgroup summaries are descriptive and are not external '
            'validation results.',
            'Aggregate feature-drift reduction does not establish a '
            'causal explanation for the paired Dice advantage.',
        ],
        'case_verification': (
            'Patient identities and patient-level independence could not '
            'be verified using the available split metadata.'
        ),
    }
    (out / 'stage17_audit.json').write_text(
        json.dumps(audit, indent=2, default=str), encoding='utf-8'
    )

    figure_descriptions = """# Stage 17: generated figure descriptions

The images in this directory are based on validated numerical experiment
outputs. These figure identifiers refer to Stage 17 output files and do not
necessarily match figure numbers in the conference manuscript.

- `fig17_01_condition_effects`: Mean adaptive-minus-frozen Dice advantage
  for six severe H&E perturbations, with exploratory Stage 16 ROI-bootstrap
  confidence intervals and the number of ROIs with positive paired effects.
- `fig17_02_paired_roi_scatter`: Mean perturbed Dice for each held-out ROI
  under the two scaling branches, averaged across the six severe conditions.
  Colors denote primary and metastatic melanoma; the dashed line is identity.
- `fig17_03_fold_consistency`: Mean paired Dice advantage within each fixed
  validation fold, with exploratory within-fold ROI-bootstrap intervals.
- `fig17_04_feature_drift`: Mean absolute change across nine image-derived
  feature channels for adaptive and frozen scaling, by perturbation condition.
- `fig17_05_drift_vs_dice`: ROI-level mean drift reduction and paired Dice
  advantage, with the corresponding Spearman rank correlation.
- `fig17_06_exceptions`: Paired Dice differences for the five ROIs with the
  lowest mean adaptive advantage, shown separately for each condition.
- `fig17_07_precision_recall`: Mean pixel-level segmentation precision and
  recall for the two branches, by perturbation condition.
- `fig17_08_four_method_benchmark`: Descriptive comparison of the four
  method-matched preprocessing–RF pipelines. This figure is generated only
  if all required Stage 12 fold-level metric files are present.

Inference is based on held-out ROI observations. The six conditions for each
ROI are repeated observations, not independent samples. The frozen branch
uses reference statistics from the corresponding clean ROI and the same RF
trained with adaptive scaling.
"""
    (out / 'stage17_figure_captions.md').write_text(
        figure_descriptions, encoding='utf-8'
    )


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    ap.add_argument('--stage16-dir',type=Path,default=root/'outputs'/'stage16_statistical_audit')
    ap.add_argument('--stage12-dir',type=Path,default=root/'outputs'/'stain_robustness')
    ap.add_argument('--out-dir',type=Path,default=root/'outputs'/'stage17_manuscript')
    args=ap.parse_args()
    src=args.stage16_dir.resolve(); stage12=args.stage12_dir.resolve();out=args.out_dir.resolve()
    check(out!=src and out!=stage12,'output cannot overwrite Stage16 or Stage12 input folders')
    df,roi,folds,cond,overall,subgroups,assoc,inputs=load_stage16(src)
    check((df[['adaptive_dice','frozen_dice','adaptive_feature_mae','frozen_feature_mae']].notna().all()).all(),'missing core mechanism values')
    out.mkdir(parents=True,exist_ok=True)
    print('STAGE 17 | 5-fold inputs verified. Generating publication figures...',flush=True)
    ci=roi_folds_boot(roi)
    same(ci,np.asarray([overall.bootstrap_95ci_low.iloc[0],overall.bootstrap_95ci_high.iloc[0]]),'Stage 17 bootstrap does not match Stage 16',tol=.003)
    plot_condition_effects(cond,out);plot_roi_pairs(roi,out);plot_fold_forest(roi,folds,out)
    plot_feature_drift(df,out);plot_mechanism_drift_relation(roi,out);plot_exceptions(df,roi,out);plot_precision_recall(df,out)
    sens=sensitivity(df,roi,out)
    benchmark=optional_stage12_benchmark(stage12,df,folds,out)
    write_outputs(src,stage12,out,df,roi,folds,cond,overall,subgroups,assoc,inputs,benchmark,sens,ci)
    print(f'PASS | {len(roi)} ROIs | {len(df)} paired conditions | advantage={df.adaptive_advantage_dice.mean():+.6f}',flush=True)
    print(f'Stage 12 four-method comparison: {benchmark["status"]}',flush=True)
    print('Outputs: '+str(out),flush=True)

if __name__=='__main__':main()
