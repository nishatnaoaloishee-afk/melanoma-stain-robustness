"""Stage 21: nine-channel drift over 205 x 6, plus five exception ROI descriptors.

Uses exact Stage14/15 feature functions, calibration and random sample policy.
Does not train models. Outputs checkpoint after each ROI.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, gc
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def require(expr, reason):
    if not bool(expr):raise RuntimeError('STAGE 21 AUDIT FAILED: '+reason)


def atomic_csv(df,p):
    tmp=p.with_suffix('.tmp');df.to_csv(tmp,index=False,float_format='%.16g');tmp.replace(p)


def load_stage15(root):
    script=root/'src'/'15_percentile_mechanism_rf.py'
    spec=importlib.util.spec_from_file_location('revision_stage15_exact',script)
    require(spec is not None and spec.loader is not None,'Missing Stage15 source')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    require(mod.ROOT.resolve()==root.resolve(),'Stage15 project root mismatch')
    return mod


def polygon_count(path):
    """Count polygon geometries, not pixels. One polygon is treated as a nuclei annotation."""
    with path.open(encoding='utf-8') as f:data=json.load(f)
    count=0
    for feat in data.get('features',[]):
        geom=feat.get('geometry') or {}
        typ=geom.get('type','')
        coords=geom.get('coordinates') or []
        if typ=='Polygon' and coords:count+=1
        elif typ=='MultiPolygon':count+=sum(bool(poly) for poly in coords)
    return count


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    root=args.root.resolve();out=root/'outputs'/'revision_stage21';out.mkdir(parents=True,exist_ok=True)
    s=load_stage15(root)
    ref_stats,frozen_stack=s.load_pilot_helpers()
    ref=pd.read_csv(root/'outputs'/'stage16_statistical_audit'/'stage16_combined_paired_rows.csv',dtype={'roi':str,'condition':str}).set_index(['fold','roi','condition'])
    require(len(ref)==1230,'Stage16 paired data incomplete')
    driftfile=out/'nine_channel_drift_checkpoint.csv'
    if driftfile.exists():
        done=pd.read_csv(driftfile,dtype={'roi':str,'condition':str})
        require({'fold','roi','condition','feature','adaptive_channel_mae','frozen_channel_mae'}.issubset(done.columns),'Checkpoint schema wrong')
        require(not done.duplicated(['fold','roi','condition','feature']).any(),'Duplicate checkpoint rows')
        full=done.groupby(['fold','roi','condition']).size()
        require((full==9).all(),'Partial 9-channel ROI-condition checkpoint; inspect before continuing')
    else:done=pd.DataFrame()
    finished=set(zip(done.fold,done.roi)) if len(done) else set()
    all_exception=pd.read_csv(root/'outputs'/'stage17_manuscript'/'stage17_five_exception_rois.csv',dtype={'roi':str})
    require(len(all_exception)==5 and all_exception.roi.nunique()==5,'Five exception ROI inventory mismatch')
    exceptions=[]
    for fold in range(1,6):
        names,types,tr,va,cfg,alpha,clean_lookup,stage12,samples=s.load_inputs(fold,load_training_pixels=False)
        require(samples is None,'Unexpected training pixels in feature-only analysis')
        for index,ii in enumerate(va,start=1):
            roi=names[ii]
            if (fold,roi) in finished and roi not in set(all_exception.roi):continue
            with Image.open(s.ROI_DIR/f'{roi}.tif') as im:raw=np.asarray(im.convert('RGB'),dtype=np.uint8)
            tissue=s.clean_reference_tissue_mask_20x(raw)
            clean20,_=s.preprocess_for_method(raw,'no_correction',cfg)
            feat_names,anchors,cleanmap=s.check_feature_equivalence(clean20,ref_stats,frozen_stack)
            require(len(feat_names)==9,'Expected nine feature channels')
            # Exact Stage 15 per-ROI deterministic sampling policy.
            rng=np.random.default_rng(s.SEED+fold*100+index)
            candidates=np.flatnonzero(tissue.reshape(-1))
            require(len(candidates)>=100,'Insufficient tissue: '+roi)
            ids=rng.choice(candidates,size=min(s.FEATURE_PIXELS_PER_ROI,len(candidates)),replace=False)
            baseline=cleanmap.reshape(-1,9)[ids].copy()
            del cleanmap
            if roi in set(all_exception.roi):
                gt_path=s.GT_DIR/f'{roi}_nuclei.geojson'
                gt=s.rasterize_geojson_aligned(str(gt_path),raw.shape[0],raw.shape[1],to_20x=True)
                require(gt.shape==tissue.shape,'GT/tissue shape mismatch '+roi)
                n_poly=polygon_count(gt_path)
                tissue_pixel_count=int(tissue.sum())
                desc=all_exception[all_exception.roi==roi].iloc[0].to_dict()
                exceptions.append({**desc,'tissue_pixels_20x':tissue_pixel_count,'roi_pixels_20x':int(tissue.size),'tissue_fraction':float(tissue.mean()),'nuclei_annotation_polygons':n_poly,'nuclei_density_per_million_tissue_pixels_20x':float(n_poly*1e6/tissue_pixel_count),'ground_truth_nuclei_occupancy_tissue':float((gt&tissue).sum()/tissue_pixel_count)})
            if (fold,roi) not in finished:
                decomp=s.decompose_he(raw);rows=[]
                for cond in s.CONDITIONS:
                    ah,ae=alpha[(fold,cond)]
                    perturbed=s.reconstruct_from_decomposition(decomp,alpha_H=ah,alpha_E=ae)
                    pert20,_=s.preprocess_for_method(perturbed,'no_correction',cfg)
                    del perturbed
                    f_names,ad=s.compute_feature_stack(pert20)
                    fr_names,fr=frozen_stack(pert20,anchors)
                    require(f_names==feat_names==fr_names,'Feature order changed')
                    ad_mae=np.abs(ad.reshape(-1,9)[ids]-baseline).mean(axis=0)
                    fr_mae=np.abs(fr.reshape(-1,9)[ids]-baseline).mean(axis=0)
                    reference=ref.loc[(fold,roi,cond)]
                    require(abs(ad_mae.mean()-float(reference.adaptive_feature_mae))<3e-7,'Adaptive drift replication failed: '+roi+'/'+cond)
                    require(abs(fr_mae.mean()-float(reference.frozen_feature_mae))<3e-7,'Frozen drift replication failed: '+roi+'/'+cond)
                    for j,feat in enumerate(feat_names):
                        rows.append({'fold':fold,'roi':roi,'slide_type':types[ii],'condition':cond,'feature':feat,'adaptive_channel_mae':float(ad_mae[j]),'frozen_channel_mae':float(fr_mae[j]),'channel_drift_reduction':float(fr_mae[j]-ad_mae[j]),'paired_dice_advantage':float(reference.adaptive_advantage_dice)})
                    del pert20,ad,fr
                done=pd.concat([done,pd.DataFrame(rows)],ignore_index=True)
                atomic_csv(done,driftfile)
                finished.add((fold,roi))
                print(f'Stage21 fold{fold} ROI {index}/41: {roi} | saved {len(finished)}/205 ROIs',flush=True)
            del baseline,clean20,raw,tissue
            gc.collect()
    require(len(done)==205*6*9 and done.groupby(['fold','roi','condition']).size().eq(9).all(),'Full 205x6x9 coverage missing')
    # Validate ALL checkpoint rows on resume as well, not just newly computed ones.
    validation=done.groupby(['fold','roi','condition']).agg(ad=('adaptive_channel_mae','mean'),fr=('frozen_channel_mae','mean'))
    for key,row in validation.iterrows():
        r=ref.loc[key]
        require(abs(float(row.ad)-float(r.adaptive_feature_mae))<3e-7,'Checkpoint adaptive drift differs from Stage15: '+str(key))
        require(abs(float(row.fr)-float(r.frozen_feature_mae))<3e-7,'Checkpoint frozen drift differs from Stage15: '+str(key))
    means=done.groupby(['condition','feature'],sort=False).agg(n_rois=('roi','nunique'),adaptive_mae=('adaptive_channel_mae','mean'),frozen_mae=('frozen_channel_mae','mean'),drift_reduction=('channel_drift_reduction','mean')).reset_index()
    atomic_csv(means,out/'channel_drift_by_condition.csv')
    overall=done.groupby('feature',sort=False).agg(n_roi_condition=('roi','size'),adaptive_mae=('adaptive_channel_mae','mean'),frozen_mae=('frozen_channel_mae','mean'),drift_reduction=('channel_drift_reduction','mean')).reset_index()
    atomic_csv(overall,out/'channel_drift_overall.csv')
    from scipy.stats import spearmanr
    corrs=[]
    for (cond,feat),g in done.groupby(['condition','feature']):
        rho,pv=spearmanr(g.channel_drift_reduction,g.paired_dice_advantage)
        corrs.append({'condition':cond,'feature':feat,'n_rois':len(g),'spearman_rho_exploratory':rho,'p_nominal_exploratory':pv})
    atomic_csv(pd.DataFrame(corrs),out/'channel_drift_dice_correlations.csv')
    exceptions=pd.DataFrame(exceptions)
    require(len(exceptions)==5 and exceptions.roi.nunique()==5,'Exception ROI characteristics incomplete')
    atomic_csv(exceptions,out/'five_exception_roi_characteristics.csv')
    with (out/'revision21_audit.json').open('w',encoding='utf-8') as f:json.dump({'status':'PASS','rows':len(done),'unique_rois':205,'conditions':6,'channels':9,'no_rf_training':True,'feature_drift_exact_replication':'Each ROI-condition channel mean reproduces Stage15 average MAE within 3e-7','nuclei_density_unit':'annotations per 10^6 tissue pixels at 20x; not per mm^2 without verified pixel size'},f,indent=2)
    print('STAGE 21 PASS ->',out,flush=True)

if __name__=='__main__':main()
