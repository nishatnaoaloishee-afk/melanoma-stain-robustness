"""Stage 22: sequential per-fold threshold robustness, exact Stage15 RF and features.

For each held-out ROI in a fold, evaluate three representative severe conditions
at thresholds 0.30, 0.35, 0.40, 0.45, 0.50 using ONE probability map per branch.
"""
from __future__ import annotations
import argparse, importlib.util, gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

CONDITIONS=('H_high_severe','E_low_severe','HE_high_severe')
THRESHOLDS=(0.30,0.35,0.40,0.45,0.50)
TOLERANCE=1e-6

def require(t,message):
    if not bool(t):raise RuntimeError('STAGE 22 AUDIT FAILED: '+message)

def atomic_csv(d,path):
    tmp=path.with_suffix('.tmp');d.to_csv(tmp,index=False,float_format='%.16g');tmp.replace(path)

def stage15(root):
    src=root/'src'/'15_percentile_mechanism_rf.py'
    spec=importlib.util.spec_from_file_location('stage15_verified',src)
    require(spec is not None and spec.loader is not None,'Stage15 source absent')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    require(mod.ROOT.resolve()==root.resolve(),'Stage15 project-root mismatch')
    return mod

def probability_map(s,rf,pos_col,feat,tissue):
    hh,ww,n=feat.shape
    require(n==9 and tissue.shape==(hh,ww),'Bad probability dimensions')
    matrix=feat.reshape(-1,9)
    indices=np.flatnonzero(tissue.reshape(-1))
    prob=np.zeros(hh*ww,dtype=np.float32)
    for st in range(0,len(indices),s.CHUNK):
        ix=indices[st:st+s.CHUNK]
        prob[ix]=rf.predict_proba(matrix[ix])[:,pos_col].astype(np.float32)
    return prob.reshape(hh,ww)

def score_threshold(s,prob,tissue,gt,t):
    binary=(prob>=t)&tissue
    instances,final=s.binary_to_instances(binary,tissue,min_area=5,max_area=1200,min_peak_distance=2,dist_sigma=1.0)
    m=s.compute_pixel_metrics(final,gt,tissue)
    return float(m['dice'])

def auc_safe(prob,tissue,gt):
    truth=np.asarray(gt[tissue],dtype=np.uint8)
    if truth.min()==truth.max():return float('nan')
    return float(roc_auc_score(truth,prob[tissue]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--fold',type=int,required=True,choices=range(1,6));ap.add_argument('--save-model',action='store_true',help='Optional: persist the reconstructed 500-tree RF; large output')
    args=ap.parse_args();root=args.root.resolve();fold=args.fold
    out=root/'outputs'/'revision_stage22';out.mkdir(parents=True,exist_ok=True)
    csvfile=out/f'fold{fold:02d}_threshold_sensitivity.csv'
    s=stage15(root)
    names,types,tr,va,cfg,alpha,clean_lookup,stage12,samples=s.load_inputs(fold,load_training_pixels=False)
    expected_rois={names[i] for i in va}
    if csvfile.exists():
        existing=pd.read_csv(csvfile,dtype={'roi':str,'condition':str})
        require({'fold','roi','condition','threshold','adaptive_dice','frozen_dice','clean_dice'}.issubset(existing.columns),'Invalid threshold checkpoint schema')
        require(not existing.duplicated(['fold','roi','condition','threshold']).any(),'Duplicate threshold checkpoint')
        require(set(existing.roi).issubset(expected_rois),'Extra ROI in threshold checkpoint')
        require(set(existing.condition).issubset(CONDITIONS),'Extra condition in threshold checkpoint')
        require(set(np.round(existing.threshold,4)).issubset(set(THRESHOLDS)),'Unexpected threshold checkpoint')
        counts=existing.groupby('roi').size()
        require(counts.eq(len(CONDITIONS)*len(THRESHOLDS)).all(),'Partial ROI checkpoint: inspect before resuming')
        for roi,g in existing.groupby('roi'):
            require(set(zip(g.condition,np.round(g.threshold,4)))==set((c,t) for c in CONDITIONS for t in THRESHOLDS),'Incomplete checkpoint threshold grid '+roi)
    else:existing=pd.DataFrame()
    finished=set(existing.roi) if len(existing) else set()
    if finished==expected_rois:
        print(f'STAGE 22 fold {fold} ALREADY COMPLETE: no retraining',flush=True);return
    print(f'STAGE22 fold {fold}/5, completed={len(finished)}/41. Rebuild exact Stage15 RF...',flush=True)
    # Stage15 will load the original common clean training pixels.
    names2,types2,tr2,va2,cfg2,alpha2,clean2,ref2,samples=s.load_inputs(fold,load_training_pixels=True)
    require(names2==names and np.array_equal(tr2,tr) and np.array_equal(va2,va),'Stage15 split changed')
    reference_stats,frozen_features=s.load_pilot_helpers()
    t_start=time.perf_counter()
    rf,pos_col,feature_secs,fit_secs=s.fit_clean_reference_rf(names,tr,cfg,samples)
    del samples;gc.collect()
    if args.save_model:
        import joblib
        dst=out/f'fold{fold:02d}_no_correction_rf.joblib'
        joblib.dump(rf,dst,compress=3)
        print('Saved opt-in RF model',dst,flush=True)
    for index,ii in enumerate(va,start=1):
        roi=names[ii]
        if roi in finished:continue
        with Image.open(s.ROI_DIR/f'{roi}.tif') as im:raw=np.asarray(im.convert('RGB'),dtype=np.uint8)
        tissue=s.clean_reference_tissue_mask_20x(raw)
        gt=s.rasterize_geojson_aligned(str(s.GT_DIR/f'{roi}_nuclei.geojson'),raw.shape[0],raw.shape[1],to_20x=True)
        clean20,_=s.preprocess_for_method(raw,'no_correction',cfg)
        feat_names,anchors,cleanmap=s.check_feature_equivalence(clean20,reference_stats,frozen_features)
        clean_prob=probability_map(s,rf,pos_col,cleanmap,tissue)
        clean_dice=score_threshold(s,clean_prob,tissue,gt,.40)
        reference_clean=float(clean_lookup.loc[roi,'dice'])
        require(abs(clean_dice-reference_clean)<=TOLERANCE,f'Clean baseline fails ROI={roi}: {clean_dice} vs {reference_clean}')
        del cleanmap,clean_prob,clean20
        dec=s.decompose_he(raw)
        roi_rows=[]
        for cond in CONDITIONS:
            ah,ae=alpha[(fold,cond)]
            perturb=s.reconstruct_from_decomposition(dec,alpha_H=ah,alpha_E=ae)
            pert20,_=s.preprocess_for_method(perturb,'no_correction',cfg)
            del perturb
            a_names,amap=s.compute_feature_stack(pert20)
            f_names,fmap=frozen_features(pert20,anchors)
            require(a_names==f_names==feat_names,'Different feature order')
            ap=probability_map(s,rf,pos_col,amap,tissue)
            fp=probability_map(s,rf,pos_col,fmap,tissue)
            del amap,fmap,pert20
            auc_a=auc_safe(ap,tissue,gt);auc_f=auc_safe(fp,tissue,gt)
            old=pd.read_csv(s.OUTDIR/f'fold{fold:02d}_paired_mechanism.csv',dtype={'roi':str,'condition':str})
            old=old[(old.roi==roi)&(old.condition==cond)]
            require(len(old)==1,'Stage15 paired reference missing '+roi+'/'+cond)
            for threshold in THRESHOLDS:
                ad=score_threshold(s,ap,tissue,gt,threshold)
                fr=score_threshold(s,fp,tissue,gt,threshold)
                if threshold==.40:
                    require(abs(ad-float(old.adaptive_dice.iloc[0]))<=TOLERANCE,f'Adaptive 0.40 reproduction failure {roi}/{cond}')
                    require(abs(fr-float(old.frozen_dice.iloc[0]))<=TOLERANCE,f'Frozen 0.40 reproduction failure {roi}/{cond}')
                roi_rows.append({'fold':fold,'roi':roi,'slide_type':types[ii],'condition':cond,'threshold':threshold,'clean_dice':reference_clean,'adaptive_dice':ad,'frozen_dice':fr,'paired_advantage':ad-fr,'adaptive_auc':auc_a,'frozen_auc':auc_f})
            del ap,fp
        existing=pd.concat([existing,pd.DataFrame(roi_rows)],ignore_index=True)
        atomic_csv(existing,csvfile)
        finished.add(roi)
        print(f'Stage22 fold{fold} ROI {index}/41 | PASS clean and 0.40 baseline | checkpoint {len(finished)}/41',flush=True)
        del raw,tissue,gt,dec
        gc.collect()
    del rf;gc.collect()
    expected=41*len(CONDITIONS)*len(THRESHOLDS)
    require(len(existing)==expected,'Threshold fold completion mismatch')
    with (out/f'fold{fold:02d}_audit.json').open('w',encoding='utf-8') as f:json.dump({'status':'PASS','fold':fold,'n_rois':41,'n_conditions':3,'thresholds':THRESHOLDS,'rows':len(existing),'zero_point_four_reproduction':'Both branches checked against Stage15 for all selected ROI-condition pairs, tolerance <=1e-6','clean_reproduction':'Clean Dice checked against Stage11','model_training':'Exact Stage15 500-tree clean RF reconstituted','training_seconds':fit_secs,'wall_seconds':time.perf_counter()-t_start,'probability_auc':'Pixel-level AUC on tissue pixels, exploratory; segmentation Dice includes postprocessing'},f,indent=2)
    print(f'STAGE22 FOLD {fold} PASS',flush=True)

if __name__=='__main__':main()
