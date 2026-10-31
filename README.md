# Stain Robustness in Melanoma Nuclei Segmentation — Research Code

This repository is a **release candidate** for the experiments in the IEEE conference manuscript *Investigating the Effects of Image Preprocessing and Feature Scaling on Stain Robustness in Melanoma Nuclei Segmentation*.

## Scientific scope
- 205 matched PUMA melanoma image–nuclei annotation ROIs, five ROI-disjoint folds (164 train / 41 test per fold), seed 13.
- Experiment I: no correction, CLAHE, Macenko, Reinhard; each uses its own clean-trained 500-tree RF, 12 fold-calibrated synthetic H&E conditions.
- Experiment II: one no-correction clean-trained RF per fold, fixed across adaptive and clean-reference-frozen inference-time feature scaling, under six severe perturbations.
- The frozen branch uses *clean held-out ROI information* and is a diagnostic counterfactual, **not** a deployable method or a separately trained frozen-scaling model.
- Patient-disjointness was **not** established; ROC/ROI bootstrap findings are exploratory and do not prove external generalization.

## Original code and stages
The original source files are **unchanged** and kept in `src/` and `legacy_code/`:
- `src/05_calibrate_stain_variation.py` and `src/07_build_enhancement_targets.py`: training-fold calibration and normalization targets.
- `src/10_create_common_training_samples.py`: balanced per-ROI training pixel coordinates, NumPy seed 13.
- `src/11_clean_method_rf_experiment.py` and `src/12_stain_robustness_rf_experiment.py`: Experiment I.
- `src/14_feature_invariance_pilot.py`: helper functions imported by the final Experiment II code; originally a small pilot.
- `src/15_percentile_mechanism_rf.py`: Experiment II.
- `src/16_fivefold_scientific_audit.py`, `src/17_manuscript_figures_analysis.py`, `src/18_export_ieee_figures.py`: statistical consolidation and publication figures.
- `src/20_revision_existing_outputs.py` through `src/24_run_all_revision.py`: review-period analyses (paired tests, nine-feature drift, threshold sensitivity, audits).
- `src/00`–`04`, `06`, `08`–`09`, `13`, `19`, and `src/pilot_stain_perturbation_fixedhed.py`: supporting QC/pilot/inventory scripts; their outputs are **not** additional final manuscript experiments.
- `legacy_code/puma_nuclei_rf_only_cv5_split.py`: nine-feature RF and postprocessing helper implementations imported by the conference scripts; `legacy_code/hybridA_utils.py` supplies metrics. The other two legacy scripts are provenance dependencies.

## Perturbation seed policy
Synthetic H&E strength factors are **deterministic**, not random draws. `src/05` calculates fold-specific mild/severe H/E multipliers from only the training ROIs, while `src/12` and `src/15` use the recorded full-precision multipliers. Thus there is *no perturbation seed*. Seed 13 applies to original ROI cross-validation, training-pixel draws and RF training. `src/15` uses a reproducible derived seed for feature-drift sample positions. Separate analysis seeds are in the scripts: Stage 16 default 13016, Stage 17 default 17017, and review analyses use their own documented seeds. All result intervals are exploratory ROI-level intervals, not patient-level intervals.

## Dataset provenance — important
The project inventory has 206 TIFF filenames, 206 GeoJSON filenames but **205 exact matching pairs**: the TIFF uses `training_set_metastatic_roi_103f`, whereas the GeoJSON uses `training_set_metastatic_roi_103_nuclei.geojson`. The pre-saved 205-ROI split is authoritative. Modern versions of PUMA may have corrected this annotation inconsistency, so **do not recompute a new five-fold split from a different dataset version and represent it as the original experiment**. Obtain the official PUMA public training dataset and verify filenames, image content, and licensing. Original source: https://puma.grand-challenge.org/dataset/ . Do not upload private PUMA test images or uncontrolled data copies to this repository.

## Local folder setup
Source scripts infer `ROOT` as the directory above `src` or `tools`. With raw data for a full rerun, the root should include:
```
01_training_dataset_tif_ROIs/           # Download original official PUMA training images separately
01_training_dataset_geojson_nuclei/     # Download original public nuclei annotations separately
src/
legacy_code/
inputs/splits/cv_splits_seed13_folds5.npz
inputs/training_samples/common_training_pixels_seed13.npz
outputs/stain_calibration/fold_perturbation_parameters.csv
outputs/enhancement_targets/fold01_enhancement_target.json ... fold05...
```

### Environment
Run these commands on the **original experiment machine** and use the **same environment** when possible. `requirements.txt` lists packages inferred from imports but is *not a pinned recreation of the original environment*.
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip freeze > requirements-lock.txt
python --version
```
If the original `.venv` still exists, instead activate that original environment and generate `requirements-lock.txt` there, so the lock represents the environment that actually produced the published results. Do not claim that a new install is the original locked environment.

### Recreate the original-format common training CSV
The original generated `common_training_pixels_seed13.csv` is approximately 62 MB. Only the 3.35 MB compact original NPZ needs to be hosted; the converter below restores the exact original row order and `slide_type` column:
```powershell
python .\tools\restore_common_training_csv.py
```
This utility was added for release convenience and was **not** part of the originally run experiment. It needs the original trusted `common_training_pixels_seed13.npz` and frozen split.

## Quick statistical check (using copied cached outputs)
```powershell
python src/16_fivefold_scientific_audit.py
python src/17_manuscript_figures_analysis.py
python src/18_export_ieee_figures.py
python src/20_revision_existing_outputs.py
python src/23_revision_final_audit.py
```
Stage 23 additionally requires completed outputs of Stages 21 and 22. If not copied, you must run Stage 21 and all five folds of Stage 22 using original PUMA images, annotations and training samples. Stage 22 refits one 500-tree RF per fold and can be slow. Original results and audits should be compared before claiming identical numeric reproduction.

## Full recomputation (computationally intensive)
Generate fold-specific calibration/targets/training pixels, train four independently matched RF pipelines, evaluate twelve perturbations, and run Experiment II:
```powershell
python src/05_calibrate_stain_variation.py
python src/07_build_enhancement_targets.py
python src/10_create_common_training_samples.py
foreach ($m in @('no_correction','clahe','macenko','reinhard')) { python src/11_clean_method_rf_experiment.py --method $m }
foreach ($m in @('no_correction','clahe','macenko','reinhard')) { python src/12_stain_robustness_rf_experiment.py --method $m --fold 0 }
foreach ($fold in 1..5) { python src/15_percentile_mechanism_rf.py --fold $fold }
python src/16_fivefold_scientific_audit.py
python src/17_manuscript_figures_analysis.py
python src/18_export_ieee_figures.py
python src/24_run_all_revision.py
```
Only execute after matching the exact 205 original images/annotations and frozen split, and after reviewing `--help` and existing output files. Several commands are costly and use checkpoints; do not set `--overwrite 1` unless deliberately redoing results.

## Research and data citation
Please cite the PUMA dataset paper (Schuiveling et al., *GigaScience*, 2025) and the official PUMA dataset archive. This repository does not include original PUMA images or nuclei annotations. The original dataset's terms and version should be checked against the archive being used.

## Release status and license
**Release candidate / source audit only.** Source parsing succeeded, but this package was **not fully re-executed** on the original image data and all original output hashes were not validated. The repository owner should check rights to distribute code derived from earlier research, add an appropriate license if authorized, and verify results/configurations before presenting it as a fully reproduced archival release.
