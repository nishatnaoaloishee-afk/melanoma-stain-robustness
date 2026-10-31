# Stage 16 — Five-fold statistical consolidation and scientific audit

**Audit status: PASS** — five complete folds, 205 unique validation ROIs, 1,230 paired severe-condition observations.

**Mean adaptive Dice:** 0.795628. **Mean frozen Dice:** 0.722657.
**Mean paired adaptive advantage:** 0.072970 Dice (ROI-stratified bootstrap 95% percentile CI 0.068073 to 0.077832).
**Feature drift:** adaptive 0.015163, frozen 0.068041, relative reduction 77.7%.

## Fold consistency

| fold | n_rois | mean_adaptive_dice | mean_frozen_dice | mean_adaptive_advantage | roi_wins |
|---|---|---|---|---|---|
| 1.00000 | 41.00000 | 0.79400 | 0.72609 | 0.06791 | 40.00000 |
| 2.00000 | 41.00000 | 0.78359 | 0.72123 | 0.06236 | 40.00000 |
| 3.00000 | 41.00000 | 0.79816 | 0.71029 | 0.08787 | 41.00000 |
| 4.00000 | 41.00000 | 0.79059 | 0.72091 | 0.06968 | 39.00000 |
| 5.00000 | 41.00000 | 0.81180 | 0.73477 | 0.07703 | 40.00000 |

## Six predefined severe perturbations

| condition | mean_adaptive_dice | mean_frozen_dice | mean_adaptive_advantage | roi_bootstrap_95ci_low | roi_bootstrap_95ci_high | roi_wins | holm_adjusted_roi_sign_test_p_exploratory |
|---|---|---|---|---|---|---|---|
| H_low_severe | 0.80009 | 0.74807 | 0.05202 | 0.03704 | 0.06687 | 134 | 0.00003 |
| H_high_severe | 0.79972 | 0.64570 | 0.15403 | 0.14237 | 0.16571 | 197 | 0.00000 |
| E_low_severe | 0.79342 | 0.78333 | 0.01009 | 0.00685 | 0.01321 | 166 | 0.00000 |
| E_high_severe | 0.79062 | 0.68902 | 0.10160 | 0.08576 | 0.11793 | 171 | 0.00000 |
| HE_low_severe | 0.79625 | 0.77150 | 0.02475 | 0.01112 | 0.03853 | 124 | 0.00325 |
| HE_high_severe | 0.79367 | 0.69833 | 0.09534 | 0.08319 | 0.10820 | 189 | 0.00000 |

## Subgroups (descriptive)

| slide_type | n_rois | mean_roi_adaptive_advantage | roi_wins |
|---|---|---|---|
| metastatic | 102 | 0.06567 | 98 |
| primary | 103 | 0.08020 | 102 |

## Feature-drift associations

| analysis | n | spearman_rho |
|---|---|---|
| ROI mean: reduction in feature MAE vs adaptive Dice advantage | 205 | -0.00191 |
| ROI mean: adaptive feature MAE vs adaptive Dice loss | 205 | 0.06332 |
| ROI mean: frozen feature MAE vs frozen Dice loss | 205 | -0.07350 |
| H_low_severe: drift reduction vs Dice advantage | 205 | 0.14717 |
| H_high_severe: drift reduction vs Dice advantage | 205 | 0.23143 |
| E_low_severe: drift reduction vs Dice advantage | 205 | -0.11923 |
| E_high_severe: drift reduction vs Dice advantage | 205 | 0.25184 |
| HE_low_severe: drift reduction vs Dice advantage | 205 | -0.05015 |
| HE_high_severe: drift reduction vs Dice advantage | 205 | 0.58668 |

## Research interpretation and limitations

- The counterfactual *freezes percentiles taken from the clean validation image*. This is an oracle reference, **not** a deployable system or a retrained, appropriately matched comparator.
- The RF was trained with adaptive normalization. Changing only inference normalization directly tests the trained pipeline's **sensitivity to that normalization policy**; it does not prove superiority to every alternative architecture or establish full causal mediation.
- The same ROI appears in six perturbation conditions. The primary unit of analysis is the **ROI-mean effect (n=205)**, not 1,230 independent cases.
- ROI identifiers and 'primary/metastatic' labels alone do not establish distinct patients or slides. Slide/patient-level correlation could make ROI-bootstrap confidence intervals and nominal p values **overconfident**. Treat all p values as exploratory until clustering metadata is obtained.
- This is internally cross-validated evidence on one dataset with **synthetic H&E concentration changes**. It is not validation on independent clinical laboratories or scanners.
- Six severe conditions were selected for the mechanism experiment. The four-method Stage-12 benchmark used 12 conditions; don't confuse their aggregate means.
- Per-condition sign tests are Holm-corrected across six conditions, but correction cannot account for unknown slide/patient clustering.
- Feature MAE is averaged across nine features and does not reveal RF feature importance or establish that drift reduction explains the Dice benefit.

## Reproducibility

All five Stage-15 integrity files read PASS. Maximum Stage-12 reproduction error: 1.110e-16; Stage-11 reproduction error: 1.110e-16.
Resampling: 10000 stratified ROI bootstrap samples, seed 13016; 41 ROI draws with replacement from each original CV fold.

**Next:** obtain reliable patient/slide identifiers if available, inspect worst-case ROIs, and prepare a focused IEEE results section with appropriately qualified inference.
