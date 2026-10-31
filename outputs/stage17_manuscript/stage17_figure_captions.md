# Stage 17 figure captions (suggested)

**Figure 1 — Condition-specific advantage.** Mean ROI-paired Dice difference (adaptive minus frozen normalization) for each of six predefined severe H&E perturbation conditions. Whiskers indicate exploratory ROI-bootstrap 95% percentile intervals from Stage 16. Fractional labels show ROIs favoring adaptive; these observations are repeated across conditions.

**Figure 2 — Paired ROI mean Dice.** Each point represents one ROI's mean Dice across six predefined severe perturbations with the same clean-trained RF. Dashed line indicates equality. Color denotes primary versus metastatic melanoma; no between-group generalization claim.

**Figure 3 — Fold-consistency forest plot.** Mean adaptive-minus-frozen Dice advantage in each of five frozen validation folds (41 ROIs/fold); exploratory within-fold ROI-bootstrap confidence intervals.

**Figure 4 — Feature-drift magnitudes.** Mean absolute feature change relative to clean image, averaged across nine handcrafted features and ROIs, comparing adaptive and clean-frozen feature percentile normalization by perturbation condition.

**Figure 5 — Feature drift versus Dice effect.** ROI-level mean feature MAE reduction versus mean adaptive Dice advantage. An approximately zero overall Spearman correlation cautions against inferring that aggregate feature-drift reduction is the causal mediator.

**Figure 6 — Exceptional ROI heatmap.** Paired Dice differences across six conditions in the five ROIs with lowest average adaptive advantage; negative cells show frozen normalization outperforming adaptive in that condition. These are numeric results, not qualitative image masks.

**Figure 7 — Precision and recall.** ROI-macro mean segmentation precision/recall under six severe conditions for both inference-normalization branches, using the same clean-trained RF and postprocessing.

**Figure 8 — Stage12 benchmark (generated only with all four methods' input files).** The clean and perturbed mean Dice and corresponding mean Dice degradation for No Correction, CLAHE, Macenko, and Reinhard, each paired across the identical 205 ROIs and 12 perturbation conditions. Model-specific training makes this a pipeline comparison rather than a fixed-model preprocessing intervention.
