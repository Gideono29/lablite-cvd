# Open questions for the maintainer

Modeling decisions the code doesn't make for you. Each one names where it applies.

## Decided (2026-10-07)
1. Tiers: T0 office → T1 + total cholesterol, HDL → T2 + HbA1c, eGFR → T3 + UACR.
2. Cohort keeps everyone with complete T0 inputs; tier comparisons use the all-labs subset; no imputation.
3. Penalized Cox model per tier; published coefficients.
4. Primary information cost: Δ net benefit at 7.5% / 20%; secondary ΔAUC, Δcalibration, reclassification.
5. Horizon 10 years (cycles with adequate 10-year follow-up). UACR enters as log(UACR). Laboratory inputs
   capped (winsorized) at their 1st and 99th percentiles in the training data.

## Open
1. **Office diabetes definition** (`pipeline/cohort.py::derive`). T0 `diabetes` is self-report or medication
   only. EquiCVD also counted HbA1c ≥ 6.5%, but that would put lab information into the office tier.
   `diabetes_any` keeps the HbA1c-inclusive version for description only. Confirm.
2. **UACR assay changes across cycles.** UACR is computed as URXUMA / URXUCR × 100 in every cycle with no
   adjustment. NHANES changed the urine creatinine method in 2007 and the albumin method in 2009 (ALB_CR_F
   carries both assays: URXUMA2 / URXUCR2). Check the NHANES lab documentation for recommended
   cross-cycle adjustments and decide whether to apply them.
3. **Informative missingness.** Participants missing any lab (6.0%) had higher all-cause mortality
   (20.4% vs 13.5%) and CVD mortality (4.2% vs 3.4%) than the all-labs subset. Decide how to report this,
   e.g. a limitations paragraph or a secondary analysis on the naturally-missing group.
4. **Decision thresholds for a mortality outcome** (`pipeline/metrics.py::THRESHOLDS`). Observed 10-year
   CVD death risk in the analysis set is 1.9%. The 7.5% / 20% thresholds were designed for incident ASCVD,
   so few people cross them (5.6% and 0.5–0.9% flagged) and net benefit at 20% is ~0 for every tier, which
   leaves the primary metric with little to compare. One option is the thresholds used for 10-year fatal CVD
   in the original ESC SCORE charts (1%, 5%, 10%), keeping 7.5% / 20% as secondary. Both sets are now
   computed; decide which is primary.
5. **Competing risk.** Each tier is a cause-specific Cox model and risk = 1 − S(10), which ignores non-CVD
   death; predictions run ~12% high (O/E ≈ 0.88–0.89). Options: keep and recalibrate, or estimate absolute
   risk with a second cause-specific model for non-CVD death (Aalen–Johansen). Decide.
6. **Defaults Claude chose; please confirm:** one model per tier with `female` as a covariate (not
   sex-stratified); linear terms only (no splines or interactions); fitted with normalized MEC weights;
   ridge penalty from {0, 1, 10, 100, 1000} by 5-fold cross-validated partial likelihood (it barely matters:
   the CV likelihood is flat between 0 and 10); capping applies to laboratory inputs only, not SBP or BMI.
7. **Bootstrap does not refit the models.** `lablite-cvd bootstrap` (Rao–Wu over PSUs within strata, B = 200)
   resamples fixed out-of-fold predictions, as EquiCVD did, so the intervals leave out model-fitting
   variability and are somewhat too narrow. A refitting bootstrap is feasible (one fit takes ~2 min, so B = 200
   is ~6 h). Decide whether the paper needs it.
