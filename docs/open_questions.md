# Open questions for the maintainer

Modeling decisions the code doesn't make for you. Each one names where it applies.

## Decided (2026-10-07)
1. Tiers: T0 office → T1 + total cholesterol, HDL → T2 + HbA1c, eGFR → T3 + UACR.
2. Cohort keeps everyone with complete T0 inputs; tier comparisons use the all-labs subset; no imputation.
3. Penalized Cox models per tier; published coefficients.
4. Primary information cost: Δ net benefit at the decision thresholds (item 6); secondary ΔAUC,
   Δcalibration, reclassification.
5. Horizon 10 years (cycles with adequate 10-year follow-up). UACR enters as log(UACR).
6. Decision thresholds: 1/5/10% primary (ESC SCORE 10-year fatal-CVD cut points); 7.5/20% secondary.
7. Absolute risk accounts for non-CVD death as a competing event: per tier, a second cause-specific Cox model
   for non-CVD death; CVD risk = cumulative incidence (O/E went from 0.88 to 0.98).
8. Office-tier `diabetes` = self-report or glucose-lowering medication only (no HbA1c); `diabetes_any` is
   descriptive only.
9. UACR: pre-2007 urine creatinine (Jaffe, Beckman CX3) adjusted to the 2007+ enzymatic method (Roche ModP)
   with the piecewise equations in the NHANES 2007–2008 ALB_CR_E documentation, applied to 1999–2006.
   Median effect on pre-2007 UACR: +4.2% (IQR 1.5–6.3%). The documentation reports no change in the
   urine albumin method. (An earlier version of this file wrongly said the albumin method changed in 2009;
   the URXUMA2 / URXUCR2 variables in ALB_CR_F aren't a second assay.)
10. Informative missingness: limitations paragraph plus a descriptive secondary analysis
    (`outputs/fit/missing_group_check.csv`). In 1999–2010, the final models applied with `tier="auto"` to the
    813 participants naturally missing a lab (41 CVD deaths) **underpredict**: observed 3.5% vs predicted
    2.6%, O/E 1.38 (1.46 before splines). This must appear in the model card as a caution for `tier="auto"`.
11. Kept defaults: one pair of models per tier with `female` as a covariate (not sex-stratified); fitted with
    normalized MEC weights; ridge penalty from {0, 1, 10, 100, 1000} by 5-fold cross-validated partial
    likelihood.
12. Continuous inputs except age (SBP, BMI and all labs) capped at the training 1st / 99th percentiles;
    `predict()` warns when age is outside 40–79.
13. Refitting bootstrap for the paper's main intervals: run after the TestPyPI release.
14. Prespecified 4-knot restricted cubic splines (knots at the 5/35/65/95th percentiles of the capped
    training values) for every continuous input except age, in both cause-specific models. Motivated by
    `scripts/nonlinearity_check.py` (linear vs spline, cross-validated likelihood;
    `outputs/fit/nonlinearity_check.csv`), which showed gains for eGFR, BMI, HDL, HbA1c and log UACR.
    Shape curves in `outputs/fit/shape_functions.csv`.

15. Age stays linear in the main model (decided 2026-10-07). The age-spline fit is reported as a sensitivity
    analysis (`outputs/sensitivity_age_spline/`): conclusions unchanged (T3 AUC 0.830 vs 0.829; UACR ΔAUC
    0.014 in both; O/E 0.98 in both), despite a +3.0 cross-validated likelihood gain for age in the CVD model.

## Open
None. Next: model card, packaging the parameters, TestPyPI release; then the refitting bootstrap (item 13).
