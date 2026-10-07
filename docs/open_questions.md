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
    2.4%, O/E 1.46. This must appear in the model card as a caution for `tier="auto"`.
11. Kept defaults: one pair of models per tier with `female` as a covariate (not sex-stratified); fitted with
    normalized MEC weights; ridge penalty from {0, 1, 10, 100, 1000} by 5-fold cross-validated partial
    likelihood.
12. Continuous inputs except age (SBP, BMI and all labs) capped at the training 1st / 99th percentiles;
    `predict()` warns when age is outside 40–79.
13. Refitting bootstrap for the paper's main intervals: run after the TestPyPI release.

## Open
1. **Linear terms vs splines.** `scripts/nonlinearity_check.py` replaces each continuous term with a 4-knot
   restricted cubic spline (2 extra df) and compares 5-fold cross-validated partial log-likelihood
   (`outputs/fit/nonlinearity_check.csv`). Gains above ~2 suggest real non-linearity:
   - CVD death: eGFR +6.9, BMI +6.3, age +3.0, log UACR +2.7, HDL +2.3; SBP, total cholesterol, HbA1c ≤ 0.
   - Non-CVD death: BMI +14.7, HDL +13.3, HbA1c +10.9, eGFR +3.5; others ≤ 0.2.
   Linear terms are not supported for several inputs, including two labs (eGFR, HbA1c) whose information
   cost may be understated under linearity. Options: (a) prespecified splines for all continuous inputs in
   both models; (b) splines only where the check shows a gain (data-driven, needs reporting as such);
   (c) keep linear and report the check as a limitation.
