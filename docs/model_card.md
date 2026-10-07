# Model card: LabLite-CVD 0.1.1

Structure follows Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). All numbers come from files in
`outputs/fit/` and `data/processed/`, produced by the commands in the README. The modeling decisions and their
rationale are in `docs/open_questions.md`.

> **Research use only.** LabLite-CVD is not a medical device. It must not be used for individual clinical
> decisions, treatment allocation, or in place of a validated clinical risk tool.

> **Verification status:** parameters and results verified by the maintainer on 2026-10-07 (`meta.status` in
> the parameter file). 0.1.0 was the unverified pre-release of the same parameters.

## Model details
- **Developer:** Gideon Owusu, Michigan Technological University
  ([ORCID 0009-0000-0540-7449](https://orcid.org/0009-0000-0540-7449)).
- **Version:** 0.1.1 (October 2026). License: MIT.
- **Model type:** four nested *tiers*, each a pair of ridge-penalized cause-specific Cox proportional hazards
  models, one for CVD death and one for non-CVD death. Predicted 10-year risk of CVD death is the cumulative
  incidence from the two models, so non-CVD death is treated as a competing event.
- **Inputs by tier:**

  | Tier | Adds | Inputs |
  |---|---|---|
  | T0 | office only | age, sex, systolic BP, BP treatment, current smoking, diabetes (self-report or medication), BMI |
  | T1 | lipids | + total cholesterol, HDL cholesterol |
  | T2 | glycemic / renal | + HbA1c, eGFR (CKD-EPI 2021, race-free) |
  | T3 | urine albumin | + urine albumin-to-creatinine ratio (UACR) |

- **Functional form:** age linear; every other continuous input capped at its training 1st/99th percentile and
  entered as a 4-knot restricted cubic spline (knots at the 5/35/65/95th percentiles); UACR on the log scale.
  Binary inputs linear. No interactions. Race/ethnicity is not an input.
- **Explainability:** all parameters (caps, knots, coefficients, baseline hazards) are published in
  `lablite_cvd/data/lablite_params.json`. `TierModel.shape()` returns each input's log hazard ratio curve;
  curves for every tier and both causes are in `outputs/fit/shape_functions.csv`.
- **Usage:** `lablite_cvd.load_model().predict(df)` returns the tier used and the risk for each row.
  `tier="auto"` (default) uses the highest tier whose inputs are all present; `tier="T0"` etc. forces one.

## Intended use
- **Primary:** research on how much predictive and decision value is lost when laboratory tests are missing
  (the *information cost* of each lab group), and on tiered, missing-data-aware risk modeling.
- **Users:** researchers in cardiovascular epidemiology, risk prediction and health services.
- **Out of scope:** individual patient care; populations outside U.S. adults aged 40–79 without prior CVD;
  outcomes other than CVD death (it does **not** predict incident ASCVD, myocardial infarction or stroke);
  horizons other than 10 years; anyone younger than 40 or older than 79 (`predict()` warns).

## Outcome
Death from cardiovascular disease within 10 years: NCHS underlying cause of death codes 001 (diseases of heart)
or 005 (cerebrovascular diseases) in the 2019 public-use linked mortality file. Non-CVD death is a competing
event. Because the outcome is death, not incident disease, absolute risks are much lower than ASCVD
incidence tools such as the Pooled Cohort Equations or PREVENT predict, and the two can't be compared directly.

## Training and evaluation data
- **Source:** NHANES 1999–2018 linked to the NCHS 2019 public-use mortality files (follow-up through
  2019-12-31), downloaded from CDC with SHA-256 checksums (`data/raw/manifest.json`).
- **Cohort** (`data/processed/cohort_flow.csv`): 101,316 participants → aged 40–79 (31,995) → examined at
  the mobile examination center (30,704) → not pregnant (30,673) → no self-reported CHD, angina, MI, heart
  failure or stroke (26,357) → linkage-eligible with follow-up (26,293) → complete office inputs: **24,861**.
  Laboratory values are left missing, never imputed; 23,374 (94%) have all labs.
- **Model development and evaluation set:** all-labs participants from the cycles with adequate 10-year
  follow-up (1999–2010): **n = 13,349, 355 CVD deaths and 1,124 non-CVD deaths within 10 years.**
- **Preprocessing:** pre-2007 urine creatinine adjusted to the 2007+ enzymatic method (NHANES ALB_CR_E
  equations); serum creatinine standardized to IDMS per NHANES guidance; office diabetes excludes HbA1c so
  that no lab information enters T0.
- **Weights:** models fitted with normalized 20-year MEC examination weights, so estimates target the U.S.
  population.

## Evaluation method
Five-fold cross-fitting: every participant's prediction comes from models fitted without them, with ridge
penalties chosen inside each training fold by cross-validated partial likelihood. Metrics are survey-weighted,
with inverse probability of censoring weights (Kaplan–Meier within cycle) for administrative censoring.
Intervals are 95% Rao–Wu survey-bootstrap percentile intervals (B = 200, PSUs resampled within strata) over
the fixed out-of-fold predictions. The models aren't refitted in each replicate, so the intervals are
somewhat too narrow; a refitting bootstrap is planned.

## Performance (out-of-fold, 95% CI)

| Metric | T0 office | T1 + lipids | T2 + HbA1c, eGFR | T3 + UACR |
|---|---|---|---|---|
| AUC | 0.813 (0.787, 0.846) | 0.810 (0.782, 0.844) | 0.815 (0.785, 0.851) | **0.829** (0.800, 0.863) |
| Observed / expected | 0.98 (0.86, 1.11) | 0.98 (0.86, 1.11) | 0.98 (0.86, 1.12) | 0.98 (0.86, 1.11) |
| Calibration slope | 0.95 (0.84, 1.11) | 0.94 (0.82, 1.09) | 0.95 (0.83, 1.09) | 0.97 (0.87, 1.10) |
| Brier score | 0.0177 | 0.0177 | 0.0177 | 0.0175 |
| % above 5% risk | 9.8% | 10.1% | 9.2% | 9.2% |

Observed 10-year CVD death risk is 1.87% (95% CI 1.62–2.13%).

## Information cost of missing labs
Metric(richer tier) − metric(poorer tier): what's lost when the labs that separate the two tiers are missing.
Net benefit is per 1,000 people; the primary decision thresholds are 1/5/10% (ESC SCORE 10-year fatal-CVD cut
points), with 7.5/20% secondary (`outputs/fit/information_cost_ci.csv`).

| Labs missing | ΔAUC | ΔNB at 1% | ΔNB at 5% | ΔNB at 10% | % changing category (1/5/10%) |
|---|---|---|---|---|---|
| UACR (T3 vs T2) | **0.014** (0.007, 0.022) | 0.5 (−0.0, 1.1) | **0.9** (0.1, 1.6) | −0.2 (−0.9, 0.4) | 14.2% |
| HbA1c + eGFR (T2 vs T1) | 0.005 (−0.001, 0.012) | −0.4 (−0.9, 0.2) | 0.2 (−0.4, 1.0) | 0.6 (−0.2, 1.4) | 11.6% |
| Lipids (T1 vs T0) | −0.003 (−0.009, 0.002) | −0.0 (−0.3, 0.3) | −0.5 (−1.3, 0.2) | −0.3 (−0.8, 0.2) | 9.7% |
| All labs (T3 vs T0) | **0.016** (0.001, 0.029) | 0.1 (−0.5, 0.7) | 0.6 (−0.3, 1.5) | 0.1 (−0.7, 0.9) | 19.3% |

**Summary:** for 10-year CVD death, UACR is the only lab group with a clear information cost. Without any labs,
discrimination falls, but net benefit at the primary thresholds doesn't fall detectably. Missing labs still
move 10–19% of people to a different risk category, so individual predictions change even where population
decision value doesn't.

## Subgroup performance (`outputs/fit/subgroup_performance.csv`)

| Group | n | CVD deaths | T0 O/E | T0 AUC | T3 O/E | T3 AUC |
|---|---|---|---|---|---|---|
| Female | 6,828 | 142 | 0.98 (0.80, 1.15) | 0.81 (0.78, 0.86) | 0.97 (0.79, 1.13) | 0.83 (0.79, 0.87) |
| Male | 6,521 | 213 | 0.98 (0.79, 1.18) | 0.81 (0.77, 0.85) | 0.99 (0.80, 1.19) | 0.82 (0.79, 0.86) |
| NH White | 6,633 | 188 | 0.94 (0.79, 1.11) | 0.84 (0.82, 0.87) | 0.96 (0.80, 1.14) | 0.86 (0.83, 0.88) |
| NH Black | 2,544 | 82 | 1.14 (0.87, 1.43) | 0.76 (0.69, 0.83) | 1.14 (0.89, 1.43) | 0.79 (0.74, 0.84) |
| Mexican American | 2,780 | 61 | 1.12 (0.72, 1.47) | 0.76 (0.69, 0.83) | 1.00 (0.64, 1.35) | 0.76 (0.69, 0.84) |
| Other Hispanic | 919 | 17 | 1.06 (0.37, 1.98) | 0.66 (0.52, 0.93) | 0.97 (0.34, 1.87) | 0.71 (0.61, 0.90) |
| Other / Multiracial | 473 | 7 | 0.97 (0.16, 1.81) | 0.56 (0.28, 0.94) | 0.91 (0.15, 1.69) | 0.59 (0.29, 0.93) |
| Age 40–59 | 7,775 | 77 | 0.99 (0.73, 1.23) | 0.76 (0.69, 0.83) | 0.94 (0.70, 1.17) | 0.80 (0.75, 0.86) |
| Age 60–79 | 5,574 | 278 | 0.97 (0.82, 1.13) | 0.71 (0.66, 0.74) | 1.00 (0.84, 1.15) | 0.73 (0.68, 0.77) |

Calibration intervals include 1 in every group. **Discrimination is lower for non-Hispanic Black and
Mexican American participants than for non-Hispanic White participants**, and the Other Hispanic and
Other/Multiracial groups have too few deaths for reliable estimates. Within-age-group AUCs are lower than
the overall AUC because age is the strongest predictor.

## Caution: participants with missing labs (`tier="auto"`)
Applied with `tier="auto"` to the 813 participants from 1999–2010 who were *naturally* missing a lab (41 CVD
deaths), the models **underpredict**: observed 3.5% vs predicted 2.6%, O/E 1.38 (O/E 1.97 in the 86
participants routed to T2; descriptive, few events; `outputs/fit/missing_group_check.csv`). People with
missing labs died more often than their measured inputs imply (in the full cohort, all-cause death was
20.4% among those missing a lab vs 13.5% among those with all labs), so missingness is informative. A lower-tier prediction for someone whose labs are missing in
practice is likely an **underestimate**.

## Sensitivity analysis
Modeling age as a spline as well (`outputs/sensitivity_age_spline/`) leaves the conclusions unchanged:
T3 AUC 0.830 vs 0.829, UACR ΔAUC 0.014 in both, O/E 0.98 in both.

## Ethical considerations and limitations
- **Outcome:** CVD mortality from public-use linked data, which NCHS perturbs for some fields; no nonfatal
  events. Cause of death on death certificates is imperfect.
- **Era:** participants examined 1999–2010 and followed to 2019; CVD mortality and treatment have changed
  since then. No external validation has been done.
- **Data:** self-reported medical history and medication; single-occasion lab and BP measurements;
  `bp_treated` and `diabetes` from questionnaires. Statin use isn't a model input.
- **Missingness:** missing labs are informative (see above), and tier comparisons use participants with all
  labs, whose risk differs from those without.
- **Equity:** lower discrimination in non-Hispanic Black and Mexican American participants; very small samples
  for other groups. Race/ethnicity isn't an input, but performance differs across groups.
- **Uncertainty:** bootstrap intervals don't include model-refitting variability.
- **Proportional hazards and functional form:** no interactions, including with age; linear age.

## Reproduction
From a clone of the repository:
```bash
pip install -e .[data,test]
lablite-cvd download && lablite-cvd cohort && lablite-cvd fit && lablite-cvd bootstrap
python scripts/nonlinearity_check.py
python scripts/subgroup_performance.py
```
