# Changelog

## 0.1.1 (2026-10-07)
- Parameters and results verified by the maintainer; `meta.status` updated. Model and parameters are
  otherwise identical to 0.1.0.

## 0.1.0 (2026-10-07)
Published on PyPI as a pre-release: the fitted parameters are not yet verified by the maintainer
(`meta.status` in the parameter file). A verified release will follow as a new version.

- Fitted parameters shipped as package data; `lablite_cvd.load_model()` (warns: research use only).
- Model card with performance, information cost, subgroup results and limitations;
  `scripts/subgroup_performance.py`.
- Clear error when `sex` / `female` is missing; SPDX license metadata.
- Prespecified restricted cubic splines (4 knots) for every continuous input except age; knots stored in the
  parameter JSON; `TierModel.shape()` and `outputs/fit/shape_functions.csv` give log hazard ratio curves.
- `cox.fit` returns zero coefficients when there are no events.
- UACR: pre-2007 urine creatinine adjusted to the 2007+ enzymatic method (NHANES ALB_CR_E equations).
- SBP and BMI also capped at training 1st/99th percentiles; `predict()` warns for age outside 40-79.
- Cox fit uses step halving (robust to near-collinear inputs).
- Secondary analysis of participants naturally missing labs (`missing_group_check.csv`).
- `scripts/nonlinearity_check.py`: splines vs linear terms by cross-validated likelihood.
- Competing risks: each tier adds a cause-specific Cox model for non-CVD death; predicted risk is the cumulative
  incidence of CVD death (O/E 0.88 -> 0.98). Parameter JSON now stores baseline hazard steps for both causes.
- Primary decision thresholds 1/5/10%; 7.5/20% secondary.
- `lablite-cvd bootstrap`: Rao–Wu survey bootstrap (B = 200) for per-tier metrics and information cost.
  Information cost now covers each lab group (T1−T0, T2−T1, T3−T2) as well as each tier vs T3; net benefit
  at 1/5/7.5/10/20%; reclassification at 1/5/10% and 7.5/20% cut points.
- Tier models: weighted ridge Cox (`lablite_cvd.cox`, checked against statsmodels PHReg), `TierModel` /
  `LabLiteModel` with JSON parameters and `predict(df, tier="auto")`; `lablite-cvd fit` writes parameters,
  coefficients, out-of-fold performance and information cost (point estimates). Draft, not yet verified.
- Data pipeline ported from EquiCVD Bench v1.0.0 with urine albumin/creatinine (LAB16, L16_B, L16_C,
  ALB_CR_D-J) added: `lablite-cvd download` (127 files, SHA-256 manifest) and `lablite-cvd cohort`.
- Cohort keeps participants with missing laboratory values; records `has_*` indicators, `all_labs` and `tier`;
  writes flow, missingness table, data dictionary and metadata. n = 24,861 (23,374 with all labs).
- Office-tier diabetes excludes HbA1c; `diabetes_any` kept for description only.
- `available_tiers(df)`; `data` extra (requests); `lablite-cvd` console script.
- `docs/open_questions.md`.
- Repository scaffolding: package layout, input tiers T0–T3 with `available_tier`, draft model card,
  citation metadata, CI.
