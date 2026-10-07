# Changelog

## Unreleased
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
