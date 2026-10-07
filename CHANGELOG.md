# Changelog

## Unreleased
- Data pipeline ported from EquiCVD Bench v1.0.0 with urine albumin/creatinine (LAB16, L16_B, L16_C,
  ALB_CR_D-J) added: `lablite-cvd download` (127 files, SHA-256 manifest) and `lablite-cvd cohort`.
- Cohort keeps participants with missing laboratory values; records `has_*` indicators, `all_labs` and `tier`;
  writes flow, missingness table, data dictionary and metadata. n = 24,861 (23,374 with all labs).
- Office-tier diabetes excludes HbA1c; `diabetes_any` kept for description only.
- `available_tiers(df)`; `data` extra (requests); `lablite-cvd` console script.
- `docs/open_questions.md`.
- Repository scaffolding: package layout, input tiers T0–T3 with `available_tier`, draft model card,
  citation metadata, CI.
