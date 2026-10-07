# LabLite-CVD

> **Research use only.** LabLite-CVD is not a medical device and must not be used for individual
> clinical decisions.

An explainable cardiovascular risk model with **tiered inputs**. It measures the **information cost of missing
laboratory data**: how much discrimination, calibration and decision value is lost when lipid, glycemic,
renal or urine-albumin measurements aren't available.

Maintainer: Gideon Owusu, Michigan Technological University
([ORCID 0009-0000-0540-7449](https://orcid.org/0009-0000-0540-7449))

**Status:** pre-alpha (`0.1.0.dev0`). Data pipeline, cohort and draft tier models built; the models are not yet
verified and not shipped in the package.

## Input tiers

| Tier | Adds | Inputs |
|---|---|---|
| T0 | office only | age, sex, systolic BP, BP treatment, smoking, diabetes (self-report/medication), BMI |
| T1 | lipids | + total cholesterol, HDL cholesterol |
| T2 | glycemic / renal | + HbA1c, eGFR |
| T3 | urine albumin | + UACR |

`lablite_cvd.available_tier(record)` returns the highest tier whose inputs are all present
(`available_tiers(df)` for a DataFrame).

## Data pipeline

```bash
pip install -e .[data,test]
lablite-cvd download   # 127 files from wwwn.cdc.gov and ftp.cdc.gov; SHA-256 manifest in data/raw/manifest.json
lablite-cvd cohort     # data/processed/{cohort.csv.gz, cohort_flow.csv, missingness.csv, data_dictionary.csv}
lablite-cvd fit        # outputs/fit/{lablite_params.json, coefficients.csv, performance.csv, information_cost.csv}
pytest -q
```

**Cohort:** NHANES 1999–2018 linked to the NCHS 2019 public-use mortality files. Adults aged 40–79 who were
examined, not pregnant, had no self-reported CVD, were eligible for linkage and had complete office (T0)
inputs. Laboratory values are **left missing**, never imputed; each participant's highest available tier is
recorded. Outcome: CVD death (heart disease or cerebrovascular), with non-CVD death as a competing event.

**Models:** one ridge-penalized Cox model per tier for 10-year CVD death, fitted on all-labs participants from
1999–2010 (cycles with adequate 10-year follow-up). log(UACR); laboratory inputs capped at the training 1st/99th
percentiles. Performance and information cost use out-of-fold (5-fold) predictions, survey-weighted with IPCW.

## Documentation

| File | Contents |
|---|---|
| `docs/model_card.md` | Model card (draft) |
| `docs/open_questions.md` | Modeling decisions taken and still open |
| `data/processed/data_dictionary.csv` | Every cohort column: units, definition, NHANES source |

## Related

The data handling is ported from [EquiCVD Bench](https://doi.org/10.5281/zenodo.23083676) v1.0.0, with urine
albumin/creatinine added.

## Citation

See `CITATION.cff`. A Zenodo DOI will be added at the first release.
