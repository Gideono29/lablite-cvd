# LabLite-CVD

> **Research use only.** LabLite-CVD is not a medical device and must not be used for individual
> clinical decisions.

An explainable cardiovascular risk model with **tiered inputs**. It measures the **information cost of missing
laboratory data**: how much discrimination, calibration and decision value is lost when lipid, glycemic,
renal or urine-albumin measurements aren't available.

Maintainer: Gideon Owusu, Michigan Technological University
([ORCID 0009-0000-0540-7449](https://orcid.org/0009-0000-0540-7449))

**Status:** `0.1.0`, pre-release. The fitted parameters ship with the package but haven't yet been independently
verified by the maintainer. See the [model card](https://github.com/Gideono29/lablite-cvd/blob/main/docs/model_card.md) for performance, subgroup results and
limitations.

## Usage

```python
import pandas as pd
import lablite_cvd

model = lablite_cvd.load_model()  # warns: research use only
patients = pd.DataFrame({
    "age": [55, 70], "sex": ["Female", "Male"], "sbp": [128, 150], "bp_treated": [0, 1],
    "smoker": [0, 1], "diabetes": [0, 1], "bmi": [27.0, 31.0],
    "total_chol": [210, None], "hdl": [60, None], "hba1c": [5.5, 7.2], "egfr": [92, 61], "uacr": [6.0, 85.0],
})
model.predict(patients)               # tier used + 10-year risk of CVD death, per row
model.predict(patients, tier="T0")    # force the office-only tier
```

Units: age years; SBP mmHg; BMI kg/m²; cholesterol mg/dL; HbA1c %; eGFR mL/min/1.73 m² (CKD-EPI 2021);
UACR mg/g; `bp_treated`, `smoker`, `diabetes` 0/1 (`diabetes` = self-reported diagnosis or glucose-lowering
medication). `tier="auto"` uses the highest tier whose inputs are all present. The model **underpredicts** for
people whose labs are missing in practice (O/E 1.38 in NHANES), so treat lower-tier risks as likely
underestimates. Predicts CVD **death**, not incident ASCVD.

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
pip install -e .[data,test]   # from a clone of this repository
lablite-cvd download   # 127 files from wwwn.cdc.gov and ftp.cdc.gov; SHA-256 manifest in data/raw/manifest.json
lablite-cvd cohort     # data/processed/{cohort.csv.gz, cohort_flow.csv, missingness.csv, data_dictionary.csv}
lablite-cvd fit        # outputs/fit/{lablite_params.json, coefficients.csv, performance.csv, information_cost.csv}
lablite-cvd bootstrap  # outputs/fit/{performance_ci.csv, information_cost_ci.csv}: survey-bootstrap 95% CIs
python scripts/nonlinearity_check.py  # outputs/fit/nonlinearity_check.csv: splines vs linear terms
pytest -q
```

**Cohort:** NHANES 1999–2018 linked to the NCHS 2019 public-use mortality files. Adults aged 40–79 who were
examined, not pregnant, had no self-reported CVD, were eligible for linkage and had complete office (T0)
inputs. Laboratory values are **left missing**, never imputed; each participant's highest available tier is
recorded. Outcome: CVD death (heart disease or cerebrovascular), with non-CVD death as a competing event.

**Models:** per tier, ridge-penalized cause-specific Cox models for CVD death and for non-CVD death; the
absolute 10-year CVD-death risk is their cumulative incidence (non-CVD death as a competing event). Fitted on all-labs participants from
1999–2010 (cycles with adequate 10-year follow-up). Every continuous input except age enters as a 4-knot
restricted cubic spline; log(UACR); SBP, BMI and laboratory inputs capped at the training
1st/99th percentiles. Performance and information cost use out-of-fold (5-fold) predictions, survey-weighted with IPCW;
net benefit and reclassification at 1/5/10% (primary) and 7.5/20% (secondary).

## Documentation

| File | Contents |
|---|---|
| `docs/model_card.md` | Model card: data, methods, performance, information cost, subgroups, limitations |
| `docs/open_questions.md` | Modeling decisions taken and still open |
| `data/processed/data_dictionary.csv` | Every cohort column: units, definition, NHANES source |

## Related

The data handling is ported from [EquiCVD Bench](https://doi.org/10.5281/zenodo.23083676) v1.0.0, with urine
albumin/creatinine added.

## Citation

See `CITATION.cff`. A Zenodo DOI will be added at the first release.
