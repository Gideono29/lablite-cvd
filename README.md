# LabLite-CVD

> **Research use only.** LabLite-CVD is not a medical device and must not be used for individual
> clinical decisions.

An explainable cardiovascular risk model with **tiered inputs**. It measures the **information cost of missing
laboratory data**: how much discrimination, calibration and decision value is lost when lipid, glycemic,
renal or urine-albumin measurements aren't available.

Maintainer: Gideon Owusu, Michigan Technological University
([ORCID 0009-0000-0540-7449](https://orcid.org/0009-0000-0540-7449))

**Status:** pre-alpha (`0.1.0.dev0`). Package scaffolding only; no fitted model yet.

## Input tiers (draft)

| Tier | Adds | Inputs |
|---|---|---|
| T0 | office only | age, sex, systolic BP, BP treatment, smoking, diabetes, BMI |
| T1 | lipids | + total cholesterol, HDL cholesterol |
| T2 | glycemic / renal | + HbA1c, eGFR |
| T3 | urine albumin | + UACR |

`lablite_cvd.available_tier(record)` returns the highest tier whose inputs are all present.

## Development

```bash
pip install -e .[test]
pytest -q
```

## Documentation

| File | Contents |
|---|---|
| `docs/model_card.md` | Model card (draft) |

## Related

Data handling will follow [EquiCVD Bench](https://doi.org/10.5281/zenodo.23083676) (NHANES 1999–2018 with NCHS
2019 public-use linked mortality).

## Citation

See `CITATION.cff`. A Zenodo DOI will be added at the first release.
