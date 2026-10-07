# Open questions for the maintainer

Modeling decisions the code doesn't make for you. Each one names where it applies.

## Decided (2026-10-07)
1. Tiers: T0 office → T1 + total cholesterol, HDL → T2 + HbA1c, eGFR → T3 + UACR.
2. Cohort keeps everyone with complete T0 inputs; tier comparisons use the all-labs subset; no imputation.
3. Penalized Cox model per tier; published coefficients.
4. Primary information cost: Δ net benefit at 7.5% / 20%; secondary ΔAUC, Δcalibration, reclassification.

## Open
1. **Office diabetes definition** (`pipeline/cohort.py::derive`). T0 `diabetes` is self-report or medication
   only. EquiCVD also counted HbA1c ≥ 6.5%, but that would put lab information into the office tier.
   `diabetes_any` keeps the HbA1c-inclusive version for description only. Confirm.
2. **UACR assay changes across cycles.** UACR is computed as URXUMA / URXUCR × 100 in every cycle with no
   adjustment. NHANES changed the urine creatinine method in 2007 and the albumin method in 2009 (ALB_CR_F
   carries both assays: URXUMA2 / URXUCR2). Check the NHANES lab documentation for recommended
   cross-cycle adjustments and decide whether to apply them.
3. **Extreme values.** The cohort keeps raw values (max UACR 92,500 mg/g, BMI 130, HbA1c 2.0%). Choose
   transforms (e.g. log UACR) and winsorization limits for model fitting.
4. **Informative missingness.** Participants missing any lab (6.0%) had higher all-cause mortality
   (20.4% vs 13.5%) and CVD mortality (4.2% vs 3.4%) than the all-labs subset. Decide how to report this,
   e.g. a limitations paragraph or a secondary analysis on the naturally-missing group.
5. **Horizon and cycles.** EquiCVD used a 10-year horizon, keeping 1999–2010 where follow-up is
   adequate. Confirm the horizon for LabLite-CVD.
