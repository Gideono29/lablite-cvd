"""Build the analytic cohort, keeping participants with missing laboratory values.

Eligibility follows EquiCVD Bench v1.0.0 (age 40-79, MEC-examined, not pregnant, no self-reported CVD,
linkage-eligible) except that only the office-tier (T0) inputs must be complete. Laboratory inputs are left
missing and each participant's highest available tier is recorded. No imputation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lablite_cvd.pipeline.config import CVD_UCOD, CYCLES, MORT_COLSPECS, MORT_NAMES, RACE_LABELS, cycle_files
from lablite_cvd.tiers import TIERS, available_tiers

LAB_GROUPS = {
    "lipids": ("total_chol", "hdl"),
    "glyc_renal": ("hba1c", "egfr"),
    "uacr": ("uacr",),
}


def _read_xpt(path: Path, cols: list) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    keep = ["SEQN"] + [c for c in dict.fromkeys(cols) if c in df.columns]
    return df[keep]


def load_cycle(raw: Path, label: str, year: int, suffix: str) -> pd.DataFrame:
    files = cycle_files(year, suffix)
    demo_stem = "DEMO" + suffix
    df = _read_xpt(raw / "nhanes" / label / f"{demo_stem}.xpt", files.pop(demo_stem))
    for stem, cols in files.items():
        path = raw / "nhanes" / label / f"{stem}.xpt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `lablite-cvd download` first")
        df = df.merge(_read_xpt(path, cols), on="SEQN", how="left")
    df["cycle"] = label
    df["cycle_start"] = year
    return df


def load_mortality(raw: Path) -> pd.DataFrame:
    frames = []
    for _, year, _ in CYCLES:
        path = raw / "mortality" / f"NHANES_{year}_{year + 1}_MORT_2019_PUBLIC.dat"
        m = pd.read_fwf(path, colspecs=MORT_COLSPECS, names=MORT_NAMES, na_values=["."], dtype=str)
        frames.append(m.apply(pd.to_numeric, errors="coerce"))
    return pd.concat(frames, ignore_index=True)


def _coalesce(df, cols):
    out = pd.Series(np.nan, index=df.index)
    for c in cols:
        if c in df.columns:
            out = out.fillna(df[c])
    return out


def _yes_no(s):
    """NHANES 1=yes/2=no coding -> 1/0, refused/don't know/missing -> NaN."""
    return s.map({1: 1.0, 2: 0.0})


def ckd_epi_2021(scr, age, female):
    """Race-free CKD-EPI 2021 creatinine equation (mL/min/1.73m2)."""
    scr = np.asarray(scr, float)
    female = np.asarray(female) == 1
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    r = scr / k
    return (142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.200 * 0.9938 ** np.asarray(age, float)
            * np.where(female, 1.012, 1.0))


def adjust_urine_creatinine_pre2007(x):
    """Urine creatinine (mg/dL) from the pre-2007 Jaffe method to the 2007+ enzymatic method.

    Piecewise equations from the NHANES 2007-2008 ALB_CR_E documentation:
    X < 75: (1.02*sqrt(X) - 0.36)^2; 75 <= X < 250: (1.05*sqrt(X) - 0.74)^2; X >= 250: (1.01*sqrt(X) - 0.10)^2.
    """
    x = np.asarray(x, float)
    r = np.sqrt(x)
    base = np.select([x < 75, x < 250], [1.02 * r - 0.36, 1.05 * r - 0.74], 1.01 * r - 0.10)
    return np.where(np.isnan(x), np.nan, np.clip(base, 0, None) ** 2)


def derive(raw: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"SEQN": raw["SEQN"].astype(int), "cycle": raw["cycle"],
                      "cycle_start": raw["cycle_start"]})
    d["age"] = raw["RIDAGEYR"]
    d["female"] = (raw["RIAGENDR"] == 2).astype(int)
    d["sex"] = raw["RIAGENDR"].map({1: "Male", 2: "Female"})
    d["race"] = raw["RIDRETH1"].map(RACE_LABELS)

    # --- Office (T0) inputs ---
    d["sbp"] = raw[[c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in raw]].mean(axis=1)
    told_htn = _yes_no(raw["BPQ020"])
    # BPQ050A asked only of those told they have hypertension; skip -> not treated
    d["bp_treated"] = np.where(told_htn.isna(), np.nan, (raw["BPQ050A"] == 1).astype(float))
    ever = _yes_no(raw["SMQ020"])
    now = raw["SMQ040"].map({1: 1.0, 2: 1.0, 3: 0.0})
    d["smoker"] = np.where(ever == 0, 0.0, np.where(ever == 1, now, np.nan))
    # Office diabetes: self-report or glucose-lowering medication only (no HbA1c, which is a T2 lab input)
    dm_q = raw["DIQ010"].map({1: 1.0, 2: 0.0, 3: 0.0})
    dm_rx = (raw["DIQ050"] == 1) | (raw["DIQ070"] == 1)
    d["diabetes"] = np.where((dm_q == 1) | dm_rx, 1.0, np.where(dm_q.notna(), 0.0, np.nan))
    d["bmi"] = raw["BMXBMI"]

    # --- Laboratory inputs (left missing when not measured) ---
    d["total_chol"] = raw["LBXTC"]
    d["hdl"] = _coalesce(raw, ["LBDHDL", "LBXHDD", "LBDHDD"])
    d["hba1c"] = raw["LBXGH"]
    # Serum creatinine standardized to IDMS-traceable method per NHANES guidance
    cr = _coalesce(raw, ["LBDSCR", "LBXSCR"])
    cr = np.where(d["cycle_start"] == 1999, 1.013 * cr + 0.147, cr)
    cr = np.where(d["cycle_start"] == 2005, -0.016 + 0.978 * cr, cr)
    d["creatinine"] = cr
    d["egfr"] = ckd_epi_2021(d["creatinine"], d["age"], d["female"])
    # UACR (mg/g) = urine albumin (ug/mL = mg/L) / urine creatinine (mg/dL) * 100, with pre-2007 urine
    # creatinine (Jaffe, Beckman CX3) adjusted to the 2007+ enzymatic Roche ModP method.
    ucr = raw["URXUCR"].where(raw["URXUCR"] > 0)
    ucr = np.where(d["cycle_start"] < 2007, adjust_urine_creatinine_pre2007(ucr), ucr)
    d["uacr"] = raw["URXUMA"] / pd.Series(ucr, index=d.index).where(lambda s: s > 0) * 100

    # Descriptive only, never a model input: diabetes including HbA1c >= 6.5%
    d["diabetes_any"] = np.where((d["diabetes"] == 1) | (d["hba1c"] >= 6.5), 1.0,
                                 np.where(d["diabetes"].notna() | d["hba1c"].notna(), 0.0, np.nan))
    d["statin"] = (raw["BPQ100D"] == 1).astype(float)

    mcq = raw[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]]
    d["prior_cvd"] = (mcq == 1).any(axis=1).astype(int)
    d["pregnant"] = (raw["RIDEXPRG"] == 1).astype(int)
    d["mec_examined"] = (raw["RIDSTATR"] == 2).astype(int)

    pir = raw["INDFMPIR"]
    d["income"] = pd.cut(pir, [-np.inf, 1.3, 3.5, np.inf], labels=["PIR<1.3", "PIR 1.3-3.5", "PIR>3.5"],
                         right=False).astype(object)
    d["education"] = raw["DMDEDUC2"].map({1: "<HS", 2: "<HS", 3: "HS/GED", 4: "Some college",
                                          5: "College+"})

    # Survey design: 20-year combined MEC weight (4-yr weight for 1999-2002 per NCHS guidance)
    early = d["cycle_start"].isin([1999, 2001])
    d["wt"] = np.where(early, raw["WTMEC4YR"] * 2 / 10, raw["WTMEC2YR"] / 10)
    d["strata"] = raw["SDMVSTRA"]
    d["psu"] = raw["SDMVPSU"]

    # Outcomes from the public-use linked mortality file (follow-up through 2019-12-31)
    d["linkage_eligible"] = (raw["ELIGSTAT"] == 1).astype(int)
    d["time"] = raw["PERMTH_EXM"] / 12.0
    died = raw["MORTSTAT"] == 1
    cvd = died & raw["UCOD_LEADING"].isin(CVD_UCOD)
    d["event"] = np.select([cvd, died], [1, 2], 0)  # 0 alive/censored, 1 CVD death, 2 other death
    return d


def add_missingness(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for group, cols in LAB_GROUPS.items():
        d[f"has_{group}"] = d[list(cols)].notna().all(axis=1).astype(int)
    d["all_labs"] = d[[f"has_{g}" for g in LAB_GROUPS]].all(axis=1).astype(int)
    d["tier"] = available_tiers(d)
    return d


# column -> (type/units, definition, NHANES/LMF source variables)
DICTIONARY = {
    "SEQN": ("int", "Respondent sequence number (unique across cycles)", "DEMO.SEQN"),
    "cycle": ("str", "NHANES survey cycle, e.g. 1999-2000", "file"),
    "cycle_start": ("int", "First calendar year of the cycle", "file"),
    "age": ("years", "Age at screening", "RIDAGEYR"),
    "female": ("0/1", "Female sex", "RIAGENDR==2"),
    "sex": ("str", "Female / Male", "RIAGENDR"),
    "race": ("str", "Race/ethnicity (Mexican American, Other Hispanic, NH White, NH Black, Other/Multiracial)",
             "RIDRETH1"),
    "sbp": ("mmHg", "Mean of available auscultatory systolic readings", "BPXSY1-BPXSY4"),
    "bp_treated": ("0/1", "Currently taking prescribed BP-lowering medication", "BPQ020, BPQ050A"),
    "smoker": ("0/1", "Current smoker: >=100 lifetime cigarettes and now smokes every day or some days",
               "SMQ020, SMQ040"),
    "diabetes": ("0/1", "Office definition: self-reported diagnosis, insulin, or oral agents (no HbA1c)",
                 "DIQ010, DIQ050, DIQ070"),
    "bmi": ("kg/m2", "Body mass index", "BMXBMI"),
    "total_chol": ("mg/dL", "Serum total cholesterol", "LBXTC (LAB13, L13_B, L13_C, TCHOL_D-J)"),
    "hdl": ("mg/dL", "HDL cholesterol", "LBDHDL (1999-2002), LBXHDD (2003-04), LBDHDD (2005-18)"),
    "hba1c": ("%", "Glycohemoglobin", "LBXGH (LAB10, L10_B, L10_C, GHB_D-J)"),
    "creatinine": ("mg/dL", "Serum creatinine standardized to IDMS (1999-2000: 1.013x+0.147; 2005-06: "
                   "-0.016+0.978x)", "LBXSCR / LBDSCR (LAB18, L40_B, L40_C, BIOPRO_D-J)"),
    "egfr": ("mL/min/1.73m2", "CKD-EPI 2021 race-free eGFR", "derived from creatinine, age, sex"),
    "uacr": ("mg/g", "Urine albumin-to-creatinine ratio, URXUMA / URXUCR * 100; pre-2007 urine creatinine adjusted "
             "to the 2007+ enzymatic method (NHANES ALB_CR_E equations)",
             "URXUMA, URXUCR (LAB16, L16_B, L16_C, ALB_CR_D-J)"),
    "diabetes_any": ("0/1", "Descriptive only: office diabetes or HbA1c >= 6.5%", "diabetes, LBXGH"),
    "statin": ("0/1", "Currently taking prescribed lipid-lowering medication (statin proxy)", "BPQ100D"),
    "prior_cvd": ("0/1", "Self-reported CHF, CHD, angina, MI or stroke (exclusion)", "MCQ160B-F"),
    "pregnant": ("0/1", "Pregnant at exam (exclusion)", "RIDEXPRG==1"),
    "mec_examined": ("0/1", "Examined at mobile examination center", "RIDSTATR==2"),
    "income": ("str", "Family income-to-poverty ratio group: <1.3, 1.3-3.5, >3.5", "INDFMPIR"),
    "education": ("str", "Highest education (adults 20+): <HS, HS/GED, Some college, College+", "DMDEDUC2"),
    "wt": ("weight", "20-year combined MEC weight: WTMEC4YR*2/10 for 1999-2002, WTMEC2YR/10 otherwise",
           "WTMEC2YR, WTMEC4YR"),
    "strata": ("int", "Masked variance pseudo-stratum", "SDMVSTRA"),
    "psu": ("int", "Masked variance pseudo-PSU", "SDMVPSU"),
    "linkage_eligible": ("0/1", "Eligible for mortality linkage", "LMF ELIGSTAT==1"),
    "time": ("years", "Follow-up from MEC exam to death or 2019-12-31", "LMF PERMTH_EXM/12"),
    "event": ("0/1/2", "0 alive at end of follow-up, 1 CVD death (UCOD 001 heart disease or 005 cerebrovascular), "
              "2 non-CVD death", "LMF MORTSTAT, UCOD_LEADING"),
    "has_lipids": ("0/1", "Total cholesterol and HDL both present", "derived"),
    "has_glyc_renal": ("0/1", "HbA1c and eGFR both present", "derived"),
    "has_uacr": ("0/1", "UACR present", "derived"),
    "all_labs": ("0/1", "All laboratory inputs present (tier comparison subset)", "derived"),
    "tier": ("str", "Highest nested tier (T0-T3) whose inputs are all present", "derived"),
}


def write_dictionary(columns, path):
    missing = [c for c in columns if c not in DICTIONARY]
    if missing:
        raise ValueError(f"undocumented cohort columns: {missing}")
    rows = [{"column": c, "type_units": DICTIONARY[c][0], "definition": DICTIONARY[c][1], "source": DICTIONARY[c][2]}
            for c in columns]
    pd.DataFrame(rows).to_csv(path, index=False)


def select(d: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Apply eligibility steps; return the cohort and the flow as (label, n) pairs."""
    flow = [("All NHANES 1999-2018 participants", len(d))]

    def step(mask, label):
        nonlocal d
        d = d[mask.loc[d.index]]
        flow.append((label, len(d)))

    step(d["age"].between(40, 79), "Age 40-79 at screening")
    step(d["mec_examined"] == 1, "Examined at mobile examination center")
    step(d["pregnant"] == 0, "Not pregnant")
    step(d["prior_cvd"] == 0, "No self-reported CHD, angina, MI, heart failure or stroke")
    step((d["linkage_eligible"] == 1) & d["time"].notna(), "Eligible for mortality linkage with follow-up")
    step(d["wt"] > 0, "Positive MEC examination weight")
    step(d[list(TIERS["T0"])].notna().all(axis=1), "Complete office inputs (" + ", ".join(TIERS["T0"]) + ")")
    return d.reset_index(drop=True), flow


def missingness_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in dict.fromkeys(c for t in TIERS.values() for c in t if c not in TIERS["T0"]):
        miss = d[var].isna()
        rows.append({"variable": var, "n_missing": int(miss.sum()), "pct_missing": round(100 * miss.mean(), 2),
                     "pct_missing_weighted": round(100 * np.average(miss, weights=d["wt"]), 2)})
    return pd.DataFrame(rows)


def build_cohort(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    print("Reading NHANES component files ...", flush=True)
    raw = pd.concat([load_cycle(raw_dir, *c) for c in CYCLES], ignore_index=True)
    raw = raw.merge(load_mortality(raw_dir), on="SEQN", how="left")
    d, flow = select(derive(raw))
    d = add_missingness(d)

    out = data_dir / "processed"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "cohort.csv.gz", index=False)
    write_dictionary(d.columns, out / "data_dictionary.csv")
    flow_df = pd.DataFrame(flow, columns=["step", "n"])
    flow_df["excluded"] = (-flow_df["n"].diff()).fillna(0).astype(int)
    flow_df.to_csv(out / "cohort_flow.csv", index=False)
    miss = missingness_table(d)
    miss.to_csv(out / "missingness.csv", index=False)

    meta = {
        "n": int(len(d)),
        "cvd_deaths": int((d["event"] == 1).sum()),
        "other_deaths": int((d["event"] == 2).sum()),
        "median_followup_years": float(d["time"].median()),
        "n_by_tier": {k: int(v) for k, v in d["tier"].value_counts().sort_index().items()},
        "n_all_labs": int(d["all_labs"].sum()),
        "cvd_deaths_all_labs": int(((d["event"] == 1) & (d["all_labs"] == 1)).sum()),
        "by_cycle": d.groupby("cycle").agg(n=("SEQN", "size"), all_labs=("all_labs", "sum"),
                                           cvd_deaths=("event", lambda e: int((e == 1).sum())),
                                           max_fu=("time", "max")).reset_index().to_dict("records"),
    }
    (out / "cohort_meta.json").write_text(json.dumps(meta, indent=1, default=float))

    print("\nCohort flow:")
    print(flow_df.to_string(index=False))
    print("\nLaboratory missingness:")
    print(miss.to_string(index=False))
    print(f"\nCohort n={meta['n']:,} (all labs present: {meta['n_all_labs']:,}); CVD deaths={meta['cvd_deaths']:,} "
          f"({meta['cvd_deaths_all_labs']:,} in all-labs subset); median follow-up "
          f"{meta['median_followup_years']:.1f} y; by tier {meta['n_by_tier']}")
    print(f"Written: {out / 'cohort.csv.gz'}")
    return d
