import numpy as np
import pandas as pd
import pytest

from lablite_cvd.pipeline.cohort import (add_missingness, adjust_urine_creatinine_pre2007, ckd_epi_2021, derive,
                                         missingness_table, select)


def test_ckd_epi_2021_reference_value():
    # 60-year-old man, creatinine 1.0 mg/dL -> 142 * (1/0.9)^-1.2 * 0.9938^60 = 86.2
    assert ckd_epi_2021([1.0], [60], [0])[0] == pytest.approx(86.2, abs=0.1)


def _raw(n=4, **overrides):
    base = dict(
        SEQN=np.arange(1, n + 1), cycle="2011-2012", cycle_start=2011,
        RIDAGEYR=55.0, RIAGENDR=2.0, RIDRETH1=3.0, RIDSTATR=2.0, RIDEXPRG=np.nan,
        BPXSY1=130.0, BPXSY2=126.0, BPXSY3=np.nan, BPXSY4=np.nan,
        BPQ020=2.0, BPQ050A=np.nan, BPQ100D=np.nan,
        SMQ020=2.0, SMQ040=np.nan, DIQ010=2.0, DIQ050=2.0, DIQ070=np.nan,
        BMXBMI=27.0, LBXTC=200.0, LBDHDD=50.0, LBXGH=5.6, LBXSCR=0.8,
        URXUMA=10.0, URXUCR=100.0,
        MCQ160B=2.0, MCQ160C=2.0, MCQ160D=2.0, MCQ160E=2.0, MCQ160F=2.0,
        INDFMPIR=2.0, DMDEDUC2=4.0, WTMEC2YR=20000.0, WTMEC4YR=np.nan, SDMVSTRA=1, SDMVPSU=1,
        ELIGSTAT=1, MORTSTAT=0, UCOD_LEADING=np.nan, PERMTH_EXM=96.0,
    )
    df = pd.DataFrame({k: [v] * n if np.isscalar(v) else v for k, v in base.items()})
    for col, vals in overrides.items():
        df[col] = vals
    return df


def test_office_diabetes_ignores_hba1c():
    d = derive(_raw(LBXGH=[5.6, 7.0, 7.0, 5.6], DIQ010=[2.0, 2.0, 1.0, 2.0]))
    assert d["diabetes"].tolist() == [0, 0, 1, 0]
    assert d["diabetes_any"].tolist() == [0, 1, 1, 0]


def test_uacr_units_and_zero_creatinine():
    d = derive(_raw(URXUMA=[10.0, 10.0, np.nan, 5.0], URXUCR=[100.0, 0.0, 100.0, 50.0]))
    assert d["uacr"].iloc[0] == pytest.approx(10.0)
    assert np.isnan(d["uacr"].iloc[1]) and np.isnan(d["uacr"].iloc[2])
    assert d["uacr"].iloc[3] == pytest.approx(10.0)


def test_events_coded_from_mortality_file():
    d = derive(_raw(MORTSTAT=[0, 1, 1, 1], UCOD_LEADING=[np.nan, 1, 5, 2]))
    assert d["event"].tolist() == [0, 1, 1, 2]


def test_missing_labs_kept_missing_office_excluded():
    raw = _raw(LBXTC=[200.0, np.nan, 200.0, 200.0], URXUMA=[10.0, 10.0, np.nan, 10.0],
               BMXBMI=[27.0, 27.0, 27.0, np.nan])
    d, flow = select(derive(raw))
    d = add_missingness(d)
    assert len(d) == 3 and flow[-1][1] == 3
    assert d["tier"].tolist() == ["T3", "T0", "T2"]
    assert d["all_labs"].tolist() == [1, 0, 0]
    table = missingness_table(d)
    assert table["variable"].is_unique
    assert table.set_index("variable").loc["total_chol", "n_missing"] == 1


def test_urine_creatinine_adjustment_equations():
    # One value per piece of the NHANES ALB_CR_E equations
    got = adjust_urine_creatinine_pre2007([50.0, 100.0, 300.0, np.nan])
    np.testing.assert_allclose(got[:3], [(1.02 * 50 ** 0.5 - 0.36) ** 2, (1.05 * 10 - 0.74) ** 2,
                                         (1.01 * 300 ** 0.5 - 0.10) ** 2])
    assert np.isnan(got[3])


def test_uacr_adjusted_only_before_2007():
    early = derive(_raw(cycle="2005-2006", cycle_start=2005))
    late = derive(_raw(cycle="2007-2008", cycle_start=2007))
    assert late["uacr"].iloc[0] == pytest.approx(10.0)
    assert early["uacr"].iloc[0] == pytest.approx(1000 / (1.05 * 10 - 0.74) ** 2)
