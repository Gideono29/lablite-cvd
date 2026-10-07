import math

import pandas as pd

from lablite_cvd import TIERS, available_tier, available_tiers

OFFICE = dict(age=55, sex="F", sbp=130, bp_treated=0, smoker=0, diabetes=0, bmi=27.0)


def test_tiers_are_nested():
    names = list(TIERS)
    for lower, upper in zip(names, names[1:]):
        assert set(TIERS[lower]) < set(TIERS[upper])


def test_office_only_is_t0():
    assert available_tier(OFFICE) == "T0"


def test_missing_office_input_gives_none():
    assert available_tier({**OFFICE, "sbp": None}) is None


def test_nan_counts_as_missing():
    assert available_tier({**OFFICE, "total_chol": 200, "hdl": math.nan}) == "T0"


def test_full_record_is_top_tier():
    rec = {**OFFICE, "total_chol": 200, "hdl": 50, "hba1c": 5.6, "egfr": 90, "uacr": 10}
    assert available_tier(rec) == "T3"


def test_gap_stops_at_lower_tier():
    # Renal and urine labs present but lipids missing: only T0 is complete.
    assert available_tier({**OFFICE, "hba1c": 5.6, "egfr": 90, "uacr": 10}) == "T0"


def test_dataframe_version_matches_record_version():
    full = {**OFFICE, "total_chol": 200, "hdl": 50, "hba1c": 5.6, "egfr": 90, "uacr": 10}
    records = [OFFICE, full, {**full, "hdl": None}, {**full, "sbp": None}, {**full, "uacr": math.nan}]
    df = pd.DataFrame(records)
    assert available_tiers(df).tolist() == [available_tier(r) for r in records]
