import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lablite_cvd

ROOT = Path(__file__).resolve().parents[1]

PATIENTS = pd.DataFrame({
    "age": [55, 70, 62], "sex": ["Female", "Male", "M"], "sbp": [128, 150, 140], "bp_treated": [0, 1, 1],
    "smoker": [0, 1, 0], "diabetes": [0, 1, 0], "bmi": [27.0, 31.0, 29.0],
    "total_chol": [210, 190, np.nan], "hdl": [60, 40, np.nan],
    "hba1c": [5.5, 7.2, 5.9], "egfr": [92, 61, 80], "uacr": [6.0, 85.0, 12.0],
})


@pytest.fixture(scope="module")
def model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lablite_cvd.load_model()


def test_packaged_parameters_match_fit_output():
    shipped = json.loads((ROOT / "src" / "lablite_cvd" / "data" / "lablite_params.json").read_text())
    fitted = json.loads((ROOT / "outputs" / "fit" / "lablite_params.json").read_text())
    assert shipped == fitted


def test_load_model_and_predict(model):
    out = model.predict(PATIENTS)
    assert out["tier"].tolist() == ["T3", "T3", "T0"]  # third patient lacks lipids
    assert out["risk"].between(0, 1).all()
    assert out["risk"].iloc[1] > out["risk"].iloc[0]  # older, smoker, diabetic, high UACR


def test_every_tier_available_for_complete_record(model):
    risks = [model.predict(PATIENTS.iloc[[0]], tier=t)["risk"].iloc[0] for t in lablite_cvd.TIERS]
    assert all(0 < r < 0.2 for r in risks)


def test_missing_sex_raises(model):
    with pytest.raises(ValueError, match="sex"):
        model.predict(PATIENTS.drop(columns="sex"))
