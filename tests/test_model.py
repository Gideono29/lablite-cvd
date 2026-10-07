import numpy as np
import pandas as pd
import pytest

from lablite_cvd import cox
from lablite_cvd.model import FEATURES, LabLiteModel, fit_tier, model_frame
from lablite_cvd.pipeline.metrics import net_benefit


def _sim(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.uniform(40, 79, n), "sex": rng.choice(["Female", "Male"], n), "sbp": rng.normal(128, 18, n),
        "bp_treated": rng.integers(0, 2, n), "smoker": rng.integers(0, 2, n), "diabetes": rng.integers(0, 2, n),
        "bmi": rng.normal(29, 6, n), "total_chol": rng.normal(200, 40, n), "hdl": rng.normal(53, 15, n),
        "hba1c": rng.normal(5.8, 0.8, n), "egfr": rng.normal(88, 18, n), "uacr": rng.lognormal(2, 1.2, n),
    })
    lp = 0.08 * (df["age"] - 60) + 0.5 * df["smoker"] + 0.3 * np.log(df["uacr"])
    t = rng.exponential(60 / np.exp(lp))
    c = rng.uniform(5, 15, n)
    return df, np.minimum(t, c), (t <= c).astype(float)


def test_weights_equal_duplication():
    df, time, event = _sim(500)
    X = model_frame(df)[FEATURES["T0"]].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    w = np.random.default_rng(1).integers(1, 4, len(df)).astype(float)
    idx = np.repeat(np.arange(len(df)), w.astype(int))
    np.testing.assert_allclose(cox.fit(X, time, event, w), cox.fit(X[idx], time[idx], event[idx]), atol=1e-8)


def test_ridge_shrinks_coefficients():
    df, time, event = _sim(500)
    X = model_frame(df)[FEATURES["T1"]].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    assert np.linalg.norm(cox.fit(X, time, event, lam=200.0)) < np.linalg.norm(cox.fit(X, time, event))


def test_fit_recovers_signal_and_json_roundtrip(tmp_path):
    df, time, event = _sim()
    mf = model_frame(df)
    w = np.ones(len(df))
    model = LabLiteModel({t: fit_tier(t, mf, time, event, 0 * event, w, 1.0, 1.0, 10.0) for t in FEATURES})
    coef = model.tiers["T3"].coefficients().set_index("feature")
    assert coef.loc["log_uacr", "beta_per_unit"] == pytest.approx(0.3, abs=0.1)
    assert coef.loc["smoker", "beta_per_unit"] == pytest.approx(0.5, abs=0.15)

    model.to_json(tmp_path / "m.json")
    again = LabLiteModel.from_json(tmp_path / "m.json")
    np.testing.assert_allclose(again.predict(df)["risk"], model.predict(df)["risk"])


def test_predict_auto_uses_highest_available_tier():
    df, time, event = _sim()
    model = LabLiteModel({t: fit_tier(t, model_frame(df), time, event, 0 * event, np.ones(len(df)), 1.0, 1.0, 10.0)
                          for t in FEATURES})
    rows = df.iloc[:3].copy()
    rows.loc[rows.index[1], "uacr"] = np.nan
    rows.loc[rows.index[2], "hdl"] = np.nan
    out = model.predict(rows)
    assert out["tier"].tolist() == ["T3", "T2", "T0"]
    assert out["risk"].between(0, 1).all()
    forced = model.predict(rows.iloc[[0]], tier="T0")["risk"].iloc[0]
    assert forced == pytest.approx(model.tiers["T0"].risk(model_frame(rows.iloc[[0]]))[0])


def test_capping_limits_extreme_inputs():
    df, time, event = _sim()
    m = fit_tier("T3", model_frame(df), time, event, 0 * event, np.ones(len(df)), 1.0, 1.0, 10.0)
    extreme = df.iloc[[0]].assign(uacr=1e6)
    capped = df.iloc[[0]].assign(uacr=np.exp(m.caps["log_uacr"][1]))
    assert m.risk(model_frame(extreme))[0] == pytest.approx(m.risk(model_frame(capped))[0])


def test_net_benefit_known_values():
    y = np.array([1, 0, 0, 0])
    W = np.ones(4)
    assert net_benefit(np.array([0.9, 0.0, 0.0, 0.0]), y, W, 0.2) == pytest.approx(0.25)
    # treat all: 1/4 - 3/4 * 0.25
    assert net_benefit(np.ones(4), y, W, 0.2) == pytest.approx(0.25 - 0.75 * 0.25)


def test_cumulative_incidence_matches_competing_risk_truth():
    # No covariate effects, constant cause-specific hazards, censoring at 12 years. With no censoring before the
    # horizon, the cumulative incidence estimate must equal the empirical proportion dying of cause 1 by 10 y,
    # and both should be near h1 / (h1 + h2) * (1 - exp(-(h1 + h2) * 10)).
    rng = np.random.default_rng(3)
    n, h1, h2 = 40000, 0.004, 0.012
    df = _sim(n, seed=4)[0]
    t_all = rng.exponential(1 / (h1 + h2), n)
    cause = np.where(rng.uniform(size=n) < h1 / (h1 + h2), 1, 2)
    time = np.minimum(np.ceil(t_all * 12) / 12, 12.0)  # monthly, like the linked mortality file
    died = t_all < 12.0
    m = fit_tier("T0", model_frame(df), time, (died & (cause == 1)).astype(float),
                 (died & (cause == 2)).astype(float), np.ones(n), 1000.0, 1000.0, 10.0)
    risk = m.risk(model_frame(df)).mean()
    assert risk == pytest.approx(np.mean(died & (cause == 1) & (t_all <= 10)), abs=1e-4)
    assert risk == pytest.approx(h1 / (h1 + h2) * (1 - np.exp(-(h1 + h2) * 10.0)), abs=0.003)


def test_competing_risk_lowers_risk():
    df, time, event = _sim()
    rng = np.random.default_rng(5)
    other = ((rng.uniform(size=len(df)) < 0.3) & (event == 0)).astype(float)
    mf = model_frame(df)
    w = np.ones(len(df))
    without = fit_tier("T0", mf, time, event, 0 * event, w, 1.0, 1.0, 10.0).risk(mf)
    with_cr = fit_tier("T0", mf, time, event, other, w, 1.0, 1.0, 10.0).risk(mf)
    assert np.all(with_cr <= without + 1e-12) and with_cr.mean() < without.mean()
