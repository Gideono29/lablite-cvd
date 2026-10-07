import numpy as np
import pandas as pd
import pytest

from lablite_cvd.pipeline.cost import bootstrap, cost_rows, rao_wu_multipliers


def test_rao_wu_resamples_psus_within_strata():
    strata = np.repeat([1, 1, 2, 2, 2], 10)
    psu = np.repeat([1, 2, 1, 2, 3], 10)
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rao_wu_multipliers(strata, psu, rng)
        # Constant within PSU; within each stratum, n_h - 1 draws rescaled by n_h / (n_h - 1)
        for s, n_h in ((1, 2), (2, 3)):
            per_psu = [m[(strata == s) & (psu == q)] for q in range(1, n_h + 1)]
            assert all(np.ptp(v) == 0 for v in per_psu)
            assert sum(v[0] for v in per_psu) == pytest.approx(n_h)


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-3 + 1.5 * x)))).astype(int)
    noise = rng.normal(size=n)
    risk = lambda lp: 1 / (1 + np.exp(-lp))
    preds = pd.DataFrame({"T0": risk(-3 + 0.3 * noise), "T1": risk(-3 + 0.5 * x + 0.3 * noise),
                          "T2": risk(-3 + 0.5 * x + 0.3 * noise), "T3": risk(-3 + 1.5 * x)})
    return preds, y, np.ones(n), rng


def test_cost_sign_is_richer_minus_poorer():
    preds, y, w, _ = _toy()
    c = cost_rows(preds, y, w, np.ones(len(y)))
    assert c[("T3 vs T0", "dAUC")] > 0.2
    assert c[("T2 vs T1", "dAUC")] == pytest.approx(0.0)
    assert c[("T2 vs T1", "pct_reclassified_1/5/10")] == 0.0


def test_bootstrap_interval_contains_estimate():
    preds, y, w, rng = _toy(2000)
    strata = np.repeat(np.arange(20), 100)
    psu = np.tile(np.repeat([1, 2], 50), 20)
    tiers, cost = bootstrap(preds, y, w, np.ones(len(y)), strata, psu, B=30, seed=1)
    row = cost[(cost.comparison == "T3 vs T0") & (cost.metric == "dAUC")].iloc[0]
    assert row.lo <= row.estimate <= row.hi and row.lo > 0
    assert set(tiers.tier) == {"T0", "T1", "T2", "T3"}
