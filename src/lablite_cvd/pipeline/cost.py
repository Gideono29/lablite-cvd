"""Information cost of missing laboratory inputs, with survey-bootstrap uncertainty.

A comparison (richer, poorer) reports metric(richer) - metric(poorer): the performance lost when the labs that
separate the two tiers are missing. Bootstrap: Rao-Wu rescaling over PSUs within strata applied to fixed
out-of-fold predictions and IPCW weights (models are not refitted per replicate).
"""
import numpy as np
import pandas as pd

from lablite_cvd.pipeline.metrics import PRIMARY_THRESHOLDS, SECONDARY_THRESHOLDS, THRESHOLDS, performance

COMPARISONS = [
    ("T3", "T0", "all labs"),
    ("T3", "T1", "HbA1c, eGFR, UACR"),
    ("T3", "T2", "UACR"),
    ("T1", "T0", "lipids"),
    ("T2", "T1", "HbA1c, eGFR"),
]
# Risk categories for reclassification: name -> inner cut points (1/5/10 primary)
CATEGORY_SETS = {"1/5/10": PRIMARY_THRESHOLDS, "7.5/20": SECONDARY_THRESHOLDS}
DELTA_METRICS = ["AUC", "cal_slope", "OE"] + [f"NB_{t * 100:g}" for t in THRESHOLDS]


def _reclassified(p_rich, p_poor, cuts, sw):
    return float(np.sum(sw * (np.digitize(p_rich, cuts) != np.digitize(p_poor, cuts))) / sw.sum())


def tier_metrics(preds: pd.DataFrame, y, w, sw) -> dict:
    return {t: performance(preds[t].to_numpy(float), y, w, sw) for t in preds.columns}


def cost_rows(preds: pd.DataFrame, y, w, sw, perf=None) -> dict:
    """Flat dict {(comparison, metric): value} for one set of weights."""
    perf = perf or tier_metrics(preds, y, w, sw)
    out = {}
    for rich, poor, _ in COMPARISONS:
        key = f"{rich} vs {poor}"
        for m in DELTA_METRICS:
            out[(key, f"d{m}")] = perf[rich][m] - perf[poor][m]
        for name, cuts in CATEGORY_SETS.items():
            out[(key, f"pct_reclassified_{name}")] = _reclassified(preds[rich].to_numpy(float),
                                                                   preds[poor].to_numpy(float), cuts, sw)
    return out


def rao_wu_multipliers(strata, psu, rng):
    """Rao-Wu rescaling bootstrap: resample n_h-1 PSUs within each stratum; returns per-person multipliers."""
    mult = np.zeros(len(strata))
    for s in np.unique(strata):
        in_s = strata == s
        psus = np.unique(psu[in_s])
        n_h = len(psus)
        if n_h < 2:
            mult[in_s] = 1.0
            continue
        draws = rng.choice(psus, size=n_h - 1, replace=True)
        for q in psus:
            mult[in_s & (psu == q)] = (draws == q).sum() * n_h / (n_h - 1)
    return mult


def bootstrap(preds: pd.DataFrame, y, w, wt, strata, psu, B=200, seed=0, level=0.95):
    """Point estimates and percentile intervals for per-tier metrics and information-cost deltas."""
    sw = wt / wt.mean()
    perf = tier_metrics(preds, y, w, sw)
    point_cost = cost_rows(preds, y, w, sw, perf)
    point_tier = {(t, m): v for t, ms in perf.items() for m, v in ms.items()}
    rng = np.random.default_rng(seed)
    reps_cost, reps_tier = [], []
    for b in range(B):
        swb = wt * rao_wu_multipliers(strata, psu, rng)
        swb = swb / swb.mean()
        pb = tier_metrics(preds, y, w, swb)
        reps_tier.append({(t, m): v for t, ms in pb.items() for m, v in ms.items()})
        reps_cost.append(cost_rows(preds, y, w, swb, pb))
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{B}", flush=True)
    a = (1 - level) / 2

    def summarize(point, reps, cols):
        rows = []
        for key, est in point.items():
            vals = np.array([r[key] for r in reps], float)
            vals = vals[np.isfinite(vals)]
            rows.append((*key, est, np.quantile(vals, a), np.quantile(vals, 1 - a), vals.std(ddof=1)))
        return pd.DataFrame(rows, columns=cols + ["estimate", "lo", "hi", "se"])

    return (summarize(point_tier, reps_tier, ["tier", "metric"]),
            summarize(point_cost, reps_cost, ["comparison", "metric"]))
