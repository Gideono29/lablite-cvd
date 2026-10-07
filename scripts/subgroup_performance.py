"""Subgroup calibration (O/E) and discrimination (AUC) per tier, for the model card.

Uses the out-of-fold predictions from `lablite-cvd fit`, survey weights x IPCW, and the same Rao-Wu bootstrap
(B = 200, 95% percentile intervals). Writes outputs/fit/subgroup_performance.csv.

Run after `lablite-cvd fit`:  python scripts/subgroup_performance.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from lablite_cvd.model import FEATURES
from lablite_cvd.pipeline.cost import rao_wu_multipliers
from lablite_cvd.pipeline.fit import analysis_set
from lablite_cvd.pipeline.metrics import ipcw, weighted_auc

HORIZON, MIN_G, B, SEED = 10.0, 0.10, 200, 20261010


def oe_auc(p, y, w, sw):
    W = sw * w
    k = W > 0
    obs = np.sum(W[k] * y[k]) / W[k].sum()
    exp_ = np.sum(sw * p) / sw.sum()
    return obs / exp_, weighted_auc(p[k], y[k], W[k])


def main(data_dir=Path("data"), out=Path("outputs/fit")):
    cohort = pd.read_csv(data_dir / "processed" / "cohort.csv.gz")
    d, _, _ = analysis_set(cohort, HORIZON, MIN_G)
    oof = pd.read_csv(out / "oof_predictions.csv.gz")
    assert oof["SEQN"].equals(d["SEQN"])
    y, w, _ = ipcw(d, HORIZON)
    d["age_group"] = np.where(d["age"] < 60, "40-59", "60-79")
    groups = [("Overall", "All", np.ones(len(d), bool))]
    for attr in ("sex", "race", "age_group"):
        for lev in sorted(d[attr].dropna().unique()):
            groups.append((attr, lev, (d[attr] == lev).to_numpy()))
    wt = d["wt"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    mults = [rao_wu_multipliers(d["strata"].to_numpy(), d["psu"].to_numpy(), rng) for _ in range(B)]
    rows = []
    for tier in FEATURES:
        p = oof[tier].to_numpy(float)
        for attr, lev, m in groups:
            oe, auc = oe_auc(p[m], y[m], w[m], wt[m])
            reps = np.array([oe_auc(p[m], y[m], w[m], (wt * mu)[m]) for mu in mults], float)
            lo, hi = np.nanquantile(reps, [0.025, 0.975], axis=0)
            rows.append({"tier": tier, "attribute": attr, "level": lev, "n": int(m.sum()),
                         "cvd_deaths": int(y[m].sum()), "OE": oe, "OE_lo": lo[0], "OE_hi": hi[0],
                         "AUC": auc, "AUC_lo": lo[1], "AUC_hi": hi[1]})
        print(f"{tier} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "subgroup_performance.csv", index=False)
    print(res[res.tier.isin(["T0", "T3"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
