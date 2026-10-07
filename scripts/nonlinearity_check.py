"""Check whether restricted cubic splines improve on linear terms (full-lab tier, CVD and non-CVD death).

For each continuous feature, replaces its linear term with a 4-knot restricted cubic spline (knots at the
5/35/65/95th percentiles, Harrell) and compares 5-fold cross-validated partial log-likelihood with the
all-linear model, at the penalty chosen for the linear model. Writes outputs/fit/nonlinearity_check.csv.

Run after `lablite-cvd cohort`:  python scripts/nonlinearity_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from lablite_cvd import cox
from lablite_cvd.model import FEATURES, model_frame, prepare
from lablite_cvd.pipeline.fit import LAMBDAS, _folds, analysis_set

HORIZON, MIN_G, SEED = 10.0, 0.10, 20261009
CONTINUOUS = ["age", "sbp", "bmi", "total_chol", "hdl", "hba1c", "egfr", "log_uacr"]


def rcs_basis(x, knots):
    """Non-linear part of a restricted cubic spline (k knots -> k-2 columns)."""
    k = np.asarray(knots, float)
    t_last, t_pen = k[-1], k[-2]
    cube = lambda u: np.clip(u, 0, None) ** 3
    cols = [cube(x - t) - cube(x - t_pen) * (t_last - t) / (t_last - t_pen)
            + cube(x - t_last) * (t_pen - t) / (t_last - t_pen) for t in k[:-2]]
    return np.column_stack(cols) / (t_last - k[0]) ** 2


def cv_ll(Z, time, event, w, folds):
    Z = (Z - Z.mean(0)) / Z.std(0)
    scores = {lam: cox.cv_log_likelihood(Z, time, event, w, lam, folds) for lam in LAMBDAS}
    lam = max(scores, key=scores.get)
    return scores[lam], lam


def main(data_dir=Path("data"), out=Path("outputs/fit")):
    cohort = pd.read_csv(data_dir / "processed" / "cohort.csv.gz")
    d, _, _ = analysis_set(cohort, HORIZON, MIN_G)
    mf = model_frame(d)
    feats = FEATURES["T3"]
    Z, *_ = prepare(mf, feats)  # capped and standardized, as in the fitted models
    capped = pd.DataFrame(Z, columns=feats)
    time = np.minimum(d["time"].to_numpy(float), HORIZON)
    w = (d["wt"] / d["wt"].mean()).to_numpy(float)
    rows = []
    for cause, code in (("cvd", 1), ("non-cvd", 2)):
        event = ((d["event"] == code) & (d["time"] <= HORIZON)).to_numpy(float)
        folds = _folds(event, 5, np.random.default_rng(SEED))
        base, lam = cv_ll(Z, time, event, w, folds)
        for f in CONTINUOUS:
            x = capped[f].to_numpy()
            extra = rcs_basis(x, np.percentile(x, [5, 35, 65, 95]))
            ll, lam_s = cv_ll(np.column_stack([Z, extra]), time, event, w, folds)
            rows.append({"cause": cause, "feature": f, "cv_loglik_linear": base, "cv_loglik_spline": ll,
                         "delta": ll - base, "extra_df": extra.shape[1], "lambda_linear": lam,
                         "lambda_spline": lam_s})
            print(f"{cause:8s} {f:11s} delta CV log-lik = {ll - base:+.2f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "nonlinearity_check.csv", index=False)
    print(f"Written: {out / 'nonlinearity_check.csv'}")


if __name__ == "__main__":
    main()
