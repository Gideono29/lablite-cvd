"""Censoring-aware, survey-weighted performance metrics.

IPCW, AUC and calibration ported from EquiCVD Bench v1.0.0. Outcome: CVD death within ``horizon`` years with
non-CVD death as a competing event; administratively censored participants are handled by inverse probability
of censoring weights, with the censoring distribution estimated by Kaplan-Meier within NHANES cycle.
"""
import numpy as np

# 1/5/10%: ESC SCORE fatal-CVD cut points; 7.5/20%: ACC/AHA ASCVD cut points. Primary set not yet chosen.
THRESHOLDS = (0.01, 0.05, 0.075, 0.10, 0.20)


def censoring_km(time, event, horizon):
    """Return G(t-) at each subject's min(time, horizon) evaluation point and G(horizon)."""
    cens = event == 0
    u = np.unique(time[cens])
    if len(u) == 0:
        return np.ones(len(time)), 1.0
    st = np.sort(time)
    n_risk = len(time) - np.searchsorted(st, u, side="left")
    d = np.bincount(np.searchsorted(u, time[cens]), minlength=len(u))
    surv = np.cumprod(1 - d / n_risk)
    k_minus = np.searchsorted(u, np.minimum(time, horizon), side="left")  # censorings strictly before t
    g_minus = np.where(k_minus > 0, surv[np.maximum(k_minus - 1, 0)], 1.0)
    kh = np.searchsorted(u, horizon, side="right")
    return g_minus, (surv[kh - 1] if kh > 0 else 1.0)


def ipcw(df, horizon, strata_col="cycle"):
    """Binary horizon outcome y, IPCW weight w, and per-stratum G(horizon)."""
    y = ((df["event"] == 1) & (df["time"] <= horizon)).to_numpy(int)
    w = np.zeros(len(df))
    g_h = {}
    for s, idx in df.groupby(strata_col).indices.items():
        t, e = df["time"].to_numpy()[idx], df["event"].to_numpy()[idx]
        g_minus, gh = censoring_km(t, e, horizon)
        g_h[s] = gh
        died = (e != 0) & (t <= horizon)
        past = t > horizon
        wi = np.zeros(len(idx))
        wi[died] = 1 / g_minus[died]
        if gh > 0:
            wi[past] = 1 / gh
        w[idx] = wi
    return y, w, g_h


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def weighted_logistic(x, y, w, offset=None, iters=25):
    """Weighted logistic regression y ~ a + b*x (or y ~ a + offset). Returns (a, b)."""
    X = np.column_stack([np.ones_like(x), x]) if offset is None else np.ones((len(x), 1))
    off = np.zeros_like(x) if offset is None else offset
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta + off, -35, 35)
        mu = 1 / (1 + np.exp(-eta))
        g = X.T @ (w * (y - mu))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return (np.nan, np.nan)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return (beta[0], beta[1]) if offset is None else (beta[0], np.nan)


def weighted_auc(p, y, w):
    case, ctrl = w * y, w * (1 - y)
    if case.sum() <= 0 or ctrl.sum() <= 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    ps, cs, ks = p[order], case[order], ctrl[order]
    uniq, start = np.unique(ps, return_index=True)
    c_case = np.add.reduceat(cs, start)
    c_ctrl = np.add.reduceat(ks, start)
    below = np.cumsum(c_ctrl) - c_ctrl
    return float(np.sum(c_case * (below + 0.5 * c_ctrl)) / (cs.sum() * ks.sum()))


def net_benefit(p, y, W, t):
    """Net benefit of treating p >= t: TP/N - FP/N * t/(1-t), with combined survey x IPCW weights W."""
    flag = p >= t
    n = W.sum()
    return float((np.sum(W * y * flag) - np.sum(W * (1 - y) * flag) * t / (1 - t)) / n)


def performance(p, y, w, sw):
    """Metrics for one prediction vector. sw = survey (or bootstrap) weight; w = IPCW."""
    W = sw * w
    known = W > 0
    pk, yk, Wk = p[known], y[known], W[known]
    obs = float(np.sum(Wk * yk) / Wk.sum())
    exp_ = float(np.sum(sw * p) / sw.sum())
    lp = _logit(pk)
    citl, _ = weighted_logistic(lp, yk, Wk / Wk.mean(), offset=lp)
    _, slope = weighted_logistic(lp, yk, Wk / Wk.mean())
    brier = float(np.sum(Wk * (yk - pk) ** 2) / Wk.sum())
    out = {"observed": obs, "expected": exp_, "OE": obs / exp_, "cal_intercept": citl, "cal_slope": slope,
           "AUC": weighted_auc(pk, yk, Wk), "brier": brier}
    for t in THRESHOLDS:
        tag = f"{t * 100:g}"
        out[f"NB_{tag}"] = net_benefit(pk, yk, Wk, t)
        out[f"NB_all_{tag}"] = net_benefit(np.ones_like(pk), yk, Wk, t)
        out[f"pct_flagged_{tag}"] = float(np.sum(sw * (p >= t)) / sw.sum())
    return out
