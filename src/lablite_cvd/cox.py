"""Weighted ridge-penalized Cox proportional hazards model (Breslow ties), numpy only."""
from __future__ import annotations

import numpy as np


def _risk_sets(time):
    """Ascending sort order and, for each sorted row, the first index of its risk set (time >= t)."""
    order = np.argsort(time, kind="mergesort")
    ts = time[order]
    return order, np.searchsorted(ts, ts, side="left")


def _sums(X, w, eta, order, start, second=True):
    """Weighted risk-set sums S0, S1 (and S2) evaluated at every sorted row."""
    r = w[order] * np.exp(eta[order])
    Xs = X[order]
    s0 = np.cumsum(r[::-1])[::-1][start]
    s1 = np.cumsum((r[:, None] * Xs)[::-1], axis=0)[::-1][start]
    s2 = None
    if second:
        outer = r[:, None, None] * Xs[:, :, None] * Xs[:, None, :]
        s2 = np.cumsum(outer[::-1], axis=0)[::-1][start]
    return s0, s1, s2


def log_partial_likelihood(beta, X, time, event, w):
    """Weighted Breslow log partial likelihood (no penalty)."""
    order, start = _risk_sets(time)
    eta = X @ beta
    s0, _, _ = _sums(X, w, eta, order, start, second=False)
    d = (w * event)[order]
    return float(np.sum(d * (eta[order] - np.log(s0))))


def fit(X, time, event, w=None, lam=0.0, max_iter=50, tol=1e-9):
    """Maximize log PL - lam/2 * ||beta||^2 by Newton-Raphson. Returns beta."""
    X = np.asarray(X, float)
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    w = np.ones(len(time)) if w is None else np.asarray(w, float)
    order, start = _risk_sets(time)
    d = (w * event)[order]
    keep = d > 0
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        s0, s1, s2 = _sums(X, w, X @ beta, order, start)
        s0, s1, s2, dk = s0[keep], s1[keep], s2[keep], d[keep]
        mean = s1 / s0[:, None]
        grad = np.sum(dk[:, None] * (X[order][keep] - mean), axis=0) - lam * beta
        info = np.einsum("i,ijk->jk", dk, s2 / s0[:, None, None] - mean[:, :, None] * mean[:, None, :])
        step = np.linalg.solve(info + lam * np.eye(len(beta)), grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def baseline_cumhaz(X, time, event, w, beta, horizon):
    """Breslow baseline cumulative hazard at ``horizon`` (for linear predictor X @ beta = 0)."""
    X = np.asarray(X, float)
    time = np.asarray(time, float)
    order, start = _risk_sets(time)
    s0, _, _ = _sums(X, np.asarray(w, float), X @ beta, order, start, second=False)
    d = (np.asarray(w, float) * np.asarray(event, float))[order]
    at = (d > 0) & (time[order] <= horizon)
    return float(np.sum(d[at] / s0[at]))


def cv_log_likelihood(X, time, event, w, lam, folds):
    """Verweij-van Houwelingen cross-validated log partial likelihood for one penalty value."""
    total = 0.0
    for k in np.unique(folds):
        tr = folds != k
        b = fit(X[tr], time[tr], event[tr], w[tr], lam)
        total += (log_partial_likelihood(b, X, time, event, w)
                  - log_partial_likelihood(b, X[tr], time[tr], event[tr], w[tr]))
    return total
