"""Tier models: preprocessing, fitting, and 10-year risk prediction from published parameters."""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lablite_cvd import cox
from lablite_cvd.tiers import TIERS, available_tiers

# Model features per tier (``female`` from ``sex``, ``log_uacr`` from ``uacr``)
OFFICE = ["age", "female", "sbp", "bp_treated", "smoker", "diabetes", "bmi"]
LABS = {"T1": ["total_chol", "hdl"], "T2": ["hba1c", "egfr"], "T3": ["log_uacr"]}
FEATURES = {"T0": OFFICE}
for _t, _prev in (("T1", "T0"), ("T2", "T1"), ("T3", "T2")):
    FEATURES[_t] = FEATURES[_prev] + LABS[_t]

# Continuous features (except age) winsorized at the training 1st / 99th percentiles
CAPPED = ["sbp", "bmi", "total_chol", "hdl", "hba1c", "egfr", "log_uacr"]
CAP_PCTL = (1, 99)
AGE_RANGE = (40, 79)  # eligibility range of the development cohort
# Prespecified restricted cubic splines (4 knots) for every continuous feature except age, which stays linear
SPLINE = CAPPED
KNOT_PCTL = (5, 35, 65, 95)


def rcs_basis(x, knots):
    """Non-linear columns of a restricted cubic spline (Harrell): k knots -> k-2 columns."""
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    t_last, t_pen = k[-1], k[-2]
    cube = lambda u: np.clip(u, 0, None) ** 3
    cols = [cube(x - t) - cube(x - t_pen) * (t_last - t) / (t_last - t_pen)
            + cube(x - t_last) * (t_pen - t) / (t_last - t_pen) for t in k[:-2]]
    return np.column_stack(cols) / (t_last - k[0]) ** 2


def expand(X: pd.DataFrame, knots: dict) -> pd.DataFrame:
    """Linear columns for every feature plus spline columns ``<feature>_s1..`` for those in ``knots``."""
    out = X.copy()
    for f, kn in knots.items():
        basis = rcs_basis(X[f], kn)
        for j in range(basis.shape[1]):
            out[f"{f}_s{j + 1}"] = basis[:, j]
    return out


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from cohort / user columns (no capping)."""
    out = pd.DataFrame(index=df.index)
    if "female" not in df and "sex" not in df:
        raise ValueError("input needs a 'sex' column ('Female'/'Male' or 'F'/'M') or a 0/1 'female' column")
    if "female" in df:
        out["female"] = df["female"].astype(float)
    else:
        sex = df["sex"].astype(str).str.strip().str.upper().str[0]
        out["female"] = sex.map({"F": 1.0, "M": 0.0})
    for col in ["age", "sbp", "bp_treated", "smoker", "diabetes", "bmi", "total_chol", "hdl", "hba1c", "egfr"]:
        if col in df:
            out[col] = df[col].astype(float)
    if "uacr" in df:
        out["log_uacr"] = np.log(df["uacr"].astype(float).where(df["uacr"] > 0))
    return out


@dataclass
class TierModel:
    tier: str
    features: list
    caps: dict  # feature -> [low, high]
    knots: dict  # feature -> restricted cubic spline knots (on the capped scale)
    columns: list  # design columns: features, then spline columns
    center: list
    scale: list
    beta: list  # CVD death, per standardized design column
    beta_other: list  # non-CVD death (competing event), per standardized design column
    lam: float
    lam_other: float
    times: list  # distinct event times (years) up to the horizon, either cause
    dh_cvd: list  # baseline hazard increments on ``times`` at the centered linear predictor
    dh_other: list
    horizon: float

    def design(self, mf: pd.DataFrame) -> np.ndarray:
        X = mf[self.features].copy()
        for f, (lo, hi) in self.caps.items():
            X[f] = X[f].clip(lo, hi)
        X = expand(X, self.knots)[self.columns]
        return (X.to_numpy(float) - np.array(self.center)) / np.array(self.scale)

    def linear_predictor(self, mf: pd.DataFrame, cause: str = "cvd") -> np.ndarray:
        return self.design(mf) @ np.array(self.beta if cause == "cvd" else self.beta_other)

    def risk(self, mf: pd.DataFrame) -> np.ndarray:
        """Absolute risk of CVD death by the horizon, accounting for non-CVD death as a competing event."""
        Z = self.design(mf)
        return cox.cumulative_incidence(np.array(self.times), np.array(self.dh_cvd), np.array(self.dh_other),
                                        np.exp(Z @ np.array(self.beta)), np.exp(Z @ np.array(self.beta_other)))

    def coefficients(self) -> pd.DataFrame:
        """Coefficients per design column (standardized and original scale), both causes."""
        b, bo = np.array(self.beta), np.array(self.beta_other)
        sd = np.array(self.scale)
        return pd.DataFrame({"column": self.columns, "beta_per_sd": b, "beta_per_unit": b / sd,
                             "beta_other_per_sd": bo, "beta_other_per_unit": bo / sd, "sd": sd,
                             "mean": self.center})

    def shape(self, feature: str, grid, reference: float, cause: str = "cvd") -> np.ndarray:
        """Log hazard ratio of ``feature`` over ``grid`` relative to ``reference`` (other inputs fixed)."""
        cols = [c for c in self.columns if c == feature or c.startswith(feature + "_s")]
        idx = [self.columns.index(c) for c in cols]
        b = np.array(self.beta if cause == "cvd" else self.beta_other)[idx] / np.array(self.scale)[idx]

        def contrib(v):
            v = np.asarray(v, float)
            if feature in self.caps:
                v = np.clip(v, *self.caps[feature])
            X = pd.DataFrame({feature: v})
            return expand(X, {feature: self.knots[feature]} if feature in self.knots else {})[cols].to_numpy() @ b

        return contrib(grid) - contrib([reference])[0]


def prepare(mf: pd.DataFrame, feats: list, spline=SPLINE):
    """Cap continuous features at training percentiles, add spline columns, standardize.

    Returns (Z, caps, knots, columns, center, scale).
    """
    caps = {f: [float(v) for v in np.percentile(mf[f], CAP_PCTL)] for f in feats if f in CAPPED}
    X = mf[feats].copy()
    for f, (lo, hi) in caps.items():
        X[f] = X[f].clip(lo, hi)
    knots = {f: [float(v) for v in np.percentile(X[f], KNOT_PCTL)] for f in feats if f in spline}
    X = expand(X, knots)
    center, scale = X.mean().to_numpy(), X.std(ddof=0).to_numpy()
    return (X.to_numpy(float) - center) / scale, caps, knots, list(X.columns), center, scale


def fit_tier(tier, mf, time, event_cvd, event_other, w, lam, lam_other, horizon, spline=SPLINE) -> TierModel:
    """Fit cause-specific Cox models for CVD and non-CVD death on the same standardized design."""
    feats = FEATURES[tier]
    Z, caps, knots, columns, center, scale = prepare(mf, feats, spline)
    beta = cox.fit(Z, time, event_cvd, w, lam)
    beta_o = cox.fit(Z, time, event_other, w, lam_other)
    t1, h1 = cox.baseline_hazard(Z, time, event_cvd, w, beta, horizon)
    t2, h2 = cox.baseline_hazard(Z, time, event_other, w, beta_o, horizon)
    times = np.union1d(t1, t2)
    dh1, dh2 = np.zeros(len(times)), np.zeros(len(times))
    dh1[np.searchsorted(times, t1)] = h1
    dh2[np.searchsorted(times, t2)] = h2
    return TierModel(tier, feats, caps, knots, columns, center.tolist(), scale.tolist(), beta.tolist(),
                     beta_o.tolist(), float(lam), float(lam_other), times.tolist(), dh1.tolist(), dh2.tolist(),
                     float(horizon))


class LabLiteModel:
    """Set of tier models; ``predict`` picks the highest tier each row's inputs allow."""

    def __init__(self, tiers: dict, meta: dict | None = None):
        self.tiers = tiers
        self.meta = meta or {}

    def predict(self, df: pd.DataFrame, tier: str = "auto") -> pd.DataFrame:
        mf = model_frame(df)
        if "age" in mf and not mf["age"].dropna().between(*AGE_RANGE).all():
            warnings.warn(f"age outside {AGE_RANGE[0]}-{AGE_RANGE[1]} (the development range); predictions for "
                          "those rows are extrapolations", stacklevel=2)
        chosen = available_tiers(df) if tier == "auto" else pd.Series(tier, index=df.index)
        risk = pd.Series(np.nan, index=df.index)
        for name, m in self.tiers.items():
            rows = chosen == name
            if rows.any():
                risk[rows] = m.risk(mf[rows])
        return pd.DataFrame({"tier": chosen, "risk": risk})

    def to_json(self, path):
        payload = {"meta": self.meta, "tiers": {k: asdict(v) for k, v in self.tiers.items()}}
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def from_json(cls, path):
        text = path.read_text() if hasattr(path, "read_text") else Path(path).read_text()
        payload = json.loads(text)
        return cls({k: TierModel(**v) for k, v in payload["tiers"].items()}, payload.get("meta"))


_NOTICE_SHOWN = False


def load_model() -> LabLiteModel:
    """Load the published LabLite-CVD parameters shipped with the package (research use only)."""
    global _NOTICE_SHOWN
    from importlib.resources import files

    from lablite_cvd import RESEARCH_USE_NOTICE
    if not _NOTICE_SHOWN:
        warnings.warn(RESEARCH_USE_NOTICE, UserWarning, stacklevel=2)
        _NOTICE_SHOWN = True
    return LabLiteModel.from_json(files("lablite_cvd") / "data" / "lablite_params.json")


assert set(FEATURES) == set(TIERS)
