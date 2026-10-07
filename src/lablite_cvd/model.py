"""Tier models: preprocessing, fitting, and 10-year risk prediction from published parameters."""
from __future__ import annotations

import json
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

# Laboratory features winsorized at the training 1st / 99th percentiles
CAPPED = ["total_chol", "hdl", "hba1c", "egfr", "log_uacr"]
CAP_PCTL = (1, 99)


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from cohort / user columns (no capping)."""
    out = pd.DataFrame(index=df.index)
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
    center: list
    scale: list
    beta: list  # CVD death, per standardized feature
    beta_other: list  # non-CVD death (competing event), per standardized feature
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
        return (X.to_numpy(float) - np.array(self.center)) / np.array(self.scale)

    def linear_predictor(self, mf: pd.DataFrame, cause: str = "cvd") -> np.ndarray:
        return self.design(mf) @ np.array(self.beta if cause == "cvd" else self.beta_other)

    def risk(self, mf: pd.DataFrame) -> np.ndarray:
        """Absolute risk of CVD death by the horizon, accounting for non-CVD death as a competing event."""
        Z = self.design(mf)
        return cox.cumulative_incidence(np.array(self.times), np.array(self.dh_cvd), np.array(self.dh_other),
                                        np.exp(Z @ np.array(self.beta)), np.exp(Z @ np.array(self.beta_other)))

    def coefficients(self) -> pd.DataFrame:
        """Cause-specific hazard ratios per SD and per original unit."""
        b, bo = np.array(self.beta), np.array(self.beta_other)
        sd = np.array(self.scale)
        return pd.DataFrame({"feature": self.features, "beta_per_sd": b, "beta_per_unit": b / sd,
                             "hr_per_unit": np.exp(b / sd), "beta_other_per_sd": bo,
                             "hr_other_per_unit": np.exp(bo / sd), "sd": sd, "mean": self.center})


def prepare(mf: pd.DataFrame, feats: list):
    """Cap laboratory features at training percentiles and standardize. Returns (Z, caps, center, scale)."""
    caps = {f: [float(v) for v in np.percentile(mf[f], CAP_PCTL)] for f in feats if f in CAPPED}
    X = mf[feats].copy()
    for f, (lo, hi) in caps.items():
        X[f] = X[f].clip(lo, hi)
    center, scale = X.mean().to_numpy(), X.std(ddof=0).to_numpy()
    return (X.to_numpy(float) - center) / scale, caps, center, scale


def fit_tier(tier, mf, time, event_cvd, event_other, w, lam, lam_other, horizon) -> TierModel:
    """Fit cause-specific Cox models for CVD and non-CVD death on the same standardized features."""
    feats = FEATURES[tier]
    Z, caps, center, scale = prepare(mf, feats)
    beta = cox.fit(Z, time, event_cvd, w, lam)
    beta_o = cox.fit(Z, time, event_other, w, lam_other)
    t1, h1 = cox.baseline_hazard(Z, time, event_cvd, w, beta, horizon)
    t2, h2 = cox.baseline_hazard(Z, time, event_other, w, beta_o, horizon)
    times = np.union1d(t1, t2)
    dh1, dh2 = np.zeros(len(times)), np.zeros(len(times))
    dh1[np.searchsorted(times, t1)] = h1
    dh2[np.searchsorted(times, t2)] = h2
    return TierModel(tier, feats, caps, center.tolist(), scale.tolist(), beta.tolist(), beta_o.tolist(),
                     float(lam), float(lam_other), times.tolist(), dh1.tolist(), dh2.tolist(), float(horizon))


class LabLiteModel:
    """Set of tier models; ``predict`` picks the highest tier each row's inputs allow."""

    def __init__(self, tiers: dict, meta: dict | None = None):
        self.tiers = tiers
        self.meta = meta or {}

    def predict(self, df: pd.DataFrame, tier: str = "auto") -> pd.DataFrame:
        mf = model_frame(df)
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
        payload = json.loads(Path(path).read_text())
        return cls({k: TierModel(**v) for k, v in payload["tiers"].items()}, payload.get("meta"))


assert set(FEATURES) == set(TIERS)
