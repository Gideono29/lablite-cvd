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
    beta: list  # per standardized feature
    lam: float
    cumhaz0: float  # baseline cumulative hazard at horizon, at the centered linear predictor
    horizon: float

    def design(self, mf: pd.DataFrame) -> np.ndarray:
        X = mf[self.features].copy()
        for f, (lo, hi) in self.caps.items():
            X[f] = X[f].clip(lo, hi)
        return (X.to_numpy(float) - np.array(self.center)) / np.array(self.scale)

    def linear_predictor(self, mf: pd.DataFrame) -> np.ndarray:
        return self.design(mf) @ np.array(self.beta)

    def risk(self, mf: pd.DataFrame) -> np.ndarray:
        return 1 - np.exp(-self.cumhaz0 * np.exp(self.linear_predictor(mf)))

    def coefficients(self) -> pd.DataFrame:
        """Hazard ratios per SD and per original unit."""
        b = np.array(self.beta)
        sd = np.array(self.scale)
        return pd.DataFrame({"feature": self.features, "beta_per_sd": b, "beta_per_unit": b / sd,
                             "hr_per_unit": np.exp(b / sd), "sd": sd, "mean": self.center})


def prepare(mf: pd.DataFrame, feats: list):
    """Cap laboratory features at training percentiles and standardize. Returns (Z, caps, center, scale)."""
    caps = {f: [float(v) for v in np.percentile(mf[f], CAP_PCTL)] for f in feats if f in CAPPED}
    X = mf[feats].copy()
    for f, (lo, hi) in caps.items():
        X[f] = X[f].clip(lo, hi)
    center, scale = X.mean().to_numpy(), X.std(ddof=0).to_numpy()
    return (X.to_numpy(float) - center) / scale, caps, center, scale


def fit_tier(tier, mf, time, event, w, lam, horizon) -> TierModel:
    feats = FEATURES[tier]
    Z, caps, center, scale = prepare(mf, feats)
    beta = cox.fit(Z, time, event, w, lam)
    h0 = cox.baseline_cumhaz(Z, time, event, w, beta, horizon)
    return TierModel(tier, feats, caps, center.tolist(), scale.tolist(), beta.tolist(), float(lam), h0,
                     float(horizon))


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
