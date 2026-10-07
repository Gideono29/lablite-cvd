"""Input tiers: which predictors each model tier requires.

Tier membership approved by the maintainer on 2026-10-07 (docs/open_questions.md).
Each tier is a strict superset of the one before it.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

TIERS: dict[str, tuple[str, ...]] = {}
TIERS["T0"] = ("age", "sex", "sbp", "bp_treated", "smoker", "diabetes", "bmi")
TIERS["T1"] = TIERS["T0"] + ("total_chol", "hdl")
TIERS["T2"] = TIERS["T1"] + ("hba1c", "egfr")
TIERS["T3"] = TIERS["T2"] + ("uacr",)


def _present(value) -> bool:
    return value is not None and not pd.isna(value)


def available_tier(record: Mapping) -> str | None:
    """Highest tier whose required inputs are all present in ``record``, or None."""
    best = None
    for name, required in TIERS.items():
        if not all(_present(record.get(col)) for col in required):
            break
        best = name
    return best


def available_tiers(df: pd.DataFrame) -> pd.Series:
    """Row-wise :func:`available_tier` for a DataFrame (None where T0 is incomplete)."""
    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    still = pd.Series(True, index=df.index)
    for name, required in TIERS.items():
        cols = [c for c in required if c in df.columns]
        complete = df[cols].notna().all(axis=1) if len(cols) == len(required) else False
        still &= complete
        out[still] = name
    return out
