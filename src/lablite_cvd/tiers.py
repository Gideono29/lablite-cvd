"""Input tiers: which predictors each model tier requires.

DRAFT: tier membership is a modeling decision owned by the maintainer and is not final.
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
