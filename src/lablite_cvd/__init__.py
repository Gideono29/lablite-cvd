"""LabLite-CVD: tiered-input cardiovascular risk model (research use only).

Not a medical device. Not validated for clinical decision-making.

    >>> import lablite_cvd
    >>> model = lablite_cvd.load_model()
    >>> model.predict(df)  # columns: tier, risk (10-year CVD death)
"""

from lablite_cvd.tiers import TIERS, available_tier, available_tiers

__version__ = "0.1.0"

RESEARCH_USE_NOTICE = (
    "LabLite-CVD is for research use only. It is not a medical device and must not be "
    "used for individual clinical decisions."
)

from lablite_cvd.model import LabLiteModel, load_model  # noqa: E402  (after RESEARCH_USE_NOTICE)

__all__ = ["TIERS", "available_tier", "available_tiers", "LabLiteModel", "load_model", "RESEARCH_USE_NOTICE",
           "__version__"]
