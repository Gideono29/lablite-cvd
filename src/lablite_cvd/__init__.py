"""LabLite-CVD: tiered-input cardiovascular risk model (research use only).

Not a medical device. Not validated for clinical decision-making.
"""

from lablite_cvd.tiers import TIERS, available_tier

__version__ = "0.1.0.dev0"

RESEARCH_USE_NOTICE = (
    "LabLite-CVD is for research use only. It is not a medical device and must not be "
    "used for individual clinical decisions."
)

__all__ = ["TIERS", "available_tier", "RESEARCH_USE_NOTICE", "__version__"]
