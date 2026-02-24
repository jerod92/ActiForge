"""
auto_actuary.analytics.market_insights
=======================================
Novel market intelligence capabilities for P&C insurance carriers.

Modules
-------
cycle_detection
    Identifies hard/soft market cycles using combined ratio trends,
    rate adequacy signals, and capacity proxies.  Outputs a market
    cycle indicator (MCI) score and phase classification.

opportunity_scoring
    Scores each portfolio segment (territory × class × LOB) on a
    multi-dimensional opportunity matrix that combines profitability,
    growth potential, competitive positioning, and retention dynamics.

anomaly_detection
    Statistical detection of loss trend breaks and emerging loss
    environment changes using CUSUM, Chow structural break tests,
    and Z-score outlier flagging.
"""

from auto_actuary.analytics.market_insights.cycle_detection import (
    MarketCycleDetector,
    MarketPhase,
)
from auto_actuary.analytics.market_insights.opportunity_scoring import (
    SegmentOpportunityScorer,
    OpportunityScore,
)
from auto_actuary.analytics.market_insights.anomaly_detection import (
    LossAnomalyDetector,
    AnomalyResult,
)

__all__ = [
    "MarketCycleDetector",
    "MarketPhase",
    "SegmentOpportunityScorer",
    "OpportunityScore",
    "LossAnomalyDetector",
    "AnomalyResult",
]
