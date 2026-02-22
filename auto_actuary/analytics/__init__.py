# auto_actuary analytics package

from auto_actuary.analytics.time_series import TimeSeriesManager, SnapshotStore
from auto_actuary.analytics.portfolio import (
    MarketBreakdownConfig,
    MarketBreakdownAnalysis,
    ProductMixAnalysis,
)
from auto_actuary.analytics.cause_of_loss import CauseOfLossAnalysis, CauseOfLossCorrelation
from auto_actuary.analytics.retention import RetentionAnalysis

__all__ = [
    # Time series
    "TimeSeriesManager",
    "SnapshotStore",
    # Portfolio / market breakdown
    "MarketBreakdownConfig",
    "MarketBreakdownAnalysis",
    "ProductMixAnalysis",
    # Cause of loss
    "CauseOfLossAnalysis",
    "CauseOfLossCorrelation",
    # Retention
    "RetentionAnalysis",
]
