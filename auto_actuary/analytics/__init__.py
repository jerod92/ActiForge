# auto_actuary analytics package
# All imports are lazy-guarded so that optional heavy dependencies (scipy,
# statsmodels, scikit-learn) do not block core analytics when not installed.

try:
    from auto_actuary.analytics.time_series import TimeSeriesManager, SnapshotStore
except ImportError:
    TimeSeriesManager = None  # type: ignore[assignment,misc]
    SnapshotStore = None  # type: ignore[assignment,misc]

try:
    from auto_actuary.analytics.portfolio import (
        MarketBreakdownConfig,
        MarketBreakdownAnalysis,
        ProductMixAnalysis,
    )
except ImportError:
    MarketBreakdownConfig = None  # type: ignore[assignment,misc]
    MarketBreakdownAnalysis = None  # type: ignore[assignment,misc]
    ProductMixAnalysis = None  # type: ignore[assignment,misc]

try:
    from auto_actuary.analytics.cause_of_loss import CauseOfLossAnalysis, CauseOfLossCorrelation
except ImportError:
    CauseOfLossAnalysis = None  # type: ignore[assignment,misc]
    CauseOfLossCorrelation = None  # type: ignore[assignment,misc]

try:
    from auto_actuary.analytics.retention import RetentionAnalysis
except ImportError:
    RetentionAnalysis = None  # type: ignore[assignment,misc]

__all__ = [
    "TimeSeriesManager",
    "SnapshotStore",
    "MarketBreakdownConfig",
    "MarketBreakdownAnalysis",
    "ProductMixAnalysis",
    "CauseOfLossAnalysis",
    "CauseOfLossCorrelation",
    "RetentionAnalysis",
]
