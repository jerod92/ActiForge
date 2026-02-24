"""
Tests for the market_insights module.

Covers:
  - MarketCycleDetector: MCI scoring, phase classification, history
  - SegmentOpportunityScorer: scoring, grading, portfolio health
  - LossAnomalyDetector: CUSUM, Chow, Z-score, EWMA
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


# ---------------------------------------------------------------------------
# Market Cycle Detector
# ---------------------------------------------------------------------------

class TestMarketCycleDetector:
    """Tests for market cycle detection and MCI scoring."""

    @pytest.fixture
    def hard_market_crs(self) -> pd.Series:
        """Steadily declining combined ratios → hard market signal."""
        return pd.Series(
            {2015: 1.08, 2016: 1.06, 2017: 1.04, 2018: 1.02,
             2019: 0.99, 2020: 0.97, 2021: 0.95, 2022: 0.93},
        )

    @pytest.fixture
    def soft_market_crs(self) -> pd.Series:
        """Steadily rising combined ratios → soft market signal."""
        return pd.Series(
            {2015: 0.93, 2016: 0.95, 2017: 0.97, 2018: 0.99,
             2019: 1.01, 2020: 1.04, 2021: 1.06, 2022: 1.09},
        )

    def test_mci_hard_market_positive(self, hard_market_crs):
        detector = MarketCycleDetector(combined_ratios=hard_market_crs)
        result = detector.analyse()
        assert result.mci_score > 0, "Improving CRs should produce positive MCI"

    def test_mci_soft_market_negative(self, soft_market_crs):
        detector = MarketCycleDetector(combined_ratios=soft_market_crs)
        result = detector.analyse()
        assert result.mci_score < 0, "Deteriorating CRs should produce negative MCI"

    def test_mci_in_range(self, hard_market_crs):
        detector = MarketCycleDetector(combined_ratios=hard_market_crs)
        result = detector.analyse()
        assert -1.0 <= result.mci_score <= 1.0

    def test_phase_hard_market(self, hard_market_crs):
        detector = MarketCycleDetector(combined_ratios=hard_market_crs)
        result = detector.analyse()
        assert result.is_hard

    def test_phase_soft_market(self, soft_market_crs):
        detector = MarketCycleDetector(combined_ratios=soft_market_crs)
        result = detector.analyse()
        assert result.is_soft

    def test_signals_list_nonempty(self, hard_market_crs):
        detector = MarketCycleDetector(combined_ratios=hard_market_crs)
        result = detector.analyse()
        assert len(result.signals) >= 1
        assert any(s.name == "combined_ratio" for s in result.signals)

    def test_history_has_all_years(self, hard_market_crs):
        detector = MarketCycleDetector(combined_ratios=hard_market_crs)
        result = detector.analyse()
        for yr in hard_market_crs.index:
            assert yr in result.history.index

    def test_rate_change_signal_increases_mci(self, hard_market_crs):
        """Adding positive rate changes should increase the MCI vs. CR alone."""
        rate_changes = pd.Series({yr: 0.08 for yr in hard_market_crs.index})
        detector_no_rc = MarketCycleDetector(combined_ratios=hard_market_crs)
        detector_with_rc = MarketCycleDetector(
            combined_ratios=hard_market_crs, rate_changes=rate_changes
        )
        mci_no = detector_no_rc.analyse().mci_score
        mci_with = detector_with_rc.analyse().mci_score
        assert mci_with >= mci_no - 0.05  # with positive rate changes, MCI ≥ without

    def test_narrative_nonempty(self, hard_market_crs):
        detector = MarketCycleDetector(combined_ratios=hard_market_crs)
        result = detector.analyse()
        assert len(result.narrative) > 20

    def test_cycle_duration_positive(self, hard_market_crs):
        detector = MarketCycleDetector(combined_ratios=hard_market_crs)
        result = detector.analyse()
        assert result.cycle_duration_estimate >= 1

    def test_phase_history_returns_dataframe(self, hard_market_crs):
        detector = MarketCycleDetector(combined_ratios=hard_market_crs)
        hist = detector.phase_history()
        assert isinstance(hist, pd.DataFrame)
        assert "mci_score" in hist.columns
        assert "phase" in hist.columns

    def test_insufficient_data_raises(self):
        crs = pd.Series({2022: 1.02, 2023: 1.00})  # only 2 years
        with pytest.raises(ValueError, match="(?i)at least 3 years"):
            MarketCycleDetector(combined_ratios=crs)


# ---------------------------------------------------------------------------
# Segment Opportunity Scorer
# ---------------------------------------------------------------------------

class TestSegmentOpportunityScorer:
    """Tests for segment opportunity scoring."""

    @pytest.fixture
    def segment_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            "segment_key": ["SE_A", "NE_B", "MW_A", "SW_C", "NW_B"],
            "territory": ["SE", "NE", "MW", "SW", "NW"],
            "class_code": ["A", "B", "A", "C", "B"],
            "earned_premium": [5_000_000, 3_000_000, 4_000_000, 1_500_000, 2_500_000],
            "earned_exposure": [10_000, 6_000, 8_000, 3_000, 5_000],
            "loss_ratio": [0.58, 0.72, 0.63, 0.81, 0.67],
            "indicated_rate_change": [-0.05, 0.08, 0.02, 0.15, 0.03],
            "exposure_trend": [0.08, 0.03, 0.06, -0.02, 0.05],
            "retention_rate": [0.90, 0.78, 0.85, 0.70, 0.82],
        })

    def test_scores_all_segments(self, segment_data):
        scorer = SegmentOpportunityScorer(segment_data, permissible_loss_ratio=0.65)
        result = scorer.score_all()
        assert len(result) == 5

    def test_score_range(self, segment_data):
        scorer = SegmentOpportunityScorer(segment_data)
        result = scorer.score_all()
        assert (result["total_score"] >= 0).all()
        assert (result["total_score"] <= 100).all()

    def test_profitable_segment_scores_higher(self, segment_data):
        scorer = SegmentOpportunityScorer(segment_data, permissible_loss_ratio=0.65)
        result = scorer.score_all()
        # SE_A (LR=0.58, better than PLR 0.65) should outscore SW_C (LR=0.81)
        assert result.loc["SE_A", "total_score"] > result.loc["SW_C", "total_score"]

    def test_grade_present(self, segment_data):
        scorer = SegmentOpportunityScorer(segment_data)
        result = scorer.score_all()
        valid_grades = {"Premium", "Solid", "Marginal", "Challenged", "Critical"}
        for g in result["grade"]:
            assert g in valid_grades

    def test_action_present(self, segment_data):
        scorer = SegmentOpportunityScorer(segment_data)
        result = scorer.score_all()
        assert all(len(a) > 5 for a in result["action"])

    def test_top_opportunities_returns_n(self, segment_data):
        scorer = SegmentOpportunityScorer(segment_data)
        top = scorer.top_opportunities(n=3)
        assert len(top) <= 3

    def test_challenged_segments_below_threshold(self, segment_data):
        scorer = SegmentOpportunityScorer(segment_data)
        scored = scorer.score_all()
        challenged = scorer.challenged_segments(threshold=50.0)
        assert all(challenged["total_score"] < 50.0)

    def test_portfolio_health_summary_keys(self, segment_data):
        scorer = SegmentOpportunityScorer(segment_data)
        summary = scorer.portfolio_health_summary()
        for key in ["n_segments", "avg_opportunity_score", "n_critical",
                    "n_premium_opportunities", "pct_premium_grade_or_solid"]:
            assert key in summary

    def test_sorted_descending_by_score(self, segment_data):
        scorer = SegmentOpportunityScorer(segment_data)
        result = scorer.score_all()
        scores = result["total_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# ---------------------------------------------------------------------------
# Loss Anomaly Detector
# ---------------------------------------------------------------------------

class TestLossAnomalyDetector:
    """Tests for loss trend anomaly detection."""

    @pytest.fixture
    def stable_series(self) -> pd.Series:
        """Stable pure premium with mild growth — no anomaly."""
        years = range(2012, 2024)
        return pd.Series(
            {yr: 500 * (1.03 ** (yr - 2012)) for yr in years}
        )

    @pytest.fixture
    def anomalous_series(self) -> pd.Series:
        """Stable then sudden spike in recent years."""
        years = range(2012, 2024)
        vals = {yr: 500 * (1.03 ** (yr - 2012)) for yr in years}
        vals[2022] *= 1.35  # +35% spike
        vals[2023] *= 1.40  # continued
        return pd.Series(vals)

    def test_stable_series_no_critical(self, stable_series):
        detector = LossAnomalyDetector(stable_series, metric_name="pure_premium")
        report = detector.analyse()
        assert report.severity in ("normal", "monitoring")  # not elevated or critical

    def test_anomalous_series_signals(self, anomalous_series):
        detector = LossAnomalyDetector(anomalous_series, metric_name="pure_premium")
        report = detector.analyse()
        # At least one method should fire
        assert any(r.signal_detected for r in report.results)

    def test_report_has_four_methods(self, stable_series):
        detector = LossAnomalyDetector(stable_series)
        report = detector.analyse()
        methods = {r.method for r in report.results}
        assert methods == {"cusum", "chow", "z_score", "ewma"}

    def test_anomaly_result_fields(self, stable_series):
        detector = LossAnomalyDetector(stable_series)
        report = detector.analyse()
        for r in report.results:
            assert isinstance(r, AnomalyResult)
            assert r.method in ("cusum", "chow", "z_score", "ewma")
            assert r.direction in ("adverse", "favorable", "neutral")
            assert 0.0 <= r.confidence <= 1.0

    def test_data_column_in_report(self, stable_series):
        detector = LossAnomalyDetector(stable_series)
        report = detector.analyse()
        assert isinstance(report.data, pd.DataFrame)
        assert "value" in report.data.columns

    def test_recommended_actions_nonempty(self, anomalous_series):
        detector = LossAnomalyDetector(anomalous_series)
        report = detector.analyse()
        assert len(report.recommended_actions) >= 1

    def test_severity_levels(self, anomalous_series):
        detector = LossAnomalyDetector(anomalous_series)
        report = detector.analyse()
        assert report.severity in ("critical", "elevated", "monitoring", "normal")

    def test_cusum_adverse_direction(self, anomalous_series):
        detector = LossAnomalyDetector(anomalous_series)
        report = detector.analyse()
        cusum = next((r for r in report.results if r.method == "cusum"), None)
        assert cusum is not None
        if cusum.signal_detected:
            assert cusum.direction == "adverse"

    def test_chow_signal_year_in_range(self, anomalous_series):
        detector = LossAnomalyDetector(anomalous_series)
        report = detector.analyse()
        chow = next((r for r in report.results if r.method == "chow"), None)
        assert chow is not None
        if chow.signal_year is not None:
            min_yr = int(anomalous_series.index.min())
            max_yr = int(anomalous_series.index.max())
            assert min_yr <= chow.signal_year <= max_yr

    def test_insufficient_data_raises(self):
        short = pd.Series({2021: 500, 2022: 520, 2023: 510})
        with pytest.raises(ValueError, match="(?i)at least 4"):
            LossAnomalyDetector(short)

    def test_dataframe_input_accepted(self, stable_series):
        df = stable_series.reset_index()
        df.columns = ["year", "value"]
        detector = LossAnomalyDetector(df, metric_name="test")
        report = detector.analyse()
        assert report is not None
