"""
Tests for ratemaking modules: trend, on-level, rate indication.
"""

import numpy as np
import pandas as pd
import pytest

from auto_actuary.analytics.ratemaking.trend import TrendAnalysis, TrendFit
from auto_actuary.analytics.ratemaking.indicated_rate import RateIndication


class TestTrendAnalysis:
    """Test log-linear trend fitting."""

    @pytest.fixture
    def clean_trend_data(self):
        """Perfect log-linear growth at 5%/yr."""
        years = list(range(2015, 2024))
        values = [1000 * (1.05 ** (y - 2015)) for y in years]
        return pd.DataFrame({"year": years, "value": values})

    def test_all_years_fit(self, clean_trend_data):
        ta = TrendAnalysis(clean_trend_data, periods=[3, 5])
        assert len(ta._fits) > 0

    def test_correct_trend_recovered(self, clean_trend_data):
        ta = TrendAnalysis(clean_trend_data, periods=[3, 5])
        fit = ta.select("all")
        assert fit is not None
        # Should recover ~5% annual trend
        assert abs(fit.annual_trend - 1.05) < 0.005

    def test_trend_factor_between(self, clean_trend_data):
        ta = TrendAnalysis(clean_trend_data)
        fit = ta.select("all")
        factor = ta.trend_factor_between(2020, 2023, fit=fit)
        expected = 1.05 ** 3
        assert abs(factor - expected) < 0.01

    def test_trend_table_has_expected_periods(self, clean_trend_data):
        ta = TrendAnalysis(clean_trend_data, periods=[3, 5])
        tbl = ta.trend_table()
        periods_found = set(tbl["period"].values)
        assert "all" in periods_found
        assert "3yr" in periods_found
        assert "5yr" in periods_found

    def test_r_squared_near_one_for_clean_data(self, clean_trend_data):
        ta = TrendAnalysis(clean_trend_data)
        fit = ta.select("all")
        assert fit.r_squared > 0.999

    def test_noisy_data_lower_r_squared(self):
        rng = np.random.default_rng(99)
        years = list(range(2015, 2024))
        values = [1000 * (1.05 ** (y - 2015)) * rng.lognormal(0, 0.2) for y in years]
        ta = TrendAnalysis(pd.DataFrame({"year": years, "value": values}))
        fit = ta.select("all")
        assert fit.r_squared < 0.99  # noisy → lower R²

    def test_period_selection_fallback(self, clean_trend_data):
        """If requested period not available, falls back to 'all'."""
        ta = TrendAnalysis(clean_trend_data, periods=[3])
        fit = ta.select("10yr")  # not computed
        assert fit is not None  # falls back, doesn't crash


class TestRateIndication:
    """Test rate indication calculation."""

    @pytest.fixture
    def simple_indication(self):
        years = [2019, 2020, 2021, 2022, 2023]
        olp = pd.Series([4_000_000, 4_200_000, 4_400_000, 4_600_000, 4_800_000], index=years)
        ult = pd.Series([2_600_000, 2_730_000, 2_860_000, 2_990_000, 3_120_000], index=years)
        return RateIndication(
            on_level_premium_by_year=olp,
            ultimate_loss_by_year=ult,
            lob="PPA",
            variable_expense_ratio=0.25,
            fixed_expense_ratio=0.05,
            target_profit_margin=0.05,
        )

    def test_permissible_lr(self, simple_indication):
        result = simple_indication.compute()
        # Permissible LR = 1 - 0.25 - 0.05 - 0.05 = 0.65
        assert result.permissible_loss_ratio == pytest.approx(0.65)

    def test_projected_loss_ratio(self, simple_indication):
        result = simple_indication.compute()
        total_ult = 2_600_000 + 2_730_000 + 2_860_000 + 2_990_000 + 3_120_000
        total_olp = 4_000_000 + 4_200_000 + 4_400_000 + 4_600_000 + 4_800_000
        expected_lr = total_ult / total_olp
        assert abs(result.projected_loss_ratio - expected_lr) < 1e-6

    def test_indicated_change_formula(self, simple_indication):
        result = simple_indication.compute()
        expected = result.projected_loss_ratio / result.permissible_loss_ratio - 1.0
        assert abs(result.indicated_change - expected) < 1e-9

    def test_trend_factor_applied(self):
        years = [2021, 2022, 2023]
        olp = pd.Series([1_000_000, 1_000_000, 1_000_000], index=years)
        ult = pd.Series([650_000, 650_000, 650_000], index=years)
        ind = RateIndication(
            on_level_premium_by_year=olp,
            ultimate_loss_by_year=ult,
            trend_factor=1.10,
            variable_expense_ratio=0.25,
            fixed_expense_ratio=0.05,
            target_profit_margin=0.05,
        )
        result = ind.compute()
        # Trended loss = 650k × 1.10 = 715k
        # Projected LR = 715k / 1000k = 0.715
        # Permissible = 0.65
        # Indicated = 0.715/0.65 - 1 ≈ +10%
        assert result.projected_loss_ratio == pytest.approx(715_000 / 1_000_000)
        assert result.indicated_change > 0

    def test_credibility_bounds(self, simple_indication):
        result = simple_indication.compute()
        assert 0.0 <= result.credibility <= 1.0

    def test_no_overlapping_years_raises(self):
        olp = pd.Series([1_000_000], index=[2020])
        ult = pd.Series([650_000], index=[2021])  # different year
        ind = RateIndication(olp, ult)
        with pytest.raises(ValueError, match="No overlapping"):
            ind.compute()

    def test_profitability_scenario_flat_book(self):
        """A book earning exactly the permissible LR should indicate 0% change."""
        years = [2020, 2021, 2022]
        perm = 0.65
        olp = pd.Series([1_000_000] * 3, index=years)
        ult = pd.Series([1_000_000 * perm] * 3, index=years)
        ind = RateIndication(
            olp, ult,
            variable_expense_ratio=0.25,
            fixed_expense_ratio=0.05,
            target_profit_margin=0.05,
        )
        result = ind.compute()
        assert abs(result.indicated_change) < 0.001
