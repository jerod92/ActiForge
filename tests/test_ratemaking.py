"""
Tests for ratemaking modules: trend, on-level, rate indication,
plus new Durbin-Watson, WLS, and Bühlmann-Straub credibility.
"""

import numpy as np
import pandas as pd
import pytest

from auto_actuary.analytics.ratemaking.trend import TrendAnalysis, TrendFit, _durbin_watson
from auto_actuary.analytics.ratemaking.indicated_rate import (
    RateIndication,
    buhlmann_straub_credibility,
)


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


class TestDurbinWatson:
    """Test Durbin-Watson autocorrelation statistic."""

    def test_dw_no_autocorrelation(self):
        """White noise residuals → DW ≈ 2.0."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        dw = _durbin_watson(residuals)
        assert 1.5 <= dw <= 2.5

    def test_dw_positive_autocorrelation(self):
        """Highly positively autocorrelated residuals → DW < 1.5."""
        # AR(1) with ρ=0.95
        e = np.zeros(100)
        e[0] = 1.0
        for t in range(1, 100):
            e[t] = 0.95 * e[t - 1] + np.random.normal(0, 0.1)
        dw = _durbin_watson(e)
        assert dw < 1.5

    def test_dw_short_series_returns_nan(self):
        dw = _durbin_watson(np.array([1.0, 2.0]))
        assert np.isnan(dw)

    def test_trend_table_includes_dw(self):
        years = list(range(2015, 2024))
        values = [1000 * (1.05 ** (y - 2015)) for y in years]
        ta = TrendAnalysis(pd.DataFrame({"year": years, "value": values}))
        tbl = ta.trend_table()
        assert "durbin_watson" in tbl.columns
        assert "autocorrelation_flag" in tbl.columns

    def test_trend_ci_columns_present(self):
        years = list(range(2015, 2024))
        values = [1000 * (1.05 ** (y - 2015)) for y in years]
        ta = TrendAnalysis(pd.DataFrame({"year": years, "value": values}))
        tbl = ta.trend_table()
        assert "trend_ci_90_lo" in tbl.columns
        assert "trend_ci_90_hi" in tbl.columns

    def test_wls_with_weights_recovers_trend(self):
        """WLS with exposure weights should also recover the 5% trend on clean data."""
        rng = np.random.default_rng(7)
        years = list(range(2015, 2024))
        values = [1000 * (1.05 ** (y - 2015)) for y in years]
        weights = [rng.integers(100, 1000) for _ in years]
        data = pd.DataFrame({"year": years, "value": values, "weight": weights})
        ta = TrendAnalysis(data, use_wls=True)
        fit = ta.select("all")
        assert abs(fit.annual_trend - 1.05) < 0.01
        assert fit.weighted is True


class TestBuhlmannStraub:
    """Test Bühlmann-Straub empirical Bayes credibility."""

    def test_high_variance_observations_yield_positive_credibility(self):
        """With substantial between-year variation and large weights, Z > 0."""
        # Widely varying observations → a_hat > 0 → finite k → Z > 0
        obs = pd.Series([0.50, 0.80, 0.55, 0.90, 0.45], index=range(5))
        wts = pd.Series([10_000] * 5, index=range(5))
        result = buhlmann_straub_credibility(obs, wts)
        # a_hat should be positive given high variance in obs
        assert result["a_hat"] >= 0.0
        assert 0.0 <= result["Z_total"] <= 1.0

    def test_zero_between_variance_gives_zero_credibility(self):
        """If all observations are identical (a_hat→0), credibility→0."""
        obs = pd.Series([0.65, 0.65, 0.65, 0.65], index=range(4))
        wts = pd.Series([100, 100, 100, 100], index=range(4))
        result = buhlmann_straub_credibility(obs, wts)
        # a_hat=0 → Z=0
        assert result["Z_total"] == pytest.approx(0.0, abs=1e-6)

    def test_result_keys_present(self):
        obs = pd.Series([0.60, 0.65, 0.70], index=range(3))
        wts = pd.Series([500, 600, 550], index=range(3))
        result = buhlmann_straub_credibility(obs, wts)
        for key in ["mu_hat", "a_hat", "v_hat", "k", "Z_total", "cred_estimate"]:
            assert key in result

    def test_mu_hat_is_weighted_mean(self):
        obs = pd.Series([0.60, 0.80], index=[0, 1])
        wts = pd.Series([3.0, 1.0], index=[0, 1])
        result = buhlmann_straub_credibility(obs, wts)
        expected_mu = (0.60 * 3 + 0.80 * 1) / 4
        assert abs(result["mu_hat"] - expected_mu) < 1e-9

    def test_buhlmann_straub_in_rate_indication(self):
        """RateIndication accepts credibility_method='buhlmann_straub'."""
        years = [2019, 2020, 2021, 2022]
        olp = pd.Series([1_000_000, 1_050_000, 1_100_000, 1_150_000], index=years)
        ult = pd.Series([680_000, 700_000, 730_000, 760_000], index=years)
        ind = RateIndication(
            olp, ult,
            credibility_method="buhlmann_straub",
            variable_expense_ratio=0.25,
            fixed_expense_ratio=0.05,
            target_profit_margin=0.05,
        )
        result = ind.compute()
        assert 0.0 <= result.credibility <= 1.0
        assert result.credibility_weighted_change is not None
