"""
Tests for triangle development math.

These test FCAS-level correctness — LDF formulas, CDF chains, tail fitting.
"""

import numpy as np
import pytest
import pandas as pd

from auto_actuary.analytics.triangles.development import LossTriangle, LDFMethods
from auto_actuary.analytics.triangles import tail as tail_mod


class TestLDFMethods:
    """Test the LDF averaging math."""

    def test_volume_weighted_ldf(self):
        # Simple 2-year example: from_col=[1000, 1100], to_col=[1350, 1430]
        # VW LDF = (1350+1430)/(1000+1100) = 2780/2100 = 1.3238...
        fc = pd.Series([1000, 1100], index=[2019, 2020])
        tc = pd.Series([1350, 1430], index=[2019, 2020])
        ldf = LDFMethods.volume_weighted(fc, tc)
        assert abs(ldf - 2780 / 2100) < 1e-9

    def test_simple_average_ldf(self):
        # Individual LDFs: 1350/1000=1.35, 1430/1100=1.30
        # Simple avg = (1.35 + 1.30) / 2 = 1.325
        fc = pd.Series([1000, 1100], index=[2019, 2020])
        tc = pd.Series([1350, 1430], index=[2019, 2020])
        ldf = LDFMethods.simple_average(fc, tc)
        assert abs(ldf - (1.35 + 1430/1100) / 2) < 1e-9

    def test_volume_weighted_n_recent(self):
        # With n_recent=1, only use most recent period (2020 row)
        fc = pd.Series([1000, 1100], index=[2019, 2020])
        tc = pd.Series([1350, 1430], index=[2019, 2020])
        ldf = LDFMethods.volume_weighted(fc, tc, n_recent=1)
        expected = 1430 / 1100
        assert abs(ldf - expected) < 1e-9

    def test_medial_drops_extremes(self):
        # 4 individual LDFs: 1.0, 1.2, 1.4, 2.0 → drop 1.0 and 2.0 → avg(1.2, 1.4) = 1.3
        fc = pd.Series([100, 100, 100, 100])
        tc = pd.Series([100, 120, 140, 200])
        ldf = LDFMethods.medial_average(fc, tc, exclude=1)
        assert abs(ldf - 1.3) < 1e-9

    def test_nan_handling(self):
        # NaN in from_col should be excluded
        fc = pd.Series([1000, np.nan, 1200])
        tc = pd.Series([1350, 1430, 1600])
        ldf = LDFMethods.volume_weighted(fc, tc)
        expected = (1350 + 1600) / (1000 + 1200)
        assert abs(ldf - expected) < 1e-9

    def test_zero_from_excluded(self):
        # from_col=0 should not cause division by zero
        fc = pd.Series([1000, 0, 1200])
        tc = pd.Series([1350, 0, 1600])
        ldf = LDFMethods.simple_average(fc, tc)
        assert np.isfinite(ldf)


class TestLossTriangle:
    """Test the LossTriangle class."""

    def test_triangle_shape(self, small_triangle):
        assert small_triangle.n_origins == 5
        assert small_triangle.n_ages == 5

    def test_latest_diagonal(self, small_triangle):
        diag = small_triangle.latest_diagonal
        # Origin 2019 should be at age 60
        assert diag[2019] == pytest.approx(1_490_000)
        # Origin 2023 should be at age 12
        assert diag[2023] == pytest.approx(950_000)

    def test_compute_all_ldfs(self, small_triangle):
        tbl = small_triangle.compute_all_ldfs()
        # Should have one row per age step
        assert len(tbl) == 4  # 12-24, 24-36, 36-48, 48-60
        # All LDFs should be > 1 (losses are still developing)
        assert (tbl["vw_all"] > 1.0).all()

    def test_selected_ldfs(self, small_triangle):
        small_triangle.compute_all_ldfs()
        selected = small_triangle.select_ldfs(method="vw_all")
        assert len(selected) == 4
        assert all(v > 1.0 for v in selected.values)

    def test_ldf_override(self, small_triangle):
        small_triangle.compute_all_ldfs()
        selected = small_triangle.select_ldfs(method="vw_all", overrides={"12-24": 1.500})
        assert selected["12-24"] == pytest.approx(1.500)

    def test_tail_factor_user_specified(self, small_triangle):
        small_triangle.develop(tail_method="user_specified", user_tail=1.025)
        assert small_triangle._tail_factor == pytest.approx(1.025)

    def test_cdfs_chain(self, developed_triangle):
        cdfs = developed_triangle._cdfs
        assert cdfs is not None
        # CDF at earliest age must be >= CDF at latest age
        ages = developed_triangle.ages
        assert cdfs[ages[0]] >= cdfs[ages[-1]]

    def test_ultimates_chain_ladder(self, developed_triangle):
        ults = developed_triangle.ultimates()
        # All ultimates >= reported (cumulative development)
        diag = developed_triangle.latest_diagonal
        for origin in developed_triangle.origins:
            assert ults[origin] >= diag[origin] - 1  # small floating point tolerance

    def test_ibnr_nonnegative(self, developed_triangle):
        ibnr = developed_triangle.ibnr()
        # For standard development patterns, IBNR should be >= 0
        assert (ibnr >= -1).all()  # -1 tolerance for float

    def test_summary_structure(self, developed_triangle):
        summ = developed_triangle.summary()
        expected_cols = {"latest_age", "reported", "cdf_to_ult", "ultimate", "ibnr", "pct_unreported"}
        assert expected_cols.issubset(set(summ.columns))
        assert len(summ) == 5

    def test_to_incremental(self, small_triangle):
        inc = small_triangle.to_incremental()
        assert inc.is_cumulative == False
        # First column of incremental should equal first column of cumulative
        assert (inc.triangle.iloc[:, 0] == small_triangle.triangle.iloc[:, 0]).all()


class TestTailFitting:
    """Test tail factor methods."""

    def test_inverse_power_monotone(self):
        # Ages 12, 24, 36, 48, 60; declining LDFs
        ages = np.array([18, 30, 42, 54])
        ldfs = np.array([1.45, 1.15, 1.06, 1.02])
        tail = tail_mod.fit_tail(ages, ldfs, curve="inverse_power")
        assert tail >= 1.0

    def test_tail_below_threshold_returns_one(self):
        # Very mature book — LDFs near 1.000 → tail should be 1.000
        ages = np.array([18, 30, 42, 54])
        ldfs = np.array([1.002, 1.001, 1.001, 1.000])
        tail = tail_mod.fit_tail(ages, ldfs, curve="inverse_power", threshold=1.005)
        assert tail == pytest.approx(1.0)

    def test_insufficient_data_returns_one(self):
        tail = tail_mod.fit_tail(np.array([18]), np.array([1.3]))
        assert tail == pytest.approx(1.0)

    def test_benchmark_tail(self):
        # At age 60 (5 years), should be around 1.060 per defaults
        tail = tail_mod.benchmark_tail(60)
        assert tail == pytest.approx(1.060)

    def test_benchmark_tail_mature(self):
        # At age 144 (12 years), should be 1.000
        tail = tail_mod.benchmark_tail(144)
        assert tail == pytest.approx(1.000)
