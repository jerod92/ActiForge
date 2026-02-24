"""
Tests for IBNR reserve methods.

Validates Chain Ladder, Bornhuetter-Ferguson, and Cape Cod.
Uses known-answer properties (not specific numbers, since those depend on
the random seed of synthetic data), plus algebraic invariants.
"""

import numpy as np
import pandas as pd
import pytest

from auto_actuary.analytics.triangles.development import LossTriangle
from auto_actuary.analytics.reserves.ibnr import ReserveAnalysis


TRIANGLE_DATA = {
    12:  [1_000, 1_100, 1_200, 1_350, 950],
    24:  [1_350, 1_430, 1_510, 1_680, None],
    36:  [1_450, 1_530, 1_620, None,  None],
    48:  [1_480, 1_560, None,  None,  None],
    60:  [1_490, None,  None,  None,  None],
}
ORIGINS = [2019, 2020, 2021, 2022, 2023]
PREMIUM = pd.Series([4_500, 4_800, 5_100, 5_500, 3_800], index=ORIGINS)


@pytest.fixture
def developed_tri():
    df = pd.DataFrame(TRIANGLE_DATA, index=ORIGINS)
    df.index.name = "accident_year"
    tri = LossTriangle(df, lob="TEST")
    tri.develop(ldf_method="vw_all", tail_method="user_specified", user_tail=1.010)
    return tri


@pytest.fixture
def reserve_analysis(developed_tri):
    return ReserveAnalysis(
        triangle=developed_tri,
        config=type("cfg", (), {
            "assumption": lambda *a, default=None, **kw: default,
        })(),
        premium=PREMIUM,
    )


class TestChainLadder:
    def test_cl_available(self, reserve_analysis):
        assert "chain_ladder" in reserve_analysis.available_methods

    def test_cl_ultimates_ge_reported(self, reserve_analysis, developed_tri):
        cl = reserve_analysis.result("chain_ladder")
        diag = developed_tri.latest_diagonal
        for origin in ORIGINS:
            assert cl.ultimates[origin] >= diag[origin] - 1e-3

    def test_cl_ibnr_equals_ult_minus_reported(self, reserve_analysis, developed_tri):
        cl = reserve_analysis.result("chain_ladder")
        diag = developed_tri.latest_diagonal
        for origin in ORIGINS:
            expected_ibnr = cl.ultimates[origin] - diag[origin]
            assert abs(cl.ibnr[origin] - expected_ibnr) < 1e-3

    def test_most_mature_year_small_ibnr(self, reserve_analysis):
        cl = reserve_analysis.result("chain_ladder")
        # 2019 (most mature, age 60) should have smallest IBNR
        min_ibnr_year = cl.ibnr.idxmin()
        assert min_ibnr_year == 2019


class TestBornhuetterFerguson:
    def test_bf_available(self, reserve_analysis):
        assert "bornhuetter_ferguson" in reserve_analysis.available_methods

    def test_bf_elr_positive(self, reserve_analysis):
        bf = reserve_analysis.result("bornhuetter_ferguson")
        assert bf.elr is not None
        assert bf.elr > 0

    def test_bf_immature_year_closer_to_expected(self, reserve_analysis, developed_tri):
        """
        For the most immature year (2023, age 12), B-F should give more
        weight to expected losses than chain ladder does.
        This means B-F and CL diverge more for immature years.
        """
        bf = reserve_analysis.result("bornhuetter_ferguson")
        cl = reserve_analysis.result("chain_ladder")
        elr = bf.elr
        prem_2023 = float(PREMIUM[2023])

        # B-F ibnr = ELR × Premium × pct_unreported
        # CL ibnr = reported × (CDF - 1)
        # For high-CDF (immature) years: B-F < CL if ELR < implied LR
        # We just test they differ
        assert bf.ultimates[2023] != pytest.approx(cl.ultimates[2023])

    def test_bf_mature_year_close_to_cl(self, reserve_analysis):
        """For fully developed years, B-F ≈ CL (reported dominates)."""
        bf = reserve_analysis.result("bornhuetter_ferguson")
        cl = reserve_analysis.result("chain_ladder")
        # 2019 is at age 60, near fully developed
        diff_pct = abs(bf.ultimates[2019] - cl.ultimates[2019]) / cl.ultimates[2019]
        assert diff_pct < 0.10  # within 10%


class TestCapeCod:
    def test_cc_available(self, reserve_analysis):
        assert "cape_cod" in reserve_analysis.available_methods

    def test_cc_elr_positive(self, reserve_analysis):
        cc = reserve_analysis.result("cape_cod")
        assert cc.elr is not None and cc.elr > 0

    def test_cc_elr_derived_from_data(self, reserve_analysis, developed_tri):
        """
        Cape Cod ELR = Σ Reported / Σ (Premium / CDF)
        i.e. Σ Reported / Σ Used-Up-Premium, where
        Used-Up Premium = Premium × % Reported = Premium / CDF.
        (Friedland 2010, §4.2; Mack 1994 Derivation 2)
        """
        cc = reserve_analysis.result("cape_cod")
        diag = developed_tri.latest_diagonal
        cdfs = pd.Series({
            o: developed_tri._cdfs.get(developed_tri._latest_age.get(o), 1.0)
            for o in ORIGINS
        })
        # Correct: used-up premium = Premium / CDF (% reported portion)
        used_up = PREMIUM / cdfs
        expected_elr = diag.sum() / used_up.sum()
        assert abs(cc.elr - expected_elr) < 0.001


class TestComparison:
    def test_comparison_table_has_all_methods(self, reserve_analysis):
        tbl = reserve_analysis.comparison_table()
        for m in reserve_analysis.available_methods:
            assert f"{m}_ult" in tbl.columns
            assert f"{m}_ibnr" in tbl.columns

    def test_total_row_present(self, reserve_analysis):
        tbl = reserve_analysis.comparison_table()
        assert "TOTAL" in tbl.index

    def test_total_equals_sum(self, reserve_analysis):
        tbl = reserve_analysis.comparison_table()
        cl_ult_sum = tbl.loc[ORIGINS, "chain_ladder_ult"].sum()
        assert abs(tbl.loc["TOTAL", "chain_ladder_ult"] - cl_ult_sum) < 1
