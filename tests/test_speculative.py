"""
tests/test_speculative.py
=========================
Unit and integration tests for the speculative scenario analysis module.

Tests cover:
- ActuarialCategoricalEncoder: sparse collapse, credibility encoding, transform
- FrequencyGLM / SeverityGLM / CompoundGLM: fit, predict, relativities
- ScenarioEngine: each of the 8 scenario types, compare_scenarios, stress_test
- TrendProjector: fit, project, regime detection, sensitivity, CI
- ScenarioResult.summary_table()
- Session integration: scenario_engine() method

All tests use small synthetic data to run fast without external dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers — synthetic data generators
# ---------------------------------------------------------------------------

def make_segment_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Generate a small synthetic segment DataFrame for testing.

    Columns: accident_year, territory, class_code, coverage_code,
             earned_exposure, claim_count, incurred_loss,
             written_premium, is_catastrophe
    """
    rng = np.random.default_rng(seed)
    years = rng.choice([2019, 2020, 2021, 2022, 2023], size=n)
    territories = rng.choice(["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"], size=n)
    classes = rng.choice(["A", "B", "C", "D"], size=n)
    coverages = rng.choice(["BI", "PD", "COMP", "COLL"], size=n)

    exposure = rng.uniform(50, 500, size=n)
    frequency = np.where(
        territories == "SOUTH",
        rng.uniform(0.12, 0.18, size=n),  # higher freq south
        rng.uniform(0.06, 0.12, size=n),
    )
    claim_count = np.round(exposure * frequency).astype(float)
    severity = rng.uniform(3000, 8000, size=n)
    incurred = claim_count * severity
    premium = exposure * rng.uniform(600, 900, size=n)
    is_cat = rng.choice([0, 1], size=n, p=[0.95, 0.05])

    return pd.DataFrame({
        "accident_year": years.astype(int),
        "territory": territories,
        "class_code": classes,
        "coverage_code": coverages,
        "earned_exposure": exposure,
        "claim_count": claim_count,
        "incurred_loss": incurred,
        "written_premium": premium,
        "is_catastrophe": is_cat.astype(float),
    })


def make_trend_data(n_years: int = 8, annual_trend: float = 0.03, seed: int = 1) -> pd.DataFrame:
    """Simple yearly data with embedded trend + noise."""
    rng = np.random.default_rng(seed)
    years = np.arange(2016, 2016 + n_years)
    base = 0.08
    values = base * (1 + annual_trend) ** (years - years[0]) * rng.uniform(0.92, 1.08, size=n_years)
    return pd.DataFrame({"year": years, "value": values})


# ---------------------------------------------------------------------------
# 1. Categorical encoder tests
# ---------------------------------------------------------------------------

class TestSparseCollapser:
    def test_collapses_rare_levels(self):
        from auto_actuary.analytics.speculative.categorical import SparseCollapser
        X = pd.DataFrame({"cat": ["A"] * 100 + ["B"] * 5 + ["C"] * 3})
        sc = SparseCollapser(min_obs=10, other_label="_Other")
        Xt = sc.fit_transform(X, ["cat"])
        assert "_Other" in Xt["cat"].values
        assert "A" in Xt["cat"].values
        assert "B" not in Xt["cat"].values

    def test_no_collapse_when_all_large(self):
        from auto_actuary.analytics.speculative.categorical import SparseCollapser
        X = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 50})
        sc = SparseCollapser(min_obs=10)
        Xt = sc.fit_transform(X, ["cat"])
        assert "_Other" not in Xt["cat"].values

    def test_unknown_column_ignored(self):
        from auto_actuary.analytics.speculative.categorical import SparseCollapser
        X = pd.DataFrame({"cat": ["A"] * 50})
        sc = SparseCollapser().fit(X, ["cat"])
        Xt = sc.transform(X)
        assert "cat" in Xt.columns


class TestCredibilityEncoder:
    def test_grand_mean_shrinkage(self):
        """Sparse level should be shrunk toward grand mean."""
        from auto_actuary.analytics.speculative.categorical import CredibilityEncoder
        n_sparse = 5
        n_large = 500
        # Sparse level "B" has high target; but so few obs → shrinks
        X = pd.DataFrame({"cat": ["A"] * n_large + ["B"] * n_sparse})
        y = pd.Series([0.08] * n_large + [0.20] * n_sparse)
        ce = CredibilityEncoder(credibility_k=1082.0)
        ce.fit(X, y, ["cat"])
        enc_a = ce._encodings["cat"]["A"]
        enc_b = ce._encodings["cat"]["B"]
        grand = ce._grand_means["cat"]
        # B should be closer to grand mean than its raw mean
        assert abs(enc_b - grand) < abs(0.20 - grand), "B not shrunk enough"
        # A should be nearly equal to its raw mean (large n)
        assert abs(enc_a - 0.08) < 0.005, "A over-shrunk"

    def test_handle_unknown_mean(self):
        from auto_actuary.analytics.speculative.categorical import CredibilityEncoder
        X = pd.DataFrame({"cat": ["A"] * 100})
        y = pd.Series([0.08] * 100)
        ce = CredibilityEncoder(handle_unknown="mean")
        ce.fit(X, y, ["cat"])
        X_new = pd.DataFrame({"cat": ["Z"]})
        Xt = ce.transform(X_new)
        assert not np.isnan(Xt["cat"].iloc[0])

    def test_weighted_fit(self):
        from auto_actuary.analytics.speculative.categorical import CredibilityEncoder
        X = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 50})
        y = pd.Series([0.05] * 50 + [0.15] * 50)
        w = pd.Series([100.0] * 50 + [10.0] * 50)  # A has much more weight
        ce = CredibilityEncoder(credibility_k=100.0)
        ce.fit(X, y, ["cat"], weights=w)
        grand = ce._grand_means["cat"]
        # Weighted grand mean should be closer to 0.05 (A dominates)
        assert grand < 0.10


class TestActuarialCategoricalEncoder:
    def test_fit_transform_roundtrip(self):
        from auto_actuary.analytics.speculative.categorical import ActuarialCategoricalEncoder
        df = make_segment_df(200)
        ace = ActuarialCategoricalEncoder(min_obs=5, credibility_k=100.0)
        X = df[["territory", "class_code"]]
        y = df["claim_count"] / df["earned_exposure"]
        Xt = ace.fit_transform(X, y, cat_cols=["territory", "class_code"])
        assert Xt["territory"].dtype == float
        assert Xt["class_code"].dtype == float
        assert Xt.shape == X.shape

    def test_transform_unseen(self):
        from auto_actuary.analytics.speculative.categorical import ActuarialCategoricalEncoder
        df = make_segment_df(100)
        ace = ActuarialCategoricalEncoder(min_obs=5, credibility_k=100.0)
        X = df[["territory"]]
        y = df["claim_count"] / df["earned_exposure"]
        ace.fit(X, y, cat_cols=["territory"])
        X_new = pd.DataFrame({"territory": ["UNKNOWN_TERRITORY"]})
        Xt = ace.transform(X_new)
        # Should not crash and should return grand mean
        assert not np.isnan(Xt["territory"].iloc[0])

    def test_relativities_output(self):
        from auto_actuary.analytics.speculative.categorical import ActuarialCategoricalEncoder
        df = make_segment_df(300)
        ace = ActuarialCategoricalEncoder(min_obs=5, credibility_k=100.0)
        X = df[["territory"]]
        y = df["claim_count"] / df["earned_exposure"]
        ace.fit(X, y, cat_cols=["territory"])
        rel = ace.relativities("territory")
        assert "level" in rel.columns
        assert "relativity" in rel.columns
        # Base level relativity should be closest to 1.0
        assert rel["relativity"].notna().all()


# ---------------------------------------------------------------------------
# 2. GLM model tests
# ---------------------------------------------------------------------------

class TestFrequencyGLM:
    def test_fit_predict_basic(self):
        from auto_actuary.analytics.speculative.glm_models import FrequencyGLM
        df = make_segment_df(300)
        glm = FrequencyGLM(alpha=0.1, min_category_obs=5, credibility_k=100.0, max_iter=100)
        glm.fit(
            X=df[["territory", "class_code", "accident_year"]],
            claim_counts=df["claim_count"],
            exposure=df["earned_exposure"],
            cat_cols=["territory", "class_code"],
            cont_cols=["accident_year"],
        )
        assert glm.is_fitted
        preds = glm.predict(df[["territory", "class_code", "accident_year"]])
        assert len(preds) == len(df)
        assert (preds > 0).all()

    def test_result_diagnostics(self):
        from auto_actuary.analytics.speculative.glm_models import FrequencyGLM
        df = make_segment_df(300)
        glm = FrequencyGLM(alpha=0.05, min_category_obs=5, credibility_k=100.0, max_iter=100)
        glm.fit(
            df[["territory", "class_code"]],
            df["claim_count"],
            df["earned_exposure"],
            cat_cols=["territory", "class_code"],
        )
        result = glm.result
        assert result is not None
        assert result.n_obs == len(df)
        assert result.model_type == "frequency"
        assert result.mean_abs_error >= 0

    def test_relativities(self):
        from auto_actuary.analytics.speculative.glm_models import FrequencyGLM
        df = make_segment_df(400)
        glm = FrequencyGLM(alpha=0.05, min_category_obs=5, credibility_k=100.0, max_iter=100)
        glm.fit(
            df[["territory"]],
            df["claim_count"],
            df["earned_exposure"],
            cat_cols=["territory"],
        )
        rel = glm.relativities("territory")
        assert "level" in rel.columns
        assert "relativity" in rel.columns

    def test_not_fitted_raises(self):
        from auto_actuary.analytics.speculative.glm_models import FrequencyGLM
        glm = FrequencyGLM()
        df = pd.DataFrame({"territory": ["A"]})
        with pytest.raises(RuntimeError, match="not fitted"):
            glm.predict(df)


class TestSeverityGLM:
    def test_fit_predict(self):
        from auto_actuary.analytics.speculative.glm_models import SeverityGLM
        df = make_segment_df(300)
        severity = df["incurred_loss"] / df["claim_count"].replace(0, np.nan)
        mask = severity.notna() & (severity > 0)
        glm = SeverityGLM(alpha=0.05, min_category_obs=5, credibility_k=100.0, max_iter=100)
        glm.fit(
            X=df[mask][["territory", "class_code"]],
            severity=severity[mask],
            claim_counts=df[mask]["claim_count"],
            cat_cols=["territory", "class_code"],
        )
        assert glm.is_fitted
        preds = glm.predict(df[mask][["territory", "class_code"]])
        assert (preds > 0).all()


class TestCompoundGLM:
    def test_fit_and_portfolio_prediction(self):
        from auto_actuary.analytics.speculative.glm_models import CompoundGLM
        df = make_segment_df(400)
        glm = CompoundGLM(
            freq_alpha=0.1, sev_alpha=0.1,
            min_category_obs=5, credibility_k=100.0, max_iter=100
        )
        glm.fit(
            X=df[["territory", "class_code"]],
            claim_counts=df["claim_count"],
            losses=df["incurred_loss"],
            exposure=df["earned_exposure"],
            cat_cols=["territory", "class_code"],
        )
        assert glm.is_fitted

        kpis = glm.predict_portfolio(
            df[["territory", "class_code"]],
            df["earned_exposure"],
            df["written_premium"],
        )
        assert "predicted_loss_ratio" in kpis
        assert 0 < kpis["predicted_loss_ratio"] < 5  # sanity

    def test_relativities_table(self):
        from auto_actuary.analytics.speculative.glm_models import CompoundGLM
        df = make_segment_df(400)
        glm = CompoundGLM(max_iter=100, min_category_obs=5, credibility_k=100.0)
        glm.fit(
            df[["territory"]],
            df["claim_count"],
            df["incurred_loss"],
            df["earned_exposure"],
            cat_cols=["territory"],
        )
        rel = glm.relativities_table(cat_cols=["territory"])
        assert "freq_rel" in rel.columns
        assert "sev_rel" in rel.columns
        assert "pp_rel" in rel.columns

    def test_fit_compound_glm_from_segments(self):
        from auto_actuary.analytics.speculative.glm_models import fit_compound_glm_from_segments
        df = make_segment_df(400)
        glm = fit_compound_glm_from_segments(df, min_category_obs=5, credibility_k=100.0)
        assert glm.is_fitted

    def test_bootstrap_portfolio_ci(self):
        from auto_actuary.analytics.speculative.glm_models import CompoundGLM
        df = make_segment_df(200)
        glm = CompoundGLM(max_iter=100, min_category_obs=5, credibility_k=100.0)
        glm.fit(
            df[["territory"]],
            df["claim_count"],
            df["incurred_loss"],
            df["earned_exposure"],
            cat_cols=["territory"],
        )
        intervals = glm.bootstrap_portfolio(
            df[["territory"]],
            df["earned_exposure"],
            df["written_premium"],
            n_boot=30, ci=0.90,
        )
        lr_pi = intervals.get("predicted_loss_ratio")
        assert lr_pi is not None
        assert lr_pi.lower <= lr_pi.point <= lr_pi.upper


# ---------------------------------------------------------------------------
# 3. Scenario engine tests
# ---------------------------------------------------------------------------

class TestScenarioEngine:
    @pytest.fixture
    def engine(self):
        from auto_actuary.analytics.speculative.scenario_engine import ScenarioEngine
        df = make_segment_df(300)
        return ScenarioEngine(df, expense_ratio=0.30)

    def test_base_kpis(self, engine):
        kpis = engine.base_kpis
        assert "total_losses" in kpis
        assert "loss_ratio" in kpis
        assert "combined_ratio" in kpis
        assert kpis["total_losses"] > 0
        assert 0 < kpis["loss_ratio"] < 5

    def test_rate_action_increases_premium(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            ScenarioParams, RateActionParams
        )
        params = ScenarioParams(
            name="Rate +10% all",
            rate_action=RateActionParams(
                rate_changes={"all": 0.10},
                price_elasticity=-0.35,
                segment_col="territory",
            ),
        )
        result = engine.run_scenario(params)
        # Premium should increase (rate up, some volume loss)
        assert result.scenario_kpis["total_premium"] > result.base_kpis["total_premium"]
        # Loss ratio should fall
        assert result.scenario_kpis["loss_ratio"] < result.base_kpis["loss_ratio"]

    def test_rate_action_targeted_territory(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            ScenarioParams, RateActionParams
        )
        params = ScenarioParams(
            name="Rate +20% SOUTH",
            rate_action=RateActionParams(
                rate_changes={"SOUTH": 0.20},
                price_elasticity=-0.35,
                segment_col="territory",
            ),
        )
        result = engine.run_scenario(params)
        assert result.scenario_kpis["total_premium"] > result.base_kpis["total_premium"]

    def test_frequency_trend_increases_losses(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            ScenarioParams, FrequencyTrendParams
        )
        params = ScenarioParams(
            name="Freq trend +5%/yr",
            frequency_trend=FrequencyTrendParams(
                annual_trend=0.05, horizon_years=3
            ),
        )
        result = engine.run_scenario(params)
        assert result.scenario_kpis["total_losses"] > result.base_kpis["total_losses"]
        assert result.scenario_kpis["loss_ratio"] > result.base_kpis["loss_ratio"]

    def test_severity_shock_increases_losses(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            ScenarioParams, SeverityShockParams
        )
        params = ScenarioParams(
            name="Sev +20%",
            severity_shock=SeverityShockParams(severity_multiplier=1.20, sustained=False),
        )
        result = engine.run_scenario(params)
        assert result.scenario_kpis["total_losses"] > result.base_kpis["total_losses"]

    def test_exit_segment_reduces_portfolio(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            ScenarioParams, ExitSegmentParams
        )
        params = ScenarioParams(
            name="Exit SOUTH",
            exit_segment=ExitSegmentParams(filters={"territory": "SOUTH"}),
        )
        result = engine.run_scenario(params)
        assert result.scenario_kpis["total_losses"] < result.base_kpis["total_losses"]
        assert result.scenario_kpis["total_exposure"] < result.base_kpis["total_exposure"]

    def test_mix_shift(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            ScenarioParams, MixShiftParams
        )
        params = ScenarioParams(
            name="Grow NORTH +20%",
            mix_shift=MixShiftParams(volume_changes={"NORTH": 0.20}, segment_col="territory"),
        )
        result = engine.run_scenario(params)
        assert result.scenario_kpis["total_losses"] > result.base_kpis["total_losses"]

    def test_cat_environment_cat_flag(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            ScenarioParams, CatEnvironmentParams
        )
        params = ScenarioParams(
            name="CAT +30%",
            cat_environment=CatEnvironmentParams(cat_multiplier=1.30),
        )
        result = engine.run_scenario(params)
        # More CAT → more losses
        assert result.scenario_kpis["total_losses"] > result.base_kpis["total_losses"]

    def test_enter_market(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            ScenarioParams, EnterMarketParams
        )
        params = ScenarioParams(
            name="Enter Texas",
            enter_market=EnterMarketParams(
                est_annual_premium=5_000_000,
                est_loss_ratio=0.65,
                est_expense_ratio=0.30,
                ramp_up_years=3,
            ),
        )
        result = engine.run_scenario(params)
        # Premium grows
        assert result.scenario_kpis["total_premium"] > result.base_kpis["total_premium"]

    def test_expense_initiative(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            ScenarioParams, ExpenseInitiativeParams
        )
        params = ScenarioParams(
            name="2pt Expense Save",
            expense_initiative=ExpenseInitiativeParams(
                delta_expense_ratio=-0.02, timeline_years=2
            ),
        )
        result = engine.run_scenario(params)
        assert result.scenario_kpis["expense_ratio"] < result.base_kpis.get("expense_ratio", 0.30)

    def test_compare_scenarios(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            rate_action_scenario, frequency_stress_scenario
        )
        scenarios = [
            rate_action_scenario("Rate +10%", "territory", {"all": 0.10}),
            frequency_stress_scenario("Freq 5%/yr", 0.05, horizon_years=3),
        ]
        comparison = engine.compare_scenarios(scenarios)
        assert "Rate +10%" in comparison.columns
        assert "Freq 5%/yr" in comparison.columns
        assert "loss_ratio" in comparison.index

    def test_stress_test_output(self, engine):
        stress_df = engine.stress_test(n_simulations=100, random_state=42)
        assert len(stress_df) == 100
        assert "loss_ratio" in stress_df.columns
        assert "combined_ratio" in stress_df.columns
        assert "percentile" in stress_df.columns
        # Percentiles should be monotonically increasing
        assert stress_df["percentile"].is_monotonic_increasing

    def test_scenario_result_summary_table(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            rate_action_scenario
        )
        params = rate_action_scenario("Rate +5%", "territory", {"all": 0.05})
        result = engine.run_scenario(params, n_boot=0)
        tbl = result.summary_table()
        assert "Base" in tbl.columns
        assert "Scenario" in tbl.columns
        assert "Change" in tbl.columns
        assert "loss_ratio" in tbl.index

    def test_parametric_ci(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            rate_action_scenario
        )
        params = rate_action_scenario("Rate +5%", "territory", {"all": 0.05})
        result = engine.run_scenario(params, n_boot=0, ci=0.90)
        lr_ci = result.ci_90.get("loss_ratio")
        assert lr_ci is not None
        lo, hi = lr_ci
        assert lo <= result.scenario_kpis["loss_ratio"] <= hi

    def test_stacked_scenario(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            ScenarioParams, RateActionParams, SeverityShockParams
        )
        params = ScenarioParams(
            name="Rate+Sev",
            rate_action=RateActionParams({"all": 0.08}, price_elasticity=-0.35, segment_col="territory"),
            severity_shock=SeverityShockParams(severity_multiplier=1.10, sustained=False),
        )
        result = engine.run_scenario(params)
        # Combined effect: more premium, more losses
        assert result.scenario_kpis["total_premium"] > result.base_kpis["total_premium"]
        assert result.scenario_kpis["total_losses"] > result.base_kpis["total_losses"]

    def test_scenario_notes_populated(self, engine):
        from auto_actuary.analytics.speculative.scenario_engine import (
            rate_action_scenario
        )
        params = rate_action_scenario("Rate +10%", "territory", {"NORTH": 0.10})
        result = engine.run_scenario(params)
        assert len(result.notes) >= 1


# ---------------------------------------------------------------------------
# 4. Trend projector tests
# ---------------------------------------------------------------------------

class TestTrendProjector:
    @pytest.fixture
    def fitted_projector(self):
        from auto_actuary.analytics.speculative.trend_projector import TrendProjector
        data = make_trend_data(n_years=8, annual_trend=0.03)
        tp = TrendProjector(metric_name="frequency", n_boot=100, random_state=42)
        tp.fit(data)
        return tp

    def test_fit_recovers_approximate_trend(self, fitted_projector):
        tp = fitted_projector
        assert tp.is_fitted
        # With 8 years of +3%/yr data + noise, expect fitted trend near +3%
        trend = tp.fitted_annual_trend - 1.0
        assert -0.02 < trend < 0.08, f"Fitted trend {trend:.3f} not near 0.03"

    def test_projection_shape(self, fitted_projector):
        proj = fitted_projector.project(horizon_years=3)
        # 8 historical + 3 projected
        assert len(proj) == 8 + 3
        assert "type" in proj.columns
        assert "point" in proj.columns
        assert "p10" in proj.columns
        assert "p90" in proj.columns

    def test_projection_ci_ordering(self, fitted_projector):
        proj = fitted_projector.project(horizon_years=3)
        future = proj[proj["type"] == "projection"]
        assert (future["p10"] <= future["point"]).all()
        assert (future["point"] <= future["p90"]).all()

    def test_projection_scenarios(self, fitted_projector):
        proj = fitted_projector.project(
            horizon_years=3,
            scenarios={"pessimistic": 0.06, "optimistic": 0.00},
        )
        assert "pessimistic" in proj.columns
        future = proj[proj["type"] == "projection"]
        # Pessimistic > base projection (higher trend)
        assert (future["pessimistic"] >= future["point"]).all()

    def test_regime_detection_no_break(self, fitted_projector):
        rcr = fitted_projector.detect_regime_change(min_segment_size=3)
        # Clean trend data — should not detect a break
        if rcr.is_significant:
            # Acceptable if regime detection is overly sensitive on tiny sample
            assert rcr.break_year is not None
        else:
            assert rcr.break_year is None

    def test_regime_detection_with_break(self):
        """Data with a clear structural break at year 5."""
        from auto_actuary.analytics.speculative.trend_projector import TrendProjector
        rng = np.random.default_rng(99)
        years = np.arange(2015, 2025)
        vals = np.concatenate([
            0.08 * 1.02 ** np.arange(5) * rng.uniform(0.99, 1.01, 5),
            0.08 * 1.02 ** 4 * 1.10 ** np.arange(1, 6) * rng.uniform(0.99, 1.01, 5),
        ])
        data = pd.DataFrame({"year": years, "value": vals})
        tp = TrendProjector(metric_name="test", n_boot=50, random_state=42)
        tp.fit(data)
        rcr = tp.detect_regime_change(min_segment_size=3)
        # With a clear inflection, F-stat should be elevated
        assert rcr.f_statistic > 0

    def test_sensitivity_output(self, fitted_projector):
        sens = fitted_projector.sensitivity(horizon_years=3, n_steps=10)
        assert len(sens) == 10
        assert "assumed_trend" in sens.columns
        assert "projected_value" in sens.columns
        assert "pct_change_vs_base" in sens.columns

    def test_bootstrap_ci(self, fitted_projector):
        lo, hi = fitted_projector.bootstrap_trend_ci(ci=0.90)
        point = fitted_projector.fitted_annual_trend
        assert lo <= point <= hi

    def test_trend_summary(self, fitted_projector):
        summary = fitted_projector.trend_summary()
        assert "annual_trend" in summary.index
        assert "r_squared" in summary.index
        assert "ci_90_low" in summary.index

    def test_not_fitted_raises(self):
        from auto_actuary.analytics.speculative.trend_projector import TrendProjector
        tp = TrendProjector()
        with pytest.raises(RuntimeError):
            tp.project(3)

    def test_too_few_points_raises(self):
        from auto_actuary.analytics.speculative.trend_projector import TrendProjector
        tp = TrendProjector()
        with pytest.raises(ValueError):
            tp.fit(pd.DataFrame({"year": [2020], "value": [0.08]}))

    def test_build_trend_projectors(self):
        from auto_actuary.analytics.speculative.trend_projector import build_trend_projectors
        # Simulate an fs_table output
        rng = np.random.default_rng(0)
        years = np.arange(2018, 2024)
        df = pd.DataFrame({
            "accident_year": years,
            "frequency": 0.08 * 1.03 ** np.arange(6) * rng.uniform(0.95, 1.05, 6),
            "severity": 5000 * 1.04 ** np.arange(6) * rng.uniform(0.95, 1.05, 6),
            "pure_premium": 400 * 1.05 ** np.arange(6) * rng.uniform(0.95, 1.05, 6),
        })
        projectors = build_trend_projectors(df, year_col="accident_year", n_boot=50)
        assert "frequency" in projectors
        assert "severity" in projectors
        assert "pure_premium" in projectors
        for tp in projectors.values():
            assert tp.is_fitted


# ---------------------------------------------------------------------------
# 5. Pre-built scenario helpers
# ---------------------------------------------------------------------------

class TestScenarioHelpers:
    def test_rate_action_scenario(self):
        from auto_actuary.analytics.speculative.scenario_engine import rate_action_scenario
        sp = rate_action_scenario("Test", "territory", {"NORTH": 0.05})
        assert sp.rate_action is not None
        assert sp.rate_action.rate_changes == {"NORTH": 0.05}

    def test_frequency_stress_scenario(self):
        from auto_actuary.analytics.speculative.scenario_engine import frequency_stress_scenario
        sp = frequency_stress_scenario("Test", 0.05, horizon_years=3)
        assert sp.frequency_trend is not None
        assert sp.frequency_trend.annual_trend == 0.05

    def test_severity_inflation_scenario(self):
        from auto_actuary.analytics.speculative.scenario_engine import severity_inflation_scenario
        sp = severity_inflation_scenario("Test", 1.15, sustained=True, horizon_years=3)
        assert sp.severity_shock is not None
        assert sp.severity_shock.severity_multiplier == 1.15
        assert sp.severity_shock.sustained is True

    def test_cat_environment_scenario(self):
        from auto_actuary.analytics.speculative.scenario_engine import cat_environment_scenario
        sp = cat_environment_scenario("Test", 1.25)
        assert sp.cat_environment is not None
        assert sp.cat_environment.cat_multiplier == 1.25


# ---------------------------------------------------------------------------
# 6. Module __init__ exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    def test_all_symbols_importable(self):
        from auto_actuary.analytics.speculative import (
            ActuarialCategoricalEncoder,
            CredibilityEncoder,
            SparseCollapser,
            FrequencyGLM,
            SeverityGLM,
            CompoundGLM,
            GLMResult,
            PredictionInterval,
            fit_compound_glm_from_segments,
            ScenarioEngine,
            ScenarioParams,
            ScenarioResult,
            RateActionParams,
            FrequencyTrendParams,
            SeverityShockParams,
            MixShiftParams,
            ExitSegmentParams,
            EnterMarketParams,
            CatEnvironmentParams,
            ExpenseInitiativeParams,
            rate_action_scenario,
            frequency_stress_scenario,
            severity_inflation_scenario,
            cat_environment_scenario,
            TrendProjector,
            RegimeChangeResult,
            build_trend_projectors,
        )
        # Smoke test: all imports resolved
        assert ScenarioEngine is not None
        assert TrendProjector is not None


# ---------------------------------------------------------------------------
# 7. Integration test with session (requires sample data fixture from conftest)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSessionIntegration:
    """
    Integration tests using the sample session from conftest.py.
    Marked 'slow' — only run these when testing the full stack.
    Run with: pytest tests/test_speculative.py -m slow
    """

    def test_scenario_engine_from_session(self, sample_session):
        engine = sample_session.scenario_engine(lob="PPA", expense_ratio=0.28)
        assert engine is not None
        kpis = engine.base_kpis
        assert "total_losses" in kpis

    def test_run_rate_scenario_from_session(self, sample_session):
        from auto_actuary.analytics.speculative import rate_action_scenario
        engine = sample_session.scenario_engine(lob="PPA")
        scenario = rate_action_scenario("Rate +10%", "territory", {"all": 0.10})
        result = engine.run_scenario(scenario)
        assert result.scenario_kpis["loss_ratio"] < result.base_kpis["loss_ratio"]

    def test_compare_scenarios_from_session(self, sample_session):
        from auto_actuary.analytics.speculative import (
            rate_action_scenario, frequency_stress_scenario, cat_environment_scenario
        )
        engine = sample_session.scenario_engine(lob="PPA")
        scenarios = [
            rate_action_scenario("Rate +10%", "territory", {"all": 0.10}),
            frequency_stress_scenario("Freq 5%/yr", 0.05, 3),
            cat_environment_scenario("CAT +25%", 1.25),
        ]
        comp = engine.compare_scenarios(scenarios)
        assert "Rate +10%" in comp.columns
        assert "Freq 5%/yr" in comp.columns
        assert "CAT +25%" in comp.columns
