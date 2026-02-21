"""
auto_actuary.analytics.speculative
====================================
Speculative business scenario analysis for executive decision support.

Modules
-------
categorical
    Robust encoding for high-cardinality / sparse categorical variables.
    SparseCollapser, CredibilityEncoder, ActuarialCategoricalEncoder.

glm_models
    GLM-based frequency (Poisson) and severity (Gamma) models.
    FrequencyGLM, SeverityGLM, CompoundGLM, fit_compound_glm_from_segments.

scenario_engine
    What-if scenario simulation across 8 business decision types.
    ScenarioEngine, ScenarioParams, ScenarioResult + component param classes.
    Pre-built helpers: rate_action_scenario, frequency_stress_scenario, etc.

trend_projector
    Forward trend projection with bootstrap prediction intervals.
    TrendProjector, RegimeChangeResult, build_trend_projectors.

Quick start
-----------
>>> from auto_actuary.analytics.speculative import ScenarioEngine, ScenarioParams
>>> from auto_actuary.analytics.speculative import rate_action_scenario

>>> engine = ScenarioEngine(segment_df, expense_ratio=0.30)
>>> scenario = rate_action_scenario("Rate +10% SE", "territory", {"SOUTHEAST": 0.10})
>>> result = engine.run_scenario(scenario)
>>> print(result.summary_table())
"""

from auto_actuary.analytics.speculative.categorical import (
    ActuarialCategoricalEncoder,
    CredibilityEncoder,
    SparseCollapser,
)
from auto_actuary.analytics.speculative.glm_models import (
    CompoundGLM,
    FrequencyGLM,
    GLMResult,
    PredictionInterval,
    SeverityGLM,
    fit_compound_glm_from_segments,
)
from auto_actuary.analytics.speculative.scenario_engine import (
    CatEnvironmentParams,
    EnterMarketParams,
    ExpenseInitiativeParams,
    ExitSegmentParams,
    FrequencyTrendParams,
    MixShiftParams,
    RateActionParams,
    ScenarioEngine,
    ScenarioParams,
    ScenarioResult,
    SeverityShockParams,
    cat_environment_scenario,
    frequency_stress_scenario,
    rate_action_scenario,
    severity_inflation_scenario,
)
from auto_actuary.analytics.speculative.trend_projector import (
    RegimeChangeResult,
    TrendProjector,
    build_trend_projectors,
)

__all__ = [
    # Categorical
    "ActuarialCategoricalEncoder",
    "CredibilityEncoder",
    "SparseCollapser",
    # GLM models
    "FrequencyGLM",
    "SeverityGLM",
    "CompoundGLM",
    "GLMResult",
    "PredictionInterval",
    "fit_compound_glm_from_segments",
    # Scenario engine
    "ScenarioEngine",
    "ScenarioParams",
    "ScenarioResult",
    "RateActionParams",
    "FrequencyTrendParams",
    "SeverityShockParams",
    "MixShiftParams",
    "ExitSegmentParams",
    "EnterMarketParams",
    "CatEnvironmentParams",
    "ExpenseInitiativeParams",
    "rate_action_scenario",
    "frequency_stress_scenario",
    "severity_inflation_scenario",
    "cat_environment_scenario",
    # Trend projector
    "TrendProjector",
    "RegimeChangeResult",
    "build_trend_projectors",
]
