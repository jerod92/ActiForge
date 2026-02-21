"""
auto_actuary.core.session
=========================
The central ActuarySession ties config, loaded data, and analysis together.
This is the primary entry-point for interactive and programmatic use.

Usage
-----
>>> session = ActuarySession.from_config("config/schema.yaml")
>>> session.load_csv("policies",   "data/policies.csv")
>>> session.load_csv("valuations", "data/valuations.csv")
>>> session.load_csv("claims",     "data/claims.csv")

>>> # Triangle development
>>> tri = session.build_triangle(lob="PPA", value="incurred_loss")
>>> tri.develop()
>>> print(tri.summary())

>>> # Rate indication
>>> ind = session.rate_indication(lob="PPA")
>>> print(ind.indicated_change)

>>> # Executive dashboard
>>> session.exec_dashboard(output_path="output/dashboard.html")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from auto_actuary.core.config import ActuaryConfig
from auto_actuary.core.data_loader import DataLoader

logger = logging.getLogger(__name__)


class ActuarySession:
    """
    Top-level session object.

    Holds references to the config, data loader, and any computed results.
    All analytics modules are accessible through this object.

    Parameters
    ----------
    config : ActuaryConfig
    """

    def __init__(self, config: ActuaryConfig) -> None:
        self.config = config
        self.loader = DataLoader(config)
        self._results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        schema_path: str | Path = "config/schema.yaml",
        assumptions_path: Optional[str | Path] = None,
        lob_path: Optional[str | Path] = None,
    ) -> "ActuarySession":
        """Create a session from a schema.yaml path."""
        cfg = ActuaryConfig(
            schema_path=schema_path,
            assumptions_path=assumptions_path,
            lob_path=lob_path,
        )
        return cls(cfg)

    @classmethod
    def from_dir(cls, config_dir: str | Path = "config") -> "ActuarySession":
        """Create a session from a config directory."""
        return cls(ActuaryConfig.from_dir(config_dir))

    # ------------------------------------------------------------------
    # Data loading (delegates to DataLoader)
    # ------------------------------------------------------------------

    def load_csv(self, table: str, path: Union[str, Path], **kwargs: Any) -> "ActuarySession":
        """Load a CSV file into *table*. Returns self for chaining."""
        self.loader.load_csv(table, path, **kwargs)
        return self

    def load_dataframe(self, table: str, df: pd.DataFrame) -> "ActuarySession":
        """Register a pre-built DataFrame. Returns self for chaining."""
        self.loader.load_dataframe(table, df)
        return self

    def load_sql(self, table: str, sql: str, engine: Any) -> "ActuarySession":
        """Execute SQL and load result into *table*. Returns self for chaining."""
        self.loader.load_sql(table, sql, engine)
        return self

    def load_sql_file(self, table: str, sql_path: Union[str, Path], engine: Any) -> "ActuarySession":
        """Execute a .sql file and load result into *table*."""
        self.loader.load_sql_file(table, sql_path, engine)
        return self

    def data(self, table: str) -> pd.DataFrame:
        """Return the loaded DataFrame for *table*."""
        return self.loader[table]

    # ------------------------------------------------------------------
    # Triangle analytics
    # ------------------------------------------------------------------

    def build_triangle(
        self,
        lob: str,
        value: str = "incurred_loss",
        origin_basis: str = "accident_year",
        dev_step_months: int = 12,
        **kwargs: Any,
    ) -> "auto_actuary.analytics.triangles.development.LossTriangle":  # type: ignore[name-defined]
        """
        Build a loss development triangle from loaded valuations data.

        Parameters
        ----------
        lob : str
            Line of business code (e.g. 'PPA', 'HO').
        value : str
            Column to aggregate: incurred_loss | paid_loss | paid_count | open_count
        origin_basis : str
            accident_year | policy_year | report_year
        dev_step_months : int
            12 for annual triangles, 3 for quarterly.

        Returns
        -------
        LossTriangle
        """
        from auto_actuary.analytics.triangles.development import build_triangle_from_session

        return build_triangle_from_session(
            session=self,
            lob=lob,
            value=value,
            origin_basis=origin_basis,
            dev_step_months=dev_step_months,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Reserve analytics
    # ------------------------------------------------------------------

    def reserve_analysis(
        self,
        lob: str,
        value: str = "incurred_loss",
        methods: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "auto_actuary.analytics.reserves.ibnr.ReserveAnalysis":  # type: ignore[name-defined]
        """
        Run a full IBNR reserve analysis for *lob*.

        Builds the triangle, selects LDFs, and applies the configured reserve
        methods (chain ladder, B-F, Cape Cod).
        """
        from auto_actuary.analytics.reserves.ibnr import ReserveAnalysis

        tri = self.build_triangle(lob=lob, value=value, **kwargs)
        tri.develop()
        return ReserveAnalysis(triangle=tri, config=self.config, methods=methods)

    # ------------------------------------------------------------------
    # Ratemaking
    # ------------------------------------------------------------------

    def rate_indication(
        self,
        lob: str,
        accident_years: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> "auto_actuary.analytics.ratemaking.indicated_rate.RateIndication":  # type: ignore[name-defined]
        """
        Compute a rate indication for *lob*.

        Requires: policies, valuations, claims, rate_changes loaded.
        """
        from auto_actuary.analytics.ratemaking.indicated_rate import RateIndication

        return RateIndication.from_session(
            session=self,
            lob=lob,
            accident_years=accident_years,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Frequency / Severity
    # ------------------------------------------------------------------

    def freq_severity(
        self,
        lob: str,
        coverage: Optional[str] = None,
        **kwargs: Any,
    ) -> "auto_actuary.analytics.frequency_severity.analysis.FreqSevAnalysis":  # type: ignore[name-defined]
        """Run frequency/severity analysis for *lob* (optionally by coverage)."""
        from auto_actuary.analytics.frequency_severity.analysis import FreqSevAnalysis

        return FreqSevAnalysis.from_session(session=self, lob=lob, coverage=coverage, **kwargs)

    # ------------------------------------------------------------------
    # Profitability
    # ------------------------------------------------------------------

    def loss_ratios(
        self,
        lob: Optional[str] = None,
        by: Optional[list[str]] = None,
    ) -> "auto_actuary.analytics.profitability.loss_ratio.LossRatioReport":  # type: ignore[name-defined]
        """
        Compute loss ratios.

        Parameters
        ----------
        lob : str, optional
            Filter to single LOB.  None = all LOBs.
        by : list[str], optional
            Additional grouping dimensions: ['territory', 'coverage_code', 'class_code']
        """
        from auto_actuary.analytics.profitability.loss_ratio import LossRatioReport

        return LossRatioReport.from_session(session=self, lob=lob, by=by)

    def combined_ratio(
        self,
        lob: Optional[str] = None,
    ) -> "auto_actuary.analytics.profitability.combined_ratio.CombinedRatioReport":  # type: ignore[name-defined]
        """Compute combined ratio (loss + expense)."""
        from auto_actuary.analytics.profitability.combined_ratio import CombinedRatioReport

        return CombinedRatioReport.from_session(session=self, lob=lob)

    def cohort_analysis(
        self,
        lob: Optional[str] = None,
    ) -> "auto_actuary.analytics.profitability.cohort.CohortReport":  # type: ignore[name-defined]
        """Policy vintage / cohort profitability analysis."""
        from auto_actuary.analytics.profitability.cohort import CohortReport

        return CohortReport.from_session(session=self, lob=lob)

    # ------------------------------------------------------------------
    # Catastrophe
    # ------------------------------------------------------------------

    def cat_analysis(
        self,
        lob: Optional[str] = None,
    ) -> "auto_actuary.analytics.catastrophe.cat_analysis.CatAnalysis":  # type: ignore[name-defined]
        """Catastrophe vs. non-cat loss split and trend."""
        from auto_actuary.analytics.catastrophe.cat_analysis import CatAnalysis

        return CatAnalysis.from_session(session=self, lob=lob)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def exec_dashboard(
        self,
        output_path: Union[str, Path] = "output/dashboard.html",
        lob: Optional[str] = None,
    ) -> Path:
        """
        Generate and save an executive HTML dashboard.

        Returns the path to the saved file.
        """
        from auto_actuary.reports.executive.dashboard import ExecDashboard

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dash = ExecDashboard(session=self, lob=lob)
        return dash.render(output_path=output_path)

    def triangle_exhibit(
        self,
        lob: str,
        value: str = "incurred_loss",
        output_path: Optional[Union[str, Path]] = None,
        fmt: str = "excel",
    ) -> Path:
        """
        Generate a formatted triangle development exhibit.

        Parameters
        ----------
        fmt : str
            'excel' or 'html'
        """
        from auto_actuary.reports.actuarial.triangle_exhibit import TriangleExhibit

        if output_path is None:
            output_path = f"output/{lob}_triangle.{'xlsx' if fmt == 'excel' else 'html'}"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        tri = self.build_triangle(lob=lob, value=value)
        tri.develop()
        exhibit = TriangleExhibit(triangle=tri, config=self.config)
        return exhibit.render(output_path=output_path, fmt=fmt)

    def reserve_exhibit(
        self,
        lob: str,
        output_path: Optional[Union[str, Path]] = None,
        fmt: str = "excel",
    ) -> Path:
        """Generate a formatted reserve / IBNR exhibit."""
        from auto_actuary.reports.actuarial.reserve_exhibit import ReserveExhibit

        if output_path is None:
            output_path = f"output/{lob}_reserve.{'xlsx' if fmt == 'excel' else 'html'}"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        analysis = self.reserve_analysis(lob=lob)
        exhibit = ReserveExhibit(analysis=analysis, config=self.config)
        return exhibit.render(output_path=output_path, fmt=fmt)

    def rate_indication_exhibit(
        self,
        lob: str,
        output_path: Optional[Union[str, Path]] = None,
        fmt: str = "excel",
    ) -> Path:
        """Generate a rate indication exhibit."""
        from auto_actuary.reports.actuarial.rate_indication import RateIndicationExhibit

        if output_path is None:
            output_path = f"output/{lob}_rate_indication.{'xlsx' if fmt == 'excel' else 'html'}"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        ind = self.rate_indication(lob=lob)
        exhibit = RateIndicationExhibit(indication=ind, config=self.config)
        return exhibit.render(output_path=output_path, fmt=fmt)

    # ------------------------------------------------------------------
    # Speculative / scenario analysis
    # ------------------------------------------------------------------

    def build_segment_df(
        self,
        lob: Optional[str] = None,
        by: Optional[list[str]] = None,
        value: str = "incurred_loss",
    ) -> pd.DataFrame:
        """
        Build a segment-level DataFrame suitable for ScenarioEngine.

        Aggregates claims, valuations, and policies data into a flat table
        at (accident_year × territory × class_code × coverage_code) grain.

        Parameters
        ----------
        lob : str, optional
            Filter to a single line of business.
        by : list of str, optional
            Additional grouping dimensions beyond accident_year.
            Default: ['territory', 'class_code', 'coverage_code'].
        value : str
            Loss column to aggregate.  Default 'incurred_loss'.

        Returns
        -------
        DataFrame suitable for passing to ScenarioEngine and CompoundGLM.
        """
        by = by or ["territory", "class_code", "coverage_code"]
        group_cols = ["accident_year"] + [c for c in by if c != "accident_year"]

        # Claims + latest valuations
        claims = self.loader["claims"].copy() if "claims" in self.loader.loaded_tables else pd.DataFrame()
        vals = self.loader["valuations"].copy() if "valuations" in self.loader.loaded_tables else pd.DataFrame()
        policies = self.loader["policies"].copy() if "policies" in self.loader.loaded_tables else pd.DataFrame()

        if claims.empty or vals.empty:
            raise ValueError(
                "Cannot build segment_df without 'claims' and 'valuations' loaded. "
                "Call session.load_csv('claims', ...) and session.load_csv('valuations', ...) first."
            )

        # Filter by LOB
        if lob and "line_of_business" in claims.columns:
            claims = claims[claims["line_of_business"] == lob].copy()
        if lob and not policies.empty and "line_of_business" in policies.columns:
            policies = policies[policies["line_of_business"] == lob].copy()

        # Accident year
        if "accident_date" in claims.columns:
            claims["accident_year"] = claims["accident_date"].dt.year

        # Latest valuation per claim
        if "valuation_date" in vals.columns and "claim_id" in vals.columns:
            latest_vals = (
                vals.sort_values("valuation_date")
                .groupby("claim_id")
                .last()
                .reset_index()[["claim_id", value, "paid_loss"]]
            ) if value in vals.columns else pd.DataFrame()
            if not latest_vals.empty:
                claims = claims.merge(latest_vals, on="claim_id", how="left")

        claims[value] = claims.get(value, pd.Series(0, index=claims.index)).fillna(0)
        claims["claim_count"] = 1

        avail_group = [c for c in group_cols if c in claims.columns]
        agg_cols = {value: "sum", "claim_count": "sum"}
        if "is_catastrophe" in claims.columns:
            agg_cols["is_catastrophe"] = "max"

        loss_df = claims.groupby(avail_group).agg(**{
            k: pd.NamedAgg(column=col if col in claims.columns else k, aggfunc=fn)
            for k, (col, fn) in {
                "incurred_loss": (value, "sum"),
                "claim_count": ("claim_count", "sum"),
            }.items()
        }).reset_index()

        if "is_catastrophe" in claims.columns:
            cat_df = claims.groupby(avail_group)["is_catastrophe"].max().reset_index()
            loss_df = loss_df.merge(cat_df, on=avail_group, how="left")

        # Exposure and premium from policies
        if not policies.empty and "effective_date" in policies.columns:
            policies["accident_year"] = policies["effective_date"].dt.year
            exp_group = [c for c in avail_group if c in policies.columns]
            exp_agg = {}
            if "written_exposure" in policies.columns:
                exp_agg["earned_exposure"] = pd.NamedAgg("written_exposure", "sum")
            if "written_premium" in policies.columns:
                exp_agg["written_premium"] = pd.NamedAgg("written_premium", "sum")
            if exp_agg:
                exp_df = policies.groupby(exp_group).agg(**exp_agg).reset_index()
                loss_df = loss_df.merge(exp_df, on=[c for c in exp_group if c in loss_df.columns], how="left")

        loss_df["incurred_loss"] = loss_df.get("incurred_loss", pd.Series(0)).fillna(0)
        loss_df["claim_count"] = loss_df.get("claim_count", pd.Series(0)).fillna(0)

        return loss_df

    def scenario_engine(
        self,
        lob: Optional[str] = None,
        by: Optional[list[str]] = None,
        expense_ratio: float = 0.30,
        fit_glm: bool = False,
    ) -> "auto_actuary.analytics.speculative.scenario_engine.ScenarioEngine":  # type: ignore[name-defined]
        """
        Build a ScenarioEngine from loaded session data.

        Parameters
        ----------
        lob : str, optional
            Line of business to filter to.
        by : list of str, optional
            Segment grouping dimensions.
        expense_ratio : float
            Portfolio expense ratio for combined ratio calculations.
        fit_glm : bool
            If True, fit a CompoundGLM to the segment data and attach it
            to the engine for GLM-assisted scenario analysis.

        Returns
        -------
        ScenarioEngine

        Example
        -------
        >>> engine = session.scenario_engine(lob="PPA", expense_ratio=0.28)
        >>> scenario = rate_action_scenario("Rate +8% SE", "territory", {"SOUTHEAST": 0.08})
        >>> result = engine.run_scenario(scenario)
        >>> print(result.summary_table())
        """
        from auto_actuary.analytics.speculative.scenario_engine import ScenarioEngine
        from auto_actuary.analytics.speculative.glm_models import fit_compound_glm_from_segments

        seg_df = self.build_segment_df(lob=lob, by=by)

        glm = None
        if fit_glm:
            cat_cols = [c for c in ["territory", "class_code", "coverage_code"] if c in seg_df.columns]
            cont_cols = [c for c in ["accident_year"] if c in seg_df.columns]
            try:
                glm = fit_compound_glm_from_segments(
                    seg_df,
                    cat_cols=cat_cols,
                    cont_cols=cont_cols,
                )
                logger.info("ScenarioEngine: CompoundGLM fitted — %r", glm)
            except Exception as e:
                logger.warning("ScenarioEngine: GLM fitting failed (%s); continuing without GLM", e)

        return ScenarioEngine(segment_df=seg_df, glm=glm, expense_ratio=expense_ratio)

    def scenario_report(
        self,
        scenarios: list,
        output_path: Union[str, Path] = "output/scenario_analysis.html",
        lob: Optional[str] = None,
        expense_ratio: float = 0.30,
        horizon_years: int = 3,
        run_stress_test: bool = True,
        n_stress_sims: int = 500,
        n_boot_ci: int = 0,
    ) -> Path:
        """
        Run scenarios and render a self-contained HTML scenario report.

        Parameters
        ----------
        scenarios : list of ScenarioParams
            Scenarios to compare.
        output_path : str or Path
            Output HTML file path.
        lob : str, optional
            Line of business filter.
        expense_ratio : float
            Portfolio expense ratio for CR calculation.
        horizon_years : int
            Trend projection horizon.
        run_stress_test : bool
            Include a Monte Carlo stress test distribution.
        n_stress_sims : int
            Number of stress test simulations.
        n_boot_ci : int
            Bootstrap iterations for CI on scenario KPIs.
            0 = parametric CI approximation (fast).

        Returns
        -------
        Path to the rendered HTML file.

        Example
        -------
        >>> from auto_actuary.analytics.speculative import (
        ...     rate_action_scenario, frequency_stress_scenario
        ... )
        >>> scenarios = [
        ...     rate_action_scenario("Rate +10% all", "all", {"all": 0.10}),
        ...     frequency_stress_scenario("Freq trend 5%/yr", 0.05, horizon_years=3),
        ... ]
        >>> session.scenario_report(scenarios, output_path="output/exec_scenarios.html")
        """
        from auto_actuary.analytics.speculative.scenario_engine import ScenarioEngine
        from auto_actuary.analytics.speculative.trend_projector import build_trend_projectors
        from auto_actuary.reports.executive.scenario_report import ScenarioReport

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        engine = self.scenario_engine(lob=lob, expense_ratio=expense_ratio)
        results = [engine.run_scenario(sp, n_boot=n_boot_ci) for sp in scenarios]

        # Build trend projectors from F/S analysis
        trend_projectors = {}
        try:
            fs = self.freq_severity(lob=lob or "")
            fs_tbl = fs.fs_table().reset_index()
            if "accident_year" in fs_tbl.columns:
                trend_projectors = build_trend_projectors(
                    fs_tbl, year_col="accident_year", n_boot=200
                )
        except Exception as e:
            logger.debug("Could not build trend projectors: %s", e)

        stress_df = None
        if run_stress_test:
            stress_df = engine.stress_test(n_simulations=n_stress_sims)

        report = ScenarioReport(
            engine=engine,
            results=results,
            trend_projectors=trend_projectors,
            stress_df=stress_df,
            lob=lob or "",
            horizon_years=horizon_years,
        )
        return report.render(output_path=output_path)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        tables = self.loader.loaded_tables
        return f"ActuarySession(config={self.config!r}, loaded={tables})"
