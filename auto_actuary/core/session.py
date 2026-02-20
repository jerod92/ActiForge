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
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        tables = self.loader.loaded_tables
        return f"ActuarySession(config={self.config!r}, loaded={tables})"
