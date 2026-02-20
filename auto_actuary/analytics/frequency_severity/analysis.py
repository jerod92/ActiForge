"""
auto_actuary.analytics.frequency_severity.analysis
====================================================
Frequency/Severity (F/S) decomposition of pure premium.

    Pure Premium = Frequency × Severity

where:
    Frequency = Claims per Exposure Unit  (losses per car-year, etc.)
    Severity  = Average Loss per Claim

This module calculates F/S by accident year, coverage code, territory, and
class code, and fits trend models to each component.

Key outputs
-----------
- F/S table by year (accident year × coverage × territory)
- Trend factors for frequency, severity, and pure premium
- Credibility-weighted pure premium by class
- Territorial/class relativities (index to base)

References
----------
- Werner & Modlin (2016) "Basic Ratemaking", Chapters 5–6
- ISO loss cost manuals (structure reference)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from auto_actuary.analytics.ratemaking.trend import TrendAnalysis, TrendFit

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession
    from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)


class FreqSevAnalysis:
    """
    Frequency/severity decomposition and trend analysis.

    Parameters
    ----------
    losses : pd.DataFrame
        Columns: accident_year | coverage_code | incurred_loss | claim_count |
                 territory (optional) | class_code (optional)
    exposures : pd.DataFrame
        Columns: accident_year | coverage_code | earned_exposure |
                 territory (optional) | class_code (optional)
    lob : str
        Line of business label.
    coverage : str, optional
        Filter to a single coverage (e.g., 'BI', 'COMP').
    exclude_cat : bool
        Exclude catastrophe losses from trend fitting.
    """

    def __init__(
        self,
        losses: pd.DataFrame,
        exposures: pd.DataFrame,
        lob: str = "",
        coverage: Optional[str] = None,
        exclude_cat: bool = True,
    ) -> None:
        self.lob = lob
        self.coverage = coverage
        self.exclude_cat = exclude_cat

        self._losses = losses.copy()
        self._exposures = exposures.copy()

        if coverage:
            self._losses = self._losses[self._losses.get("coverage_code", pd.Series()) == coverage] if "coverage_code" in self._losses.columns else self._losses
            self._exposures = self._exposures[self._exposures.get("coverage_code", pd.Series()) == coverage] if "coverage_code" in self._exposures.columns else self._exposures

        if exclude_cat and "is_catastrophe" in self._losses.columns:
            self._losses = self._losses[self._losses["is_catastrophe"] != 1]

        self._fs_table: Optional[pd.DataFrame] = None
        self._trends: Dict[str, TrendAnalysis] = {}

    # ------------------------------------------------------------------
    # Core F/S table
    # ------------------------------------------------------------------

    def fs_table(self, by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute frequency/severity by accident year (and optional dimensions).

        Parameters
        ----------
        by : list of str, optional
            Additional grouping columns, e.g. ['coverage_code', 'territory'].

        Returns
        -------
        pd.DataFrame
            Index: accident_year (+ any *by* columns)
            Columns: earned_exposure | claim_count | incurred_loss |
                     frequency | severity | pure_premium
        """
        group_cols = ["accident_year"] + (by or [])
        loss_cols = ["incurred_loss", "claim_count"]
        exp_cols = ["earned_exposure"]

        # Aggregate losses
        avail_loss_cols = [c for c in loss_cols if c in self._losses.columns]
        loss_agg = (
            self._losses.groupby(group_cols)[avail_loss_cols]
            .sum()
            .reset_index()
        )

        # Aggregate exposures
        avail_exp_cols = [c for c in exp_cols if c in self._exposures.columns]
        exp_agg = (
            self._exposures.groupby(group_cols)[avail_exp_cols]
            .sum()
            .reset_index()
        )

        # Merge
        merged = loss_agg.merge(exp_agg, on=group_cols, how="outer")
        merged = merged.sort_values(group_cols)

        # Compute ratios
        merged["earned_exposure"] = merged.get("earned_exposure", 1)
        merged["claim_count"] = merged.get("claim_count", merged.get("incurred_loss", 0) / 1000)
        merged["incurred_loss"] = merged.get("incurred_loss", 0)

        exposure = merged["earned_exposure"].replace(0, np.nan)
        count = merged["claim_count"].replace(0, np.nan)

        merged["frequency"] = merged["claim_count"] / exposure
        merged["severity"] = merged["incurred_loss"] / count
        merged["pure_premium"] = merged["incurred_loss"] / exposure

        if group_cols:
            merged = merged.set_index(group_cols)

        self._fs_table = merged
        return merged

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def fit_trends(
        self,
        periods: Optional[List[int]] = None,
    ) -> Dict[str, TrendAnalysis]:
        """
        Fit log-linear trends to frequency, severity, and pure premium.

        Returns dict with keys: 'frequency', 'severity', 'pure_premium'
        """
        tbl = self.fs_table().reset_index()
        periods = periods or [3, 5, 10]

        for metric in ["frequency", "severity", "pure_premium"]:
            if metric not in tbl.columns:
                continue
            data = tbl[["accident_year", metric]].rename(
                columns={"accident_year": "year", metric: "value"}
            ).dropna()
            if len(data) >= 2:
                self._trends[metric] = TrendAnalysis(data, periods=periods, metric_name=metric)

        return self._trends

    def selected_trends(
        self,
        period: str = "5yr",
    ) -> Dict[str, Optional[TrendFit]]:
        """
        Return selected trend fits for F, S, PP.
        """
        if not self._trends:
            self.fit_trends()

        result = {}
        for metric, ta in self._trends.items():
            fits = ta._fits
            result[metric] = ta.select(period) if fits else None
        return result

    # ------------------------------------------------------------------
    # Territorial / class relativities
    # ------------------------------------------------------------------

    def relativities(
        self,
        dimension: str = "territory",
        base: Optional[str] = None,
        metric: str = "pure_premium",
    ) -> pd.DataFrame:
        """
        Compute indicated relativities for a classification variable.

        Parameters
        ----------
        dimension : str
            'territory' | 'class_code' | 'coverage_code'
        base : str, optional
            Value to index to 1.000.  Defaults to the dimension value with
            the highest exposure.
        metric : str
            'pure_premium' | 'frequency' | 'severity'

        Returns
        -------
        pd.DataFrame
            dimension_value | exposure | claim_count | pure_premium | relativity | credibility
        """
        tbl = self.fs_table(by=[dimension]).reset_index()
        group = tbl.groupby(dimension)

        summary = group.agg(
            earned_exposure=("earned_exposure", "sum"),
            claim_count=("claim_count", "sum"),
            incurred_loss=("incurred_loss", "sum"),
        ).reset_index()

        summary["pure_premium"] = summary["incurred_loss"] / summary["earned_exposure"].replace(0, np.nan)
        summary["frequency"] = summary["claim_count"] / summary["earned_exposure"].replace(0, np.nan)
        summary["severity"] = summary["incurred_loss"] / summary["claim_count"].replace(0, np.nan)

        # Base class
        if base is None:
            base = summary.loc[summary["earned_exposure"].idxmax(), dimension]

        base_val = summary.loc[summary[dimension] == base, metric].values
        if len(base_val) == 0 or base_val[0] == 0:
            logger.warning("Base %s='%s' not found or zero — using overall mean", dimension, base)
            base_pp = summary[metric].mean()
        else:
            base_pp = float(base_val[0])

        summary["relativity"] = summary[metric] / base_pp

        # Classical credibility
        full_cred = 1082
        summary["credibility"] = (np.sqrt(summary["claim_count"] / full_cred)).clip(0, 1)
        summary["cred_relativity"] = (
            summary["credibility"] * summary["relativity"]
            + (1 - summary["credibility"]) * 1.0  # complement = base class
        )

        return summary.set_index(dimension)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, float]:
        """Return key scalar metrics for dashboard use."""
        tbl = self.fs_table().reset_index()
        overall_exp = tbl["earned_exposure"].sum()
        overall_loss = tbl["incurred_loss"].sum()
        overall_cnt = tbl["claim_count"].sum()
        return {
            "total_earned_exposure": float(overall_exp),
            "total_losses": float(overall_loss),
            "total_claims": float(overall_cnt),
            "overall_frequency": float(overall_cnt / overall_exp) if overall_exp else np.nan,
            "overall_severity": float(overall_loss / overall_cnt) if overall_cnt else np.nan,
            "overall_pure_premium": float(overall_loss / overall_exp) if overall_exp else np.nan,
        }

    def __repr__(self) -> str:
        summ = self.summary()
        return (
            f"FreqSevAnalysis(lob={self.lob!r}, "
            f"coverage={self.coverage!r}, "
            f"pp={summ.get('overall_pure_premium', 0):,.2f})"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: str,
        coverage: Optional[str] = None,
        **kwargs,
    ) -> "FreqSevAnalysis":
        """
        Build FreqSevAnalysis from loaded session data.

        Requires: claims, valuations, policies loaded in session.
        """
        claims = session.loader["claims"].copy()
        vals = session.loader["valuations"].copy()

        # Filter LOB
        claims_lob = claims[claims["line_of_business"] == lob].copy()
        claims_lob["accident_year"] = claims_lob["accident_date"].dt.year

        # Latest valuation per claim
        latest_vals = (
            vals.sort_values("valuation_date")
            .groupby("claim_id")
            .last()
            .reset_index()
        )

        # Join claims to valuations
        merged = claims_lob.merge(latest_vals[["claim_id", "incurred_loss"]], on="claim_id", how="left")
        merged["claim_count"] = 1
        merged["incurred_loss"] = merged["incurred_loss"].fillna(0)

        loss_cols = ["accident_year", "coverage_code", "territory", "class_code",
                     "incurred_loss", "claim_count", "is_catastrophe"]
        avail = [c for c in loss_cols if c in merged.columns]
        losses_df = merged[avail]

        # Build exposure from policies
        policies = session.loader["policies"].copy()
        policies_lob = policies[policies["line_of_business"] == lob].copy()
        policies_lob["accident_year"] = policies_lob["effective_date"].dt.year

        exp_cols = ["accident_year", "territory", "class_code", "written_exposure"]
        avail_exp = [c for c in exp_cols if c in policies_lob.columns]
        exposures_df = policies_lob[avail_exp].rename(columns={"written_exposure": "earned_exposure"})

        return cls(
            losses=losses_df,
            exposures=exposures_df,
            lob=lob,
            coverage=coverage,
            **kwargs,
        )
