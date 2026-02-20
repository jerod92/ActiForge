"""
auto_actuary.analytics.reserves.adequacy
=========================================
Reserve adequacy testing — measures how held reserves compare to developed
actuary estimates and tracks reserve runoff (development since last evaluation).

Key metrics produced
--------------------
- Reserve development (favorable / adverse) since prior evaluation
- Percent developed relative to prior ultimate estimate
- Redundancy / deficiency of held reserves vs. actuarial ultimates
- Calendar-year loss development triangle (useful for ULAE allocation)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.analytics.reserves.ibnr import ReserveAnalysis

logger = logging.getLogger(__name__)


class ReserveAdequacy:
    """
    Compare held (booked) reserves to actuarially estimated IBNR.

    Parameters
    ----------
    analysis_current : ReserveAnalysis
        Reserve analysis at the current evaluation date.
    held_reserves : pd.Series, optional
        Booked IBNR by origin year.  If not provided, assumed = 0.
    analysis_prior : ReserveAnalysis, optional
        Reserve analysis from the prior evaluation date (for development).
    prior_ultimate : pd.Series, optional
        Prior estimate of ultimate by origin year (for % developed).
    """

    def __init__(
        self,
        analysis_current: "ReserveAnalysis",
        held_reserves: Optional[pd.Series] = None,
        analysis_prior: Optional["ReserveAnalysis"] = None,
        prior_ultimate: Optional[pd.Series] = None,
    ) -> None:
        self.analysis = analysis_current
        self.held_reserves = held_reserves
        self.analysis_prior = analysis_prior
        self.prior_ultimate = prior_ultimate

    def adequacy_table(self, method: Optional[str] = None) -> pd.DataFrame:
        """
        Return reserve adequacy by origin year.

        Columns
        -------
        origin | reported | actuary_ultimate | actuary_ibnr |
        held_ibnr | redundancy | redundancy_pct | prior_ultimate | development
        """
        res = self.analysis.selected(method)
        tri = self.analysis.triangle
        origins = tri.origins
        diag = tri.latest_diagonal

        df = pd.DataFrame(index=origins)
        df.index.name = "origin"
        df["reported"] = diag
        df["actuary_ultimate"] = res.ultimates.reindex(origins)
        df["actuary_ibnr"] = res.ibnr.reindex(origins)

        if self.held_reserves is not None:
            df["held_ibnr"] = self.held_reserves.reindex(origins).fillna(0)
            df["redundancy"] = df["held_ibnr"] - df["actuary_ibnr"]
            df["redundancy_pct"] = df["redundancy"] / df["actuary_ultimate"].replace(0, np.nan)
        else:
            df["held_ibnr"] = np.nan
            df["redundancy"] = np.nan
            df["redundancy_pct"] = np.nan

        if self.prior_ultimate is not None:
            df["prior_ultimate"] = self.prior_ultimate.reindex(origins)
            df["calendar_yr_development"] = df["actuary_ultimate"] - df["prior_ultimate"]
            df["pct_developed"] = df["calendar_yr_development"] / df["prior_ultimate"].replace(0, np.nan)
        else:
            df["prior_ultimate"] = np.nan
            df["calendar_yr_development"] = np.nan
            df["pct_developed"] = np.nan

        # Totals
        num_cols = df.select_dtypes("number").columns
        totals = df[num_cols].sum()
        totals["redundancy_pct"] = (
            totals["redundancy"] / totals["actuary_ultimate"]
            if totals.get("actuary_ultimate", 0) != 0
            else np.nan
        )
        totals["pct_developed"] = (
            totals["calendar_yr_development"] / totals["prior_ultimate"]
            if totals.get("prior_ultimate", 0) != 0
            else np.nan
        )
        df.loc["TOTAL"] = totals

        return df

    def development_summary(self) -> Dict[str, float]:
        """
        Return a dict with key scalar adequacy metrics.
        """
        tbl = self.adequacy_table()
        tot = tbl.loc["TOTAL"]
        return {
            "total_actuary_ultimate": float(tot.get("actuary_ultimate", np.nan)),
            "total_actuary_ibnr": float(tot.get("actuary_ibnr", np.nan)),
            "total_held_ibnr": float(tot.get("held_ibnr", np.nan)),
            "total_redundancy": float(tot.get("redundancy", np.nan)),
            "redundancy_pct": float(tot.get("redundancy_pct", np.nan)),
            "calendar_yr_development": float(tot.get("calendar_yr_development", np.nan)),
            "pct_developed": float(tot.get("pct_developed", np.nan)),
        }

    def calendar_year_triangle(self) -> pd.DataFrame:
        """
        Build a calendar-year (anti-diagonal) development triangle.

        Each anti-diagonal represents a calendar year of development.
        Useful for ULAE analysis and identifying systematic development patterns.
        """
        tri = self.analysis.triangle.triangle
        origins = tri.index.values
        ages = tri.columns.values

        records = []
        for i, origin in enumerate(origins):
            for j, age in enumerate(ages):
                val = tri.loc[origin, age]
                if pd.notna(val):
                    cal_year = origin + age // 12  # approximate calendar year
                    records.append(
                        {"origin": origin, "dev_age": age, "calendar_year": cal_year, "value": val}
                    )

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Pivot so rows = origin, cols = calendar year
        cal_tri = df.pivot_table(index="origin", columns="calendar_year", values="value", aggfunc="sum")
        return cal_tri.sort_index(axis=0).sort_index(axis=1)

    def __repr__(self) -> str:
        summ = self.development_summary()
        redund = summ.get("total_redundancy", np.nan)
        return (
            f"ReserveAdequacy(actuary_ibnr={summ.get('total_actuary_ibnr', 0):,.0f}, "
            f"held_ibnr={summ.get('total_held_ibnr', 0):,.0f}, "
            f"redundancy={redund:+,.0f})"
        )
