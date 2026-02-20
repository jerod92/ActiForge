"""
auto_actuary.analytics.profitability.loss_ratio
================================================
Loss ratio analysis — the most fundamental P&C profitability metric.

    Loss Ratio = Losses / Earned Premium

Multiple flavors are computed:
  - Paid Loss Ratio    = Paid Losses / Earned Premium
  - Incurred Loss Ratio = Incurred Losses (paid + case) / Earned Premium
  - Ultimate Loss Ratio = Actuarial Ultimate Losses / On-Level EP
  - LAE Ratio          = Loss Adjustment Expense / Earned Premium
  - Combined Loss & LAE Ratio

Sliced by:
  - Accident Year
  - Calendar Year
  - Line of Business
  - Coverage Code
  - Territory
  - Class Code
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession

logger = logging.getLogger(__name__)


class LossRatioReport:
    """
    Loss ratio analysis, sliced and diced across dimensions.

    Parameters
    ----------
    losses : pd.DataFrame
        Columns: accident_year | line_of_business | coverage_code | territory |
                 class_code | paid_loss | incurred_loss | paid_alae | case_alae
    premiums : pd.DataFrame
        Columns: accident_year | line_of_business | coverage_code | territory |
                 class_code | earned_premium
    lob : str, optional
        Filter to a single LOB.
    """

    def __init__(
        self,
        losses: pd.DataFrame,
        premiums: pd.DataFrame,
        lob: Optional[str] = None,
    ) -> None:
        self.lob = lob
        self._losses = losses[losses["line_of_business"] == lob].copy() if lob else losses.copy()
        self._premiums = premiums[premiums["line_of_business"] == lob].copy() if lob else premiums.copy()

    # ------------------------------------------------------------------
    # Core table builder
    # ------------------------------------------------------------------

    def _build(
        self,
        group_cols: List[str],
        loss_col: str = "incurred_loss",
    ) -> pd.DataFrame:
        """Aggregate losses and premium by group_cols and compute ratios."""
        # Aggregate losses
        loss_agg_cols = [c for c in ["paid_loss", "incurred_loss", "paid_alae", "case_alae"]
                         if c in self._losses.columns]
        loss_grp = self._losses.groupby(group_cols)[loss_agg_cols].sum().reset_index()

        # Aggregate premium
        prem_grp = self._premiums.groupby(group_cols)[["earned_premium"]].sum().reset_index()

        # Merge
        df = loss_grp.merge(prem_grp, on=group_cols, how="outer").fillna(0)
        ep = df["earned_premium"].replace(0, np.nan)

        if "paid_loss" in df.columns:
            df["paid_loss_ratio"] = df["paid_loss"] / ep
        if "incurred_loss" in df.columns:
            df["incurred_loss_ratio"] = df["incurred_loss"] / ep
        if "paid_alae" in df.columns and "case_alae" in df.columns:
            df["lae"] = df["paid_alae"] + df["case_alae"]
            df["lae_ratio"] = df["lae"] / ep
        elif "paid_alae" in df.columns:
            df["lae"] = df["paid_alae"]
            df["lae_ratio"] = df["lae"] / ep

        return df.set_index(group_cols)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def by_accident_year(
        self,
        extra_dims: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Loss ratios by accident year."""
        group_cols = ["accident_year"] + (extra_dims or [])
        return self._build(group_cols)

    def by_calendar_year(
        self,
        extra_dims: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Loss ratios by calendar year (paid basis)."""
        if "calendar_year" not in self._losses.columns:
            logger.warning("'calendar_year' not in losses — cannot compute calendar-year LR")
            return pd.DataFrame()
        group_cols = ["calendar_year"] + (extra_dims or [])
        return self._build(group_cols, loss_col="paid_loss")

    def by_lob(self) -> pd.DataFrame:
        """Portfolio summary by LOB."""
        return self._build(["line_of_business"])

    def by_coverage(self, accident_year: Optional[int] = None) -> pd.DataFrame:
        """Loss ratios by coverage code."""
        df = self._losses.copy()
        if accident_year:
            df = df[df["accident_year"] == accident_year]
        group_cols = ["coverage_code"]
        l_grp = df.groupby(group_cols)[
            [c for c in ["paid_loss", "incurred_loss", "paid_alae"] if c in df.columns]
        ].sum()
        p_grp = self._premiums.groupby(group_cols)[["earned_premium"]].sum()
        result = l_grp.join(p_grp, how="outer").fillna(0)
        ep = result["earned_premium"].replace(0, np.nan)
        if "incurred_loss" in result:
            result["incurred_loss_ratio"] = result["incurred_loss"] / ep
        return result

    def by_territory(self) -> pd.DataFrame:
        """Loss ratios by territory."""
        if "territory" not in self._losses.columns:
            return pd.DataFrame()
        return self._build(["territory"])

    def trending(self, metric: str = "incurred_loss_ratio") -> pd.DataFrame:
        """
        Return the metric by accident year for trend/spark-line display.
        """
        tbl = self.by_accident_year()
        if metric not in tbl.columns:
            return pd.DataFrame()
        return tbl[[metric]].sort_index()

    def summary(self) -> Dict[str, float]:
        """Overall scalar summary."""
        total_loss = self._losses.get("incurred_loss", pd.Series([0])).sum()
        total_prem = self._premiums.get("earned_premium", pd.Series([0])).sum()
        total_paid = self._losses.get("paid_loss", pd.Series([0])).sum()
        total_alae = (
            self._losses.get("paid_alae", pd.Series([0])).sum()
            + self._losses.get("case_alae", pd.Series([0])).sum()
        )
        ep = total_prem if total_prem != 0 else np.nan
        return {
            "earned_premium": float(total_prem),
            "incurred_loss": float(total_loss),
            "paid_loss": float(total_paid),
            "lae": float(total_alae),
            "incurred_loss_ratio": float(total_loss / ep) if ep else np.nan,
            "paid_loss_ratio": float(total_paid / ep) if ep else np.nan,
            "lae_ratio": float(total_alae / ep) if ep else np.nan,
            "loss_lae_ratio": float((total_loss + total_alae) / ep) if ep else np.nan,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"LossRatioReport(lob={self.lob!r}, "
            f"EP={s.get('earned_premium', 0):,.0f}, "
            f"ILR={s.get('incurred_loss_ratio', 0):.3f})"
        )

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: Optional[str] = None,
        by: Optional[List[str]] = None,
    ) -> "LossRatioReport":
        """Build from session data."""
        claims = session.loader["claims"].copy()
        vals = session.loader["valuations"].copy()
        policies = session.loader["policies"].copy()

        # Latest valuation per claim
        latest = vals.sort_values("valuation_date").groupby("claim_id").last().reset_index()
        merged = claims.merge(latest[["claim_id", "paid_loss", "incurred_loss",
                                      "paid_alae", "case_alae"]], on="claim_id", how="left")
        merged["accident_year"] = merged["accident_date"].dt.year
        for col in ["paid_loss", "incurred_loss", "paid_alae", "case_alae"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)

        policies_e = policies.copy()
        policies_e["accident_year"] = policies_e["effective_date"].dt.year
        prem_ep_col = "earned_premium" if "earned_premium" in policies_e.columns else "written_premium"
        premiums = policies_e.rename(columns={prem_ep_col: "earned_premium"})

        return cls(losses=merged, premiums=premiums, lob=lob)
