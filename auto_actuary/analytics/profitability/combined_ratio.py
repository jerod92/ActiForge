"""
auto_actuary.analytics.profitability.combined_ratio
====================================================
Combined ratio — the headline P&C profitability KPI.

    Combined Ratio = Loss Ratio + Expense Ratio

    Loss Ratio    = (Losses + ALAE) / Earned Premium
    Expense Ratio = (ULAE + Commissions + G&A) / Written Premium*

    * Convention: expense ratio is often on written premium basis for LAG

    Operating Ratio = Combined Ratio − Investment Income Ratio

A combined ratio < 100% means an underwriting profit.
A combined ratio > 100% means an underwriting loss.

Also computes:
  - Loss Ratio by accident year (AY basis)
  - Expense Ratio by calendar year (CY basis)
  - Combined Ratio trend (calendar year)
  - Underwriting profit / loss
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession

logger = logging.getLogger(__name__)


class CombinedRatioReport:
    """
    Combined ratio computation.

    Parameters
    ----------
    losses : pd.DataFrame
        Columns: calendar_year | line_of_business | incurred_loss | paid_alae | case_alae
    expenses : pd.DataFrame
        Columns: calendar_year | line_of_business | expense_type | amount |
                 written_premium | earned_premium
    premiums : pd.DataFrame
        Columns: calendar_year | line_of_business | earned_premium | written_premium
    lob : str, optional
        Filter to single LOB.
    """

    def __init__(
        self,
        losses: pd.DataFrame,
        expenses: pd.DataFrame,
        premiums: pd.DataFrame,
        lob: Optional[str] = None,
    ) -> None:
        self.lob = lob
        filt = lambda df: df[df["line_of_business"] == lob].copy() if lob and "line_of_business" in df.columns else df.copy()
        self._losses = filt(losses)
        self._expenses = filt(expenses)
        self._premiums = filt(premiums)

    # ------------------------------------------------------------------
    # Core table
    # ------------------------------------------------------------------

    def by_year(self) -> pd.DataFrame:
        """
        Combined ratio by calendar year.

        Returns
        -------
        pd.DataFrame
            calendar_year | earned_premium | written_premium |
            incurred_loss | lae | total_loss_lae | expense_amount |
            loss_ratio | lae_ratio | loss_lae_ratio | expense_ratio | combined_ratio |
            uw_profit_loss
        """
        # Premiums
        prem_grp = self._premiums.groupby("calendar_year")[
            ["earned_premium", "written_premium"]
        ].sum()

        # Losses
        loss_cols = [c for c in ["incurred_loss", "paid_alae", "case_alae"] if c in self._losses.columns]
        if "calendar_year" not in self._losses.columns:
            # Derive from accident_year (approximate calendar = accident for paid losses)
            self._losses["calendar_year"] = self._losses.get("accident_year", pd.Series(dtype=int))
        loss_grp = self._losses.groupby("calendar_year")[loss_cols].sum()

        # Expenses
        if not self._expenses.empty:
            exp_grp = self._expenses.groupby("calendar_year")["amount"].sum().rename("expense_amount")
        else:
            exp_grp = pd.Series(dtype=float, name="expense_amount")

        # Join
        df = prem_grp.join(loss_grp, how="outer").join(exp_grp, how="outer").fillna(0)

        ep = df["earned_premium"].replace(0, np.nan)
        wp = df.get("written_premium", df["earned_premium"]).replace(0, np.nan)

        df["lae"] = df.get("paid_alae", 0) + df.get("case_alae", 0)
        df["total_loss_lae"] = df.get("incurred_loss", 0) + df["lae"]
        df["loss_ratio"] = df.get("incurred_loss", 0) / ep
        df["lae_ratio"] = df["lae"] / ep
        df["loss_lae_ratio"] = df["total_loss_lae"] / ep
        df["expense_ratio"] = df.get("expense_amount", 0) / wp
        df["combined_ratio"] = df["loss_lae_ratio"] + df["expense_ratio"]
        df["uw_profit_loss"] = df["earned_premium"] - df["total_loss_lae"] - df.get("expense_amount", 0)

        return df.sort_index()

    def current_year(self) -> Dict[str, float]:
        """KPIs for the most recent calendar year."""
        tbl = self.by_year()
        if tbl.empty:
            return {}
        row = tbl.iloc[-1]
        return {
            "calendar_year": int(tbl.index[-1]),
            "earned_premium": float(row.get("earned_premium", 0)),
            "loss_ratio": float(row.get("loss_ratio", np.nan)),
            "lae_ratio": float(row.get("lae_ratio", np.nan)),
            "loss_lae_ratio": float(row.get("loss_lae_ratio", np.nan)),
            "expense_ratio": float(row.get("expense_ratio", np.nan)),
            "combined_ratio": float(row.get("combined_ratio", np.nan)),
            "uw_profit_loss": float(row.get("uw_profit_loss", np.nan)),
        }

    def three_year_avg(self) -> Dict[str, float]:
        """Simple 3-year average of key ratios."""
        tbl = self.by_year().tail(3)
        if tbl.empty:
            return {}
        return {
            "avg_loss_ratio": float(tbl["loss_ratio"].mean()),
            "avg_loss_lae_ratio": float(tbl["loss_lae_ratio"].mean()),
            "avg_expense_ratio": float(tbl["expense_ratio"].mean()),
            "avg_combined_ratio": float(tbl["combined_ratio"].mean()),
        }

    def trend_series(self, metric: str = "combined_ratio") -> pd.Series:
        """Return a time series of *metric* for charting."""
        tbl = self.by_year()
        return tbl[metric] if metric in tbl.columns else pd.Series(dtype=float)

    def __repr__(self) -> str:
        cy = self.current_year()
        cr = cy.get("combined_ratio", np.nan)
        return f"CombinedRatioReport(lob={self.lob!r}, combined_ratio={cr:.3f})"

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: Optional[str] = None,
    ) -> "CombinedRatioReport":
        """Build from session."""
        claims = session.loader["claims"].copy()
        vals = session.loader["valuations"].copy()
        policies = session.loader["policies"].copy()

        latest = vals.sort_values("valuation_date").groupby("claim_id").last().reset_index()
        merged = claims.merge(latest, on="claim_id", how="left")
        merged["calendar_year"] = merged["accident_date"].dt.year
        if "line_of_business" not in merged.columns and "line_of_business" in claims.columns:
            merged["line_of_business"] = claims["line_of_business"]

        policies["calendar_year"] = policies["effective_date"].dt.year
        ep_col = "earned_premium" if "earned_premium" in policies.columns else "written_premium"
        prem_df = policies.rename(columns={ep_col: "earned_premium"})
        if "written_premium" not in prem_df.columns:
            prem_df["written_premium"] = prem_df["earned_premium"]

        expenses_df = session.loader["expenses"].copy() if "expenses" in session.loader else pd.DataFrame()

        return cls(losses=merged, expenses=expenses_df, premiums=prem_df, lob=lob)
