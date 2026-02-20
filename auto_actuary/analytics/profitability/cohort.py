"""
auto_actuary.analytics.profitability.cohort
============================================
Policy vintage (cohort) profitability analysis.

Groups policies by their effective year (vintage/cohort) and tracks
the development of losses and profitability over time.

Key questions answered:
  - Which policy years are running at a loss after full development?
  - How does loss ratio develop from initial report to ultimate?
  - Which new business cohorts vs. renewals are most profitable?
  - How does retention interact with ultimate profitability?

Output tables
-------------
  - Cohort P&L: by vintage | earned_premium | ultimate_loss | loss_ratio | uw_profit
  - Cohort development: how loss ratio has evolved at each evaluation date
  - NB vs Renewal split by vintage
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession

logger = logging.getLogger(__name__)


class CohortReport:
    """
    Vintage cohort profitability analysis.

    Parameters
    ----------
    policies : pd.DataFrame
        Must have: policy_id | effective_date | written_premium | line_of_business |
                   transaction_type (NB vs RN) | written_exposure
    claims : pd.DataFrame
        Must have: policy_id | accident_date | claim_id | line_of_business
    valuations : pd.DataFrame
        Must have: claim_id | valuation_date | incurred_loss
    lob : str, optional
    """

    def __init__(
        self,
        policies: pd.DataFrame,
        claims: pd.DataFrame,
        valuations: pd.DataFrame,
        lob: Optional[str] = None,
    ) -> None:
        self.lob = lob
        filt_lob = lambda df, col="line_of_business": (
            df[df[col] == lob].copy() if lob and col in df.columns else df.copy()
        )
        self._policies = filt_lob(policies)
        self._claims = filt_lob(claims)
        self._vals = valuations.copy()

        self._policies["vintage"] = self._policies["effective_date"].dt.year

    # ------------------------------------------------------------------
    # Premium by vintage
    # ------------------------------------------------------------------

    def _premium_by_vintage(self) -> pd.DataFrame:
        pol = self._policies.copy()
        ep_col = "earned_premium" if "earned_premium" in pol.columns else "written_premium"
        wp = pol.groupby("vintage")["written_premium"].sum().rename("written_premium")
        ep = pol.groupby("vintage")[ep_col].sum().rename("earned_premium")
        exp = pol.groupby("vintage")["written_exposure"].sum().rename("written_exposure") if "written_exposure" in pol.columns else pd.Series(dtype=float)

        nb_mask = pol.get("transaction_type", pd.Series("RN", index=pol.index)).isin(["NB"])
        nb_prem = pol[nb_mask].groupby("vintage")["written_premium"].sum().rename("nb_written_premium")
        rn_prem = pol[~nb_mask].groupby("vintage")["written_premium"].sum().rename("rn_written_premium")
        policy_count = pol.groupby("vintage")["policy_id"].count().rename("policy_count")

        df = pd.concat([wp, ep, exp, nb_prem, rn_prem, policy_count], axis=1).fillna(0)
        df["nb_pct"] = df.get("nb_written_premium", 0) / df["written_premium"].replace(0, np.nan)
        return df

    # ------------------------------------------------------------------
    # Loss by vintage (latest valuation)
    # ------------------------------------------------------------------

    def _loss_by_vintage(self) -> pd.DataFrame:
        # Join claims → policies to get vintage
        pol = self._policies[["policy_id", "vintage"]].drop_duplicates()
        claims_v = self._claims.merge(pol, on="policy_id", how="inner")

        # Latest valuation per claim
        latest = (
            self._vals.sort_values("valuation_date")
            .groupby("claim_id")
            .last()
            .reset_index()
        )
        merged = claims_v.merge(latest[["claim_id", "incurred_loss"]], on="claim_id", how="left")
        merged["incurred_loss"] = merged["incurred_loss"].fillna(0)
        merged["claim_count"] = 1

        return merged.groupby("vintage")[["incurred_loss", "claim_count"]].sum()

    # ------------------------------------------------------------------
    # Cohort P&L
    # ------------------------------------------------------------------

    def cohort_pl(self) -> pd.DataFrame:
        """
        Return cohort profitability table.

        Columns: vintage | written_premium | earned_premium | policy_count |
                 nb_pct | incurred_loss | claim_count | loss_ratio |
                 uw_profit | uw_profit_pct | frequency | severity
        """
        prem = self._premium_by_vintage()
        loss = self._loss_by_vintage()

        df = prem.join(loss, how="outer").fillna(0)
        ep = df["earned_premium"].replace(0, np.nan)
        df["loss_ratio"] = df["incurred_loss"] / ep
        df["uw_profit"] = df["earned_premium"] - df["incurred_loss"]
        df["uw_profit_pct"] = df["uw_profit"] / df["earned_premium"].replace(0, np.nan)
        df["frequency"] = df["claim_count"] / df.get("written_exposure", df["policy_count"]).replace(0, np.nan)
        df["severity"] = df["incurred_loss"] / df["claim_count"].replace(0, np.nan)

        return df.sort_index()

    # ------------------------------------------------------------------
    # Development of loss ratio by cohort
    # ------------------------------------------------------------------

    def cohort_development(self) -> pd.DataFrame:
        """
        Show how each cohort's loss ratio has changed at each valuation date.

        Returns a pivot table:
            vintage (rows) × valuation_year (cols) = loss_ratio

        Useful for spotting adverse reserve development in specific vintages.
        """
        pol = self._policies[["policy_id", "vintage", "written_premium"]].drop_duplicates()
        prem_by_vintage = pol.groupby("vintage")["written_premium"].sum()

        claims_v = self._claims.merge(pol[["policy_id", "vintage"]], on="policy_id", how="inner")

        merged = claims_v.merge(
            self._vals[["claim_id", "valuation_date", "incurred_loss"]],
            on="claim_id",
            how="left",
        )
        merged["valuation_year"] = merged["valuation_date"].dt.year
        merged["incurred_loss"] = merged["incurred_loss"].fillna(0)

        pivot = merged.groupby(["vintage", "valuation_year"])["incurred_loss"].sum().unstack(level=1)

        # Divide each cell by earned premium for that vintage → loss ratio
        lr_pivot = pivot.div(prem_by_vintage, axis=0)
        return lr_pivot.sort_index(axis=0).sort_index(axis=1)

    def summary(self) -> Dict[str, float]:
        """Portfolio-level cohort summary."""
        pl = self.cohort_pl()
        return {
            "total_written_premium": float(pl["written_premium"].sum()),
            "total_earned_premium": float(pl["earned_premium"].sum()),
            "total_incurred_loss": float(pl["incurred_loss"].sum()),
            "total_claim_count": float(pl["claim_count"].sum()),
            "overall_loss_ratio": float(pl["incurred_loss"].sum() / pl["earned_premium"].sum()) if pl["earned_premium"].sum() else np.nan,
            "overall_uw_profit": float(pl["uw_profit"].sum()),
            "worst_vintage_lr": float(pl["loss_ratio"].max()),
            "best_vintage_lr": float(pl["loss_ratio"].min()),
            "nb_pct_recent": float(pl["nb_pct"].iloc[-1]) if not pl.empty else np.nan,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"CohortReport(lob={self.lob!r}, "
            f"vintages={len(self.cohort_pl())}, "
            f"overall_lr={s.get('overall_loss_ratio', 0):.3f})"
        )

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: Optional[str] = None,
    ) -> "CohortReport":
        return cls(
            policies=session.loader["policies"],
            claims=session.loader["claims"],
            valuations=session.loader["valuations"],
            lob=lob,
        )
