"""
auto_actuary.analytics.catastrophe.cat_analysis
================================================
Catastrophe (CAT) loss analysis for P&C carriers.

Separates CAT from non-CAT losses and provides:
  - CAT loss by event (cat_code) and accident year
  - CAT vs. non-CAT loss ratios
  - CAT frequency (events per year) and average severity
  - Expected CAT load for ratemaking
  - Territory-level CAT concentration

Important actuarial note:
  CAT losses are excluded from loss trend analysis (since they are
  not predictable from historical frequency/severity trends).
  Instead, expected CAT loads are typically sourced from a catastrophe
  model (AIR, RMS, Verisk) or historical average gross-up.

  auto_actuary supports a simple expected-value approach using historical
  data as a placeholder when cat model output is not available.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession
    from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)


class CatAnalysis:
    """
    Catastrophe loss analysis.

    Parameters
    ----------
    claims : pd.DataFrame
        Must have: claim_id | accident_date | is_catastrophe | cat_code |
                   coverage_code | territory | line_of_business
    valuations : pd.DataFrame
        Must have: claim_id | valuation_date | incurred_loss
    premiums : pd.DataFrame
        Must have: accident_year | line_of_business | earned_premium
    lob : str, optional
    """

    def __init__(
        self,
        claims: pd.DataFrame,
        valuations: pd.DataFrame,
        premiums: pd.DataFrame,
        lob: Optional[str] = None,
    ) -> None:
        self.lob = lob
        filt = lambda df: df[df["line_of_business"] == lob].copy() if lob and "line_of_business" in df.columns else df.copy()
        self._claims = filt(claims)
        self._premiums = filt(premiums)
        self._vals = valuations.copy()

        self._claims["accident_year"] = self._claims["accident_date"].dt.year

        # Latest valuation per claim
        self._latest = (
            self._vals.sort_values("valuation_date")
            .groupby("claim_id")
            .last()
            .reset_index()
        )
        self._merged = self._claims.merge(
            self._latest[["claim_id", "incurred_loss"]], on="claim_id", how="left"
        )
        self._merged["incurred_loss"] = self._merged["incurred_loss"].fillna(0)
        is_cat_col = "is_catastrophe" if "is_catastrophe" in self._merged.columns else None
        if is_cat_col:
            self._cat = self._merged[self._merged[is_cat_col] == 1]
            self._non_cat = self._merged[self._merged[is_cat_col] != 1]
        else:
            logger.warning("No 'is_catastrophe' column — treating all claims as non-CAT")
            self._cat = self._merged.iloc[:0]  # empty
            self._non_cat = self._merged

    # ------------------------------------------------------------------
    # CAT vs non-CAT split
    # ------------------------------------------------------------------

    def split_by_year(self) -> pd.DataFrame:
        """
        Annual CAT vs. non-CAT loss split.

        Columns: accident_year | cat_loss | non_cat_loss | total_loss |
                 cat_ratio | earned_premium | cat_loss_ratio | non_cat_loss_ratio
        """
        cat_agg = self._cat.groupby("accident_year")["incurred_loss"].sum().rename("cat_loss")
        noncat_agg = self._non_cat.groupby("accident_year")["incurred_loss"].sum().rename("non_cat_loss")

        # Premium
        prem = self._premiums.copy()
        if "accident_year" not in prem.columns:
            prem["accident_year"] = prem.get("calendar_year", pd.Series(dtype=int))
        prem_agg = prem.groupby("accident_year")["earned_premium"].sum()

        df = pd.concat([cat_agg, noncat_agg], axis=1).fillna(0)
        df = df.join(prem_agg, how="outer").fillna(0)
        df["total_loss"] = df["cat_loss"] + df["non_cat_loss"]
        ep = df["earned_premium"].replace(0, np.nan)
        df["cat_loss_ratio"] = df["cat_loss"] / ep
        df["non_cat_loss_ratio"] = df["non_cat_loss"] / ep
        df["cat_pct_of_total"] = df["cat_loss"] / df["total_loss"].replace(0, np.nan)

        return df.sort_index()

    # ------------------------------------------------------------------
    # Event analysis
    # ------------------------------------------------------------------

    def by_event(self) -> pd.DataFrame:
        """
        CAT losses by event (cat_code).

        Columns: cat_code | accident_year | claim_count | incurred_loss | avg_severity | territories
        """
        if self._cat.empty:
            return pd.DataFrame()

        cat_code_col = "cat_code" if "cat_code" in self._cat.columns else None
        if not cat_code_col:
            return pd.DataFrame()

        grp = self._cat.groupby([cat_code_col, "accident_year"])
        agg = grp.agg(
            claim_count=("claim_id", "count"),
            incurred_loss=("incurred_loss", "sum"),
        ).reset_index()

        agg["avg_severity"] = agg["incurred_loss"] / agg["claim_count"].replace(0, np.nan)

        if "territory" in self._cat.columns:
            territory_by_event = (
                self._cat.groupby(cat_code_col)["territory"]
                .apply(lambda x: ", ".join(sorted(x.unique())))
                .rename("territories")
            )
            agg = agg.merge(territory_by_event, on=cat_code_col, how="left")

        return agg.sort_values("incurred_loss", ascending=False)

    # ------------------------------------------------------------------
    # CAT load for ratemaking
    # ------------------------------------------------------------------

    def expected_cat_load(
        self,
        method: str = "expected_value",
        years: int = 5,
    ) -> Dict[str, float]:
        """
        Compute expected CAT load for ratemaking.

        Parameters
        ----------
        method : str
            'expected_value' — simple historical average
        years : int
            Number of most recent years to use.

        Returns
        -------
        dict with keys:
            avg_cat_loss | avg_earned_premium | expected_cat_lr | cat_load_pct
        """
        tbl = self.split_by_year().tail(years)
        avg_cat = float(tbl["cat_loss"].mean())
        avg_ep = float(tbl["earned_premium"].mean())
        cat_lr = avg_cat / avg_ep if avg_ep else np.nan

        return {
            "avg_cat_loss": avg_cat,
            "avg_earned_premium": avg_ep,
            "expected_cat_lr": cat_lr,
            "cat_load_pct": cat_lr,
            "years_used": min(years, len(tbl)),
            "method": method,
        }

    # ------------------------------------------------------------------
    # Territory concentration
    # ------------------------------------------------------------------

    def territory_concentration(self) -> pd.DataFrame:
        """
        CAT loss concentration by territory.

        Useful for identifying geographic accumulation risk.
        """
        if self._cat.empty or "territory" not in self._cat.columns:
            return pd.DataFrame()

        terr = self._cat.groupby("territory").agg(
            cat_loss=("incurred_loss", "sum"),
            cat_claim_count=("claim_id", "count"),
            event_count=("cat_code", "nunique"),
        )
        terr["avg_loss_per_event"] = terr["cat_loss"] / terr["event_count"].replace(0, np.nan)
        terr["pct_of_total"] = terr["cat_loss"] / terr["cat_loss"].sum()
        return terr.sort_values("cat_loss", ascending=False)

    def summary(self) -> Dict[str, float]:
        """Key scalar CAT metrics."""
        split = self.split_by_year()
        if split.empty:
            return {}
        recent = split.tail(3)
        return {
            "total_cat_loss": float(split["cat_loss"].sum()),
            "total_non_cat_loss": float(split["non_cat_loss"].sum()),
            "avg_annual_cat_loss": float(split["cat_loss"].mean()),
            "avg_cat_loss_ratio": float(split["cat_loss_ratio"].mean()),
            "cat_years_count": int((split["cat_loss"] > 0).sum()),
            "3yr_avg_cat_lr": float(recent["cat_loss_ratio"].mean()),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"CatAnalysis(lob={self.lob!r}, "
            f"total_cat_loss={s.get('total_cat_loss', 0):,.0f}, "
            f"avg_cat_lr={s.get('avg_cat_loss_ratio', 0):.3f})"
        )

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: Optional[str] = None,
    ) -> "CatAnalysis":
        claims = session.loader["claims"]
        vals = session.loader["valuations"]
        policies = session.loader["policies"].copy()
        policies["accident_year"] = policies["effective_date"].dt.year
        ep_col = "earned_premium" if "earned_premium" in policies.columns else "written_premium"
        premiums = policies.rename(columns={ep_col: "earned_premium"})
        return cls(claims=claims, valuations=vals, premiums=premiums, lob=lob)
