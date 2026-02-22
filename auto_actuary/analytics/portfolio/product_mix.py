"""
auto_actuary.analytics.portfolio.product_mix
=============================================
Product mix and portfolio composition analysis.

Portfolio mix matters to actuaries for two reasons:
  1. **Pricing adequacy**: the aggregate loss ratio is a weighted average of
     segment loss ratios; if the mix drifts toward worse-performing segments,
     the overall book deteriorates even if individual segment rates are adequate.
  2. **Diversification**: a well-diversified mix of LOBs and geographies reduces
     volatility relative to a concentrated portfolio.

This module quantifies:
  - Proportion of premium, exposure, and claims by LOB, coverage, territory,
    class_code, and any custom dimension
  - Mix shift over time (how composition has changed year-over-year)
  - Herfindahl–Hirschman Index (HHI) as a portfolio concentration metric
  - Profitability by mix segment (loss ratio per slice)

References
----------
- Werner & Modlin (2016) "Basic Ratemaking" Ch. 12 (mix of business)
- Herfindahl, O. (1950) "Concentration in the Steel Industry"
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession

logger = logging.getLogger(__name__)


class ProductMixAnalysis:
    """
    Portfolio product-mix and concentration analytics.

    Parameters
    ----------
    policies : pd.DataFrame
        Must contain: policy_id | written_premium | written_exposure |
                      effective_date | line_of_business.
        Optional: territory | class_code | sub_line | coverage_code
    claims : pd.DataFrame, optional
        Must contain: policy_id | claim_id | accident_date | line_of_business
    valuations : pd.DataFrame, optional
        Must contain: claim_id | incurred_loss
    lob : str, optional
        Filter to a single LOB.
    """

    def __init__(
        self,
        policies: pd.DataFrame,
        claims: Optional[pd.DataFrame] = None,
        valuations: Optional[pd.DataFrame] = None,
        lob: Optional[str] = None,
    ) -> None:
        self.lob = lob
        pol = policies.copy()
        if lob and "line_of_business" in pol.columns:
            pol = pol[pol["line_of_business"] == lob]
        self._policies = pol

        clm = claims.copy() if claims is not None else pd.DataFrame()
        if lob and not clm.empty and "line_of_business" in clm.columns:
            clm = clm[clm["line_of_business"] == lob]
        self._claims = clm
        self._vals = valuations.copy() if valuations is not None else pd.DataFrame()

        if "effective_date" in self._policies.columns:
            self._policies["policy_year"] = pd.to_datetime(
                self._policies["effective_date"], errors="coerce"
            ).dt.year

        self._loss_by_policy: pd.DataFrame = self._build_loss_by_policy()

    def _build_loss_by_policy(self) -> pd.DataFrame:
        if self._claims.empty or self._vals.empty:
            return pd.DataFrame(columns=["policy_id", "incurred_loss", "claim_count"])

        loss_cols = [c for c in ["claim_id", "incurred_loss"] if c in self._vals.columns]
        if "valuation_date" in self._vals.columns:
            latest = (
                self._vals.sort_values("valuation_date")
                .groupby("claim_id").last().reset_index()[loss_cols]
            )
        else:
            latest = self._vals[loss_cols].copy()

        if "policy_id" not in self._claims.columns:
            return pd.DataFrame(columns=["policy_id", "incurred_loss", "claim_count"])

        merged = self._claims[["claim_id", "policy_id"]].merge(latest, on="claim_id", how="left")
        merged["incurred_loss"] = merged["incurred_loss"].fillna(0)
        merged["claim_count"] = 1
        return merged.groupby("policy_id").agg(
            incurred_loss=("incurred_loss", "sum"),
            claim_count=("claim_count", "sum"),
        ).reset_index()

    # ------------------------------------------------------------------
    # Mix tables
    # ------------------------------------------------------------------

    def mix_by(
        self,
        dimension: str,
        year: Optional[int] = None,
        metric: str = "written_premium",
    ) -> pd.DataFrame:
        """
        Portfolio composition breakdown by *dimension*.

        Parameters
        ----------
        dimension : str
            Column to group by: 'line_of_business', 'territory', 'class_code',
            'sub_line', 'coverage_code', or any column in policies.
        year : int, optional
            Filter to a single policy_year.
        metric : str
            'written_premium' | 'written_exposure' | 'policy_count'

        Returns
        -------
        pd.DataFrame
            Columns: <dimension> | <metric> | pct_of_total | cumulative_pct
        """
        pol = self._policies.copy()
        if year is not None and "policy_year" in pol.columns:
            pol = pol[pol["policy_year"] == year]

        if dimension not in pol.columns:
            raise ValueError(f"Column '{dimension}' not found in policies DataFrame.")

        if metric == "policy_count":
            pol["policy_count"] = 1
            grp = pol.groupby(dimension)["policy_count"].sum()
        elif metric in pol.columns:
            grp = pol.groupby(dimension)[metric].sum()
        else:
            raise ValueError(f"Metric '{metric}' not found in policies DataFrame.")

        df = grp.to_frame(name=metric).sort_values(metric, ascending=False)
        total = df[metric].sum()
        df["pct_of_total"] = df[metric] / total if total > 0 else np.nan
        df["cumulative_pct"] = df["pct_of_total"].cumsum()
        return df

    def mix_with_loss_ratio(
        self,
        dimension: str,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Mix breakdown with loss ratio per segment.

        Returns
        -------
        pd.DataFrame
            Columns: <dimension> | written_premium | pct_of_total |
                     incurred_loss | claim_count | loss_ratio | frequency | severity
        """
        pol = self._policies.copy()
        if year is not None and "policy_year" in pol.columns:
            pol = pol[pol["policy_year"] == year]

        if dimension not in pol.columns:
            raise ValueError(f"Column '{dimension}' not found in policies DataFrame.")

        prem_grp = pol.groupby(dimension).agg(
            written_premium=("written_premium", "sum") if "written_premium" in pol.columns else ("policy_id", "count"),
            written_exposure=("written_exposure", "sum") if "written_exposure" in pol.columns else ("policy_id", "count"),
            policy_count=("policy_id", "count"),
        )

        if not self._loss_by_policy.empty:
            merged = pol[["policy_id", dimension]].merge(self._loss_by_policy, on="policy_id", how="left")
            merged["incurred_loss"] = merged["incurred_loss"].fillna(0)
            merged["claim_count"] = merged["claim_count"].fillna(0)
            loss_grp = merged.groupby(dimension).agg(
                incurred_loss=("incurred_loss", "sum"),
                claim_count=("claim_count", "sum"),
            )
            result = prem_grp.join(loss_grp, how="left").fillna(0)
        else:
            result = prem_grp.copy()
            result["incurred_loss"] = 0.0
            result["claim_count"] = 0

        total_prem = float(result["written_premium"].sum()) if "written_premium" in result.columns else 1.0
        result["pct_of_total"] = result["written_premium"] / total_prem if total_prem > 0 else np.nan
        ep = result["written_premium"].replace(0, np.nan)
        result["loss_ratio"] = result["incurred_loss"] / ep

        exp = result["written_exposure"].replace(0, np.nan) if "written_exposure" in result.columns else np.nan
        cnt = result["claim_count"].replace(0, np.nan)
        result["frequency"] = result["claim_count"] / exp
        result["severity"] = result["incurred_loss"] / cnt

        return result.sort_values("written_premium", ascending=False)

    # ------------------------------------------------------------------
    # Mix shift (year-over-year)
    # ------------------------------------------------------------------

    def mix_shift(
        self,
        dimension: str,
        metric: str = "written_premium",
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Compute year-over-year mix shift for *dimension*.

        Parameters
        ----------
        dimension : str
            Column to track (e.g. 'line_of_business', 'territory').
        metric : str
            What to measure ('written_premium', 'written_exposure', 'policy_count').
        years : list of int, optional
            Subset of policy years to show.  None = all years.

        Returns
        -------
        pd.DataFrame
            Pivot: rows = policy_year, columns = dimension values, values = pct_of_total.
        """
        pol = self._policies.copy()
        if years:
            pol = pol[pol["policy_year"].isin(years)]

        if "policy_year" not in pol.columns or dimension not in pol.columns:
            return pd.DataFrame()

        if metric == "policy_count":
            pol["policy_count"] = 1
            raw = pol.groupby(["policy_year", dimension])["policy_count"].sum().reset_index()
            val_col = "policy_count"
        elif metric in pol.columns:
            raw = pol.groupby(["policy_year", dimension])[metric].sum().reset_index()
            val_col = metric
        else:
            raise ValueError(f"Metric '{metric}' not in policies.")

        # Normalize to % within each year
        totals = raw.groupby("policy_year")[val_col].sum()
        raw = raw.merge(totals.rename("year_total"), on="policy_year")
        raw["pct"] = raw[val_col] / raw["year_total"].replace(0, np.nan)

        pivot = raw.pivot_table(index="policy_year", columns=dimension, values="pct", fill_value=0)
        return pivot

    # ------------------------------------------------------------------
    # Concentration (HHI)
    # ------------------------------------------------------------------

    def herfindahl_index(
        self,
        dimension: str = "territory",
        year: Optional[int] = None,
        metric: str = "written_premium",
    ) -> float:
        """
        Herfindahl–Hirschman Index (HHI) of portfolio concentration.

        HHI = Σ (market_share_i²)

        where market_share_i is the fraction of total *metric* in segment i.

        HHI ∈ (0, 1]:
          - Near 0   = highly diversified
          - 1        = completely concentrated in one segment
          - > 0.25   = highly concentrated (DoJ antitrust standard)

        Parameters
        ----------
        dimension : str
            Column to compute concentration over.
        year : int, optional
            Filter to a single policy year.
        metric : str
            Measure to compute shares from.

        Returns
        -------
        float
            HHI value.
        """
        mix = self.mix_by(dimension=dimension, year=year, metric=metric)
        shares = mix["pct_of_total"].dropna()
        if shares.empty:
            return np.nan
        return float((shares ** 2).sum())

    def concentration_summary(
        self,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        HHI for all available dimensions.

        Returns
        -------
        pd.DataFrame
            Columns: dimension | hhi | interpretation
        """
        dims = [d for d in ["line_of_business", "territory", "class_code", "sub_line"]
                if d in self._policies.columns]
        rows = []
        for dim in dims:
            hhi = self.herfindahl_index(dimension=dim, year=year)
            rows.append({
                "dimension": dim,
                "hhi": round(hhi, 4),
                "interpretation": _interpret_hhi(hhi),
            })
        return pd.DataFrame(rows).set_index("dimension")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, year: Optional[int] = None) -> Dict[str, object]:
        """Scalar portfolio summary."""
        pol = self._policies.copy()
        if year and "policy_year" in pol.columns:
            pol = pol[pol["policy_year"] == year]

        total_premium = float(pol["written_premium"].sum()) if "written_premium" in pol.columns else 0.0
        total_exposure = float(pol["written_exposure"].sum()) if "written_exposure" in pol.columns else 0.0
        n_lobs = int(pol["line_of_business"].nunique()) if "line_of_business" in pol.columns else 0
        top_lob = str(pol.groupby("line_of_business")["written_premium"].sum().idxmax()) \
            if "line_of_business" in pol.columns and total_premium > 0 else ""

        return {
            "total_written_premium": total_premium,
            "total_written_exposure": total_exposure,
            "total_policies": int(len(pol)),
            "n_lines_of_business": n_lobs,
            "top_lob": top_lob,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ProductMixAnalysis(lob={self.lob!r}, "
            f"policies={s['total_policies']}, "
            f"lobs={s['n_lines_of_business']}, "
            f"top_lob={s['top_lob']!r})"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: Optional[str] = None,
    ) -> "ProductMixAnalysis":
        """Build from a loaded ActuarySession."""
        policies = session.loader["policies"] if "policies" in session.loader.loaded_tables else pd.DataFrame()
        claims = session.loader["claims"] if "claims" in session.loader.loaded_tables else None
        vals = session.loader["valuations"] if "valuations" in session.loader.loaded_tables else None
        return cls(policies=policies, claims=claims, valuations=vals, lob=lob)


def _interpret_hhi(hhi: float) -> str:
    if np.isnan(hhi):
        return "N/A"
    if hhi < 0.10:
        return "unconcentrated"
    if hhi < 0.18:
        return "moderately concentrated"
    if hhi < 0.25:
        return "concentrated"
    return "highly concentrated"
