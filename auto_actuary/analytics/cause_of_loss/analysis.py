"""
auto_actuary.analytics.cause_of_loss.analysis
==============================================
Cause-of-loss (peril) analysis and cross-dimension correlations.

Insurance losses arise from distinct causes: wind, hail, water damage,
collision, theft, fire, medical, etc.  Understanding the cause mix and its
evolution over time is essential for:

  - Peril-specific pricing and underwriting
  - Reinsurance treaty design (cat vs. non-cat split)
  - Reserve adequacy by cause
  - Trend monitoring (e.g., rising water damage frequency)

Data requirement
----------------
The ``claims`` table must contain a ``cause_code`` column (canonical alias for
the ``cause_of_loss`` field mapped in schema.yaml).

CauseOfLossAnalysis
    Summary: frequency, severity, pure premium, loss ratio by cause code.
    Supports slicing by LOB, territory, coverage, and accident year.

CauseOfLossCorrelation
    Quantifies the statistical association between cause codes and other
    categorical dimensions (territory, class_code, coverage_code).
    Uses Cramér's V as a symmetric, bounded (0–1) measure of association.

References
----------
- ISO cause-of-loss codes (industry standard classification)
- CAS "Casualty Actuarial Society Statement of Principles" on cause of loss
- Cramér, H. (1946) "Mathematical Methods of Statistics"
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CauseOfLossAnalysis
# ---------------------------------------------------------------------------

class CauseOfLossAnalysis:
    """
    Frequency / severity / pure-premium breakdown by cause of loss.

    Parameters
    ----------
    claims : pd.DataFrame
        Columns: claim_id | policy_id | accident_date | cause_code |
                 line_of_business | territory | coverage_code |
                 claim_status | is_catastrophe (optional)
    valuations : pd.DataFrame
        Columns: claim_id | valuation_date | incurred_loss | paid_loss |
                 paid_alae | case_alae (optional)
    policies : pd.DataFrame
        Columns: policy_id | written_premium | written_exposure |
                 line_of_business | effective_date
    lob : str, optional
        Filter to a single line of business.
    exclude_cat : bool
        Exclude catastrophe claims from the analysis (default False).
    """

    def __init__(
        self,
        claims: pd.DataFrame,
        valuations: pd.DataFrame,
        policies: pd.DataFrame,
        lob: Optional[str] = None,
        exclude_cat: bool = False,
    ) -> None:
        self.lob = lob
        self.exclude_cat = exclude_cat

        # Filter LOB
        filt_lob = lambda df: (
            df[df["line_of_business"] == lob].copy()
            if lob and "line_of_business" in df.columns else df.copy()
        )
        self._claims = filt_lob(claims)
        self._policies = filt_lob(policies)
        self._vals = valuations.copy()

        if exclude_cat and "is_catastrophe" in self._claims.columns:
            self._claims = self._claims[self._claims["is_catastrophe"] != 1]

        # Enrich claims with accident_year
        if "accident_date" in self._claims.columns:
            self._claims["accident_year"] = self._claims["accident_date"].dt.year

        # Join latest valuation → claims
        self._merged = self._join_valuations()

        # Exposure denominator from policies
        self._total_exposure = (
            float(self._policies["written_exposure"].sum())
            if "written_exposure" in self._policies.columns else np.nan
        )

    def _join_valuations(self) -> pd.DataFrame:
        """Return claims with incurred_loss, paid_loss, paid_alae from latest valuation."""
        if self._vals.empty or "valuation_date" not in self._vals.columns:
            merged = self._claims.copy()
            for col in ["incurred_loss", "paid_loss", "paid_alae", "case_alae"]:
                if col not in merged.columns:
                    merged[col] = 0.0
            return merged

        loss_cols = ["claim_id"]
        for col in ["incurred_loss", "paid_loss", "paid_alae", "case_alae"]:
            if col in self._vals.columns:
                loss_cols.append(col)

        latest = (
            self._vals.sort_values("valuation_date")
            .groupby("claim_id")
            .last()
            .reset_index()[loss_cols]
        )

        merged = self._claims.merge(latest, on="claim_id", how="left")
        for col in ["incurred_loss", "paid_loss", "paid_alae", "case_alae"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        return merged

    # ------------------------------------------------------------------
    # Core summary table
    # ------------------------------------------------------------------

    def by_cause(
        self,
        extra_dims: Optional[List[str]] = None,
        accident_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Loss summary by cause_code.

        Parameters
        ----------
        extra_dims : list of str, optional
            Additional grouping columns, e.g. ['territory', 'coverage_code'].
        accident_year : int, optional
            Filter to a single accident year.

        Returns
        -------
        pd.DataFrame
            Index: cause_code (+ any extra_dims)
            Columns: claim_count | incurred_loss | paid_loss | paid_alae |
                     severity | pct_of_total_loss | cumulative_pct
        """
        df = self._merged.copy()
        if accident_year:
            df = df[df["accident_year"] == accident_year]

        group_cols = ["cause_code"] + (extra_dims or [])
        group_cols = [c for c in group_cols if c in df.columns]

        if "cause_code" not in df.columns:
            logger.warning("CauseOfLossAnalysis: 'cause_code' not in claims — "
                           "check schema.yaml cause_of_loss mapping.")
            return pd.DataFrame()

        agg_dict: Dict[str, str] = {"claim_count": "sum"}
        df["claim_count"] = 1
        for col in ["incurred_loss", "paid_loss", "paid_alae", "case_alae"]:
            if col in df.columns:
                agg_dict[col] = "sum"

        result = df.groupby(group_cols).agg(**{k: pd.NamedAgg(column=k, aggfunc=v)
                                               for k, v in agg_dict.items()}).reset_index()

        # Derived metrics
        cnt = result["claim_count"].replace(0, np.nan)
        if "incurred_loss" in result.columns:
            result["severity"] = result["incurred_loss"] / cnt
            total_loss = result["incurred_loss"].sum()
            result["pct_of_total_loss"] = result["incurred_loss"] / total_loss if total_loss else np.nan
            result = result.sort_values("incurred_loss", ascending=False)
            result["cumulative_pct"] = result["pct_of_total_loss"].cumsum()

        if not np.isnan(self._total_exposure):
            result["frequency"] = result["claim_count"] / self._total_exposure
            if "incurred_loss" in result.columns:
                result["pure_premium"] = result["incurred_loss"] / self._total_exposure

        return result.set_index(group_cols)

    # ------------------------------------------------------------------
    # Trend by cause
    # ------------------------------------------------------------------

    def trend_by_cause(
        self,
        cause: Optional[str] = None,
        metric: str = "incurred_loss",
    ) -> pd.DataFrame:
        """
        Annual time series of *metric* for each cause (or a single cause).

        Parameters
        ----------
        cause : str, optional
            Specific cause_code to return.  None = all causes.
        metric : str
            One of: incurred_loss | paid_loss | claim_count | severity | frequency

        Returns
        -------
        pd.DataFrame
            Columns: accident_year | cause_code | <metric>
            Pivotable for trend charting.
        """
        df = self._merged.copy()
        if cause:
            df = df[df.get("cause_code", pd.Series()) == cause] if "cause_code" in df.columns else df

        if "accident_year" not in df.columns or "cause_code" not in df.columns:
            return pd.DataFrame()

        df["claim_count"] = 1
        agg_cols = ["claim_count"]
        for col in ["incurred_loss", "paid_loss"]:
            if col in df.columns:
                agg_cols.append(col)

        trend = df.groupby(["accident_year", "cause_code"])[agg_cols].sum().reset_index()

        if "incurred_loss" in trend.columns:
            trend["severity"] = trend["incurred_loss"] / trend["claim_count"].replace(0, np.nan)

        if metric not in trend.columns:
            logger.warning("CauseOfLossAnalysis.trend_by_cause: metric '%s' not available", metric)
            return trend

        return trend[["accident_year", "cause_code", metric]]

    # ------------------------------------------------------------------
    # Cat vs. non-cat split
    # ------------------------------------------------------------------

    def cat_noncat_split(self) -> pd.DataFrame:
        """
        Split losses into catastrophe and non-catastrophe by cause_code.

        Returns
        -------
        pd.DataFrame
            Columns: cause_code | cat_loss | noncat_loss | cat_pct
        """
        if "is_catastrophe" not in self._merged.columns:
            logger.warning("CauseOfLossAnalysis: 'is_catastrophe' not in claims — "
                           "cannot split cat/non-cat.")
            return pd.DataFrame()

        df = self._merged.copy()
        cat = df[df["is_catastrophe"] == 1].groupby("cause_code")["incurred_loss"].sum()
        noncat = df[df["is_catastrophe"] != 1].groupby("cause_code")["incurred_loss"].sum()
        result = pd.DataFrame({"cat_loss": cat, "noncat_loss": noncat}).fillna(0)
        total = (result["cat_loss"] + result["noncat_loss"]).replace(0, np.nan)
        result["cat_pct"] = result["cat_loss"] / total
        return result.sort_values("cat_loss", ascending=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        """Scalar summary metrics."""
        df = self._merged.copy()
        total_claims = int(len(df))
        total_loss = float(df.get("incurred_loss", pd.Series([0])).sum())
        n_causes = int(df["cause_code"].nunique()) if "cause_code" in df.columns else 0

        top_cause = ""
        if "cause_code" in df.columns and "incurred_loss" in df.columns:
            by_cause = df.groupby("cause_code")["incurred_loss"].sum()
            top_cause = str(by_cause.idxmax()) if not by_cause.empty else ""

        return {
            "total_claims": total_claims,
            "total_incurred_loss": total_loss,
            "n_distinct_causes": n_causes,
            "top_cause_by_loss": top_cause,
            "total_exposure": self._total_exposure,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"CauseOfLossAnalysis(lob={self.lob!r}, "
            f"claims={s['total_claims']}, "
            f"causes={s['n_distinct_causes']}, "
            f"top_cause={s['top_cause_by_loss']!r})"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: Optional[str] = None,
        **kwargs,
    ) -> "CauseOfLossAnalysis":
        """Build from a loaded ActuarySession."""
        claims = session.loader["claims"] if "claims" in session.loader.loaded_tables else pd.DataFrame()
        vals = session.loader["valuations"] if "valuations" in session.loader.loaded_tables else pd.DataFrame()
        policies = session.loader["policies"] if "policies" in session.loader.loaded_tables else pd.DataFrame()
        return cls(claims=claims, valuations=vals, policies=policies, lob=lob, **kwargs)


# ---------------------------------------------------------------------------
# CauseOfLossCorrelation
# ---------------------------------------------------------------------------

class CauseOfLossCorrelation:
    """
    Statistical association between cause of loss and other categorical factors.

    Uses Cramér's V, a normalised chi-squared statistic:

        V = sqrt(chi² / (n × (min(r,c) − 1)))

    where r = # rows (cause levels), c = # cols (other variable levels).

    V ∈ [0, 1]:  0 = no association,  1 = perfect association.

    Parameters
    ----------
    claims : pd.DataFrame
        Claims data with cause_code and any other categorical columns.
    lob : str, optional
        Filter to a single LOB.
    """

    def __init__(
        self,
        claims: pd.DataFrame,
        lob: Optional[str] = None,
    ) -> None:
        self.lob = lob
        df = claims.copy()
        if lob and "line_of_business" in df.columns:
            df = df[df["line_of_business"] == lob]
        self._claims = df

    @staticmethod
    def _cramers_v(contingency: pd.DataFrame) -> float:
        """Compute Cramér's V from a contingency table."""
        ct = contingency.values
        n = ct.sum()
        if n == 0:
            return np.nan
        chi2, p, dof, expected = scipy_stats.chi2_contingency(ct, correction=False)
        min_dim = min(ct.shape) - 1
        if min_dim <= 0:
            return np.nan
        v = float(np.sqrt(chi2 / (n * min_dim)))
        return min(v, 1.0)

    def pairwise_associations(
        self,
        dimensions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute Cramér's V between cause_code and each dimension.

        Parameters
        ----------
        dimensions : list of str, optional
            Columns to test against cause_code.
            Default: ['territory', 'coverage_code', 'class_code', 'line_of_business'].

        Returns
        -------
        pd.DataFrame
            Columns: dimension | cramers_v | chi2_pvalue | interpretation
            Sorted by association strength (descending).
        """
        dims = dimensions or ["territory", "coverage_code", "class_code", "line_of_business"]
        dims = [d for d in dims if d in self._claims.columns and d != "cause_code"]

        if "cause_code" not in self._claims.columns:
            logger.warning("CauseOfLossCorrelation: 'cause_code' not in claims.")
            return pd.DataFrame()

        rows = []
        for dim in dims:
            sub = self._claims[["cause_code", dim]].dropna()
            if sub.empty:
                continue
            ct = pd.crosstab(sub["cause_code"], sub[dim])
            v = self._cramers_v(ct)
            chi2_stat, p_val, _, _ = scipy_stats.chi2_contingency(ct.values, correction=False)
            rows.append({
                "dimension": dim,
                "cramers_v": round(v, 4),
                "chi2_pvalue": round(float(p_val), 6),
                "n_obs": int(len(sub)),
                "interpretation": _interpret_v(v),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("cramers_v", ascending=False)
            .set_index("dimension")
        )

    def heatmap_data(
        self,
        dimension: str,
        metric: str = "incurred_loss",
        valuations: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Pivot table: cause_code × dimension-value = total metric.

        Suitable for rendering as a heatmap to spot concentration patterns.

        Parameters
        ----------
        dimension : str
            Column to cross against cause_code (e.g. 'territory').
        metric : str
            Metric column to aggregate (default 'incurred_loss').
            If 'claim_count', counts claims instead.
        valuations : pd.DataFrame, optional
            If metric is 'incurred_loss', pass the valuations table to join losses.

        Returns
        -------
        pd.DataFrame
            Rows = cause_code, Columns = dimension values.
        """
        df = self._claims.copy()

        if metric == "claim_count":
            df["claim_count"] = 1
            pivot = df.groupby(["cause_code", dimension])["claim_count"].sum().unstack(fill_value=0)
        elif metric == "incurred_loss" and valuations is not None:
            latest = (
                valuations.sort_values("valuation_date")
                .groupby("claim_id").last().reset_index()[["claim_id", "incurred_loss"]]
            ) if "valuation_date" in valuations.columns else valuations[["claim_id", "incurred_loss"]]
            df = df.merge(latest, on="claim_id", how="left")
            df["incurred_loss"] = df["incurred_loss"].fillna(0)
            pivot = df.groupby(["cause_code", dimension])["incurred_loss"].sum().unstack(fill_value=0)
        else:
            if metric not in df.columns:
                raise ValueError(f"Metric '{metric}' not in claims DataFrame.")
            pivot = df.groupby(["cause_code", dimension])[metric].sum().unstack(fill_value=0)

        return pivot

    def top_causes_by_dimension(
        self,
        dimension: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        For each value of *dimension*, show the top N causes by claim count.

        Returns
        -------
        pd.DataFrame
            Columns: <dimension> | rank | cause_code | claim_count | pct_of_segment
        """
        if "cause_code" not in self._claims.columns or dimension not in self._claims.columns:
            return pd.DataFrame()

        df = self._claims.copy()
        df["claim_count"] = 1
        grp = df.groupby([dimension, "cause_code"])["claim_count"].sum().reset_index()
        grp["rank"] = grp.groupby(dimension)["claim_count"].rank(method="first", ascending=False)

        seg_totals = grp.groupby(dimension)["claim_count"].sum()
        grp = grp.merge(seg_totals.rename("seg_total"), on=dimension)
        grp["pct_of_segment"] = grp["claim_count"] / grp["seg_total"]

        return (
            grp[grp["rank"] <= top_n]
            .sort_values([dimension, "rank"])
            .drop(columns=["seg_total"])
            .set_index([dimension, "rank"])
        )

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: Optional[str] = None,
    ) -> "CauseOfLossCorrelation":
        """Build from a loaded ActuarySession."""
        claims = session.loader["claims"] if "claims" in session.loader.loaded_tables else pd.DataFrame()
        return cls(claims=claims, lob=lob)


def _interpret_v(v: float) -> str:
    """Qualitative label for Cramér's V."""
    if np.isnan(v):
        return "N/A"
    if v < 0.10:
        return "negligible"
    if v < 0.20:
        return "weak"
    if v < 0.40:
        return "moderate"
    if v < 0.60:
        return "strong"
    return "very strong"
