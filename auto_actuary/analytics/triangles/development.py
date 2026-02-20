"""
auto_actuary.analytics.triangles.development
=============================================
Loss development triangle construction and LDF selection.

A development triangle organizes historical losses by:
  - Rows  = origin periods (accident year, policy year, or report year)
  - Cols  = development age in months since the start of the origin period

The key actuarial workflow is:
  1. Build the triangle from loss valuation snapshots
  2. Compute LDFs (link ratios) between adjacent development ages
  3. Select representative LDFs using averaging methods
  4. Project each origin period to ultimate using CDF-to-ultimate
  5. Hand off to reserve methods (chain ladder, B-F, Cape Cod)

References
----------
- Friedland, J.F. (2010) "Estimating Unpaid Claims Using Basic Techniques"
- Werner, G. & Modlin, C. (2016) "Basic Ratemaking"  (CAS study note)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from auto_actuary.analytics.triangles import tail as tail_module

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession
    from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LDF averaging methods
# ---------------------------------------------------------------------------

class LDFMethods:
    """Compute individual LDFs between adjacent development ages."""

    @staticmethod
    def individual(from_col: pd.Series, to_col: pd.Series) -> pd.Series:
        """Return element-wise LDF = to / from."""
        mask = from_col.notna() & to_col.notna() & (from_col != 0)
        result = pd.Series(index=from_col.index, dtype=float)
        result[mask] = to_col[mask] / from_col[mask]
        return result

    @staticmethod
    def volume_weighted(from_col: pd.Series, to_col: pd.Series, n_recent: Optional[int] = None) -> float:
        """Volume-weighted (link-ratio) average: sum(to) / sum(from)."""
        mask = from_col.notna() & to_col.notna() & (from_col > 0)
        if n_recent is not None and mask.sum() > n_recent:
            valid_idx = from_col.index[mask][-n_recent:]
            mask = from_col.index.isin(valid_idx)
        if mask.sum() == 0:
            return np.nan
        return float(to_col[mask].sum() / from_col[mask].sum())

    @staticmethod
    def simple_average(from_col: pd.Series, to_col: pd.Series, n_recent: Optional[int] = None) -> float:
        """Simple (arithmetic) average of individual LDFs."""
        indiv = LDFMethods.individual(from_col, to_col).dropna()
        if n_recent is not None and len(indiv) > n_recent:
            indiv = indiv.iloc[-n_recent:]
        if len(indiv) == 0:
            return np.nan
        return float(indiv.mean())

    @staticmethod
    def medial_average(
        from_col: pd.Series,
        to_col: pd.Series,
        n_recent: Optional[int] = None,
        exclude: int = 1,
    ) -> float:
        """Drop *exclude* highest and lowest individual LDFs, then average."""
        indiv = LDFMethods.individual(from_col, to_col).dropna().sort_values()
        if n_recent is not None and len(indiv) > n_recent:
            indiv = indiv.iloc[-n_recent:]
        if len(indiv) <= 2 * exclude:
            return float(indiv.mean()) if len(indiv) > 0 else np.nan
        return float(indiv.iloc[exclude:-exclude].mean())

    @staticmethod
    def geometric_average(from_col: pd.Series, to_col: pd.Series, n_recent: Optional[int] = None) -> float:
        """Geometric average of individual LDFs."""
        indiv = LDFMethods.individual(from_col, to_col).dropna()
        if n_recent is not None and len(indiv) > n_recent:
            indiv = indiv.iloc[-n_recent:]
        if len(indiv) == 0:
            return np.nan
        return float(np.exp(np.log(indiv).mean()))


# ---------------------------------------------------------------------------
# Core triangle class
# ---------------------------------------------------------------------------

class LossTriangle:
    """
    A cumulative (or incremental) development triangle.

    Rows  = origin periods (integer years or period labels)
    Cols  = development age in months (12, 24, 36, …)

    Parameters
    ----------
    triangle : pd.DataFrame
        Pre-pivoted DataFrame with origin periods as the index and
        development ages as columns.
    lob : str
        Line of business code (for labeling).
    value_type : str
        What the values represent (incurred_loss, paid_loss, etc.)
    origin_basis : str
        accident_year | policy_year | report_year
    is_cumulative : bool
        True (default) = cumulative.  False = incremental.
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        lob: str = "",
        value_type: str = "incurred_loss",
        origin_basis: str = "accident_year",
        is_cumulative: bool = True,
    ) -> None:
        self.lob = lob
        self.value_type = value_type
        self.origin_basis = origin_basis
        self.is_cumulative = is_cumulative

        # Sort
        self._tri = triangle.sort_index(axis=0).sort_index(axis=1).astype(float)

        # Will be filled by .develop()
        self._selected_ldfs: Optional[pd.Series] = None
        self._tail_factor: float = 1.0
        self._cdfs: Optional[pd.Series] = None
        self._ultimates: Optional[pd.Series] = None

        # LDF summary table (all methods)
        self._ldf_table: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def triangle(self) -> pd.DataFrame:
        return self._tri

    @property
    def ages(self) -> np.ndarray:
        return self._tri.columns.to_numpy()

    @property
    def origins(self) -> np.ndarray:
        return self._tri.index.to_numpy()

    @property
    def latest_diagonal(self) -> pd.Series:
        """Most recent observed value for each origin period."""
        result = {}
        self._latest_age: Dict[Any, int] = {}
        for origin in self._tri.index:
            row = self._tri.loc[origin].dropna()
            if len(row) > 0:
                result[origin] = row.iloc[-1]
                self._latest_age[origin] = row.index[-1]
            else:
                result[origin] = np.nan
                self._latest_age[origin] = self.ages[0]
        return pd.Series(result, name="latest_diagonal")

    @property
    def n_origins(self) -> int:
        return len(self._tri)

    @property
    def n_ages(self) -> int:
        return len(self._tri.columns)

    # ------------------------------------------------------------------
    # LDF computation
    # ------------------------------------------------------------------

    def compute_all_ldfs(self) -> pd.DataFrame:
        """
        Compute LDFs for all averaging methods and store as self._ldf_table.

        Returns
        -------
        pd.DataFrame
            Index = age step labels ("12-24", "24-36", …)
            Columns = method names
        """
        ages = self.ages
        steps = [f"{ages[i]}-{ages[i+1]}" for i in range(len(ages) - 1)]
        methods: Dict[str, List[float]] = {
            "vw_all": [],
            "vw_5yr": [],
            "vw_3yr": [],
            "sa_all": [],
            "sa_5yr": [],
            "sa_3yr": [],
            "medial_5x1": [],
            "geometric_all": [],
        }

        for i in range(len(ages) - 1):
            fc = self._tri[ages[i]]
            tc = self._tri[ages[i + 1]]
            methods["vw_all"].append(LDFMethods.volume_weighted(fc, tc))
            methods["vw_5yr"].append(LDFMethods.volume_weighted(fc, tc, n_recent=5))
            methods["vw_3yr"].append(LDFMethods.volume_weighted(fc, tc, n_recent=3))
            methods["sa_all"].append(LDFMethods.simple_average(fc, tc))
            methods["sa_5yr"].append(LDFMethods.simple_average(fc, tc, n_recent=5))
            methods["sa_3yr"].append(LDFMethods.simple_average(fc, tc, n_recent=3))
            methods["medial_5x1"].append(LDFMethods.medial_average(fc, tc, n_recent=5, exclude=1))
            methods["geometric_all"].append(LDFMethods.geometric_average(fc, tc))

        self._ldf_table = pd.DataFrame(methods, index=steps)
        return self._ldf_table

    def select_ldfs(
        self,
        method: str = "vw_5yr",
        overrides: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Select LDFs using *method*, with optional manual overrides.

        Parameters
        ----------
        method : str
            Column name from compute_all_ldfs() output, or one of:
            'vw_all' | 'vw_5yr' | 'vw_3yr' | 'sa_all' | 'sa_5yr' |
            'sa_3yr' | 'medial_5x1' | 'geometric_all'
        overrides : dict, optional
            {"12-24": 1.350, "24-36": 1.100}  — override specific steps.

        Returns
        -------
        pd.Series  (index = age step labels, values = selected LDFs)
        """
        if self._ldf_table is None:
            self.compute_all_ldfs()

        if method not in self._ldf_table.columns:
            raise ValueError(
                f"Unknown LDF method '{method}'.  "
                f"Valid: {list(self._ldf_table.columns)}"
            )

        selected = self._ldf_table[method].copy()

        if overrides:
            for step, val in overrides.items():
                if step in selected.index:
                    selected[step] = val
                    logger.info("LDF override: %s = %.4f", step, val)

        self._selected_ldfs = selected
        return selected

    # ------------------------------------------------------------------
    # Tail factor
    # ------------------------------------------------------------------

    def fit_tail(
        self,
        method: str = "curve_fit",
        curve: str = "inverse_power",
        threshold: float = 1.005,
        user_tail: float = 1.0,
    ) -> float:
        """
        Fit or set the tail factor (development from last observed age to ultimate).

        Parameters
        ----------
        method : str
            'curve_fit' | 'user_specified'
        curve : str
            'inverse_power' | 'exponential'  (used when method='curve_fit')
        threshold : float
            If fitted tail < threshold, set to 1.000.
        user_tail : float
            Used when method='user_specified'.
        """
        if method == "user_specified":
            self._tail_factor = user_tail
            return user_tail

        if self._ldf_table is None:
            self.compute_all_ldfs()

        # Use all-years volume-weighted LDFs for tail fitting
        ldfs = self._ldf_table["vw_all"].dropna()
        ages_mid = self.ages[:-1] + np.diff(self.ages) / 2  # midpoint of each step
        ages_mid = ages_mid[: len(ldfs)]

        tail = tail_module.fit_tail(ages_mid, ldfs.values, curve=curve, threshold=threshold)
        self._tail_factor = tail
        logger.info("Tail factor fitted (%s, %s): %.5f", method, curve, tail)
        return tail

    # ------------------------------------------------------------------
    # CDF-to-ultimate and ultimates
    # ------------------------------------------------------------------

    def compute_cdfs(self) -> pd.Series:
        """
        Compute cumulative development factors (CDF) to ultimate for each age.

        CDF(age) = product of selected LDFs from *age* forward × tail factor.
        """
        if self._selected_ldfs is None:
            self.select_ldfs()

        ldfs = self._selected_ldfs.values.astype(float)
        tail = self._tail_factor
        n = len(ldfs)
        cdfs = np.ones(n + 1)

        # CDF for the last observed age = tail
        cdfs[n] = tail
        for i in range(n - 1, -1, -1):
            cdfs[i] = ldfs[i] * cdfs[i + 1]

        self._cdfs = pd.Series(cdfs, index=list(self.ages) + ["tail"])
        return self._cdfs

    def ultimates(self, method: str = "chain_ladder") -> pd.Series:
        """
        Estimate ultimate losses using chain ladder (latest × CDF-to-ult).

        For B-F and Cape Cod, use the ReserveAnalysis class.
        """
        if self._cdfs is None:
            self.compute_cdfs()

        diag = self.latest_diagonal
        result = {}
        for origin in self.origins:
            lat_age = self._latest_age.get(origin, self.ages[0])
            if lat_age in self._cdfs.index:
                cdf = self._cdfs[lat_age]
            else:
                cdf = self._tail_factor
            result[origin] = diag[origin] * cdf

        self._ultimates = pd.Series(result, name="chain_ladder_ultimate")
        return self._ultimates

    def ibnr(self) -> pd.Series:
        """IBNR = Ultimate − Latest Diagonal (chain ladder)."""
        if self._ultimates is None:
            self.ultimates()
        return (self._ultimates - self.latest_diagonal).rename("ibnr_chain_ladder")

    # ------------------------------------------------------------------
    # Master workflow
    # ------------------------------------------------------------------

    def develop(
        self,
        ldf_method: str = "vw_5yr",
        ldf_overrides: Optional[Dict[str, float]] = None,
        tail_method: str = "curve_fit",
        tail_curve: str = "inverse_power",
        tail_threshold: float = 1.005,
        user_tail: float = 1.0,
    ) -> "LossTriangle":
        """
        Full development workflow: compute LDFs → select → fit tail → CDFs.

        Returns self for chaining.
        """
        self.compute_all_ldfs()
        self.select_ldfs(method=ldf_method, overrides=ldf_overrides)
        self.fit_tail(
            method=tail_method,
            curve=tail_curve,
            threshold=tail_threshold,
            user_tail=user_tail,
        )
        self.compute_cdfs()
        return self

    # ------------------------------------------------------------------
    # Diagnostics and display
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Return a summary table:
        origin | latest_age | latest_diag | cdf_to_ult | ultimate | ibnr | %_unreported
        """
        if self._cdfs is None:
            self.compute_cdfs()
        if self._ultimates is None:
            self.ultimates()

        rows = []
        diag = self.latest_diagonal
        for origin in self.origins:
            lat_age = self._latest_age.get(origin, self.ages[0])
            cdf = self._cdfs.get(lat_age, self._tail_factor)
            ult = self._ultimates[origin]
            rep = diag[origin]
            ibnr = ult - rep
            pct_unreported = ibnr / ult if ult != 0 else np.nan
            rows.append(
                {
                    "origin": origin,
                    "latest_age": lat_age,
                    "reported": rep,
                    "cdf_to_ult": cdf,
                    "ultimate": ult,
                    "ibnr": ibnr,
                    "pct_unreported": pct_unreported,
                }
            )
        return pd.DataFrame(rows).set_index("origin")

    def ldf_exhibit(self) -> pd.DataFrame:
        """Return the full LDF averaging table with selected row appended."""
        if self._ldf_table is None:
            self.compute_all_ldfs()
        exhibit = self._ldf_table.copy()
        if self._selected_ldfs is not None:
            exhibit["SELECTED"] = self._selected_ldfs
        return exhibit

    def to_incremental(self) -> "LossTriangle":
        """Return a new LossTriangle with incremental (period) values."""
        inc = self._tri.diff(axis=1)
        inc.iloc[:, 0] = self._tri.iloc[:, 0]  # first age is already incremental
        return LossTriangle(
            inc,
            lob=self.lob,
            value_type=self.value_type,
            origin_basis=self.origin_basis,
            is_cumulative=False,
        )

    def __repr__(self) -> str:
        return (
            f"LossTriangle(lob={self.lob!r}, value={self.value_type!r}, "
            f"origins={self.n_origins}, ages={self.n_ages})"
        )


# ---------------------------------------------------------------------------
# Factory: build from session
# ---------------------------------------------------------------------------

def build_triangle_from_session(
    session: "ActuarySession",
    lob: str,
    value: str = "incurred_loss",
    origin_basis: str = "accident_year",
    dev_step_months: int = 12,
    coverage: Optional[str] = None,
) -> LossTriangle:
    """
    Build a LossTriangle from loaded session data.

    Requires:
        session.data("valuations") — claim-level loss snapshots
        session.data("claims")     — accident_date, lob, coverage fields

    The valuation data must have (after schema mapping):
        claim_id | valuation_date | <value column>

    The claims data must have:
        claim_id | accident_date | line_of_business
    """
    claims = session.loader["claims"].copy()
    vals = session.loader["valuations"].copy()

    # Filter by LOB
    claims_lob = claims[claims["line_of_business"] == lob].copy()
    if coverage:
        claims_lob = claims_lob[claims_lob["coverage_code"] == coverage]

    if claims_lob.empty:
        raise ValueError(f"No claims found for LOB='{lob}'" + (f", coverage='{coverage}'" if coverage else ""))

    # Join valuations to claims to get accident_date
    merged = vals.merge(
        claims_lob[["claim_id", "accident_date", "line_of_business"]],
        on="claim_id",
        how="inner",
    )

    if merged.empty:
        raise ValueError(f"No valuation data matched for LOB='{lob}'")

    # Determine origin period
    if origin_basis == "accident_year":
        merged["origin"] = merged["accident_date"].dt.year
    elif origin_basis == "policy_year":
        # Would need policy effective date; use accident_year as fallback
        merged["origin"] = merged["accident_date"].dt.year
        logger.warning("policy_year basis not yet implemented — using accident_year")
    elif origin_basis == "report_year":
        if "report_date" in merged.columns:
            merged["origin"] = merged["report_date"].dt.year
        else:
            merged["origin"] = merged["accident_date"].dt.year
    else:
        raise ValueError(f"Unknown origin_basis: {origin_basis!r}")

    # Determine development age
    # Origin start = Jan 1 of the origin year
    merged["origin_start"] = pd.to_datetime(merged["origin"].astype(str) + "-01-01")
    merged["dev_months_raw"] = (
        (merged["valuation_date"].dt.year - merged["origin_start"].dt.year) * 12
        + (merged["valuation_date"].dt.month - merged["origin_start"].dt.month)
    ).clip(lower=1)

    # Round to nearest dev_step_months
    merged["dev_age"] = (
        ((merged["dev_months_raw"] / dev_step_months).round() * dev_step_months)
        .astype(int)
        .clip(lower=dev_step_months)
    )

    if value not in merged.columns:
        raise ValueError(
            f"Value column '{value}' not found after joining claims + valuations. "
            f"Available: {list(merged.columns)}"
        )

    # Aggregate (sum) by origin × dev_age
    agg = (
        merged.groupby(["origin", "dev_age"])[value]
        .sum()
        .reset_index()
    )

    # Pivot
    tri_df = agg.pivot_table(index="origin", columns="dev_age", values=value, aggfunc="sum")
    tri_df = tri_df.sort_index(axis=0).sort_index(axis=1)

    return LossTriangle(
        triangle=tri_df,
        lob=lob,
        value_type=value,
        origin_basis=origin_basis,
        is_cumulative=True,
    )
