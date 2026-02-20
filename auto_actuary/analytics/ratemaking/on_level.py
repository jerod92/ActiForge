"""
auto_actuary.analytics.ratemaking.on_level
===========================================
On-level premium adjustment — restate historical written premiums to what
they would have been under current rates.

The Parallelogram Method
------------------------
The parallelogram method is the standard FCAS approach for adjusting
historical premium to current rate level.

1. Track cumulative rate index relative to a base period.
2. For each historical policy year, determine what fraction of earned
   premium was written before/after each rate change (using the area
   of parallelograms under the rate-change timeline).
3. Compute an average rate level factor (ARLF) for each policy year.
4. On-level earned premium = Historical Earned Premium × (Current Rate Index / ARLF)

Extension of Exposures Method
------------------------------
Re-rate every individual policy at current rates.  More precise but requires
individual policy data.  We approximate by applying on-level factors from the
parallelogram if individual re-rating data is not available.

References
----------
- Werner & Modlin (2016) "Basic Ratemaking", Chapter 4
- CAS Exam 5 syllabus
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)


class OnLevelPremium:
    """
    Compute on-level earned premium using the parallelogram method.

    Parameters
    ----------
    rate_changes : pd.DataFrame
        Must have columns: effective_date | line_of_business | rate_change_pct
        rate_change_pct is a decimal (0.05 = +5%)
    policies : pd.DataFrame
        Must have columns: effective_date | expiration_date | earned_premium |
                           line_of_business
    lob : str
        Line of business to filter on.
    base_year : int, optional
        Year from which the cumulative rate index starts at 1.000.
        Defaults to the earliest year in the data.
    """

    def __init__(
        self,
        rate_changes: pd.DataFrame,
        policies: pd.DataFrame,
        lob: str,
        base_year: Optional[int] = None,
    ) -> None:
        self.lob = lob

        # Filter to LOB
        self.rate_changes = rate_changes[rate_changes["line_of_business"] == lob].copy()
        self.rate_changes = self.rate_changes.sort_values("effective_date").reset_index(drop=True)

        self.policies = policies[policies["line_of_business"] == lob].copy()

        # Determine year range
        if self.policies.empty:
            raise ValueError(f"No policies found for LOB='{lob}'")

        self.years = sorted(self.policies["effective_date"].dt.year.unique())
        self.base_year = base_year or self.years[0]

        # Build cumulative rate index series
        self._rate_index: Dict[pd.Timestamp, float] = {}
        self._build_rate_index()

        # On-level factors by policy year
        self._on_level_factors: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Rate index
    # ------------------------------------------------------------------

    def _build_rate_index(self) -> None:
        """
        Build cumulative rate index keyed to each rate-change effective date.
        Index = 1.000 at base_year start; multiplied by (1 + pct) at each change.
        """
        base_date = pd.Timestamp(f"{self.base_year}-01-01")
        index = 1.0
        self._rate_index[base_date] = 1.0

        for _, row in self.rate_changes.iterrows():
            eff = row["effective_date"]
            pct = row["rate_change_pct"]
            if eff >= base_date:
                index *= (1.0 + pct)
                self._rate_index[eff] = index

    def current_rate_index(self) -> float:
        """Return the current cumulative rate index (most recent)."""
        if not self._rate_index:
            return 1.0
        return float(max(self._rate_index.values()))

    def rate_index_at(self, date: pd.Timestamp) -> float:
        """Return the rate index in effect on *date*."""
        sorted_dates = sorted(self._rate_index.keys())
        idx = 1.0
        for d in sorted_dates:
            if d <= date:
                idx = self._rate_index[d]
            else:
                break
        return idx

    # ------------------------------------------------------------------
    # Parallelogram method
    # ------------------------------------------------------------------

    def _parallelogram_arlf(self, policy_year: int) -> float:
        """
        Compute the Average Rate Level Factor (ARLF) for a policy year
        using the parallelogram method.

        Assumes policies are uniformly distributed throughout the year
        and have annual terms (12-month policies).

        Returns: ARLF = weighted average rate index for earned premium
        """
        # For a policy year: policies written uniformly Jan 1 → Dec 31
        # A policy written on date t (0 ≤ t ≤ 1) in the year earns over [t, t+1]
        # Earned premium from a policy written at t is uniform over its term.

        # Rate changes that affect this policy year's earned premium
        py_start = pd.Timestamp(f"{policy_year}-01-01")
        py_end = pd.Timestamp(f"{policy_year + 1}-12-31")

        relevant = [
            (d, v)
            for d, v in sorted(self._rate_index.items())
            if d <= py_end
        ]

        if not relevant:
            return 1.0

        # Numerical integration: sample 1000 points
        t_points = np.linspace(0, 1, 1000)  # fraction of policy year written
        arlf_sum = 0.0

        for t in t_points:
            # This policy written at t in the policy year, earns over [t, t+1] years
            earn_start_frac = t
            earn_end_frac = t + 1.0  # 1 year later

            # Convert to calendar date
            earn_start_date = py_start + pd.Timedelta(days=int(t * 365))
            earn_end_date = earn_start_date + pd.DateOffset(years=1)

            # Average rate level over the earning period
            # Sample mid-point of earning period
            earn_mid_date = earn_start_date + (earn_end_date - earn_start_date) / 2
            arlf_sum += self.rate_index_at(earn_mid_date)

        return arlf_sum / len(t_points)

    def on_level_factors(self) -> pd.Series:
        """
        Compute on-level factors for each policy year.

        on_level_factor(year) = Current Rate Index / ARLF(year)

        Returns
        -------
        pd.Series  (index = policy_year, values = on-level factor)
        """
        cri = self.current_rate_index()
        factors = {}
        for yr in self.years:
            arlf = self._parallelogram_arlf(yr)
            factors[yr] = cri / arlf if arlf != 0 else 1.0

        self._on_level_factors = pd.Series(factors, name="on_level_factor")
        return self._on_level_factors

    def on_level_premium(self) -> pd.DataFrame:
        """
        Compute on-level earned premium by policy year.

        Returns
        -------
        pd.DataFrame
            policy_year | earned_premium | on_level_factor | on_level_earned_premium
        """
        olfs = self.on_level_factors()

        # Aggregate earned premium by policy year
        pol = self.policies.copy()
        pol["policy_year"] = pol["effective_date"].dt.year

        # Use pre-computed earned premium if available, else use written
        ep_col = "earned_premium" if "earned_premium" in pol.columns else "written_premium"
        by_year = pol.groupby("policy_year")[ep_col].sum().rename("earned_premium")

        result = by_year.to_frame()
        result["on_level_factor"] = olfs
        result["on_level_earned_premium"] = result["earned_premium"] * result["on_level_factor"]

        return result

    def rate_change_table(self) -> pd.DataFrame:
        """
        Return a formatted table of rate changes and cumulative index.
        """
        rows = []
        for date, idx in sorted(self._rate_index.items()):
            # Find the rate change at this date
            match = self.rate_changes[self.rate_changes["effective_date"] == date]
            pct = match["rate_change_pct"].iloc[0] if not match.empty else 0.0
            rows.append(
                {
                    "effective_date": date,
                    "rate_change_pct": pct,
                    "cumulative_index": idx,
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        cri = self.current_rate_index()
        return (
            f"OnLevelPremium(lob={self.lob!r}, "
            f"rate_changes={len(self.rate_changes)}, "
            f"current_index={cri:.4f})"
        )
