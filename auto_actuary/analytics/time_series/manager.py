"""
auto_actuary.analytics.time_series.manager
==========================================
Historical / time-series data management for insurance analytics.

Insurance data is inherently temporal: policies earn over time, claims develop
over multiple valuation dates, and loss ratios shift with market cycles.
This module provides two complementary abstractions:

SnapshotStore
    A keyed container of (as_of_date → DataFrames) snapshots.  Each snapshot
    is a point-in-time picture of the portfolio.  Use this when you need to
    compare the same KPI across multiple evaluation dates (e.g., year-end
    actuarial reviews, quarterly reserve monitoring).

TimeSeriesManager
    Higher-level manager built on SnapshotStore.  Adds:
      - Period-over-period change tables (absolute and %)
      - Compound annual growth rate (CAGR) computation
      - Rolling N-period averages
      - Trend fitting (log-linear OLS) with confidence intervals
      - Spark-line export for dashboard embedding

Typical workflow
----------------
>>> store = SnapshotStore()
>>> store.add_snapshot("2022-12-31", {"policies": df_22, "claims": df_22_c})
>>> store.add_snapshot("2023-12-31", {"policies": df_23, "claims": df_23_c})

>>> ts = TimeSeriesManager(store)
>>> lr_series = ts.metric_series("loss_ratio", compute_fn=my_lr_function)
>>> ts.period_change(lr_series)          # year-over-year deltas
>>> ts.cagr(lr_series, periods=3)        # 3-year CAGR
>>> ts.trend_fit(lr_series)              # log-linear slope + CI
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SnapshotStore
# ---------------------------------------------------------------------------

class SnapshotStore:
    """
    An ordered, date-keyed store of DataFrame snapshots.

    Each snapshot is a dict mapping table name → DataFrame.  Snapshots are
    sorted chronologically on insertion.

    Parameters
    ----------
    name : str
        Optional label (e.g. LOB code or portfolio name) for display.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._snapshots: Dict[pd.Timestamp, Dict[str, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_snapshot(
        self,
        as_of_date: Any,
        tables: Dict[str, pd.DataFrame],
        overwrite: bool = False,
    ) -> "SnapshotStore":
        """
        Register a point-in-time snapshot.

        Parameters
        ----------
        as_of_date : str, date, or Timestamp
            The evaluation date for this snapshot.
        tables : dict
            Mapping of table name → DataFrame (e.g. {"policies": df, "claims": df}).
        overwrite : bool
            If True, replace an existing snapshot at the same date.

        Returns
        -------
        self (for chaining)
        """
        ts = pd.Timestamp(as_of_date)
        if ts in self._snapshots and not overwrite:
            raise ValueError(
                f"Snapshot for {ts.date()} already exists.  "
                "Pass overwrite=True to replace it."
            )
        self._snapshots[ts] = {k: v.copy() for k, v in tables.items()}
        logger.info("SnapshotStore '%s': added snapshot at %s (%d tables)",
                    self.name, ts.date(), len(tables))
        return self

    def remove_snapshot(self, as_of_date: Any) -> "SnapshotStore":
        """Remove the snapshot at *as_of_date*."""
        ts = pd.Timestamp(as_of_date)
        if ts not in self._snapshots:
            raise KeyError(f"No snapshot for {ts.date()}")
        del self._snapshots[ts]
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def dates(self) -> List[pd.Timestamp]:
        """Sorted list of snapshot dates."""
        return sorted(self._snapshots)

    @property
    def n_snapshots(self) -> int:
        return len(self._snapshots)

    def get(self, as_of_date: Any) -> Dict[str, pd.DataFrame]:
        """Return the snapshot dict for *as_of_date*."""
        ts = pd.Timestamp(as_of_date)
        if ts not in self._snapshots:
            raise KeyError(f"No snapshot for {ts.date()}.  Available: {[d.date() for d in self.dates]}")
        return self._snapshots[ts]

    def get_table(self, as_of_date: Any, table: str) -> pd.DataFrame:
        """Return a single table from the snapshot at *as_of_date*."""
        return self.get(as_of_date)[table]

    def latest(self) -> Tuple[pd.Timestamp, Dict[str, pd.DataFrame]]:
        """Return (date, snapshot_dict) for the most recent snapshot."""
        if not self._snapshots:
            raise ValueError("SnapshotStore is empty.")
        latest_date = self.dates[-1]
        return latest_date, self._snapshots[latest_date]

    def prior(self, periods: int = 1) -> Tuple[pd.Timestamp, Dict[str, pd.DataFrame]]:
        """Return the snapshot *periods* steps before the most recent one."""
        dates = self.dates
        if len(dates) <= periods:
            raise ValueError(
                f"Only {len(dates)} snapshots exist; cannot go back {periods} period(s)."
            )
        idx = -(periods + 1)
        d = dates[idx]
        return d, self._snapshots[d]

    def __contains__(self, as_of_date: Any) -> bool:
        return pd.Timestamp(as_of_date) in self._snapshots

    def __len__(self) -> int:
        return len(self._snapshots)

    def __repr__(self) -> str:
        dates_str = ", ".join(str(d.date()) for d in self.dates)
        return f"SnapshotStore(name={self.name!r}, dates=[{dates_str}])"


# ---------------------------------------------------------------------------
# TrendResult
# ---------------------------------------------------------------------------

@dataclass
class TrendResult:
    """Output of a log-linear trend fit."""
    slope: float           # annual log-change (e.g. 0.03 = 3% per year)
    intercept: float
    r_squared: float
    p_value: float
    std_err: float
    ci_lower: float        # 95% CI lower bound on slope
    ci_upper: float        # 95% CI upper bound on slope
    n_obs: int

    @property
    def annual_pct_change(self) -> float:
        """Approximate annual % change ≈ slope (valid for small slopes)."""
        return float(np.expm1(self.slope))

    @property
    def is_significant(self) -> bool:
        """True if slope is statistically significant at 5% level."""
        return self.p_value < 0.05

    def __repr__(self) -> str:
        sig = "***" if self.p_value < 0.01 else ("*" if self.p_value < 0.05 else "")
        return (
            f"TrendResult(slope={self.slope:+.4f}{sig}, "
            f"ann_chg={self.annual_pct_change:+.2%}, "
            f"R²={self.r_squared:.3f}, p={self.p_value:.4f})"
        )


# ---------------------------------------------------------------------------
# TimeSeriesManager
# ---------------------------------------------------------------------------

class TimeSeriesManager:
    """
    High-level time-series analytics on top of a SnapshotStore.

    Parameters
    ----------
    store : SnapshotStore
        The underlying snapshot container.
    year_end_only : bool
        If True, only use December 31 snapshots for year-over-year calculations.
        Quarterly snapshots are still stored but skipped in YoY comparisons.
    """

    def __init__(
        self,
        store: Optional[SnapshotStore] = None,
        year_end_only: bool = False,
    ) -> None:
        self.store = store or SnapshotStore()
        self.year_end_only = year_end_only

    # ------------------------------------------------------------------
    # Snapshot delegation helpers
    # ------------------------------------------------------------------

    def add_snapshot(self, as_of_date: Any, tables: Dict[str, pd.DataFrame], **kw) -> "TimeSeriesManager":
        self.store.add_snapshot(as_of_date, tables, **kw)
        return self

    # ------------------------------------------------------------------
    # Metric series construction
    # ------------------------------------------------------------------

    def metric_series(
        self,
        metric_name: str,
        compute_fn: Callable[[pd.Timestamp, Dict[str, pd.DataFrame]], float],
        dates: Optional[List[Any]] = None,
    ) -> pd.Series:
        """
        Build a time series of a scalar metric by applying *compute_fn* to
        each snapshot.

        Parameters
        ----------
        metric_name : str
            Label for the series (used as Series name).
        compute_fn : callable
            A function (as_of_date, snapshot_dict) → float.
        dates : list, optional
            Subset of snapshot dates to evaluate.  None = all dates.

        Returns
        -------
        pd.Series indexed by pd.Timestamp (snapshot dates).

        Example
        -------
        >>> def my_lr(dt, snap):
        ...     pol = snap["policies"]
        ...     clm = snap["claims"]
        ...     return clm["incurred_loss"].sum() / pol["written_premium"].sum()
        >>> ts.metric_series("loss_ratio", my_lr)
        """
        eval_dates = [pd.Timestamp(d) for d in dates] if dates else self.store.dates
        if self.year_end_only:
            eval_dates = [d for d in eval_dates if d.month == 12 and d.day == 31]

        values = {}
        for dt in eval_dates:
            try:
                snap = self.store.get(dt)
                values[dt] = float(compute_fn(dt, snap))
            except Exception as exc:
                logger.warning("metric_series '%s' failed at %s: %s", metric_name, dt.date(), exc)
                values[dt] = np.nan

        return pd.Series(values, name=metric_name).sort_index()

    def dataframe_series(
        self,
        compute_fn: Callable[[pd.Timestamp, Dict[str, pd.DataFrame]], pd.DataFrame],
        dates: Optional[List[Any]] = None,
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Like metric_series but for DataFrame-returning functions.

        Returns
        -------
        dict mapping pd.Timestamp → pd.DataFrame
        """
        eval_dates = [pd.Timestamp(d) for d in dates] if dates else self.store.dates
        result: Dict[pd.Timestamp, pd.DataFrame] = {}
        for dt in eval_dates:
            try:
                snap = self.store.get(dt)
                result[dt] = compute_fn(dt, snap)
            except Exception as exc:
                logger.warning("dataframe_series failed at %s: %s", dt.date(), exc)
        return result

    # ------------------------------------------------------------------
    # Period-over-period analysis
    # ------------------------------------------------------------------

    def period_change(
        self,
        series: pd.Series,
        periods: int = 1,
        pct: bool = True,
    ) -> pd.DataFrame:
        """
        Compute period-over-period change for a metric series.

        Parameters
        ----------
        series : pd.Series
            Time-indexed metric series (e.g. from metric_series()).
        periods : int
            Number of periods to look back (1 = prior period).
        pct : bool
            If True, include % change column.

        Returns
        -------
        pd.DataFrame
            Columns: value | prior_value | change | pct_change (if pct=True)
        """
        s = series.sort_index().dropna()
        df = pd.DataFrame({"value": s})
        df["prior_value"] = df["value"].shift(periods)
        df["change"] = df["value"] - df["prior_value"]
        if pct:
            df["pct_change"] = df["change"] / df["prior_value"].replace(0, np.nan)
        return df

    def rolling_average(
        self,
        series: pd.Series,
        window: int = 3,
        min_periods: int = 1,
    ) -> pd.Series:
        """
        Rolling N-period average of a metric series.

        Parameters
        ----------
        window : int
            Look-back window (number of snapshots).
        min_periods : int
            Minimum observations required to produce a value.
        """
        return series.sort_index().rolling(window=window, min_periods=min_periods).mean()

    def cagr(self, series: pd.Series, periods: Optional[int] = None) -> float:
        """
        Compound annual growth rate over *periods* years.

        Parameters
        ----------
        series : pd.Series
            Time-indexed metric series.
        periods : int, optional
            Number of annual periods.  None = full series length in years.

        Returns
        -------
        float
            CAGR (e.g. 0.05 = 5% per year).
        """
        s = series.sort_index().dropna()
        if len(s) < 2:
            return np.nan

        if periods is not None:
            s = s.iloc[-(periods + 1):]

        start_val = float(s.iloc[0])
        end_val = float(s.iloc[-1])
        n_years = (s.index[-1] - s.index[0]).days / 365.25

        if start_val <= 0 or n_years <= 0:
            return np.nan

        return float((end_val / start_val) ** (1.0 / n_years) - 1.0)

    # ------------------------------------------------------------------
    # Trend fitting
    # ------------------------------------------------------------------

    def trend_fit(
        self,
        series: pd.Series,
        log_transform: bool = True,
    ) -> TrendResult:
        """
        Fit a log-linear trend to a metric series using OLS.

        The regression is:
            log(y_t) = a + b * t    [if log_transform=True]
            y_t      = a + b * t    [if log_transform=False]

        where t is measured in fractional years from the series start.

        Parameters
        ----------
        series : pd.Series
            Time-indexed metric series (index must be datetime-like).
        log_transform : bool
            Apply natural log to the dependent variable (recommended for
            ratio / frequency metrics).

        Returns
        -------
        TrendResult
        """
        s = series.sort_index().dropna()
        if len(s) < 3:
            raise ValueError(f"Need ≥ 3 observations for trend fit; got {len(s)}.")

        origin = s.index[0]
        t = np.array([(d - origin).days / 365.25 for d in s.index])
        y = s.values.astype(float)

        if log_transform:
            if np.any(y <= 0):
                raise ValueError("log_transform=True requires all values > 0.")
            y = np.log(y)

        slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)

        # 95% CI on slope
        t_crit = stats.t.ppf(0.975, df=len(t) - 2)
        ci_lower = float(slope - t_crit * std_err)
        ci_upper = float(slope + t_crit * std_err)

        return TrendResult(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_value ** 2),
            p_value=float(p_value),
            std_err=float(std_err),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_obs=len(s),
        )

    # ------------------------------------------------------------------
    # Comparative table (across snapshots)
    # ------------------------------------------------------------------

    def comparison_table(
        self,
        metrics: Dict[str, Callable[[pd.Timestamp, Dict[str, pd.DataFrame]], float]],
        dates: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """
        Build a wide comparison table: dates × metrics.

        Parameters
        ----------
        metrics : dict
            Mapping metric_name → compute_fn.
        dates : list, optional
            Snapshot dates to include.  None = all.

        Returns
        -------
        pd.DataFrame
            Rows = snapshot dates, columns = metric names.

        Example
        -------
        >>> table = ts.comparison_table({
        ...     "loss_ratio": lambda dt, s: s["claims"]["incurred_loss"].sum() / s["policies"]["written_premium"].sum(),
        ...     "policy_count": lambda dt, s: len(s["policies"]),
        ... })
        """
        series_dict = {}
        for name, fn in metrics.items():
            series_dict[name] = self.metric_series(name, fn, dates=dates)
        return pd.DataFrame(series_dict)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Summarise the snapshot store: dates, table names, row counts.

        Returns
        -------
        pd.DataFrame
            Columns: as_of_date | tables | total_rows
        """
        rows = []
        for dt in self.store.dates:
            snap = self.store.get(dt)
            total_rows = sum(len(v) for v in snap.values())
            rows.append({
                "as_of_date": dt,
                "tables": ", ".join(sorted(snap.keys())),
                "total_rows": total_rows,
            })
        return pd.DataFrame(rows).set_index("as_of_date")

    def __repr__(self) -> str:
        return (
            f"TimeSeriesManager(snapshots={self.store.n_snapshots}, "
            f"dates={[d.date() for d in self.store.dates]})"
        )
