"""
auto_actuary.analytics.speculative.trend_projector
====================================================
Forward trend projection with uncertainty bands and regime detection.

The standard actuarial trend approach (log-linear regression) gives a point
estimate — a single annual trend factor.  That's fine for the exhibits, but
executives need to understand *how much the projection could be wrong* and
*whether the trend itself is changing*.

This module adds:

1. **Prediction intervals** — Bootstrap-derived bands showing plausible
   outcomes at the 10th/25th/75th/90th percentile.

2. **Multiple scenarios** — Pessimistic / Base / Optimistic projections
   using trend assumption overrides or the fitted CI bands.

3. **Regime change detection** — A simple Chow test flags whether the trend
   appears to have structurally changed (e.g., post-COVID inflection in
   severity, post-distracted-driving inflection in frequency).

4. **Sensitivity analysis** — Shows how the projected KPI changes as the
   assumed annual trend varies from pessimistic to optimistic.

Why this matters for executives
--------------------------------
"Our model says frequency grows at +2%/yr" is one number.
"If frequency grows at +4%/yr instead (90th percentile), our combined ratio
is 104% by 2027" is a decision-support insight.

References
----------
- Chow, G.C. (1960) "Tests of Equality between Sets of Coefficients in Two
  Linear Regressions", Econometrica 28(3)
- Mahler, H. (2010) CAS Exam 5 study note — Trend & Development
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as spstats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ProjectionPoint:
    """Single projected data point with uncertainty."""
    year: float
    point: float        # central estimate
    p10: float          # 10th percentile (pessimistic tail)
    p25: float          # 25th percentile
    p75: float          # 75th percentile
    p90: float          # 90th percentile (adverse tail)
    is_observed: bool   # True = historical, False = projection


@dataclass
class RegimeChangeResult:
    """Result of a regime-change / structural-break test."""
    break_year: Optional[float]   # None if no significant break detected
    f_statistic: float
    p_value: float
    pre_break_trend: Optional[float]   # annual trend before break
    post_break_trend: Optional[float]  # annual trend after break
    is_significant: bool

    @property
    def trend_acceleration(self) -> Optional[float]:
        if self.pre_break_trend is None or self.post_break_trend is None:
            return None
        return self.post_break_trend - self.pre_break_trend

    def __repr__(self) -> str:
        if not self.is_significant:
            return "RegimeChangeResult(no significant break)"
        return (
            f"RegimeChangeResult(break={self.break_year}, "
            f"pre={self.pre_break_trend:+.2%}/yr, "
            f"post={self.post_break_trend:+.2%}/yr, "
            f"p={self.p_value:.3f})"
        )


@dataclass
class SensitivityPoint:
    """One row in a sensitivity analysis table."""
    assumed_trend: float     # the annual trend assumption
    horizon_years: int
    projected_value: float   # KPI value under this assumption
    pct_change_vs_base: float


# ---------------------------------------------------------------------------
# Main TrendProjector
# ---------------------------------------------------------------------------

class TrendProjector:
    """
    Forward-looking trend projection with uncertainty quantification.

    Wraps a fitted log-linear trend (from TrendAnalysis or raw data) and
    adds prediction intervals, scenario forks, and regime detection.

    Parameters
    ----------
    metric_name : str
        Label for display (e.g., "Frequency", "Severity", "Pure Premium").
    annual_trend : float, optional
        Override the fitted trend with a specific assumption (e.g., 0.03 = +3%).
        If None, the trend is estimated from the data.
    n_boot : int
        Bootstrap iterations for prediction intervals.  Default 500.
        Lower values (100) for speed; 1000+ for publication-quality CI.
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        metric_name: str = "metric",
        annual_trend: Optional[float] = None,
        n_boot: int = 500,
        random_state: int = 42,
    ) -> None:
        self.metric_name = metric_name
        self._annual_trend_override = annual_trend
        self.n_boot = n_boot
        self.random_state = random_state

        # Fitted from data
        self._data: Optional[pd.DataFrame] = None
        self._slope: float = 0.0          # log-space annual slope
        self._intercept: float = 0.0
        self._base_year: float = 0.0
        self._fitted_annual_trend: float = 1.0
        self._r_squared: float = 0.0
        self._p_value: float = 1.0
        self._residual_std: float = 0.0
        self._boot_slopes: Optional[np.ndarray] = None

        self.is_fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        year_col: str = "year",
        value_col: str = "value",
    ) -> "TrendProjector":
        """
        Fit the log-linear trend to historical data.

        Parameters
        ----------
        data : DataFrame
            Must have *year_col* (int/float) and *value_col* (positive float).
        year_col : str
        value_col : str

        Returns
        -------
        self (for chaining)
        """
        df = data[[year_col, value_col]].dropna().copy()
        df = df[df[value_col] > 0].sort_values(year_col)
        df.columns = ["year", "value"]

        if len(df) < 2:
            raise ValueError(f"Need at least 2 data points to fit a trend; got {len(df)}")

        self._data = df.copy()
        self._base_year = float(df["year"].min())

        x = (df["year"] - self._base_year).values.astype(float)
        y = np.log(df["value"].values.astype(float))

        slope, intercept, r, p, se = spstats.linregress(x, y)
        self._slope = float(slope)
        self._intercept = float(intercept)
        self._r_squared = float(r ** 2)
        self._p_value = float(p)
        self._fitted_annual_trend = float(np.exp(slope))
        self._residual_std = float(np.std(y - (intercept + slope * x)))

        # Bootstrap slope distribution
        self._boot_slopes = self._bootstrap_slopes(x, y)

        self.is_fitted = True
        logger.debug(
            "TrendProjector.fit: %s — slope=%.4f, trend=%.4f/yr, R²=%.3f, p=%.3f",
            self.metric_name,
            self._slope,
            self._fitted_annual_trend - 1,
            self._r_squared,
            self._p_value,
        )
        return self

    def _bootstrap_slopes(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bootstrap distribution of the log-linear slope."""
        rng = np.random.default_rng(self.random_state)
        n = len(x)
        slopes = np.zeros(self.n_boot)
        for i in range(self.n_boot):
            idx = rng.choice(n, size=n, replace=True)
            try:
                s, *_ = spstats.linregress(x[idx], y[idx])
                slopes[i] = s
            except Exception:
                slopes[i] = self._slope
        return slopes

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(
        self,
        horizon_years: int,
        scenarios: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Project the metric forward with prediction intervals.

        Parameters
        ----------
        horizon_years : int
            Number of years to project beyond the last observed year.
        scenarios : dict, optional
            Named annual trend overrides for scenario forks.
            e.g., {"pessimistic": 0.06, "base": 0.03, "optimistic": 0.01}
            If None, uses the fitted trend + bootstrap bands.

        Returns
        -------
        DataFrame with columns:
            year | type (historical/projection) | point |
            p10 | p25 | p75 | p90 |
            [scenario columns if provided]
        """
        self._check_fitted()
        df = self._data.copy()
        last_year = float(df["year"].max())
        last_value = float(df.loc[df["year"] == last_year, "value"].iloc[0])

        proj_years = np.arange(last_year + 1, last_year + horizon_years + 1)

        # Effective annual trend (fitted or overridden)
        if self._annual_trend_override is not None:
            eff_slope = float(np.log(1 + self._annual_trend_override))
            boot_slopes = np.full(self.n_boot, eff_slope)
        else:
            eff_slope = self._slope
            boot_slopes = self._boot_slopes

        # Bootstrap projection paths
        boot_paths = np.zeros((self.n_boot, len(proj_years)))
        for i, s in enumerate(boot_slopes):
            # Start from last observed value and project forward
            for j, yr in enumerate(proj_years):
                years_forward = yr - last_year
                boot_paths[i, j] = last_value * np.exp(s * years_forward)

        # Point projection
        proj_values = [last_value * np.exp(eff_slope * (yr - last_year)) for yr in proj_years]

        rows = []

        # Historical points (no CI — these are observed)
        for _, row in df.iterrows():
            rows.append({
                "year": float(row["year"]),
                "type": "historical",
                "point": float(row["value"]),
                "p10": float(row["value"]),
                "p25": float(row["value"]),
                "p75": float(row["value"]),
                "p90": float(row["value"]),
            })

        # Projected points with CI bands
        for j, (yr, pv) in enumerate(zip(proj_years, proj_values)):
            boot_col = boot_paths[:, j]
            row: dict = {
                "year": float(yr),
                "type": "projection",
                "point": float(pv),
                "p10": float(np.percentile(boot_col, 10)),
                "p25": float(np.percentile(boot_col, 25)),
                "p75": float(np.percentile(boot_col, 75)),
                "p90": float(np.percentile(boot_col, 90)),
            }

            # Named scenario overrides
            if scenarios:
                for scen_name, scen_trend in scenarios.items():
                    row[scen_name] = float(
                        last_value * (1 + scen_trend) ** (yr - last_year)
                    )

            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Regime / structural-break detection
    # ------------------------------------------------------------------

    def detect_regime_change(
        self,
        min_segment_size: int = 3,
    ) -> RegimeChangeResult:
        """
        Test for a structural break in the trend using a Chow-style F-test.

        Tries all possible break points between *min_segment_size* and
        (n - min_segment_size), and returns the most significant one.

        The F-statistic compares:
        - Unrestricted: two separate regressions (pre and post break)
        - Restricted: one common regression

        Parameters
        ----------
        min_segment_size : int
            Minimum number of observations in each segment.

        Returns
        -------
        RegimeChangeResult
        """
        self._check_fitted()
        df = self._data.copy()
        n = len(df)

        if n < 2 * min_segment_size + 1:
            return RegimeChangeResult(
                break_year=None,
                f_statistic=0.0,
                p_value=1.0,
                pre_break_trend=None,
                post_break_trend=None,
                is_significant=False,
            )

        x = (df["year"] - self._base_year).values.astype(float)
        y = np.log(df["value"].values.astype(float))

        # Restricted (full) model SSE
        s_r, i_r, *_ = spstats.linregress(x, y)
        y_hat_r = i_r + s_r * x
        sse_r = float(np.sum((y - y_hat_r) ** 2))

        best_break = None
        best_f = -np.inf
        best_p = 1.0
        best_pre_slope = None
        best_post_slope = None

        for bp_idx in range(min_segment_size, n - min_segment_size):
            x1, y1 = x[:bp_idx], y[:bp_idx]
            x2, y2 = x[bp_idx:], y[bp_idx:]

            try:
                s1, i1, *_ = spstats.linregress(x1, y1)
                s2, i2, *_ = spstats.linregress(x2, y2)
            except Exception:
                continue

            sse1 = float(np.sum((y1 - (i1 + s1 * x1)) ** 2))
            sse2 = float(np.sum((y2 - (i2 + s2 * x2)) ** 2))
            sse_u = sse1 + sse2

            # Chow F-statistic: (SSE_R - SSE_U) / k / (SSE_U / (n - 2k))
            k = 2  # two parameters (slope, intercept)
            df_num = k
            df_den = n - 2 * k
            if df_den <= 0 or sse_u <= 0:
                continue

            f_stat = ((sse_r - sse_u) / df_num) / (sse_u / df_den)
            p_val = float(1 - spstats.f.cdf(f_stat, df_num, df_den))

            if f_stat > best_f:
                best_f = f_stat
                best_p = p_val
                best_break = float(df["year"].iloc[bp_idx])
                best_pre_slope = float(s1)
                best_post_slope = float(s2)

        is_sig = (best_p < 0.10) if best_p is not None else False

        return RegimeChangeResult(
            break_year=best_break if is_sig else None,
            f_statistic=float(best_f),
            p_value=float(best_p) if best_p is not None else 1.0,
            pre_break_trend=float(np.exp(best_pre_slope) - 1) if best_pre_slope is not None and is_sig else None,
            post_break_trend=float(np.exp(best_post_slope) - 1) if best_post_slope is not None and is_sig else None,
            is_significant=is_sig,
        )

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity(
        self,
        horizon_years: int,
        trend_range: Optional[np.ndarray] = None,
        n_steps: int = 20,
    ) -> pd.DataFrame:
        """
        How does the projected value change across different trend assumptions?

        Returns a DataFrame with one row per trend assumption showing the
        projected KPI and % change vs. the base (fitted) trend.

        Parameters
        ----------
        horizon_years : int
            Projection horizon.
        trend_range : ndarray, optional
            Array of annual trend fractions to sweep.
            Default: fitted_trend ± 5 percentage points in 20 steps.
        n_steps : int
            Number of steps in the default range.

        Returns
        -------
        DataFrame: assumed_trend | horizon_years | projected_value | pct_change_vs_base
        """
        self._check_fitted()
        last_value = float(self._data.loc[self._data["year"] == self._data["year"].max(), "value"].iloc[0])
        base_trend = self._fitted_annual_trend - 1.0

        if trend_range is None:
            lo = max(-0.20, base_trend - 0.05)
            hi = min(0.30, base_trend + 0.05)
            trend_range = np.linspace(lo, hi, n_steps)

        base_proj = last_value * (1 + base_trend) ** horizon_years

        rows = []
        for t in trend_range:
            proj = last_value * (1 + t) ** horizon_years
            rows.append(SensitivityPoint(
                assumed_trend=float(t),
                horizon_years=horizon_years,
                projected_value=float(proj),
                pct_change_vs_base=float((proj - base_proj) / max(abs(base_proj), 1e-10)),
            ))

        return pd.DataFrame([
            {
                "assumed_trend": p.assumed_trend,
                "horizon_years": p.horizon_years,
                "projected_value": p.projected_value,
                "pct_change_vs_base": p.pct_change_vs_base,
            }
            for p in rows
        ])

    # ------------------------------------------------------------------
    # Properties & diagnostics
    # ------------------------------------------------------------------

    @property
    def fitted_annual_trend(self) -> float:
        """Fitted annual trend factor (e.g., 1.032 = +3.2%/yr)."""
        self._check_fitted()
        return self._fitted_annual_trend

    @property
    def r_squared(self) -> float:
        self._check_fitted()
        return self._r_squared

    @property
    def p_value(self) -> float:
        self._check_fitted()
        return self._p_value

    @property
    def is_significant(self) -> bool:
        """True if the trend is statistically significant at 10% level."""
        return self.p_value < 0.10

    def bootstrap_trend_ci(self, ci: float = 0.90) -> Tuple[float, float]:
        """
        Bootstrap confidence interval on the fitted annual trend.

        Returns (lower, upper) as annual trend factors.
        """
        self._check_fitted()
        alpha = (1 - ci) / 2
        boot_trends = np.exp(self._boot_slopes)
        return (
            float(np.quantile(boot_trends, alpha)),
            float(np.quantile(boot_trends, 1 - alpha)),
        )

    def trend_summary(self) -> pd.Series:
        """Quick scalar summary of the fitted trend."""
        self._check_fitted()
        lo, hi = self.bootstrap_trend_ci()
        return pd.Series({
            "metric": self.metric_name,
            "annual_trend": self._fitted_annual_trend - 1.0,
            "annual_trend_pct": f"{self._fitted_annual_trend - 1:.2%}",
            "r_squared": self._r_squared,
            "p_value": self._p_value,
            "significant": self.is_significant,
            "ci_90_low": lo - 1.0,
            "ci_90_high": hi - 1.0,
            "n_obs": len(self._data),
        })

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("TrendProjector not fitted. Call .fit() first.")

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"TrendProjector(metric={self.metric_name!r}, not fitted)"
        return (
            f"TrendProjector(metric={self.metric_name!r}, "
            f"trend={self._fitted_annual_trend - 1:+.2%}/yr, "
            f"R²={self._r_squared:.3f}, p={self._p_value:.3f})"
        )


# ---------------------------------------------------------------------------
# Convenience: build multiple TrendProjectors from a freq/sev table
# ---------------------------------------------------------------------------

def build_trend_projectors(
    fs_table: pd.DataFrame,
    year_col: str = "accident_year",
    metrics: Optional[List[str]] = None,
    n_boot: int = 300,
) -> Dict[str, TrendProjector]:
    """
    Fit TrendProjectors for frequency, severity, and pure premium.

    Parameters
    ----------
    fs_table : DataFrame
        Output of FreqSevAnalysis.fs_table() with year and metric columns.
    year_col : str
        Column containing the year (default "accident_year").
    metrics : list of str, optional
        Metric columns to project.  Defaults to ["frequency", "severity", "pure_premium"].
    n_boot : int
        Bootstrap iterations for each projector.

    Returns
    -------
    dict of metric_name -> TrendProjector
    """
    metrics = metrics or ["frequency", "severity", "pure_premium"]
    projectors: Dict[str, TrendProjector] = {}

    df = fs_table.reset_index() if not isinstance(fs_table.index, pd.RangeIndex) else fs_table.copy()

    for metric in metrics:
        if metric not in df.columns:
            continue
        data = df[[year_col, metric]].rename(columns={year_col: "year", metric: "value"}).dropna()
        data = data[data["value"] > 0]
        if len(data) < 2:
            logger.warning("build_trend_projectors: not enough data for '%s' (%d rows)", metric, len(data))
            continue
        tp = TrendProjector(metric_name=metric, n_boot=n_boot)
        tp.fit(data)
        projectors[metric] = tp
        logger.info("Fitted TrendProjector: %s", tp)

    return projectors
