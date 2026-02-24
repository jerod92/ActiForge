"""
auto_actuary.analytics.ratemaking.trend
========================================
Loss and premium trend analysis for ratemaking.

Loss Trend
----------
Measures the change in pure premium (or frequency / severity separately) over
time due to social, economic, and medical inflation.  Pure premium is trended
forward to the future policy period under review.

The standard approach (CAS Exam 5) is log-linear regression:

    ln(PP_t) = a + b·t

    Annual trend factor = e^b
    Trended PP = PP_base × trend_factor^n_years

Premium Trend
-------------
Corrects for changes in the premium-generating characteristics of the book
(e.g., average limits drift, deductible changes, insured value changes).

Trend Selection
---------------
- Multiple periods (3yr, 5yr, all-yr) are computed and presented for
  actuarial judgment.
- Statistical tests (R², p-value, Durbin-Watson) flag unreliable trends.

References
----------
- Werner & Modlin (2016) "Basic Ratemaking", Chapter 6
- Mahler, H. (2010) CAS Exam 5 study note — Trend
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2

logger = logging.getLogger(__name__)


def _durbin_watson(residuals: np.ndarray) -> float:
    """
    Compute the Durbin-Watson statistic for autocorrelation in residuals.

        DW = Σ_{t=2}^{n} (e_t - e_{t-1})² / Σ e_t²

    Interpretation:
      DW ≈ 2  → no autocorrelation (ideal)
      DW < 1.5 → positive autocorrelation (common in loss cost trends)
      DW > 2.5 → negative autocorrelation

    Reference: Durbin & Watson (1950, 1951).
    """
    if len(residuals) < 3:
        return np.nan
    diff = np.diff(residuals)
    dw = float(np.sum(diff ** 2) / np.sum(residuals ** 2)) if np.sum(residuals ** 2) > 0 else np.nan
    return dw


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrendFit:
    """Result of a single log-linear trend fit."""
    period_label: str        # e.g., "5yr", "3yr", "all"
    n_points: int
    annual_trend: float      # e.g., 1.042 = +4.2% per year
    r_squared: float
    p_value: float
    std_err: float
    intercept: float
    slope: float
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    durbin_watson: Optional[float] = None  # DW statistic for serial autocorrelation
    weighted: bool = False               # True if fit used WLS (exposure-weighted)
    slope_ci_90: Optional[Tuple[float, float]] = None  # 90% CI on annual_trend

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.10  # 90% confidence

    @property
    def has_autocorrelation(self) -> bool:
        """True if DW < 1.5 (positive autocorrelation warning)."""
        return self.durbin_watson is not None and self.durbin_watson < 1.5

    @property
    def trend_pct(self) -> float:
        return self.annual_trend - 1.0

    def trend_factor(self, n_years: float) -> float:
        return self.annual_trend ** n_years

    def __repr__(self) -> str:
        dw_str = f", DW={self.durbin_watson:.2f}" if self.durbin_watson is not None else ""
        wgt_str = " [WLS]" if self.weighted else ""
        return (
            f"TrendFit({self.period_label}{wgt_str}: {self.trend_pct:+.2%}/yr, "
            f"R²={self.r_squared:.3f}, p={self.p_value:.3f}{dw_str})"
        )


# ---------------------------------------------------------------------------
# Core trend analysis
# ---------------------------------------------------------------------------

class TrendAnalysis:
    """
    Log-linear trend analysis for loss costs (pure premium, frequency, severity).

    Parameters
    ----------
    data : pd.DataFrame
        Must have columns: year (int) | value (float)
        Optional column: weight (float) — earned exposure or claim counts for WLS.
        where value is pure premium, frequency, or severity.
    periods : list[int]
        Number of years to use in each trend fit, e.g. [3, 5, 10].
        'all' is always included.
    metric_name : str
        Label for the metric being trended (for display).
    use_wls : bool
        If True and a 'weight' column is present, fit using weighted least squares
        (WLS) with weights proportional to earned exposure.  WLS is preferred for
        pure premium trends because years with more exposure should carry more
        weight (Werner & Modlin 2016, §6.3).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        periods: Optional[List[int]] = None,
        metric_name: str = "pure_premium",
        use_wls: bool = True,
    ) -> None:
        if "year" not in data.columns or "value" not in data.columns:
            raise ValueError("data must have 'year' and 'value' columns")

        self.data = data.dropna(subset=["year", "value"]).sort_values("year").copy()
        self.data = self.data[self.data["value"] > 0]  # log-linear requires positive
        self.periods = periods or [3, 5, 10]
        self.metric_name = metric_name
        self.use_wls = use_wls and "weight" in self.data.columns

        self._fits: List[TrendFit] = []
        self._selected: Optional[TrendFit] = None
        self._fit_all()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_period(self, years_subset: pd.DataFrame, label: str) -> Optional[TrendFit]:
        """
        Fit log-linear trend to a subset of the data.

        Uses WLS when exposure weights are available (self.use_wls=True).
        The Durbin-Watson statistic is computed on residuals to flag
        serial autocorrelation.
        """
        if len(years_subset) < 2:
            return None

        x = years_subset["year"].values.astype(float)
        y = np.log(years_subset["value"].values.astype(float))

        # Center x for numerical stability
        x_min = x.min()
        x_c = x - x_min

        weighted = False
        if self.use_wls and "weight" in years_subset.columns:
            w = years_subset["weight"].values.astype(float)
            w = np.where(w > 0, w, 1e-10)  # guard against zero weights
            w = w / w.sum()  # normalise
            # WLS via the weighted normal equations
            # β = (X'WX)^{-1} X'Wy  with X = [1, x_c]
            W = np.diag(w)
            X = np.column_stack([np.ones_like(x_c), x_c])
            try:
                XtW = X.T @ W
                beta = np.linalg.solve(XtW @ X, XtW @ y)
                intercept, slope = float(beta[0]), float(beta[1])
                y_hat = intercept + slope * x_c
                residuals = y - y_hat
                ss_res = float(w @ residuals ** 2)
                ss_tot = float(w @ (y - float(np.average(y, weights=w))) ** 2)
                r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                # Approximate standard error of slope via WLS variance formula
                se_slope = float(
                    np.sqrt(ss_res / max(len(x) - 2, 1) / float(w @ (x_c - np.average(x_c, weights=w)) ** 2))
                    if float(w @ (x_c - np.average(x_c, weights=w)) ** 2) > 0 else np.nan
                )
                # t-test for slope significance
                t_stat = slope / se_slope if se_slope and se_slope > 0 else np.nan
                df = max(len(x) - 2, 1)
                p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df))) if not np.isnan(t_stat) else np.nan
                weighted = True
            except np.linalg.LinAlgError:
                # Fallback to OLS
                slope, intercept, r, p_val, se_slope = stats.linregress(x_c, y)
                r2, residuals = float(r ** 2), y - (intercept + slope * x_c)
        else:
            slope, intercept, r, p_val, se_slope = stats.linregress(x_c, y)
            r2 = float(r ** 2)
            y_hat = intercept + slope * x_c
            residuals = y - y_hat

        annual_trend = np.exp(slope)

        # Durbin-Watson on log-residuals
        dw = _durbin_watson(residuals)

        # 90% CI on slope → transform to CI on annual trend
        df_ci = max(len(x) - 2, 1)
        t90 = float(stats.t.ppf(0.95, df=df_ci))
        if not np.isnan(se_slope):
            slope_lo = slope - t90 * se_slope
            slope_hi = slope + t90 * se_slope
            trend_ci = (float(np.exp(slope_lo)), float(np.exp(slope_hi)))
        else:
            trend_ci = None

        return TrendFit(
            period_label=label,
            n_points=len(years_subset),
            annual_trend=float(annual_trend),
            r_squared=float(r2),
            p_value=float(p_val),
            std_err=float(se_slope),
            intercept=float(intercept),
            slope=float(slope),
            start_year=int(x.min()),
            end_year=int(x.max()),
            durbin_watson=dw,
            weighted=weighted,
            slope_ci_90=trend_ci,
        )

    def _fit_all(self) -> None:
        """Fit trend for all requested periods."""
        self._fits = []

        # All-years fit
        fit = self._fit_period(self.data, "all")
        if fit:
            self._fits.append(fit)

        # Recent-N-year fits
        sorted_years = sorted(self.data["year"].unique())
        for n in self.periods:
            if len(sorted_years) >= n:
                recent_years = sorted_years[-n:]
                subset = self.data[self.data["year"].isin(recent_years)]
                fit = self._fit_period(subset, f"{n}yr")
                if fit:
                    self._fits.append(fit)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, period_label: str = "5yr") -> TrendFit:
        """
        Select a trend fit by period label.

        Falls back to 'all' if the requested period is unavailable.
        """
        for fit in self._fits:
            if fit.period_label == period_label:
                self._selected = fit
                return fit
        # Fall back
        logger.warning(
            "Trend period '%s' not available — using 'all'", period_label
        )
        self._selected = self._fits[0]
        return self._selected

    def selected(self) -> Optional[TrendFit]:
        if self._selected is None and self._fits:
            self._selected = self._fits[0]
        return self._selected

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def trend_table(self) -> pd.DataFrame:
        """
        Return a comparison table of all fitted trends.

        Columns: period | start_year | end_year | n_pts | annual_trend | trend_pct |
                 trend_ci_lo | trend_ci_hi | r_squared | p_value | significant |
                 durbin_watson | autocorrelation_flag | weighted
        """
        rows = []
        for fit in self._fits:
            ci_lo = fit.slope_ci_90[0] if fit.slope_ci_90 else np.nan
            ci_hi = fit.slope_ci_90[1] if fit.slope_ci_90 else np.nan
            rows.append(
                {
                    "period": fit.period_label,
                    "start_year": fit.start_year,
                    "end_year": fit.end_year,
                    "n_pts": fit.n_points,
                    "annual_trend": fit.annual_trend,
                    "trend_pct": fit.trend_pct,
                    "trend_ci_90_lo": ci_lo,
                    "trend_ci_90_hi": ci_hi,
                    "r_squared": fit.r_squared,
                    "p_value": fit.p_value,
                    "significant": fit.is_significant,
                    "durbin_watson": fit.durbin_watson,
                    "autocorrelation_flag": fit.has_autocorrelation,
                    "weighted": fit.weighted,
                }
            )
        return pd.DataFrame(rows)

    def projected_values(self, fit: Optional[TrendFit] = None) -> pd.DataFrame:
        """
        Return observed and fitted values for plotting / diagnostics.
        """
        if fit is None:
            fit = self.selected()
        if fit is None:
            return pd.DataFrame()

        x = self.data["year"].values.astype(float)
        x_c = x - fit.start_year
        y_hat = np.exp(fit.intercept + fit.slope * x_c)

        df = self.data.copy()
        df["fitted"] = y_hat
        df["residual"] = df["value"] - df["fitted"]
        return df

    def trend_factor_between(
        self,
        from_year: float,
        to_year: float,
        fit: Optional[TrendFit] = None,
    ) -> float:
        """
        Compute the trend factor from *from_year* to *to_year*.

        trend_factor = annual_trend ^ (to_year - from_year)
        """
        f = fit or self.selected()
        if f is None:
            return 1.0
        return f.trend_factor(to_year - from_year)

    def __repr__(self) -> str:
        n = len(self._fits)
        sel = self.selected()
        sel_str = repr(sel) if sel else "none selected"
        return f"TrendAnalysis(metric={self.metric_name!r}, fits={n}, selected={sel_str})"


# ---------------------------------------------------------------------------
# Convenience: compute pure premium trend from session-level data
# ---------------------------------------------------------------------------

def build_trend_from_session(
    losses_by_year: pd.DataFrame,
    exposure_by_year: pd.DataFrame,
    year_col: str = "accident_year",
    loss_col: str = "incurred_loss",
    exposure_col: str = "earned_exposure",
    exclude_cat: bool = True,
    cat_col: Optional[str] = None,
) -> Dict[str, TrendAnalysis]:
    """
    Build TrendAnalysis objects for pure premium, frequency, and severity.

    Parameters
    ----------
    losses_by_year : pd.DataFrame
        Columns: accident_year | incurred_loss | (cat_flag optional) | claim_count
    exposure_by_year : pd.DataFrame
        Columns: accident_year | earned_exposure
    exclude_cat : bool
        Remove catastrophe losses before fitting trend (standard practice).

    Returns
    -------
    dict with keys: 'pure_premium', 'frequency', 'severity'
    """
    df = losses_by_year.copy()

    if exclude_cat and cat_col and cat_col in df.columns:
        df = df[df[cat_col] == 0]

    # Join exposure
    merged = df.merge(
        exposure_by_year.rename(columns={year_col: "year", exposure_col: "exposure"}),
        on="year" if "year" in df.columns else year_col,
        how="left",
    )
    merged = merged.rename(columns={year_col: "year", loss_col: "loss"})
    merged = merged[merged["exposure"] > 0]

    merged["pure_premium"] = merged["loss"] / merged["exposure"]

    results = {}

    results["pure_premium"] = TrendAnalysis(
        merged.rename(columns={"pure_premium": "value"})[["year", "value"]],
        metric_name="Pure Premium",
    )

    if "claim_count" in merged.columns:
        merged["frequency"] = merged["claim_count"] / merged["exposure"]
        merged["severity"] = merged["loss"] / merged["claim_count"].replace(0, np.nan)

        results["frequency"] = TrendAnalysis(
            merged.rename(columns={"frequency": "value"})[["year", "value"]].dropna(),
            metric_name="Frequency",
        )
        results["severity"] = TrendAnalysis(
            merged.rename(columns={"severity": "value"})[["year", "value"]].dropna(),
            metric_name="Severity",
        )

    return results
