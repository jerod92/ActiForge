"""
auto_actuary.analytics.market_insights.anomaly_detection
=========================================================
Statistical detection of loss trend breaks and emerging loss environment changes.

Early detection of adverse loss development is critical for:
  - Reserving: adjust IBNR estimates before full development is visible
  - Ratemaking: incorporate emerging trends before they contaminate historical data
  - CAT management: identify geographic or peril clustering
  - Fraud/litigation: detect claim frequency spikes before they become chronic

Methods implemented
-------------------
1. CUSUM (Cumulative Sum)
   Sequential change-point detection.  Accumulates deviations from the
   historical mean; signals when the cumulative sum exceeds a threshold.
   Classical Page-Hinkley CUSUM (Page 1954).

2. Chow Structural Break Test
   Tests for a structural break (shift in mean and/or trend) at each potential
   breakpoint.  Reports the most likely breakpoint and its F-statistic.
   (Chow 1960; Andrews 1993 for unknown breakpoints)

3. Z-Score Outlier Detection
   Simple robust outlier flagging using median absolute deviation (MAD)
   instead of standard deviation (more robust to heavy tails in loss data).

4. Exponentially Weighted Moving Average (EWMA)
   Industry-standard control chart for process monitoring.  Signals when
   the EWMA crosses ±k standard deviations from the historical mean.

References
----------
- Page, E.S. (1954), "Continuous Inspection Schemes", Biometrika 41(1/2)
- Chow, G.C. (1960), "Tests of Equality Between Sets of Coefficients",
  Econometrica 28(3)
- Andrews, D.W.K. (1993), "Tests for Parameter Instability and Structural Change",
  Econometrica 61(4)
- Montgomery, D.C. (2020) "Introduction to Statistical Quality Control", §9.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of a single anomaly detection test."""
    method: str                      # 'cusum' | 'chow' | 'z_score' | 'ewma'
    metric_name: str
    signal_detected: bool            # True if anomaly/change-point found
    signal_year: Optional[int]       # Year when signal first triggered (if applicable)
    signal_magnitude: float          # Effect size or test statistic
    p_value: Optional[float]         # Statistical significance where applicable
    direction: str                   # 'adverse' | 'favorable' | 'neutral'
    confidence: float                # Confidence in the signal (0–1)
    details: Dict[str, object] = field(default_factory=dict)
    narrative: str = ""

    @property
    def is_significant(self) -> bool:
        return self.signal_detected and self.confidence >= 0.80


@dataclass
class LossAnomalyReport:
    """Full anomaly detection report for a loss metric series."""
    metric_name: str
    evaluation_year: int
    results: List[AnomalyResult]
    overall_signal: bool              # True if ≥2 methods agree on anomaly
    severity: str                     # 'critical' | 'elevated' | 'monitoring' | 'normal'
    recommended_actions: List[str]
    data: pd.DataFrame               # Input data + test statistics by year


class LossAnomalyDetector:
    """
    Detect statistical anomalies in loss metrics (pure premium, frequency, severity).

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time series indexed by year.  If DataFrame, must have a 'value' column.
        If a 'weight' column is present, it is used for weighted statistics.
    metric_name : str
        Label for the metric (e.g., 'pure_premium', 'frequency').
    baseline_years : int
        Number of years to use as the stable baseline.  Default 5.
    cusum_k : float
        CUSUM reference value (half the minimum detectable shift in sigma units).
        Typical: 0.5 (detect 1-sigma shift quickly).
    cusum_h : float
        CUSUM decision boundary in sigma units.  Typical: 4–5 (h=5 → ~0.005 ARL₀ false-alarm rate).
    ewma_lambda : float
        EWMA smoothing constant λ ∈ (0, 1].  λ=0.2 is standard for ARL₀ ≈ 370.
    ewma_L : float
        EWMA control limit multiplier.  L=3 gives 3σ limits.
    """

    def __init__(
        self,
        data: pd.Series | pd.DataFrame,
        metric_name: str = "pure_premium",
        baseline_years: int = 5,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        ewma_lambda: float = 0.20,
        ewma_L: float = 3.0,
    ) -> None:
        if isinstance(data, pd.Series):
            self.df = data.reset_index()
            self.df.columns = ["year", "value"]
        else:
            self.df = data.copy()

        self.df = self.df.sort_values("year").dropna(subset=["value"])
        self.df = self.df[self.df["value"] > 0]

        self.metric_name = metric_name
        self.baseline_years = baseline_years
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.ewma_lambda = ewma_lambda
        self.ewma_L = ewma_L

        if len(self.df) < 4:
            raise ValueError(f"At least 4 data points required; got {len(self.df)}")

        # Robust baseline statistics (median-based to handle skewness)
        baseline = self.df.head(self.baseline_years)["value"]
        self._mu0 = float(baseline.median())
        self._sigma0 = float(
            1.4826 * (baseline - baseline.median()).abs().median()  # MAD estimator
        )
        if self._sigma0 < 1e-10:
            self._sigma0 = float(baseline.std())  # fallback to std

    # ------------------------------------------------------------------
    # CUSUM
    # ------------------------------------------------------------------

    def _cusum(self) -> AnomalyResult:
        """
        Two-sided CUSUM for detecting shifts in the mean.

        The log-transformed series is tested (appropriate for multiplicative
        loss processes): x_t = ln(PP_t / mu0)

        S_t+ = max(0, S_{t-1}+ + z_t − k)   [upper: adverse trend]
        S_t− = max(0, S_{t-1}− − z_t − k)   [lower: favorable trend]

        Signal: |S_t| > h
        """
        values = self.df["value"].values.astype(float)
        years = self.df["year"].values.astype(int)
        n = len(values)

        mu_ln = float(np.log(self._mu0))
        sigma_ln = float(self._sigma0 / self._mu0)  # delta method: σ_ln ≈ σ/μ
        if sigma_ln < 1e-10:
            sigma_ln = float(np.log(values).std())

        k = self.cusum_k
        h = self.cusum_h

        s_plus = np.zeros(n)
        s_minus = np.zeros(n)

        for t in range(1, n):
            z_t = (np.log(values[t]) - mu_ln) / sigma_ln
            s_plus[t] = max(0, s_plus[t-1] + z_t - k)
            s_minus[t] = max(0, s_minus[t-1] - z_t - k)

        signal_idx_plus = np.argmax(s_plus > h) if np.any(s_plus > h) else -1
        signal_idx_minus = np.argmax(s_minus > h) if np.any(s_minus > h) else -1

        if signal_idx_plus > 0 and (signal_idx_minus < 0 or s_plus[signal_idx_plus] > s_minus[signal_idx_minus]):
            signal_detected = True
            signal_year = int(years[signal_idx_plus])
            direction = "adverse"
            magnitude = float(s_plus[signal_idx_plus] / h)
        elif signal_idx_minus > 0:
            signal_detected = True
            signal_year = int(years[signal_idx_minus])
            direction = "favorable"
            magnitude = float(s_minus[signal_idx_minus] / h)
        else:
            signal_detected = False
            signal_year = None
            direction = "neutral"
            magnitude = float(max(s_plus.max(), s_minus.max()) / h)

        confidence = float(np.clip(magnitude, 0, 1)) if signal_detected else 0.0

        narrative = ""
        if signal_detected:
            narrative = (
                f"CUSUM detected {direction} shift in {self.metric_name} "
                f"starting {signal_year}. Signal strength {magnitude:.1f}× threshold."
            )

        return AnomalyResult(
            method="cusum",
            metric_name=self.metric_name,
            signal_detected=signal_detected,
            signal_year=signal_year,
            signal_magnitude=magnitude,
            p_value=None,  # CUSUM doesn't produce a p-value directly
            direction=direction,
            confidence=confidence,
            details={
                "s_plus_max": float(s_plus.max()),
                "s_minus_max": float(s_minus.max()),
                "threshold_h": h,
                "s_plus": s_plus.tolist(),
                "s_minus": s_minus.tolist(),
                "years": years.tolist(),
            },
            narrative=narrative,
        )

    # ------------------------------------------------------------------
    # Chow structural break test
    # ------------------------------------------------------------------

    def _chow_test(self) -> AnomalyResult:
        """
        Chow (1960) test for structural break in log-linear trend.

        Tests each possible break point t* in the middle 60% of the sample.
        The break point with the highest F-statistic is selected.

        F = [(RSS_restricted - RSS_unrestricted) / k] / [RSS_unrestricted / (n - 2k)]
        where k = number of parameters (= 2 for intercept + slope).
        """
        df = self.df.copy()
        x = df["year"].values.astype(float)
        y = np.log(df["value"].values.astype(float))
        n = len(x)

        if n < 8:
            return AnomalyResult(
                method="chow",
                metric_name=self.metric_name,
                signal_detected=False,
                signal_year=None,
                signal_magnitude=0.0,
                p_value=1.0,
                direction="neutral",
                confidence=0.0,
                narrative="Insufficient data for Chow test (need ≥ 8 years).",
            )

        # Restricted model (no break)
        x_c = x - x.min()
        slope_r, int_r, _, _, _ = stats.linregress(x_c, y)
        y_hat_r = int_r + slope_r * x_c
        rss_r = float(np.sum((y - y_hat_r) ** 2))

        k = 2  # intercept + slope

        # Search breakpoints in middle 40% to 60% of sample
        lo = max(k + 1, int(n * 0.30))
        hi = min(n - k - 1, int(n * 0.70))

        best_f = 0.0
        best_bp = None
        best_p = 1.0

        for bp in range(lo, hi + 1):
            x1, y1 = x_c[:bp], y[:bp]
            x2, y2 = x_c[bp:], y[bp:]

            if len(x1) < 3 or len(x2) < 3:
                continue

            s1, i1, _, _, _ = stats.linregress(x1, y1)
            s2, i2, _, _, _ = stats.linregress(x2, y2)

            rss1 = float(np.sum((y1 - (i1 + s1 * x1)) ** 2))
            rss2 = float(np.sum((y2 - (i2 + s2 * x2)) ** 2))
            rss_u = rss1 + rss2

            if rss_u < 1e-12:
                continue

            f_stat = ((rss_r - rss_u) / k) / (rss_u / max(n - 2 * k, 1))
            p_val = float(1 - stats.f.cdf(f_stat, dfn=k, dfd=max(n - 2 * k, 1)))

            if f_stat > best_f:
                best_f = f_stat
                best_bp = bp
                best_p = p_val

        signal_detected = best_p < 0.10 if best_bp is not None else False
        signal_year = int(x[best_bp]) if best_bp is not None else None

        if signal_detected and best_bp is not None:
            # Assess direction of break
            slope_pre, _, _, _, _ = stats.linregress(x_c[:best_bp], y[:best_bp])
            slope_post, _, _, _, _ = stats.linregress(x_c[best_bp:], y[best_bp:])
            direction = "adverse" if slope_post > slope_pre else "favorable"
            magnitude = best_f / 10.0  # normalise (F>10 = strong break)
            confidence = float(np.clip(1.0 - best_p, 0, 1))
            narrative = (
                f"Chow test detects structural break at {signal_year} "
                f"(F={best_f:.2f}, p={best_p:.3f}). Trend shifted "
                f"{'adversely' if direction=='adverse' else 'favorably'}: "
                f"{np.exp(slope_pre):.3f}→{np.exp(slope_post):.3f} annual trend."
            )
        else:
            direction = "neutral"
            magnitude = best_f / 10.0 if best_f else 0.0
            confidence = 0.0
            narrative = f"Chow test: no significant structural break detected (best F={best_f:.2f}, p={best_p:.3f})."

        return AnomalyResult(
            method="chow",
            metric_name=self.metric_name,
            signal_detected=signal_detected,
            signal_year=signal_year,
            signal_magnitude=magnitude,
            p_value=best_p,
            direction=direction,
            confidence=confidence,
            details={"f_statistic": best_f, "breakpoint_index": best_bp},
            narrative=narrative,
        )

    # ------------------------------------------------------------------
    # Z-Score (MAD-based)
    # ------------------------------------------------------------------

    def _z_score(self) -> AnomalyResult:
        """
        Robust outlier detection using Modified Z-Score (Iglewicz & Hoaglin 1993).

        Modified Z_i = 0.6745 × (x_i − median) / MAD

        |Z| > 3.5 → outlier (analogous to 3.5σ in Normal distribution).
        Uses log-transformed values to handle multiplicative loss processes.
        """
        values = self.df["value"].values.astype(float)
        years = self.df["year"].values.astype(int)
        log_vals = np.log(values)

        med = np.median(log_vals)
        mad = np.median(np.abs(log_vals - med))
        if mad < 1e-10:
            mad = np.std(log_vals)

        z_scores = 0.6745 * (log_vals - med) / mad

        # Flag the most recent year as the signal (if it's an outlier)
        recent_z = z_scores[-1]
        most_recent_year = int(years[-1])

        threshold = 3.0
        signal_detected = abs(recent_z) > threshold
        signal_year = most_recent_year if signal_detected else None
        direction = "adverse" if recent_z > threshold else ("favorable" if recent_z < -threshold else "neutral")
        magnitude = abs(recent_z) / threshold
        confidence = float(np.clip(abs(recent_z) / (threshold + 2), 0, 1))

        outlier_years = years[np.abs(z_scores) > threshold].tolist()

        narrative = ""
        if signal_detected:
            narrative = (
                f"{self.metric_name} at {most_recent_year} is a statistical outlier "
                f"(modified Z={recent_z:+.2f}, threshold ±{threshold}). "
                f"{'Well above' if direction=='adverse' else 'Well below'} historical range. "
                f"All outlier years: {outlier_years}."
            )

        return AnomalyResult(
            method="z_score",
            metric_name=self.metric_name,
            signal_detected=signal_detected,
            signal_year=signal_year,
            signal_magnitude=magnitude,
            p_value=None,
            direction=direction,
            confidence=confidence,
            details={
                "z_scores": dict(zip(years.tolist(), z_scores.tolist())),
                "outlier_years": outlier_years,
                "threshold": threshold,
            },
            narrative=narrative,
        )

    # ------------------------------------------------------------------
    # EWMA control chart
    # ------------------------------------------------------------------

    def _ewma(self) -> AnomalyResult:
        """
        EWMA control chart for loss trend monitoring.

        z_t = λ × x_t + (1 − λ) × z_{t−1}

        Control limits: z_0 ± L × σ × sqrt(λ / (2 − λ))

        Uses log-transformed values.
        """
        values = self.df["value"].values.astype(float)
        years = self.df["year"].values.astype(int)
        log_vals = np.log(values)

        lam = self.ewma_lambda
        L = self.ewma_L
        n = len(log_vals)

        # Baseline from first baseline_years observations
        n_base = min(self.baseline_years, n - 1)
        mu0 = float(np.mean(log_vals[:n_base]))
        sigma0 = float(np.std(log_vals[:n_base]))
        if sigma0 < 1e-10:
            sigma0 = 0.1

        # EWMA statistic
        ewma = np.zeros(n)
        ewma[0] = mu0
        for t in range(1, n):
            ewma[t] = lam * log_vals[t] + (1 - lam) * ewma[t - 1]

        # Asymptotic control limits
        cl = L * sigma0 * np.sqrt(lam / (2 - lam))
        ucl = mu0 + cl
        lcl = mu0 - cl

        # Detect first out-of-control point
        in_control = (ewma >= lcl) & (ewma <= ucl)
        out_of_control = ~in_control

        if np.any(out_of_control[n_base:]):
            first_oc = n_base + int(np.argmax(out_of_control[n_base:]))
            signal_detected = True
            signal_year = int(years[first_oc])
            direction = "adverse" if ewma[first_oc] > ucl else "favorable"
            # Effect size: multiples of control limit width
            limit_distance = abs(ewma[first_oc] - mu0)
            magnitude = float(limit_distance / cl)
            confidence = float(np.clip((limit_distance - cl) / cl, 0, 1))
            narrative = (
                f"EWMA control chart: {self.metric_name} went out of control in "
                f"{signal_year} ({'above UCL' if direction=='adverse' else 'below LCL'}). "
                f"Signal strength {magnitude:.1f}× control limit."
            )
        else:
            signal_detected = False
            signal_year = None
            direction = "neutral"
            magnitude = float(max(np.abs(ewma[n_base:] - mu0)) / cl) if cl > 0 else 0.0
            confidence = 0.0
            narrative = f"EWMA: {self.metric_name} remains within control limits."

        return AnomalyResult(
            method="ewma",
            metric_name=self.metric_name,
            signal_detected=signal_detected,
            signal_year=signal_year,
            signal_magnitude=magnitude,
            p_value=None,
            direction=direction,
            confidence=confidence,
            details={
                "ewma": dict(zip(years.tolist(), ewma.tolist())),
                "ucl": ucl,
                "lcl": lcl,
                "lambda": lam,
                "L": L,
            },
            narrative=narrative,
        )

    # ------------------------------------------------------------------
    # Master analysis
    # ------------------------------------------------------------------

    def analyse(self, evaluation_year: Optional[int] = None) -> LossAnomalyReport:
        """
        Run all anomaly detection methods and synthesise results.

        Parameters
        ----------
        evaluation_year : int, optional
            Restrict data to up to and including this year.

        Returns
        -------
        LossAnomalyReport
        """
        year_filtered = evaluation_year is not None
        if year_filtered:
            working_df = self.df[self.df["year"] <= evaluation_year].copy()
            if len(working_df) < 4:
                raise ValueError("Insufficient data after year filter.")
            orig_df = self.df
            self.df = working_df
        else:
            evaluation_year = int(self.df["year"].max())

        # Run all tests
        results = [
            self._cusum(),
            self._chow_test(),
            self._z_score(),
            self._ewma(),
        ]

        if year_filtered:
            self.df = orig_df

        # Synthesise: how many methods signal?
        n_signals = sum(1 for r in results if r.signal_detected)
        avg_confidence = float(np.mean([r.confidence for r in results]))

        # Overall signal: ≥ 2 methods agree
        overall_signal = n_signals >= 2

        # Severity classification
        if n_signals >= 3 and avg_confidence >= 0.70:
            severity = "critical"
        elif n_signals >= 2 and avg_confidence >= 0.50:
            severity = "elevated"
        elif n_signals >= 1:
            severity = "monitoring"
        else:
            severity = "normal"

        # Recommended actions
        actions = []
        if severity == "critical":
            actions = [
                "Immediately review reserve adequacy — significant adverse development possible",
                "Place affected segments on underwriting watch list",
                "Consider emergency rate action in impacted territories/classes",
                "Flag for management board discussion",
            ]
        elif severity == "elevated":
            actions = [
                "Review reserve margins in affected accident years",
                "Accelerate next ratemaking cycle",
                "Investigate claim drivers: social inflation, litigation climate, new loss sources",
            ]
        elif severity == "monitoring":
            actions = [
                "Continue quarterly monitoring of the flagged metric",
                "Ensure next trend analysis uses full data including recent years",
            ]
        else:
            actions = ["No action required — metric within normal historical range."]

        # Build diagnostic data table
        data_df = self.df.copy()
        cusum_details = [r for r in results if r.method == "cusum"]
        if cusum_details:
            c = cusum_details[0]
            years_list = c.details.get("years", [])
            s_plus = c.details.get("s_plus", [])
            s_minus = c.details.get("s_minus", [])
            if len(years_list) == len(data_df):
                data_df["cusum_s_plus"] = s_plus
                data_df["cusum_s_minus"] = s_minus

        ewma_details = [r for r in results if r.method == "ewma"]
        if ewma_details:
            e = ewma_details[0]
            ewma_vals = e.details.get("ewma", {})
            data_df["ewma"] = data_df["year"].map(ewma_vals)
            data_df["ewma_ucl"] = e.details.get("ucl", np.nan)
            data_df["ewma_lcl"] = e.details.get("lcl", np.nan)

        z_details = [r for r in results if r.method == "z_score"]
        if z_details:
            z_scores_map = z_details[0].details.get("z_scores", {})
            data_df["z_score"] = data_df["year"].map(z_scores_map)

        logger.info(
            "Loss anomaly detection: metric=%s, year=%d, severity=%s, n_signals=%d",
            self.metric_name, evaluation_year, severity, n_signals,
        )

        return LossAnomalyReport(
            metric_name=self.metric_name,
            evaluation_year=evaluation_year,
            results=results,
            overall_signal=overall_signal,
            severity=severity,
            recommended_actions=actions,
            data=data_df,
        )

    def __repr__(self) -> str:
        return (
            f"LossAnomalyDetector(metric={self.metric_name!r}, "
            f"n={len(self.df)}, "
            f"baseline_mu={self._mu0:.4f})"
        )
