"""
auto_actuary.analytics.triangles.tail
======================================
Tail factor estimation methods.

The tail factor represents development from the last observed age to ultimate.
A tail > 1.000 means losses are still expected to develop even after the last
available data point.

Methods implemented:
  1. Inverse Power Curve: LDF(t) = a * t^b  →  integrated to ultimate
  2. Exponential Decay:   LDF(t) = a * exp(b * t) where b < 0
  3. User-specified (no fitting)

Reference: Friedland (2010), Chapter 13 — Tail Factors
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curve functions
# ---------------------------------------------------------------------------

def _inverse_power(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """LDF(t) = a * t^b  (b should be negative for declining LDFs)."""
    return a * np.power(t, b)


def _exponential(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """LDF(t) = a * exp(b * t)  (b should be negative)."""
    return a * np.exp(b * t)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_tail(
    ages: np.ndarray,
    ldfs: np.ndarray,
    curve: str = "inverse_power",
    threshold: float = 1.005,
    max_age: float = 360.0,
) -> float:
    """
    Fit a curve to observed LDFs and extrapolate to ultimate.

    Parameters
    ----------
    ages : array-like
        Midpoint ages (in months) corresponding to each LDF step.
    ldfs : array-like
        Observed LDF values (should be > 1.0).
    curve : str
        'inverse_power' | 'exponential'
    threshold : float
        If the computed tail < threshold, return 1.000.
    max_age : float
        Integration stops at this age (months).  Default 360 = 30 years.

    Returns
    -------
    float
        Tail factor ≥ 1.000
    """
    ages = np.asarray(ages, dtype=float)
    ldfs = np.asarray(ldfs, dtype=float)

    # Drop NaN and enforce LDF > 1 for meaningful tail (LDFs very close to 1 are noise)
    mask = np.isfinite(ldfs) & (ldfs > 1.0) & np.isfinite(ages) & (ages > 0)
    ages_clean = ages[mask]
    ldfs_clean = ldfs[mask]

    if len(ages_clean) < 2:
        logger.warning("Not enough LDF observations to fit tail curve — using tail = 1.0")
        return 1.0

    # Convert LDFs to annualized log-excess over 1.0 for fitting stability
    # We fit ln(LDF - 1) ~ f(age)  which works for standard development patterns
    y = ldfs_clean - 1.0
    log_y = np.log(y)

    try:
        if curve == "inverse_power":
            # ln(LDF-1) = ln(a) + b*ln(t)
            # → linear regression in log-log space
            log_t = np.log(ages_clean)
            b, log_a = np.polyfit(log_t, log_y, 1)
            a = np.exp(log_a)

            # Fitted function: excess LDF(t) = a * t^b
            last_age = ages_clean[-1]

            def excess_ldf(t):
                return a * (t ** b)

        elif curve == "exponential":
            # ln(LDF-1) = ln(a) + b*t
            b, log_a = np.polyfit(ages_clean, log_y, 1)
            a = np.exp(log_a)
            last_age = ages_clean[-1]

            def excess_ldf(t):
                return a * np.exp(b * t)

        else:
            raise ValueError(f"Unknown tail curve: {curve!r}")

        # Tail = product of (1 + excess_ldf(t)) from last_age to max_age
        # Approximate using sum of log LDFs via integration
        # ln(tail) = integral from last_age to max_age of ln(1 + excess_ldf(t)) dt / step
        # Numerical integration over remaining ages
        step = 12.0  # 12-month steps
        proj_ages = np.arange(last_age + step, max_age + step, step)
        proj_ldfs = 1.0 + np.array([max(excess_ldf(t), 0) for t in proj_ages])
        tail = float(np.prod(proj_ldfs))

        if tail < threshold:
            tail = 1.0
            logger.debug("Tail factor %.5f below threshold %.3f → set to 1.000", tail, threshold)
        else:
            logger.info("Tail factor (curve=%s): %.5f", curve, tail)

    except Exception as exc:
        logger.warning("Tail fitting failed: %s — using tail = 1.000", exc)
        tail = 1.0

    return max(1.0, tail)


def benchmark_tail(last_observed_age: int, tail_factors: Optional[dict] = None) -> float:
    """
    Return a benchmark tail factor by development age using industry defaults.

    These are rough CAS study note benchmarks — actuaries should override
    with company-specific or reinsurance cedant experience.

    Parameters
    ----------
    last_observed_age : int
        Last development age in months (e.g., 120 = 10-year tail).
    tail_factors : dict, optional
        Custom lookup {age_months: tail_factor}.
    """
    defaults = {
        12: 2.500, 24: 1.500, 36: 1.200, 48: 1.100,
        60: 1.060, 72: 1.035, 84: 1.020, 96: 1.012,
        108: 1.007, 120: 1.003, 132: 1.001, 144: 1.000,
    }
    lookup = {**defaults, **(tail_factors or {})}

    # Find nearest age ≥ last_observed_age
    valid = {age: tf for age, tf in lookup.items() if age >= last_observed_age}
    if not valid:
        return 1.0
    nearest = min(valid.keys())
    return lookup[nearest]
