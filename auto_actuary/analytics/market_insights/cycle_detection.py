"""
auto_actuary.analytics.market_insights.cycle_detection
=======================================================
Insurance market cycle detection and phase classification.

The P&C insurance market moves through well-documented hard/soft cycles
(Venezian 1985; Cummins & Outreville 1987) driven by:
  - Loss shock events (CAT, social inflation, interest rate changes)
  - Capacity constraints (policyholder surplus levels)
  - Underwriting discipline (combined ratio trends)
  - Rate adequacy (current rates vs. needed rates)

This module synthesises several signals into a Market Cycle Indicator (MCI)
and classifies the current phase.

Market Phases
-------------
  HARD_MARKET   — Rates rising, combined ratios improving, supply restricted
  SOFTENING     — Rates still adequate but starting to decline, competition increases
  SOFT_MARKET   — Rates below adequate levels, combined ratios deteriorating
  HARDENING     — Losses mounting, price discipline returning, rate increases begin

MCI Score
---------
The MCI is a composite score in [−1, +1]:
  +1.0  → deeply hard market (rising rates, improving CR, tight capacity)
  0.0   → neutral / transitioning
  −1.0  → deeply soft market (falling rates, deteriorating CR, excess capacity)

References
----------
- Venezian, E. (1985), "Ratemaking Methods and Profit Cycles in P/L Insurance"
- Cummins, J.D. & Outreville, J.F. (1987), "An International Analysis of
  Underwriting Cycles in Property-Liability Insurance", JRI 54(2)
- CAS Monograph 7 (2005): "Understanding Insurance Market Cycles"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Enumeration of insurance market cycle phases."""
    HARD_MARKET = "hard_market"
    HARDENING = "hardening"
    SOFT_MARKET = "soft_market"
    SOFTENING = "softening"
    INDETERMINATE = "indeterminate"


@dataclass
class CycleSignal:
    """Individual signal contributing to the Market Cycle Indicator."""
    name: str
    value: float        # raw value (e.g., combined ratio = 0.98)
    score: float        # standardised score in [−1, +1]
    direction: str      # 'hard' | 'soft' | 'neutral'
    weight: float       # contribution weight in the MCI
    description: str    # human-readable interpretation


@dataclass
class MarketCycleResult:
    """Full market cycle analysis output."""
    evaluation_year: int
    mci_score: float                    # Composite MCI in [−1, +1]
    phase: MarketPhase
    phase_label: str
    signals: List[CycleSignal]          # Individual contributing signals
    trend_slope: float                  # Trend in MCI over recent years
    cycle_duration_estimate: int        # Estimated years to next phase shift
    narrative: str                      # Plain-English summary
    history: pd.DataFrame               # MCI by year for charting

    @property
    def is_hard(self) -> bool:
        return self.phase in (MarketPhase.HARD_MARKET, MarketPhase.HARDENING)

    @property
    def is_soft(self) -> bool:
        return self.phase in (MarketPhase.SOFT_MARKET, MarketPhase.SOFTENING)


class MarketCycleDetector:
    """
    Detect and classify insurance market cycle phase from historical metrics.

    Parameters
    ----------
    combined_ratios : pd.Series
        Calendar-year combined ratios (e.g., {2018: 0.98, 2019: 1.04, ...}).
        Index = year (int), values = combined ratio (float).
    rate_changes : pd.Series, optional
        Average filed rate changes by year.  Positive = hardening.
        e.g., {2021: 0.08, 2022: 0.12} means +8% and +12% rate increases.
    loss_ratios : pd.Series, optional
        Calendar-year loss ratios.  High LR → soft-market signal.
    surplus_change : pd.Series, optional
        Year-over-year change in industry policyholder surplus (proxy for capacity).
        Positive = expanding capacity → soft signal.
    cat_load : pd.Series, optional
        Catastrophe loss as % of premium by year.  Spikes trigger hardening.
    lob : str
        Line of business label (for output labeling).
    window : int
        Number of years to use for trend computation (default 5).
    """

    # Signal weights in the composite MCI
    _WEIGHTS = {
        "combined_ratio": 0.35,
        "rate_change": 0.30,
        "loss_ratio_trend": 0.20,
        "capacity": 0.10,
        "cat_shock": 0.05,
    }

    def __init__(
        self,
        combined_ratios: pd.Series,
        rate_changes: Optional[pd.Series] = None,
        loss_ratios: Optional[pd.Series] = None,
        surplus_change: Optional[pd.Series] = None,
        cat_load: Optional[pd.Series] = None,
        lob: str = "ALL",
        window: int = 5,
    ) -> None:
        self.lob = lob
        self.window = window
        self.combined_ratios = combined_ratios.sort_index().dropna()
        self.rate_changes = rate_changes.sort_index().dropna() if rate_changes is not None else pd.Series(dtype=float)
        self.loss_ratios = loss_ratios.sort_index().dropna() if loss_ratios is not None else pd.Series(dtype=float)
        self.surplus_change = surplus_change.sort_index().dropna() if surplus_change is not None else pd.Series(dtype=float)
        self.cat_load = cat_load.sort_index().dropna() if cat_load is not None else pd.Series(dtype=float)

        if len(self.combined_ratios) < 3:
            raise ValueError("At least 3 years of combined ratios required for cycle detection.")

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _cr_signal(self, year: int) -> CycleSignal:
        """
        Combined ratio signal.

        CR < 0.97 (profitable)    → hard market signal (+1)
        CR = 1.00 (break-even)   → neutral (0)
        CR > 1.05 (unprofitable) → soft market signal (−1)

        Uses trailing 3-year average to reduce noise.
        """
        recent = self.combined_ratios[self.combined_ratios.index <= year].tail(3)
        cr_avg = float(recent.mean())

        # Piecewise linear: score = clamp((1.02 − CR) / 0.08, −1, +1)
        raw_score = (1.02 - cr_avg) / 0.08
        score = float(np.clip(raw_score, -1.0, 1.0))
        direction = "hard" if score > 0.2 else ("soft" if score < -0.2 else "neutral")

        return CycleSignal(
            name="combined_ratio",
            value=cr_avg,
            score=score,
            direction=direction,
            weight=self._WEIGHTS["combined_ratio"],
            description=f"3yr avg combined ratio {cr_avg:.1%}; {'profitable' if cr_avg < 1.0 else 'unprofitable'}",
        )

    def _rate_signal(self, year: int) -> Optional[CycleSignal]:
        """Rate change signal. Positive rate change → hard market."""
        if self.rate_changes.empty:
            return None

        recent = self.rate_changes[self.rate_changes.index <= year].tail(3)
        if recent.empty:
            return None
        rate_avg = float(recent.mean())

        # Clamp: +10% change = fully hard (+1), −10% = fully soft (−1)
        score = float(np.clip(rate_avg / 0.10, -1.0, 1.0))
        direction = "hard" if score > 0.1 else ("soft" if score < -0.1 else "neutral")

        return CycleSignal(
            name="rate_change",
            value=rate_avg,
            score=score,
            direction=direction,
            weight=self._WEIGHTS["rate_change"],
            description=f"3yr avg rate change {rate_avg:+.1%}; {'hardening' if rate_avg > 0 else 'softening'}",
        )

    def _loss_trend_signal(self, year: int) -> Optional[CycleSignal]:
        """
        Loss ratio trend signal.  Deteriorating LR trend → soft-market precursor.
        Uses log-linear trend slope over *window* years.
        """
        if self.loss_ratios.empty:
            return None

        recent = self.loss_ratios[self.loss_ratios.index <= year].tail(self.window)
        if len(recent) < 3:
            return None

        x = np.arange(len(recent), dtype=float)
        y = np.log(recent.values.astype(float))
        slope, _, _, p, _ = stats.linregress(x, y)

        # Positive slope = LR rising = soft signal
        # Annual change of +5% = fully soft (score = −1)
        score = float(np.clip(-slope / 0.05, -1.0, 1.0))
        trend_pct = float(np.exp(slope) - 1.0)
        direction = "hard" if score > 0.2 else ("soft" if score < -0.2 else "neutral")

        return CycleSignal(
            name="loss_ratio_trend",
            value=trend_pct,
            score=score,
            direction=direction,
            weight=self._WEIGHTS["loss_ratio_trend"],
            description=(
                f"Loss ratio annual trend {trend_pct:+.1%}/yr "
                f"({'deteriorating' if trend_pct > 0 else 'improving'})"
                + (f", p={p:.2f}" if p < 0.2 else " [low significance]")
            ),
        )

    def _capacity_signal(self, year: int) -> Optional[CycleSignal]:
        """
        Capacity signal from surplus change.

        Expanding surplus → more capacity → soft-market signal (−).
        Contracting surplus → less capacity → hard-market signal (+).
        """
        if self.surplus_change.empty:
            return None

        recent = self.surplus_change[self.surplus_change.index <= year].tail(3)
        if recent.empty:
            return None
        surplus_avg = float(recent.mean())

        # Surplus growing >10%/yr = very soft capacity (score = −1)
        # Surplus shrinking >10%/yr = tight capacity (score = +1)
        score = float(np.clip(-surplus_avg / 0.10, -1.0, 1.0))
        direction = "hard" if score > 0.2 else ("soft" if score < -0.2 else "neutral")

        return CycleSignal(
            name="capacity",
            value=surplus_avg,
            score=score,
            direction=direction,
            weight=self._WEIGHTS["capacity"],
            description=f"Surplus change {surplus_avg:+.1%}/yr; {'expanding' if surplus_avg > 0 else 'contracting'} capacity",
        )

    def _cat_signal(self, year: int) -> Optional[CycleSignal]:
        """
        CAT shock signal.  Large CAT year = market-hardening catalyst.
        """
        if self.cat_load.empty:
            return None

        if year not in self.cat_load.index:
            return None

        cat_pct = float(self.cat_load[year])
        historical_avg = float(self.cat_load[self.cat_load.index < year].mean()) if len(self.cat_load) > 1 else cat_pct

        # CAT exceeding 2× historical average = strong hardening signal
        deviation = (cat_pct - historical_avg) / max(historical_avg, 0.01)
        score = float(np.clip(deviation, -1.0, 1.0))
        direction = "hard" if score > 0.3 else "neutral"

        return CycleSignal(
            name="cat_shock",
            value=cat_pct,
            score=score,
            direction=direction,
            weight=self._WEIGHTS["cat_shock"],
            description=f"CAT load {cat_pct:.1%} vs. avg {historical_avg:.1%} — {deviation:+.1f}× deviation",
        )

    # ------------------------------------------------------------------
    # Phase classification
    # ------------------------------------------------------------------

    def _classify_phase(self, mci: float, mci_slope: float) -> Tuple[MarketPhase, str]:
        """
        Classify the market phase from the MCI score and its recent trend.

        The trend slope tells us whether MCI is moving toward hard or soft.
        """
        if mci > 0.4 and mci_slope >= 0:
            return MarketPhase.HARD_MARKET, "Hard Market: rates above adequate, combined ratios profitable, capacity constrained"
        elif mci > 0.1 or (mci > -0.1 and mci_slope > 0.05):
            return MarketPhase.HARDENING, "Hardening: rates rising, underwriting discipline returning after soft period"
        elif mci < -0.4 and mci_slope <= 0:
            return MarketPhase.SOFT_MARKET, "Soft Market: rates below adequate, combined ratios deteriorating, excess capacity"
        elif mci < -0.1 or (mci > -0.4 and mci_slope < -0.05):
            return MarketPhase.SOFTENING, "Softening: rate adequacy eroding, competition increasing, capacity expanding"
        else:
            return MarketPhase.INDETERMINATE, "Indeterminate: mixed signals, market in transition"

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyse(self, evaluation_year: Optional[int] = None) -> MarketCycleResult:
        """
        Run the market cycle analysis for a given evaluation year.

        Parameters
        ----------
        evaluation_year : int, optional
            Year to evaluate.  Defaults to most recent year in combined_ratios.

        Returns
        -------
        MarketCycleResult
        """
        if evaluation_year is None:
            evaluation_year = int(self.combined_ratios.index.max())

        # Compute signals
        signals: List[CycleSignal] = []
        cr_sig = self._cr_signal(evaluation_year)
        signals.append(cr_sig)

        for sig_fn in [self._rate_signal, self._loss_trend_signal,
                       self._capacity_signal, self._cat_signal]:
            sig = sig_fn(evaluation_year)
            if sig is not None:
                signals.append(sig)

        # Composite MCI = Σ(weight × score) / Σ(weight)
        total_weight = sum(s.weight for s in signals)
        if total_weight > 0:
            mci = float(sum(s.score * s.weight for s in signals) / total_weight)
        else:
            mci = 0.0

        # Build historical MCI series
        history_rows = []
        years = sorted(self.combined_ratios.index)
        for yr in years:
            # Compute MCI for each year with available data
            yr_signals = [self._cr_signal(yr)]
            for sig_fn in [self._rate_signal, self._loss_trend_signal,
                           self._capacity_signal, self._cat_signal]:
                s = sig_fn(yr)
                if s is not None:
                    yr_signals.append(s)
            yr_tw = sum(s.weight for s in yr_signals)
            yr_mci = float(sum(s.score * s.weight for s in yr_signals) / yr_tw) if yr_tw > 0 else 0.0
            history_rows.append({"year": yr, "mci_score": yr_mci})
        history = pd.DataFrame(history_rows).set_index("year")

        # MCI trend over recent window (slope tells direction of movement)
        recent_history = history.tail(self.window)
        if len(recent_history) >= 2:
            x = np.arange(len(recent_history), dtype=float)
            slope, _, _, _, _ = stats.linregress(x, recent_history["mci_score"].values)
            mci_slope = float(slope)
        else:
            mci_slope = 0.0

        # Phase classification
        phase, phase_label = self._classify_phase(mci, mci_slope)

        # Cycle duration estimate (empirical: avg P&C cycle ≈ 5−7 years; Venezian 1985)
        # Distance from phase boundary / rate of change
        hard_boundary = 0.2 if phase in (MarketPhase.SOFT_MARKET, MarketPhase.SOFTENING) else -0.2
        if abs(mci_slope) > 0.01:
            years_to_shift = abs((hard_boundary - mci) / mci_slope)
            cycle_est = int(np.clip(years_to_shift, 1, 15))
        else:
            cycle_est = 5  # default when trend is flat

        # Narrative
        signal_summary = "; ".join(
            f"{s.name.replace('_', ' ')} ({s.direction})" for s in signals
        )
        narrative = (
            f"{self.lob} market cycle — {evaluation_year}: MCI={mci:+.2f}, "
            f"Phase: {phase.value.replace('_', ' ').title()}. "
            f"Signals: {signal_summary}. "
            f"Trend: MCI moving {'positively' if mci_slope > 0 else 'negatively'} "
            f"at {mci_slope:+.3f}/yr. Est. {cycle_est}yr to next phase shift."
        )

        logger.info("Market cycle: year=%d, MCI=%.2f, phase=%s", evaluation_year, mci, phase.value)

        return MarketCycleResult(
            evaluation_year=evaluation_year,
            mci_score=mci,
            phase=phase,
            phase_label=phase_label,
            signals=signals,
            trend_slope=mci_slope,
            cycle_duration_estimate=cycle_est,
            narrative=narrative,
            history=history,
        )

    def phase_history(self) -> pd.DataFrame:
        """Return year-by-year phase classification for charting."""
        years = sorted(self.combined_ratios.index)
        rows = []
        for yr in years:
            try:
                result = self.analyse(evaluation_year=yr)
                rows.append({
                    "year": yr,
                    "mci_score": result.mci_score,
                    "phase": result.phase.value,
                    "combined_ratio": float(self.combined_ratios.get(yr, np.nan)),
                })
            except Exception:
                pass
        return pd.DataFrame(rows).set_index("year")

    def __repr__(self) -> str:
        latest = self.analyse()
        return (
            f"MarketCycleDetector(lob={self.lob!r}, "
            f"year={latest.evaluation_year}, "
            f"MCI={latest.mci_score:+.2f}, "
            f"phase={latest.phase.value})"
        )
