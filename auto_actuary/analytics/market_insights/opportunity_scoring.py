"""
auto_actuary.analytics.market_insights.opportunity_scoring
===========================================================
Segment opportunity scoring for strategic portfolio management.

Each portfolio segment (territory × class × LOB) is scored on a multi-
dimensional opportunity matrix that reveals where the carrier should:
  - Grow aggressively (high profitability + high growth potential)
  - Manage carefully (profitable but saturated)
  - Re-price or exit (unprofitable, no path to adequacy)
  - Watch selectively (marginal profitability, some growth potential)

The OpportunityScore synthesises five dimensions into a single actionable score:

  1. Profitability Score   — current loss ratio vs. permissible
  2. Rate Adequacy Score   — indicated rate change (from ratemaking analysis)
  3. Growth Potential Score— volume trend (exposure growth) and market penetration
  4. Retention Score       — policy renewal retention rate
  5. Risk Concentration Score — HHI-based concentration risk (too much in one segment)

Final Score: weighted combination in [0, 100]
  ≥ 80 = Premium opportunity (grow)
  60–80 = Solid performer (maintain / optimise)
  40–60 = Marginal (monitor closely, re-price)
  < 40  = Challenged (exit or remediate)

References
----------
- McKinsey Insurance Practice (2021) "Growth in P&C insurance"
- CAS Research Paper: "Segmentation and Risk Selection" (2019)
- Herfindahl-Hirschman Index (HHI) for concentration measurement
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
class OpportunityScore:
    """Opportunity score for a single portfolio segment."""
    segment_key: str              # e.g., "SOUTHEAST|class_A"
    segment_values: Dict[str, str]  # {"territory": "SOUTHEAST", "class_code": "A"}
    total_score: float            # 0–100 composite
    grade: str                    # "Premium" | "Solid" | "Marginal" | "Challenged"
    action: str                   # strategic action label

    # Dimension scores (0–100 each)
    profitability_score: float
    rate_adequacy_score: float
    growth_potential_score: float
    retention_score: float
    concentration_score: float   # 100 = well diversified, 0 = dangerously concentrated

    # Raw metrics for transparency
    loss_ratio: float
    indicated_rate_change: float  # from ratemaking
    exposure_trend: float         # annual % change in earned exposure
    retention_rate: float
    hhi: float                    # Herfindahl-Hirschman Index (0–1)

    # Additional context
    earned_premium: float
    earned_exposure: float
    notes: List[str] = field(default_factory=list)

    @property
    def is_opportunity(self) -> bool:
        return self.total_score >= 70

    @property
    def is_challenged(self) -> bool:
        return self.total_score < 40

    def __repr__(self) -> str:
        return (
            f"OpportunityScore({self.segment_key}: {self.total_score:.1f}/100 "
            f"[{self.grade}] — {self.action})"
        )


class SegmentOpportunityScorer:
    """
    Score portfolio segments on a multi-dimensional opportunity matrix.

    Parameters
    ----------
    segment_metrics : pd.DataFrame
        One row per segment.  Required columns:
            segment_key    — unique string identifier for the segment
            earned_premium — total earned premium in the segment
            earned_exposure— total earned exposure
            loss_ratio     — historical loss ratio (LR = losses / premium)
        Optional but strongly recommended:
            indicated_rate_change — from ratemaking analysis (e.g., 0.08 = +8%)
            retention_rate        — policy renewal retention (0–1)
            exposure_trend        — annual % change in exposure (e.g., 0.05 = +5%/yr)
            <dimension columns>   — any categorical columns (territory, class_code, lob)
    permissible_loss_ratio : float
        Target loss ratio (= 1 − V − Q).  Default 0.65.
    dimension_cols : list of str
        Column names to use as segment dimensions for HHI calculation.
    weights : dict, optional
        Override default scoring weights.  Keys: profitability, rate_adequacy,
        growth_potential, retention, concentration.  Must sum to 1.0.
    """

    _DEFAULT_WEIGHTS = {
        "profitability": 0.35,
        "rate_adequacy": 0.25,
        "growth_potential": 0.20,
        "retention": 0.12,
        "concentration": 0.08,
    }

    def __init__(
        self,
        segment_metrics: pd.DataFrame,
        permissible_loss_ratio: float = 0.65,
        dimension_cols: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.df = segment_metrics.copy()
        self.plr = permissible_loss_ratio
        self.dimension_cols = dimension_cols or []

        wts = {**self._DEFAULT_WEIGHTS, **(weights or {})}
        total_w = sum(wts.values())
        self.weights = {k: v / total_w for k, v in wts.items()}  # normalise

        # Compute HHI for each dimension
        self._hhi_by_dim: Dict[str, pd.Series] = {}
        if "earned_premium" in self.df.columns:
            for dim in self.dimension_cols:
                if dim in self.df.columns:
                    self._hhi_by_dim[dim] = self._compute_hhi(dim)

    # ------------------------------------------------------------------
    # HHI concentration
    # ------------------------------------------------------------------

    def _compute_hhi(self, dimension: str) -> pd.Series:
        """
        Compute Herfindahl-Hirschman Index (HHI) for each segment value
        within *dimension*.

        HHI = Σ s_i²  where s_i = market share of sub-segment i.
        HHI = 1.0 → perfect monopoly concentration (bad for risk management)
        HHI → 0   → highly fragmented (diversified)

        US DOJ considers:
          HHI < 0.10 → unconcentrated
          0.10–0.18  → moderately concentrated
          HHI > 0.18 → highly concentrated

        Returns a Series indexed by segment_key with the HHI for that key's
        dimension value.
        """
        if dimension not in self.df.columns or "earned_premium" not in self.df.columns:
            return pd.Series(0.0, index=self.df["segment_key"] if "segment_key" in self.df.columns else self.df.index)

        total_premium = float(self.df["earned_premium"].sum())
        if total_premium <= 0:
            return pd.Series(0.0, index=range(len(self.df)))

        group_shares = (
            self.df.groupby(dimension)["earned_premium"].sum() / total_premium
        )
        hhi_by_group = (group_shares ** 2).to_dict()

        if "segment_key" in self.df.columns:
            return self.df.apply(lambda row: hhi_by_group.get(row.get(dimension), 0.0), axis=1)
        return pd.Series(0.0, index=range(len(self.df)))

    # ------------------------------------------------------------------
    # Individual dimension scoring (0–100)
    # ------------------------------------------------------------------

    def _score_profitability(self, lr: float) -> float:
        """
        Profitability score based on loss ratio vs. permissible.

        score = 100 when LR = PLR × 0.85 (15% better than permissible)
        score = 50  when LR = PLR         (exactly at target)
        score = 0   when LR = PLR × 1.25  (25% worse than permissible)
        """
        if np.isnan(lr) or lr <= 0:
            return 50.0
        ratio = lr / self.plr  # < 1 = profitable
        score = 100 * (1.0 - (ratio - 0.85) / 0.40)
        return float(np.clip(score, 0, 100))

    def _score_rate_adequacy(self, indicated_change: float) -> float:
        """
        Rate adequacy score from the indicated rate change.

        A negative indication (rates too high) → high score (opportunity to grow)
        A positive indication (rates inadequate) → low score (avoid growing)

        score = 100 when indicated = −10% (over-priced → growth opportunity)
        score = 50  when indicated = 0%   (rates adequate)
        score = 0   when indicated = +20% (significantly under-priced)
        """
        if np.isnan(indicated_change):
            return 50.0
        # Linear scale: -0.10 → 100, 0.0 → 50, +0.20 → 0
        score = 50.0 + (-indicated_change / 0.20) * 50.0
        return float(np.clip(score, 0, 100))

    def _score_growth_potential(self, exposure_trend: float) -> float:
        """
        Growth potential from exposure trend.

        A segment growing in exposure signals demand strength.
        +10%/yr trend → score 80  (strong growth market)
        0%/yr         → score 50  (stable)
        −10%/yr       → score 20  (shrinking market)
        """
        if np.isnan(exposure_trend):
            return 50.0
        score = 50.0 + (exposure_trend / 0.10) * 30.0
        return float(np.clip(score, 0, 100))

    def _score_retention(self, retention_rate: float) -> float:
        """
        Retention score.

        >90% retention → score 90+ (loyal book, high lifetime value)
        70% retention  → score 50  (average)
        <50% retention → score 0   (churn problem)
        """
        if np.isnan(retention_rate) or retention_rate <= 0:
            return 50.0
        # score = (retention − 0.50) / (0.45) × 100
        score = (retention_rate - 0.50) / 0.45 * 100
        return float(np.clip(score, 0, 100))

    def _score_concentration(self, hhi: float) -> float:
        """
        Concentration score (inverted: high HHI = bad = low score).

        HHI = 0.05 → score 90  (well diversified)
        HHI = 0.18 → score 50  (moderately concentrated)
        HHI = 0.50 → score 0   (dangerously concentrated)
        """
        if np.isnan(hhi):
            return 50.0
        score = 100 * (1 - (hhi - 0.05) / 0.45)
        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Grade and action
    # ------------------------------------------------------------------

    @staticmethod
    def _grade(total_score: float) -> Tuple[str, str]:
        if total_score >= 80:
            return "Premium", "Grow aggressively — top profitability + growth opportunity"
        elif total_score >= 65:
            return "Solid", "Maintain and optimise — strong performer, selective growth"
        elif total_score >= 50:
            return "Marginal", "Monitor closely — re-price to adequacy, limit new business"
        elif total_score >= 35:
            return "Challenged", "Remediate or exit — below-adequacy, limited growth prospects"
        else:
            return "Critical", "Exit segment — persistent underperformance, no viable path"

    # ------------------------------------------------------------------
    # Main scoring
    # ------------------------------------------------------------------

    def score_all(self) -> pd.DataFrame:
        """
        Compute opportunity scores for all segments.

        Returns
        -------
        pd.DataFrame
            One row per segment, indexed by segment_key.
            Includes total_score, grade, action, and all dimension scores.
        """
        scores: List[OpportunityScore] = []
        seg_key_col = "segment_key" if "segment_key" in self.df.columns else None

        # Portfolio-level HHI for each segment (worst-case across dimensions)
        for idx, row in self.df.iterrows():
            seg_key = str(row.get("segment_key", idx)) if seg_key_col else str(idx)

            lr = float(row.get("loss_ratio", np.nan))
            indicated = float(row.get("indicated_rate_change", np.nan))
            exp_trend = float(row.get("exposure_trend", np.nan))
            retention = float(row.get("retention_rate", np.nan))

            # HHI: use maximum concentration across dimensions (conservative)
            hhi_vals = []
            for dim, hhi_series in self._hhi_by_dim.items():
                if idx in hhi_series.index:
                    hhi_vals.append(float(hhi_series[idx]))
            hhi = max(hhi_vals) if hhi_vals else 0.1

            # Dimension scores
            prof_s = self._score_profitability(lr)
            rate_s = self._score_rate_adequacy(indicated)
            grow_s = self._score_growth_potential(exp_trend)
            ret_s = self._score_retention(retention)
            conc_s = self._score_concentration(hhi)

            # Composite score
            total = (
                self.weights["profitability"] * prof_s
                + self.weights["rate_adequacy"] * rate_s
                + self.weights["growth_potential"] * grow_s
                + self.weights["retention"] * ret_s
                + self.weights["concentration"] * conc_s
            )

            grade, action = self._grade(total)

            # Segment dimension values
            dim_vals = {d: str(row.get(d, "")) for d in self.dimension_cols if d in row}

            # Notes
            notes = []
            if not np.isnan(lr) and lr > self.plr * 1.10:
                notes.append(f"Loss ratio {lr:.1%} exceeds permissible {self.plr:.1%} by {(lr/self.plr-1):.1%}")
            if not np.isnan(indicated) and indicated > 0.05:
                notes.append(f"Rate indication +{indicated:.1%} signals inadequate pricing")
            if not np.isnan(retention) and retention < 0.75:
                notes.append(f"Retention {retention:.1%} below typical threshold 75%")
            if hhi > 0.25:
                notes.append(f"HHI {hhi:.2f} exceeds concentration threshold 0.25")

            scores.append(OpportunityScore(
                segment_key=seg_key,
                segment_values=dim_vals,
                total_score=round(total, 1),
                grade=grade,
                action=action,
                profitability_score=round(prof_s, 1),
                rate_adequacy_score=round(rate_s, 1),
                growth_potential_score=round(grow_s, 1),
                retention_score=round(ret_s, 1),
                concentration_score=round(conc_s, 1),
                loss_ratio=lr,
                indicated_rate_change=indicated,
                exposure_trend=exp_trend,
                retention_rate=retention,
                hhi=hhi,
                earned_premium=float(row.get("earned_premium", np.nan)),
                earned_exposure=float(row.get("earned_exposure", np.nan)),
                notes=notes,
            ))

        result_df = pd.DataFrame([{
            "segment_key": s.segment_key,
            "grade": s.grade,
            "action": s.action,
            "total_score": s.total_score,
            "profitability_score": s.profitability_score,
            "rate_adequacy_score": s.rate_adequacy_score,
            "growth_potential_score": s.growth_potential_score,
            "retention_score": s.retention_score,
            "concentration_score": s.concentration_score,
            "loss_ratio": s.loss_ratio,
            "indicated_rate_change": s.indicated_rate_change,
            "exposure_trend": s.exposure_trend,
            "retention_rate": s.retention_rate,
            "hhi": s.hhi,
            "earned_premium": s.earned_premium,
            "earned_exposure": s.earned_exposure,
            "notes": "; ".join(s.notes),
        } for s in scores]).set_index("segment_key").sort_values("total_score", ascending=False)

        self._scores = scores
        return result_df

    def top_opportunities(self, n: int = 5) -> pd.DataFrame:
        """Return the top *n* segments by opportunity score."""
        scored = self.score_all()
        return scored.head(n)

    def challenged_segments(self, threshold: float = 40.0) -> pd.DataFrame:
        """Return segments with total_score below *threshold*."""
        scored = self.score_all()
        return scored[scored["total_score"] < threshold]

    def opportunity_matrix(self) -> pd.DataFrame:
        """
        Return a 2×2 opportunity matrix pivot (profitability × growth).

        Quadrants:
          Q1 (high profit + high growth) → Stars
          Q2 (high profit + low growth)  → Cash cows
          Q3 (low profit + high growth)  → Problem children (re-price & grow)
          Q4 (low profit + low growth)   → Dogs (exit)
        """
        scored = self.score_all()
        scored["profit_quartile"] = pd.qcut(scored["profitability_score"], 2, labels=["Low Profit", "High Profit"])
        scored["growth_quartile"] = pd.qcut(scored["growth_potential_score"], 2, labels=["Low Growth", "High Growth"])

        matrix = scored.groupby(["profit_quartile", "growth_quartile"], observed=True).agg(
            n_segments=("total_score", "count"),
            avg_score=("total_score", "mean"),
            total_premium=("earned_premium", "sum"),
            avg_loss_ratio=("loss_ratio", "mean"),
        ).round(2)

        return matrix

    def portfolio_health_summary(self) -> Dict[str, object]:
        """Return a scalar summary of portfolio health."""
        scored = self.score_all()
        return {
            "n_segments": int(len(scored)),
            "avg_opportunity_score": float(scored["total_score"].mean()),
            "premium_in_premium_segments": float(
                scored.loc[scored["grade"] == "Premium", "earned_premium"].sum()
            ),
            "premium_in_challenged_segments": float(
                scored.loc[scored["grade"].isin(["Challenged", "Critical"]), "earned_premium"].sum()
            ),
            "pct_premium_grade_or_solid": float(
                scored.loc[scored["grade"].isin(["Premium", "Solid"]), "earned_premium"].sum()
                / scored["earned_premium"].sum()
            ) if scored["earned_premium"].sum() > 0 else np.nan,
            "n_critical": int((scored["grade"] == "Critical").sum()),
            "n_premium_opportunities": int((scored["grade"] == "Premium").sum()),
        }

    def __repr__(self) -> str:
        summary = self.portfolio_health_summary()
        return (
            f"SegmentOpportunityScorer("
            f"n_segments={summary['n_segments']}, "
            f"avg_score={summary['avg_opportunity_score']:.1f}/100)"
        )
