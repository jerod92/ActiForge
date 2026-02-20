"""
auto_actuary.analytics.ratemaking.indicated_rate
=================================================
Rate indication — the ultimate output of the ratemaking process.

The indicated rate change tells management how much rates need to change
(up or down) to achieve the target underwriting profit margin.

Formula
-------
    Permissible Loss Ratio  = 1 − V − Q
    where:
        V = variable expense ratio (commissions, taxes, assessments)
        Q = target profit margin

    Indicated Change = (Projected Loss Ratio / Permissible Loss Ratio) − 1

Projected Loss Ratio
--------------------
    = Σ (Trended, Developed Ultimate Losses) / On-Level Earned Premium

    Each accident year's ultimate losses are:
    1. Developed to ultimate via the selected reserve method
    2. Trended to the future policy period midpoint using loss trend

On-Level Earned Premium
-----------------------
    Historical earned premium restated to current rate level using the
    parallelogram method or extension of exposures.

Credibility
-----------
    Classical credibility:
        Z = min(sqrt(n / n_full), 1.0)
        n_full = 1082 (FCAS standard: ±5% accuracy, 90% CI)
    Indicated change (credibility-weighted):
        = Z × Indicated + (1 − Z) × Complement (e.g., industry trend)

References
----------
- Werner & Modlin (2016) "Basic Ratemaking", Chapters 7–11
- CAS Exam 5 (Ratemaking & Estimating Unpaid Claims)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession
    from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RateIndicationResult:
    """Scalar indication result for a single accident-year range."""
    lob: str
    accident_years: List[int]

    on_level_premium: float
    trended_ultimate_loss: float
    projected_loss_ratio: float

    variable_expense_ratio: float
    fixed_expense_ratio: float
    target_profit_margin: float
    permissible_loss_ratio: float

    indicated_change: float         # e.g., 0.08 = +8%
    credibility: float              # 0–1
    complement_indication: float    # e.g., industry indication
    credibility_weighted_change: float

    # Details
    by_year: Optional[pd.DataFrame] = None  # year-level detail
    trend_factor: float = 1.0
    development_factor: float = 1.0
    notes: List[str] = field(default_factory=list)

    @property
    def indicated_pct(self) -> str:
        return f"{self.indicated_change:+.1%}"

    @property
    def credibility_weighted_pct(self) -> str:
        return f"{self.credibility_weighted_change:+.1%}"

    def __repr__(self) -> str:
        return (
            f"RateIndication(lob={self.lob!r}, "
            f"indicated={self.indicated_pct}, "
            f"cred={self.credibility:.2f}, "
            f"cred_wtd={self.credibility_weighted_pct})"
        )


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class RateIndication:
    """
    Compute a rate indication for a line of business.

    Parameters
    ----------
    on_level_premium_by_year : pd.Series
        On-level earned premium indexed by accident year.
    ultimate_loss_by_year : pd.Series
        Actuarial ultimate losses by accident year (post-development).
    trend_factor : float
        Cumulative loss trend factor to project to future period midpoint.
    premium_trend_factor : float
        Premium trend factor (corrects for limit/deductible drift).
    variable_expense_ratio : float
        V in the formula above.  Default from config.
    fixed_expense_ratio : float
        Fixed expense as % of premium.
    target_profit_margin : float
        Q in the formula.
    claim_count : int, optional
        For credibility calculation.
    complement : float
        Industry or company-wide indication used as complement.  Default 0.0.
    """

    def __init__(
        self,
        on_level_premium_by_year: pd.Series,
        ultimate_loss_by_year: pd.Series,
        lob: str = "",
        trend_factor: float = 1.0,
        premium_trend_factor: float = 1.0,
        variable_expense_ratio: float = 0.25,
        fixed_expense_ratio: float = 0.05,
        target_profit_margin: float = 0.05,
        claim_count: Optional[int] = None,
        complement: float = 0.0,
        full_cred_claims: int = 1082,
    ) -> None:
        self.lob = lob
        self.on_level_premium = on_level_premium_by_year
        self.ultimate_loss = ultimate_loss_by_year
        self.trend_factor = trend_factor
        self.premium_trend_factor = premium_trend_factor
        self.var_exp = variable_expense_ratio
        self.fixed_exp = fixed_expense_ratio
        self.profit = target_profit_margin
        self.claim_count = claim_count
        self.complement = complement
        self.full_cred_claims = full_cred_claims

    def compute(self) -> RateIndicationResult:
        """Run the full indication and return a RateIndicationResult."""
        years = sorted(
            set(self.on_level_premium.index) & set(self.ultimate_loss.index)
        )
        if not years:
            raise ValueError("No overlapping accident years between premium and ultimate loss data.")

        olp = self.on_level_premium.reindex(years).fillna(0)
        ult = self.ultimate_loss.reindex(years).fillna(0)

        # Trend losses to future period
        trended_ult = ult * self.trend_factor

        # Adjust premium for premium trend (if applicable)
        adjusted_olp = olp * self.premium_trend_factor

        # Projected loss ratio
        total_olp = float(adjusted_olp.sum())
        total_trended_ult = float(trended_ult.sum())

        if total_olp <= 0:
            raise ValueError("On-level earned premium is zero — cannot compute loss ratio.")

        proj_lr = total_trended_ult / total_olp

        # Permissible loss ratio
        permissible_lr = 1.0 - self.var_exp - self.fixed_exp - self.profit

        # Indicated change
        indicated_change = (proj_lr / permissible_lr) - 1.0

        # Credibility
        n = self.claim_count or int(total_olp / 1000)  # rough fallback
        cred = min(np.sqrt(n / self.full_cred_claims), 1.0)

        # Credibility-weighted indication
        cred_wtd = cred * indicated_change + (1.0 - cred) * self.complement

        # Year-level detail
        by_year = pd.DataFrame(
            {
                "on_level_premium": olp,
                "ultimate_loss": ult,
                "trended_ultimate_loss": trended_ult,
                "loss_ratio": ult / olp.replace(0, np.nan),
                "trended_loss_ratio": trended_ult / adjusted_olp.replace(0, np.nan),
            },
            index=years,
        )

        return RateIndicationResult(
            lob=self.lob,
            accident_years=years,
            on_level_premium=total_olp,
            trended_ultimate_loss=total_trended_ult,
            projected_loss_ratio=proj_lr,
            variable_expense_ratio=self.var_exp,
            fixed_expense_ratio=self.fixed_exp,
            target_profit_margin=self.profit,
            permissible_loss_ratio=permissible_lr,
            indicated_change=indicated_change,
            credibility=cred,
            complement_indication=self.complement,
            credibility_weighted_change=cred_wtd,
            by_year=by_year,
            trend_factor=self.trend_factor,
            development_factor=1.0,  # already baked in to ultimate_loss
        )

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: str,
        accident_years: Optional[List[int]] = None,
        **kwargs,
    ) -> "RateIndication":
        """
        Build a RateIndication from a session with loaded data.

        Requires session to have loaded:
            - policies (for on-level premium via OnLevelPremium)
            - valuations + claims (for reserve analysis → ultimates)
            - rate_changes (for on-level factors)
        """
        from auto_actuary.analytics.ratemaking.on_level import OnLevelPremium
        from auto_actuary.analytics.ratemaking.trend import TrendAnalysis

        cfg = session.config

        # On-level premium
        olp_table = None
        if "policies" in session.loader and "rate_changes" in session.loader:
            try:
                olp_calc = OnLevelPremium(
                    rate_changes=session.loader["rate_changes"],
                    policies=session.loader["policies"],
                    lob=lob,
                )
                olp_df = olp_calc.on_level_premium()
                olp_table = olp_df["on_level_earned_premium"]
            except Exception as exc:
                logger.warning("On-level premium failed: %s — using raw written premium", exc)

        if olp_table is None:
            # Fallback: sum written premium from policies
            pol = session.loader["policies"]
            pol_lob = pol[pol["line_of_business"] == lob].copy()
            pol_lob["accident_year"] = pol_lob["effective_date"].dt.year
            olp_table = pol_lob.groupby("accident_year")["written_premium"].sum()

        # Reserve analysis → ultimates
        reserve = session.reserve_analysis(lob=lob)
        selected = reserve.selected()
        ultimate_by_year = selected.ultimates

        # Filter to requested accident years
        if accident_years:
            olp_table = olp_table.reindex(accident_years)
            ultimate_by_year = ultimate_by_year.reindex(accident_years)

        # Expense / profit assumptions from config
        var_exp = kwargs.pop("variable_expense_ratio", cfg.assumption("ratemaking", "variable_expense_ratio", default=0.25))
        fixed_exp = kwargs.pop("fixed_expense_ratio", cfg.assumption("ratemaking", "fixed_expense_ratio", default=0.05))
        profit = kwargs.pop("target_profit_margin", cfg.assumption("ratemaking", "target_profit_margin", default=0.05))

        return cls(
            on_level_premium_by_year=olp_table,
            ultimate_loss_by_year=ultimate_by_year,
            lob=lob,
            variable_expense_ratio=float(var_exp),
            fixed_expense_ratio=float(fixed_exp),
            target_profit_margin=float(profit),
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"RateIndication(lob={self.lob!r})"
