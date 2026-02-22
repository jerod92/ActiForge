"""
auto_actuary.analytics.ratemaking.irpm
=======================================
Individual Risk Premium Modification (IRPM) usage efficiency analysis.

IRPM (also called "schedule rating" or "experience rating mod") allows
underwriters to deviate from the filed manual rate for individual risks based
on risk-specific characteristics.  Common forms include:

  - Schedule rating credits/debits (for commercial lines, WC, GL)
  - Experience rating modifications (e-mod, for WC)
  - Individual risk deviations (IRD) for large accounts

The actuarial question is: **how well is IRPM being used?**

A properly functioning IRPM system should:
  1. Predict risk — accounts with large credits should have lower-than-average
     loss ratios; accounts with large debits should have higher loss ratios.
  2. Not exhibit systematic bias — the overall distribution of modifications
     should be centred near zero, and the premium generated should be adequate.
  3. Distinguish between risks — a narrow modification distribution means the
     tool is not differentiating risks, defeating its purpose.

This module computes:
  - **Adequacy test**: is average modified premium adequate (loss ratio test)?
  - **Predictive lift**: does IRPM effectively rank order risk? (Gini / lift curve)
  - **Dispersion analysis**: distribution of modification factors across the book.
  - **Efficiency ratio**: Var(predicted LR) / Var(actual LR) — a perfect IRPM
    would explain all variation in actual loss ratios.
  - **Bias test**: are credits / debits systematically mis-calibrated?

Data requirements
-----------------
The ``policies`` table must contain:
  - ``irpm_factor``  (the modification applied, e.g. 0.85 = 15% credit)
                     OR ``schedule_credit_pct`` (as a signed %)
  - ``written_premium``
  - ``written_exposure``

Loss data is joined from ``claims`` + ``valuations`` as usual.

References
----------
- Mahler, H. (1998) "Introduction to Credibility"
- Gillam, W. & Snader, R. (1992) "Fundamentals of Individual Risk Rating"
- CAS Study Note: "Schedule Rating" (multiple authors)
- ISO ERM-14 Experience Rating Plan documentation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class IRPMAdequacyResult:
    """Adequacy test result."""
    overall_loss_ratio: float
    target_loss_ratio: float         # = permissible loss ratio
    adequacy_ratio: float            # actual LR / target LR  (1.0 = adequate)
    is_adequate: bool
    credits_loss_ratio: float        # LR for policies with credits (IRPM < 1)
    debits_loss_ratio: float         # LR for policies with debits (IRPM > 1)
    neutral_loss_ratio: float        # LR for policies at/near IRPM = 1.0

    @property
    def adequacy_pct(self) -> str:
        return f"{(self.adequacy_ratio - 1):+.1%}"

    def __repr__(self) -> str:
        status = "ADEQUATE" if self.is_adequate else "INADEQUATE"
        return (
            f"IRPMAdequacyResult({status}, "
            f"LR={self.overall_loss_ratio:.3f}, "
            f"target={self.target_loss_ratio:.3f})"
        )


@dataclass
class IRPMEfficiencyResult:
    """Efficiency metrics for the IRPM system."""
    gini_coefficient: float         # 0 = no lift, 1 = perfect lift
    efficiency_ratio: float         # Var(predicted)/Var(actual); 1 = perfect
    correlation_irpm_lr: float      # Pearson r between IRPM factor and actual LR
    modification_std: float         # Std dev of IRPM factors (dispersion)
    modification_iqr: float         # Interquartile range of IRPM factors
    pct_at_unity: float             # % of policies with IRPM = 1.0 (not modified)
    pct_credits: float              # % of policies with IRPM < 1.0 (credits)
    pct_debits: float               # % of policies with IRPM > 1.0 (debits)

    def __repr__(self) -> str:
        return (
            f"IRPMEfficiencyResult("
            f"gini={self.gini_coefficient:.3f}, "
            f"corr(IRPM,LR)={self.correlation_irpm_lr:.3f}, "
            f"spread_IQR={self.modification_iqr:.3f})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class IRPMAnalysis:
    """
    IRPM usage efficiency analysis.

    Parameters
    ----------
    policies : pd.DataFrame
        Must contain: policy_id | written_premium | written_exposure |
                      irpm_factor (or schedule_credit_pct).
        Optional: effective_date | territory | class_code | agent_id |
                  line_of_business
    claims : pd.DataFrame, optional
        Must contain: claim_id | policy_id
    valuations : pd.DataFrame, optional
        Must contain: claim_id | incurred_loss
    lob : str, optional
        Filter to a single LOB.
    target_loss_ratio : float
        Permissible loss ratio (= 1 - V - Q).  Used for adequacy test.
        Default 0.65 (roughly a 35% expense ratio).
    unity_tolerance : float
        IRPM factors within ±*unity_tolerance* of 1.0 are considered "at unity".
        Default 0.01 (i.e., within 1%).
    """

    def __init__(
        self,
        policies: pd.DataFrame,
        claims: Optional[pd.DataFrame] = None,
        valuations: Optional[pd.DataFrame] = None,
        lob: Optional[str] = None,
        target_loss_ratio: float = 0.65,
        unity_tolerance: float = 0.01,
    ) -> None:
        self.lob = lob
        self.target_loss_ratio = target_loss_ratio
        self.unity_tolerance = unity_tolerance

        pol = policies.copy()
        if lob and "line_of_business" in pol.columns:
            pol = pol[pol["line_of_business"] == lob]

        # Normalise IRPM column name
        if "irpm_factor" not in pol.columns:
            if "schedule_credit_pct" in pol.columns:
                # Convert signed credit % to multiplicative factor
                # e.g. +15% credit → 0.85,  -10% debit → 1.10
                pol["irpm_factor"] = 1.0 - pol["schedule_credit_pct"] / 100.0
                logger.info("IRPMAnalysis: derived irpm_factor from schedule_credit_pct")
            elif "schedule_mod" in pol.columns:
                pol["irpm_factor"] = pol["schedule_mod"]
            else:
                logger.warning(
                    "IRPMAnalysis: no IRPM factor column found "
                    "(expected 'irpm_factor', 'schedule_credit_pct', or 'schedule_mod'). "
                    "Analysis will be limited."
                )
                pol["irpm_factor"] = np.nan

        self._policies = pol
        self._claims = claims.copy() if claims is not None else pd.DataFrame()
        self._vals = valuations.copy() if valuations is not None else pd.DataFrame()

        # Build policy-level loss table
        self._policy_data: pd.DataFrame = self._build_policy_loss_table()

    def _build_policy_loss_table(self) -> pd.DataFrame:
        """Join losses to policies to build a policy-level flat table."""
        pol = self._policies.copy()

        if not self._claims.empty and not self._vals.empty:
            loss_cols = [c for c in ["claim_id", "incurred_loss", "paid_loss"]
                         if c in self._vals.columns]
            if "valuation_date" in self._vals.columns:
                latest = (
                    self._vals.sort_values("valuation_date")
                    .groupby("claim_id").last().reset_index()[loss_cols]
                )
            else:
                latest = self._vals[loss_cols].copy()

            if "policy_id" in self._claims.columns:
                claim_loss = self._claims[["claim_id", "policy_id"]].merge(latest, on="claim_id", how="left")
                claim_loss["incurred_loss"] = claim_loss.get("incurred_loss", pd.Series(0)).fillna(0)
                claim_loss["claim_count"] = 1
                pol_loss = claim_loss.groupby("policy_id").agg(
                    incurred_loss=("incurred_loss", "sum"),
                    claim_count=("claim_count", "sum"),
                ).reset_index()
                pol = pol.merge(pol_loss, on="policy_id", how="left")
                pol["incurred_loss"] = pol["incurred_loss"].fillna(0)
                pol["claim_count"] = pol["claim_count"].fillna(0)
            else:
                pol["incurred_loss"] = 0.0
                pol["claim_count"] = 0
        else:
            pol["incurred_loss"] = 0.0
            pol["claim_count"] = 0

        # Policy-level loss ratio
        prem = pol.get("written_premium", pd.Series(np.nan)).replace(0, np.nan)
        pol["policy_loss_ratio"] = pol["incurred_loss"] / prem

        # Manual premium (before IRPM) = written_premium / irpm_factor
        irpm = pol["irpm_factor"].replace(0, np.nan)
        pol["manual_premium"] = pol["written_premium"] / irpm

        # Pure premium (loss per exposure unit)
        exp = pol.get("written_exposure", pd.Series(np.nan)).replace(0, np.nan)
        pol["pure_premium"] = pol["incurred_loss"] / exp

        return pol

    # ------------------------------------------------------------------
    # Distribution analysis
    # ------------------------------------------------------------------

    def modification_distribution(
        self,
        bins: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Frequency distribution of IRPM modification factors.

        Parameters
        ----------
        bins : list of float, optional
            Bin edges.  Default: [0.5, 0.7, 0.8, 0.85, 0.90, 0.95, 1.0, 1.05,
                                   1.10, 1.15, 1.20, 1.30, 1.50]

        Returns
        -------
        pd.DataFrame
            Columns: bin_label | policy_count | written_premium |
                     pct_of_policies | pct_of_premium |
                     avg_loss_ratio (if loss data available)
        """
        bins = bins or [0.50, 0.70, 0.80, 0.85, 0.90, 0.95,
                        1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50, 2.00]
        df = self._policy_data.copy()
        df = df[df["irpm_factor"].notna()]

        labels = [f"{bins[i]:.2f}–{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
        df["irpm_bin"] = pd.cut(df["irpm_factor"], bins=bins, labels=labels, right=False)

        agg = df.groupby("irpm_bin", observed=True).agg(
            policy_count=("policy_id", "count"),
            written_premium=("written_premium", "sum") if "written_premium" in df.columns else ("policy_id", "count"),
            avg_irpm_factor=("irpm_factor", "mean"),
        )

        total_pol = agg["policy_count"].sum()
        total_prem = agg["written_premium"].sum()
        agg["pct_of_policies"] = agg["policy_count"] / total_pol if total_pol > 0 else np.nan
        agg["pct_of_premium"] = agg["written_premium"] / total_prem if total_prem > 0 else np.nan

        if "policy_loss_ratio" in df.columns:
            lr = df.groupby("irpm_bin", observed=True).apply(
                lambda g: g["incurred_loss"].sum() / g["written_premium"].replace(0, np.nan).sum()
                if "written_premium" in g.columns else np.nan
            )
            agg["avg_loss_ratio"] = lr

        return agg

    # ------------------------------------------------------------------
    # Adequacy test
    # ------------------------------------------------------------------

    def adequacy_test(self) -> IRPMAdequacyResult:
        """
        Test whether IRPM-modified premiums are adequate for different buckets.

        The adequacy test answers: *given the credits/debits applied, does
        premium still cover losses at the target loss ratio?*

        Returns
        -------
        IRPMAdequacyResult
        """
        df = self._policy_data.copy()

        def _lr(subset: pd.DataFrame) -> float:
            prem = float(subset["written_premium"].sum()) if "written_premium" in subset.columns else 0.0
            loss = float(subset["incurred_loss"].sum())
            return loss / prem if prem > 0 else np.nan

        overall_lr = _lr(df)

        tol = self.unity_tolerance
        credits_mask = df["irpm_factor"] < (1.0 - tol)
        debits_mask = df["irpm_factor"] > (1.0 + tol)
        neutral_mask = ~credits_mask & ~debits_mask

        return IRPMAdequacyResult(
            overall_loss_ratio=overall_lr,
            target_loss_ratio=self.target_loss_ratio,
            adequacy_ratio=overall_lr / self.target_loss_ratio if self.target_loss_ratio else np.nan,
            is_adequate=overall_lr <= self.target_loss_ratio,
            credits_loss_ratio=_lr(df[credits_mask]),
            debits_loss_ratio=_lr(df[debits_mask]),
            neutral_loss_ratio=_lr(df[neutral_mask]),
        )

    # ------------------------------------------------------------------
    # Efficiency / predictive lift
    # ------------------------------------------------------------------

    def efficiency_test(self) -> IRPMEfficiencyResult:
        """
        Compute IRPM efficiency metrics (Gini, correlation, dispersion).

        Returns
        -------
        IRPMEfficiencyResult
        """
        df = self._policy_data.dropna(subset=["irpm_factor"]).copy()

        # Dispersion metrics
        irpm_vals = df["irpm_factor"]
        mod_std = float(irpm_vals.std())
        mod_iqr = float(irpm_vals.quantile(0.75) - irpm_vals.quantile(0.25))

        tol = self.unity_tolerance
        pct_at_unity = float((irpm_vals.between(1.0 - tol, 1.0 + tol)).mean())
        pct_credits = float((irpm_vals < 1.0 - tol).mean())
        pct_debits = float((irpm_vals > 1.0 + tol).mean())

        # Correlation: IRPM factor vs. actual loss ratio
        # Negative correlation is expected and desirable (high credits → low LR)
        lr_col = "policy_loss_ratio"
        valid = df[[lr_col, "irpm_factor"]].dropna()
        if len(valid) >= 5:
            corr, _ = scipy_stats.pearsonr(valid["irpm_factor"], valid[lr_col])
        else:
            corr = np.nan

        # Gini coefficient (lift curve area)
        # Rank policies by IRPM factor ascending (best predicted → worst)
        # Compare cumulative actual loss vs. random baseline
        gini = self._compute_gini(df)

        # Efficiency ratio: explained variance (R² of IRPM predicting actual LR)
        eff_ratio = float(corr ** 2) if not np.isnan(corr) else np.nan

        return IRPMEfficiencyResult(
            gini_coefficient=gini,
            efficiency_ratio=eff_ratio,
            correlation_irpm_lr=float(corr) if not np.isnan(corr) else np.nan,
            modification_std=mod_std,
            modification_iqr=mod_iqr,
            pct_at_unity=pct_at_unity,
            pct_credits=pct_credits,
            pct_debits=pct_debits,
        )

    @staticmethod
    def _compute_gini(df: pd.DataFrame) -> float:
        """Compute Gini coefficient of lift: IRPM factor vs. actual loss."""
        required = ["irpm_factor", "incurred_loss", "written_premium"]
        if any(c not in df.columns for c in required):
            return np.nan
        sub = df[required].dropna()
        if sub.empty or sub["written_premium"].sum() == 0:
            return np.nan

        # Sort by predicted risk (IRPM factor ascending = best → worst predicted)
        sub = sub.sort_values("irpm_factor")
        cum_prem = sub["written_premium"].cumsum() / sub["written_premium"].sum()
        cum_loss = sub["incurred_loss"].cumsum() / sub["incurred_loss"].sum()

        # Gini = 2 × (area under Lorenz curve - 0.5)
        # Use trapezoidal integration
        area = float(np.trapezoid(cum_loss.values, cum_prem.values))
        return float(2 * area - 1)

    # ------------------------------------------------------------------
    # Bias test
    # ------------------------------------------------------------------

    def bias_test(
        self,
        n_quantiles: int = 10,
    ) -> pd.DataFrame:
        """
        Test for systematic bias in the IRPM system.

        Splits policies into *n_quantiles* buckets by IRPM factor and
        computes, for each bucket:
          - Predicted loss ratio (= manual_premium × target_LR × irpm_factor / written_premium)
          - Actual loss ratio
          - Bias = (actual - predicted) / predicted

        A well-calibrated IRPM should show bias ≈ 0 across all buckets.

        Parameters
        ----------
        n_quantiles : int
            Number of equal-frequency buckets (default 10 = deciles).

        Returns
        -------
        pd.DataFrame
            Rows = IRPM quantile bucket.
            Columns: irpm_factor_mean | irpm_factor_range |
                     predicted_loss_ratio | actual_loss_ratio | bias
        """
        df = self._policy_data.dropna(subset=["irpm_factor"]).copy()
        if df.empty:
            return pd.DataFrame()

        df["irpm_quantile"] = pd.qcut(df["irpm_factor"], q=n_quantiles, duplicates="drop")

        def _summarise(g: pd.DataFrame) -> pd.Series:
            prem = g["written_premium"].sum()
            loss = g["incurred_loss"].sum()
            irpm_mean = g["irpm_factor"].mean()
            # Predicted LR = IRPM factor × target_LR (assuming manual rate is adequate)
            pred_lr = irpm_mean * self.target_loss_ratio
            actual_lr = loss / prem if prem > 0 else np.nan
            bias = (actual_lr - pred_lr) / pred_lr if pred_lr and pred_lr != 0 else np.nan
            irpm_min = g["irpm_factor"].min()
            irpm_max = g["irpm_factor"].max()
            return pd.Series({
                "irpm_factor_mean": round(irpm_mean, 4),
                "irpm_factor_range": f"{irpm_min:.3f}–{irpm_max:.3f}",
                "n_policies": int(len(g)),
                "written_premium": round(prem, 0),
                "predicted_loss_ratio": round(pred_lr, 4) if not np.isnan(pred_lr) else np.nan,
                "actual_loss_ratio": round(actual_lr, 4) if not np.isnan(actual_lr) else np.nan,
                "bias": round(bias, 4) if not np.isnan(bias) else np.nan,
            })

        return df.groupby("irpm_quantile", observed=True).apply(_summarise)

    # ------------------------------------------------------------------
    # Segment breakdown
    # ------------------------------------------------------------------

    def by_segment(
        self,
        dimension: str = "territory",
    ) -> pd.DataFrame:
        """
        IRPM usage statistics broken down by a portfolio dimension.

        Returns
        -------
        pd.DataFrame
            Columns: avg_irpm_factor | pct_credits | pct_debits |
                     actual_loss_ratio | target_loss_ratio | bias
        """
        df = self._policy_data.copy()
        if dimension not in df.columns:
            raise ValueError(f"Dimension '{dimension}' not found in data.")

        def _seg_stats(g: pd.DataFrame) -> pd.Series:
            irpm = g["irpm_factor"].dropna()
            prem = g["written_premium"].sum() if "written_premium" in g.columns else 0
            loss = g["incurred_loss"].sum()
            tol = self.unity_tolerance
            return pd.Series({
                "policy_count": int(len(g)),
                "avg_irpm_factor": float(irpm.mean()) if not irpm.empty else np.nan,
                "pct_credits": float((irpm < 1.0 - tol).mean()) if not irpm.empty else np.nan,
                "pct_debits": float((irpm > 1.0 + tol).mean()) if not irpm.empty else np.nan,
                "irpm_std": float(irpm.std()) if not irpm.empty else np.nan,
                "written_premium": float(prem),
                "actual_loss_ratio": float(loss / prem) if prem > 0 else np.nan,
            })

        result = df.groupby(dimension).apply(_seg_stats)
        result["target_loss_ratio"] = self.target_loss_ratio
        result["bias"] = (result["actual_loss_ratio"] - self.target_loss_ratio) / self.target_loss_ratio
        return result.sort_values("policy_count", ascending=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        """Scalar summary of IRPM usage."""
        df = self._policy_data.copy()
        irpm = df["irpm_factor"].dropna()
        adeq = self.adequacy_test()
        eff = self.efficiency_test()

        return {
            "n_policies": int(len(df)),
            "n_policies_with_irpm": int(irpm.notna().sum()),
            "avg_irpm_factor": float(irpm.mean()) if not irpm.empty else np.nan,
            "irpm_factor_std": float(irpm.std()) if not irpm.empty else np.nan,
            "pct_credits": float(eff.pct_credits),
            "pct_debits": float(eff.pct_debits),
            "pct_at_unity": float(eff.pct_at_unity),
            "overall_loss_ratio": float(adeq.overall_loss_ratio),
            "is_adequate": bool(adeq.is_adequate),
            "gini_coefficient": float(eff.gini_coefficient) if not np.isnan(eff.gini_coefficient) else None,
            "correlation_irpm_lr": float(eff.correlation_irpm_lr) if not np.isnan(eff.correlation_irpm_lr) else None,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"IRPMAnalysis(lob={self.lob!r}, "
            f"n_policies={s['n_policies']}, "
            f"avg_irpm={s.get('avg_irpm_factor', 0):.3f}, "
            f"adequate={s.get('is_adequate', False)})"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: Optional[str] = None,
        target_loss_ratio: float = 0.65,
        **kwargs,
    ) -> "IRPMAnalysis":
        """
        Build IRPMAnalysis from a loaded ActuarySession.

        Requires 'policies' table to contain irpm_factor or schedule_credit_pct.
        """
        policies = session.loader["policies"] if "policies" in session.loader.loaded_tables else pd.DataFrame()
        claims = session.loader["claims"] if "claims" in session.loader.loaded_tables else None
        vals = session.loader["valuations"] if "valuations" in session.loader.loaded_tables else None
        return cls(
            policies=policies,
            claims=claims,
            valuations=vals,
            lob=lob,
            target_loss_ratio=target_loss_ratio,
            **kwargs,
        )
