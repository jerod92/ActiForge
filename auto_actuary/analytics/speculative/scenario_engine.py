"""
auto_actuary.analytics.speculative.scenario_engine
====================================================
Speculative business scenario engine for executive decision support.

"What happens if we raise rates 10% in the Southeast?"
"What if frequency keeps climbing at 5% per year instead of 3%?"
"How does entering the Texas commercial auto market change our combined ratio?"
"What's the P&L impact of a 20% increase in CAT activity?"

This module answers those questions by perturbing a baseline portfolio state
and measuring the downstream KPI effects: written premium, loss ratio,
combined ratio, IBNR, and net income.

Architecture
------------
Each ScenarioParams dataclass describes *one* business decision or assumption
change.  ScenarioEngine holds the baseline data + fitted GLM, then applies
the scenario transforms and returns a ScenarioResult.

Scenarios are composable — you can stack rate action + mix shift + expense
initiative to model a multi-pronged strategic response.

Scenario types
--------------
1. rate_action       — price change in targeted segments (with elasticity)
2. frequency_trend   — assume a different annual frequency trend going forward
3. severity_shock    — one-time or sustained severity multiplier
4. mix_shift         — reweight portfolio across territories / classes
5. exit_segment      — remove a territory, class, or LOB from the portfolio
6. enter_market      — add a new segment with estimated P&L parameters
7. cat_environment   — change CAT loss load (climate, concentration)
8. expense_initiative— operating expense ratio improvement

Uncertainty
-----------
For any scenario, confidence intervals (90% by default) are produced by
propagating bootstrap uncertainty from the GLM through the scenario transforms.

References
----------
- Werner & Modlin (2016) "Basic Ratemaking" Chapter 9 (risk classification)
- Casualty Actuarial Society (2014) "Ratemaking" study note
- Own-price elasticity of insurance: Elasticity ≈ −0.3 to −0.7 (Harrington 1988)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from auto_actuary.analytics.speculative.glm_models import CompoundGLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario parameter containers
# ---------------------------------------------------------------------------

@dataclass
class RateActionParams:
    """
    Targeted rate change in specific segments.

    Parameters
    ----------
    rate_changes : dict
        Maps (territory | class_code | lob_code | "all") -> pct_change.
        e.g., {"SOUTHEAST": 0.10, "NORTHEAST": 0.05} means +10% SE, +5% NE.
    price_elasticity : float
        Own-price elasticity of demand (negative, typically -0.3 to -0.5).
        A 10% rate increase with elasticity -0.3 → -3% volume loss.
    segment_col : str
        Column used to match rate_changes keys (default "territory").
    """
    rate_changes: Dict[str, float]
    price_elasticity: float = -0.35
    segment_col: str = "territory"


@dataclass
class FrequencyTrendParams:
    """
    Override the assumed annual frequency trend.

    Parameters
    ----------
    annual_trend : float
        New assumed annual frequency trend (e.g., 0.05 = +5%/yr).
        Replaces the historically-fitted trend for projection.
    horizon_years : int
        Number of years to project forward.
    base_year : int, optional
        Year from which trend is applied.  Defaults to the max year in data.
    """
    annual_trend: float          # e.g., 0.05 for +5%/yr
    horizon_years: int = 3
    base_year: Optional[int] = None


@dataclass
class SeverityShockParams:
    """
    One-time or sustained severity multiplier.

    Parameters
    ----------
    severity_multiplier : float
        Multiplicative shock.  1.15 = +15% severity.
    sustained : bool
        If True, the multiplier compounds annually over horizon_years.
        If False, it's a one-time shock to the current year.
    horizon_years : int
        Projection horizon (only used when sustained=True).
    """
    severity_multiplier: float
    sustained: bool = False
    horizon_years: int = 3


@dataclass
class MixShiftParams:
    """
    Change in portfolio volume distribution across segments.

    Parameters
    ----------
    volume_changes : dict
        Maps segment label -> pct change in written exposure.
        e.g., {"FLORIDA": -0.20, "OHIO": 0.10} = reduce FL by 20%, grow OH 10%.
    segment_col : str
        Column name to match segment labels against.
    """
    volume_changes: Dict[str, float]
    segment_col: str = "territory"


@dataclass
class ExitSegmentParams:
    """
    Remove a territory, class code, or LOB from the portfolio.

    Parameters
    ----------
    filters : dict
        Column -> value pairs to identify the segment to exit.
        e.g., {"territory": "FLORIDA"} exits Florida.
        e.g., {"lob_code": "WC"} exits Workers Comp.
    """
    filters: Dict[str, str]


@dataclass
class EnterMarketParams:
    """
    Enter a new market segment not currently in the portfolio.

    Parameters
    ----------
    est_annual_premium : float
        Estimated first-year written premium for the new segment.
    est_loss_ratio : float
        Expected loss ratio for the new segment (actuarial estimate).
    est_expense_ratio : float
        Expected expense ratio (may differ from existing portfolio).
    est_frequency : float, optional
        Expected frequency (claims per unit) if GLM projection is desired.
    est_severity : float, optional
        Expected average severity.
    ramp_up_years : int
        Years to reach full premium volume (linear ramp).
    """
    est_annual_premium: float
    est_loss_ratio: float
    est_expense_ratio: float
    est_frequency: Optional[float] = None
    est_severity: Optional[float] = None
    ramp_up_years: int = 3


@dataclass
class CatEnvironmentParams:
    """
    Change the expected CAT loss environment.

    Parameters
    ----------
    cat_multiplier : float
        Multiplicative change to historical CAT losses.
        1.20 = +20% CAT load (climate risk increase).
        0.80 = -20% (concentration reduction after reinsurance).
    noncat_unchanged : bool
        If True, only CAT losses are affected.  If False, total loss trend shifts.
    """
    cat_multiplier: float
    noncat_unchanged: bool = True


@dataclass
class ExpenseInitiativeParams:
    """
    Operational expense reduction initiative.

    Parameters
    ----------
    delta_expense_ratio : float
        Absolute change in expense ratio (negative = improvement).
        -0.02 = two-point expense ratio improvement.
    timeline_years : int
        Years over which the savings are phased in.
    """
    delta_expense_ratio: float    # e.g., -0.02 = 2pt improvement
    timeline_years: int = 2


@dataclass
class ScenarioParams:
    """
    Container for a complete speculative scenario.

    A scenario can include any combination of the component params above.
    Leave unused components as None to apply only the specified changes.
    """
    name: str
    description: str = ""
    rate_action: Optional[RateActionParams] = None
    frequency_trend: Optional[FrequencyTrendParams] = None
    severity_shock: Optional[SeverityShockParams] = None
    mix_shift: Optional[MixShiftParams] = None
    exit_segment: Optional[ExitSegmentParams] = None
    enter_market: Optional[EnterMarketParams] = None
    cat_environment: Optional[CatEnvironmentParams] = None
    expense_initiative: Optional[ExpenseInitiativeParams] = None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """
    Result of running a scenario through the engine.

    Attributes
    ----------
    name : str
        Scenario name.
    base_kpis : dict
        Baseline KPIs before any scenario changes.
    scenario_kpis : dict
        KPIs under the scenario assumptions.
    deltas : dict
        Absolute change: scenario_kpis - base_kpis.
    delta_pcts : dict
        Percentage change: (scenario - base) / |base|.
    segment_breakdown : DataFrame
        Segment-level KPI comparison (base vs. scenario).
    ci_90 : dict
        90% confidence intervals on key scenario KPIs.
        Keys match scenario_kpis; values are (lower, upper) tuples.
    notes : list of str
        Human-readable notes about the scenario assumptions.
    """
    name: str
    base_kpis: Dict[str, float]
    scenario_kpis: Dict[str, float]
    deltas: Dict[str, float]
    delta_pcts: Dict[str, float]
    segment_breakdown: pd.DataFrame
    ci_90: Dict[str, Tuple[float, float]]
    notes: List[str] = field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        """
        Return a tidy comparison table suitable for display.

        Returns
        -------
        DataFrame with index=KPI, columns=[Base, Scenario, Change, Change%]
        """
        kpis = sorted(set(list(self.base_kpis.keys()) + list(self.scenario_kpis.keys())))
        rows = []
        for kpi in kpis:
            base = self.base_kpis.get(kpi, np.nan)
            scen = self.scenario_kpis.get(kpi, np.nan)
            delta = self.deltas.get(kpi, np.nan)
            delta_pct = self.delta_pcts.get(kpi, np.nan)
            ci = self.ci_90.get(kpi, (np.nan, np.nan))
            rows.append({
                "KPI": kpi,
                "Base": base,
                "Scenario": scen,
                "Change": delta,
                "Change%": delta_pct,
                "CI_Low": ci[0],
                "CI_High": ci[1],
            })
        return pd.DataFrame(rows).set_index("KPI")

    def __repr__(self) -> str:
        lr_base = self.base_kpis.get("loss_ratio", np.nan)
        lr_scen = self.scenario_kpis.get("loss_ratio", np.nan)
        return (
            f"ScenarioResult(name={self.name!r}, "
            f"LR: {lr_base:.1%} → {lr_scen:.1%})"
        )


# ---------------------------------------------------------------------------
# Scenario Engine
# ---------------------------------------------------------------------------

class ScenarioEngine:
    """
    Speculative business scenario engine.

    Applies scenario parameter changes to a baseline portfolio and computes
    the resulting KPI impact.  Designed for executive "what-if" analysis.

    Parameters
    ----------
    segment_df : DataFrame
        Aggregated portfolio data at (year × territory × class) grain.
        Required columns: earned_exposure, claim_count, incurred_loss,
                          written_premium (if loss ratio desired)
        Optional: is_catastrophe, expense_ratio, territory, class_code, lob_code
    glm : CompoundGLM, optional
        Pre-fitted compound GLM.  If None, the engine uses aggregate statistics
        rather than GLM predictions for scenario impacts.
    expense_ratio : float
        Portfolio-level expense ratio (ULAE + ALAE + other expenses / premium).
        Used for combined ratio calculations.  Default 0.30.
    current_year : int, optional
        Reference year for trend projections.  Defaults to max accident_year.
    """

    def __init__(
        self,
        segment_df: pd.DataFrame,
        glm: Optional[CompoundGLM] = None,
        expense_ratio: float = 0.30,
        current_year: Optional[int] = None,
    ) -> None:
        self._df = segment_df.copy()
        self.glm = glm
        self.expense_ratio = expense_ratio

        if current_year is None:
            yr_col = "accident_year" if "accident_year" in self._df.columns else None
            self.current_year = int(self._df[yr_col].max()) if yr_col else 2024
        else:
            self.current_year = current_year

        self._base_kpis = self._compute_kpis(self._df)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        params: ScenarioParams,
        n_boot: int = 0,
        ci: float = 0.90,
        random_state: int = 42,
    ) -> ScenarioResult:
        """
        Run a speculative scenario and return the KPI impact.

        Parameters
        ----------
        params : ScenarioParams
            Scenario definition (which levers to pull).
        n_boot : int
            Bootstrap iterations for confidence intervals.  0 = skip CI.
        ci : float
            Confidence level for intervals.  Default 0.90.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        ScenarioResult
        """
        df_scen = self._df.copy()
        notes = []

        # Apply scenario transforms in logical order
        if params.exit_segment is not None:
            df_scen, n_exit = self._apply_exit_segment(df_scen, params.exit_segment)
            notes.append(f"Exited segment: {params.exit_segment.filters} ({n_exit:,} rows removed)")

        if params.mix_shift is not None:
            df_scen, mix_note = self._apply_mix_shift(df_scen, params.mix_shift)
            notes.append(mix_note)

        if params.rate_action is not None:
            df_scen, rate_note = self._apply_rate_action(df_scen, params.rate_action)
            notes.append(rate_note)

        if params.frequency_trend is not None:
            df_scen, freq_note = self._apply_frequency_trend(df_scen, params.frequency_trend)
            notes.append(freq_note)

        if params.severity_shock is not None:
            df_scen, sev_note = self._apply_severity_shock(df_scen, params.severity_shock)
            notes.append(sev_note)

        if params.cat_environment is not None:
            df_scen, cat_note = self._apply_cat_environment(df_scen, params.cat_environment)
            notes.append(cat_note)

        if params.expense_initiative is not None:
            df_scen, exp_note = self._apply_expense_initiative(df_scen, params.expense_initiative)
            notes.append(exp_note)

        scen_kpis = self._compute_kpis(df_scen)

        # Add new market entry on top of computed KPIs
        if params.enter_market is not None:
            scen_kpis, em_note = self._apply_enter_market(scen_kpis, params.enter_market)
            notes.append(em_note)

        deltas = {k: scen_kpis.get(k, 0) - self._base_kpis.get(k, 0) for k in scen_kpis}
        delta_pcts = {}
        for k in deltas:
            base_val = self._base_kpis.get(k, 0)
            delta_pcts[k] = deltas[k] / abs(base_val) if abs(base_val) > 1e-10 else 0.0

        # Segment breakdown
        seg_breakdown = self._build_segment_breakdown(df_scen)

        # Bootstrap CI (parametric approximation if n_boot=0)
        ci_90 = self._compute_ci(scen_kpis, n_boot, ci, random_state)

        return ScenarioResult(
            name=params.name,
            base_kpis=self._base_kpis,
            scenario_kpis=scen_kpis,
            deltas=deltas,
            delta_pcts=delta_pcts,
            segment_breakdown=seg_breakdown,
            ci_90=ci_90,
            notes=notes,
        )

    def compare_scenarios(
        self,
        scenarios: List[ScenarioParams],
        n_boot: int = 0,
        ci: float = 0.90,
    ) -> pd.DataFrame:
        """
        Run multiple scenarios and return a side-by-side comparison table.

        Rows = KPIs, Columns = Base + each scenario name.

        Parameters
        ----------
        scenarios : list of ScenarioParams
        n_boot : int
            Bootstrap iterations per scenario.
        ci : float
            Confidence level for intervals.

        Returns
        -------
        DataFrame with KPIs as index and scenario columns.
        """
        results = {}
        for sp in scenarios:
            logger.info("Running scenario: %s", sp.name)
            res = self.run_scenario(sp, n_boot=n_boot, ci=ci)
            results[sp.name] = res

        # Build comparison table
        all_kpis = sorted(
            set(k for r in results.values() for k in r.scenario_kpis.keys())
        )

        rows = []
        for kpi in all_kpis:
            row = {"KPI": kpi, "Base": self._base_kpis.get(kpi, np.nan)}
            for name, res in results.items():
                row[name] = res.scenario_kpis.get(kpi, np.nan)
                row[f"{name}_Δ%"] = res.delta_pcts.get(kpi, np.nan)
            rows.append(row)

        return pd.DataFrame(rows).set_index("KPI")

    def stress_test(
        self,
        freq_shocks: Optional[np.ndarray] = None,
        sev_shocks: Optional[np.ndarray] = None,
        n_simulations: int = 500,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Monte Carlo stress test over a range of frequency and severity shocks.

        Samples frequency shock and severity shock from provided distributions
        (or uniform range if arrays given), computes the resulting loss ratio
        and combined ratio for each draw.

        Parameters
        ----------
        freq_shocks : ndarray, optional
            Array of frequency multipliers to sample from.
            Default: uniform [0.85, 1.25].
        sev_shocks : ndarray, optional
            Array of severity multipliers to sample from.
            Default: uniform [0.85, 1.30].
        n_simulations : int
            Number of Monte Carlo draws.
        random_state : int
            Random seed.

        Returns
        -------
        DataFrame with columns: freq_shock | sev_shock | loss_ratio |
            combined_ratio | pure_premium | percentile
        Sorted by loss_ratio, with percentile column showing the empirical CDF.
        """
        rng = np.random.default_rng(random_state)

        if freq_shocks is None:
            freq_shocks = np.linspace(0.80, 1.30, 50)
        if sev_shocks is None:
            sev_shocks = np.linspace(0.80, 1.35, 50)

        base_losses = self._base_kpis.get("total_losses", 1.0)
        base_premium = self._base_kpis.get("total_premium", 1.0)
        base_freq = self._base_kpis.get("overall_frequency", 0.08)
        base_sev = self._base_kpis.get("overall_severity", 5000.0)
        base_exp = self._base_kpis.get("total_exposure", 1.0)

        records = []
        for _ in range(n_simulations):
            fs = float(rng.choice(freq_shocks))
            ss = float(rng.choice(sev_shocks))

            new_freq = base_freq * fs
            new_sev = base_sev * ss
            new_losses = new_freq * new_sev * base_exp

            lr = new_losses / max(base_premium, 1e-6)
            cr = lr + self.expense_ratio

            records.append({
                "freq_shock": fs,
                "sev_shock": ss,
                "loss_ratio": lr,
                "combined_ratio": cr,
                "pure_premium": new_freq * new_sev,
            })

        df = pd.DataFrame(records).sort_values("loss_ratio").reset_index(drop=True)
        df["percentile"] = (df.index + 1) / len(df)
        return df

    @property
    def base_kpis(self) -> Dict[str, float]:
        return self._base_kpis.copy()

    # ------------------------------------------------------------------
    # Scenario transforms (internal)
    # ------------------------------------------------------------------

    def _apply_rate_action(
        self, df: pd.DataFrame, params: RateActionParams
    ) -> Tuple[pd.DataFrame, str]:
        """
        Apply rate changes to written_premium with elasticity volume effect.

        Formula:
            new_premium_per_unit = old_premium_per_unit * (1 + rate_change)
            volume_factor = 1 + elasticity * rate_change
            new_exposure = old_exposure * volume_factor
            new_written_premium = new_premium_per_unit * new_exposure

        For the loss side: fewer policies → fewer losses proportional to exposure change.
        """
        df = df.copy()
        col = params.segment_col
        if col not in df.columns and "all" not in params.rate_changes:
            logger.warning("rate_action: segment column '%s' not in data", col)
            return df, "rate_action skipped (column not found)"

        applied = []
        for segment, rate_chg in params.rate_changes.items():
            if segment == "all":
                mask = pd.Series(True, index=df.index)
            elif col in df.columns:
                mask = df[col].astype(str) == str(segment)
            else:
                continue

            vol_factor = 1.0 + params.price_elasticity * rate_chg

            if "written_premium" in df.columns:
                df.loc[mask, "written_premium"] = (
                    df.loc[mask, "written_premium"] * (1 + rate_chg) * vol_factor
                )
            if "earned_exposure" in df.columns:
                df.loc[mask, "earned_exposure"] = df.loc[mask, "earned_exposure"] * vol_factor
            if "incurred_loss" in df.columns:
                df.loc[mask, "incurred_loss"] = df.loc[mask, "incurred_loss"] * vol_factor
            if "claim_count" in df.columns:
                df.loc[mask, "claim_count"] = df.loc[mask, "claim_count"] * vol_factor

            applied.append(f"{segment}: {rate_chg:+.1%} rate, {vol_factor - 1:+.1%} volume")

        note = f"Rate action (elasticity={params.price_elasticity}): " + "; ".join(applied)
        return df, note

    def _apply_frequency_trend(
        self, df: pd.DataFrame, params: FrequencyTrendParams
    ) -> Tuple[pd.DataFrame, str]:
        """
        Project claims forward using an overridden annual frequency trend.

        Multiplies incurred_loss and claim_count by the trend factor for
        each row based on (base_year + horizon) - accident_year.
        """
        df = df.copy()
        base_yr = params.base_year or self.current_year
        project_yr = base_yr + params.horizon_years
        annual_trend = 1.0 + params.annual_trend

        if "accident_year" in df.columns:
            years_forward = (project_yr - df["accident_year"].clip(upper=base_yr))
        else:
            years_forward = pd.Series(float(params.horizon_years), index=df.index)

        trend_factor = annual_trend ** years_forward.clip(lower=0)

        if "incurred_loss" in df.columns:
            df["incurred_loss"] = df["incurred_loss"] * trend_factor
        if "claim_count" in df.columns:
            df["claim_count"] = df["claim_count"] * trend_factor

        note = (
            f"Frequency trend override: {params.annual_trend:+.2%}/yr × {params.horizon_years}yr "
            f"(compound factor {annual_trend**params.horizon_years:.4f})"
        )
        return df, note

    def _apply_severity_shock(
        self, df: pd.DataFrame, params: SeverityShockParams
    ) -> Tuple[pd.DataFrame, str]:
        """
        Apply a severity multiplier to all (or sustained) losses.

        For sustained shocks, losses for each accident year are multiplied
        by multiplier^(horizon_years) to simulate compounding inflation.
        For one-time shocks, all rows get the flat multiplier.
        """
        df = df.copy()
        if params.sustained and "accident_year" in df.columns:
            base_yr = self.current_year
            years_forward = (base_yr + params.horizon_years - df["accident_year"]).clip(lower=0, upper=params.horizon_years)
            factor = params.severity_multiplier ** years_forward
        else:
            factor = params.severity_multiplier

        if "incurred_loss" in df.columns:
            df["incurred_loss"] = df["incurred_loss"] * factor

        mode = "sustained" if params.sustained else "one-time"
        note = f"Severity shock ({mode}): {params.severity_multiplier:.3f}× multiplier"
        return df, note

    def _apply_mix_shift(
        self, df: pd.DataFrame, params: MixShiftParams
    ) -> Tuple[pd.DataFrame, str]:
        """
        Reweight portfolio volume by segment.

        Scales earned_exposure, incurred_loss, claim_count, and written_premium
        by the provided volume change factors for each segment.
        """
        df = df.copy()
        col = params.segment_col
        if col not in df.columns:
            return df, f"mix_shift skipped ('{col}' not in data)"

        applied = []
        for segment, vol_chg in params.volume_changes.items():
            mask = df[col].astype(str) == str(segment)
            factor = 1.0 + vol_chg
            for metric in ["earned_exposure", "incurred_loss", "claim_count", "written_premium"]:
                if metric in df.columns:
                    df.loc[mask, metric] = df.loc[mask, metric] * factor
            applied.append(f"{segment}: {vol_chg:+.1%}")

        note = "Mix shift: " + "; ".join(applied)
        return df, note

    def _apply_exit_segment(
        self, df: pd.DataFrame, params: ExitSegmentParams
    ) -> Tuple[pd.DataFrame, int]:
        """Remove rows matching the exit filters."""
        mask = pd.Series(True, index=df.index)
        for col, val in params.filters.items():
            if col in df.columns:
                mask &= df[col].astype(str) == str(val)
        n_exit = int(mask.sum())
        return df[~mask].copy(), n_exit

    def _apply_cat_environment(
        self, df: pd.DataFrame, params: CatEnvironmentParams
    ) -> Tuple[pd.DataFrame, str]:
        """Adjust CAT loss load."""
        df = df.copy()
        cat_col = "is_catastrophe"

        if cat_col in df.columns:
            cat_mask = df[cat_col].fillna(0) == 1
            if "incurred_loss" in df.columns:
                df.loc[cat_mask, "incurred_loss"] = (
                    df.loc[cat_mask, "incurred_loss"] * params.cat_multiplier
                )
            note = (
                f"CAT environment: {params.cat_multiplier:.2f}× CAT load "
                f"({int(cat_mask.sum())} CAT rows adjusted)"
            )
        else:
            # No CAT flag — apply to estimated CAT share (assume 5% historical)
            cat_share = 0.05
            if "incurred_loss" in df.columns:
                non_cat = df["incurred_loss"] * (1 - cat_share)
                cat_losses = df["incurred_loss"] * cat_share * params.cat_multiplier
                df["incurred_loss"] = non_cat + cat_losses
            note = (
                f"CAT environment: {params.cat_multiplier:.2f}× CAT load "
                f"(assumed {cat_share:.0%} CAT share, no is_catastrophe flag in data)"
            )

        return df, note

    def _apply_expense_initiative(
        self, df: pd.DataFrame, params: ExpenseInitiativeParams
    ) -> Tuple[pd.DataFrame, str]:
        """
        Record the expense ratio improvement in the engine for CR calculation.

        Does not modify the loss data — instead adjusts the engine's
        expense_ratio attribute used in combined ratio computation.
        """
        # Linear phase-in of expense savings
        full_savings = -params.delta_expense_ratio  # sign flip: delta is negative
        annual_savings = full_savings / max(params.timeline_years, 1)

        if "expense_ratio" in df.columns:
            df["expense_ratio"] = df["expense_ratio"] + params.delta_expense_ratio
        else:
            # Store for KPI computation
            df["_expense_ratio_adj"] = params.delta_expense_ratio

        note = (
            f"Expense initiative: {params.delta_expense_ratio:+.1%} expense ratio "
            f"phased in over {params.timeline_years}yr"
        )
        return df, note

    def _apply_enter_market(
        self, kpis: Dict[str, float], params: EnterMarketParams
    ) -> Tuple[Dict[str, float], str]:
        """
        Add new market segment P&L to existing KPIs.

        Assumes gradual ramp-up to full premium volume over ramp_up_years.
        Year 1 = 1/ramp_up_years of full volume, Year N = full volume.
        """
        kpis = dict(kpis)
        # Year 1 ramp fraction (conservative: half of full in first year)
        ramp_fraction = 1.0 / max(params.ramp_up_years, 1)
        new_premium = params.est_annual_premium * ramp_fraction
        new_losses = new_premium * params.est_loss_ratio
        new_expenses = new_premium * params.est_expense_ratio

        kpis["total_premium"] = kpis.get("total_premium", 0) + new_premium
        kpis["total_losses"] = kpis.get("total_losses", 0) + new_losses

        # Recalculate ratios
        tp = kpis["total_premium"]
        tl = kpis["total_losses"]
        te = kpis.get("total_expense", 0) + new_expenses

        kpis["loss_ratio"] = tl / tp if tp else np.nan
        kpis["expense_ratio"] = te / tp if tp else np.nan
        kpis["combined_ratio"] = kpis["loss_ratio"] + kpis.get("expense_ratio", self.expense_ratio)
        kpis["total_expense"] = te

        note = (
            f"New market entry: ${new_premium:,.0f} premium (year 1, "
            f"{ramp_fraction:.0%} ramp), expected LR={params.est_loss_ratio:.1%}"
        )
        return kpis, note

    # ------------------------------------------------------------------
    # KPI computation
    # ------------------------------------------------------------------

    def _compute_kpis(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute portfolio-level KPIs from a segment DataFrame.

        Returns
        -------
        dict with keys:
            total_exposure | total_losses | total_premium | total_claims |
            overall_frequency | overall_severity | overall_pure_premium |
            loss_ratio | expense_ratio | combined_ratio |
            cat_loss_ratio (if is_catastrophe column exists) |
            noncat_loss_ratio (if is_catastrophe column exists)
        """
        kpis: Dict[str, float] = {}

        total_exp = float(df["earned_exposure"].sum()) if "earned_exposure" in df.columns else 1.0
        total_loss = float(df["incurred_loss"].sum()) if "incurred_loss" in df.columns else 0.0
        total_claims = float(df["claim_count"].sum()) if "claim_count" in df.columns else 0.0

        kpis["total_exposure"] = total_exp
        kpis["total_losses"] = total_loss
        kpis["total_claims"] = total_claims

        kpis["overall_frequency"] = total_claims / max(total_exp, 1e-6)
        kpis["overall_severity"] = total_loss / max(total_claims, 1e-6)
        kpis["overall_pure_premium"] = total_loss / max(total_exp, 1e-6)

        # Premium
        if "written_premium" in df.columns:
            total_prem = float(df["written_premium"].sum())
            kpis["total_premium"] = total_prem
            kpis["loss_ratio"] = total_loss / max(total_prem, 1e-6)
        elif "earned_premium" in df.columns:
            total_prem = float(df["earned_premium"].sum())
            kpis["total_premium"] = total_prem
            kpis["loss_ratio"] = total_loss / max(total_prem, 1e-6)
        else:
            kpis["total_premium"] = np.nan
            kpis["loss_ratio"] = np.nan

        # Expense ratio
        if "expense_ratio" in df.columns:
            kpis["expense_ratio"] = float(df["expense_ratio"].mean())
        elif "_expense_ratio_adj" in df.columns:
            kpis["expense_ratio"] = self.expense_ratio + float(df["_expense_ratio_adj"].mean())
        else:
            kpis["expense_ratio"] = self.expense_ratio

        kpis["combined_ratio"] = kpis["loss_ratio"] + kpis["expense_ratio"] if not np.isnan(kpis["loss_ratio"]) else np.nan

        # CAT / non-CAT split
        if "is_catastrophe" in df.columns:
            cat_mask = df["is_catastrophe"].fillna(0) == 1
            cat_losses = float(df.loc[cat_mask, "incurred_loss"].sum()) if "incurred_loss" in df.columns else 0.0
            noncat_losses = total_loss - cat_losses
            total_prem_safe = kpis.get("total_premium", np.nan)
            kpis["cat_losses"] = cat_losses
            kpis["noncat_losses"] = noncat_losses
            if not np.isnan(total_prem_safe):
                kpis["cat_loss_ratio"] = cat_losses / max(total_prem_safe, 1e-6)
                kpis["noncat_loss_ratio"] = noncat_losses / max(total_prem_safe, 1e-6)

        return kpis

    def _build_segment_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a segment-level comparison DataFrame.

        Groups by territory (or lob_code if territory not available) and
        returns base vs. scenario KPIs side-by-side.
        """
        seg_col = None
        for candidate in ["territory", "lob_code", "class_code"]:
            if candidate in df.columns:
                seg_col = candidate
                break
        if seg_col is None:
            return pd.DataFrame()

        def _agg(data):
            g = data.groupby(seg_col).agg(
                exposure=("earned_exposure", "sum") if "earned_exposure" in data.columns else ("claim_count", "count"),
                losses=("incurred_loss", "sum") if "incurred_loss" in data.columns else ("claim_count", "sum"),
                claims=("claim_count", "sum") if "claim_count" in data.columns else ("incurred_loss", "count"),
                premium=("written_premium", "sum") if "written_premium" in data.columns else ("incurred_loss", "count"),
            )
            if "earned_exposure" in data.columns:
                g["pure_premium"] = g["losses"] / g["exposure"].replace(0, np.nan)
            if "written_premium" in data.columns:
                g["loss_ratio"] = g["losses"] / g["premium"].replace(0, np.nan)
            return g

        base_seg = _agg(self._df).add_suffix("_base")
        scen_seg = _agg(df).add_suffix("_scen")
        combined = base_seg.join(scen_seg, how="outer")

        if "losses_base" in combined.columns and "losses_scen" in combined.columns:
            combined["losses_delta"] = combined["losses_scen"] - combined["losses_base"]

        if "loss_ratio_base" in combined.columns and "loss_ratio_scen" in combined.columns:
            combined["lr_delta"] = combined["loss_ratio_scen"] - combined["loss_ratio_base"]

        return combined

    def _compute_ci(
        self,
        scen_kpis: Dict[str, float],
        n_boot: int,
        ci: float,
        random_state: int,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals on scenario KPIs.

        If n_boot > 0, bootstrap by resampling rows.
        If n_boot == 0, use a parametric approximation based on Poisson/Gamma
        variance assumptions.
        """
        if n_boot > 0:
            return self._bootstrap_ci(scen_kpis, n_boot, ci, random_state)
        else:
            return self._parametric_ci(scen_kpis, ci)

    def _parametric_ci(
        self, kpis: Dict[str, float], ci: float
    ) -> Dict[str, Tuple[float, float]]:
        """
        Parametric CI approximation using actuarial variance assumptions.

        For loss ratio: use normal approximation with CV estimated from
        Poisson frequency variance (1/n) and Gamma severity variance (1/n).
        """
        from scipy import stats as spstats

        z = spstats.norm.ppf((1 + ci) / 2)
        ci_dict = {}

        n_claims = kpis.get("total_claims", 100)
        n_claims = max(n_claims, 1)

        for kpi, val in kpis.items():
            if np.isnan(val):
                ci_dict[kpi] = (np.nan, np.nan)
                continue

            # Coefficient of variation approximations
            if kpi == "loss_ratio":
                # CV(LR) ≈ sqrt(1/n) from Poisson frequency + 1/n from Gamma severity
                cv = np.sqrt(2 / n_claims)
            elif kpi == "overall_frequency":
                cv = np.sqrt(1 / n_claims)
            elif kpi == "overall_severity":
                cv = np.sqrt(1 / n_claims) * 1.5  # severity more volatile
            elif kpi == "combined_ratio":
                cv = np.sqrt(2 / n_claims) * 0.9
            else:
                # Generic 5% for dollar amounts (premium, losses, etc.)
                cv = 0.05

            half_width = val * cv * z
            ci_dict[kpi] = (val - half_width, val + half_width)

        return ci_dict

    def _bootstrap_ci(
        self,
        scen_kpis: Dict[str, float],
        n_boot: int,
        ci: float,
        random_state: int,
    ) -> Dict[str, Tuple[float, float]]:
        """Bootstrap CI by resampling segment rows."""
        rng = np.random.default_rng(random_state)
        n = len(self._df)
        boot_kpis = {k: [] for k in scen_kpis}

        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            df_b = self._df.iloc[idx].copy()
            kpis_b = self._compute_kpis(df_b)
            for k in scen_kpis:
                boot_kpis[k].append(kpis_b.get(k, scen_kpis[k]))

        alpha_tail = (1 - ci) / 2
        ci_dict = {}
        for k, vals in boot_kpis.items():
            arr = np.array(vals)
            ci_dict[k] = (
                float(np.quantile(arr, alpha_tail)),
                float(np.quantile(arr, 1 - alpha_tail)),
            )
        return ci_dict

    def __repr__(self) -> str:
        lr = self._base_kpis.get("loss_ratio", np.nan)
        cr = self._base_kpis.get("combined_ratio", np.nan)
        return (
            f"ScenarioEngine(rows={len(self._df):,}, "
            f"base_LR={lr:.1%}, base_CR={cr:.1%})"
        )


# ---------------------------------------------------------------------------
# Pre-built scenario templates for common executive questions
# ---------------------------------------------------------------------------

def rate_action_scenario(
    name: str,
    segment_col: str,
    rate_changes: Dict[str, float],
    price_elasticity: float = -0.35,
) -> ScenarioParams:
    """Quick builder for a rate action scenario."""
    changes_str = ", ".join(f"{k}: {v:+.1%}" for k, v in rate_changes.items())
    return ScenarioParams(
        name=name,
        description=f"Rate action — {changes_str}",
        rate_action=RateActionParams(
            rate_changes=rate_changes,
            price_elasticity=price_elasticity,
            segment_col=segment_col,
        ),
    )


def frequency_stress_scenario(
    name: str,
    annual_trend: float,
    horizon_years: int = 3,
) -> ScenarioParams:
    """Quick builder for a frequency trend stress test."""
    return ScenarioParams(
        name=name,
        description=f"Frequency trend {annual_trend:+.1%}/yr for {horizon_years}yr",
        frequency_trend=FrequencyTrendParams(
            annual_trend=annual_trend,
            horizon_years=horizon_years,
        ),
    )


def severity_inflation_scenario(
    name: str,
    severity_multiplier: float,
    sustained: bool = False,
    horizon_years: int = 3,
) -> ScenarioParams:
    """Quick builder for a severity shock scenario."""
    mode = f"sustained {horizon_years}yr" if sustained else "one-time"
    return ScenarioParams(
        name=name,
        description=f"Severity {severity_multiplier:.2f}× ({mode})",
        severity_shock=SeverityShockParams(
            severity_multiplier=severity_multiplier,
            sustained=sustained,
            horizon_years=horizon_years,
        ),
    )


def cat_environment_scenario(
    name: str,
    cat_multiplier: float,
) -> ScenarioParams:
    """Quick builder for a CAT environment scenario."""
    direction = "increase" if cat_multiplier > 1 else "decrease"
    return ScenarioParams(
        name=name,
        description=f"CAT {direction}: {cat_multiplier:.2f}× expected CAT load",
        cat_environment=CatEnvironmentParams(cat_multiplier=cat_multiplier),
    )
