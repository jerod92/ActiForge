"""
auto_actuary.analytics.portfolio.segment_analytics
====================================================
Segment Analytics — the engine behind the Segment Analytics Dashboard.

Answers the core business questions a carrier needs for segment-level strategy:
  - Which segments are growing / shrinking?
  - Which retain good business and lose bad?
  - How profitable is each segment over time?
  - What is the estimated Customer Lifetime Value (CLV) per segment?
  - Where should we take rate action or change appetite?

Time-series orientation
-----------------------
All outputs are indexed by (period, segment_value) so they can be rendered
as interactive trend charts.  "Period" is either accident/effective year or
calendar quarter depending on config.time_granularity.

Data requirements
-----------------
policies table — required columns:
    policy_id, risk_id, policy_number, effective_date, expiration_date,
    written_premium, earned_premium (or written_premium as proxy),
    line_of_business, <segment columns>

Optional but enriching:
    transaction_type (NB vs RN), agent_id

claims + valuations — for loss metrics per segment:
    standard columns as loaded by DataLoader

Retention calculation
---------------------
A policy is considered "retained" if there is another policy row with the same
policy_number effective within [expiration_date - 60 days, expiration_date + 60 days].
This handles minor date drift and off-cycle renewals.

CLV estimation
--------------
Expected CLV = (average annual premium per account) × (expected tenure in years)
             × (1 - combined_ratio)   [underwriting profit margin]

Expected tenure = 1 / (1 - retention_rate)  [geometric series]

This is a simple first-order CLV; the dashboard surfaces this alongside actual
retention and loss data so users can prioritize segment actions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession
    from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)

# Grace window for renewal matching (days)
_RENEWAL_GRACE_DAYS = 60


class SegmentAnalytics:
    """
    Segment-level time-series analytics.

    Parameters
    ----------
    policies : pd.DataFrame
        Canonical policies table (post-schema-rename).
    losses : pd.DataFrame
        Merged claims+latest-valuation table with incurred_loss, accident_date.
    segment_cols : list of str
        Column names to use as segment dimensions (e.g. ['territory', 'class_code']).
    lob : str, optional
        Filter to a single line of business.
    time_granularity : str
        'year' (default) or 'quarter'.
    """

    def __init__(
        self,
        policies: pd.DataFrame,
        losses: pd.DataFrame,
        segment_cols: List[str],
        lob: Optional[str] = None,
        time_granularity: str = "year",
    ) -> None:
        self.lob = lob
        self.segment_cols = segment_cols
        self.time_granularity = time_granularity

        # Filter to LOB
        filt = lambda df, col="line_of_business": (
            df[df[col] == lob].copy() if lob and col in df.columns else df.copy()
        )
        self._policies = filt(policies)
        self._losses = filt(losses)

        # Derive period column
        self._policies = self._policies.copy()
        if "effective_date" in self._policies.columns:
            self._policies["_period"] = self._period_from_date(
                self._policies["effective_date"]
            )
        else:
            self._policies["_period"] = np.nan

        if "accident_date" in self._losses.columns:
            self._losses = self._losses.copy()
            self._losses["_period"] = self._period_from_date(
                self._losses["accident_date"]
            )

        # Earned premium fallback
        if "earned_premium" not in self._policies.columns:
            self._policies["earned_premium"] = self._policies.get(
                "written_premium", pd.Series(0, index=self._policies.index)
            )

        # Transaction type fallback
        if "transaction_type" not in self._policies.columns:
            self._policies["transaction_type"] = "NB"

        # Determine valid underwriting periods — years/quarters where policies
        # were actually written (i.e., have positive earned premium).  Claims
        # can have accident dates outside this range (spillover into next year),
        # so we use this set to filter trend outputs.
        if "_period" in self._policies.columns and "earned_premium" in self._policies.columns:
            valid_mask = self._policies["earned_premium"] > 0
            self._valid_periods: set = set(self._policies.loc[valid_mask, "_period"].dropna().unique())
        else:
            self._valid_periods = set()

    # ------------------------------------------------------------------
    # Period helper
    # ------------------------------------------------------------------

    def _period_from_date(self, date_series: pd.Series) -> pd.Series:
        if self.time_granularity == "quarter":
            return date_series.dt.to_period("Q").astype(str)
        return date_series.dt.year

    # ------------------------------------------------------------------
    # Premium & policy count trends
    # ------------------------------------------------------------------

    def premium_trend(self, segment: str) -> pd.DataFrame:
        """
        Written and earned premium by (period, segment_value).

        Returns
        -------
        DataFrame indexed by (period, segment_value) with columns:
            written_premium | earned_premium | policy_count | nb_count | rn_count
        """
        if segment not in self._policies.columns:
            logger.warning("Segment column '%s' not in policies table", segment)
            return pd.DataFrame()

        grp = self._policies.groupby(["_period", segment])
        agg = grp.agg(
            written_premium=("written_premium", "sum"),
            earned_premium=("earned_premium", "sum"),
            policy_count=("policy_id", "nunique"),
        ).reset_index()

        # NB vs RN breakdown
        if "transaction_type" in self._policies.columns:
            nb_df = (
                self._policies[self._policies["transaction_type"] == "NB"]
                .groupby(["_period", segment])["policy_id"]
                .nunique()
                .reset_index(name="nb_count")
            )
            rn_df = (
                self._policies[self._policies["transaction_type"] == "RN"]
                .groupby(["_period", segment])["policy_id"]
                .nunique()
                .reset_index(name="rn_count")
            )
            agg = agg.merge(nb_df, on=["_period", segment], how="left")
            agg = agg.merge(rn_df, on=["_period", segment], how="left")
            agg["nb_count"] = agg["nb_count"].fillna(0).astype(int)
            agg["rn_count"] = agg["rn_count"].fillna(0).astype(int)

        agg.rename(columns={"_period": "period", segment: "segment_value"}, inplace=True)
        agg["segment_col"] = segment
        return agg.set_index(["period", "segment_value"])

    # ------------------------------------------------------------------
    # Loss metrics per segment
    # ------------------------------------------------------------------

    def loss_trend(self, segment: str) -> pd.DataFrame:
        """
        Incurred loss, claim count, loss ratio by (period, segment_value).

        Loss ratio = incurred_loss / earned_premium for the same period/segment.
        """
        if self._losses.empty or segment not in self._losses.columns:
            return pd.DataFrame()

        loss_grp = (
            self._losses.groupby(["_period", segment])
            .agg(
                incurred_loss=("incurred_loss", "sum"),
                claim_count=("claim_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"_period": "period", segment: "segment_value"})
        )

        # Join with earned premium for loss ratio
        prem = self.premium_trend(segment).reset_index()[
            ["period", "segment_value", "earned_premium"]
        ]
        merged = loss_grp.merge(prem, on=["period", "segment_value"], how="left")
        ep = merged["earned_premium"].replace(0, np.nan)
        merged["loss_ratio"] = merged["incurred_loss"] / ep
        merged["pure_premium"] = merged["incurred_loss"] / merged.get(
            "earned_exposure", ep
        ).replace(0, np.nan)
        merged["segment_col"] = segment

        # Drop spillover periods where no premiums were written (e.g. claims
        # with accident dates falling one year after the last writing year).
        if self._valid_periods:
            merged = merged[merged["period"].isin(self._valid_periods)]

        return merged.set_index(["period", "segment_value"])

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def retention_trend(self, segment: str) -> pd.DataFrame:
        """
        Policy retention rate by (period, segment_value).

        Retention = policies that renewed / policies that expired in the period.
        A policy is matched as retained if the same policy_number appears again
        within _RENEWAL_GRACE_DAYS of its expiration date.

        Returns
        -------
        DataFrame indexed by (period, segment_value) with columns:
            expiring_count | retained_count | retention_rate | lapsed_premium
        """
        pol = self._policies.copy()
        if "policy_number" not in pol.columns or "expiration_date" not in pol.columns:
            logger.warning("Retention requires policy_number and expiration_date columns")
            return pd.DataFrame()
        if segment not in pol.columns:
            return pd.DataFrame()

        # Each policy's expiration period
        pol["_exp_period"] = self._period_from_date(pol["expiration_date"])

        # Build a lookup: policy_number → set of effective dates
        eff_lookup = pol.groupby("policy_number")["effective_date"].apply(set).to_dict()

        def is_retained(row) -> int:
            eff_dates = eff_lookup.get(row["policy_number"], set())
            exp = row["expiration_date"]
            if pd.isna(exp):
                return 0
            for eff in eff_dates:
                delta = abs((eff - exp).days)
                if delta <= _RENEWAL_GRACE_DAYS and eff > row["effective_date"]:
                    return 1
            return 0

        pol["_retained"] = pol.apply(is_retained, axis=1)

        grp = pol.groupby(["_exp_period", segment])
        agg = grp.agg(
            expiring_count=("policy_id", "nunique"),
            retained_count=("_retained", "sum"),
            expiring_premium=("written_premium", "sum"),
        ).reset_index()

        agg["retention_rate"] = agg["retained_count"] / agg["expiring_count"].replace(0, np.nan)
        agg["lapsed_premium"] = agg["expiring_premium"] * (1 - agg["retention_rate"].fillna(0))

        agg.rename(
            columns={"_exp_period": "period", segment: "segment_value"},
            inplace=True,
        )
        agg["segment_col"] = segment

        # Drop periods beyond the last writing year — policies that expire in the
        # year after the final underwriting year show 0% retention (no next year
        # renewals exist in the data), which is misleading.
        if self._valid_periods:
            agg = agg[agg["period"].isin(self._valid_periods)]

        return agg.set_index(["period", "segment_value"])

    # ------------------------------------------------------------------
    # Customer Lifetime Value
    # ------------------------------------------------------------------

    def clv_by_segment(self, segment: str, expense_ratio: float = 0.30) -> pd.DataFrame:
        """
        Estimated Customer Lifetime Value (CLV) by segment.

        CLV = avg_annual_premium × expected_tenure × (1 - combined_ratio)

        where:
          expected_tenure   = 1 / (1 - retention_rate)  [geometric series]
          combined_ratio    = loss_ratio + expense_ratio

        Returns
        -------
        DataFrame indexed by segment_value with columns:
            avg_annual_premium | retention_rate | expected_tenure_years |
            loss_ratio | expense_ratio | combined_ratio |
            uw_profit_margin | estimated_clv
        """
        if segment not in self._policies.columns:
            return pd.DataFrame()

        # Retention across all periods
        ret_df = self.retention_trend(segment).reset_index()
        if ret_df.empty:
            return pd.DataFrame()
        ret_agg = (
            ret_df.groupby("segment_value")
            .agg(
                total_expiring=("expiring_count", "sum"),
                total_retained=("retained_count", "sum"),
            )
            .reset_index()
        )
        ret_agg["retention_rate"] = (
            ret_agg["total_retained"] / ret_agg["total_expiring"].replace(0, np.nan)
        ).clip(0, 0.99)

        # Average annual premium by segment
        prem_agg = (
            self._policies.groupby(segment)
            .agg(
                avg_annual_premium=("earned_premium", "mean"),
            )
            .reset_index()
            .rename(columns={segment: "segment_value"})
        )

        # Loss ratio by segment
        if not self._losses.empty and segment in self._losses.columns:
            # Only include losses from valid underwriting periods so that
            # spillover claims (accident dates in year N+1 with zero EP) do not
            # inflate the aggregate loss ratio.
            losses_valid = self._losses
            if self._valid_periods and "_period" in self._losses.columns:
                losses_valid = self._losses[self._losses["_period"].isin(self._valid_periods)]
            loss_agg = (
                losses_valid.groupby(segment)
                .agg(incurred_loss=("incurred_loss", "sum"))
                .reset_index()
                .rename(columns={segment: "segment_value"})
            )
            prem_tot = (
                self._policies.groupby(segment)["earned_premium"]
                .sum()
                .reset_index()
                .rename(columns={segment: "segment_value", "earned_premium": "total_ep"})
            )
            lr_df = loss_agg.merge(prem_tot, on="segment_value", how="left")
            lr_df["loss_ratio"] = lr_df["incurred_loss"] / lr_df["total_ep"].replace(0, np.nan)
        else:
            lr_df = pd.DataFrame(
                {"segment_value": prem_agg["segment_value"], "loss_ratio": np.nan}
            )

        # Assemble
        result = ret_agg.merge(prem_agg, on="segment_value", how="outer")
        result = result.merge(lr_df[["segment_value", "loss_ratio"]], on="segment_value", how="left")

        result["expected_tenure_years"] = 1 / (1 - result["retention_rate"].clip(0, 0.99)).replace(0, np.nan)
        result["expense_ratio"] = expense_ratio
        result["combined_ratio"] = result["loss_ratio"].fillna(0.65) + expense_ratio
        result["uw_profit_margin"] = (1 - result["combined_ratio"]).clip(-1, 1)
        result["estimated_clv"] = (
            result["avg_annual_premium"]
            * result["expected_tenure_years"]
            * result["uw_profit_margin"].clip(0, None)
        )

        result["segment_col"] = segment
        return result.set_index("segment_value")

    # ------------------------------------------------------------------
    # Summary table (all periods, all segments combined)
    # ------------------------------------------------------------------

    def segment_scorecard(self, segment: str) -> pd.DataFrame:
        """
        One-row-per-segment scorecard combining latest period metrics.

        Columns: written_premium | earned_premium | policy_count |
                 incurred_loss | loss_ratio | retention_rate | estimated_clv
        """
        prem = self.premium_trend(segment).reset_index()
        if prem.empty:
            return pd.DataFrame()

        # Use the last two periods — "latest" and "prior" for YoY
        periods = sorted(prem["period"].unique())
        latest_period = periods[-1] if periods else None
        prior_period = periods[-2] if len(periods) >= 2 else None

        latest = prem[prem["period"] == latest_period].set_index("segment_value")

        # Loss ratio for latest period
        loss = self.loss_trend(segment).reset_index()
        if not loss.empty:
            loss_latest = loss[loss["period"] == latest_period].set_index("segment_value")
            latest = latest.join(loss_latest[["loss_ratio", "incurred_loss"]], how="left")

        # Retention for the prior→latest transition
        ret = self.retention_trend(segment).reset_index()
        if not ret.empty and prior_period is not None:
            ret_prior = ret[ret["period"] == prior_period].set_index("segment_value")
            latest = latest.join(ret_prior[["retention_rate"]], how="left")

        # YoY premium growth
        if prior_period is not None:
            prior = prem[prem["period"] == prior_period].set_index("segment_value")
            latest = latest.join(prior[["written_premium"]].rename(
                columns={"written_premium": "prior_wp"}
            ), how="left")
            latest["wp_yoy"] = (
                (latest["written_premium"] - latest["prior_wp"])
                / latest["prior_wp"].replace(0, np.nan)
            )

        # CLV
        clv = self.clv_by_segment(segment)
        if not clv.empty:
            latest = latest.join(clv[["estimated_clv"]], how="left")

        latest["segment_col"] = segment
        latest["latest_period"] = latest_period
        return latest

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: Optional[str] = None,
        segment_cols: Optional[List[str]] = None,
    ) -> "SegmentAnalytics":
        """Build SegmentAnalytics from a loaded ActuarySession."""
        cfg = session.config
        if segment_cols is None:
            segment_cols = cfg.all_segments

        policies = session.loader["policies"].copy() if "policies" in session.loader else pd.DataFrame()
        claims = session.loader["claims"].copy() if "claims" in session.loader else pd.DataFrame()
        vals = session.loader["valuations"].copy() if "valuations" in session.loader else pd.DataFrame()

        # Build losses: latest valuation per claim merged with claim header
        losses = pd.DataFrame()
        if not claims.empty and not vals.empty and "valuation_date" in vals.columns:
            latest_vals = (
                vals.sort_values("valuation_date")
                .groupby("claim_id")
                .last()
                .reset_index()[["claim_id", "incurred_loss", "paid_loss", "paid_alae", "case_alae"]]
            )
            losses = claims.merge(latest_vals, on="claim_id", how="left")
            for col in ["incurred_loss", "paid_loss", "paid_alae", "case_alae"]:
                if col in losses.columns:
                    losses[col] = losses[col].fillna(0)

        return cls(
            policies=policies,
            losses=losses,
            segment_cols=segment_cols,
            lob=lob,
            time_granularity=cfg.time_granularity,
        )
