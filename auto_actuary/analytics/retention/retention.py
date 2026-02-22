"""
auto_actuary.analytics.retention.retention
==========================================
Account and policy retention analysis.

Retention is one of the most fundamental KPIs for a P&C carrier.  A 1%
improvement in retention can have an outsized impact on profitability because:
  - Renewals are cheaper to acquire (no agent commission on marketing)
  - Renewing insureds carry lower selection risk over time (the "survivor" effect)
  - Long-tenured customers tend to have better loss ratios (Werner & Modlin Ch. 12)

This module computes retention at two levels:

Policy Retention
    The proportion of policies that survive to renew at expiration.
    Excludes mid-term cancellations; uses the expiration date as the
    opportunity window.

Account Retention
    The proportion of insured accounts (risk_id / insured_id) that remain
    active (at least one in-force policy) across consecutive policy years.
    An account may be retained even if an individual policy is not (e.g.,
    they switch from one LOB to another with the same carrier).

Segmentation
    Both metrics can be sliced by: LOB, territory, class_code, agent_id,
    and any other column present in the policies table.

Profitability lift
    The module can join loss data to compute the loss ratio split between
    retained and non-retained (lapsed) policies — the "retention lift" that
    quantifies how much better retained business performs.

Data requirements
-----------------
policies table must contain:
  - policy_id          unique policy term identifier
  - policy_number      shared across consecutive terms of the same policy
  - risk_id            insured / account identifier
  - effective_date     coverage start date
  - expiration_date    coverage end date
  - written_premium
  - line_of_business
  - transaction_type   (from transactions table; NB = new business, RN = renewal)

The transaction_type column is sourced from the ``transactions`` table, not the
policies table directly.  Load it separately via session.load_csv('transactions')
and pass it in, or join it to your policies extract before calling this class.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession

logger = logging.getLogger(__name__)

# Days of grace period for matching renewals (accounts for minor date drift)
_RENEWAL_GRACE_DAYS = 30


class RetentionAnalysis:
    """
    Account and policy retention analytics.

    Parameters
    ----------
    policies : pd.DataFrame
        One row per policy term.  Required columns: policy_id, policy_number,
        risk_id, effective_date, expiration_date, written_premium,
        line_of_business.  Optional: territory, class_code, agent_id,
        sub_line, transaction_type.
    transactions : pd.DataFrame, optional
        Transaction-level data.  Used to determine whether a policy originated
        as new business (NB) or renewal (RN) when transaction_type is not
        already in *policies*.  Must have: policy_id | transaction_type.
    valuations : pd.DataFrame, optional
        For computing loss ratios on retained vs. lapsed segments.
        Must have: claim_id | incurred_loss.
    claims : pd.DataFrame, optional
        Must have: claim_id | policy_id.
    lob : str, optional
        Filter to a single line of business.
    """

    def __init__(
        self,
        policies: pd.DataFrame,
        transactions: Optional[pd.DataFrame] = None,
        valuations: Optional[pd.DataFrame] = None,
        claims: Optional[pd.DataFrame] = None,
        lob: Optional[str] = None,
    ) -> None:
        self.lob = lob

        pol = policies.copy()
        if lob and "line_of_business" in pol.columns:
            pol = pol[pol["line_of_business"] == lob]

        # Attach transaction_type from transactions table if not already present
        if "transaction_type" not in pol.columns and transactions is not None:
            txn = transactions.copy()
            # For each policy, use the first transaction type (NB or RN)
            base_txn = (
                txn[txn.get("transaction_type", pd.Series()).isin(["NB", "RN"])]
                if "transaction_type" in txn.columns else txn
            )
            if "transaction_type" in base_txn.columns:
                first_txn = (
                    base_txn.sort_values("transaction_date" if "transaction_date" in base_txn.columns else base_txn.columns[0])
                    .groupby("policy_id")["transaction_type"]
                    .first()
                    .reset_index()
                )
                pol = pol.merge(first_txn, on="policy_id", how="left")

        self._policies = pol
        self._vals = valuations.copy() if valuations is not None else pd.DataFrame()
        self._claims = claims.copy() if claims is not None else pd.DataFrame()

        # Pre-compute: ensure dates are datetime
        for col in ["effective_date", "expiration_date", "cancel_date"]:
            if col in self._policies.columns:
                self._policies[col] = pd.to_datetime(self._policies[col], errors="coerce")

        # Policy year (based on effective date)
        if "effective_date" in self._policies.columns:
            self._policies["policy_year"] = self._policies["effective_date"].dt.year

        # Loss by policy (latest valuation)
        self._loss_by_policy: pd.DataFrame = self._build_loss_by_policy()

    def _build_loss_by_policy(self) -> pd.DataFrame:
        if self._claims.empty or self._vals.empty:
            return pd.DataFrame(columns=["policy_id", "incurred_loss"])

        loss_cols = ["claim_id", "incurred_loss"]
        avail = [c for c in loss_cols if c in self._vals.columns]
        if "valuation_date" in self._vals.columns:
            latest = (
                self._vals.sort_values("valuation_date")
                .groupby("claim_id").last().reset_index()[avail]
            )
        else:
            latest = self._vals[avail].copy()

        if "policy_id" not in self._claims.columns:
            return pd.DataFrame(columns=["policy_id", "incurred_loss"])

        merged = self._claims[["claim_id", "policy_id"]].merge(latest, on="claim_id", how="left")
        merged["incurred_loss"] = merged["incurred_loss"].fillna(0)
        return merged.groupby("policy_id")["incurred_loss"].sum().reset_index()

    # ------------------------------------------------------------------
    # Policy retention
    # ------------------------------------------------------------------

    def policy_retention(
        self,
        by: Optional[List[str]] = None,
        min_exposure_year: Optional[int] = None,
        max_exposure_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compute policy-level retention rates by policy year.

        Logic:
          For each policy term expiring in year Y, check whether a subsequent
          term with the same policy_number exists with an effective date within
          ``_RENEWAL_GRACE_DAYS`` days of the expiration date.

        Parameters
        ----------
        by : list of str, optional
            Additional grouping columns (e.g. ['territory', 'line_of_business']).
        min_exposure_year : int, optional
            Earliest expiration year to include.
        max_exposure_year : int, optional
            Latest expiration year to include.

        Returns
        -------
        pd.DataFrame
            Columns: expiration_year | policies_expiring | policies_renewed |
                     retention_rate | written_premium_expiring |
                     written_premium_renewed | premium_retention_rate
            (plus any *by* columns as additional index levels)
        """
        pol = self._policies.copy()

        if "expiration_date" not in pol.columns or "effective_date" not in pol.columns:
            logger.warning("RetentionAnalysis.policy_retention: "
                           "expiration_date or effective_date not in policies.")
            return pd.DataFrame()

        pol["expiration_year"] = pol["expiration_date"].dt.year

        if min_exposure_year:
            pol = pol[pol["expiration_year"] >= min_exposure_year]
        if max_exposure_year:
            pol = pol[pol["expiration_year"] <= max_exposure_year]

        # Build lookup: policy_number → set of effective_dates of subsequent terms
        next_terms: Dict[str, pd.DatetimeTZNaive] = {}
        if "policy_number" in pol.columns:
            for pnum, grp in pol.groupby("policy_number"):
                dates = sorted(grp["effective_date"].dropna())
                next_terms[str(pnum)] = dates

        def _has_renewal(row: pd.Series) -> bool:
            pnum = str(row.get("policy_number", ""))
            exp = row.get("expiration_date")
            if pd.isna(exp) or pnum not in next_terms:
                return False
            for eff in next_terms[pnum]:
                if eff is not None and pd.notna(eff):
                    delta = abs((eff - exp).days)
                    if delta <= _RENEWAL_GRACE_DAYS and eff > row.get("effective_date"):
                        return True
            return False

        if "policy_number" in pol.columns:
            pol["renewed"] = pol.apply(_has_renewal, axis=1).astype(int)
        else:
            # Fallback: use transaction_type if present
            if "transaction_type" in pol.columns:
                pol["renewed"] = pol["transaction_type"].eq("RN").shift(-1).fillna(False).astype(int)
            else:
                logger.warning("RetentionAnalysis: neither 'policy_number' nor 'transaction_type' "
                               "available — retention rates will not be accurate.")
                pol["renewed"] = 0

        group_cols = ["expiration_year"] + (by or [])
        group_cols = [c for c in group_cols if c in pol.columns]

        agg = pol.groupby(group_cols).agg(
            policies_expiring=("policy_id", "count"),
            policies_renewed=("renewed", "sum"),
            written_premium_expiring=("written_premium", "sum") if "written_premium" in pol.columns else ("policy_id", "count"),
        ).reset_index()

        agg["retention_rate"] = agg["policies_renewed"] / agg["policies_expiring"].replace(0, np.nan)

        if "written_premium" in pol.columns:
            renewed_prem = pol[pol["renewed"] == 1].groupby(group_cols)["written_premium"].sum().reset_index()
            renewed_prem.rename(columns={"written_premium": "written_premium_renewed"}, inplace=True)
            agg = agg.merge(renewed_prem, on=group_cols, how="left")
            agg["written_premium_renewed"] = agg["written_premium_renewed"].fillna(0)
            agg["premium_retention_rate"] = (
                agg["written_premium_renewed"] / agg["written_premium_expiring"].replace(0, np.nan)
            )

        return agg.set_index(group_cols)

    # ------------------------------------------------------------------
    # Account retention
    # ------------------------------------------------------------------

    def account_retention(
        self,
        by: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute account-level retention by policy year.

        An account is considered "retained" in year Y+1 if the same risk_id
        has at least one in-force policy with an effective_date in year Y+1.

        Parameters
        ----------
        by : list of str, optional
            Additional grouping columns (must be stable per account, e.g. LOB).

        Returns
        -------
        pd.DataFrame
            Columns: policy_year | accounts_active | accounts_retained |
                     account_retention_rate
        """
        pol = self._policies.copy()

        if "risk_id" not in pol.columns or "policy_year" not in pol.columns:
            logger.warning("RetentionAnalysis.account_retention: "
                           "risk_id or effective_date not in policies.")
            return pd.DataFrame()

        # Unique accounts per year
        years = sorted(pol["policy_year"].dropna().unique().astype(int))
        rows = []

        for yr in years:
            active_this = set(pol[pol["policy_year"] == yr]["risk_id"].dropna().unique())
            active_next = set(pol[pol["policy_year"] == yr + 1]["risk_id"].dropna().unique()) if (yr + 1) in years else set()
            retained = active_this & active_next
            rows.append({
                "policy_year": int(yr),
                "accounts_active": len(active_this),
                "accounts_retained": len(retained),
                "account_retention_rate": len(retained) / len(active_this) if active_this else np.nan,
            })

        return pd.DataFrame(rows).set_index("policy_year")

    # ------------------------------------------------------------------
    # Retention by segment
    # ------------------------------------------------------------------

    def retention_by_segment(
        self,
        segment: str = "territory",
    ) -> pd.DataFrame:
        """
        Policy retention rate broken down by a single segment dimension.

        Parameters
        ----------
        segment : str
            Column to segment by: 'territory', 'class_code', 'agent_id', etc.

        Returns
        -------
        pd.DataFrame
            Columns: <segment> | policies_expiring | retention_rate |
                     premium_retention_rate (if premium available)
        """
        return self.policy_retention(by=[segment])

    # ------------------------------------------------------------------
    # Profitability lift from retention
    # ------------------------------------------------------------------

    def retention_profitability_lift(self) -> pd.DataFrame:
        """
        Compare loss ratios between retained and lapsed policies.

        Returns
        -------
        pd.DataFrame
            Columns: segment | written_premium | incurred_loss | loss_ratio
            where segment ∈ {'Renewed', 'Lapsed (Did Not Renew)'}

        Requires claims and valuations to have been loaded.
        """
        if self._loss_by_policy.empty:
            logger.warning("RetentionAnalysis.retention_profitability_lift: "
                           "no loss data — load claims and valuations first.")
            return pd.DataFrame()

        pol = self._policies.copy()
        if "policy_number" not in pol.columns:
            return pd.DataFrame()

        # Compute renewed flag (same logic as policy_retention)
        next_terms_map: Dict[str, list] = {}
        for pnum, grp in pol.groupby("policy_number"):
            next_terms_map[str(pnum)] = sorted(grp["effective_date"].dropna())

        def _renewed(row: pd.Series) -> bool:
            pnum = str(row.get("policy_number", ""))
            exp = row.get("expiration_date")
            if pd.isna(exp) or pnum not in next_terms_map:
                return False
            for eff in next_terms_map[pnum]:
                if eff is not None and pd.notna(eff):
                    if abs((eff - exp).days) <= _RENEWAL_GRACE_DAYS and eff > row["effective_date"]:
                        return True
            return False

        pol["renewed"] = pol.apply(_renewed, axis=1)
        pol = pol.merge(self._loss_by_policy, on="policy_id", how="left")
        pol["incurred_loss"] = pol["incurred_loss"].fillna(0)
        pol["segment"] = pol["renewed"].map({True: "Renewed", False: "Lapsed (Did Not Renew)"})

        result = pol.groupby("segment").agg(
            written_premium=("written_premium", "sum"),
            incurred_loss=("incurred_loss", "sum"),
            policy_count=("policy_id", "count"),
        )
        result["loss_ratio"] = result["incurred_loss"] / result["written_premium"].replace(0, np.nan)
        return result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, float]:
        """Scalar summary of overall retention metrics."""
        pol_ret = self.policy_retention()
        acct_ret = self.account_retention()

        overall_pol_ret = (
            float(pol_ret["policies_renewed"].sum() / pol_ret["policies_expiring"].sum())
            if not pol_ret.empty and pol_ret["policies_expiring"].sum() > 0 else np.nan
        )
        overall_acct_ret = float(acct_ret["account_retention_rate"].mean()) if not acct_ret.empty else np.nan

        return {
            "overall_policy_retention_rate": overall_pol_ret,
            "overall_account_retention_rate": overall_acct_ret,
            "total_policies_expiring": int(pol_ret["policies_expiring"].sum()) if not pol_ret.empty else 0,
            "total_policies_renewed": int(pol_ret["policies_renewed"].sum()) if not pol_ret.empty else 0,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"RetentionAnalysis(lob={self.lob!r}, "
            f"policy_retention={s.get('overall_policy_retention_rate', 0):.1%}, "
            f"account_retention={s.get('overall_account_retention_rate', 0):.1%})"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        lob: Optional[str] = None,
    ) -> "RetentionAnalysis":
        """Build from a loaded ActuarySession."""
        policies = session.loader["policies"] if "policies" in session.loader.loaded_tables else pd.DataFrame()
        txns = session.loader["transactions"] if "transactions" in session.loader.loaded_tables else None
        vals = session.loader["valuations"] if "valuations" in session.loader.loaded_tables else None
        claims = session.loader["claims"] if "claims" in session.loader.loaded_tables else None
        return cls(policies=policies, transactions=txns, valuations=vals, claims=claims, lob=lob)
