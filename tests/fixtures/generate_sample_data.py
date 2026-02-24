"""
Generate synthetic but actuarially realistic sample data for testing.

Run directly to create CSVs:
    python tests/fixtures/generate_sample_data.py --output-dir data/

The generated data follows these actuarial assumptions:
  - PPA line, 2018-2023 accident years
  - ~2,000 policies per year, 12-month terms
  - Frequency: ~8% (80 claims per 1,000 car-years)
  - Severity: ~$8,000 average BI; $3,500 COMP/COLL
  - Loss development follows a realistic emergence pattern
  - Three rate changes: +4% (2019), -2% (2021), +6% (2023)
  - ~5% of losses are CAT (hail events)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)  # reproducible

COVERAGES = ["BI", "PD", "COMP", "COLL", "MED"]
TERRITORIES = ["North", "South", "East", "West", "Metro"]
CLASS_CODES = ["01", "02", "03", "04"]  # preferred, standard, nonstandard, youthful
YEARS = list(range(2018, 2024))

# Loss emergence pattern (% reported by age 12, 24, …)
EMERGENCE = {
    "BI":   [0.35, 0.65, 0.82, 0.91, 0.96, 0.99, 1.00],
    "PD":   [0.75, 0.92, 0.97, 0.99, 1.00, 1.00, 1.00],
    "COMP": [0.85, 0.96, 0.99, 1.00, 1.00, 1.00, 1.00],
    "COLL": [0.80, 0.94, 0.98, 1.00, 1.00, 1.00, 1.00],
    "MED":  [0.50, 0.78, 0.90, 0.96, 0.99, 1.00, 1.00],
}

ULTIMATE_SEVERITY = {
    "BI":   8_500, "PD": 3_200, "COMP": 2_800, "COLL": 4_100, "MED": 1_200,
}

FREQUENCY = {
    "BI": 0.030, "PD": 0.065, "COMP": 0.045, "COLL": 0.055, "MED": 0.015,
}

ANNUAL_LOSS_TREND = 0.04   # +4% per year
ANNUAL_FREQ_TREND = 0.02   # +2%
ANNUAL_SEV_TREND  = 0.025  # +2.5%

RATE_CHANGES = [
    {"eff_date": "2019-01-01", "lob_code": "PPA", "terr_code": None, "class_code": None, "rate_chg_pct": 0.04},
    {"eff_date": "2021-07-01", "lob_code": "PPA", "terr_code": None, "class_code": None, "rate_chg_pct": -0.02},
    {"eff_date": "2023-01-01", "lob_code": "PPA", "terr_code": None, "class_code": None, "rate_chg_pct": 0.06},
]

VAL_DATES = pd.to_datetime([f"{yr}-12-31" for yr in range(2018, 2025)])


def generate_policies(n_per_year: int = 2_000) -> pd.DataFrame:
    """
    Generate a realistic policy portfolio with proper renewal chains.

    Each account (insured) has a stable policy_number that persists across
    renewal terms so that the retention analysis can match consecutive terms.
    Retention rates vary by class and territory to make segment analytics
    interesting.
    """
    base_premium = 900  # base annual premium at 2018 rates
    cutoff = pd.Timestamp("2023-12-31")

    rate_index = 1.0
    rate_schedule = {pd.Timestamp(r["eff_date"]): r["rate_chg_pct"] for r in RATE_CHANGES}
    _applied_changes: set = set()

    # -----------------------------------------------------------------------
    # Build rate index per year (applied once at year boundary)
    # -----------------------------------------------------------------------
    rate_by_year: dict = {}
    ri = 1.0
    applied: set = set()
    for yr in YEARS:
        for chg_date, pct in sorted(rate_schedule.items()):
            if chg_date.year <= yr and chg_date not in applied:
                ri *= (1 + pct)
                applied.add(chg_date)
        rate_by_year[yr] = ri

    # -----------------------------------------------------------------------
    # Model insured accounts
    # Each account starts in a random year and may lapse/renew each year.
    # Retention probability varies by class (class 01 = best, 04 = worst).
    # -----------------------------------------------------------------------
    RETENTION_PROB = {"01": 0.88, "02": 0.82, "03": 0.75, "04": 0.65}
    TERR_RETENTION = {"North": 1.00, "South": 0.98, "East": 0.97, "West": 1.01, "Metro": 0.95}

    records = []
    pid = 1
    acct_id = 1
    # Pool of active accounts: dict of acct_id → {eff, exp, terr, cls, pol_num, agent}
    active_accounts: dict = {}

    # Seed the pool with n_per_year new business in the first year (2018)
    yr0 = YEARS[0]
    ri0 = rate_by_year[yr0]
    for _ in range(n_per_year):
        eff = pd.Timestamp(f"{yr0}-01-01") + pd.Timedelta(days=int(RNG.integers(0, 365)))
        exp = eff + pd.DateOffset(years=1)
        terr = RNG.choice(TERRITORIES)
        cls  = RNG.choice(CLASS_CODES)
        pol_num = f"PPA-{acct_id:07d}"
        agent = f"AGT-{RNG.integers(1, 50):03d}"
        cls_mult  = {"01": 0.85, "02": 1.00, "03": 1.20, "04": 1.50}[cls]
        terr_mult = {"North": 0.95, "South": 1.05, "East": 1.00, "West": 0.92, "Metro": 1.15}[terr]
        prem = base_premium * ri0 * cls_mult * terr_mult * RNG.lognormal(0, 0.1)
        policy_days = max((exp - eff).days, 1)
        earn_end = min(pd.Timestamp(exp), cutoff)
        earn_days = max((earn_end - eff).days, 0)
        records.append({
            "policy_id":     pid,
            "policy_number": pol_num,
            "written_date":  eff,
            "eff_date":      eff,
            "exp_date":      pd.Timestamp(exp),
            "cancel_date":   None,
            "lob_code":      "PPA",
            "sub_line":      "PPA",
            "terr_code":     terr,
            "class_code":    cls,
            "insured_id":    f"INS-{acct_id:07d}",
            "agent_code":    agent,
            "wrt_prem":      round(prem, 2),
            "ern_prem":      round(prem * earn_days / policy_days, 2),
            "wrt_exposure":  1.0,
            "exp_unit":      "car-year",
            "txn_type":      "NB",
        })
        active_accounts[acct_id] = {
            "eff": eff, "exp": pd.Timestamp(exp),
            "terr": terr, "cls": cls, "pol_num": pol_num, "agent": agent
        }
        acct_id += 1
        pid += 1

    for yr in YEARS:
        ri = rate_by_year[yr]
        year_start = pd.Timestamp(f"{yr}-01-01")
        year_end   = pd.Timestamp(f"{yr}-12-31")

        # ---- Process renewals for accounts expiring this year ----
        still_active: dict = {}
        for aid, acc in active_accounts.items():
            exp = acc["exp"]
            if exp.year != yr:
                still_active[aid] = acc
                continue
            # Renewal decision
            ret_p = RETENTION_PROB[acc["cls"]] * TERR_RETENTION[acc["terr"]]
            if RNG.random() < ret_p:
                # Renew
                new_eff = exp
                new_exp = new_eff + pd.DateOffset(years=1)
                still_active[aid] = {**acc, "eff": new_eff, "exp": new_exp}
                cls_mult  = {"01": 0.85, "02": 1.00, "03": 1.20, "04": 1.50}[acc["cls"]]
                terr_mult = {"North": 0.95, "South": 1.05, "East": 1.00, "West": 0.92, "Metro": 1.15}[acc["terr"]]
                prem = base_premium * ri * cls_mult * terr_mult * RNG.lognormal(0, 0.08)
                policy_days = max((new_exp - new_eff).days, 1)
                earn_end = min(new_exp, cutoff)
                earn_days = max((earn_end - new_eff).days, 0)
                records.append({
                    "policy_id":     pid,
                    "policy_number": acc["pol_num"],
                    "written_date":  new_eff,
                    "eff_date":      new_eff,
                    "exp_date":      pd.Timestamp(new_exp),
                    "cancel_date":   None,
                    "lob_code":      "PPA",
                    "sub_line":      "PPA",
                    "terr_code":     acc["terr"],
                    "class_code":    acc["cls"],
                    "insured_id":    f"INS-{aid:07d}",
                    "agent_code":    acc["agent"],
                    "wrt_prem":      round(prem, 2),
                    "ern_prem":      round(prem * earn_days / policy_days, 2),
                    "wrt_exposure":  1.0,
                    "exp_unit":      "car-year",
                    "txn_type":      "RN",
                })
                pid += 1
            # else: lapsed — account exits

        active_accounts = still_active

        # ---- Write new business to maintain portfolio size ----
        current_size = sum(1 for a in active_accounts.values() if a["eff"].year <= yr)
        nb_needed = max(0, n_per_year - len(active_accounts))
        for _ in range(nb_needed):
            eff = year_start + pd.Timedelta(days=int(RNG.integers(0, 365)))
            exp = eff + pd.DateOffset(years=1)
            terr = RNG.choice(TERRITORIES)
            cls  = RNG.choice(CLASS_CODES)
            pol_num = f"PPA-{acct_id:07d}"
            agent = f"AGT-{RNG.integers(1, 50):03d}"
            cls_mult  = {"01": 0.85, "02": 1.00, "03": 1.20, "04": 1.50}[cls]
            terr_mult = {"North": 0.95, "South": 1.05, "East": 1.00, "West": 0.92, "Metro": 1.15}[terr]
            prem = base_premium * ri * cls_mult * terr_mult * RNG.lognormal(0, 0.1)
            policy_days = max((exp - eff).days, 1)
            earn_end = min(pd.Timestamp(exp), cutoff)
            earn_days = max((earn_end - eff).days, 0)
            records.append({
                "policy_id":     pid,
                "policy_number": pol_num,
                "written_date":  eff,
                "eff_date":      eff,
                "exp_date":      pd.Timestamp(exp),
                "cancel_date":   None,
                "lob_code":      "PPA",
                "sub_line":      "PPA",
                "terr_code":     terr,
                "class_code":    cls,
                "insured_id":    f"INS-{acct_id:07d}",
                "agent_code":    agent,
                "wrt_prem":      round(prem, 2),
                "ern_prem":      round(prem * earn_days / policy_days, 2),
                "wrt_exposure":  1.0,
                "exp_unit":      "car-year",
                "txn_type":      "NB",
            })
            active_accounts[acct_id] = {"eff": eff, "exp": exp, "terr": terr, "cls": cls,
                                         "pol_num": pol_num, "agent": agent}
            acct_id += 1
            pid += 1

        # ---- Record existing in-force accounts that were written before this year ----
        # (capture accounts whose effective date is in this year but not yet recorded)
        for aid, acc in active_accounts.items():
            if acc["eff"].year == yr and not any(
                r["policy_number"] == acc["pol_num"] and r["eff_date"] == acc["eff"]
                for r in records[-nb_needed:]
            ):
                pass  # already recorded above in the renewal or NB block

    df = pd.DataFrame(records)
    return df


def generate_claims(policies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    claim_records = []
    val_records = []
    cid = 1

    for _, pol in policies.iterrows():
        yr = pol["eff_date"].year
        trend_yrs = yr - 2018
        terr = pol["terr_code"]
        cls  = pol["class_code"]

        for cov in COVERAGES:
            base_freq = FREQUENCY[cov]
            trended_freq = base_freq * ((1 + ANNUAL_FREQ_TREND) ** trend_yrs)

            # Territory/class adjustments
            terr_mult = {"North": 0.9, "South": 1.1, "East": 1.0, "West": 0.85, "Metro": 1.2}[terr]
            cls_mult  = {"01": 0.8, "02": 1.0, "03": 1.3, "04": 1.8}[cls]
            adj_freq  = trended_freq * terr_mult * cls_mult

            # Poisson claim count
            n_claims = RNG.poisson(adj_freq)

            for _ in range(n_claims):
                # Accident date (uniform over policy year)
                loss_date = pol["eff_date"] + pd.Timedelta(days=int(RNG.integers(0, 365)))
                if loss_date > pol["exp_date"]:
                    continue

                base_sev = ULTIMATE_SEVERITY[cov]
                trended_sev = base_sev * ((1 + ANNUAL_SEV_TREND) ** trend_yrs)
                ultimate = max(0, float(RNG.lognormal(np.log(trended_sev), 0.8)))

                is_cat = 1 if (cov in ("COMP", "COLL") and RNG.random() < 0.05) else 0
                cat_code = f"CAT{loss_date.year}" if is_cat else None

                report_lag_days = int(RNG.exponential(30 if cov == "PD" else 90))
                report_date = loss_date + pd.Timedelta(days=report_lag_days)

                # Status: claim is closed if accident year is old enough
                years_old = 2024 - loss_date.year
                close_prob = 0.95 if years_old >= 5 else (0.80 if years_old >= 3 else 0.55)
                status = "C" if RNG.random() < close_prob else "O"
                close_date = None
                if status == "C":
                    close_date = loss_date + pd.Timedelta(days=int(RNG.exponential(365 * 2)))

                claim_records.append({
                    "claim_id":      cid,
                    "policy_id":     pol["policy_id"],
                    "loss_date":     loss_date,
                    "report_date":   report_date,
                    "close_date":    close_date,
                    "reopen_date":   None,
                    "cov_code":      cov,
                    "lob_code":      "PPA",
                    "terr_code":     terr,
                    "cause_code":    RNG.choice(["ACCIDENT", "WEATHER", "THEFT", "OTHER"]),
                    "status":        status,
                    "is_cat":        is_cat,
                    "cat_code":      cat_code,
                })

                # Loss valuations at each year-end
                emergence = EMERGENCE[cov]
                for val_idx, val_date in enumerate(VAL_DATES):
                    if val_date < report_date:
                        continue

                    # Age since accident date
                    age_months = (val_date.year - loss_date.year) * 12 + (val_date.month - loss_date.month)
                    age_idx = min(age_months // 12, len(emergence) - 1)
                    pct = emergence[age_idx]

                    # Add random noise to emergence
                    pct_noisy = min(1.0, max(0, pct + RNG.normal(0, 0.02)))
                    reported = ultimate * pct_noisy

                    # Split into paid and case
                    if status == "C" and val_date >= (close_date or val_date):
                        paid = reported
                        case = 0.0
                    else:
                        paid_frac = pct_noisy * 0.7
                        paid = ultimate * max(0, min(paid_frac, pct_noisy))
                        case = max(0, reported - paid)

                    alae_pct = {"BI": 0.15, "PD": 0.05, "COMP": 0.04, "COLL": 0.04, "MED": 0.10}[cov]

                    val_records.append({
                        "claim_id":      cid,
                        "val_date":      val_date,
                        "paid_loss":     round(paid, 2),
                        "case_reserve":  round(case, 2),
                        "incurred_loss": round(paid + case, 2),
                        "paid_alae":     round(paid * alae_pct, 2),
                        "case_alae":     round(case * alae_pct * 0.5, 2),
                        "paid_cnt":      1 if status == "C" and val_date >= (close_date or val_date) else 0,
                        "open_cnt":      0 if status == "C" and val_date >= (close_date or val_date) else 1,
                    })

                cid += 1

    claims_df = pd.DataFrame(claim_records)
    vals_df   = pd.DataFrame(val_records)
    return claims_df, vals_df


def generate_rate_changes() -> pd.DataFrame:
    return pd.DataFrame(RATE_CHANGES)


def generate_transactions(policies: pd.DataFrame) -> pd.DataFrame:
    """Generate a flat transactions table derived from the policies DataFrame."""
    records = []
    tid = 1
    for _, pol in policies.iterrows():
        records.append({
            "transaction_id":   tid,
            "policy_id":        pol["policy_id"],
            "transaction_date": pol["written_date"],
            "effective_date":   pol["eff_date"],
            "transaction_type": pol["txn_type"],
            "written_premium":  pol["wrt_prem"],
            "lob_code":         pol["lob_code"],
            "terr_code":        pol["terr_code"],
            "class_code":       pol["class_code"],
        })
        tid += 1
    return pd.DataFrame(records)


def generate_expenses(policies: pd.DataFrame) -> pd.DataFrame:
    records = []
    for yr in YEARS:
        ep = policies[policies["eff_date"].dt.year == yr]["wrt_prem"].sum()
        records.extend([
            {"cal_year": yr, "lob_code": "PPA", "exp_type": "Commissions", "exp_amount": ep * 0.12, "wrt_prem": ep, "ern_prem": ep},
            {"cal_year": yr, "lob_code": "PPA", "exp_type": "Taxes & Fees", "exp_amount": ep * 0.03, "wrt_prem": ep, "ern_prem": ep},
            {"cal_year": yr, "lob_code": "PPA", "exp_type": "G&A", "exp_amount": ep * 0.05, "wrt_prem": ep, "ern_prem": ep},
            {"cal_year": yr, "lob_code": "PPA", "exp_type": "ULAE", "exp_amount": ep * 0.025, "wrt_prem": ep, "ern_prem": ep},
        ])
    return pd.DataFrame(records)


def generate_all(output_dir: str = "data", n_per_year: int = 500) -> dict[str, Path]:
    """
    Generate all sample CSVs and write to output_dir.

    Parameters
    ----------
    n_per_year : int
        Policies per accident year.  500 is fast; use 2000 for realistic volume.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_per_year} policies/year × {len(YEARS)} years…")
    policies = generate_policies(n_per_year)
    policies.to_csv(out / "policies.csv", index=False)
    print(f"  policies.csv: {len(policies):,} rows")

    print("Generating claims and valuations…")
    claims, valuations = generate_claims(policies)
    claims.to_csv(out / "claims.csv", index=False)
    valuations.to_csv(out / "valuations.csv", index=False)
    print(f"  claims.csv: {len(claims):,} rows")
    print(f"  valuations.csv: {len(valuations):,} rows")

    transactions = generate_transactions(policies)
    transactions.to_csv(out / "transactions.csv", index=False)
    print(f"  transactions.csv: {len(transactions):,} rows")

    rate_changes = generate_rate_changes()
    rate_changes.to_csv(out / "rate_changes.csv", index=False)
    print(f"  rate_changes.csv: {len(rate_changes)} rows")

    expenses = generate_expenses(policies)
    expenses.to_csv(out / "expenses.csv", index=False)
    print(f"  expenses.csv: {len(expenses)} rows")

    print(f"\n✓ Sample data written to {out.resolve()}/")
    return {
        "policies": out / "policies.csv",
        "claims": out / "claims.csv",
        "valuations": out / "valuations.csv",
        "transactions": out / "transactions.csv",
        "rate_changes": out / "rate_changes.csv",
        "expenses": out / "expenses.csv",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--n-per-year", type=int, default=500)
    args = parser.parse_args()
    generate_all(args.output_dir, args.n_per_year)
