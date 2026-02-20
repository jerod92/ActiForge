"""
auto_actuary.core.data_loader
==============================
Load raw data from CSV files, SQL queries, SQLAlchemy engines, or plain
DataFrames and normalize column names to canonical aliases via the schema config.

Data Contract
-------------
Each table has a canonical set of column names.  See config/schema.yaml for
the mapping.  The loader renames incoming columns to canonical names and
casts date columns automatically.

Supported sources:
  - CSV file (pd.read_csv)
  - SQL query result file (CSV output from user-written SQL)
  - SQLAlchemy connection + SQL string
  - Raw pandas DataFrame (caller handles aliasing)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)

# Canonical date columns per table
_DATE_COLS: Dict[str, list[str]] = {
    "policies": ["written_date", "effective_date", "expiration_date", "cancel_date"],
    "transactions": ["transaction_date", "effective_date"],
    "claims": ["accident_date", "report_date", "close_date", "reopen_date"],
    "valuations": ["valuation_date"],
    "rate_changes": ["effective_date"],
    "expenses": [],
}

# Required canonical columns per table (loader warns if missing)
_REQUIRED_COLS: Dict[str, list[str]] = {
    "policies": ["policy_id", "effective_date", "expiration_date", "written_premium", "line_of_business"],
    "transactions": ["transaction_id", "policy_id", "transaction_date", "transaction_type", "written_premium"],
    "claims": ["claim_id", "policy_id", "accident_date", "coverage_code", "claim_status"],
    "valuations": ["claim_id", "valuation_date", "paid_loss", "incurred_loss"],
    "rate_changes": ["effective_date", "line_of_business", "rate_change_pct"],
    "expenses": ["calendar_year", "line_of_business", "expense_type", "amount"],
}


class DataLoader:
    """
    Loads and normalizes data for the actuary session.

    Parameters
    ----------
    config : ActuaryConfig
    """

    def __init__(self, config: ActuaryConfig) -> None:
        self.config = config
        self._frames: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public load methods
    # ------------------------------------------------------------------

    def load_csv(
        self,
        table: str,
        path: Union[str, Path],
        **read_csv_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load a CSV file, rename columns to canonical aliases, and store.

        Parameters
        ----------
        table : str
            One of: policies | transactions | claims | valuations |
                    coverages | rate_changes | expenses
        path : str or Path
            Path to the CSV file.
        **read_csv_kwargs
            Passed directly to pd.read_csv.
        """
        path = Path(path)
        logger.info("Loading %s from %s", table, path)
        df = pd.read_csv(path, **read_csv_kwargs)
        return self._ingest(table, df)

    def load_dataframe(self, table: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Register a pre-loaded DataFrame.  The DataFrame may use either
        DB column names (will be renamed per schema) or canonical names.
        """
        return self._ingest(table, df.copy())

    def load_sql(
        self,
        table: str,
        sql: str,
        engine: Any,  # sqlalchemy.Engine
    ) -> pd.DataFrame:
        """
        Execute *sql* against *engine* and ingest the result.

        Parameters
        ----------
        table : str
            Canonical table name.
        sql : str
            SQL query returning the columns described in docs/query_specifications.md
        engine : sqlalchemy.Engine
        """
        try:
            df = pd.read_sql(sql, engine)
        except Exception as exc:
            raise RuntimeError(
                f"SQL query for '{table}' failed: {exc}\n"
                "See docs/query_specifications.md for required output format."
            ) from exc
        return self._ingest(table, df)

    def load_sql_file(
        self,
        table: str,
        sql_path: Union[str, Path],
        engine: Any,
    ) -> pd.DataFrame:
        """Execute a .sql file and ingest the result."""
        sql_path = Path(sql_path)
        sql = sql_path.read_text()
        return self.load_sql(table, sql, engine)

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------

    def get(self, table: str) -> pd.DataFrame:
        """Return the canonical DataFrame for *table*, or raise KeyError."""
        if table not in self._frames:
            raise KeyError(
                f"Table '{table}' not loaded.  Call load_csv / load_dataframe first."
            )
        return self._frames[table]

    def __getitem__(self, table: str) -> pd.DataFrame:
        return self.get(table)

    def __contains__(self, table: str) -> bool:
        return table in self._frames

    @property
    def loaded_tables(self) -> list[str]:
        return list(self._frames.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ingest(self, table: str, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns, parse dates, validate, and store."""
        # Rename DB columns → canonical
        rename = self.config.rename_map(table)
        df = df.rename(columns=rename)

        # Parse date columns
        for col in _DATE_COLS.get(table, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Derive incurred_loss if not present
        if table == "valuations":
            if "incurred_loss" not in df.columns:
                if "paid_loss" in df.columns and "case_reserve" in df.columns:
                    df["incurred_loss"] = df["paid_loss"].fillna(0) + df["case_reserve"].fillna(0)
                    logger.debug("valuations: computed incurred_loss = paid_loss + case_reserve")

        # Validate required columns
        missing = [c for c in _REQUIRED_COLS.get(table, []) if c not in df.columns]
        if missing:
            logger.warning(
                "Table '%s' is missing canonical columns: %s.  "
                "Check schema.yaml mappings and your query output.",
                table,
                missing,
            )

        self._frames[table] = df
        logger.info("  → %d rows loaded into '%s'", len(df), table)
        return df


# ---------------------------------------------------------------------------
# Convenience: compute earned premium from policy dates
# ---------------------------------------------------------------------------

def compute_earned_premium(
    policies: pd.DataFrame,
    as_of_date: pd.Timestamp,
    year_col: str = "accident_year",
) -> pd.DataFrame:
    """
    Compute earned premium for each policy for each calendar year using
    the straight-line (pro-rata) method.

    Returns a DataFrame with columns:
        policy_id | calendar_year | earned_premium | earned_exposure
    """
    records = []
    for _, row in policies.iterrows():
        eff = row["effective_date"]
        exp = row.get("expiration_date") or (eff + pd.DateOffset(years=1))
        wrt_prem = row.get("written_premium", 0)
        wrt_exp = row.get("written_exposure", 0)
        policy_days = max((exp - eff).days, 1)

        # Determine which calendar years this policy spans
        for yr in range(eff.year, min(exp.year, as_of_date.year) + 1):
            yr_start = pd.Timestamp(f"{yr}-01-01")
            yr_end = pd.Timestamp(f"{yr}-12-31")
            earn_start = max(eff, yr_start)
            earn_end = min(exp, yr_end, as_of_date)
            if earn_end <= earn_start:
                continue
            earn_days = (earn_end - earn_start).days
            earn_pct = earn_days / policy_days
            records.append(
                {
                    "policy_id": row["policy_id"],
                    "calendar_year": yr,
                    "earned_premium": wrt_prem * earn_pct,
                    "earned_exposure": wrt_exp * earn_pct,
                }
            )

    return pd.DataFrame(records)


def dev_age_months(origin_date: pd.Timestamp, valuation_date: pd.Timestamp) -> int:
    """
    Compute development age in months (rounded to nearest 12 for annual triangles).
    """
    months = (
        (valuation_date.year - origin_date.year) * 12
        + (valuation_date.month - origin_date.month)
    )
    return max(months, 0)
