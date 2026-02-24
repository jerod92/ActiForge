"""
auto_actuary.core.config
========================
Load and resolve schema.yaml + actuarial_assumptions.yaml.

The schema maps user DB column names → canonical column aliases.
The assumptions hold all actuarial parameters (LDF methods, trend, expenses…).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_CANONICAL_TABLES = {
    "policies",
    "transactions",
    "claims",
    "valuations",
    "coverages",
    "rate_changes",
    "expenses",
}


class ActuaryConfig:
    """
    Unified configuration object.

    Parameters
    ----------
    schema_path : str or Path
        Path to schema.yaml
    assumptions_path : str or Path, optional
        Path to actuarial_assumptions.yaml.  Defaults to same directory as schema.
    lob_path : str or Path, optional
        Path to lines_of_business.yaml.  Defaults to same directory as schema.
    """

    def __init__(
        self,
        schema_path: str | Path,
        assumptions_path: Optional[str | Path] = None,
        lob_path: Optional[str | Path] = None,
    ) -> None:
        schema_path = Path(schema_path).expanduser().resolve()
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema config not found: {schema_path}")

        config_dir = schema_path.parent

        if assumptions_path is None:
            assumptions_path = config_dir / "actuarial_assumptions.yaml"
        if lob_path is None:
            lob_path = config_dir / "lines_of_business.yaml"

        with open(schema_path) as fh:
            self._schema: Dict[str, Any] = yaml.safe_load(fh) or {}

        self._assumptions: Dict[str, Any] = {}
        if Path(assumptions_path).exists():
            with open(assumptions_path) as fh:
                self._assumptions = yaml.safe_load(fh) or {}

        self._lob: Dict[str, Any] = {}
        if Path(lob_path).exists():
            with open(lob_path) as fh:
                self._lob = yaml.safe_load(fh) or {}

        # Build reverse maps: canonical → db_column, per table
        self._reverse_maps: Dict[str, Dict[str, str]] = {}
        for table in _CANONICAL_TABLES:
            if table in self._schema:
                self._reverse_maps[table] = {
                    canonical: db_col
                    for canonical, db_col in self._schema[table].items()
                    if db_col  # skip blank entries
                }

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def db_col(self, table: str, canonical: str) -> str:
        """Return the actual DB column name for a canonical alias."""
        try:
            return self._reverse_maps[table][canonical]
        except KeyError:
            # Fall back to canonical itself (user may already use canonical names)
            return canonical

    def rename_map(self, table: str) -> Dict[str, str]:
        """
        Return a dict suitable for ``pd.DataFrame.rename(columns=…)``
        that maps DB column names → canonical names for *table*.
        """
        if table not in self._reverse_maps:
            return {}
        return {db: can for can, db in self._reverse_maps[table].items()}

    def canonical_cols(self, table: str) -> list[str]:
        """List all canonical column names defined for *table*."""
        return list(self._reverse_maps.get(table, {}).keys())

    # ------------------------------------------------------------------
    # Source / connection helpers
    # ------------------------------------------------------------------

    @property
    def source_type(self) -> str:
        return self._schema.get("source", {}).get("type", "csv")

    @property
    def base_path(self) -> Path:
        raw = self._schema.get("source", {}).get("base_path", "./data")
        return Path(raw).expanduser()

    @property
    def connection_string(self) -> Optional[str]:
        return self._schema.get("source", {}).get("connection_string")

    # ------------------------------------------------------------------
    # Assumption helpers (dot-path access)
    # ------------------------------------------------------------------

    def assumption(self, *keys: str, default: Any = None) -> Any:
        """
        Get a nested assumption value.

        Example
        -------
        >>> cfg.assumption("triangles", "selected_ldf_method")
        'volume_weighted_5yr'
        """
        node = self._assumptions
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
            if node is default:
                return default
        return node

    # ------------------------------------------------------------------
    # LOB helpers
    # ------------------------------------------------------------------

    @property
    def lobs(self) -> Dict[str, Any]:
        return self._lob.get("lines_of_business", {})

    def lob_label(self, lob_code: str) -> str:
        return self.lobs.get(lob_code, {}).get("label", lob_code)

    def lob_exposure_unit(self, lob_code: str) -> str:
        return self.lobs.get(lob_code, {}).get("exposure_unit", "unit")

    def coverage_label(self, code: str) -> str:
        return self._lob.get("coverage_codes", {}).get(code, code)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def company_name(self) -> str:
        return self._assumptions.get("reporting", {}).get("company_name", "P&C Carrier")

    @property
    def output_dir(self) -> Path:
        raw = self._assumptions.get("reporting", {}).get("output_dir", "./output")
        return Path(raw).expanduser()

    @property
    def primary_color(self) -> str:
        return self._assumptions.get("reporting", {}).get("primary_color", "#003366")

    @property
    def accent_color(self) -> str:
        return self._assumptions.get("reporting", {}).get("accent_color", "#0099CC")

    # ------------------------------------------------------------------
    # Segmentation helpers
    # ------------------------------------------------------------------

    @property
    def geo_segments(self) -> list:
        """Column names for geographic segmentation (from schema.yaml segmentation.geo_segments)."""
        return self._schema.get("segmentation", {}).get("geo_segments", ["territory"])

    @property
    def market_segments(self) -> list:
        """Column names for market segmentation (from schema.yaml segmentation.market_segments)."""
        return self._schema.get("segmentation", {}).get("market_segments", ["class_code"])

    @property
    def all_segments(self) -> list:
        """Union of geo and market segment column names, deduplicated."""
        seen: set = set()
        result = []
        for col in self.geo_segments + self.market_segments:
            if col not in seen:
                seen.add(col)
                result.append(col)
        return result

    def segment_label(self, col: str) -> str:
        """Human-readable label for a segment column."""
        labels = self._schema.get("segmentation", {}).get("segment_labels", {})
        return labels.get(col, col.replace("_", " ").title())

    @property
    def time_granularity(self) -> str:
        """Time granularity for segment trend charts: 'year' or 'quarter'."""
        return self._schema.get("segmentation", {}).get("time_granularity", "year")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(cls, config_dir: str | Path = "config") -> "ActuaryConfig":
        """Load from a directory containing schema.yaml."""
        config_dir = Path(config_dir)
        return cls(
            schema_path=config_dir / "schema.yaml",
            assumptions_path=config_dir / "actuarial_assumptions.yaml",
            lob_path=config_dir / "lines_of_business.yaml",
        )

    def __repr__(self) -> str:
        return (
            f"ActuaryConfig(source={self.source_type!r}, "
            f"tables={list(self._reverse_maps.keys())})"
        )
