"""
auto_actuary.analytics.portfolio.market_breakdown
==================================================
Custom market breakdown category support.

Carriers segment their book of business in many different ways: personal vs.
commercial, preferred vs. non-standard, monoline vs. package, high-value homes
vs. standard.  These groupings and sub-groupings don't always map directly onto
line-of-business codes, and they vary significantly across carriers.

This module lets you define *named segment hierarchies* in plain Python (or via
YAML), then slice any loaded DataFrame against them — with loss ratio, exposure,
premium, and count summaries at every level.

Concepts
--------
MarketBreakdownConfig
    A tree of named groups and sub-groups, each defined by a predicate
    (a callable or a dict of column-filter conditions).

MarketBreakdownAnalysis
    Applies a MarketBreakdownConfig to policy / claim DataFrames and
    produces summary tables at the group, sub-group, and leaf levels.

Quick example
-------------
>>> from auto_actuary.analytics.portfolio import MarketBreakdownConfig, MarketBreakdownAnalysis

>>> config = MarketBreakdownConfig.from_dict({
...     "Personal Lines": {
...         "Preferred Auto": {"line_of_business": "PPA", "class_code": ["P1", "P2"]},
...         "Non-Standard Auto": {"line_of_business": "PPA", "class_code": ["NS1", "NS2"]},
...         "Homeowners": {"line_of_business": ["HO", "DP"]},
...     },
...     "Commercial Lines": {
...         "Small Commercial Auto": {
...             "line_of_business": "CA",
...             "written_premium": ("<=", 10_000),
...         },
...         "Large Commercial Auto": {
...             "line_of_business": "CA",
...             "written_premium": (">", 10_000),
...         },
...     },
... })

>>> mba = MarketBreakdownAnalysis(config, policies=policies_df, claims=claims_df,
...                               valuations=vals_df)
>>> mba.summary()          # DataFrame: group | subgroup | premium | losses | LR
>>> mba.by_group()         # DataFrame: one row per top-level group
>>> mba.by_subgroup()      # DataFrame: one row per sub-group
>>> mba.drilldown("Personal Lines", "Preferred Auto")  # policy-level slice
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------

_Predicate = Union[
    Callable[[pd.DataFrame], pd.Series],  # raw boolean mask function
    Dict[str, Any],                         # column-filter dict
]


def _build_mask(df: pd.DataFrame, predicate: _Predicate) -> pd.Series:
    """
    Convert a predicate to a boolean mask over *df*.

    Dict predicates support:
      - exact match:       "col": "value"  or  "col": ["v1", "v2"]
      - numeric compare:  "col": (">", 1000)  (operators: <, <=, >, >=, ==, !=)
      - callable:         "col": lambda s: s.str.startswith("P")
    """
    if callable(predicate) and not isinstance(predicate, dict):
        return predicate(df)

    mask = pd.Series(True, index=df.index)
    for col, condition in predicate.items():
        if col not in df.columns:
            logger.warning("MarketBreakdown: column '%s' not in DataFrame — condition skipped", col)
            continue

        col_series = df[col]

        if isinstance(condition, list):
            mask &= col_series.isin(condition)
        elif isinstance(condition, tuple) and len(condition) == 2:
            op, val = condition
            ops = {
                "<": col_series < val,
                "<=": col_series <= val,
                ">": col_series > val,
                ">=": col_series >= val,
                "==": col_series == val,
                "!=": col_series != val,
            }
            if op not in ops:
                raise ValueError(f"Unsupported operator '{op}' in MarketBreakdown predicate.")
            mask &= ops[op]
        elif callable(condition):
            mask &= condition(col_series)
        else:
            mask &= col_series == condition

    return mask


# ---------------------------------------------------------------------------
# SegmentNode
# ---------------------------------------------------------------------------

@dataclass
class SegmentNode:
    """
    A single named segment with an optional predicate and children.

    Parameters
    ----------
    name : str
        Display name for this segment.
    predicate : _Predicate, optional
        Filter condition.  None = matches all rows (useful for top-level groups
        that are defined purely by their children's union).
    children : list of SegmentNode
        Sub-groups of this segment.
    """
    name: str
    predicate: Optional[_Predicate] = None
    children: List["SegmentNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """Boolean mask for rows matching this segment."""
        if self.predicate is None:
            if self.children:
                # Union of children's masks
                combined = pd.Series(False, index=df.index)
                for child in self.children:
                    combined |= child.mask(df)
                return combined
            return pd.Series(True, index=df.index)
        return _build_mask(df, self.predicate)


# ---------------------------------------------------------------------------
# MarketBreakdownConfig
# ---------------------------------------------------------------------------

class MarketBreakdownConfig:
    """
    Hierarchical definition of market segments.

    Construct directly via :meth:`from_dict` for the most common use case.

    Parameters
    ----------
    segments : list of SegmentNode
        Top-level segment nodes.
    """

    def __init__(self, segments: List[SegmentNode]) -> None:
        self.segments = segments

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, definition: Dict[str, Any]) -> "MarketBreakdownConfig":
        """
        Build a MarketBreakdownConfig from a nested dict.

        Dict format::

            {
                "Group Name": {
                    "SubGroup A": { <predicate dict> },
                    "SubGroup B": { <predicate dict> },
                },
                "Leaf Group": { <predicate dict> },   # no children — treated as leaf
            }

        If a value is itself a nested dict of dicts (all values are dicts),
        it is treated as a group-with-children.  Otherwise it's a leaf predicate.
        """
        top_nodes = []
        for group_name, group_def in definition.items():
            node = cls._parse_node(group_name, group_def)
            top_nodes.append(node)
        return cls(top_nodes)

    @classmethod
    def from_yaml(cls, path: str) -> "MarketBreakdownConfig":
        """
        Load from a YAML file.

        YAML structure mirrors the dict format of :meth:`from_dict`.
        """
        import yaml  # type: ignore[import-untyped]
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @staticmethod
    def _parse_node(name: str, definition: Any) -> SegmentNode:
        """Recursively parse a segment definition."""
        if not isinstance(definition, dict):
            # Scalar value — treat as a predicate (e.g., a list of LOBs)
            return SegmentNode(name=name, predicate={"line_of_business": definition})

        # Check if all values are dicts (group with children) or mixed (leaf predicate)
        all_dict_values = all(isinstance(v, dict) for v in definition.values())

        if all_dict_values and definition:
            # This is a group with named sub-groups
            children = [
                MarketBreakdownConfig._parse_node(child_name, child_def)
                for child_name, child_def in definition.items()
            ]
            return SegmentNode(name=name, predicate=None, children=children)
        else:
            # This is a leaf node with a predicate dict
            return SegmentNode(name=name, predicate=definition)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_segments(self, indent: int = 0) -> List[str]:
        """Return a flat list of segment names with indentation."""
        lines = []
        for node in self.segments:
            lines.append("  " * indent + node.name)
            for child in node.children:
                lines.append("  " * (indent + 1) + child.name)
                for grandchild in child.children:
                    lines.append("  " * (indent + 2) + grandchild.name)
        return lines

    def __repr__(self) -> str:
        names = [n.name for n in self.segments]
        return f"MarketBreakdownConfig(groups={names})"


# ---------------------------------------------------------------------------
# MarketBreakdownAnalysis
# ---------------------------------------------------------------------------

class MarketBreakdownAnalysis:
    """
    Apply a MarketBreakdownConfig to portfolio data and produce summaries.

    Parameters
    ----------
    config : MarketBreakdownConfig
    policies : pd.DataFrame
        Must have: policy_id | written_premium | written_exposure |
                   line_of_business | and any columns used in predicates.
    claims : pd.DataFrame, optional
        Must have: policy_id | claim_id | accident_date
    valuations : pd.DataFrame, optional
        Must have: claim_id | incurred_loss (latest valuation)
    as_of_year : int, optional
        Filter policies/claims to a single accident/effective year.
    """

    def __init__(
        self,
        config: MarketBreakdownConfig,
        policies: pd.DataFrame,
        claims: Optional[pd.DataFrame] = None,
        valuations: Optional[pd.DataFrame] = None,
        as_of_year: Optional[int] = None,
    ) -> None:
        self.config = config
        self._policies = policies.copy()
        self._claims = claims.copy() if claims is not None else pd.DataFrame()
        self._vals = valuations.copy() if valuations is not None else pd.DataFrame()

        if as_of_year:
            if "effective_date" in self._policies.columns:
                self._policies = self._policies[
                    self._policies["effective_date"].dt.year == as_of_year
                ]
            if not self._claims.empty and "accident_date" in self._claims.columns:
                self._claims = self._claims[
                    self._claims["accident_date"].dt.year == as_of_year
                ]

        # Pre-join claims → latest valuation
        self._loss_by_policy = self._build_loss_by_policy()

    def _build_loss_by_policy(self) -> pd.DataFrame:
        """Join claims to latest valuations → losses summarised by policy_id."""
        if self._claims.empty or self._vals.empty:
            return pd.DataFrame(columns=["policy_id", "claim_count", "incurred_loss"])

        latest = (
            self._vals.sort_values("valuation_date")
            .groupby("claim_id")
            .last()
            .reset_index()[["claim_id", "incurred_loss"]]
        ) if "valuation_date" in self._vals.columns else self._vals[["claim_id", "incurred_loss"]]

        merged = self._claims[["claim_id", "policy_id"]].merge(latest, on="claim_id", how="left")
        merged["incurred_loss"] = merged["incurred_loss"].fillna(0)
        merged["claim_count"] = 1
        return merged.groupby("policy_id")[["claim_count", "incurred_loss"]].sum().reset_index()

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def _agg_segment(self, node: SegmentNode) -> Dict[str, float]:
        """Compute KPIs for a single SegmentNode over self._policies."""
        mask = node.mask(self._policies)
        pol_slice = self._policies[mask]

        wrt_prem = float(pol_slice["written_premium"].sum()) if "written_premium" in pol_slice.columns else 0.0
        wrt_exp = float(pol_slice["written_exposure"].sum()) if "written_exposure" in pol_slice.columns else 0.0
        pol_count = int(len(pol_slice))

        # Join losses
        if not self._loss_by_policy.empty and pol_count > 0:
            pol_ids = pol_slice["policy_id"].unique()
            loss_slice = self._loss_by_policy[self._loss_by_policy["policy_id"].isin(pol_ids)]
            claim_count = int(loss_slice["claim_count"].sum())
            incurred_loss = float(loss_slice["incurred_loss"].sum())
        else:
            claim_count = 0
            incurred_loss = 0.0

        ep = wrt_prem if wrt_prem != 0 else np.nan
        return {
            "policy_count": pol_count,
            "written_premium": wrt_prem,
            "written_exposure": wrt_exp,
            "claim_count": claim_count,
            "incurred_loss": incurred_loss,
            "loss_ratio": incurred_loss / ep if ep else np.nan,
            "frequency": claim_count / wrt_exp if wrt_exp else np.nan,
            "severity": incurred_loss / claim_count if claim_count else np.nan,
            "pure_premium": incurred_loss / wrt_exp if wrt_exp else np.nan,
        }

    # ------------------------------------------------------------------
    # Public summary tables
    # ------------------------------------------------------------------

    def by_group(self) -> pd.DataFrame:
        """
        Summary at the top-level group level.

        Returns
        -------
        pd.DataFrame indexed by group name.
        """
        rows = []
        for node in self.config.segments:
            row = {"group": node.name}
            row.update(self._agg_segment(node))
            rows.append(row)
        return pd.DataFrame(rows).set_index("group")

    def by_subgroup(self) -> pd.DataFrame:
        """
        Summary at the sub-group level (group × subgroup).

        Returns
        -------
        pd.DataFrame with MultiIndex (group, subgroup).
        """
        rows = []
        for node in self.config.segments:
            if node.is_leaf:
                row = {"group": node.name, "subgroup": "(all)"}
                row.update(self._agg_segment(node))
                rows.append(row)
            else:
                for child in node.children:
                    row = {"group": node.name, "subgroup": child.name}
                    row.update(self._agg_segment(child))
                    rows.append(row)
        return pd.DataFrame(rows).set_index(["group", "subgroup"])

    def summary(self) -> pd.DataFrame:
        """
        Full three-level summary: group | subgroup | KPIs.

        Leaves appear as their own group with subgroup='(all)'.
        """
        return self.by_subgroup()

    def drilldown(
        self,
        group: str,
        subgroup: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return the raw policy rows for a named segment.

        Parameters
        ----------
        group : str
            Top-level group name.
        subgroup : str, optional
            Sub-group name.  None = return all policies in the group.

        Returns
        -------
        pd.DataFrame
            Policy rows matching the segment, with loss columns joined.
        """
        # Find the node
        node = next((n for n in self.config.segments if n.name == group), None)
        if node is None:
            raise KeyError(f"Group '{group}' not found in MarketBreakdownConfig.")

        if subgroup is not None:
            child = next((c for c in node.children if c.name == subgroup), None)
            if child is None:
                raise KeyError(f"Subgroup '{subgroup}' not found under group '{group}'.")
            target = child
        else:
            target = node

        mask = target.mask(self._policies)
        pol_slice = self._policies[mask].copy()

        if not self._loss_by_policy.empty:
            pol_slice = pol_slice.merge(self._loss_by_policy, on="policy_id", how="left")
            pol_slice["incurred_loss"] = pol_slice["incurred_loss"].fillna(0)
            pol_slice["claim_count"] = pol_slice["claim_count"].fillna(0)

        return pol_slice

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        session: "ActuarySession",
        config: MarketBreakdownConfig,
        as_of_year: Optional[int] = None,
    ) -> "MarketBreakdownAnalysis":
        """Build from a loaded ActuarySession."""
        policies = session.loader["policies"] if "policies" in session.loader.loaded_tables else pd.DataFrame()
        claims = session.loader["claims"] if "claims" in session.loader.loaded_tables else None
        valuations = session.loader["valuations"] if "valuations" in session.loader.loaded_tables else None
        return cls(config, policies=policies, claims=claims, valuations=valuations, as_of_year=as_of_year)

    def __repr__(self) -> str:
        return (
            f"MarketBreakdownAnalysis("
            f"groups={[n.name for n in self.config.segments]}, "
            f"policies={len(self._policies)})"
        )
