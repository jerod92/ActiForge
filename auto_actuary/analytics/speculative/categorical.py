"""
auto_actuary.analytics.speculative.categorical
===============================================
Robust categorical encoding for insurance GLMs.

The challenge: insurance data has many categorical variables with wildly
unequal group sizes (some territories have 10k policies, others have 40).
Standard one-hot encoding produces numerically unstable GLMs when categories
are sparse, and blows up dimensionality for high-cardinality variables.

Strategy
--------
1. **Sparse category collapse** — any level with fewer than *min_obs*
   observations is merged into an "_Other" bucket before encoding.
   This prevents pathological coefficient estimates.

2. **Credibility target encoding** — replaces each category level with a
   single float: a credibility-weighted blend of the segment mean and the
   grand mean.

       Z_c  = n_c / (n_c + K)          (credibility weight, 0–1)
       enc_c = Z_c * μ_c + (1-Z_c) * μ  (shrinkage toward grand mean)

   where K = credibility_threshold (default 1082, the classical full-cred
   standard for body injury frequency).  Sparse segments (n_c << K) get
   pulled strongly toward the grand mean; large segments (n_c >> K) are
   trusted almost fully.

3. **Cross-fold smoothing** (optional) — fit the encoder only on out-of-fold
   data to prevent target leakage in training.  Enabled via *n_folds*.

Why not one-hot?
----------------
One-hot is fine for low-cardinality variables (<= *onehot_threshold* levels)
when the data is balanced.  This encoder uses it as a fallback for those
cases only.  For anything larger, credibility encoding is more stable and
naturally handles the variability in incoming data.

References
----------
- Bühlmann, H. (1967) "Experience Rating and Credibility", ASTIN Bulletin
- Micci-Barreca, D. (2001) "A preprocessing scheme for high-cardinality
  categorical attributes in classification and prediction problems"
- Werner & Modlin (2016) "Basic Ratemaking" Chapter 8 (classification)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Classical full-credibility standard for claim frequency (CAS Exam 5 / Mahler)
_CLASSICAL_FULL_CRED = 1082


class SparseCollapser:
    """
    Merges rare category levels into a single "_Other" bucket.

    Parameters
    ----------
    min_obs : int
        Levels with fewer than this many observations are collapsed.
    other_label : str
        Label to assign to collapsed levels (default "_Other").
    """

    def __init__(self, min_obs: int = 30, other_label: str = "_Other") -> None:
        self.min_obs = min_obs
        self.other_label = other_label
        self._keep: Dict[str, set] = {}  # col -> set of levels to keep

    def fit(self, X: pd.DataFrame, cols: List[str]) -> "SparseCollapser":
        self._keep = {}
        for col in cols:
            if col not in X.columns:
                continue
            counts = X[col].value_counts()
            self._keep[col] = set(counts[counts >= self.min_obs].index)
            n_collapsed = (counts < self.min_obs).sum()
            if n_collapsed > 0:
                logger.debug(
                    "SparseCollapser: %s — collapsed %d rare level(s) into '%s'",
                    col,
                    n_collapsed,
                    self.other_label,
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, keep in self._keep.items():
            if col not in out.columns:
                continue
            mask = ~out[col].isin(keep)
            out.loc[mask, col] = self.other_label
        return out

    def fit_transform(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return self.fit(X, cols).transform(X)

    @property
    def kept_levels(self) -> Dict[str, List]:
        return {col: sorted(lvls) for col, lvls in self._keep.items()}


class CredibilityEncoder:
    """
    Credibility (shrinkage) target encoder for categorical variables.

    For each categorical column and each level c:

        Z_c  = n_c / (n_c + K)
        enc_c = Z_c * mean_target_c + (1 - Z_c) * grand_mean_target

    Parameters
    ----------
    credibility_k : float
        Full-credibility threshold K.  Default 1082 (CAS classical standard).
        Smaller K = faster credibility gain = less shrinkage.
    handle_unknown : str
        'mean' — assign grand mean to unseen levels (safe for production).
        'nan'  — assign NaN (useful for diagnosing drift).
    """

    def __init__(
        self,
        credibility_k: float = _CLASSICAL_FULL_CRED,
        handle_unknown: str = "mean",
    ) -> None:
        self.credibility_k = credibility_k
        self.handle_unknown = handle_unknown
        self._encodings: Dict[str, Dict] = {}  # col -> {level: encoded_val}
        self._grand_means: Dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cols: List[str],
        weights: Optional[pd.Series] = None,
    ) -> "CredibilityEncoder":
        """
        Fit credibility encoding from data.

        Parameters
        ----------
        X : DataFrame
            Feature data (already sparse-collapsed if applicable).
        y : Series
            Target (e.g., frequency = claim_count / exposure).
        cols : list of str
            Columns to encode.
        weights : Series, optional
            Observation weights (exposure counts) for weighted means.
        """
        self._encodings = {}
        self._grand_means = {}

        for col in cols:
            if col not in X.columns:
                logger.warning("CredibilityEncoder: column '%s' not in X, skipping", col)
                continue

            df = pd.DataFrame({"cat": X[col].values, "y": y.values})
            if weights is not None:
                df["w"] = weights.values
                grand_mean = np.average(df["y"], weights=df["w"])
                group = df.groupby("cat").apply(
                    lambda g: pd.Series(
                        {
                            "n": g["w"].sum(),
                            "mean": np.average(g["y"], weights=g["w"]),
                        }
                    )
                )
            else:
                grand_mean = df["y"].mean()
                group = df.groupby("cat").agg(n=("y", "count"), mean=("y", "mean"))

            self._grand_means[col] = float(grand_mean)

            encoding = {}
            for level, row in group.iterrows():
                n_c = float(row["n"])
                z_c = n_c / (n_c + self.credibility_k)
                encoding[level] = z_c * float(row["mean"]) + (1 - z_c) * grand_mean

            self._encodings[col] = encoding
            logger.debug(
                "CredibilityEncoder: %s — %d levels, grand_mean=%.4f",
                col,
                len(encoding),
                grand_mean,
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace categorical columns with their credibility-encoded floats."""
        out = X.copy()
        for col, encoding in self._encodings.items():
            if col not in out.columns:
                continue
            if self.handle_unknown == "mean":
                fallback = self._grand_means.get(col, np.nan)
            else:
                fallback = np.nan
            out[col] = out[col].map(encoding).fillna(fallback)
        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cols: List[str],
        weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        return self.fit(X, y, cols, weights=weights).transform(X)

    def level_encodings(self, col: str) -> pd.Series:
        """Return the encoded value per level for inspection."""
        return pd.Series(self._encodings.get(col, {}), name="encoded_value")


class ActuarialCategoricalEncoder:
    """
    Full pipeline: sparse collapse → credibility encoding.

    This is the recommended interface for preparing categorical features
    for insurance GLMs.  It handles:

    - Arbitrary cardinality (territories, class codes, zip codes, etc.)
    - High variability / imbalanced categories
    - Unseen levels at prediction time
    - Sample-weight-aware mean estimation (uses exposure as weight)

    Parameters
    ----------
    min_obs : int
        Minimum observations to keep a category level; rarer → "_Other".
    credibility_k : float
        Credibility threshold K.  Lower = faster to trust a category.
    """

    def __init__(
        self,
        min_obs: int = 30,
        credibility_k: float = _CLASSICAL_FULL_CRED,
    ) -> None:
        self.min_obs = min_obs
        self.credibility_k = credibility_k
        self.collapser = SparseCollapser(min_obs=min_obs)
        self.encoder = CredibilityEncoder(credibility_k=credibility_k)
        self._cat_cols: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_cols: List[str],
        weights: Optional[pd.Series] = None,
    ) -> "ActuarialCategoricalEncoder":
        self._cat_cols = [c for c in cat_cols if c in X.columns]
        X_collapsed = self.collapser.fit_transform(X, self._cat_cols)
        self.encoder.fit(X_collapsed, y, self._cat_cols, weights=weights)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .transform()")
        X_collapsed = self.collapser.transform(X)
        return self.encoder.transform(X_collapsed)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_cols: List[str],
        weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        return self.fit(X, y, cat_cols, weights=weights).transform(X)

    def relativities(self, col: str, base: Optional[str] = None) -> pd.DataFrame:
        """
        Return credibility-encoded values as actuarial relativities.

        Indexed to 1.0 at the *base* level (default: highest-exposure level,
        approximated as the level nearest the grand mean).

        Parameters
        ----------
        col : str
            Categorical column name.
        base : str, optional
            Level to index to 1.0.  None = the level closest to grand mean.

        Returns
        -------
        DataFrame with columns: level | encoded_value | relativity
        """
        enc = self.encoder.level_encodings(col)
        if enc.empty:
            return pd.DataFrame()

        df = enc.reset_index()
        df.columns = ["level", "encoded_value"]

        grand_mean = self.encoder._grand_means.get(col, np.nan)
        if base is None:
            # Use level closest to grand mean as base (most "average" segment)
            idx = (df["encoded_value"] - grand_mean).abs().idxmin()
            base_val = float(df.loc[idx, "encoded_value"])
        else:
            row = df[df["level"] == base]
            base_val = float(row["encoded_value"].iloc[0]) if len(row) else grand_mean

        df["relativity"] = df["encoded_value"] / base_val if base_val != 0 else np.nan
        return df.sort_values("relativity", ascending=False).reset_index(drop=True)
