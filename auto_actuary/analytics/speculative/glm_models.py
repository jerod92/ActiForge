"""
auto_actuary.analytics.speculative.glm_models
==============================================
GLM-based frequency and severity models for speculative scenario analysis.

These models decompose pure premium into:

    Pure Premium = Frequency × Severity
    Frequency  = Claims per Exposure Unit   (Poisson GLM, log link)
    Severity   = Average Loss per Claim     (Gamma GLM, log link)

Why GLMs?
---------
- Log link naturally models multiplicative relativities (standard in insurance)
- Poisson family handles count data properly (frequency per unit exposure)
- Gamma family handles right-skewed, positive continuous data (loss severity)
- L2 regularization (Ridge) prevents overfitting on sparse categorical segments
- Coefficients can be back-transformed into actuarial relativities

Categorical handling
--------------------
All categorical features are processed through ActuarialCategoricalEncoder:
- Sparse levels (< min_obs) collapsed into "_Other"
- Remaining levels credibility-encoded to single floats
- No dimensionality explosion from one-hot on 50+ territory codes

Uncertainty quantification
--------------------------
Bootstrap resampling (n_boot iterations) gives prediction intervals for any
scenario.  This directly answers the executive question: "how confident are
we in this projection?"

References
----------
- Nelder & Wedderburn (1972) "Generalized Linear Models"
- Mildenhall (1999) "A Systematic Relationship Between Minimum Bias and GLM"
  Proceedings of the CAS
- sklearn docs: PoissonRegressor, GammaRegressor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from auto_actuary.analytics.speculative.categorical import ActuarialCategoricalEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class GLMResult:
    """Summary of a fitted GLM."""
    model_type: str          # "frequency" | "severity" | "compound"
    n_obs: int
    features_used: List[str]
    cat_cols: List[str]
    cont_cols: List[str]
    alpha: float             # L2 regularization strength
    train_deviance: float    # null deviance - residual deviance (higher is better)
    mean_abs_error: float
    feature_importances: pd.Series  # |coef| ranking (after scaling)

    def __repr__(self) -> str:
        return (
            f"GLMResult(type={self.model_type}, n={self.n_obs:,}, "
            f"deviance={self.train_deviance:.4f}, mae={self.mean_abs_error:.4f})"
        )


@dataclass
class PredictionInterval:
    """Point estimate and bootstrap confidence interval for a prediction."""
    point: float
    lower: float
    upper: float
    ci_level: float  # e.g., 0.90

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def __repr__(self) -> str:
        return (
            f"PI(point={self.point:.4f}, "
            f"{self.ci_level:.0%} CI=[{self.lower:.4f}, {self.upper:.4f}])"
        )


# ---------------------------------------------------------------------------
# Frequency GLM  (Poisson, log link)
# ---------------------------------------------------------------------------

class FrequencyGLM:
    """
    Poisson GLM for claim frequency modeling.

    Models log(E[frequency]) = Xβ where frequency = claims / exposure.
    Uses sklearn's PoissonRegressor with L2 (Ridge) regularization.
    Categorical features are credibility-encoded before model fitting.

    Parameters
    ----------
    alpha : float
        L2 regularization strength.  Larger = more shrinkage.  0.01 is a
        reasonable default for credibility-encoded features; increase if the
        model overfits sparse territories.
    min_category_obs : int
        Minimum observations to keep a category level distinct.
    credibility_k : float
        Full-credibility threshold for target encoding.
    max_iter : int
        Solver iteration limit.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        min_category_obs: int = 30,
        credibility_k: float = 1082.0,
        max_iter: int = 300,
    ) -> None:
        self.alpha = alpha
        self.min_category_obs = min_category_obs
        self.credibility_k = credibility_k
        self.max_iter = max_iter

        self._cat_encoder: Optional[ActuarialCategoricalEncoder] = None
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[PoissonRegressor] = None
        self._cat_cols: List[str] = []
        self._cont_cols: List[str] = []
        self._result: Optional[GLMResult] = None
        self._feature_names: List[str] = []
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        claim_counts: pd.Series,
        exposure: pd.Series,
        cat_cols: Optional[List[str]] = None,
        cont_cols: Optional[List[str]] = None,
    ) -> "FrequencyGLM":
        """
        Fit the frequency GLM.

        Parameters
        ----------
        X : DataFrame
            Feature data (policies/claims grain).
        claim_counts : Series
            Number of claims per row (integer counts).
        exposure : Series
            Earned exposure per row (car-years, house-years, etc.).
        cat_cols : list of str
            Categorical feature columns to credibility-encode.
        cont_cols : list of str
            Continuous feature columns (scaled before fitting).
        """
        self._cat_cols = [c for c in (cat_cols or []) if c in X.columns]
        self._cont_cols = [c for c in (cont_cols or []) if c in X.columns]

        # Target: frequency (claims per unit exposure)
        exp_safe = exposure.clip(lower=1e-6)
        y_freq = claim_counts / exp_safe

        # Only fit on rows with positive exposure
        mask = (exp_safe > 0) & y_freq.notna()
        X_fit = X[mask].copy()
        y_fit = y_freq[mask]
        w_fit = exp_safe[mask]  # exposure as sample weights

        # Categorical encoding
        self._cat_encoder = ActuarialCategoricalEncoder(
            min_obs=self.min_category_obs,
            credibility_k=self.credibility_k,
        )
        X_encoded = self._cat_encoder.fit_transform(
            X_fit, y_fit, self._cat_cols, weights=w_fit
        )

        # Continuous scaling
        self._scaler = StandardScaler()
        X_matrix = self._build_matrix(X_encoded)
        X_scaled = X_matrix.copy()
        if len(self._cont_cols) > 0:
            cont_idx = [list(X_matrix.columns).index(c) for c in self._cont_cols if c in X_matrix.columns]
            if cont_idx:
                X_scaled_arr = X_scaled.values.copy().astype(float)
                X_scaled_arr[:, cont_idx] = self._scaler.fit_transform(
                    X_scaled_arr[:, cont_idx]
                )
                X_scaled = pd.DataFrame(X_scaled_arr, columns=X_scaled.columns)
        else:
            self._scaler.fit(np.zeros((1, 1)))  # dummy fit

        self._feature_names = list(X_scaled.columns)

        # Fit Poisson GLM
        self._model = PoissonRegressor(
            alpha=self.alpha,
            max_iter=self.max_iter,
            fit_intercept=True,
        )
        self._model.fit(X_scaled.values, y_fit.values, sample_weight=w_fit.values)

        # Diagnostics
        y_pred = self._model.predict(X_scaled.values)
        mae = float(np.average(np.abs(y_fit.values - y_pred), weights=w_fit.values))

        # Null model deviance (intercept only)
        grand_mean = float(np.average(y_fit.values, weights=w_fit.values))
        null_dev = self._poisson_deviance(y_fit.values, np.full_like(y_fit.values, grand_mean), w_fit.values)
        res_dev = self._poisson_deviance(y_fit.values, y_pred, w_fit.values)
        deviance_explained = float(null_dev - res_dev)

        # Feature importances (absolute coefficients, first is intercept)
        coefs = pd.Series(
            np.abs(self._model.coef_),
            index=self._feature_names,
        ).sort_values(ascending=False)

        self._result = GLMResult(
            model_type="frequency",
            n_obs=int(mask.sum()),
            features_used=self._feature_names,
            cat_cols=self._cat_cols,
            cont_cols=self._cont_cols,
            alpha=self.alpha,
            train_deviance=deviance_explained,
            mean_abs_error=mae,
            feature_importances=coefs,
        )

        self.is_fitted = True
        logger.info("FrequencyGLM fitted: n=%d, deviance_explained=%.4f, mae=%.5f",
                    self._result.n_obs, deviance_explained, mae)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict frequency (claims per exposure unit)."""
        self._check_fitted()
        X_enc = self._cat_encoder.transform(X)
        X_mat = self._build_matrix(X_enc, fit=False)
        return pd.Series(self._model.predict(X_mat.values), index=X.index, name="predicted_frequency")

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        exposure_train: pd.Series,
        n_boot: int = 200,
        ci: float = 0.90,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Bootstrap prediction intervals for frequency estimates.

        Returns DataFrame with columns: point | lower | upper | ci_level.
        One row per observation in X.
        """
        self._check_fitted()
        rng = np.random.default_rng(random_state)

        point = self.predict(X).values
        boot_preds = np.zeros((n_boot, len(X)))

        for b in range(n_boot):
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            X_b = X_train.iloc[idx].copy()
            y_b = y_train.iloc[idx]
            exp_b = exposure_train.iloc[idx]

            boot_model = FrequencyGLM(
                alpha=self.alpha,
                min_category_obs=self.min_category_obs,
                credibility_k=self.credibility_k,
                max_iter=self.max_iter,
            )
            try:
                boot_model.fit(X_b, y_b, exp_b, self._cat_cols, self._cont_cols)
                boot_preds[b] = boot_model.predict(X).values
            except Exception:
                boot_preds[b] = point  # fall back to point estimate

        alpha_tail = (1 - ci) / 2
        lower = np.quantile(boot_preds, alpha_tail, axis=0)
        upper = np.quantile(boot_preds, 1 - alpha_tail, axis=0)

        return pd.DataFrame(
            {"point": point, "lower": lower, "upper": upper, "ci_level": ci},
            index=X.index,
        )

    # ------------------------------------------------------------------
    # Actuarial outputs
    # ------------------------------------------------------------------

    def relativities(self, col: str) -> pd.DataFrame:
        """Return credibility-encoded relativities for a categorical column."""
        self._check_fitted()
        if col not in self._cat_cols:
            raise ValueError(f"'{col}' is not a categorical column in this model")
        return self._cat_encoder.relativities(col)

    @property
    def result(self) -> Optional[GLMResult]:
        return self._result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_matrix(self, X_enc: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Assemble encoded features into a float matrix."""
        cols = self._cat_cols + self._cont_cols
        available = [c for c in cols if c in X_enc.columns]
        mat = X_enc[available].copy().astype(float)
        if mat.empty:
            mat = pd.DataFrame(np.ones((len(X_enc), 1)), columns=["intercept_only"], index=X_enc.index)
        return mat

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    @staticmethod
    def _poisson_deviance(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
        """Weighted Poisson deviance."""
        eps = 1e-10
        mu = np.clip(mu, eps, None)
        y = np.clip(y, eps, None)
        return float(2 * np.sum(w * (y * np.log(y / mu) - (y - mu))))


# ---------------------------------------------------------------------------
# Severity GLM  (Gamma, log link)
# ---------------------------------------------------------------------------

class SeverityGLM:
    """
    Gamma GLM for claim severity modeling.

    Models log(E[severity]) = Xβ where severity = loss / claim_count.
    Gamma family is appropriate for right-skewed, strictly positive losses.
    Uses L2 regularization to handle sparse category segments.

    Parameters
    ----------
    alpha : float
        L2 regularization strength.
    min_category_obs : int
        Minimum claims per category level (not policies — need actual losses).
    credibility_k : float
        Full-credibility threshold (default 1082 classical standard).
    """

    def __init__(
        self,
        alpha: float = 0.01,
        min_category_obs: int = 30,
        credibility_k: float = 1082.0,
        max_iter: int = 300,
    ) -> None:
        self.alpha = alpha
        self.min_category_obs = min_category_obs
        self.credibility_k = credibility_k
        self.max_iter = max_iter

        self._cat_encoder: Optional[ActuarialCategoricalEncoder] = None
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[GammaRegressor] = None
        self._cat_cols: List[str] = []
        self._cont_cols: List[str] = []
        self._result: Optional[GLMResult] = None
        self._feature_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        severity: pd.Series,
        claim_counts: Optional[pd.Series] = None,
        cat_cols: Optional[List[str]] = None,
        cont_cols: Optional[List[str]] = None,
    ) -> "SeverityGLM":
        """
        Fit severity GLM.

        Parameters
        ----------
        X : DataFrame
            Feature data at claim (or accident-year/segment) grain.
        severity : Series
            Average severity per row (must be strictly positive).
        claim_counts : Series, optional
            Claim counts used as sample weights for the GLM.
        cat_cols, cont_cols : list of str
            Categorical and continuous feature columns.
        """
        self._cat_cols = [c for c in (cat_cols or []) if c in X.columns]
        self._cont_cols = [c for c in (cont_cols or []) if c in X.columns]

        mask = severity.notna() & (severity > 0)
        X_fit = X[mask].copy()
        y_fit = severity[mask]
        w_fit = claim_counts[mask] if claim_counts is not None else pd.Series(
            np.ones(mask.sum()), index=X_fit.index
        )

        self._cat_encoder = ActuarialCategoricalEncoder(
            min_obs=self.min_category_obs,
            credibility_k=self.credibility_k,
        )
        X_encoded = self._cat_encoder.fit_transform(
            X_fit, y_fit, self._cat_cols, weights=w_fit
        )

        self._scaler = StandardScaler()
        X_matrix = self._build_matrix(X_encoded)
        X_scaled = X_matrix.copy().astype(float)

        self._feature_names = list(X_scaled.columns)

        self._model = GammaRegressor(
            alpha=self.alpha,
            max_iter=self.max_iter,
            fit_intercept=True,
        )
        self._model.fit(X_scaled.values, y_fit.values, sample_weight=w_fit.values)

        y_pred = self._model.predict(X_scaled.values)
        mae = float(np.average(np.abs(y_fit.values - y_pred), weights=w_fit.values))

        grand_mean = float(np.average(y_fit.values, weights=w_fit.values))
        null_dev = self._gamma_deviance(y_fit.values, np.full_like(y_fit.values, grand_mean), w_fit.values)
        res_dev = self._gamma_deviance(y_fit.values, y_pred, w_fit.values)
        deviance_explained = float(null_dev - res_dev)

        coefs = pd.Series(
            np.abs(self._model.coef_),
            index=self._feature_names,
        ).sort_values(ascending=False)

        self._result = GLMResult(
            model_type="severity",
            n_obs=int(mask.sum()),
            features_used=self._feature_names,
            cat_cols=self._cat_cols,
            cont_cols=self._cont_cols,
            alpha=self.alpha,
            train_deviance=deviance_explained,
            mean_abs_error=mae,
            feature_importances=coefs,
        )

        self.is_fitted = True
        logger.info("SeverityGLM fitted: n=%d, deviance_explained=%.4f, mae=%.2f",
                    self._result.n_obs, deviance_explained, mae)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict severity (average loss per claim)."""
        self._check_fitted()
        X_enc = self._cat_encoder.transform(X)
        X_mat = self._build_matrix(X_enc).astype(float)
        return pd.Series(self._model.predict(X_mat.values), index=X.index, name="predicted_severity")

    def relativities(self, col: str) -> pd.DataFrame:
        self._check_fitted()
        if col not in self._cat_cols:
            raise ValueError(f"'{col}' is not a categorical column in this model")
        return self._cat_encoder.relativities(col)

    @property
    def result(self) -> Optional[GLMResult]:
        return self._result

    def _build_matrix(self, X_enc: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        cols = self._cat_cols + self._cont_cols
        available = [c for c in cols if c in X_enc.columns]
        mat = X_enc[available].copy().astype(float)
        if mat.empty:
            mat = pd.DataFrame(np.ones((len(X_enc), 1)), columns=["intercept_only"], index=X_enc.index)
        return mat

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    @staticmethod
    def _gamma_deviance(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
        """Weighted Gamma deviance."""
        eps = 1e-10
        mu = np.clip(mu, eps, None)
        y = np.clip(y, eps, None)
        return float(2 * np.sum(w * (-np.log(y / mu) + (y - mu) / mu)))


# ---------------------------------------------------------------------------
# Compound GLM  (Frequency × Severity = Pure Premium)
# ---------------------------------------------------------------------------

class CompoundGLM:
    """
    Combined frequency × severity model for pure premium prediction.

    Pure Premium = Frequency × Severity

    Both components are modelled independently (separate GLMs) and combined
    for predictions.  This decomposition lets executives understand *which*
    driver is causing loss cost changes — frequency trends vs. severity inflation.

    Parameters
    ----------
    freq_alpha, sev_alpha : float
        L2 regularization for the respective sub-models.
    min_category_obs : int
        Sparse category threshold (applied to both models).
    credibility_k : float
        Credibility threshold for target encoding.
    """

    def __init__(
        self,
        freq_alpha: float = 0.01,
        sev_alpha: float = 0.01,
        min_category_obs: int = 30,
        credibility_k: float = 1082.0,
        max_iter: int = 300,
    ) -> None:
        self.freq_model = FrequencyGLM(
            alpha=freq_alpha,
            min_category_obs=min_category_obs,
            credibility_k=credibility_k,
            max_iter=max_iter,
        )
        self.sev_model = SeverityGLM(
            alpha=sev_alpha,
            min_category_obs=min_category_obs,
            credibility_k=credibility_k,
            max_iter=max_iter,
        )
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        claim_counts: pd.Series,
        losses: pd.Series,
        exposure: pd.Series,
        cat_cols: Optional[List[str]] = None,
        cont_cols: Optional[List[str]] = None,
    ) -> "CompoundGLM":
        """
        Fit both frequency and severity GLMs.

        Parameters
        ----------
        X : DataFrame
            Feature data.
        claim_counts : Series
            Number of claims per row.
        losses : Series
            Total incurred losses per row.
        exposure : Series
            Earned exposure per row.
        """
        cat_cols = cat_cols or []
        cont_cols = cont_cols or []

        # Frequency model (all rows with positive exposure)
        self.freq_model.fit(X, claim_counts, exposure, cat_cols, cont_cols)

        # Severity model (only rows with claims)
        claim_mask = claim_counts > 0
        severity = losses / claim_counts.replace(0, np.nan)
        if claim_mask.sum() < 10:
            logger.warning("CompoundGLM: very few claims (%d) for severity model", claim_mask.sum())

        self.sev_model.fit(
            X[claim_mask],
            severity[claim_mask],
            claim_counts[claim_mask],
            cat_cols,
            cont_cols,
        )

        self.is_fitted = True
        return self

    def predict_pure_premium(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict pure premium, frequency, and severity for each row.

        Returns
        -------
        DataFrame with columns: frequency | severity | pure_premium
        """
        self._check_fitted()
        freq = self.freq_model.predict(X)
        sev = self.sev_model.predict(X)
        pp = freq * sev
        return pd.DataFrame(
            {"frequency": freq.values, "severity": sev.values, "pure_premium": pp.values},
            index=X.index,
        )

    def predict_portfolio(
        self,
        X: pd.DataFrame,
        exposure: pd.Series,
        premium_per_unit: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Aggregate GLM predictions to portfolio KPIs.

        Returns
        -------
        dict with keys:
            total_exposure | predicted_frequency | predicted_severity |
            predicted_pure_premium | predicted_losses |
            predicted_loss_ratio (if premium_per_unit provided)
        """
        self._check_fitted()
        preds = self.predict_pure_premium(X)
        preds["exposure"] = exposure.values

        total_exp = float(exposure.sum())
        pred_losses = float((preds["pure_premium"] * preds["exposure"]).sum())
        pred_freq = float((preds["frequency"] * preds["exposure"]).sum() / max(total_exp, 1e-6))
        pred_sev = float((preds["severity"] * preds["pure_premium"] * preds["exposure"]).sum()
                         / max(pred_losses, 1e-6))

        result = {
            "total_exposure": total_exp,
            "predicted_frequency": pred_freq,
            "predicted_severity": pred_sev,
            "predicted_pure_premium": pred_losses / max(total_exp, 1e-6),
            "predicted_losses": pred_losses,
        }

        if premium_per_unit is not None:
            total_premium = float((premium_per_unit * exposure).sum())
            result["total_premium"] = total_premium
            result["predicted_loss_ratio"] = pred_losses / max(total_premium, 1e-6)

        return result

    def bootstrap_portfolio(
        self,
        X: pd.DataFrame,
        exposure: pd.Series,
        premium_per_unit: Optional[pd.Series] = None,
        n_boot: int = 300,
        ci: float = 0.90,
        random_state: int = 42,
    ) -> Dict[str, PredictionInterval]:
        """
        Bootstrap confidence intervals on portfolio-level KPIs.

        Resamples rows of (X, exposure, premium) and re-predicts, giving a
        distribution of portfolio outcomes.  Captures model uncertainty from
        the GLM coefficients — *not* process variance (use simulation for that).

        Returns
        -------
        dict[KPI name] -> PredictionInterval
        """
        self._check_fitted()
        rng = np.random.default_rng(random_state)
        n = len(X)

        point = self.predict_portfolio(X, exposure, premium_per_unit)
        keys = list(point.keys())
        boot_results = {k: [] for k in keys}

        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            X_b = X.iloc[idx]
            exp_b = exposure.iloc[idx]
            prem_b = premium_per_unit.iloc[idx] if premium_per_unit is not None else None

            try:
                kpis = self.predict_portfolio(X_b, exp_b, prem_b)
                for k in keys:
                    boot_results[k].append(kpis.get(k, point[k]))
            except Exception:
                for k in keys:
                    boot_results[k].append(point[k])

        alpha_tail = (1 - ci) / 2
        intervals = {}
        for k in keys:
            arr = np.array(boot_results[k])
            intervals[k] = PredictionInterval(
                point=point[k],
                lower=float(np.quantile(arr, alpha_tail)),
                upper=float(np.quantile(arr, 1 - alpha_tail)),
                ci_level=ci,
            )

        return intervals

    def relativities_table(self, cat_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Return a combined frequency × severity relativity table.

        For each categorical column and level, shows:
        freq_rel | sev_rel | pp_rel  (where pp_rel = freq_rel * sev_rel)
        """
        self._check_fitted()
        cols = cat_cols or self.freq_model._cat_cols
        rows = []
        for col in cols:
            try:
                freq_rel = self.freq_model.relativities(col).set_index("level")["relativity"]
                sev_rel = self.sev_model.relativities(col).set_index("level")["relativity"]
                combined = pd.DataFrame({"freq_rel": freq_rel, "sev_rel": sev_rel}).dropna()
                combined["pp_rel"] = combined["freq_rel"] * combined["sev_rel"]
                combined["variable"] = col
                combined = combined.reset_index().rename(columns={"index": "level"})
                rows.append(combined)
            except Exception as e:
                logger.debug("relativities_table: skipping %s — %s", col, e)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "CompoundGLM(not fitted)"
        fr = self.freq_model.result
        sr = self.sev_model.result
        return (
            f"CompoundGLM(freq_n={fr.n_obs if fr else 0}, sev_n={sr.n_obs if sr else 0}, "
            f"freq_mae={fr.mean_abs_error:.5f if fr else 0:.5f}, "
            f"sev_mae={sr.mean_abs_error:.2f if sr else 0:.2f})"
        )


# ---------------------------------------------------------------------------
# Factory: fit from aggregated session data
# ---------------------------------------------------------------------------

def fit_compound_glm_from_segments(
    segment_df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    cont_cols: Optional[List[str]] = None,
    freq_alpha: float = 0.05,
    sev_alpha: float = 0.05,
    credibility_k: float = 1082.0,
    min_category_obs: int = 20,
) -> CompoundGLM:
    """
    Fit a CompoundGLM from a pre-aggregated segment DataFrame.

    Expected columns
    ----------------
    accident_year, territory, class_code, coverage_code (categoricals)
    earned_exposure, claim_count, incurred_loss (numerics)
    Any additional columns in cat_cols / cont_cols.

    Parameters
    ----------
    segment_df : DataFrame
        Aggregated loss data at (year × territory × class) grain.
    cat_cols : list of str, optional
        Categorical feature columns.  Defaults to common actuarial dims.
    cont_cols : list of str, optional
        Continuous feature columns.  Defaults to accident_year (numeric).

    Returns
    -------
    Fitted CompoundGLM
    """
    df = segment_df.copy()

    # Defaults
    default_cats = ["territory", "class_code", "coverage_code", "lob_code"]
    default_conts = ["accident_year"]
    cat_cols = cat_cols if cat_cols is not None else [c for c in default_cats if c in df.columns]
    cont_cols = cont_cols if cont_cols is not None else [c for c in default_conts if c in df.columns]

    required = {"earned_exposure", "claim_count", "incurred_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"segment_df missing required columns: {missing}")

    glm = CompoundGLM(
        freq_alpha=freq_alpha,
        sev_alpha=sev_alpha,
        min_category_obs=min_category_obs,
        credibility_k=credibility_k,
    )
    glm.fit(
        X=df[cat_cols + cont_cols],
        claim_counts=df["claim_count"],
        losses=df["incurred_loss"],
        exposure=df["earned_exposure"],
        cat_cols=cat_cols,
        cont_cols=cont_cols,
    )
    return glm
