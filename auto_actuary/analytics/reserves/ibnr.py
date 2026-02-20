"""
auto_actuary.analytics.reserves.ibnr
======================================
IBNR (Incurred But Not Reported) reserve estimation.

Implements three primary FCAS-level reserve methods, each with a different
philosophy about how much to trust the data vs. prior expectations:

  Chain Ladder (Development Method)
  ----------------------------------
  Fully data-driven. Projects each accident year's reported losses to ultimate
  using the selected age-to-age development factors.

      Ultimate(AY) = Latest Reported(AY) × CDF-to-Ultimate(latest age)

  Bornhuetter-Ferguson (B-F)
  ---------------------------
  A credibility-weighted blend of Chain Ladder and the Expected Loss method.
  Particularly useful for immature accident years where limited data exists.

      IBNR(AY)    = ELR × Premium(AY) × (1 − 1/CDF)
      Ultimate(AY) = Latest Reported(AY) + IBNR(AY)

  Cape Cod
  ---------
  Like B-F, but derives the ELR from the data itself rather than requiring
  an a-priori assumption.

      ELR_cape_cod = Σ Reported / Σ (Premium × Used-Up %)
      Used-Up %(AY) = 1 − 1/CDF(latest age)
      Then proceeds as B-F.

  Benktander (Iterated B-F)
  -------------------------
  One iteration of B-F using the B-F ultimate as the new "expected" — gives
  greater weight to actual reported losses; converges to chain ladder.

      IBNR_BK(AY)    = BF_Ultimate(AY) × (1 − 1/CDF) − Reported × (1 − 1/CDF)²  (approx)
      Actually: IBNR_BK(AY) = ELR_BF × Premium × (1−1/CDF)²

References
----------
- Bornhuetter & Ferguson (1972), PCAS — "The Actuary and IBNR"
- Mack, T. (2000), ASTIN — "Credible Claims Reserves: The Benktander Method"
- Friedland (2010), CAS — "Estimating Unpaid Claims Using Basic Techniques"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.analytics.triangles.development import LossTriangle
    from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)

_ALL_METHODS = ["chain_ladder", "bornhuetter_ferguson", "cape_cod", "benktander"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ReserveResult:
    """Reserve estimate for a single method."""
    method: str
    ultimates: pd.Series          # by origin year
    ibnr: pd.Series               # by origin year
    elr: Optional[float] = None   # Expected Loss Ratio used (B-F / Cape Cod)
    notes: str = ""

    @property
    def total_ultimate(self) -> float:
        return float(self.ultimates.sum())

    @property
    def total_ibnr(self) -> float:
        return float(self.ibnr.sum())


# ---------------------------------------------------------------------------
# ReserveAnalysis
# ---------------------------------------------------------------------------

class ReserveAnalysis:
    """
    Runs multiple IBNR reserve methods on a developed LossTriangle.

    Parameters
    ----------
    triangle : LossTriangle
        Must have .develop() already called (CDFs available).
    config : ActuaryConfig
    methods : list[str], optional
        Subset of ['chain_ladder', 'bornhuetter_ferguson', 'cape_cod', 'benktander'].
        Defaults to all four.
    premium : pd.Series, optional
        Earned premium by origin year (required for B-F, Cape Cod).
        Index must match triangle.origins.
    elr_override : float or pd.Series, optional
        Override the ELR for B-F.  Scalar applies to all years; Series by year.
    """

    def __init__(
        self,
        triangle: "LossTriangle",
        config: "ActuaryConfig",
        methods: Optional[List[str]] = None,
        premium: Optional[pd.Series] = None,
        elr_override: Optional[float | pd.Series] = None,
    ) -> None:
        self.triangle = triangle
        self.config = config
        self.methods = methods or _ALL_METHODS
        self.premium = premium
        self.elr_override = elr_override

        self._results: Dict[str, ReserveResult] = {}
        self._run()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _run(self) -> None:
        tri = self.triangle
        if tri._cdfs is None:
            raise RuntimeError("Triangle CDFs not computed.  Call triangle.develop() first.")

        diag = tri.latest_diagonal
        origins = tri.origins

        # CDF-to-ultimate for each origin's latest age
        cdfs = pd.Series(
            {
                o: tri._cdfs.get(tri._latest_age.get(o, tri.ages[0]), tri._tail_factor)
                for o in origins
            }
        )

        # % unreported = 1 - 1/CDF
        pct_unreported = 1.0 - 1.0 / cdfs

        # ---- Chain Ladder ----
        if "chain_ladder" in self.methods:
            cl_ult = diag * cdfs
            cl_ibnr = cl_ult - diag
            self._results["chain_ladder"] = ReserveResult(
                method="chain_ladder",
                ultimates=cl_ult.rename("chain_ladder"),
                ibnr=cl_ibnr.rename("chain_ladder_ibnr"),
            )

        # ---- Cape Cod (derive ELR from data) ----
        elr_cc = None
        if "cape_cod" in self.methods or "bornhuetter_ferguson" in self.methods or "benktander" in self.methods:
            if self.premium is not None:
                prem_aligned = self.premium.reindex(origins)
                used_up_prem = prem_aligned * (1.0 - 1.0 / cdfs)
                used_up_total = used_up_prem.sum()
                if used_up_total > 0:
                    elr_cc = float(diag.sum() / used_up_total)
                else:
                    logger.warning("Used-up premium is zero; Cape Cod ELR undefined.")
                    elr_cc = 0.65  # fallback

            if "cape_cod" in self.methods:
                if self.premium is not None:
                    cc_ibnr = prem_aligned * pct_unreported * elr_cc
                    cc_ult = diag + cc_ibnr
                    self._results["cape_cod"] = ReserveResult(
                        method="cape_cod",
                        ultimates=cc_ult.rename("cape_cod"),
                        ibnr=cc_ibnr.rename("cape_cod_ibnr"),
                        elr=elr_cc,
                        notes=f"ELR derived from data: {elr_cc:.4f}",
                    )
                else:
                    logger.warning("Cape Cod skipped — no premium data provided.")

        # ---- Bornhuetter-Ferguson ----
        if "bornhuetter_ferguson" in self.methods:
            if self.premium is not None:
                # Determine ELR
                if self.elr_override is not None:
                    elr_bf_series = (
                        pd.Series(self.elr_override, index=origins)
                        if isinstance(self.elr_override, (int, float))
                        else self.elr_override.reindex(origins)
                    )
                    elr_label = "user_specified"
                elif elr_cc is not None:
                    # Use Cape Cod ELR as the a-priori for B-F
                    elr_bf_series = pd.Series(elr_cc, index=origins)
                    elr_label = "cape_cod"
                else:
                    elr_bf_series = pd.Series(0.65, index=origins)
                    elr_label = "fallback_0.65"
                    logger.warning("B-F ELR defaulted to 0.65 — provide premium or elr_override.")

                bf_ibnr = prem_aligned * elr_bf_series * pct_unreported
                bf_ult = diag + bf_ibnr
                elr_scalar = float(elr_bf_series.mean())
                self._results["bornhuetter_ferguson"] = ReserveResult(
                    method="bornhuetter_ferguson",
                    ultimates=bf_ult.rename("bf"),
                    ibnr=bf_ibnr.rename("bf_ibnr"),
                    elr=elr_scalar,
                    notes=f"ELR source: {elr_label}, avg={elr_scalar:.4f}",
                )

                # ---- Benktander ----
                if "benktander" in self.methods:
                    # BK: uses B-F ultimate as the "expected" for a second iteration
                    # IBNR_BK = BF_Ultimate × (1 - 1/CDF) - Reported × (1 - 1/CDF)²... approx:
                    # More precisely: BK_Ultimate = Reported + (BF_Ultimate/Prem) × Prem × pct_unrpt
                    bf_ult_aligned = bf_ult.reindex(origins)
                    implied_elr = bf_ult_aligned / prem_aligned.replace(0, np.nan)
                    bk_ibnr = prem_aligned * implied_elr * pct_unreported
                    bk_ult = diag + bk_ibnr
                    self._results["benktander"] = ReserveResult(
                        method="benktander",
                        ultimates=bk_ult.rename("benktander"),
                        ibnr=bk_ibnr.rename("benktander_ibnr"),
                        elr=float(implied_elr.mean()),
                        notes="Iterated from B-F ultimate",
                    )
            else:
                logger.warning("B-F and Benktander skipped — no premium data.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def result(self, method: str) -> ReserveResult:
        if method not in self._results:
            raise KeyError(f"Method '{method}' not available. Run methods: {list(self._results.keys())}")
        return self._results[method]

    @property
    def available_methods(self) -> List[str]:
        return list(self._results.keys())

    def comparison_table(self) -> pd.DataFrame:
        """
        Return a side-by-side comparison of ultimates and IBNR by method.

        Columns: origin | reported | cdf_to_ult | <method>_ult | <method>_ibnr | …
        """
        tri = self.triangle
        diag = tri.latest_diagonal
        cdfs = pd.Series(
            {
                o: tri._cdfs.get(tri._latest_age.get(o, tri.ages[0]), tri._tail_factor)
                for o in tri.origins
            }
        )

        df = pd.DataFrame({"reported": diag, "cdf_to_ult": cdfs})

        for method, res in self._results.items():
            df[f"{method}_ult"] = res.ultimates
            df[f"{method}_ibnr"] = res.ibnr

        # Totals row
        totals = df.sum(numeric_only=True)
        totals["cdf_to_ult"] = np.nan  # meaningless to sum
        df.loc["TOTAL"] = totals

        return df

    def selected(self, method: Optional[str] = None) -> ReserveResult:
        """Return the 'selected' reserve estimate (uses config default or explicit method)."""
        if method is None:
            method = self.config.assumption("reserves", "primary_method", default="bornhuetter_ferguson")
        if method not in self._results:
            # Fall back to chain ladder which always runs
            method = "chain_ladder"
        return self._results[method]

    def total_ibnr(self, method: Optional[str] = None) -> float:
        return self.selected(method).total_ibnr

    def total_ultimate(self, method: Optional[str] = None) -> float:
        return self.selected(method).total_ultimate

    def __repr__(self) -> str:
        return (
            f"ReserveAnalysis(lob={self.triangle.lob!r}, "
            f"methods={self.available_methods}, "
            f"selected_ibnr={self.total_ibnr():,.0f})"
        )
