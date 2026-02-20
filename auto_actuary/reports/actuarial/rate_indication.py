"""
auto_actuary.reports.actuarial.rate_indication
================================================
Rate indication exhibit — the official output of the ratemaking analysis.

Shows the full chain of logic:
  Projected Loss Ratio / Permissible Loss Ratio − 1 = Indicated Change

Exhibit format mirrors standard FCAS ratemaking exhibits (Werner/Modlin).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from auto_actuary.reports.renderers.excel import ExcelWriter
from auto_actuary.reports.renderers.html import df_to_html_table, PLOTLY_CDN, _DASHBOARD_CSS

if TYPE_CHECKING:
    from auto_actuary.analytics.ratemaking.indicated_rate import RateIndication, RateIndicationResult
    from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)


class RateIndicationExhibit:
    """
    Generate a rate indication exhibit.

    Parameters
    ----------
    indication : RateIndication  (the computation object, not the result)
    config : ActuaryConfig
    """

    def __init__(
        self,
        indication: "RateIndication",
        config: "ActuaryConfig",
    ) -> None:
        self.indication = indication
        self.cfg = config
        self._result: Optional["RateIndicationResult"] = None

    @property
    def result(self) -> "RateIndicationResult":
        if self._result is None:
            self._result = self.indication.compute()
        return self._result

    def render(self, output_path: Union[str, Path], fmt: str = "excel") -> Path:
        output_path = Path(output_path)
        if fmt == "excel":
            return self._render_excel(output_path)
        elif fmt == "html":
            return self._render_html(output_path)
        raise ValueError(f"Unknown format: {fmt!r}")

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------

    def _render_excel(self, path: Path) -> Path:
        res = self.result
        lob_label = self.cfg.lob_label(res.lob)
        writer = ExcelWriter(
            title=f"Rate Indication — {lob_label}",
            company=self.cfg.company_name,
        )
        writer.add_cover(as_of_date=pd.Timestamp.now().strftime("%B %d, %Y"))

        # Summary exhibit (key scalar table)
        summary_data = self._summary_df(res)
        writer.add_sheet(
            "Rate Indication",
            summary_data,
            title=f"Rate Indication Summary — {lob_label}",
            subtitle=f"Indicated Change: {res.indicated_pct}  |  Cred-Wtd: {res.credibility_weighted_pct}",
            number_format="#,##0.0",
        )

        # By-year detail
        if res.by_year is not None:
            writer.add_sheet(
                "Accident Year Detail",
                res.by_year,
                title="Accident Year Loss Development Detail",
                number_format="$#,##0",
                pct_cols=["loss_ratio", "trended_loss_ratio"],
                currency_cols=["on_level_premium", "ultimate_loss", "trended_ultimate_loss"],
            )

        return writer.save(path)

    def _summary_df(self, res: "RateIndicationResult") -> pd.DataFrame:
        rows = [
            ("On-Level Earned Premium", f"${res.on_level_premium:,.0f}", "Historical EP restated to current rates"),
            ("Trended Ultimate Loss", f"${res.trended_ultimate_loss:,.0f}", f"Trend factor: {res.trend_factor:.4f}"),
            ("Projected Loss Ratio", f"{res.projected_loss_ratio:.4f}", "Trended Ult / On-Level EP"),
            ("", "", ""),
            ("Variable Expense Ratio (V)", f"{res.variable_expense_ratio:.4f}", "Commissions, taxes, fees"),
            ("Fixed Expense Ratio", f"{res.fixed_expense_ratio:.4f}", "G&A expenses"),
            ("Target Profit Margin (Q)", f"{res.target_profit_margin:.4f}", ""),
            ("Permissible Loss Ratio", f"{res.permissible_loss_ratio:.4f}", "1 − V − Fixed − Q"),
            ("", "", ""),
            ("INDICATED CHANGE", res.indicated_pct, "Proj LR / Permissible LR − 1"),
            ("Credibility (Z)", f"{res.credibility:.4f}", f"Based on {self.indication.claim_count or 'estimated'} claims"),
            ("Complement Indication", f"{res.complement_indication:+.4f}", "Industry / company complement"),
            ("CREDIBILITY-WEIGHTED CHANGE", res.credibility_weighted_pct, "Z × Indicated + (1−Z) × Complement"),
        ]
        return pd.DataFrame(rows, columns=["Item", "Value", "Notes"])

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _render_html(self, path: Path) -> Path:
        res = self.result
        lob_label = self.cfg.lob_label(res.lob)
        company = self.cfg.company_name
        as_of = pd.Timestamp.now().strftime("%B %d, %Y")

        summ_df = self._summary_df(res)
        summ_html = df_to_html_table(summ_df, table_id="summ_tbl")

        by_year_html = ""
        if res.by_year is not None:
            by_year_html = df_to_html_table(
                res.by_year,
                pct_cols=["loss_ratio", "trended_loss_ratio"],
                currency_cols=["on_level_premium", "ultimate_loss", "trended_ultimate_loss"],
                table_id="by_year_tbl",
            )

        # Gauge-style KPI box
        ind_color = "#C00000" if res.indicated_change > 0.03 else ("#E8A000" if res.indicated_change > 0 else "#375623")
        gauge_html = f"""
        <div style="display:flex;gap:20px;margin-bottom:24px">
          <div class="kpi-card" style="border-top-color:{ind_color};min-width:200px">
            <div class="kpi-label">Indicated Change</div>
            <div class="kpi-value" style="color:{ind_color}">{res.indicated_pct}</div>
            <div class="kpi-sub">Proj LR / Permissible LR − 1</div>
          </div>
          <div class="kpi-card" style="min-width:200px">
            <div class="kpi-label">Credibility-Weighted</div>
            <div class="kpi-value">{res.credibility_weighted_pct}</div>
            <div class="kpi-sub">Z = {res.credibility:.2f}</div>
          </div>
          <div class="kpi-card" style="min-width:200px">
            <div class="kpi-label">Projected Loss Ratio</div>
            <div class="kpi-value">{res.projected_loss_ratio:.1%}</div>
            <div class="kpi-sub">vs. Permissible {res.permissible_loss_ratio:.1%}</div>
          </div>
        </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{company} — Rate Indication</title>
{PLOTLY_CDN}
<style>{_DASHBOARD_CSS}</style>
</head>
<body>
<div class="header">
  <div><h1>{company}</h1><div style="font-size:.9rem;opacity:.8;margin-top:4px">Rate Indication Exhibit — {lob_label}</div></div>
  <div class="meta"><div>As of: {as_of}</div><div style="font-size:.75rem;opacity:.7">auto_actuary</div></div>
</div>
<div class="container">
  <div class="section-title">Rate Indication Summary</div>
  {gauge_html}
  <div class="table-card">{summ_html}</div>

  {"<div class='section-title'>Accident Year Detail</div><div class='table-card' style='overflow-x:auto'>" + by_year_html + "</div>" if by_year_html else ""}
</div>
<div class="footer">auto_actuary — {as_of} — For actuarial review</div>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
        return path
