"""
auto_actuary.reports.actuarial.reserve_exhibit
===============================================
Reserve / IBNR exhibit — the core actuarial reserve report.

Shows all reserve methods side-by-side with the actuary's selected indication.
Standard FCAS CAS exam 6 / Schedule P format.

Sheets produced (Excel):
  1. Cover
  2. Reserve Comparison (all methods side-by-side)
  3. Selected Method Detail
  4. Adequacy vs. Held Reserves (if held reserves provided)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from auto_actuary.reports.renderers.excel import ExcelWriter
from auto_actuary.reports.renderers.html import (
    df_to_html_table, reserve_waterfall_chart, PLOTLY_CDN, _DASHBOARD_CSS
)

if TYPE_CHECKING:
    from auto_actuary.analytics.reserves.ibnr import ReserveAnalysis
    from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)


class ReserveExhibit:
    """
    Generate reserve / IBNR exhibit.

    Parameters
    ----------
    analysis : ReserveAnalysis
    config : ActuaryConfig
    held_reserves : pd.Series, optional
        Booked IBNR by accident year.
    """

    def __init__(
        self,
        analysis: "ReserveAnalysis",
        config: "ActuaryConfig",
        held_reserves: Optional[pd.Series] = None,
    ) -> None:
        self.analysis = analysis
        self.cfg = config
        self.held_reserves = held_reserves

    def render(self, output_path: Union[str, Path], fmt: str = "excel") -> Path:
        output_path = Path(output_path)
        if fmt == "excel":
            return self._render_excel(output_path)
        elif fmt == "html":
            return self._render_html(output_path)
        else:
            raise ValueError(f"Unsupported format: {fmt!r}")

    def _render_excel(self, path: Path) -> Path:
        tri = self.analysis.triangle
        lob_label = self.cfg.lob_label(tri.lob)

        writer = ExcelWriter(
            title=f"Reserve / IBNR Exhibit — {lob_label}",
            company=self.cfg.company_name,
        )
        writer.add_cover(as_of_date=pd.Timestamp.now().strftime("%B %d, %Y"))

        # Sheet: Reserve comparison
        comp = self.analysis.comparison_table()
        writer.add_sheet(
            "Reserve Comparison",
            comp,
            title="Reserve Estimate Comparison — All Methods",
            subtitle=f"{lob_label} | Chain Ladder, B-F, Cape Cod, Benktander",
            number_format="$#,##0",
        )

        # Sheet: Selected method
        sel = self.analysis.selected()
        sel_df = pd.DataFrame({
            "ultimate": sel.ultimates,
            "ibnr": sel.ibnr,
        })
        sel_df.index.name = "accident_year"
        # Add total row
        totals = sel_df.sum()
        totals.name = "TOTAL"
        sel_df = pd.concat([sel_df, totals.to_frame().T])
        writer.add_sheet(
            f"Selected ({sel.method.replace('_', ' ').title()})",
            sel_df,
            title=f"Selected Reserve Estimate — {sel.method.replace('_', ' ').title()}",
            subtitle=f"ELR: {sel.elr:.4f}" if sel.elr else "",
            number_format="$#,##0",
        )

        # Sheet: Adequacy (if held)
        if self.held_reserves is not None:
            from auto_actuary.analytics.reserves.adequacy import ReserveAdequacy
            adeq = ReserveAdequacy(self.analysis, held_reserves=self.held_reserves)
            adeq_tbl = adeq.adequacy_table()
            writer.add_sheet(
                "Reserve Adequacy",
                adeq_tbl,
                title="Reserve Adequacy — Held vs. Actuarial Estimate",
                number_format="$#,##0",
                pct_cols=["redundancy_pct", "pct_developed"],
            )

        return writer.save(path)

    def _render_html(self, path: Path) -> Path:
        tri = self.analysis.triangle
        lob_label = self.cfg.lob_label(tri.lob)
        company = self.cfg.company_name
        as_of = pd.Timestamp.now().strftime("%B %d, %Y")
        pc = self.cfg.primary_color
        ac = self.cfg.accent_color

        comp = self.analysis.comparison_table()
        comp_html = df_to_html_table(
            comp,
            currency_cols=[c for c in comp.columns if "ult" in c or "ibnr" in c or "reported" in c],
            table_id="comp_tbl",
        )

        # Waterfall chart
        sel = self.analysis.selected()
        origins = [o for o in tri.origins]
        reported = [float(tri.latest_diagonal.get(o, 0)) for o in origins]
        ibnr = [max(0.0, float(sel.ibnr.get(o, 0))) for o in origins]
        chart_html = reserve_waterfall_chart(origins, reported, ibnr, primary_color=pc, accent_color=ac)

        # Method notes
        notes_rows = ""
        for m, res in self.analysis._results.items():
            n = len(sel.ultimates)
            elr_str = f"{res.elr:.4f}" if res.elr else "—"
            notes_rows += f"<tr><td>{m.replace('_',' ').title()}</td><td class='num'>{_fmt_m(res.total_ibnr)}</td><td class='num'>{elr_str}</td><td>{res.notes}</td></tr>"

        def _fmt_m(v):
            if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
            return f"${v:,.0f}"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{company} — Reserve Exhibit</title>
{PLOTLY_CDN}
<style>{_DASHBOARD_CSS}</style>
</head>
<body>
<div class="header">
  <div><h1>{company}</h1><div style="font-size:.9rem;opacity:.8;margin-top:4px">Reserve / IBNR Exhibit — {lob_label}</div></div>
  <div class="meta"><div>As of: {as_of}</div><div style="font-size:.75rem;opacity:.7">auto_actuary</div></div>
</div>
<div class="container">
  <div class="section-title">Reserve Position by Accident Year</div>
  <div class="chart-grid single"><div class="chart-card">{chart_html}</div></div>

  <div class="section-title">Reserve Comparison — All Methods</div>
  <div class="table-card" style="overflow-x:auto">{comp_html}</div>

  <div class="section-title">Method Summary</div>
  <div class="table-card">
    <table class="aa-table">
      <thead><tr><th>Method</th><th>Total IBNR</th><th>ELR Used</th><th>Notes</th></tr></thead>
      <tbody>{notes_rows}</tbody>
    </table>
  </div>
</div>
<div class="footer">auto_actuary — {as_of} — Actuarial use only</div>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
        logger.info("Reserve exhibit saved: %s", path)
        return path
