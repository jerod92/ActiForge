"""
auto_actuary.reports.actuarial.triangle_exhibit
================================================
Format and export a loss development triangle exhibit.

Exhibits are the bread-and-butter of actuarial presentations.
This module generates:
  - Sheet 1: Cumulative triangle (standard layout)
  - Sheet 2: LDF averaging table (all methods + selected)
  - Sheet 3: CDF-to-ultimate and projected ultimates
  - Sheet 4: Incremental triangle (for ILM / diagnostics)

Supports Excel (.xlsx) and HTML output.
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
    from auto_actuary.analytics.triangles.development import LossTriangle
    from auto_actuary.core.config import ActuaryConfig

logger = logging.getLogger(__name__)


class TriangleExhibit:
    """
    Generate a triangle development exhibit.

    Parameters
    ----------
    triangle : LossTriangle
        Must have .develop() called.
    config : ActuaryConfig
    """

    def __init__(
        self,
        triangle: "LossTriangle",
        config: "ActuaryConfig",
    ) -> None:
        self.tri = triangle
        self.cfg = config

    def render(
        self,
        output_path: Union[str, Path],
        fmt: str = "excel",
    ) -> Path:
        output_path = Path(output_path)
        if fmt == "excel":
            return self._render_excel(output_path)
        elif fmt == "html":
            return self._render_html(output_path)
        else:
            raise ValueError(f"Unknown format: {fmt!r}. Use 'excel' or 'html'.")

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------

    def _render_excel(self, path: Path) -> Path:
        tri = self.tri
        lob_label = self.cfg.lob_label(tri.lob)
        value_label = tri.value_type.replace("_", " ").title()

        writer = ExcelWriter(
            title=f"Loss Development Exhibit — {lob_label}",
            company=self.cfg.company_name,
        )
        writer.add_cover(
            produced_by="auto_actuary",
            as_of_date=pd.Timestamp.now().strftime("%B %d, %Y"),
        )

        # Sheet 1: Cumulative triangle
        writer.add_sheet(
            "Cumulative Triangle",
            tri.triangle,
            title=f"Cumulative {value_label} Development Triangle",
            subtitle=f"{lob_label} | {tri.origin_basis.replace('_', ' ').title()}",
            number_format="$#,##0",
        )

        # Sheet 2: LDF exhibit
        if tri._ldf_table is not None:
            ldf_ex = tri.ldf_exhibit()
            writer.add_sheet(
                "LDF Exhibit",
                ldf_ex,
                title="Age-to-Age Development Factors",
                subtitle="All Methods + Selected",
                number_format="0.0000",
            )

        # Sheet 3: CDF and ultimates
        summary = tri.summary()
        writer.add_sheet(
            "CDF & Ultimates",
            summary,
            title="CDF-to-Ultimate and Projected Ultimates",
            subtitle=f"Chain Ladder Method | {lob_label}",
            number_format="$#,##0",
            pct_cols=["pct_unreported"],
        )

        # Sheet 4: Incremental
        inc_tri = tri.to_incremental()
        writer.add_sheet(
            "Incremental Triangle",
            inc_tri.triangle,
            title=f"Incremental {value_label} Triangle",
            number_format="$#,##0",
        )

        return writer.save(path)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _render_html(self, path: Path) -> Path:
        tri = self.tri
        lob_label = self.cfg.lob_label(tri.lob)
        value_label = tri.value_type.replace("_", " ").title()
        company = self.cfg.company_name
        as_of = pd.Timestamp.now().strftime("%B %d, %Y")

        # Cumulative triangle table
        tri_html = df_to_html_table(
            tri.triangle,
            currency_cols=list(str(c) for c in tri.triangle.columns),
            table_id="cum_tri",
        )

        # LDF table
        ldf_html = ""
        if tri._ldf_table is not None:
            ldf_html = df_to_html_table(
                tri.ldf_exhibit(),
                ldf_cols=list(tri.ldf_exhibit().columns),
                table_id="ldf_tbl",
            )

        # Summary table
        summ_html = df_to_html_table(
            tri.summary(),
            currency_cols=["reported", "ultimate", "ibnr"],
            pct_cols=["pct_unreported"],
            table_id="summary_tbl",
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{company} — Triangle Exhibit</title>
{PLOTLY_CDN}
<style>
{_DASHBOARD_CSS}
.tri-wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>{company}</h1>
    <div style="font-size:0.95rem;margin-top:4px;opacity:0.8">Loss Development Exhibit — {lob_label}</div>
  </div>
  <div class="meta">
    <div>{value_label}</div>
    <div>As of: {as_of}</div>
    <div style="font-size:0.75rem;opacity:0.7">auto_actuary</div>
  </div>
</div>
<div class="container">
  <div class="section-title">Cumulative {value_label} Development Triangle</div>
  <div class="table-card tri-wrap">{tri_html}</div>

  {"<div class='section-title'>Age-to-Age Development Factors (All Methods)</div><div class='table-card tri-wrap'>" + ldf_html + "</div>" if ldf_html else ""}

  <div class="section-title">CDF-to-Ultimate & Projected Ultimates (Chain Ladder)</div>
  <div class="table-card">{summ_html}</div>
</div>
<div class="footer">auto_actuary — {as_of} — Actuarial use only</div>
</body>
</html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Triangle exhibit saved: %s", path)
        return path
