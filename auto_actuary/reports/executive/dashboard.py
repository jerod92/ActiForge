"""
auto_actuary.reports.executive.dashboard
=========================================
Executive HTML dashboard — a polished single-page report for management.

Layout:
  ┌─────────────────────────────────────────────────┐
  │  Company Name          [Logo]    Date / LOB      │
  ├──────┬──────┬──────┬──────┬──────┬──────────────┤
  │  WP  │  EP  │  LR  │  CR  │ IBNR │ Rate Ind.    │  ← KPI Cards
  ├──────┴──────┴──────┴──────┴──────┴──────────────┤
  │  Combined Ratio Trend  │  Written Premium Trend  │  ← Charts row 1
  ├──────────────────────────┬──────────────────────┤
  │  Loss Ratio by Coverage  │  Reserve Waterfall   │  ← Charts row 2
  ├──────────────────────────┴──────────────────────┤
  │  Frequency/Severity Trend                        │  ← Charts row 3
  ├─────────────────────────────────────────────────┤
  │  Portfolio Summary Table (by LOB / Territory)    │  ← Tables
  └─────────────────────────────────────────────────┘

The output is a self-contained HTML file (Plotly CDN embedded).
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from auto_actuary.reports.renderers.html import (
    combined_ratio_chart,
    premium_trend_chart,
    fs_trend_chart,
    reserve_waterfall_chart,
    df_to_html_table,
)

if TYPE_CHECKING:
    from auto_actuary.core.session import ActuarySession

logger = logging.getLogger(__name__)

# Plotly CDN script tag (pinned version for reproducibility)
PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>'

# ---------------------------------------------------------------------------
# Dashboard HTML template (inline — no Jinja2 required)
# ---------------------------------------------------------------------------

_DASHBOARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Inter, Arial, sans-serif; background: #F0F2F5; color: #1A1A2E; }
.header { background: linear-gradient(135deg, #002060 0%, #004080 100%);
           color: white; padding: 24px 40px; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: 0.5px; }
.header .meta { text-align: right; font-size: 0.85rem; opacity: 0.85; }
.container { max-width: 1400px; margin: 0 auto; padding: 24px 32px; }
.section-title { font-size: 1.1rem; font-weight: 600; color: #002060;
                  border-left: 4px solid #0099CC; padding-left: 12px;
                  margin: 28px 0 16px; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; margin-bottom: 28px; }
.kpi-card { background: white; border-radius: 10px; padding: 20px 22px;
            box-shadow: 0 2px 8px rgba(0,32,96,0.10);
            border-top: 4px solid #0099CC; }
.kpi-card.warn { border-top-color: #E8A000; }
.kpi-card.danger { border-top-color: #C00000; }
.kpi-card.good { border-top-color: #375623; }
.kpi-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #666; }
.kpi-value { font-size: 2rem; font-weight: 700; margin: 6px 0 4px; color: #002060; }
.kpi-sub { font-size: 0.78rem; color: #888; }
.kpi-delta { font-size: 0.82rem; font-weight: 600; margin-top: 6px; }
.kpi-delta.up { color: #C00000; }
.kpi-delta.down { color: #375623; }

/* Charts */
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.chart-grid.single { grid-template-columns: 1fr; }
.chart-card { background: white; border-radius: 10px; padding: 20px;
              box-shadow: 0 2px 8px rgba(0,32,96,0.08); }

/* Tables */
.aa-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.aa-table th { background: #002060; color: white; padding: 9px 12px;
               text-align: center; font-weight: 600; letter-spacing: 0.3px; }
.aa-table td { padding: 7px 12px; border-bottom: 1px solid #E5E9F0; }
.aa-table td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: monospace; }
.aa-table tbody tr:nth-child(even) { background: #F5F8FC; }
.aa-table tbody tr:hover { background: #EEF4FF; }
.aa-table .total-row td { background: #FFC000 !important; font-weight: 700; }
.aa-table td.pct { color: #333; }
.table-card { background: white; border-radius: 10px; padding: 20px;
              box-shadow: 0 2px 8px rgba(0,32,96,0.08); margin-bottom: 20px; overflow-x: auto; }
.footer { text-align: center; padding: 20px; color: #999; font-size: 0.78rem; margin-top: 20px; }
@media (max-width: 900px) { .chart-grid { grid-template-columns: 1fr; } }
"""


def _kpi_card(label: str, value: str, sub: str = "", delta: str = "", delta_dir: str = "") -> str:
    card_class = "kpi-card"
    delta_html = f'<div class="kpi-delta {delta_dir}">{delta}</div>' if delta else ""
    return f"""
    <div class="{card_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {delta_html}
    </div>"""


def _fmt_m(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v/1e9:.1f}B"
    elif abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    elif abs(v) >= 1e3:
        return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------

class ExecDashboard:
    """
    Build a self-contained executive HTML dashboard from session data.

    Parameters
    ----------
    session : ActuarySession
    lob : str, optional
        Filter dashboard to a single LOB.
    """

    def __init__(self, session: "ActuarySession", lob: Optional[str] = None) -> None:
        self.session = session
        self.lob = lob
        self.cfg = session.config
        self._as_of = datetime.date.today().strftime("%B %d, %Y")

    def render(self, output_path: Union[str, Path] = "output/dashboard.html") -> Path:
        """Build and save the dashboard HTML."""
        output_path = Path(output_path)
        html = self._build_html()
        output_path.write_text(html, encoding="utf-8")
        logger.info("Executive dashboard saved: %s", output_path)
        return output_path

    def _build_html(self) -> str:
        """Assemble the full HTML document."""
        company = self.cfg.company_name
        lob_label = self.cfg.lob_label(self.lob) if self.lob else "All Lines"

        # Gather data
        kpis = self._kpis()
        charts = self._charts()
        tables = self._tables()

        # KPI grid
        kpi_html = '<div class="kpi-grid">'
        kpi_html += _kpi_card("Written Premium", _fmt_m(kpis.get("written_premium", 0)), "Current Year")
        kpi_html += _kpi_card("Earned Premium", _fmt_m(kpis.get("earned_premium", 0)), "Current Year")
        lr = kpis.get("incurred_loss_ratio", np.nan)
        lr_class = "danger" if (not np.isnan(lr) and lr > 0.75) else ("warn" if not np.isnan(lr) and lr > 0.65 else "good")
        kpi_html += f'<div class="kpi-card {lr_class}"><div class="kpi-label">Loss Ratio</div><div class="kpi-value">{_fmt_pct(lr) if not np.isnan(lr) else "N/A"}</div><div class="kpi-sub">Incurred / Earned</div></div>'
        cr = kpis.get("combined_ratio", np.nan)
        cr_class = "danger" if (not np.isnan(cr) and cr > 1.05) else ("warn" if not np.isnan(cr) and cr > 0.98 else "good")
        kpi_html += f'<div class="kpi-card {cr_class}"><div class="kpi-label">Combined Ratio</div><div class="kpi-value">{_fmt_pct(cr) if not np.isnan(cr) else "N/A"}</div><div class="kpi-sub">Current Year</div></div>'
        ibnr = kpis.get("total_ibnr", np.nan)
        kpi_html += _kpi_card("IBNR Reserve", _fmt_m(ibnr) if not np.isnan(ibnr) else "N/A", "All Open Years")
        rate_ind = kpis.get("rate_indication", np.nan)
        ri_class = "warn" if (not np.isnan(rate_ind) and abs(rate_ind) > 0.05) else "good"
        kpi_html += f'<div class="kpi-card {ri_class}"><div class="kpi-label">Rate Indication</div><div class="kpi-value">{rate_ind:+.1%}</div><div class="kpi-sub">Latest indication</div></div>' if not np.isnan(rate_ind) else _kpi_card("Rate Indication", "N/A", "Run rate_indication()")
        kpi_html += "</div>"

        # Charts
        charts_row1 = f"""
        <div class="chart-grid">
            <div class="chart-card">{charts.get("combined_ratio", "<p>No data</p>")}</div>
            <div class="chart-card">{charts.get("premium_trend", "<p>No data</p>")}</div>
        </div>"""

        charts_row2 = f"""
        <div class="chart-grid">
            <div class="chart-card">{charts.get("reserve_waterfall", "<p>No data</p>")}</div>
            <div class="chart-card">{charts.get("fs_trend", "<p>No data</p>")}</div>
        </div>"""

        tables_html = ""
        for tbl_title, tbl_html in tables.items():
            tables_html += f'<div class="section-title">{tbl_title}</div><div class="table-card">{tbl_html}</div>'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{company} — Executive Dashboard</title>
{PLOTLY_CDN}
<style>{_DASHBOARD_CSS}</style>
</head>
<body>
<div class="header">
  <div>
    <h1>{company}</h1>
    <div style="font-size:1rem;margin-top:4px;opacity:0.8">Executive Actuarial Dashboard</div>
  </div>
  <div class="meta">
    <div style="font-size:1.1rem;font-weight:600">{lob_label}</div>
    <div>As of: {self._as_of}</div>
    <div style="margin-top:4px;font-size:0.75rem;opacity:0.7">Produced by auto_actuary</div>
  </div>
</div>

<div class="container">
  <div class="section-title">Key Performance Indicators</div>
  {kpi_html}

  <div class="section-title">Profitability Trends</div>
  {charts_row1}

  <div class="section-title">Reserve Position & Loss Cost Trends</div>
  {charts_row2}

  {tables_html}
</div>

<div class="footer">
  Generated by <strong>auto_actuary</strong> &mdash; {self._as_of}
  &mdash; For actuarial use; verify assumptions before presenting externally.
</div>
</body>
</html>"""
        return html

    # ------------------------------------------------------------------
    # Data gathering (gracefully handles missing tables)
    # ------------------------------------------------------------------

    def _safe(self, func, default=None):
        try:
            return func()
        except Exception as exc:
            logger.debug("Dashboard data collection warning: %s", exc)
            return default

    def _kpis(self) -> dict:
        out = {}
        loader = self.session.loader

        # Premium KPIs
        if "policies" in loader:
            pol = loader["policies"].copy()
            if self.lob:
                pol = pol[pol["line_of_business"] == self.lob]
            pol["year"] = pol["effective_date"].dt.year
            latest_yr = pol["year"].max()
            pol_ly = pol[pol["year"] == latest_yr]
            out["written_premium"] = float(pol_ly["written_premium"].sum())
            ep_col = "earned_premium" if "earned_premium" in pol_ly.columns else "written_premium"
            out["earned_premium"] = float(pol_ly[ep_col].sum())

        # Loss ratio
        if "claims" in loader and "valuations" in loader:
            cr = self._safe(lambda: self.session.combined_ratio(lob=self.lob).current_year(), {})
            out.update(cr)

        # IBNR
        if "claims" in loader and "valuations" in loader and self.lob:
            res = self._safe(lambda: self.session.reserve_analysis(lob=self.lob))
            if res:
                out["total_ibnr"] = float(res.total_ibnr())

        # Rate indication
        if "claims" in loader and "valuations" in loader and self.lob:
            ind = self._safe(lambda: self.session.rate_indication(lob=self.lob).compute())
            if ind:
                out["rate_indication"] = float(ind.indicated_change)

        return out

    def _charts(self) -> dict:
        charts = {}
        loader = self.session.loader
        pc = self.cfg.primary_color
        ac = self.cfg.accent_color

        # Combined ratio trend
        if "claims" in loader and "valuations" in loader:
            cr_report = self._safe(lambda: self.session.combined_ratio(lob=self.lob))
            if cr_report:
                cr_df = self._safe(lambda: cr_report.by_year())
                if cr_df is not None and not cr_df.empty:
                    charts["combined_ratio"] = combined_ratio_chart(
                        cr_df.reset_index(), year_col="calendar_year",
                        primary_color=pc, accent_color=ac
                    )

        # Premium trend
        if "policies" in loader:
            pol = loader["policies"].copy()
            if self.lob:
                pol = pol[pol["line_of_business"] == self.lob]
            pol["calendar_year"] = pol["effective_date"].dt.year
            ep_col = "earned_premium" if "earned_premium" in pol.columns else "written_premium"
            prem_trend = pol.groupby("calendar_year").agg(
                written_premium=("written_premium", "sum"),
                earned_premium=(ep_col, "sum"),
            ).reset_index()
            charts["premium_trend"] = premium_trend_chart(
                prem_trend, year_col="calendar_year",
                primary_color=pc, accent_color=ac
            )

        # Reserve waterfall
        if "claims" in loader and "valuations" in loader and self.lob:
            res = self._safe(lambda: self.session.reserve_analysis(lob=self.lob))
            if res:
                sel = res.selected()
                summ = self.session.build_triangle(lob=self.lob).develop().summary()
                origins = list(summ.index)
                reported = [float(v) for v in summ["reported"].values]
                ibnr = [max(0.0, float(v)) for v in sel.ibnr.reindex(origins).fillna(0).values]
                charts["reserve_waterfall"] = reserve_waterfall_chart(
                    origins, reported, ibnr,
                    primary_color=pc, accent_color=ac
                )

        # Frequency/severity trend
        if "claims" in loader and "valuations" in loader and self.lob:
            fs = self._safe(lambda: self.session.freq_severity(lob=self.lob))
            if fs:
                fs_tbl = self._safe(lambda: fs.fs_table().reset_index())
                if fs_tbl is not None and not fs_tbl.empty:
                    charts["fs_trend"] = fs_trend_chart(
                        fs_tbl.rename(columns={"accident_year": "year"}),
                        primary_color=pc, title="Loss Cost Trend"
                    )

        return charts

    def _tables(self) -> dict:
        tables = {}
        loader = self.session.loader

        # Portfolio by LOB
        if "claims" in loader and "valuations" in loader and "policies" in loader:
            lr = self._safe(lambda: self.session.loss_ratios())
            if lr:
                lob_tbl = self._safe(lambda: lr.by_lob())
                if lob_tbl is not None and not lob_tbl.empty:
                    tables["Portfolio by Line of Business"] = df_to_html_table(
                        lob_tbl,
                        pct_cols=["incurred_loss_ratio", "paid_loss_ratio", "lae_ratio"],
                        currency_cols=["incurred_loss", "paid_loss", "lae", "earned_premium"],
                        table_id="lob_table",
                    )

        # Cohort summary
        if "claims" in loader and "valuations" in loader and "policies" in loader and self.lob:
            cohort = self._safe(lambda: self.session.cohort_analysis(lob=self.lob))
            if cohort:
                pl = self._safe(lambda: cohort.cohort_pl().tail(7))
                if pl is not None and not pl.empty:
                    display_cols = ["written_premium", "earned_premium", "policy_count",
                                    "incurred_loss", "loss_ratio", "uw_profit"]
                    pl_disp = pl[[c for c in display_cols if c in pl.columns]]
                    tables[f"Cohort Profitability — {self.lob}"] = df_to_html_table(
                        pl_disp,
                        pct_cols=["loss_ratio", "nb_pct"],
                        currency_cols=["written_premium", "earned_premium", "incurred_loss", "uw_profit"],
                        table_id="cohort_table",
                    )

        return tables
