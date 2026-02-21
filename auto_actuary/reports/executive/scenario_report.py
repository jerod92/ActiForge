"""
auto_actuary.reports.executive.scenario_report
===============================================
Executive-grade HTML report for speculative scenario analysis.

Layout
------
┌──────────────────────────────────────────────────────────────────┐
│  SPECULATIVE SCENARIO ANALYSIS          As of: {date}  LOB: {lob}│
├──────────────────────────────────────────────────────────────────┤
│  BASE PORTFOLIO KPIs  (6 cards: Premium / LR / CR / Freq / Sev / IBNR)
├─────────────────────────────┬────────────────────────────────────┤
│  SCENARIO COMPARISON TABLE  │  WATERFALL (LR change by scenario) │
├─────────────────────────────┴────────────────────────────────────┤
│  TREND PROJECTIONS (Freq / Severity / PP with CI bands)          │
├──────────────────────────────────────────────────────────────────┤
│  STRESS TEST DISTRIBUTION (loss ratio percentiles)               │
├──────────────────────────────────────────────────────────────────┤
│  SENSITIVITY TORNADO (how much each assumption moves the result) │
├──────────────────────────────────────────────────────────────────┤
│  SEGMENT DRILL-DOWN (per-territory impact of selected scenario)  │
└──────────────────────────────────────────────────────────────────┘

All Plotly charts are embedded inline — the output is a standalone HTML file
that can be emailed to executives without any web server or dependencies.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from auto_actuary.analytics.speculative.scenario_engine import ScenarioEngine, ScenarioResult
from auto_actuary.analytics.speculative.trend_projector import TrendProjector

logger = logging.getLogger(__name__)

PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>'


# ---------------------------------------------------------------------------
# CSS / HTML scaffolding
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Inter, Arial, sans-serif; background: #F0F2F5; color: #1A1A2E; }

.header {
  background: linear-gradient(135deg, #002060 0%, #003d8a 100%);
  color: white; padding: 24px 40px;
  display: flex; justify-content: space-between; align-items: center;
}
.header h1 { font-size: 1.55rem; font-weight: 700; letter-spacing: 0.3px; }
.header .badge {
  background: rgba(255,255,255,0.15); border-radius: 20px;
  padding: 4px 14px; font-size: 0.82rem; margin-left: 12px; font-weight: 500;
}
.header .meta { text-align: right; font-size: 0.82rem; opacity: 0.80; line-height: 1.6; }

.container { max-width: 1440px; margin: 0 auto; padding: 24px 36px; }

.section-title {
  font-size: 1.05rem; font-weight: 700; color: #002060;
  border-left: 4px solid #0099CC; padding-left: 12px;
  margin: 32px 0 16px;
}

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 14px; margin-bottom: 28px; }
.kpi-card {
  background: white; border-radius: 10px; padding: 18px 20px;
  box-shadow: 0 2px 8px rgba(0,32,96,0.08); border-top: 4px solid #0099CC;
}
.kpi-card.warn  { border-top-color: #E8A000; }
.kpi-card.danger{ border-top-color: #C00000; }
.kpi-card.good  { border-top-color: #375623; }
.kpi-label { font-size: 0.70rem; text-transform: uppercase; letter-spacing: 1.1px; color: #666; margin-bottom: 4px; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #002060; }
.kpi-sub   { font-size: 0.76rem; color: #888; margin-top: 4px; }

/* Scenario comparison table */
.scenario-table-wrap { overflow-x: auto; }
table.scenario-table {
  width: 100%; border-collapse: collapse; font-size: 0.88rem;
  background: white; border-radius: 8px; overflow: hidden;
  box-shadow: 0 1px 6px rgba(0,32,96,0.08);
}
table.scenario-table th {
  background: #002060; color: white; padding: 10px 14px;
  text-align: right; font-weight: 600; font-size: 0.80rem;
}
table.scenario-table th:first-child { text-align: left; }
table.scenario-table td { padding: 9px 14px; text-align: right; border-bottom: 1px solid #EEE; }
table.scenario-table td:first-child { text-align: left; font-weight: 600; color: #002060; }
table.scenario-table tr:hover td { background: #F7F9FC; }
.delta-pos { color: #C00000; font-weight: 600; }
.delta-neg { color: #375623; font-weight: 600; }

/* Chart containers */
.chart-row { display: grid; gap: 20px; margin-bottom: 24px; }
.chart-row.two { grid-template-columns: 1fr 1fr; }
.chart-row.three { grid-template-columns: 1fr 1fr 1fr; }
.chart-row.one { grid-template-columns: 1fr; }
.chart-card {
  background: white; border-radius: 10px; padding: 18px;
  box-shadow: 0 2px 8px rgba(0,32,96,0.08);
}
.chart-title { font-size: 0.85rem; font-weight: 600; color: #002060; margin-bottom: 10px; }

/* Notes box */
.notes-box {
  background: #FFF8E7; border: 1px solid #E8D070; border-radius: 8px;
  padding: 14px 18px; margin-top: 6px; font-size: 0.82rem; line-height: 1.6;
}
.notes-box .note-title { font-weight: 700; color: #7A5800; margin-bottom: 6px; }
.notes-box li { margin-left: 18px; color: #555; }

/* Stress test */
.stress-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }

@media (max-width: 900px) {
  .chart-row.two, .chart-row.three, .stress-grid { grid-template-columns: 1fr; }
}
"""


# ---------------------------------------------------------------------------
# Plotly helpers (return JSON strings for embedding)
# ---------------------------------------------------------------------------

def _plotly_div(fig_json: str, div_id: str) -> str:
    return f'<div id="{div_id}"></div><script>Plotly.newPlot("{div_id}", {fig_json});</script>'


def _scenario_waterfall_json(results: List[ScenarioResult], kpi: str = "loss_ratio") -> str:
    """Waterfall: how each scenario changes the selected KPI vs. base."""
    names = [r.name for r in results]
    base_val = results[0].base_kpis.get(kpi, 0) if results else 0
    deltas = [r.deltas.get(kpi, 0) for r in results]
    colors = ["#C00000" if d > 0 else "#375623" for d in deltas]
    texts = [f"{d:+.1%}" if "ratio" in kpi else f"{d:+,.0f}" for d in deltas]

    data = [{
        "type": "waterfall",
        "name": kpi,
        "orientation": "v",
        "x": ["Base"] + names,
        "y": [base_val] + deltas,
        "measure": ["absolute"] + ["relative"] * len(names),
        "text": [f"{base_val:.1%}" if "ratio" in kpi else f"{base_val:,.0f}"] + texts,
        "textposition": "outside",
        "connector": {"line": {"color": "#BBBBBB", "width": 1}},
        "increasing": {"marker": {"color": "#C00000"}},
        "decreasing": {"marker": {"color": "#375623"}},
        "totals": {"marker": {"color": "#002060"}},
    }]
    layout = {
        "title": {"text": f"{kpi.replace('_', ' ').title()} Impact by Scenario", "font": {"size": 14, "color": "#002060"}},
        "height": 320, "margin": {"t": 40, "b": 60, "l": 60, "r": 20},
        "yaxis": {"tickformat": ".1%" if "ratio" in kpi else ",.0f", "gridcolor": "#EEE"},
        "plot_bgcolor": "white", "paper_bgcolor": "white",
        "font": {"family": "Segoe UI, Arial, sans-serif", "size": 12},
    }
    return json.dumps({"data": data, "layout": layout})


def _trend_projection_json(
    proj_df: pd.DataFrame,
    metric_name: str,
    scenarios: Optional[Dict[str, str]] = None,
) -> str:
    """Line chart with CI band for a trend projection DataFrame."""
    hist = proj_df[proj_df["type"] == "historical"]
    proj = proj_df[proj_df["type"] == "projection"]

    data = []

    # CI fill (p10 to p90)
    if "p10" in proj.columns and "p90" in proj.columns:
        all_years = list(proj["year"]) + list(proj["year"])[::-1]
        all_vals = list(proj["p90"]) + list(proj["p10"])[::-1]
        data.append({
            "type": "scatter",
            "x": all_years, "y": all_vals,
            "fill": "toself", "fillcolor": "rgba(0,153,204,0.12)",
            "line": {"color": "transparent"},
            "name": "90% CI band", "showlegend": True,
            "hoverinfo": "skip",
        })

    # P25-P75 fill
    if "p25" in proj.columns and "p75" in proj.columns:
        all_years2 = list(proj["year"]) + list(proj["year"])[::-1]
        all_vals2 = list(proj["p75"]) + list(proj["p25"])[::-1]
        data.append({
            "type": "scatter",
            "x": all_years2, "y": all_vals2,
            "fill": "toself", "fillcolor": "rgba(0,153,204,0.22)",
            "line": {"color": "transparent"},
            "name": "50% CI band", "showlegend": True,
            "hoverinfo": "skip",
        })

    # Historical series
    data.append({
        "type": "scatter",
        "x": list(hist["year"]), "y": list(hist["point"]),
        "mode": "lines+markers",
        "name": "Historical",
        "line": {"color": "#002060", "width": 2.5},
        "marker": {"size": 8, "color": "#002060"},
    })

    # Point projection
    data.append({
        "type": "scatter",
        "x": list(proj["year"]), "y": list(proj["point"]),
        "mode": "lines+markers",
        "name": "Base projection",
        "line": {"color": "#0099CC", "width": 2, "dash": "dash"},
        "marker": {"size": 7, "color": "#0099CC"},
    })

    # Scenario forks
    scen_colors = ["#E8A000", "#C00000", "#375623", "#6A0572", "#1A6B3C"]
    if scenarios:
        for i, (scen_col, scen_label) in enumerate(scenarios.items()):
            if scen_col in proj.columns:
                data.append({
                    "type": "scatter",
                    "x": list(proj["year"]), "y": list(proj[scen_col]),
                    "mode": "lines",
                    "name": scen_label,
                    "line": {"color": scen_colors[i % len(scen_colors)], "width": 1.8, "dash": "dot"},
                })

    layout = {
        "title": {"text": metric_name, "font": {"size": 13, "color": "#002060"}},
        "height": 300, "margin": {"t": 40, "b": 50, "l": 60, "r": 20},
        "yaxis": {"gridcolor": "#EEE"},
        "xaxis": {"gridcolor": "#EEE"},
        "plot_bgcolor": "white", "paper_bgcolor": "white",
        "font": {"family": "Segoe UI, Arial, sans-serif", "size": 11},
        "legend": {"font": {"size": 10}},
    }
    return json.dumps({"data": data, "layout": layout})


def _stress_distribution_json(stress_df: pd.DataFrame) -> str:
    """Histogram + CDF of stress test loss ratios."""
    lr = stress_df["loss_ratio"].values

    data = [{
        "type": "histogram",
        "x": lr.tolist(),
        "nbinsx": 30,
        "marker": {"color": "#0099CC", "opacity": 0.75, "line": {"color": "white", "width": 0.5}},
        "name": "Simulated LR",
    }]

    # Vertical lines at percentiles
    p50 = float(np.percentile(lr, 50))
    p90 = float(np.percentile(lr, 90))

    shapes = [
        {"type": "line", "x0": p50, "x1": p50, "y0": 0, "y1": 1,
         "yref": "paper", "line": {"color": "#E8A000", "width": 2, "dash": "dash"}},
        {"type": "line", "x0": p90, "x1": p90, "y0": 0, "y1": 1,
         "yref": "paper", "line": {"color": "#C00000", "width": 2, "dash": "dash"}},
    ]

    annotations = [
        {"x": p50, "y": 1.02, "yref": "paper", "text": f"P50: {p50:.1%}",
         "showarrow": False, "font": {"color": "#E8A000", "size": 11}},
        {"x": p90, "y": 1.08, "yref": "paper", "text": f"P90: {p90:.1%}",
         "showarrow": False, "font": {"color": "#C00000", "size": 11}},
    ]

    layout = {
        "title": {"text": "Stress-Test Loss Ratio Distribution", "font": {"size": 13, "color": "#002060"}},
        "height": 300, "margin": {"t": 60, "b": 50, "l": 60, "r": 20},
        "xaxis": {"title": "Loss Ratio", "tickformat": ".1%", "gridcolor": "#EEE"},
        "yaxis": {"title": "Count", "gridcolor": "#EEE"},
        "plot_bgcolor": "white", "paper_bgcolor": "white",
        "font": {"family": "Segoe UI, Arial, sans-serif", "size": 11},
        "shapes": shapes, "annotations": annotations,
    }
    return json.dumps({"data": data, "layout": layout})


def _sensitivity_tornado_json(sensitivity_df: pd.DataFrame, scenario_name: str = "") -> str:
    """Horizontal bar chart showing sensitivity of LR to trend assumption."""
    df = sensitivity_df.sort_values("pct_change_vs_base")
    colors = ["#C00000" if v > 0 else "#375623" for v in df["pct_change_vs_base"]]

    data = [{
        "type": "bar",
        "orientation": "h",
        "x": df["pct_change_vs_base"].tolist(),
        "y": [f"{t:+.1%}/yr" for t in df["assumed_trend"].tolist()],
        "marker": {"color": colors},
        "text": [f"{v:+.1%}" for v in df["pct_change_vs_base"].tolist()],
        "textposition": "outside",
        "name": "vs. Base",
    }]
    layout = {
        "title": {"text": f"Trend Sensitivity — Impact on Projected Value {scenario_name}",
                  "font": {"size": 13, "color": "#002060"}},
        "height": max(300, 20 * len(df) + 80),
        "margin": {"t": 50, "b": 60, "l": 90, "r": 80},
        "xaxis": {"tickformat": ".1%", "gridcolor": "#EEE", "zeroline": True,
                  "zerolinecolor": "#002060", "zerolinewidth": 1.5},
        "yaxis": {"gridcolor": "#EEE"},
        "plot_bgcolor": "white", "paper_bgcolor": "white",
        "font": {"family": "Segoe UI, Arial, sans-serif", "size": 11},
    }
    return json.dumps({"data": data, "layout": layout})


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float, decimals: int = 1) -> str:
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}%}"


def _fmt_dollar(v: float) -> str:
    if np.isnan(v):
        return "—"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def _delta_class(delta: float) -> str:
    """CSS class for a KPI delta (positive = bad for losses/LR; negative = good)."""
    if delta > 0:
        return "delta-pos"
    if delta < 0:
        return "delta-neg"
    return ""


def _kpi_card_color(kpi: str, value: float) -> str:
    """Assign card accent color based on KPI and value thresholds."""
    if "combined_ratio" in kpi:
        return "danger" if value > 1.05 else ("warn" if value > 1.0 else "good")
    if "loss_ratio" in kpi:
        return "danger" if value > 0.80 else ("warn" if value > 0.70 else "good")
    return ""


# ---------------------------------------------------------------------------
# Scenario comparison table
# ---------------------------------------------------------------------------

def _render_scenario_table(results: List[ScenarioResult]) -> str:
    if not results:
        return ""

    kpis = sorted(set(k for r in results for k in r.scenario_kpis.keys()))

    # Formatting rules per KPI
    def _fmt_kpi(kpi: str, val: float) -> str:
        if np.isnan(val):
            return "—"
        if "ratio" in kpi:
            return f"{val:.1%}"
        if kpi in ("overall_frequency",):
            return f"{val:.4f}"
        if kpi in ("overall_severity", "overall_pure_premium"):
            return f"${val:,.0f}"
        return _fmt_dollar(val)

    def _fmt_delta(kpi: str, val: float) -> str:
        if np.isnan(val):
            return "—"
        if "ratio" in kpi:
            return f"{val:+.1%}"
        return _fmt_dollar(val)

    # Build header
    scen_names = [r.name for r in results]
    header_cells = "".join(
        f'<th colspan="2">{n}</th>' for n in scen_names
    )
    sub_header = "".join(
        "<th>Value</th><th>Δ vs Base</th>" for _ in scen_names
    )
    html = f"""
    <div class="scenario-table-wrap">
    <table class="scenario-table">
      <thead>
        <tr>
          <th>KPI</th>
          <th colspan="2">Base</th>
          {header_cells}
        </tr>
        <tr>
          <th></th>
          <th>Value</th><th></th>
          {sub_header}
        </tr>
      </thead>
      <tbody>
    """

    for kpi in kpis:
        base_val = results[0].base_kpis.get(kpi, np.nan)
        base_str = _fmt_kpi(kpi, base_val)

        scen_cells = ""
        for r in results:
            sv = r.scenario_kpis.get(kpi, np.nan)
            dv = r.deltas.get(kpi, np.nan)
            dclass = _delta_class(dv if not np.isnan(dv) else 0)
            ci = r.ci_90.get(kpi, (np.nan, np.nan))
            ci_str = ""
            if not np.isnan(ci[0]) and not np.isnan(ci[1]) and "ratio" in kpi:
                ci_str = f'<br><span style="font-size:0.72rem;color:#888">[{_fmt_kpi(kpi, ci[0])}, {_fmt_kpi(kpi, ci[1])}]</span>'
            scen_cells += f"<td>{_fmt_kpi(kpi, sv)}{ci_str}</td>"
            scen_cells += f'<td class="{dclass}">{_fmt_delta(kpi, dv)}</td>'

        label = kpi.replace("_", " ").title()
        html += f"<tr><td>{label}</td><td>{base_str}</td><td></td>{scen_cells}</tr>"

    html += "</tbody></table></div>"
    return html


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

def _render_kpi_cards(kpis: Dict[str, float]) -> str:
    card_defs = [
        ("total_premium", "Total Premium", _fmt_dollar, ""),
        ("total_losses", "Total Losses", _fmt_dollar, ""),
        ("total_claims", "Total Claims", lambda v: f"{v:,.0f}", ""),
        ("loss_ratio", "Loss Ratio", _fmt_pct, ""),
        ("expense_ratio", "Expense Ratio", _fmt_pct, ""),
        ("combined_ratio", "Combined Ratio", _fmt_pct, ""),
        ("overall_frequency", "Frequency", lambda v: f"{v:.4f}", "per unit"),
        ("overall_severity", "Severity", _fmt_dollar, "avg per claim"),
        ("overall_pure_premium", "Pure Premium", _fmt_dollar, "per unit"),
    ]

    cards = ""
    for key, label, fmt, sub in card_defs:
        if key not in kpis:
            continue
        val = kpis[key]
        color_class = _kpi_card_color(key, val)
        formatted = fmt(val) if not np.isnan(val) else "—"
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        cards += f"""
        <div class="kpi-card {color_class}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{formatted}</div>
          {sub_html}
        </div>"""

    return f'<div class="kpi-grid">{cards}</div>'


# ---------------------------------------------------------------------------
# Main report class
# ---------------------------------------------------------------------------

class ScenarioReport:
    """
    Self-contained HTML report for speculative scenario analysis.

    Parameters
    ----------
    engine : ScenarioEngine
        The scenario engine with baseline data.
    results : list of ScenarioResult
        Scenario results to include in the report.
    trend_projectors : dict, optional
        Dict of metric_name -> TrendProjector for trend charts.
        Keys are metric names: "frequency", "severity", "pure_premium".
    stress_df : DataFrame, optional
        Output of ScenarioEngine.stress_test() for the distribution chart.
    lob : str, optional
        Line of business label for display.
    company : str, optional
        Company name for the report header.
    horizon_years : int
        Projection horizon for trend charts.
    """

    def __init__(
        self,
        engine: ScenarioEngine,
        results: List[ScenarioResult],
        trend_projectors: Optional[Dict[str, TrendProjector]] = None,
        stress_df: Optional[pd.DataFrame] = None,
        lob: str = "",
        company: str = "",
        horizon_years: int = 3,
    ) -> None:
        self.engine = engine
        self.results = results
        self.trend_projectors = trend_projectors or {}
        self.stress_df = stress_df
        self.lob = lob
        self.company = company
        self.horizon_years = horizon_years

    def render(self, output_path: Union[str, Path]) -> Path:
        """
        Render the scenario report to an HTML file.

        Parameters
        ----------
        output_path : str or Path

        Returns
        -------
        Path to the rendered file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self._build_html()
        output_path.write_text(html, encoding="utf-8")
        logger.info("ScenarioReport written to: %s", output_path)
        return output_path

    def _build_html(self) -> str:
        now = datetime.datetime.now().strftime("%B %d, %Y")
        title_lob = f" — {self.lob}" if self.lob else ""
        company_str = self.company or "auto_actuary"
        div_counter = [0]

        def next_div() -> str:
            div_counter[0] += 1
            return f"chart_{div_counter[0]}"

        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        header = f"""
        <div class="header">
          <div>
            <h1>{company_str}
              <span class="badge">Speculative Scenario Analysis</span>
            </h1>
          </div>
          <div class="meta">
            As of: {now}{title_lob}<br>
            Scenarios: {len(self.results)}
          </div>
        </div>"""

        # ------------------------------------------------------------------
        # Base KPI cards
        # ------------------------------------------------------------------
        base_kpis = self.engine.base_kpis
        kpi_cards = _render_kpi_cards(base_kpis)

        # ------------------------------------------------------------------
        # Scenario comparison table + LR waterfall
        # ------------------------------------------------------------------
        table_html = _render_scenario_table(self.results)

        wf_div = next_div()
        waterfall_json = _scenario_waterfall_json(self.results, "loss_ratio")
        waterfall_chart = _plotly_div(waterfall_json, wf_div)

        # ------------------------------------------------------------------
        # Trend projection charts
        # ------------------------------------------------------------------
        trend_charts_html = ""
        scenario_forks = {
            "pessimistic": "Pessimistic",
            "base_override": "Base (override)",
            "optimistic": "Optimistic",
        }

        if self.trend_projectors:
            trend_divs = []
            for metric, tp in self.trend_projectors.items():
                try:
                    proj_df = tp.project(
                        horizon_years=self.horizon_years,
                        scenarios={
                            "pessimistic": (tp.fitted_annual_trend - 1) + 0.03,
                            "optimistic": (tp.fitted_annual_trend - 1) - 0.02,
                        },
                    )
                    div_id = next_div()
                    fig_json = _trend_projection_json(
                        proj_df,
                        metric_name=f"{metric.replace('_', ' ').title()} Trend Projection",
                        scenarios={"pessimistic": "Pessimistic (+3%)", "optimistic": "Optimistic (−2%)"},
                    )
                    trend_divs.append(
                        f'<div class="chart-card"><div class="chart-title"></div>'
                        f'{_plotly_div(fig_json, div_id)}</div>'
                    )
                except Exception as e:
                    logger.debug("Skipping trend chart for %s: %s", metric, e)

            if trend_divs:
                ncols = min(len(trend_divs), 3)
                col_class = {1: "one", 2: "two", 3: "three"}.get(ncols, "three")
                trend_charts_html = f'<div class="chart-row {col_class}">{"".join(trend_divs)}</div>'

        # ------------------------------------------------------------------
        # Stress test distribution
        # ------------------------------------------------------------------
        stress_html = ""
        if self.stress_df is not None and len(self.stress_df) > 0:
            stress_div = next_div()
            stress_json = _stress_distribution_json(self.stress_df)
            stress_html = (
                f'<div class="chart-card">{_plotly_div(stress_json, stress_div)}</div>'
            )

        # ------------------------------------------------------------------
        # Sensitivity tornado (for first projector if available)
        # ------------------------------------------------------------------
        tornado_html = ""
        if self.trend_projectors:
            first_metric = next(iter(self.trend_projectors))
            tp = self.trend_projectors[first_metric]
            try:
                sens_df = tp.sensitivity(horizon_years=self.horizon_years, n_steps=15)
                tornado_div = next_div()
                tornado_json = _sensitivity_tornado_json(sens_df, f"({first_metric})")
                tornado_html = (
                    f'<div class="chart-card">{_plotly_div(tornado_json, tornado_div)}</div>'
                )
            except Exception as e:
                logger.debug("Skipping tornado: %s", e)

        # ------------------------------------------------------------------
        # Regime change annotations
        # ------------------------------------------------------------------
        regime_notes = []
        for metric, tp in self.trend_projectors.items():
            try:
                rcr = tp.detect_regime_change()
                if rcr.is_significant:
                    regime_notes.append(
                        f"{metric.title()}: structural break detected at {rcr.break_year:.0f} — "
                        f"pre-break trend {rcr.pre_break_trend:+.1%}/yr, "
                        f"post-break {rcr.post_break_trend:+.1%}/yr (p={rcr.p_value:.3f})"
                    )
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Scenario notes
        # ------------------------------------------------------------------
        all_notes = []
        for r in self.results:
            for note in r.notes:
                all_notes.append(f"<b>{r.name}:</b> {note}")

        notes_html = ""
        if all_notes or regime_notes:
            notes_items = "".join(f"<li>{n}</li>" for n in regime_notes + all_notes)
            notes_html = f"""
            <div class="notes-box">
              <div class="note-title">Actuarial Assumptions &amp; Model Notes</div>
              <ul>{notes_items}</ul>
            </div>"""

        # ------------------------------------------------------------------
        # Assemble full HTML
        # ------------------------------------------------------------------
        body = f"""
        {header}
        <div class="container">

          <div class="section-title">Base Portfolio KPIs</div>
          {kpi_cards}

          <div class="section-title">Scenario Comparison</div>
          <div class="chart-row two">
            <div>{table_html}</div>
            <div class="chart-card">{waterfall_chart}</div>
          </div>

          {"<div class='section-title'>Trend Projections with Uncertainty Bands</div>" + trend_charts_html if trend_charts_html else ""}

          {"<div class='section-title'>Stress Test Distribution</div><div class='chart-row two'>" + stress_html + tornado_html + "</div>" if stress_html or tornado_html else ""}

          {notes_html}

        </div>
        """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Scenario Analysis — {company_str}{title_lob}</title>
  {PLOTLY_CDN}
  <style>{_CSS}</style>
</head>
<body>{body}</body>
</html>"""

    def __repr__(self) -> str:
        return f"ScenarioReport(scenarios={len(self.results)}, lob={self.lob!r})"
