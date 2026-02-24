"""
auto_actuary.reports.executive.segment_dashboard
==================================================
Dynamic Segment Analytics Dashboard — a self-contained interactive HTML report
that answers segment-level strategic questions:

  "How well are we gaining and keeping good business?"
  "How are we losing or improving bad segments?"
  "Where should we take rate action or change appetite?"
  "What is the expected value of each market segment?"

Layout
------
  ┌─────────────────────────────────────────────────────────────┐
  │  Header (company, LOB, as-of date)                          │
  ├─────────────────────────────────────────────────────────────┤
  │  Segment Selector  [GEO: Territory ▼]  [MARKET: Class ▼]   │
  ├─────────────────────────────────────────────────────────────┤
  │  Scorecard Table (latest period KPIs per segment value)      │
  ├─────────────┬───────────────────────────────────────────────┤
  │ Premium     │  Retention Rate over Time                     │  row 1
  │ Trend       │                                               │
  ├─────────────┼───────────────────────────────────────────────┤
  │ Loss Ratio  │  Pure Premium Trend                           │  row 2
  │ over Time   │                                               │
  ├─────────────┴───────────────────────────────────────────────┤
  │  CLV Comparison Bar Chart                                   │  row 3
  └─────────────────────────────────────────────────────────────┘

All charts are Plotly.js — interactive (hover, zoom, legend toggle).
The segment selector is a pure-JS tab switcher — no server required.
The entire dashboard is a single self-contained HTML file.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_actuary.analytics.portfolio.segment_analytics import SegmentAnalytics
    from auto_actuary.core.session import ActuarySession

logger = logging.getLogger(__name__)

PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>'

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Inter, Arial, sans-serif; background: #F0F2F5; color: #1A1A2E; }
.header { background: linear-gradient(135deg, #002060 0%, #004080 100%);
          color: white; padding: 20px 36px; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 1.5rem; font-weight: 700; }
.header .meta { text-align: right; font-size: 0.85rem; opacity: 0.85; }
.container { max-width: 1500px; margin: 0 auto; padding: 20px 28px; }
.section-title { font-size: 1.05rem; font-weight: 600; color: #002060;
                 border-left: 4px solid #0099CC; padding-left: 10px; margin: 24px 0 14px; }

/* Segment selector tabs */
.seg-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
.seg-group { display: flex; align-items: center; gap: 6px; background: white;
             border-radius: 8px; padding: 8px 14px; box-shadow: 0 1px 6px rgba(0,32,96,.08); }
.seg-group label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: .8px;
                   color: #666; font-weight: 600; }
.seg-btn { border: 1.5px solid #CBD3E0; background: white; border-radius: 6px;
           padding: 5px 14px; cursor: pointer; font-size: 0.82rem; transition: all .15s; }
.seg-btn:hover { background: #EEF4FF; border-color: #0099CC; }
.seg-btn.active { background: #002060; color: white; border-color: #002060; }

/* Scorecard table */
.aa-table { width: 100%; border-collapse: collapse; font-size: 0.81rem; }
.aa-table th { background: #002060; color: white; padding: 8px 11px;
               text-align: center; font-weight: 600; letter-spacing: .3px; }
.aa-table td { padding: 6px 11px; border-bottom: 1px solid #E5E9F0; }
.aa-table td.num { text-align: right; font-family: monospace; }
.aa-table tbody tr:nth-child(even) { background: #F5F8FC; }
.aa-table tbody tr:hover { background: #EEF4FF; }
.aa-table .best { color: #375623; font-weight: 600; }
.aa-table .worst { color: #C00000; font-weight: 600; }
.table-card { background: white; border-radius: 10px; padding: 18px;
              box-shadow: 0 2px 8px rgba(0,32,96,.08); margin-bottom: 20px; overflow-x: auto; }

/* Chart grid */
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }
.chart-card { background: white; border-radius: 10px; padding: 16px;
              box-shadow: 0 2px 8px rgba(0,32,96,.08); }
.chart-grid.single { grid-template-columns: 1fr; }

.footer { text-align: center; padding: 18px; color: #999; font-size: 0.76rem; margin-top: 16px; }
@media (max-width: 900px) { .chart-grid { grid-template-columns: 1fr; } }

/* Segment panel visibility */
.seg-panel { display: none; }
.seg-panel.active { display: block; }
"""

_JS_CONTROLLER = """
function showSegment(segCol, btnEl) {
    // Deactivate all buttons in the same group
    const group = btnEl.closest('.seg-group');
    group.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    btnEl.classList.add('active');
    // Hide all panels for this group's category (geo or market)
    const category = group.dataset.category;
    document.querySelectorAll('.seg-panel[data-category="' + category + '"]')
            .forEach(p => p.classList.remove('active'));
    // Show the selected panel
    const panel = document.getElementById('panel_' + segCol);
    if (panel) panel.classList.add('active');
}

// Auto-activate first button in each group on load
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.seg-group').forEach(function(group) {
        const firstBtn = group.querySelector('.seg-btn');
        if (firstBtn) firstBtn.click();
    });
});
"""


def _to_json(obj: Any) -> str:
    """Serialize numpy/pandas scalars to JSON safely."""
    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if pd.isna(o):
                return None
            return super().default(o)
    return json.dumps(obj, cls=_Enc)


def _safe_list(series: pd.Series) -> list:
    return [None if pd.isna(v) else (int(v) if isinstance(v, (np.integer,)) else
            float(v) if isinstance(v, (np.floating, float)) else v)
            for v in series]


# ---------------------------------------------------------------------------
# Per-segment chart builders
# ---------------------------------------------------------------------------

def _premium_trend_div(df: pd.DataFrame, segment: str, seg_value_col: str = "segment_value",
                        pc: str = "#003366", ac: str = "#0099CC", div_id: str = "pt") -> str:
    """Stacked/grouped bar of written premium by period, one trace per segment value."""
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        return "<p>plotly not installed</p>"

    if df.empty or "period" not in df.columns:
        return "<p>No premium data</p>"

    fig = go.Figure()
    palette = [pc, ac, "#E8A000", "#375623", "#CC2200", "#7B2D8B", "#0D7680"]
    for i, seg_val in enumerate(sorted(df[seg_value_col].unique())):
        sub = df[df[seg_value_col] == seg_val].sort_values("period")
        fig.add_trace(go.Bar(
            x=[str(p) for p in sub["period"]],
            y=sub["written_premium"] / 1e3,
            name=str(seg_val),
            marker_color=palette[i % len(palette)],
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text=f"Written Premium by Period — {segment}", font=dict(size=14, color=pc)),
        xaxis_title="Period", yaxis_title="Written Premium ($K)",
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF",
        height=320, margin=dict(l=50, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Inter, sans-serif", size=10),
    )
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)


def _retention_trend_div(df: pd.DataFrame, segment: str, seg_value_col: str = "segment_value",
                          pc: str = "#003366", ac: str = "#0099CC", div_id: str = "ret") -> str:
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        return "<p>plotly not installed</p>"

    if df.empty or "retention_rate" not in df.columns:
        return "<p>No retention data</p>"

    fig = go.Figure()
    palette = ["#003366", "#0099CC", "#E8A000", "#375623", "#CC2200", "#7B2D8B", "#0D7680"]
    for i, seg_val in enumerate(sorted(df[seg_value_col].unique())):
        sub = df[df[seg_value_col] == seg_val].sort_values("period").dropna(subset=["retention_rate"])
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=[str(p) for p in sub["period"]],
            y=sub["retention_rate"] * 100,
            mode="lines+markers",
            name=str(seg_val),
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=6),
        ))

    fig.add_hline(y=80, line_dash="dot", line_color="#888",
                  annotation_text="80% target", annotation_position="right")
    fig.update_layout(
        title=dict(text=f"Retention Rate by Period — {segment}", font=dict(size=14, color=pc)),
        xaxis_title="Period", yaxis_title="Retention Rate (%)",
        yaxis=dict(range=[0, 105]),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF",
        height=320, margin=dict(l=50, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Inter, sans-serif", size=10),
    )
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)


def _loss_ratio_trend_div(df: pd.DataFrame, segment: str, seg_value_col: str = "segment_value",
                           pc: str = "#003366", div_id: str = "lr") -> str:
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        return "<p>plotly not installed</p>"

    if df.empty or "loss_ratio" not in df.columns:
        return "<p>No loss ratio data</p>"

    fig = go.Figure()
    palette = ["#003366", "#0099CC", "#E8A000", "#375623", "#CC2200", "#7B2D8B", "#0D7680"]
    for i, seg_val in enumerate(sorted(df[seg_value_col].unique())):
        sub = df[df[seg_value_col] == seg_val].sort_values("period").dropna(subset=["loss_ratio"])
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=[str(p) for p in sub["period"]],
            y=sub["loss_ratio"] * 100,
            mode="lines+markers",
            name=str(seg_val),
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=6),
        ))

    fig.add_hline(y=65, line_dash="dot", line_color="#375623",
                  annotation_text="65% target", annotation_position="right")
    fig.add_hline(y=80, line_dash="dash", line_color="#C00000",
                  annotation_text="80% alert", annotation_position="right")
    fig.update_layout(
        title=dict(text=f"Loss Ratio Trend — {segment}", font=dict(size=14, color=pc)),
        xaxis_title="Period", yaxis_title="Loss Ratio (%)",
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF",
        height=320, margin=dict(l=50, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Inter, sans-serif", size=10),
    )
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)


def _pure_premium_trend_div(df: pd.DataFrame, segment: str, seg_value_col: str = "segment_value",
                              pc: str = "#003366", div_id: str = "pp") -> str:
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        return ""

    if df.empty or "incurred_loss" not in df.columns or "earned_premium" not in df.columns:
        return "<p>No pure premium data</p>"

    # Compute pure premium = incurred / earned_premium (per unit earned premium proxy)
    df = df.copy()
    df["pure_premium"] = df["incurred_loss"] / df["earned_premium"].replace(0, np.nan)

    fig = go.Figure()
    palette = ["#003366", "#0099CC", "#E8A000", "#375623", "#CC2200", "#7B2D8B", "#0D7680"]
    for i, seg_val in enumerate(sorted(df[seg_value_col].unique())):
        sub = df[df[seg_value_col] == seg_val].sort_values("period").dropna(subset=["pure_premium"])
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=[str(p) for p in sub["period"]],
            y=sub["pure_premium"],
            name=str(seg_val),
            marker_color=palette[i % len(palette)],
            opacity=0.8,
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text=f"Loss Ratio Trend (Pure Premium) — {segment}", font=dict(size=14, color=pc)),
        xaxis_title="Period", yaxis_title="Loss / EP (Pure Premium Ratio)",
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF",
        height=320, margin=dict(l=50, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Inter, sans-serif", size=10),
    )
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)


def _clv_bar_div(df: pd.DataFrame, segment: str, pc: str = "#003366",
                  ac: str = "#0099CC", div_id: str = "clv") -> str:
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        return ""

    if df.empty or "estimated_clv" not in df.columns:
        return "<p>No CLV data</p>"

    df_sorted = df.sort_values("estimated_clv", ascending=True).dropna(subset=["estimated_clv"])
    if df_sorted.empty:
        return "<p>CLV data unavailable</p>"

    colors = [pc if v >= 0 else "#CC2200" for v in df_sorted["estimated_clv"]]
    fig = go.Figure(go.Bar(
        x=df_sorted["estimated_clv"],
        y=[str(v) for v in df_sorted.index],
        orientation="h",
        marker_color=colors,
        text=[f"${v:,.0f}" for v in df_sorted["estimated_clv"]],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(text=f"Estimated Customer Lifetime Value — {segment}", font=dict(size=14, color=pc)),
        xaxis_title="Estimated CLV ($)", yaxis_title=segment,
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF",
        height=max(280, len(df_sorted) * 38 + 100),
        margin=dict(l=90, r=60, t=60, b=50),
        font=dict(family="Inter, sans-serif", size=10),
    )
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Scorecard table
# ---------------------------------------------------------------------------

def _scorecard_table(scorecard: pd.DataFrame, segment_label: str) -> str:
    if scorecard.empty:
        return "<p>No data</p>"

    sc = scorecard.reset_index()
    cols_display = {
        "segment_value": segment_label,
        "written_premium": "Written Premium",
        "earned_premium": "Earned Premium",
        "policy_count": "Policy Count",
        "incurred_loss": "Incurred Loss",
        "loss_ratio": "Loss Ratio",
        "retention_rate": "Retention",
        "wp_yoy": "WP YoY",
        "estimated_clv": "Est. CLV",
    }
    available = {k: v for k, v in cols_display.items() if k in sc.columns}
    sc = sc[[c for c in available.keys()]]

    # Format
    def _fmt(col: str, val: Any) -> str:
        if pd.isna(val):
            return "—"
        if col in ("loss_ratio", "retention_rate", "wp_yoy"):
            pct = float(val) * 100
            css = ""
            if col == "loss_ratio":
                css = " class='worst'" if pct > 75 else (" class='best'" if pct < 60 else "")
            elif col == "retention_rate":
                css = " class='best'" if pct >= 80 else (" class='worst'" if pct < 65 else "")
            elif col == "wp_yoy":
                css = " class='best'" if pct > 0 else " class='worst'"
            return f"<td class='num pct'{css}>{pct:+.1f}%</td>" if col == "wp_yoy" else f"<td class='num pct'{css}>{pct:.1f}%</td>"
        if col in ("written_premium", "earned_premium", "incurred_loss", "estimated_clv"):
            v = float(val)
            if abs(v) >= 1e6:
                return f"<td class='num'>${v/1e6:.2f}M</td>"
            elif abs(v) >= 1e3:
                return f"<td class='num'>${v/1e3:.1f}K</td>"
            return f"<td class='num'>${v:,.0f}</td>"
        if col == "policy_count":
            return f"<td class='num'>{int(val):,}</td>"
        return f"<td>{val}</td>"

    headers = "".join(f"<th>{lbl}</th>" for lbl in available.values())
    rows = []
    for _, row in sc.iterrows():
        cells = "".join(_fmt(col, row[col]) for col in available.keys())
        rows.append(f"<tr>{cells}</tr>")

    return f'<table class="aa-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


# ---------------------------------------------------------------------------
# Main dashboard class
# ---------------------------------------------------------------------------

class SegmentDashboard:
    """
    Build and render the Segment Analytics Dashboard.

    Parameters
    ----------
    analytics : SegmentAnalytics
    config : ActuaryConfig
    lob : str, optional
    """

    def __init__(
        self,
        analytics: "SegmentAnalytics",
        config: Any,
        lob: Optional[str] = None,
    ) -> None:
        self.analytics = analytics
        self.cfg = config
        self.lob = lob
        self._as_of = datetime.date.today().strftime("%B %d, %Y")

    def render(self, output_path: Union[str, Path] = "output/segment_dashboard.html") -> Path:
        output_path = Path(output_path)
        html = self._build_html()
        output_path.write_text(html, encoding="utf-8")
        logger.info("Segment dashboard saved: %s", output_path)
        return output_path

    def _build_html(self) -> str:
        company = self.cfg.company_name
        lob_label = self.cfg.lob_label(self.lob) if self.lob else "All Lines"
        pc = self.cfg.primary_color
        ac = self.cfg.accent_color

        geo_segs = self.cfg.geo_segments
        mkt_segs = self.cfg.market_segments
        all_segs = self.analytics.segment_cols

        # Build tab buttons and panels
        tab_buttons_html = self._build_tab_buttons(geo_segs, mkt_segs)
        panels_html = self._build_panels(all_segs, pc, ac)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{company} — Segment Analytics Dashboard</title>
{PLOTLY_CDN}
<style>{_CSS}</style>
</head>
<body>
<div class="header">
  <div>
    <h1>{company}</h1>
    <div style="font-size:.95rem;margin-top:4px;opacity:.8">Segment Analytics Dashboard</div>
  </div>
  <div class="meta">
    <div style="font-size:1.05rem;font-weight:600">{lob_label}</div>
    <div>As of: {self._as_of}</div>
    <div style="margin-top:4px;font-size:.73rem;opacity:.7">Produced by auto_actuary</div>
  </div>
</div>

<div class="container">
  <div class="section-title">Segment Selector</div>
  <div class="seg-tabs">{tab_buttons_html}</div>

  {panels_html}
</div>

<div class="footer">
  Generated by <strong>auto_actuary</strong> &mdash; {self._as_of}
  &mdash; For management use; verify assumptions before presenting externally.
</div>

<script>{_JS_CONTROLLER}</script>
</body>
</html>"""

    def _build_tab_buttons(self, geo_segs: List[str], mkt_segs: List[str]) -> str:
        html = ""
        if geo_segs:
            btns = " ".join(
                f'<button class="seg-btn" onclick="showSegment(\'{s}\', this)">'
                f'{self.cfg.segment_label(s)}</button>'
                for s in geo_segs if s in self.analytics.segment_cols
            )
            if btns:
                html += f'<div class="seg-group" data-category="geo"><label>Geography</label>{btns}</div>'
        if mkt_segs:
            btns = " ".join(
                f'<button class="seg-btn" onclick="showSegment(\'{s}\', this)">'
                f'{self.cfg.segment_label(s)}</button>'
                for s in mkt_segs if s in self.analytics.segment_cols
            )
            if btns:
                html += f'<div class="seg-group" data-category="market"><label>Market</label>{btns}</div>'
        return html

    def _build_panels(self, segments: List[str], pc: str, ac: str) -> str:
        html = ""
        geo_segs = self.cfg.geo_segments
        for seg in segments:
            category = "geo" if seg in geo_segs else "market"
            seg_label = self.cfg.segment_label(seg)
            panel_html = self._build_single_panel(seg, seg_label, pc, ac)
            html += (
                f'<div id="panel_{seg}" class="seg-panel" data-category="{category}">'
                f'{panel_html}</div>'
            )
        return html

    def _build_single_panel(self, seg: str, seg_label: str, pc: str, ac: str) -> str:
        """Build all charts and scorecard for one segment dimension."""
        sa = self.analytics

        def _df(val: Optional[pd.DataFrame]) -> pd.DataFrame:
            """Return empty DataFrame when _safe() returns None."""
            return pd.DataFrame() if val is None else val

        # --- Scorecard ---
        scorecard = self._safe(lambda: sa.segment_scorecard(seg))
        scorecard_html = _scorecard_table(_df(scorecard), seg_label)

        # --- Premium trend ---
        prem_df = self._safe(lambda: sa.premium_trend(seg).reset_index())
        prem_chart = _premium_trend_div(
            _df(prem_df), seg_label,
            pc=pc, ac=ac, div_id=f"pt_{seg}"
        )

        # --- Retention trend ---
        ret_df = self._safe(lambda: sa.retention_trend(seg).reset_index())
        ret_chart = _retention_trend_div(
            _df(ret_df), seg_label,
            pc=pc, ac=ac, div_id=f"ret_{seg}"
        )

        # --- Loss ratio trend ---
        loss_df = self._safe(lambda: sa.loss_trend(seg).reset_index())
        lr_chart = _loss_ratio_trend_div(
            _df(loss_df), seg_label,
            pc=pc, div_id=f"lr_{seg}"
        )

        # --- Pure premium / profitability trend ---
        pp_chart = _pure_premium_trend_div(
            _df(loss_df), seg_label,
            pc=pc, div_id=f"pp_{seg}"
        )

        # --- CLV ---
        clv_df = self._safe(lambda: sa.clv_by_segment(seg).reset_index())
        clv_chart = _clv_bar_div(
            _df(clv_df), seg_label,
            pc=pc, ac=ac, div_id=f"clv_{seg}"
        )

        return f"""
<div class="section-title">Segment Scorecard — {seg_label}</div>
<div class="table-card">{scorecard_html}</div>

<div class="section-title">Growth & Premium Trends</div>
<div class="chart-grid">
  <div class="chart-card">{prem_chart}</div>
  <div class="chart-card">{ret_chart}</div>
</div>

<div class="section-title">Loss Cost & Profitability Trends</div>
<div class="chart-grid">
  <div class="chart-card">{lr_chart}</div>
  <div class="chart-card">{pp_chart}</div>
</div>

<div class="section-title">Estimated Customer Lifetime Value</div>
<div class="chart-grid single">
  <div class="chart-card">{clv_chart}</div>
</div>
"""

    def _safe(self, func: Any, default: Any = None) -> Any:
        try:
            return func()
        except Exception as exc:
            logger.debug("Segment dashboard data warning: %s", exc)
            return default
