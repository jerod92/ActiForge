"""
auto_actuary.reports.renderers.html
=====================================
HTML rendering via Jinja2 + Plotly.

All charts use Plotly.js (served via CDN in generated HTML files).
Reports are self-contained single-file HTML — no server required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.io import to_html
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "templates"


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def combined_ratio_chart(
    data: pd.DataFrame,
    year_col: str = "calendar_year",
    loss_col: str = "loss_lae_ratio",
    expense_col: str = "expense_ratio",
    combined_col: str = "combined_ratio",
    primary_color: str = "#003366",
    accent_color: str = "#0099CC",
    title: str = "Combined Ratio by Calendar Year",
) -> str:
    """Return Plotly HTML div for a stacked-bar + line combined ratio chart."""
    if not HAS_PLOTLY:
        return "<p>plotly not installed — pip install plotly</p>"

    df = data.reset_index()
    if year_col not in df.columns:
        return "<p>No year data</p>"

    fig = go.Figure()

    if loss_col in df.columns:
        fig.add_trace(go.Bar(
            x=df[year_col], y=df[loss_col] * 100,
            name="Loss & LAE Ratio",
            marker_color=primary_color,
            opacity=0.85,
        ))

    if expense_col in df.columns:
        fig.add_trace(go.Bar(
            x=df[year_col], y=df[expense_col] * 100,
            name="Expense Ratio",
            marker_color=accent_color,
            opacity=0.85,
        ))

    if combined_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[year_col], y=df[combined_col] * 100,
            mode="lines+markers+text",
            name="Combined Ratio",
            line=dict(color="#E8A000", width=2.5),
            marker=dict(size=7),
            text=[f"{v:.1f}%" for v in df[combined_col] * 100],
            textposition="top center",
        ))

    # 100% reference line
    fig.add_hline(y=100, line_dash="dash", line_color="#CC2200",
                  annotation_text="100%", annotation_position="right")

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=primary_color)),
        barmode="stack",
        xaxis_title="Calendar Year",
        yaxis_title="Ratio (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        height=380,
        margin=dict(l=50, r=30, t=70, b=50),
        font=dict(family="Inter, sans-serif", size=11),
    )

    return to_html(fig, include_plotlyjs=False, full_html=False, div_id="cr_chart")


def premium_trend_chart(
    data: pd.DataFrame,
    year_col: str,
    written_col: str = "written_premium",
    earned_col: str = "earned_premium",
    primary_color: str = "#003366",
    accent_color: str = "#0099CC",
    title: str = "Written vs. Earned Premium",
) -> str:
    if not HAS_PLOTLY:
        return ""

    df = data.reset_index()
    fig = go.Figure()

    if written_col in df.columns:
        fig.add_trace(go.Bar(
            x=df[year_col], y=df[written_col] / 1e6,
            name="Written Premium ($M)",
            marker_color=primary_color,
            opacity=0.7,
        ))

    if earned_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[year_col], y=df[earned_col] / 1e6,
            mode="lines+markers",
            name="Earned Premium ($M)",
            line=dict(color=accent_color, width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=primary_color)),
        xaxis_title="Year",
        yaxis_title="Premium ($M)",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        height=340,
        margin=dict(l=50, r=30, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Inter, sans-serif", size=11),
    )
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id="prem_chart")


def loss_ratio_heatmap(
    data: pd.DataFrame,
    row_label: str = "territory",
    col_label: str = "accident_year",
    value_label: str = "incurred_loss_ratio",
    title: str = "Loss Ratio Heatmap",
    primary_color: str = "#003366",
) -> str:
    if not HAS_PLOTLY:
        return ""

    try:
        pivot = data.pivot_table(index=row_label, columns=col_label,
                                 values=value_label, aggfunc="mean")
    except Exception:
        return ""

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale=[
            [0, "#375623"],      # green — profitable
            [0.5, "#FFEB84"],    # yellow — near break-even
            [1.0, "#C00000"],    # red — unprofitable
        ],
        zmid=65,
        text=np.round(pivot.values * 100, 1),
        texttemplate="%{text}%",
        showscale=True,
        colorbar=dict(title="LR %"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=primary_color)),
        xaxis_title=col_label.replace("_", " ").title(),
        yaxis_title=row_label.replace("_", " ").title(),
        height=max(300, len(pivot) * 28 + 120),
        margin=dict(l=80, r=30, t=70, b=50),
        font=dict(family="Inter, sans-serif", size=11),
    )
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id="lr_heatmap")


def fs_trend_chart(
    data: pd.DataFrame,
    year_col: str = "year",
    freq_col: str = "frequency",
    sev_col: str = "severity",
    pp_col: str = "pure_premium",
    primary_color: str = "#003366",
    title: str = "Frequency, Severity & Pure Premium Trend",
) -> str:
    if not HAS_PLOTLY:
        return ""

    df = data.reset_index() if data.index.name else data.copy()
    fig = go.Figure()

    colors = [primary_color, "#0099CC", "#E8A000"]

    for metric, color, label, yaxis in [
        (freq_col, colors[0], "Frequency (per exposure)", "y"),
        (pp_col, colors[2], "Pure Premium ($)", "y2"),
    ]:
        if metric not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df[year_col], y=df[metric],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            yaxis=yaxis,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=primary_color)),
        xaxis_title="Accident Year",
        yaxis=dict(title="Frequency", side="left"),
        yaxis2=dict(title="Pure Premium ($)", side="right", overlaying="y"),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        height=340,
        margin=dict(l=60, r=60, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Inter, sans-serif", size=11),
    )
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id="fs_chart")


def reserve_waterfall_chart(
    origins: List[Any],
    reported: List[float],
    ibnr: List[float],
    primary_color: str = "#003366",
    accent_color: str = "#0099CC",
    title: str = "Reserve Position — Reported vs. IBNR",
) -> str:
    if not HAS_PLOTLY:
        return ""

    fig = go.Figure(data=[
        go.Bar(name="Reported Losses", x=[str(o) for o in origins],
               y=reported, marker_color=primary_color),
        go.Bar(name="IBNR", x=[str(o) for o in origins],
               y=ibnr, marker_color=accent_color),
    ])

    fig.update_layout(
        barmode="stack",
        title=dict(text=title, font=dict(size=15, color=primary_color)),
        xaxis_title="Accident Year",
        yaxis_title="Loss Amount ($)",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        height=360,
        margin=dict(l=60, r=30, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Inter, sans-serif", size=11),
    )
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id="reserve_chart")


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def df_to_html_table(
    df: pd.DataFrame,
    pct_cols: Optional[List[str]] = None,
    currency_cols: Optional[List[str]] = None,
    ldf_cols: Optional[List[str]] = None,
    highlight_total: bool = True,
    table_id: str = "data_table",
) -> str:
    """Convert a DataFrame to a styled HTML table string."""
    pct_cols = set(pct_cols or [])
    currency_cols = set(currency_cols or [])
    ldf_cols = set(ldf_cols or [])

    df_out = df.reset_index()
    rows_html = []

    # Header
    headers = "".join(
        f"<th>{str(col).replace('_', ' ').title()}</th>"
        for col in df_out.columns
    )
    rows_html.append(f"<thead><tr>{headers}</tr></thead>")

    # Body
    body_rows = []
    for _, row in df_out.iterrows():
        is_total = str(row.iloc[0]).upper() in ("TOTAL", "SUBTOTAL", "GRAND TOTAL")
        row_class = "total-row" if is_total else ""
        cells = []
        for col, val in zip(df_out.columns, row):
            if pd.isna(val) or val == "—":
                cells.append("<td class='num'>—</td>")
            elif col in pct_cols and isinstance(val, (int, float)):
                cells.append(f"<td class='num pct'>{val:.1%}</td>")
            elif col in currency_cols and isinstance(val, (int, float)):
                cells.append(f"<td class='num cur'>${val:,.0f}</td>")
            elif col in ldf_cols and isinstance(val, (int, float)):
                cells.append(f"<td class='num ldf'>{val:.4f}</td>")
            elif isinstance(val, float):
                if abs(val) < 10 and val != 0:
                    cells.append(f"<td class='num'>{val:.4f}</td>")
                else:
                    cells.append(f"<td class='num'>{val:,.0f}</td>")
            elif isinstance(val, int):
                cells.append(f"<td class='num'>{val:,}</td>")
            else:
                cells.append(f"<td>{val}</td>")
        body_rows.append(f"<tr class='{row_class}'>{''.join(cells)}</tr>")

    rows_html.append(f"<tbody>{''.join(body_rows)}</tbody>")
    return f'<table id="{table_id}" class="aa-table">{"".join(rows_html)}</table>'
