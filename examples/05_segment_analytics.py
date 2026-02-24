"""
Example 05 — Segment Analytics Dashboard
==========================================
Demonstrates the full segment analytics workflow:
  1. Generate sample data (or load your own)
  2. Build an ActuarySession
  3. Run segment_analytics() to get time-series metrics per segment
  4. Generate the interactive Segment Analytics Dashboard HTML

The dashboard answers:
  - Which territories / classes retain the most profitable business?
  - How are loss ratios trending per segment?
  - Where is premium growing vs. shrinking?
  - What is the estimated Customer Lifetime Value per segment?

Segment dimensions are configured in config/schema.yaml under:
    segmentation:
        geo_segments:    [territory, ...]
        market_segments: [class_code, agent_id, ...]

Add your own DB column names to those lists to automatically create
additional tabs in the dashboard.

Run from repo root:
    python examples/05_segment_analytics.py
"""

import webbrowser
from pathlib import Path

data_dir = Path("data")
if not (data_dir / "policies.csv").exists():
    from tests.fixtures.generate_sample_data import generate_all
    print("Generating sample data…")
    generate_all(output_dir="data", n_per_year=500)

from auto_actuary import ActuarySession

session = (
    ActuarySession.from_config("config/schema.yaml")
    .load_csv("policies",     "data/policies.csv")
    .load_csv("claims",       "data/claims.csv")
    .load_csv("valuations",   "data/valuations.csv")
    .load_csv("rate_changes", "data/rate_changes.csv")
    .load_csv("expenses",     "data/expenses.csv")
)

# -----------------------------------------------------------------------
# Optional: explore individual segment metrics programmatically
# -----------------------------------------------------------------------
sa = session.segment_analytics(lob="PPA")

print("\n--- Retention by Territory ---")
ret = sa.retention_trend("territory")
if not ret.empty:
    print(ret.reset_index()[["period","segment_value","expiring_count","retained_count","retention_rate"]].to_string())

print("\n--- CLV by Class Code ---")
clv = sa.clv_by_segment("class_code")
if not clv.empty:
    print(clv[["avg_annual_premium","retention_rate","expected_tenure_years","loss_ratio","estimated_clv"]].round(2).to_string())

print("\n--- Loss Ratio Trend by Territory ---")
lr = sa.loss_trend("territory")
if not lr.empty:
    print(lr.reset_index()[["period","segment_value","incurred_loss","loss_ratio"]].to_string())

# -----------------------------------------------------------------------
# Generate the full dashboard
# -----------------------------------------------------------------------
Path("output").mkdir(exist_ok=True)
out = session.segment_dashboard(
    output_path="output/segment_dashboard.html",
    lob="PPA",
)

print(f"\n✓ Segment Dashboard: {out.resolve()}")
print("Opening in browser…")
try:
    webbrowser.open(f"file://{out.resolve()}")
except Exception:
    print("(Could not auto-open browser — open the HTML file manually.)")
