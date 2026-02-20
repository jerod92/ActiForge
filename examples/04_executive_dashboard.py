"""
Example 04 — Executive Dashboard
==================================
Generates the full executive HTML dashboard — the management-facing output
that synthesizes all analytics into a single polished report.

Open the generated HTML file in any browser.

Run from repo root:
    python examples/04_executive_dashboard.py
"""

import webbrowser
from pathlib import Path

data_dir = Path("data")
if not (data_dir / "claims.csv").exists():
    from tests.fixtures.generate_sample_data import generate_all
    print("Generating sample data (this takes ~30 seconds for 300 policies/yr)…")
    generate_all(output_dir="data", n_per_year=300)

from auto_actuary import ActuarySession

session = (
    ActuarySession.from_config("config/schema.yaml")
    .load_csv("policies",     "data/policies.csv")
    .load_csv("claims",       "data/claims.csv")
    .load_csv("valuations",   "data/valuations.csv")
    .load_csv("rate_changes", "data/rate_changes.csv")
    .load_csv("expenses",     "data/expenses.csv")
)

print("\nBuilding executive dashboard…")
out = session.exec_dashboard(
    output_path="output/dashboard.html",
    lob="PPA",
)

print(f"\n✓ Dashboard: {out.resolve()}")
print("\nOpening in browser…")

try:
    webbrowser.open(f"file://{out.resolve()}")
except Exception:
    print("(Could not auto-open browser — open the HTML file manually.)")
