"""
Example 02 — IBNR Reserve Analysis
=====================================
Demonstrates the full reserve workflow:
  Chain Ladder, Bornhuetter-Ferguson, Cape Cod, Benktander

Run from repo root:
    python examples/02_reserve_analysis.py
"""

from pathlib import Path
import pandas as pd

data_dir = Path("data")
if not (data_dir / "claims.csv").exists():
    from tests.fixtures.generate_sample_data import generate_all
    generate_all(output_dir="data", n_per_year=300)

from auto_actuary import ActuarySession

session = (
    ActuarySession.from_config("config/schema.yaml")
    .load_csv("policies",   "data/policies.csv")
    .load_csv("claims",     "data/claims.csv")
    .load_csv("valuations", "data/valuations.csv")
)

# ---------------------------------------------------------------------------
# Reserve analysis
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("IBNR RESERVE ANALYSIS — PPA")
print("="*60)

analysis = session.reserve_analysis(lob="PPA")

print(f"\nAvailable methods: {analysis.available_methods}")
print(f"\nSelected method: {analysis.selected().method}")
print(f"Selected ELR:    {analysis.selected().elr:.4f}" if analysis.selected().elr else "")

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
print("\n--- Reserve Comparison ---")
comp = analysis.comparison_table()
print(comp.to_string())

print(f"\n{'Method':<30} {'Total IBNR':>15} {'ELR':>8}")
print("-" * 55)
for m in analysis.available_methods:
    res = analysis.result(m)
    elr_str = f"{res.elr:.4f}" if res.elr else "   —  "
    print(f"{m.replace('_',' ').title():<30} ${res.total_ibnr:>14,.0f}  {elr_str}")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
Path("output").mkdir(exist_ok=True)
out = session.reserve_exhibit(
    lob="PPA",
    output_path="output/PPA_reserve.xlsx",
    fmt="excel",
)
print(f"\n✓ Reserve exhibit: {out}")

out_html = session.reserve_exhibit(
    lob="PPA",
    output_path="output/PPA_reserve.html",
    fmt="html",
)
print(f"✓ HTML exhibit:    {out_html}")
