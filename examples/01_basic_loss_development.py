"""
Example 01 — Basic Loss Development
====================================
Demonstrates the core triangle workflow:
  1. Generate sample data
  2. Load into session
  3. Build triangle
  4. Develop with LDF selection
  5. Compute ultimates
  6. Export exhibit

Run from repo root:
    python examples/01_basic_loss_development.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Generate sample data if it doesn't exist
data_dir = Path("data")
if not (data_dir / "claims.csv").exists():
    print("Generating sample data...")
    from tests.fixtures.generate_sample_data import generate_all
    generate_all(output_dir="data", n_per_year=300)

# ---------------------------------------------------------------------------
# Build session
# ---------------------------------------------------------------------------
from auto_actuary import ActuarySession

session = (
    ActuarySession.from_config("config/schema.yaml")
    .load_csv("policies",   "data/policies.csv")
    .load_csv("claims",     "data/claims.csv")
    .load_csv("valuations", "data/valuations.csv")
)

print(f"\nSession: {session}")
print(f"Policies loaded: {len(session.data('policies')):,}")
print(f"Claims loaded:   {len(session.data('claims')):,}")
print(f"Valuations:      {len(session.data('valuations')):,}")

# ---------------------------------------------------------------------------
# Build and develop the incurred loss triangle
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("INCURRED LOSS DEVELOPMENT TRIANGLE — PPA")
print("="*60)

tri = session.build_triangle(lob="PPA", value="incurred_loss")
print(f"\nTriangle: {tri}")
print("\nRaw triangle ($000s):")
print((tri.triangle / 1000).round(1).to_string())

# Develop with 5-year volume-weighted LDFs + curve-fit tail
tri.develop(
    ldf_method="vw_5yr",
    tail_method="curve_fit",
    tail_curve="inverse_power",
    tail_threshold=1.005,
)

print("\nSelected LDFs:")
print(tri._selected_ldfs.round(4).to_string())
print(f"\nTail Factor: {tri._tail_factor:.5f}")

print("\nCDF-to-Ultimate by Age:")
print(tri._cdfs.round(4).to_string())

# ---------------------------------------------------------------------------
# Summary: ultimates and IBNR
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("DEVELOPMENT SUMMARY (Chain Ladder)")
print("="*60)
summ = tri.summary()
print(summ.to_string())

total_reported  = summ["reported"].sum()
total_ultimate  = summ["ultimate"].sum()
total_ibnr      = summ["ibnr"].sum()

print(f"\nTotal Reported:  ${total_reported:>15,.0f}")
print(f"Total Ultimate:  ${total_ultimate:>15,.0f}")
print(f"Total IBNR:      ${total_ibnr:>15,.0f}")
print(f"% Unreported:    {total_ibnr/total_ultimate:.2%}")

# ---------------------------------------------------------------------------
# LDF comparison table
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("LDF AVERAGING TABLE")
print("="*60)
print(tri.ldf_exhibit().round(4).to_string())

# ---------------------------------------------------------------------------
# Export exhibit
# ---------------------------------------------------------------------------
Path("output").mkdir(exist_ok=True)
out = session.triangle_exhibit(lob="PPA", value="incurred_loss",
                                output_path="output/PPA_incurred_triangle.xlsx",
                                fmt="excel")
print(f"\n✓ Excel exhibit: {out}")

out_html = session.triangle_exhibit(lob="PPA", value="incurred_loss",
                                     output_path="output/PPA_incurred_triangle.html",
                                     fmt="html")
print(f"✓ HTML exhibit:  {out_html}")
