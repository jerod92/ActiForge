"""
Example 03 — Rate Indication
==============================
Demonstrates the full ratemaking workflow:
  On-Level Premium → Loss Trend → Indicated Change

Run from repo root:
    python examples/03_rate_indication.py
"""

from pathlib import Path

data_dir = Path("data")
if not (data_dir / "claims.csv").exists():
    from tests.fixtures.generate_sample_data import generate_all
    generate_all(output_dir="data", n_per_year=300)

from auto_actuary import ActuarySession

session = (
    ActuarySession.from_config("config/schema.yaml")
    .load_csv("policies",     "data/policies.csv")
    .load_csv("claims",       "data/claims.csv")
    .load_csv("valuations",   "data/valuations.csv")
    .load_csv("rate_changes", "data/rate_changes.csv")
)

# ---------------------------------------------------------------------------
# Frequency / Severity trend
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("FREQUENCY / SEVERITY ANALYSIS — PPA")
print("="*60)

fs = session.freq_severity(lob="PPA")
fs_tbl = fs.fs_table()
print("\nF/S by Accident Year:")
print(fs_tbl.round(4).to_string())

# Fit trends
trends = fs.fit_trends(periods=[3, 5])
print("\nTrend Summary:")
for metric, ta in trends.items():
    print(f"\n  {metric.title()}:")
    print(ta.trend_table().round(4).to_string(index=False))

# Select pure premium trend
pp_trend = trends.get("pure_premium")
if pp_trend:
    sel_trend = pp_trend.select("5yr") or pp_trend.select("all")
    print(f"\n  Selected PP trend: {sel_trend}")

# ---------------------------------------------------------------------------
# Rate indication
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("RATE INDICATION — PPA")
print("="*60)

ind_obj = session.rate_indication(
    lob="PPA",
    trend_factor=1.04,          # 4% cumulative loss trend
    variable_expense_ratio=0.25,
    fixed_expense_ratio=0.05,
    target_profit_margin=0.05,
)
result = ind_obj.compute()
print(f"\n{result}")

print(f"\n{'Item':<35} {'Value':>12}")
print("-" * 48)
print(f"{'On-Level Earned Premium':<35} ${result.on_level_premium:>11,.0f}")
print(f"{'Trended Ultimate Loss':<35} ${result.trended_ultimate_loss:>11,.0f}")
print(f"{'Projected Loss Ratio':<35} {result.projected_loss_ratio:>12.4f}")
print(f"{'Permissible Loss Ratio':<35} {result.permissible_loss_ratio:>12.4f}")
print(f"{'':35} {'':12}")
print(f"{'INDICATED CHANGE':<35} {result.indicated_pct:>12}")
print(f"{'Credibility (Z)':<35} {result.credibility:>12.4f}")
print(f"{'CREDIBILITY-WEIGHTED CHANGE':<35} {result.credibility_weighted_pct:>12}")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
Path("output").mkdir(exist_ok=True)
out = session.rate_indication_exhibit(
    lob="PPA",
    output_path="output/PPA_rate_indication.xlsx",
    fmt="excel",
)
print(f"\n✓ Rate indication exhibit: {out}")
