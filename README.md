# ActiForge

> **Actuarial mathematics. Strategic output.**

A Python analytics platform for P&C insurance carriers. Built for actuaries and
data scientists who need to move fast — and produce work that looks good in the boardroom.

---

## What It Does

auto_actuary covers the core quantitative workflows of a P&C actuarial department:

| Domain | Analytics | Output |
|---|---|---|
| **Reserving** | Chain Ladder, Bornhuetter-Ferguson, Cape Cod, Benktander | Excel exhibits, HTML reserve pages |
| **Ratemaking** | On-level premium (parallelogram), loss trend, rate indication, IRPM | Rate indication exhibit |
| **Development** | Triangle construction, LDF selection (6 methods × 3 periods), tail fitting | Triangle exhibit (all methods) |
| **Freq/Severity** | F/S decomposition, log-linear trend, territorial relativities | F/S summary, trend table |
| **Profitability** | Loss ratio (paid/incurred/ultimate), combined ratio, cohort analysis | Dashboard, LOB table |
| **Catastrophe** | CAT vs. non-CAT split, event analysis, territory concentration, expected load | CAT summary page |
| **Portfolio** | Market breakdown, product mix, HHI concentration, retention analysis | Segment tables, mix charts |
| **Cause of Loss** | Cause-of-loss F/S breakdown, correlation analysis | Cause-of-loss summary |
| **Scenario** | What-if modeling, compound GLM, multi-year trend projection, stress testing | Scenario comparison report |
| **Time Series** | Multi-snapshot tracking, metric trends across evaluation dates | Time-series tables |
| **Executive** | KPI cards, combined ratio trend, reserve waterfall, premium trend | Self-contained HTML dashboard |

---

## Philosophy

**You write the SQL. We write the math.**

auto_actuary does not know what your database looks like.
Instead, it:

1. Specifies exactly what data it needs (see `queries/README.md`)
2. Lets you write the SQL that extracts it from your schema
3. Maps your column names to canonical aliases via `config/schema.yaml`
4. Does all the actuarial math on the canonical form

This means zero vendor lock-in, zero ORM magic, and full auditability.

---

## Quick Start

```bash
pip install -e ".[dev]"

# 1. Generate sample data (realistic synthetic P&C data)
python tests/fixtures/generate_sample_data.py --output-dir data --n-per-year 500

# 2. Run the full suite and open the dashboard
auto-actuary all --lob PPA --output output/

# 3. Or run individual analyses
auto-actuary triangle   --lob PPA
auto-actuary reserve    --lob PPA
auto-actuary ratemaking --lob PPA --trend-factor 1.04
auto-actuary dashboard  --lob PPA

# 4. Validate your data against the expected schema
auto-actuary validate --data-dir data/
```

---

## Python API

```python
from auto_actuary import ActuarySession

# 1. Load data
session = (
    ActuarySession.from_config("config/schema.yaml")
    .load_csv("policies",     "data/policies.csv")
    .load_csv("claims",       "data/claims.csv")
    .load_csv("valuations",   "data/valuations.csv")
    .load_csv("rate_changes", "data/rate_changes.csv")
)

# 2. Triangle development
tri = session.build_triangle(lob="PPA", value="incurred_loss")
tri.develop(ldf_method="vw_5yr", tail_method="curve_fit")
print(tri.summary())

# 3. Reserve analysis (Chain Ladder, B-F, Cape Cod, Benktander)
analysis = session.reserve_analysis(lob="PPA")
print(analysis.comparison_table())
print(f"Selected IBNR: ${analysis.total_ibnr():,.0f}")

# 4. Rate indication
indication = session.rate_indication(lob="PPA", trend_factor=1.04)
result = indication.compute()
print(f"Indicated change: {result.indicated_pct}")
print(f"Credibility-weighted: {result.credibility_weighted_pct}")

# 5. Frequency / Severity
fs = session.freq_severity(lob="PPA")
trends = fs.fit_trends()
print(trends["pure_premium"].trend_table())

# 6. Profitability
lr = session.loss_ratios(lob="PPA")
print(lr.by_accident_year())
cr = session.combined_ratio(lob="PPA")
print(cr.by_year())

# 7. Catastrophe analysis
cat = session.cat_analysis(lob="PPA")
print(cat.split_by_year())
print(cat.expected_cat_load())

# 8. Portfolio analysis
mix = session.product_mix(lob="PPA")
print(mix.summary())
retention = session.retention_analysis(lob="PPA")
print(retention.by_year())

# 9. Scenario modeling
report = session.scenario_report(lob="PPA", output_path="output/scenarios.html")

# 10. Export everything
session.triangle_exhibit(lob="PPA",      output_path="output/tri.xlsx")
session.reserve_exhibit(lob="PPA",       output_path="output/res.xlsx")
session.rate_indication_exhibit(lob="PPA", output_path="output/rate.xlsx")
session.exec_dashboard(output_path="output/dashboard.html", lob="PPA")
```

---

## File Tree

```
auto_actuary/
├── config/
│   ├── schema.yaml                  # DB column → canonical alias mapping
│   ├── actuarial_assumptions.yaml   # LDF methods, trend, expense assumptions
│   └── lines_of_business.yaml       # LOB definitions and coverage codes
│
├── queries/
│   ├── README.md                    # Query output specifications (READ THIS)
│   ├── valuations.sql.example       # Template SQL for loss valuations
│   └── policies.sql.example         # Template SQL for policies
│
├── auto_actuary/
│   ├── core/
│   │   ├── config.py                # Schema + assumption loader
│   │   ├── data_loader.py           # CSV / SQL / DataFrame ingestion
│   │   └── session.py               # Top-level orchestration object
│   │
│   ├── analytics/
│   │   ├── triangles/
│   │   │   ├── development.py       # LossTriangle + 6 LDF methods + CDF chain
│   │   │   └── tail.py              # Inverse power & exponential tail fitting
│   │   ├── reserves/
│   │   │   ├── ibnr.py              # CL, B-F, Cape Cod, Benktander
│   │   │   └── adequacy.py          # Reserve adequacy testing
│   │   ├── ratemaking/
│   │   │   ├── on_level.py          # Parallelogram on-level premium
│   │   │   ├── trend.py             # Log-linear loss/premium trend
│   │   │   ├── indicated_rate.py    # Rate indication with credibility
│   │   │   └── irpm.py              # Individual Risk Premium Modification
│   │   ├── frequency_severity/
│   │   │   └── analysis.py          # F/S decomposition + relativities
│   │   ├── profitability/
│   │   │   ├── loss_ratio.py        # Paid/incurred/ultimate LR by slice
│   │   │   ├── combined_ratio.py    # Combined ratio trend
│   │   │   └── cohort.py            # Policy vintage profitability
│   │   ├── catastrophe/
│   │   │   └── cat_analysis.py      # CAT vs. non-CAT split + event analysis
│   │   ├── cause_of_loss/
│   │   │   └── analysis.py          # Cause-of-loss F/S breakdown + correlation
│   │   ├── retention/
│   │   │   └── retention.py         # Policy/account retention rates
│   │   ├── portfolio/
│   │   │   ├── market_breakdown.py  # Hierarchical market segmentation
│   │   │   └── product_mix.py       # Premium mix + HHI concentration
│   │   ├── speculative/
│   │   │   ├── scenario_engine.py   # What-if scenario modeling
│   │   │   ├── glm_models.py        # Compound GLM for rate optimization
│   │   │   ├── trend_projector.py   # Multi-year F/S trend projection
│   │   │   └── categorical.py       # Categorical variable encoding
│   │   └── time_series/
│   │       └── manager.py           # Multi-snapshot time-series tracking
│   │
│   ├── reports/
│   │   ├── renderers/
│   │   │   ├── excel.py             # Styled openpyxl workbook writer
│   │   │   └── html.py              # Plotly chart builders + table formatter
│   │   ├── executive/
│   │   │   ├── dashboard.py         # Self-contained HTML dashboard
│   │   │   └── scenario_report.py   # Scenario comparison HTML report
│   │   └── actuarial/
│   │       ├── triangle_exhibit.py  # Triangle + LDF + CDF exhibit
│   │       ├── reserve_exhibit.py   # Reserve comparison exhibit
│   │       └── rate_indication.py   # Rate indication exhibit
│   │
│   └── cli/
│       └── main.py                  # typer CLI: triangle/reserve/ratemaking/dashboard/all
│
├── tests/
│   ├── fixtures/
│   │   └── generate_sample_data.py  # Synthetic P&C data generator
│   ├── test_triangles.py            # LDF math + tail fitting tests
│   ├── test_reserves.py             # B-F / Cape Cod algebraic invariant tests
│   ├── test_ratemaking.py           # Trend + rate indication tests
│   └── test_speculative.py          # Scenario engine + GLM tests
│
├── examples/
│   ├── 01_basic_loss_development.py
│   ├── 02_reserve_analysis.py
│   ├── 03_rate_indication.py
│   └── 04_executive_dashboard.py
│
└── docs/
    └── fcas_methodology.md          # Formulas, derivations, and references
```

---

## Configuration

### `config/schema.yaml`

Maps your database column names to the canonical aliases used in the codebase.
Seven canonical tables are defined: `policies`, `transactions`, `claims`,
`valuations`, `coverages`, `rate_changes`, and `expenses`.

```yaml
policies:
  policy_id:      policy_id      # if your DB already uses canonical names, leave as-is
  effective_date: eff_date       # maps "eff_date" in DB → "effective_date" in code
  written_premium: premium_amt   # maps "premium_amt" → "written_premium"
```

### `config/actuarial_assumptions.yaml`

Controls all actuarial parameters — every value is overridable programmatically:

```yaml
triangles:
  selected_ldf_method: vw_5yr    # volume-weighted 5-year
  tail_method: curve_fit

ratemaking:
  variable_expense_ratio: 0.25
  target_profit_margin: 0.05

reserves:
  primary_method: bornhuetter_ferguson
  elr_source: cape_cod
```

### `config/lines_of_business.yaml`

Defines LOBs, coverage codes, and transaction types:

```yaml
lines_of_business:
  PPA:
    label: "Personal Auto"
    short_code: PPA
    exposure_unit: car_year
    coverages: [BI, PD, COMP, COLL, MED, PIP, UM, UIM]
```

### Loading SQL from your database

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host:5432/prod_db")
session.load_sql("valuations", open("queries/valuations.sql").read(), engine)
```

---

## Supported Data Sources

| Source | How |
|---|---|
| CSV files | `session.load_csv("table", "path/to/file.csv")` |
| Pandas DataFrame | `session.load_dataframe("table", df)` |
| SQL query (any DB) | `session.load_sql("table", sql_string, sqlalchemy_engine)` |
| SQL file | `session.load_sql_file("table", "queries/valuations.sql", engine)` |

---

## Business Functionalities

### For Actuaries (FCAS Level)

| Feature | Detail |
|---|---|
| **Triangle Development** | 6 LDF methods × all/5yr/3yr periods; medial average; geometric |
| **Tail Fitting** | Inverse-power and exponential curve fitting; benchmark fallback |
| **Reserve Methods** | Chain Ladder, Bornhuetter-Ferguson, Cape Cod, Benktander — all four, side-by-side |
| **ELR Derivation** | Cape Cod ELR automatically derived from data; override capability |
| **Reserve Adequacy** | Held vs. actuarial; calendar-year development; redundancy/deficiency |
| **On-Level Premium** | Parallelogram method; cumulative rate index; ARLF by year |
| **Loss Trend** | Log-linear regression; R², p-value, Durbin-Watson; multi-period comparison |
| **Credibility** | Classical Z = √(n/1082); credibility-weighted indication |
| **IRPM** | Individual Risk Premium Modification; adequacy/efficiency analysis; Gini coefficient |
| **F/S Decomposition** | Frequency/severity trend by coverage, territory, class |
| **Relativities** | Territorial and class relativities with credibility |
| **CAT Analysis** | Event analysis; territory concentration; expected CAT load |
| **Cause of Loss** | Cause-of-loss F/S breakdown; inter-peril correlation |
| **Cohort Analysis** | Policy vintage P&L; development of loss ratio over time |
| **Retention** | Policy and account retention rates by year and segment |
| **Portfolio Mix** | Premium/exposure mix analysis; HHI concentration index; loss ratio by segment |
| **Market Breakdown** | Hierarchical market segmentation analysis |
| **Scenario Modeling** | What-if rate, frequency, severity adjustments; stress testing with bootstrap CIs |
| **Compound GLM** | GLM-based rate optimization from segment-level data |
| **Trend Projection** | Multi-year frequency and severity trend projections |
| **Time Series** | Multi-snapshot metric tracking across evaluation dates |

### For Executives / Management

| Feature | Detail |
|---|---|
| **HTML Dashboard** | Single-file, browser-ready; no server required |
| **KPI Cards** | Written Premium, Earned Premium, Loss Ratio, Combined Ratio, IBNR, Rate Indication |
| **Combined Ratio Trend** | Stacked bar (loss + expense) + combined ratio line + 100% reference |
| **Premium Growth** | Written vs. Earned Premium trend chart |
| **Reserve Waterfall** | Reported vs. IBNR stacked bar by accident year |
| **Loss Cost Trend** | Frequency and pure premium dual-axis trend |
| **Portfolio Table** | Loss ratios by LOB, territory, or coverage |
| **Cohort Table** | Vintage profitability — loss ratio and UW profit/loss |
| **Scenario Report** | Side-by-side scenario comparison with combined ratio impact |
| **Excel Exhibits** | Professionally formatted with cover page, navy/teal theme, auto-widths |

---

## FCAS Methodology Reference

See [`docs/fcas_methodology.md`](docs/fcas_methodology.md) for formulas,
derivations, and CAS study note references.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Tests cover LDF math, tail factor behavior, B-F/Cape Cod algebraic invariants,
rate indication formula, trend fitting accuracy, and scenario engine modeling.

---

## License

MIT
