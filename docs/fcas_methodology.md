# FCAS Actuarial Methodology Reference

This document explains the actuarial methods implemented in auto_actuary,
with references to the CAS syllabus and study materials.

---

## 1. Loss Development (Triangle Methods)

### Data Structure
A loss development triangle organizes cumulative losses by:
- **Rows** = Origin periods (accident year, policy year, or report year)
- **Columns** = Development age in months since the start of the origin period

### Link Ratio (LDF) Methods

All methods compute age-to-age link ratios:  LDF(age_from → age_to) = Losses(age_to) / Losses(age_from)

| Method | Formula | Notes |
|---|---|---|
| **Volume-Weighted** | Σ to / Σ from | Standard FCAS selection method |
| **Simple Average** | mean(individual LDFs) | Gives equal weight to each year |
| **Medial Average** | Exclude ±n extremes, then average | Reduces impact of outliers |
| **Geometric** | exp(mean(ln(LDFs))) | Multiplicatively centered |

N-year variants (5yr, 3yr) use only the most recent N origin periods.

### Tail Factor
Represents development from the last observed age to ultimate.

**Inverse Power Curve:**  LDF(t) = a × t^b (b < 0)

Fit by log-linear regression:  ln(LDF−1) = ln(a) + b × ln(t)

Tail is computed as the product of projected LDFs from last age to 360 months.

**Reference:** Friedland (2010), Chapters 7–13; CAS Exam 5 materials.

---

## 2. Reserve Methods

### Chain Ladder
The "development method" — fully data-driven.

    Ultimate(AY) = Latest Reported(AY) × CDF-to-Ultimate(latest age)
    IBNR(AY)     = Ultimate(AY) − Latest Reported(AY)

**When to use:** Mature accident years with credible development data.

### Bornhuetter-Ferguson (B-F)
A credibility blend of chain ladder and expected losses.

    % Unreported(AY) = 1 − 1 / CDF(latest age)
    IBNR(AY)         = ELR × Premium(AY) × % Unreported
    Ultimate(AY)     = Latest Reported(AY) + IBNR(AY)

**Key:** The Expected Loss Ratio (ELR) is the a-priori assumption.
auto_actuary derives ELR via Cape Cod by default.

**When to use:** Immature accident years, volatile lines, new books.

**Reference:** Bornhuetter & Ferguson (1972), PCAS.

### Cape Cod
Like B-F, but derives the ELR from the data itself.

    Used-Up Premium(AY) = Premium(AY) × (1 − 1/CDF)
    ELR_CC              = Σ Latest Reported / Σ Used-Up Premium
    IBNR(AY)            = ELR_CC × Premium(AY) × % Unreported

**When to use:** When you don't have a credible a-priori ELR.

### Benktander
One iteration of B-F using B-F ultimate as the new "expected".
Gives more weight to actual data than B-F; converges to chain ladder.

    Implied ELR(AY) = BF Ultimate(AY) / Premium(AY)
    IBNR_BK(AY)     = Implied ELR(AY) × Premium(AY) × % Unreported

**Reference:** Mack (2000), ASTIN.

---

## 3. Ratemaking

### On-Level Premium (Parallelogram Method)
Adjusts historical earned premium to current rate level.

    Current Rate Index (CRI)     = Product of (1 + rate_change) over all filings
    Average Rate Level Factor    = Weighted average CRI during policy year's earning period
    On-Level Factor              = CRI / ARLF
    On-Level EP                  = Historical EP × On-Level Factor

**Reference:** Werner & Modlin (2016), Chapter 4.

### Loss Trend
Log-linear regression of historical pure premium on time.

    ln(PP_t) = a + b × t
    Annual Trend = e^b
    Trend Factor = Annual Trend ^ (trend_period_years)

The trend period runs from the **effective date of historical rates**
to the **midpoint of the prospective policy period** (typically 1–2 years ahead).

**Reference:** Werner & Modlin (2016), Chapter 6.

### Rate Indication

    Permissible Loss Ratio = 1 − V − Fixed Expense Ratio − Q

where:
- V = Variable expense ratio (commissions, taxes, fees)
- Q = Target profit margin

    Projected Loss Ratio = Trended Ultimate Losses / On-Level Earned Premium

    Indicated Change = (Projected Loss Ratio / Permissible Loss Ratio) − 1

### Credibility
Classical credibility (full credibility at 1,082 claims, ±5%, 90% CI):

    Z = min( √(n / n_full), 1.0 )    where n_full = 1,082

    Credibility-Weighted Indication = Z × Indicated + (1−Z) × Complement

---

## 4. Frequency / Severity

    Frequency  = Claim Count / Earned Exposure
    Severity   = Paid (or Incurred) Losses / Claim Count
    Pure Premium = Frequency × Severity = Total Loss / Earned Exposure

Trend analysis fits separate log-linear models to each component.
This allows actuaries to distinguish between:
- Social inflation (severity increases independent of accident frequency)
- Coverage broadening (frequency increases, severity stable)
- Systemic issues (both trending)

---

## 5. Catastrophe Analysis

CAT losses are excluded from trend analysis because:
1. CAT frequency and severity are not predictable from historical patterns
2. CAT losses are highly correlated across policies (not independent)

**Expected CAT Load (Simple Historical):**

    Expected CAT Load = Average Annual CAT Losses / Average Earned Premium

For production use, supplement with output from a cat model
(AIR Touchstone, RMS RiskLink, Verisk Catastrophe Risk Management).

---

## References

| Document | Source |
|---|---|
| "Estimating Unpaid Claims Using Basic Techniques" | Friedland (2010), CAS |
| "Basic Ratemaking" | Werner & Modlin (2016), CAS |
| "The Actuary and IBNR" | Bornhuetter & Ferguson (1972), PCAS |
| "Credible Claims Reserves: The Benktander Method" | Mack (2000), ASTIN |
| CAS Exam 5 Study Materials | CAS website |
| CAS Exam 6 Study Materials | CAS website |
| "Loss Models: From Data to Decisions" | Klugman et al. (4th ed.) |
