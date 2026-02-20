# Query Specifications

auto_actuary requires certain data to be available but **does not write SQL for you**.
Instead, it specifies exactly what the output of your queries must look like —
column names, data types, and grain.  You write the SQL that fits your schema;
the library reads the result.

## How to Use

1. Write a `.sql` file using the specifications below.
2. Execute it against your database.
3. Save the result as CSV, or pass a live `sqlalchemy` engine to `session.load_sql()`.

```python
from sqlalchemy import create_engine
from auto_actuary import ActuarySession

engine  = create_engine("postgresql://user:pass@host/db")
session = ActuarySession.from_config("config/schema.yaml")
session.load_sql("valuations", open("queries/valuations.sql").read(), engine)
```

> **Important:** Map your actual column names to the canonical aliases in
> `config/schema.yaml`.  The loader will rename them automatically.
> Alternatively, alias your columns in SQL to match the canonical names directly.

---

## Table Specifications

### 1. `policies`
**Grain:** One row per policy term

| Canonical Column   | Type      | Required | Description |
|--------------------|-----------|----------|-------------|
| `policy_id`        | string    | YES      | Unique internal policy key |
| `policy_number`    | string    | no       | Display identifier |
| `written_date`     | date      | YES      | Date transaction was recorded |
| `effective_date`   | date      | YES      | Coverage start date |
| `expiration_date`  | date      | YES      | Coverage end date |
| `cancel_date`      | date      | no       | Date cancelled (null if in-force) |
| `line_of_business` | string    | YES      | LOB code: PPA, HO, CA, GL, etc. |
| `sub_line`         | string    | no       | Sub-classification |
| `territory`        | string    | YES      | State/territory code |
| `class_code`       | string    | YES      | Underwriting class |
| `written_premium`  | float     | YES      | Written premium for this term |
| `written_exposure` | float     | YES      | Exposure units (car-years, etc.) |
| `transaction_type` | string    | YES      | NB, RN, EN, XL, CN, RE |

---

### 2. `transactions`
**Grain:** One row per premium transaction (endorsements, cancellations, etc.)

| Canonical Column   | Type   | Required | Description |
|--------------------|--------|----------|-------------|
| `transaction_id`   | string | YES      | Unique transaction key |
| `policy_id`        | string | YES      | Links to policies |
| `transaction_date` | date   | YES      | Date recorded |
| `effective_date`   | date   | YES      | Economic effective date |
| `transaction_type` | string | YES      | NB, RN, EN, XL, CN |
| `written_premium`  | float  | YES      | Transaction premium (+ or -) |
| `written_exposure` | float  | no       | Exposure delta |
| `line_of_business` | string | YES      | LOB code |
| `territory`        | string | no       | Territory code |

---

### 3. `claims`
**Grain:** One row per claim (not per coverage — join to coverages if needed)

| Canonical Column   | Type    | Required | Description |
|--------------------|---------|----------|-------------|
| `claim_id`         | string  | YES      | Unique claim key |
| `policy_id`        | string  | YES      | Links to policies |
| `accident_date`    | date    | YES      | Date of loss |
| `report_date`      | date    | YES      | Date claim was first reported |
| `close_date`       | date    | no       | Date closed (null if open) |
| `coverage_code`    | string  | YES      | BI, PD, COMP, COLL, MED, etc. |
| `line_of_business` | string  | YES      | LOB code |
| `territory`        | string  | no       | Territory of loss |
| `cause_of_loss`    | string  | no       | Cause code |
| `claim_status`     | string  | YES      | O=Open, C=Closed, R=Reopened |
| `is_catastrophe`   | int     | no       | 1 if CAT event, 0 otherwise |
| `cat_code`         | string  | no       | Event identifier (e.g. "IRMA17") |

---

### 4. `valuations`  ← **Most Critical**
**Grain:** One row per (claim_id, valuation_date)

This is the key table for triangle development.  Each row is a snapshot of a
claim's financials at a specific evaluation date.  Typically you'll have
year-end snapshots (12/31/2018, 12/31/2019, …).

| Canonical Column   | Type   | Required | Description |
|--------------------|--------|----------|-------------|
| `claim_id`         | string | YES      | Links to claims |
| `valuation_date`   | date   | YES      | Snapshot date |
| `paid_loss`        | float  | YES      | Cumulative paid losses |
| `case_reserve`     | float  | YES      | Outstanding case reserve |
| `incurred_loss`    | float  | no       | paid + case (computed if absent) |
| `paid_alae`        | float  | no       | Cumulative paid ALAE |
| `case_alae`        | float  | no       | Case ALAE reserve |
| `paid_count`       | int    | no       | 1 if closed at this eval, else 0 |
| `open_count`       | int    | no       | 1 if open at this eval, else 0 |

**SQL template:**
```sql
-- valuations.sql
-- Returns all year-end loss snapshots for triangle development.
-- Adjust table/column names to match your schema.

SELECT
    c.claim_id,
    CAST(v.val_date AS DATE)             AS valuation_date,
    COALESCE(v.paid_loss, 0)             AS paid_loss,
    COALESCE(v.case_reserve, 0)          AS case_reserve,
    COALESCE(v.paid_loss, 0)
      + COALESCE(v.case_reserve, 0)      AS incurred_loss,
    COALESCE(v.paid_alae, 0)             AS paid_alae,
    COALESCE(v.case_alae, 0)             AS case_alae,
    CASE WHEN v.status = 'C' THEN 1
         ELSE 0 END                       AS paid_count,
    CASE WHEN v.status = 'O' THEN 1
         ELSE 0 END                       AS open_count
FROM
    your_schema.claim_valuations v
    JOIN your_schema.claims c ON c.id = v.claim_id
WHERE
    EXTRACT(MONTH FROM v.val_date) = 12  -- year-end snapshots only
ORDER BY
    c.claim_id, v.val_date
;
```

---

### 5. `rate_changes`
**Grain:** One row per rate change filing

| Canonical Column    | Type   | Required | Description |
|---------------------|--------|----------|-------------|
| `effective_date`    | date   | YES      | When rate change takes effect |
| `line_of_business`  | string | YES      | LOB code |
| `territory`         | string | no       | Territory (blank = statewide) |
| `class_code`        | string | no       | Class (blank = all classes) |
| `rate_change_pct`   | float  | YES      | Decimal: 0.05 = +5%, -0.02 = -2% |

---

### 6. `expenses`
**Grain:** One row per (calendar_year, line_of_business, expense_type)

| Canonical Column    | Type   | Required | Description |
|---------------------|--------|----------|-------------|
| `calendar_year`     | int    | YES      | Calendar year |
| `line_of_business`  | string | YES      | LOB code |
| `expense_type`      | string | YES      | Commissions, Taxes, G&A, ULAE, ALAE |
| `amount`            | float  | YES      | Expense amount |
| `written_premium`   | float  | no       | WP denominator for ratio |
| `earned_premium`    | float  | no       | EP denominator for ratio |

---

## Example Files

See `.sql.example` files in this directory for ready-to-adapt SQL templates.
