# MIP to SAM Conversion Guide

## Overview

This guide explains how to convert a standard Input-Output Matrix (MIP) from national accounts into a complete Social Accounting Matrix (SAM) compatible with PEP CGE models using the `equilibria` package.

## Quick Visual Comparison: MIP vs SAM

### MIP (Input-Output Matrix) - What You Have
```
Simple structure: Sectors + aggregate VA + final demand

           agr   ind   ser │  HH   GOV   INV   EXP
        ────────────────────┼──────────────────────
agr       20    10    15   │  30    10     5     5
ind       30    40    25   │  35    10    10    10
ser       20    30    20   │  30    15    10     5
        ────────────────────┼──────────────────────
VA        40    50    40   │   0     0     0     0  ← AGGREGATE (no detail)
IMP       10    20    20   │   0     0     0     0

Missing:
❌ No factor breakdown (labor vs capital)
❌ No institution accounts (households, firms, govt)
❌ No tax flows
❌ No income distribution
```

### SAM (Social Accounting Matrix) - What You Need for CGE
```
Extended structure: Factors + institutions + fiscal flows

           agr  ind  ser │  L    K  │  hh   gvt  firm  row │ inv   X
        ───────────────────────────────────────────────────────────
I.agr      20   10   15 │  0    0  │  30    0     0     0  │  0    0
I.ind      30   40   25 │  0    0  │  35    0     0     0  │  0    0
I.ser      20   30   20 │  0    0  │  30    0     0     0  │  0    0
        ───────────────────────────────────────────────────────────
L          26  32.5  26 │  0    0  │   0    0     0     0  │  0    0  ← Labor factor
K          14  17.5  14 │  0    0  │   0    0     0     0  │  0    0  ← Capital factor
        ───────────────────────────────────────────────────────────
AG.hh       0    0    0 │ 80.8 22.7│   0    0     0     0  │  0    0  ← Household
AG.gvt      0    0    0 │ 4.2  2.3 │   0    0     0     0  │  0    0  ← Government
AG.firm     0    0    0 │  0  20.5 │   0    0     0     0  │  0    0  ← Firms
AG.row     10   20   20 │  0    0  │   0    0     0     0  │  0    0  ← Rest of world
AG.ti       5    6    5 │  0    0  │   0    0     0     0  │  0    0  ← Indirect taxes
        ───────────────────────────────────────────────────────────
OTH.inv     0    0    0 │  0    0  │  20    5    20     0  │  0    0  ← Investment
X.agr       0    0    0 │  0    0  │   0    0     0     0  │  0    5  ← Exports
X.ind       0    0    0 │  0    0  │   0    0     0     0  │  0   10
X.ser       0    0    0 │  0    0  │   0    0     0     0  │  0    5

Complete:
✅ Factors disaggregated (L, K)
✅ Institutions explicit (hh, gvt, firm, row)
✅ Tax accounts (ti, tm)
✅ Income flows (factor → institutions)
✅ Savings-investment closure
```

**The Gap**: `run_mip_to_sam()` fills this gap by disaggregating and extending the MIP structure using external parameters.

## What is a MIP?

A **Matrix Insumo-Producto (MIP)** or Input-Output table is a standard economic accounting framework that shows:
- **Intermediate flows**: How sectors use inputs from other sectors
- **Value Added**: Aggregate payments to factors (without breakdown)
- **Final Demand**: Consumption by households, government, investment, and exports
- **Imports**: External supply of goods

### Example: Simple 3-Sector MIP

Here's what a minimal MIP looks like (values in millions):

```
                    agr    ind    ser  │  HH   GOV   INV   EXP
────────────────────────────────────────┼──────────────────────
agr (agriculture)    20     10     15  │  30    10     5     5
ind (industry)       30     40     25  │  35    10    10    10
ser (services)       20     30     20  │  30    15    10     5
────────────────────────────────────────┼──────────────────────
Valor Agregado       40     50     40  │   0     0     0     0
Importaciones        10     20     20  │   0     0     0     0
```

**Reading the table:**
- **Columns** (agr, ind, ser): Sector inputs - what each sector buys
- **Rows** (agr, ind, ser): Sector outputs - what each sector sells
- **Valor Agregado (VA)**: Payments to labor + capital (aggregated)
- **Final Demand columns**: HH=Households, GOV=Government, INV=Investment, EXP=Exports

**Key limitation**: VA is aggregated - we don't know how much goes to labor vs capital, or how factor income is distributed to households, firms, etc.

## Why Convert to SAM?

CGE models like PEP require a complete SAM with:
- **Explicit factors**: Labor (L) and Capital (K) separated
- **Institutions**: Households, firms, government, rest-of-world
- **Fiscal flows**: Taxes and transfers
- **Balanced accounts**: Row sums = column sums

### Example: What the SAM Looks Like After Conversion

The same economy as a complete SAM (simplified view):

```
           │  agr   ind   ser │  L     K   │  hh   gvt  firm  row │  inv    X
───────────┼─────────────────┼────────────┼─────────────────────┼─────────────
agr        │   0     0     0 │  0     0   │  30     0     0     0 │   0     5
ind        │   0     0     0 │  0     0   │  35     0     0     0 │   0    10
ser        │   0     0     0 │  0     0   │  30     0     0     0 │   0     5
───────────┼─────────────────┼────────────┼─────────────────────┼─────────────
I.agr      │  20    10    15 │  0     0   │   0     0     0     0 │   0     0
I.ind      │  30    40    25 │  0     0   │   0     0     0     0 │   0     0
I.ser      │  20    30    20 │  0     0   │   0     0     0     0 │   0     0
───────────┼─────────────────┼────────────┼─────────────────────┼─────────────
L (labor)  │  26    32.5  26 │  0     0   │   0     0     0     0 │   0     0
K (capital)│  14    17.5  14 │  0     0   │   0     0     0     0 │   0     0
───────────┼─────────────────┼────────────┼─────────────────────┼─────────────
AG.hh      │   0     0     0 │ 80.8  22.7 │   0     0     0     0 │   0     0
AG.gvt     │   0     0     0 │  4.2   2.3 │   0     0     0     0 │   0     0
AG.firm    │   0     0     0 │  0    20.5 │   0     0     0     0 │   0     0
AG.row     │  10    20    20 │  0     0   │   0     0     0     0 │   0     0
AG.ti      │   5     6     5 │  0     0   │   0     0     0     0 │   0     0
AG.tm      │ 0.5     1     1 │  0     0   │   0     0     0     0 │   0     0
───────────┼─────────────────┼────────────┼─────────────────────┼─────────────
OTH.inv    │   0     0     0 │  0     0   │  20     5    20     0 │   0     0
```

**What's new in the SAM:**
- **L and K rows**: VA disaggregated into labor (65%) and capital (35%)
- **Institutions (AG.*)**: Households (hh), government (gvt), firms (firm), rest-of-world (row)
- **Tax accounts**: ti (indirect taxes), tm (import tariffs)
- **Factor income flows**: L and K distribute income to households, firms, government
- **Investment closure**: Savings from institutions → investment
- **Export accounts (X)**: Separate export block for CGE modeling

## What You Need to Extend MIP to SAM

The conversion requires **external data** not present in the standard MIP:

### 1. Factor Shares (How to Split VA)

**Question**: Of the total Value Added, what % goes to labor vs capital?

**Source options**:
- National accounts detailed tables (employee compensation vs operating surplus)
- Extended input-output tables with factor decomposition
- Labor force surveys + wage data
- International comparisons (World Bank, ILO)

**Example**:
```python
# From national accounts: Labor compensation = 68%, Capital income = 32%
va_factor_shares = {"L": 0.68, "K": 0.32}
```

**If unavailable**: Use literature default (L=0.65, K=0.35 from Lofgren et al. 2002)

### 2. Income Distribution (Who Owns the Factors)

**Question**: Where does factor income go?
- Labor income → mostly households, some taxes
- Capital income → households (dividends), firms (retained earnings), taxes

**Source options**:
- Household income surveys (ENIGH, EPH, etc.)
- Tax authority data (income declarations)
- Central bank household accounts
- Corporate balance sheets (dividend distribution)

**Example**:
```python
factor_to_household_shares = {
    "L": {
        "hh": 0.95,    # 95% of wages to households
        "gvt": 0.05    # 5% direct taxes on labor income
    },
    "K": {
        "hh": 0.50,    # 50% capital income to households (dividends, rent)
        "firm": 0.45,  # 45% retained by firms
        "gvt": 0.05    # 5% taxes on capital
    }
}
```

**If unavailable**: Use default simple 1-household distribution

### 3. Tax Rates (Fiscal Flows)

**Question**: What are the effective tax rates on production and trade?

**Source options**:
- Tax authority reports (SAT, DGI, AFIP, etc.)
- Ministry of finance revenue statistics
- Customs data for import tariffs
- OECD tax statistics

**Example**:
```python
tax_rates = {
    "production_tax": 0.12,  # Effective VAT/sales tax rate
    "import_tariff": 0.06,   # Average tariff rate
    "direct_tax": 0.18       # Effective income tax rate
}
```

**If unavailable**: Use defaults from literature (production=0.10, tariff=0.05, direct=0.15)

### Summary Table: Data Requirements

| Data Item | In MIP? | External Source Needed | Default Available? |
|-----------|---------|------------------------|-------------------|
| Intermediate flows (I×J) | ✅ Yes | - | - |
| Final demand (C, G, I, X) | ✅ Yes | - | - |
| Imports by commodity | ✅ Yes | - | - |
| Value Added (aggregate) | ✅ Yes | - | - |
| **Factor shares (L/K)** | ❌ No | National accounts | ✅ L=65%, K=35% |
| **Income distribution** | ❌ No | Household surveys | ✅ Simple 1-household |
| **Tax rates** | ❌ No | Fiscal data | ✅ Typical rates |
| **Number of households** | ❌ No | Survey data | ✅ 1 household |

### Minimum Requirements for Conversion

**Absolute minimum** (everything uses defaults):
```python
run_mip_to_sam("mip.xlsx")  # Uses all defaults
```

**Recommended** (at least factor shares from national data):
```python
run_mip_to_sam(
    "mip.xlsx",
    va_factor_shares={"L": 0.68, "K": 0.32}  # From national accounts
)
```

**Best practice** (all country-specific data):
```python
run_mip_to_sam(
    "mip.xlsx",
    va_factor_shares={"L": 0.68, "K": 0.32},
    tax_rates={"production_tax": 0.12, "import_tariff": 0.06},
    factor_to_household_shares={...}  # From household surveys
)
```

## Quick Start

### Minimal Example (Using Defaults)

```python
from equilibria.sam_tools import run_mip_to_sam

# Convert MIP to SAM using literature-based defaults
result = run_mip_to_sam(
    input_path="path/to/mip_2020.xlsx",
    output_path="path/to/sam_2020.xlsx"
)

# Check the result
print(f"Conversion completed in {len(result.steps)} steps")
print(f"Final SAM saved to: {result.output_path}")
```

### Recommended Example (With Country-Specific Data)

```python
from equilibria.sam_tools import run_mip_to_sam

result = run_mip_to_sam(
    input_path="data/mip_ecuador_2020.xlsx",

    # Factor shares from national accounts
    va_factor_shares={"L": 0.68, "K": 0.32},

    # Tax rates from fiscal authority
    tax_rates={
        "production_tax": 0.12,    # Effective VAT rate
        "import_tariff": 0.06,     # Average tariff
        "direct_tax": 0.18         # Income tax rate
    },

    # Output paths
    output_path="output/sam_ecuador_2020.xlsx",
    report_path="output/conversion_report.json"
)

# Verify SAM quality
balance = result.steps[-1]["balance"]
print(f"Max row-column diff: {balance['max_row_col_abs_diff']:.2e}")
```

## Input Data Requirements

### Required: MIP Excel File

Your Excel file must contain:

**Rows:**
- Commodity/sector labels (e.g., "agr", "ind", "ser")
- A row labeled "Valor Agregado" or similar (Value Added)
- A row labeled "Importaciones" or similar (Imports) - optional

**Columns:**
- Sector labels matching the row labels
- Final demand columns: "HH" (households), "GOV" (government), "INV" (investment), "EXP" (exports)

**Example Structure:**

```
           agr   ind   ser   HH   GOV   INV   EXP
agr         20    10    15   30    10     5     5
ind         30    40    25   35    10    10    10
ser         20    30    20   30    15    10     5
VA          40    50    40    0     0     0     0
IMP         10    20    20    0     0     0     0
```

### Optional: External Parameters

#### 1. Factor Shares (`va_factor_shares`)

**What it is**: Proportion of Value Added going to Labor vs Capital

**How to get it**:
- National accounts detailed factor compensation tables
- Extended input-output tables if available
- Literature: Typically L=60-70%, K=30-40%

**Default**: `{"L": 0.65, "K": 0.35}` (from IFPRI standard CGE model)

**Example**:
```python
va_factor_shares = {
    "L": 0.68,  # 68% to labor
    "K": 0.32   # 32% to capital
}
```

#### 2. Factor Income Distribution (`factor_to_household_shares`)

**What it is**: How factor income is distributed to institutions

**How to get it**:
- Household surveys (income sources)
- Central bank household accounts
- Corporate balance sheets (for capital income)

**Default**:
```python
{
    "L": {"hh": 0.95, "gvt": 0.05},              # 95% labor to HH, 5% taxes
    "K": {"hh": 0.50, "firm": 0.45, "gvt": 0.05}  # 50% capital to HH, 45% to firms
}
```

#### 3. Tax Rates (`tax_rates`)

**What it is**: Effective tax rates for production, imports, and direct taxes

**How to get it**:
- Tax authority reports (SAT, AFIP, etc.)
- Customs statistics for tariffs
- Tax revenue data

**Default**:
```python
{
    "production_tax": 0.10,  # 10% VAT/sales tax
    "import_tariff": 0.05,   # 5% average tariff
    "direct_tax": 0.15       # 15% income tax
}
```

## Function Reference

### `run_mip_to_sam()`

```python
def run_mip_to_sam(
    input_path: Path | str,
    *,
    # Disaggregation parameters
    va_factor_shares: dict[str, float] | None = None,
    factor_to_household_shares: dict[str, dict[str, float]] | None = None,
    tax_rates: dict[str, float] | None = None,

    # Format options
    sheet_name: str = "MIP",
    va_row_label: str = "Valor Agregado",
    import_row_label: str = "Importaciones",

    # Balancing
    ras_type: str = "arithmetic",
    ras_tol: float = 1e-9,
    ras_max_iter: int = 200,

    # Output
    output_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> MIPToSAMResult
```

**Parameters:**

- `input_path`: Path to MIP Excel file
- `va_factor_shares`: Factor decomposition of VA (default: L=65%, K=35%)
- `factor_to_household_shares`: Income distribution matrix (default: 1-household)
- `tax_rates`: Tax rates dict (default: typical rates from literature)
- `sheet_name`: Excel sheet name containing MIP
- `va_row_label`: Label to identify VA row
- `import_row_label`: Label to identify imports row
- `ras_type`: RAS balancing algorithm ("arithmetic" or "multiplicative")
- `ras_tol`: Tolerance for RAS convergence
- `ras_max_iter`: Maximum RAS iterations
- `output_path`: Where to save resulting SAM (Excel)
- `report_path`: Where to save transformation report (JSON)

**Returns:**

`MIPToSAMResult` with:
- `sam`: The transformed SAM object
- `steps`: List of transformation steps with diagnostics
- `output_path`: Path where SAM was saved
- `report_path`: Path where report was saved

## Transformation Steps Explained

The conversion applies these transformations in sequence. Here's what happens at each step:

### Step 1: Initial Balancing (RAS)

**Purpose**: Ensure MIP is internally consistent before transformations

**Input**: Raw MIP from Excel
**Output**: Balanced MIP with row sums ≈ column sums

This step uses RAS algorithm to make small adjustments ensuring accounting consistency.

### Step 2: Normalize Accounts

**Purpose**: Convert generic labels to SAM structure

**Before** (RAW labels):
```
           agr    ind    ser    HH    GOV
agr        20     10     15     30     10
VA         40     50     40      0      0
```

**After** (categorized):
```
           J.agr  J.ind  J.ser  FD.HH  FD.GOV
I.agr       20     10     15     30      10
VA.agg      40     50     40      0       0
```

**What changed**: Labels now have categories (I=commodities, J=sectors, VA=value added, FD=final demand)

### Step 3: Disaggregate VA to Factors

**Purpose**: Split aggregate VA into Labor (L) and Capital (K)

**Before**:
```
           J.agr  J.ind  J.ser
VA.agg      40     50     40    ← Aggregate VA
```

**After** (with shares L=0.65, K=0.35):
```
           J.agr  J.ind  J.ser
L.labor     26    32.5    26    ← 65% of VA
K.capital   14    17.5    14    ← 35% of VA
```

**Formula**: `L = VA × 0.65`, `K = VA × 0.35` for each sector

### Step 4: Factor Income Distribution

**Purpose**: Show where factor income goes (households, firms, government)

**Before**: Factors only appear as costs to sectors
```
           J.agr  J.ind  J.ser
L.labor     26    32.5    26
K.capital   14    17.5    14
```

**After**: Income flows created
```
           J.agr  J.ind  J.ser │  L     K
L.labor     26    32.5    26   │  -     -
K.capital   14    17.5    14   │  -     -
─────────────────────────────────────────
AG.hh        -      -      -   │ 80.8  22.7  ← Household income from L and K
AG.gvt       -      -      -   │  4.2   2.3  ← Government taxes
AG.firm      -      -      -   │  0    20.5  ← Firm retained earnings
```

**Interpretation**:
- Total labor income (84.5) distributed: 95.3% to households, 4.7% to government
- Total capital income (45.5) distributed: 50% to households, 45% to firms, 5% to taxes

### Step 5: Household Expenditure

**Purpose**: Convert final demand "HH" to institution expenditure

**Before**:
```
           FD.HH
I.agr       30    ← Anonymous household demand
I.ind       35
I.ser       30
```

**After**:
```
           AG.hh
I.agr       30    ← Household institution buys commodities
I.ind       35
I.ser       30
```

**What changed**: Final demand becomes institutional expenditure (AG.hh → I.*)

### Step 6: Government Flows

**Purpose**: Create tax accounts and government consumption

**New accounts created**:
```
           J.agr  J.ind  J.ser │  AG.gvt
AG.ti       5      6      5    │   -      ← Indirect taxes (production)
AG.tm      0.5     1      1    │   -      ← Import tariffs
─────────────────────────────────────────
AG.ti       -      -      -    │   16     ← ti flows to government
AG.tm       -      -      -    │   2.5    ← tm flows to government
AG.gvt      -      -      -    │   -
```

Plus government consumption:
```
           AG.gvt
I.ser       15     ← Government buys services (from FD.GOV)
```

### Step 7: ROW Account (Rest of World)

**Purpose**: Handle imports and exports through ROW institution

**Before**:
```
            I.agr  I.ind  I.ser
IMP.total    10     20     20    ← Generic imports
```

**After**:
```
            I.agr  I.ind  I.ser
AG.row       10     20     20    ← ROW supplies imports
```

Plus exports (initially from FD.EXP):
```
            AG.row
I.agr         5      ← Exports to ROW (will move to X.agr)
I.ind        10
I.ser         5
```

### Step 8: Investment Account

**Purpose**: Create savings-investment closure

**Savings** (institution → investment):
```
            OTH.inv
AG.hh         20     ← Household savings
AG.firm       20     ← Firm retained earnings
AG.gvt         5     ← Government surplus
```

**Investment demand** (from FD.INV):
```
            OTH.inv
I.agr         5      ← Investment demand by commodity
I.ind        10
I.ser        10
```

**Balance condition**: Total savings (45) ≈ Total investment (25) + adjustments

### Step 9-10: Export Block (X)

**Purpose**: Create separate export accounts for CGE modeling

**Before**:
```
            AG.row
I.agr         5     ← Exports as commodity flow
I.ind        10
I.ser         5
```

**After**:
```
            AG.row
X.agr         5     ← Exports in dedicated X account
X.ind        10
X.ser         5
```

**Why**: CGE models treat exports differently from domestic sales (different prices, elasticities)

### Complete Transformation Summary

**Original MIP (6 accounts)**:
- 3 sectors (agr, ind, ser)
- 1 VA aggregate
- 1 imports row
- 4 final demand columns

**Final SAM (20+ accounts)**:
- 3 commodities (I.agr, I.ind, I.ser)
- 3 sectors (J.agr, J.ind, J.ser)
- 2 factors (L.labor, K.capital)
- 5+ institutions (AG.hh, AG.gvt, AG.firm, AG.row, AG.ti, AG.tm)
- 3 exports (X.agr, X.ind, X.ser)
- 1 investment (OTH.inv)

**Accounting identities preserved**:
- Production = Intermediate use + Final demand
- Income = Expenditure for each institution
- Savings = Investment (with RoW balance)
- Imports + Domestic production = Total supply

## Validation

### Check Conversion Quality

```python
result = run_mip_to_sam("mip.xlsx", output_path="sam.xlsx")

# 1. Check balance
final_balance = result.steps[-1]["balance"]
max_diff = final_balance["max_row_col_abs_diff"]
print(f"Max imbalance: {max_diff:.2e}")
assert max_diff < 1e-6, "SAM not well balanced"

# 2. Check accounts created
sam = result.sam
categories = {cat for cat, _ in sam.row_keys}
assert "L" in categories  # Labor factor
assert "K" in categories  # Capital factor

elements = {elem for _, elem in sam.row_keys}
assert "hh" in elements   # Households
assert "gvt" in elements  # Government
assert "row" in elements  # Rest of world

# 3. Review transformation steps
for step in result.steps:
    print(f"{step['step']:25s} - balance: {step['balance']['max_row_col_abs_diff']:.2e}")
```

### Use in PEP Model

```python
from equilibria.templates import PEP

# Load SAM into PEP model
model = PEP.from_sam("sam.xlsx")

# Verify calibration
model.calibrate()
assert model.is_calibrated(), "Model failed to calibrate"

# Run baseline
baseline = model.solve()
print(f"Baseline GDP: {baseline['GDP']:.2f}")

# Simulate productivity shock
results = model.simulate(shocks={"tfp": {"agr": 1.10}})
print(f"New GDP: {results['GDP']:.2f}")
print(f"GDP change: {(results['GDP']/baseline['GDP'] - 1)*100:.2f}%")
```

## Troubleshooting

### Problem: "No commodity/sector labels found"

**Cause**: Parser couldn't detect sector labels in MIP

**Solution**:
- Ensure first column has sector names ("agr", "ind", etc.)
- Check that labels are text, not numbers
- Verify Excel sheet name is correct

### Problem: "VA row not found"

**Cause**: Value Added row label doesn't match expected pattern

**Solution**:
- Specify custom label: `va_row_label="Value Added"` or `va_row_label="VA"`
- Check spelling and accents

### Problem: "SAM not balanced after conversion"

**Cause**: Tax rates or shares create accounting inconsistencies

**Solution**:
- Use default parameters first to isolate issue
- Check that tax_rates are realistic (< 0.5)
- Verify factor shares sum to 1.0
- Increase `ras_max_iter` to 500

### Problem: "Factor shares must sum to 1.0"

**Cause**: `va_factor_shares` doesn't add up to 1.0

**Solution**:
```python
# Incorrect
va_factor_shares = {"L": 0.6, "K": 0.3}  # Sums to 0.9

# Correct
va_factor_shares = {"L": 0.6, "K": 0.4}  # Sums to 1.0
```

## Advanced Usage

### Multiple Household Types (Future)

The architecture supports extension to multiple household types:

```python
# Future feature (not yet implemented)
factor_to_household_shares = {
    "L": {
        "hh_rural": 0.30,
        "hh_urban_poor": 0.25,
        "hh_urban_middle": 0.25,
        "hh_urban_rich": 0.15,
        "gvt": 0.05
    },
    "K": {
        "hh_urban_rich": 0.40,
        "firm": 0.50,
        "gvt": 0.10
    }
}
```

### Sector-Specific Tax Rates (Future)

```python
# Future feature (not yet implemented)
tax_rates = {
    "production_tax": {
        "agr": 0.05,   # Lower tax on agriculture
        "ind": 0.15,   # Higher tax on industry
        "ser": 0.12
    },
    "import_tariff": 0.06
}
```

## References

### Academic Literature

- **Lofgren, H., Harris, R. L., & Robinson, S. (2002)**. "A Standard Computable General Equilibrium (CGE) Model in GAMS". IFPRI Microcomputers in Policy Research 5.
  - Source for default factor shares and model structure

- **United Nations (2009)**. "System of National Accounts 2008", Chapter 26: Social Accounting Matrix.
  - Official methodology for SAM construction

- **Pyatt, G., & Round, J. I. (1985)**. "Social Accounting Matrices: A Basis for Planning". World Bank.
  - Foundational work on SAM theory

### Software Documentation

- Equilibria documentation: `/docs/`
- PEP model guide: `/docs/guides/pep_user_flow_examples.md`
- SAM tools API: `src/equilibria/sam_tools/`

## Examples

### Complete Workflow

```python
from pathlib import Path
from equilibria.sam_tools import run_mip_to_sam
from equilibria.templates import PEP

# 1. Convert MIP to SAM
mip_path = Path("data/mip_mexico_2019.xlsx")
sam_path = Path("output/sam_mexico_2019.xlsx")

result = run_mip_to_sam(
    mip_path,
    va_factor_shares={"L": 0.67, "K": 0.33},
    tax_rates={
        "production_tax": 0.11,  # Mexico VAT ~11% effective
        "import_tariff": 0.04,
        "direct_tax": 0.16
    },
    output_path=sam_path,
    report_path="output/report.json"
)

print(f"✓ SAM created with {len(result.steps)} transformation steps")
print(f"✓ Balance: {result.steps[-1]['balance']['max_row_col_abs_diff']:.2e}")

# 2. Load into PEP model
model = PEP.from_sam(sam_path)
model.calibrate()
print("✓ Model calibrated successfully")

# 3. Run scenario: 10% productivity increase in agriculture
results = model.simulate(
    shocks={"tfp": {"agr": 1.10}},
    name="Agricultural Productivity Shock"
)

# 4. Analyze results
print("\n=== Simulation Results ===")
print(f"GDP change: {results.pct_change('GDP'):.2f}%")
print(f"Household income change: {results.pct_change('Y_hh'):.2f}%")
print(f"Agricultural output change: {results.pct_change('X_agr'):.2f}%")

# 5. Export results
results.to_excel("output/scenario_results.xlsx")
print("✓ Results exported")
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/equilibria-project/equilibria/issues
- Documentation: https://equilibria.readthedocs.io
- Examples: `/docs/examples/`
