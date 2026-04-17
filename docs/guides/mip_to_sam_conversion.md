# MIP to SAM Conversion Guide

This guide explains how to convert an Input-Output Matrix (MIP) to a Social Accounting Matrix (SAM) compatible with CGE models like PEP.

---

## 1. Data Requirements

### 1.1 Required: Separated MIP

A separated MIP contains distinct matrices for domestic and imported flows:

| Matrix | Description | Dimensions |
|--------|-------------|------------|
| **Z_d** | Domestic intermediate consumption | n_products x n_sectors |
| **Z_m** | Imported intermediate consumption | n_products x n_sectors |
| **F_d** | Domestic final demand | n_products x n_fd_components |
| **F_m** | Imported final demand | n_products x n_fd_components |
| **VA** | Value added | n_va_components x n_sectors |
| **X** | Total production by sector | n_sectors |

### 1.2 Final Demand Components (F_d, F_m)

Typical components:
- **C_hh**: Household consumption
- **C_gov**: Government consumption
- **FBKF**: Gross fixed capital formation (investment)
- **Var.S**: Change in inventories
- **X**: Exports

### 1.3 Value Added Components (VA)

Typical components:
- **L**: Remunerations (wages, salaries)
- **K**: Gross operating surplus (profits, rents)
- **Taxes**: Net taxes on production

### 1.4 Optional: Tax Rates

If your CGE model requires explicit product taxes:

```python
product_tax_rates = {
    "effective_rate": 0.0767,  # Average rate
    # Or by sector:
    "agr": 0.03,
    "ind": 0.08,
    "ser": 0.05,
}
```

### 1.5 Optional: Factor Shares

Distribution of value added between labor and capital:

```python
va_factor_shares = {
    "L": 0.65,  # Labor share
    "K": 0.35,  # Capital share
}
```

### 1.6 Optional: Income Distribution

Distribution of factor income to institutions:

```python
factor_to_household_shares = {
    "L": {"hh": 0.95, "gvt": 0.05},
    "K": {"hh": 0.35, "firm": 0.60, "gvt": 0.05}
}
```

---

## 2. System of Equations

The MIP must satisfy these fundamental identities:

### 2.1 Product Balance (Domestic)

```
X = Z_d @ 1 + F_d

Where:
  X     = Production totals (n,)
  Z_d   = Domestic intermediate consumption (n x n)
  F_d   = Domestic final demand (n x k)
```

Each product's domestic production equals its intermediate use plus final demand.

### 2.2 Import Balance

```
M = Z_m @ 1 + F_m

Where:
  M     = Total imports by product (n,)
  Z_m   = Imported intermediate consumption (n x n)
  F_m   = Imported final demand (n x k)
```

Total imports of each product equal imported intermediate plus final demand.

### 2.3 Industry Balance

```
X = Z_d.T @ 1 + Z_m.T @ 1 + VA

Where:
  X     = Production totals (n,)
  Z_d.T = Domestic inputs purchased (transposed)
  Z_m.T = Imported inputs purchased (transposed)
  VA    = Value added (summed over components)
```

Each sector's output equals its total inputs (domestic + imported) plus value added.

### 2.4 Aggregate Identity

```
sum(F_d) = sum(VA) + sum(Z_m)

Or equivalently (PIB identity):
sum(VA) = sum(F_d) - sum(IMP_F)
```

Total domestic final demand equals total value added plus imported intermediate consumption.

### 2.5 Supply-Demand Balance (Per Product)

**IMPORTANT**: This constraint is mathematically incompatible with constraints 2.1-2.3 when IMP_F > 0.

```
Supply_i = Demand_i

Where:
  Supply_i  = X_i + M_i
  Demand_i  = Z_d[i,:].sum() + Z_m[i,:].sum() + F_d[i,:].sum() + F_m[i,:].sum()
```

**Mathematical proof of incompatibility**:
- If Z is balanced (Z_d rows = cols) AND PIB is satisfied, then S-D balance requires F_d[i] = VA[i] for each product
- But PIB identity also requires sum(IMP_F) = 0
- When IMP_F > 0 (Bolivia: 6,323 USD), this is impossible

**CGE models handle this**: They equilibrate S-D endogenously via prices.

---

## 3. Conversion Steps

### Step 1: Load MIP

```python
from equilibria.sam_tools.mip_loader import load_mip_excel, MIPConfig

# Auto-detect format
mip = load_mip_excel("mip_bolivia.xlsx")

# Or specify format
config = MIPConfig.bolivia_format()
mip = load_mip_excel("mip_bolivia.xlsx", config)
```

### Step 2: Validate Balances

```python
from equilibria.sam_tools.mip_loader import validate_mip_balances, mip_summary

# Check balance errors
errors = validate_mip_balances(mip)
print(f"PIB error: {errors['pib_error_pct']:.2%}")
print(f"Industry balance max: {errors['industry_error_max']:.2f}")

# Or get full summary
print(mip_summary(mip))
```

### Step 3: Balance MIP (if needed)

```python
from equilibria.sam_tools.balancing import balance_mip_gras

result = balance_mip_gras(
    mip.Z_d.values,
    mip.Z_m.values,
    mip.F_d.values,
    mip.F_m.values,
    mip.VA.values,
    mip.X.values,
)

print(f"Converged: {result.converged}")
print(f"PIB error: {result.error_pib:.2f}")
print(f"Industry error: {result.error_industry:.2f}")
```

### Step 4: Convert to SAM

```python
from equilibria.sam_tools import run_mip_to_sam

result = run_mip_to_sam(
    "mip_balanced.xlsx",
    balancing_method="gras",
    va_factor_shares={"L": 0.39, "K": 0.61},
    factor_to_household_shares={
        "L": {"hh": 0.95, "gvt": 0.05},
        "K": {"hh": 0.35, "firm": 0.60, "gvt": 0.05}
    },
    output_path="sam_output.xlsx",
    report_path="conversion_report.json",
)

print(f"SAM created: {result.output_path}")
print(f"Steps: {len(result.steps)}")
```

### Step 5: Validate SAM

```python
# Check SAM balance
sam = result.sam
matrix = sam.matrix
row_sums = matrix.sum(axis=1)
col_sums = matrix.sum(axis=0)
max_diff = abs(row_sums - col_sums).max()

print(f"SAM max imbalance: {max_diff:.6f}")
assert max_diff < 1e-6, "SAM not balanced!"
```

---

## 4. Balancing Methods

### 4.1 GRAS (Recommended)

Generalized RAS for matrices with negative values (Junius & Oosterhaven, 2003).

```python
result = run_mip_to_sam(
    "mip.xlsx",
    balancing_method="gras",
    ...
)
```

**Characteristics**:
- Handles negative values (e.g., inventory changes)
- Prioritizes PIB = 0% and Z balance
- Accepts S-D imbalance

### 4.2 RAS

Classic iterative proportional fitting for non-negative matrices.

```python
result = run_mip_to_sam(
    "mip.xlsx",
    balancing_method="ras",
    ...
)
```

**Characteristics**:
- Requires non-negative matrix
- Fast convergence
- Good for simple cases

### 4.3 SUT-RAS

Supply-Use Table RAS for simultaneous balancing.

```python
result = run_mip_to_sam(
    "mip.xlsx",
    balancing_method="sut_ras",
    ...
)
```

**Characteristics**:
- Treats Z and F as extended system
- Simultaneous constraint satisfaction
- Good for complex SUTs

### 4.4 Cross-Entropy

Minimizes information distance from original matrix.

```python
result = run_mip_to_sam(
    "mip.xlsx",
    balancing_method="entropy",
    ...
)
```

**Characteristics**:
- Theoretically optimal
- Equivalent to GRAS for linear constraints
- Good for uncertainty quantification

### 4.5 None (Pre-balanced)

Use when input MIP is already balanced.

```python
result = run_mip_to_sam(
    "mip_balanced.xlsx",
    balancing_method="none",
    ...
)
```

---

## 5. Optional Parameters

### 5.1 Product Tax Rates

Add explicit product taxes to reach PIB at market prices:

```python
result = run_mip_to_sam(
    "mip.xlsx",
    product_tax_rates={"effective_rate": 0.08},
    # Creates AG.ti and AG.tm accounts
    # PIB = VAB + product_taxes
)
```

### 5.2 Factor Shares

Distribution of VA between L and K:

```python
result = run_mip_to_sam(
    "mip.xlsx",
    va_factor_shares={"L": 0.65, "K": 0.35},
    # If MIP already has L/K disaggregated, this is ignored
)
```

### 5.3 Income Distribution

Factor income distribution to institutions:

```python
result = run_mip_to_sam(
    "mip.xlsx",
    factor_to_household_shares={
        "L": {"hh": 0.93, "gvt": 0.07},
        "K": {"hh": 0.35, "firm": 0.60, "gvt": 0.05}
    },
)
```

---

## 6. SAM Structure Output

The resulting SAM has the following account structure:

### 6.1 Activities/Industries (J)

Production activities corresponding to MIP sectors.

### 6.2 Commodities/Products (I)

Products/goods corresponding to MIP products.

### 6.3 Factors

- **L.labor**: Labor income
- **K.capital**: Capital income

### 6.4 Institutions

- **AG.hh**: Households
- **AG.gvt**: Government
- **AG.firm**: Firms
- **AG.row**: Rest of World

### 6.5 Tax Accounts (if product_tax_rates provided)

- **AG.ti**: Indirect taxes (product taxes)
- **AG.tm**: Import tariffs

### 6.6 Other

- **OTH.inv**: Investment/Savings
- **X.*** : Export accounts by product

---

## 7. Troubleshooting

### 7.1 PIB Mismatch

**Problem**: PIB (production) != PIB (expenditure)

**Cause**: Unbalanced MIP from mixed data sources

**Solution**:
```python
result = run_mip_to_sam(
    "mip.xlsx",
    balancing_method="gras",  # Prioritizes PIB = 0
)
```

### 7.2 Large S-D Imbalance

**Problem**: Supply-Demand imbalance warning

**This is expected!** When IMP_F > 0, perfect S-D balance is mathematically impossible with PIB + Z constraints satisfied.

**Solution**: Accept the imbalance. CGE models equilibrate S-D via prices.

### 7.3 Negative Values Error

**Problem**: RAS fails with negative values

**Cause**: Inventory changes (Var.S) can be negative

**Solution**:
```python
result = run_mip_to_sam(
    "mip.xlsx",
    balancing_method="gras",  # Handles negatives
)
```

### 7.4 Non-convergence

**Problem**: Balancing does not converge

**Solutions**:
1. Increase iterations: `ras_max_iter=500`
2. Relax tolerance: `ras_tol=1e-4`
3. Check data quality (NaN, extreme values)

---

## 8. References

### Methodology
- Junius, T., & Oosterhaven, J. (2003). "The Solution of Updating or Regionalizing a Matrix with Both Positive and Negative Entries." Economic Systems Research.
- Robinson, S., Cattaneo, A., & El-Said, M. (2001). "Updating and Estimating a Social Accounting Matrix Using Cross Entropy Methods."
- Lofgren, H., Harris, R. L., & Robinson, S. (2002). "A Standard CGE Model in GAMS." IFPRI.

### Technical Documentation
- `docs/analysis/bolivia_mip_technical_report.md`: Mathematical proof of constraint incompatibility
- `docs/analysis/bolivia_mip_implementation_guide.md`: Bolivia-specific parameters
- `docs/technical/matrix_balancing_methods.md`: Algorithm details

---

**Date**: April 2025
**Version**: 1.0
