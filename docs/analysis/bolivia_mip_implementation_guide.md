# Bolivia MIP Implementation Guide

**Date**: April 2025
**Purpose**: Practical guide for using Bolivia MIP in CGE models

---

## Quick Start

```python
from equilibria.sam_tools import run_mip_to_sam

# Basic usage with recommended balanced MIP
result = run_mip_to_sam(
    "mip_bol_balanced_hybrid.xlsx",
    va_factor_shares={"L": 0.393, "K": 0.607},  # From MIP data
    # S-D imbalance will be equilibrated by CGE model
)
```

---

## 1. Recommended Files

### For CGE Models

| File | PIB Error | Z Balance | S-D Balance | Use Case |
|------|-----------|-----------|-------------|----------|
| `mip_bol_balanced_hybrid.xlsx` | 0.00% | 5 USD | 6,323 USD | CGE calibration |
| `mip_bol_balanced_gras_fixed.xlsx` | 0.38% | 11 USD | 6,300 USD | Alternative |

**Recommended**: `mip_bol_balanced_hybrid.xlsx`

### For Input-Output Analysis

| File | PIB Error | Z Balance | S-D Balance | Use Case |
|------|-----------|-----------|-------------|----------|
| `mip_bol_unbalanced2.xlsx` | 5.81% | 2,698 USD | 1,494 USD | I-O multipliers |

---

## 2. Balancing Method Selection

### GRAS Complete (Default)

```python
result = run_mip_to_sam(
    input_path="mip_bol_unbalanced.xlsx",
    balancing_method="gras",  # Default
    # Prioritizes: PIB = 0%, Z cols = Z rows
    # Accepts: S-D imbalance (~6,000 USD)
)
```

**When to use**:
- CGE models (PEP, GTAP)
- When exact PIB is critical
- When model equilibrates S-D via prices

### No Balancing

```python
result = run_mip_to_sam(
    input_path="mip_bol_unbalanced.xlsx",
    balancing_method="none",
    # Uses data as-is
    # PIB error ~6%, S-D better
)
```

**When to use**:
- Input-Output analysis
- When S-D balance is critical
- When PIB error <10% is acceptable

---

## 3. Factor Shares Configuration

### Bolivia Actual Shares (from MIP data)

```python
va_factor_shares = {
    "L": 0.393,  # Remunerations / VA = 19,093 / 48,614
    "K": 0.607,  # GOS / VA = 29,275 / 48,614
}
```

### Labor/Capital Share Decomposition

The MIP already has VA disaggregated:
- **L (Labor)**: Remunerations = 19,093 USD (39.3% of VA)
- **K (Capital)**: Gross Operating Surplus = 29,275 USD (60.2% of VA)
- **Production taxes**: 246 USD (0.5% of VA)

**Note**: Bolivia is capital-intensive (extractive economy: gas, minerals).

---

## 4. Income Distribution Parameters

### Default Distribution

```python
factor_to_household_shares = {
    "L": {
        "hh": 0.95,   # 95% wages to households
        "gvt": 0.05   # 5% wage taxes
    },
    "K": {
        "hh": 0.35,   # 35% capital income to households
        "firm": 0.60, # 60% retained by firms
        "gvt": 0.05   # 5% capital taxes
    }
}
```

### Bolivia-Specific (if data available)

Sources for Bolivia-specific parameters:
- **ENAHO**: National Household Survey (INE)
- **SIN**: Tax declarations (Servicio de Impuestos Nacionales)
- **BCB**: Central Bank financial accounts

---

## 5. Product Taxes (Optional)

### Basic Prices (Default)

```python
# Default: Work with VAB (basic prices)
result = run_mip_to_sam(
    input_path="mip_bol_balanced_hybrid.xlsx",
    product_tax_rates=None,  # Default
    # PIB = VAB = 48,614 USD
)
```

### Market Prices (If Needed)

```python
# Add product taxes to reach PIB at market prices
result = run_mip_to_sam(
    input_path="mip_bol_balanced_hybrid.xlsx",
    product_tax_rates={
        "effective_rate": 0.0767,  # 3,726 / 48,614
        # Or by sector if available
    },
    # PIB = VAB + taxes = 48,614 + 3,726 = 52,340 USD
)
```

**When to use market prices**:
- PEP models requiring explicit tax accounts
- Analysis involving tax policy changes
- Models with AG.ti and AG.tm accounts

---

## 6. Complete Example

```python
from equilibria.sam_tools import run_mip_to_sam

# Full configuration for Bolivia
result = run_mip_to_sam(
    input_path="mip_bol_balanced_hybrid.xlsx",

    # Factor shares (from MIP data)
    va_factor_shares={
        "L": 0.393,
        "K": 0.607,
    },

    # Income distribution (estimated)
    factor_to_household_shares={
        "L": {"hh": 0.93, "gvt": 0.07},
        "K": {"hh": 0.35, "firm": 0.60, "gvt": 0.05}
    },

    # Fiscal parameters
    tax_rates={
        "import_tariff": 0.09,      # Effective tariff ~9%
        "production_tax": 0.13,     # Effective VAT ~13%
    },

    # Balancing
    balancing_method="gras",

    # Output
    output_path="sam_bolivia_2023.xlsx",
    report_path="conversion_report.json"
)

# Verify result
print(f"SAM created: {result.output_path}")
print(f"Steps: {len(result.steps)}")
print(f"Final balance: {result.steps[-1]['balance']['max_row_col_abs_diff']:.2e}")
```

---

## 7. Validation Checklist

After conversion, verify:

### 1. PIB Identity
```python
va_total = sam["L"].sum() + sam["K"].sum()
gdp_expenditure = sam["C_hh"] + sam["C_gov"] + sam["I"] + sam["X"] - sam["M"]
assert abs(va_total - gdp_expenditure) < 1.0
```

### 2. SAM Balance
```python
row_sums = sam.sum(axis=1)
col_sums = sam.sum(axis=0)
max_diff = abs(row_sums - col_sums).max()
assert max_diff < 1e-6
```

### 3. Factor Shares (Bolivia-specific)
```python
l_share = sam["L"].sum() / (sam["L"].sum() + sam["K"].sum())
# Should be ~39-40% for Bolivia
assert 0.35 < l_share < 0.45
```

### 4. Trade Openness
```python
openness = (sam["X"].sum() + sam["M"].sum()) / gdp
# Should be ~60-80% for small open economy
assert 0.5 < openness < 1.0
```

---

## 8. Troubleshooting

### PIB Mismatch After Conversion

**Problem**: PIB (production) != PIB (expenditure)

**Solution**: Ensure input MIP is already balanced:
```python
# Use pre-balanced file
run_mip_to_sam("mip_bol_balanced_hybrid.xlsx", ...)
```

### S-D Imbalance Warning

**Problem**: Large Supply-Demand imbalance reported

**Solution**: This is expected and acceptable for CGE:
```
S-D imbalance of ~6,000 USD (12% PIB) is normal for Bolivia MIP.
CGE models equilibrate this endogenously via prices.
```

### Negative Values in SAM

**Problem**: Unexpected negative values

**Check**:
1. `Var.S` (inventory change) can be negative - normal
2. `IMP_F` components can be negative for inventory adjustments
3. Large negatives elsewhere indicate data issues

---

## 9. File Locations

| Type | Location |
|------|----------|
| Balanced MIP files | `/Users/marmol/proyectos/cge_babel/playground/bol/` |
| Analysis documents | `/docs/analysis/bolivia_mip_*.md` |
| Technical docs | `/docs/technical/matrix_balancing_methods.md` |
| MIP-to-SAM guide | `/docs/guides/mip_to_sam_conversion.md` |

---

## 10. References

### Methodology
- Junius & Oosterhaven (2003): GRAS algorithm
- Lofgren et al. (2002): Standard CGE Model in GAMS
- Miller & Blair (2009): Input-Output Analysis

### Bolivia Data
- INE Bolivia: https://www.ine.gob.bo/
- BCB: https://www.bcb.gob.bo/
- SIN: Servicio de Impuestos Nacionales
- ANB: Aduana Nacional de Bolivia

---

**Date**: April 2025
**Recommended MIP**: `mip_bol_balanced_hybrid.xlsx` (PIB=0%, Z=5 USD)
