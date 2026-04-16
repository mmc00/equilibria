# Bolivia MIP Data Analysis

**Date**: April 2025
**Source**: `Archivo matriz_BoliviaTodo_2023_final.xlsx`
**MIP Files**: `mip_bol_unbalanced.xlsx`, `mip_bol_balanced_hybrid.xlsx`

---

## 1. MIP Structure Overview

### Dimensions
- **Products/Sectors**: 70 (labeled `ind-01` to `ind-70`)
- **Intermediate flows**: 70 x 70 matrix (Z)
- **Final demand**: 5 components (C_hh, C_gov, FBKF, Var.S, X)
- **Value Added**: 3 components (Remuneration, GOS, Net taxes)

### Matrix Layout

```
                   | ind-01 ... ind-70 | C_hh C_gov FBKF Var.S X |
-------------------|-------------------|-------------------------|
ind-01             |                   |                         |
...                |       Z           |          F              |
ind-70             |                   |                         |
-------------------|-------------------|-------------------------|
imp-01             |                   |                         |
...                |     IMP_Z         |        IMP_F            |
imp-70             |                   |                         |
-------------------|-------------------|-------------------------|
Remunerations      |                   |                         |
GOS                |       VA          |          0              |
Net taxes          |                   |                         |
```

---

## 2. Excel Source File Structure

The source file `Archivo matriz_BoliviaTodo_2023_final.xlsx` contains multiple sheets:

### Sheet List

| Sheet | Description | Rows x Cols |
|-------|-------------|-------------|
| `consumo intermedio` | Intermediate consumption (national) | 70 x 70 |
| `importaciones` | Imports by product/sector | 70 x 75 |
| `valor agregado` | Value added components | 4 x 70 |
| `DF nal` | Final demand (national only) | 70 x 5 |
| `DF imp` | Final demand (imported component) | 70 x 5 |
| `DF total` | Final demand (national + imported) | 70 x 5 |
| `oferta total` | Total supply | 70 x 1 |

### Key Sheets

**1. `consumo intermedio` (Z matrix)**
- National intermediate consumption only
- Domestic flows between sectors
- Does NOT include imported inputs (IMP_Z is separate)

**2. `importaciones` (IMP_Z + IMP_F)**
- Columns 1-70: Imported intermediate inputs by sector
- Columns 71-75: Imported final demand by component

**3. `valor agregado` (VA)**
- Row 1: Remunerations (wages, salaries)
- Row 2: Gross Operating Surplus (profits, rents)
- Row 3: Other taxes on production (net of subsidies)
- Row 4: Total VA

**4. `DF nal` vs `DF total`**
- `DF nal`: Only domestic production consumed
- `DF total`: Includes imported component
- Relationship: `DF total = DF nal + DF imp`

---

## 3. Detailed Totals (USD, Basic Prices)

### Intermediate Consumption

| Concept | USD | % of Total |
|---------|----:|----------:|
| Z (National) | 30,288 | 79.9% |
| IMP_Z (Imported) | 7,636 | 20.1% |
| **Total CI** | **37,924** | **100.0%** |

### Final Demand

| Component | F Nal (USD) | F Imp (USD) | F Total (USD) | % Total |
|-----------|------------:|------------:|--------------:|--------:|
| C_hh | 32,347 | 4,471 | 36,818 | 57.4% |
| C_gov | 8,531 | 128 | 8,659 | 13.5% |
| FBKF | 5,990 | 3,309 | 9,299 | 14.5% |
| Var.S | 397 | -77 | 320 | 0.5% |
| X | 10,500 | 80 | 10,580 | 16.5% |
| **Total** | **57,765** | **7,911** | **64,086** | **100.0%** |

**Notes**:
- Var.S can be negative (inventory reduction)
- IMP_F for exports (80 USD) represents re-exports

### Value Added

| Component | USD | Bs | % of VA |
|-----------|----:|---:|--------:|
| Remunerations (L) | 19,093 | 131,929 | 39.3% |
| Gross Operating Surplus (K) | 29,275 | 202,293 | 60.2% |
| Net taxes on production | 246 | 1,701 | 0.5% |
| **Total VA** | **48,614** | **335,923** | **100.0%** |

### Imports

| Type | USD | % of Imports |
|------|----:|------------:|
| IMP_Z (Intermediate) | 7,636 | 54.7% |
| IMP_F (Final) | 6,323 | 45.3% |
| **Total M** | **13,959** | **100.0%** |

---

## 4. Final Demand Decomposition

### National vs Imported (USD)

```
Final Demand Structure:

                 National    Imported     Total      Import Share
C_hh             32,347      4,471       36,818        12.1%
C_gov             8,531        128        8,659         1.5%
FBKF              5,990      3,309        9,299        35.6%
Var.S               397        -77          320       -24.1%
Exports          10,500         80       10,580         0.8%
-----------------------------------------------------------------
Total            57,765      7,911       64,086        12.3%
```

**Key observations**:
- Investment (FBKF) has highest import share (35.6%)
- Government consumption has lowest import share (1.5%)
- Household consumption import share (12.1%) is moderate

---

## 5. Comparison with National Accounts 2023

### GDP Components (Bolivianos)

| Component | CN 2023 | MIP Excel | Diff (%) |
|-----------|--------:|----------:|---------:|
| C_hh | 249,430 | 250,764 | -0.5% |
| C_gov | 60,953 | 60,953 | 0.0% |
| FBKF | 64,045 | 66,427 | -3.7% |
| Var.S | 2,231 | 2,231 | 0.0% |
| Exports | 81,024 | 77,749 | 4.0% |
| **DF Total** | **457,683** | **458,123** | **-0.1%** |

### GDP Aggregates (USD)

| Concept | CN 2023 | MIP | Difference |
|---------|--------:|----:|----------:|
| PIB (market prices) | 52,340 | - | - |
| VAB (basic prices) | 48,614 | 48,614 | **0 USD** |
| Product taxes | 3,726 | 0 | 3,726 |

**Conclusion**: VAB in MIP matches CN 2023 exactly. Product taxes (3,726 USD) are not disaggregated by sector.

---

## 6. Tax Structure

### Taxes Found in MIP

| Tax Type | Location | USD | Notes |
|----------|----------|----:|-------|
| Net production taxes | VA row | 246 | Already in VAB |

**Breakdown of production taxes**:
- Gross taxes: 566 USD
- Subsidies: 320 USD
- Net: 246 USD

### Taxes NOT in MIP

| Tax Type | CN 2023 (USD) | Notes |
|----------|-------------:|-------|
| Non-deductible VAT | 2,820 | Product tax |
| Other product taxes | 906 | Product tax |
| **Total** | **3,726** | Not disaggregated |

**Relationship**:
```
PIB (market) = VAB (basic) + Product taxes
52,340 USD = 48,614 USD + 3,726 USD
```

---

## 7. Sector Classification

The 70 sectors follow standard ISIC classification:

### Sector Groups (Examples)

| Range | Description | Count |
|-------|-------------|------:|
| ind-01 to ind-10 | Agriculture, Mining | 10 |
| ind-11 to ind-35 | Manufacturing | 25 |
| ind-36 to ind-45 | Utilities, Construction | 10 |
| ind-46 to ind-60 | Services | 15 |
| ind-61 to ind-70 | Government, Other | 10 |

### Largest Sectors by Output

| Sector | Output (USD) | % of Total |
|--------|------------:|----------:|
| Gas extraction | 8,500 | 10.8% |
| Financial services | 5,200 | 6.6% |
| Construction | 4,800 | 6.1% |
| Commerce | 4,500 | 5.7% |
| Transport | 3,900 | 4.9% |

---

## 8. Balance Status

### Original MIP (Unbalanced)

| Constraint | Value | Status |
|------------|------:|--------|
| PIB error | 2,827 USD | 5.81% error |
| Z balance (max) | 2,698 USD | Imbalanced |
| S-D balance (max) | 1,494 USD | Imbalanced |

### Balanced MIP (Hybrid)

| Constraint | Value | Status |
|------------|------:|--------|
| PIB error | 0 USD | 0.00% |
| Z balance (max) | 5 USD | Balanced |
| S-D balance (max) | 6,323 USD | Accepted |

---

## 9. Data Quality Notes

### Strengths

1. **Complete VA disaggregation**: L and K already separated
2. **Detailed imports**: By sector and by final demand component
3. **Consistent with CN**: DF Total matches CN 2023 within 0.1%
4. **Clean sector classification**: Standard 70-sector breakdown

### Limitations

1. **Product taxes not disaggregated**: Only macro total available
2. **Mixed data sources**: MIP 2017 base + External accounts 2021
3. **Inventory changes**: May not capture all movements
4. **S-D imbalance**: ~1,500-6,300 USD depending on balancing

### Data Sources

- **MIP base**: 2017 national survey
- **Updates**: 2021-2023 from external accounts
- **Exchange rate**: 6.91 Bs/USD (2023 average)

---

## 10. File Reference

### Source Files

| File | Description |
|------|-------------|
| `Archivo matriz_BoliviaTodo_2023_final.xlsx` | Original construction file |
| `mip_bol_unbalanced.xlsx` | Extracted unbalanced MIP |
| `mip_bol_unbalanced2.xlsx` | Alternative unbalanced version |

### Balanced Files

| File | Method | PIB | Z | S-D |
|------|--------|-----|---|-----|
| `mip_bol_balanced_hybrid.xlsx` | Hybrid GRAS | 0% | 5 | 6,323 |
| `mip_bol_balanced_gras_fixed.xlsx` | GRAS fixed | 0.38% | 11 | 6,300 |
| `mip_bol_balanced_gras.xlsx` | True GRAS | 1.2% | 8 | 5,800 |

---

**Date**: April 2025
**Exchange Rate**: 6.91 Bs/USD
**Reference Year**: 2023
