# GTAP CGE Template (Standard GTAP 7)

Complete GTAP CGE model implementation following the GTAP Standard 7 specification with 9×10 database.

## Overview

This template provides a full multi-regional CGE model based on the GTAP (Global Trade Analysis Project) Standard 7 model. It includes:

- **Multi-regional trade**: CET exports + CES Armington imports with bilateral flows
- **Production**: Nested CES technology with value-added and intermediates
- **Demand**: Private consumption, government, and investment
- **Factor markets**: Mobile and sector-specific factors
- **Global savings**: Endogenous investment allocation
- **Taxes**: Production, consumption, trade, and factor taxes

## Quick Start

```python
from equilibria.templates.gtap import GTAPSets, GTAPParameters, GTAPModelEquations, GTAPSolver
from equilibria.templates.gtap import build_gtap_contract

# Load data from Standard GTAP 7 9x10 database
sets = GTAPSets()
sets.load_from_gdx("basedata-9x10.gdx")

params = GTAPParameters()
params.load_from_gdx("basedata-9x10.gdx")

# Build model
contract = build_gtap_contract("gtap_standard")
equations = GTAPModelEquations(sets, params, contract.closure)
model = equations.build_model()

# Solve
solver = GTAPSolver(model, contract.closure, solver_name="ipopt")
result = solver.solve()

print(f"Status: {result.status}")
print(f"Walras check: {result.walras_value:.2e}")
```

## CLI Usage

### Display Data Information
```bash
python scripts/gtap/run_gtap.py info --gdx-file /Users/marmol/proyectos2/cge_babel/standard_gtap_7/basedata-9x10.gdx
```

### Solve Baseline
```bash
python scripts/gtap/run_gtap.py solve --gdx-file /Users/marmol/proyectos2/cge_babel/standard_gtap_7/basedata-9x10.gdx --solver ipopt
```

### Apply Shock
```bash
# 10% tariff on agricultural imports from EastAsia to Oceania
python scripts/gtap/run_gtap.py shock \
    --gdx-file /Users/marmol/proyectos2/cge_babel/standard_gtap_7/basedata-9x10.gdx \
    --variable rtms \
    --index '(Oceania,c_Crops,EastAsia)' \
    --value 0.10 \
    --solver ipopt
```

## Model Structure

### Sets
- **r**: Regions (e.g., EUR, USA, CHN, BRA, IND)
- **i**: Commodities/goods (e.g., agr, food, mfg, srv, ene)
- **a**: Activities/sectors (alias of i)
- **f**: Factors (lnd, skl, unsk, cap, nrs)
- **mf**: Mobile factors (subset)
- **sf**: Sector-specific factors (subset)
- **m**: Transport modes (air, sea, road, rail)

### Parameters

#### Elasticities
- `esubva`: CES elasticity between VA and intermediates
- `esubd`: CES elasticity between domestic and imported (top Armington)
- `esubm`: CES elasticity across import sources (bottom Armington)
- `omegax`: CET elasticity between domestic sales and exports
- `omegaw`: CET elasticity across export destinations
- `etrae`: CET elasticity for factor mobility

#### Benchmark Values (from SAM)
- `vom`: Output at market prices
- `vfm`: Factor payments
- `vdfm/vifm`: Domestic/imported intermediate demand
- `vpm/vgm/vim`: Private/gov/investment demand
- `vxmd`: Exports (FOB)
- `viws/vims`: Imports (CIF/tariff-inclusive)

#### Tax Rates
- `rto`: Output tax
- `rtf`: Factor tax
- `rtms`: Import tariff
- `rtxs`: Export subsidy (negative = tax)

### Equation Blocks

1. **Production Block**: Technology nests, factor demands, output allocation
2. **Trade Block**: CET exports, CES Armington, bilateral trade, transport margins
3. **Demand Block**: Private consumption, government, investment (CES/CDE)
4. **Factor Block**: Mobile vs sluggish factor markets
5. **Income Block**: Regional income, tax revenues
6. **Investment Block**: Global savings allocation mechanism
7. **Market Clearing**: Walras conditions

## Closures

### Standard GTAP (`gtap_standard`)
- Fixed: Taxes, technology, factor endowments
- Endogenous: Savings, investment, prices, quantities
- Numeraire: World price index (pnum = 1.0)

### Trade Policy (`trade_policy`)
- Allows import/export tax changes
- Useful for tariff/subsidy simulations

### Full Model (`gtap_full`)
- All equations active
- Full closure flexibility

### Single Region (`single_region`)
- Fixes world prices
- For single-country analysis

## Solvers

### IPOPT (Default)
- Interior Point OPTimizer
- For CNS (Constrained Nonlinear System)
- Good for large models
- Handles bounds well

### PATH (Optional)
- Mixed Complementarity Problem solver
- For MCP formulation
- Requires PATH installation

## Calibration

Share parameters are calibrated from benchmark SAM:
- Production shares: `p_gx`, `p_ax`
- Armington shares: `p_alphad`, `p_alpham`
- Trade shares: `p_amw`, `p_gw`
- Factor shares: `p_gf`

## Shocks

Apply shocks to exogenous variables:

```python
# Tax shock
shock = {
    "variable": "rtms",
    "index": ("USA", "agr", "EUR"),
    "value": 0.10  # 10% tariff
}
solver.apply_shock(shock)

# Technology shock
shock = {
    "variable": "axp",
    "index": ("USA", "agr"),
    "value": 1.05  # 5% productivity increase
}
solver.apply_shock(shock)
```

## Validation

Run parity checks against GAMS baseline:

```python
from equilibria.templates.gtap import GTAPParameters

params = GTAPParameters()
params.load_from_gdx("asa7x5.gdx")

is_valid, errors = params.validate()
if is_valid:
    print("✓ Parameters are valid")
else:
    print("Validation errors:", errors)
```

## Testing

```bash
# Run GTAP-specific tests
python -m pytest tests/templates/gtap/ -v

# Run with coverage
python -m pytest tests/templates/gtap/ --cov=equilibria.templates.gtap
```

## Data Sources and References

### GTAP Standard 7 (9×10 Database)
- **Location**: `/Users/marmol/proyectos2/cge_babel/standard_gtap_7/`
- **Base Data**: `basedata-9x10.gdx` (167KB) - SAM and benchmark values
- **Parameters**: `default-9x10.gdx` (9.2KB) - Elasticities (CES, CET, Armington)
- **Sets**: `sets-9x10.gdx` - Regions, commodities, activities, factors
- **Results**: `COMP.gdx` / `COMP.csv` - Simulation results

### Model Files
- **Main**: `comp.gms` - Simulation driver
- **Model**: `model.gms` - Equation definitions  
- **Data**: `getData.gms` - Data loading
- **Calibration**: `cal.gms` - Parameter calibration

### Database Dimensions (9×10)
- **9 Regions**: Oceania, EastAsia, China, SEAsia, SAsiaRofW, LatinAmer, MENAfrica, SSAfrica, RestofWorld
- **9 Commodities**: c_Crops, c_MeatLstk, c_Extraction, c_ProcFood, c_TextWapp, c_LightMnfc, c_HeavyMnfc, c_Util_Cons, c_TransComm
- **9 Activities**: a_Crops, a_MeatLstk, a_Extraction, a_ProcFood, a_TextWapp, a_LightMnfc, a_HeavyMnfc, a_Util_Cons, a_TransComm
- **5 Factors**: Land, UnSkLab, SkLab, Capital, NatRes

### External References
- GTAP Database: [https://www.gtap.agecon.purdue.edu/](https://www.gtap.agecon.purdue.edu/)
- GTAP 7 Documentation: Released 2008, base year 2004
- GAMS Documentation: [https://www.gams.com/](https://www.gams.com/)

## Implementation Notes

1. **GDX Reading**: Uses `equilibria.babel.gdx.reader` for native GDX file reading
2. **Pyomo**: All equations built as Pyomo model (sets, params, vars, constraints)
3. **Warm Start**: Variables initialized at benchmark (1.0) for faster convergence
4. **Scaling**: Automatic scaling of variables and equations
5. **Bounds**: Economic bounds applied (positive variables, etc.)

## Parity Testing (Python vs GAMS)

The template includes a complete parity testing system to validate Python results against GTAP Standard 7 GAMS baseline.

### Quick Parity Check

```bash
# Run parity check comparing Python vs GAMS using standard_gtap_7 results
python scripts/gtap/run_gtap_parity.py \
    --gdx-file /Users/marmol/proyectos2/cge_babel/standard_gtap_7/basedata-9x10.gdx \
    --gams-results /Users/marmol/proyectos2/cge_babel/standard_gtap_7/COMP.gdx \
    --tolerance 1e-6
```

### Python API

```python
from equilibria.templates.gtap import run_gtap_parity_test

# Simple parity check
result = run_gtap_parity_test(
    gdx_file="data/asa7x5.gdx",
    gams_results_gdx="results/gams_baseline.gdx",
    tolerance=1e-6,
)

if result.passed:
    print("✓ Parity check passed!")
else:
    print(f"✗ {result.n_mismatches} mismatches found")
    for m in result.mismatches[:5]:
        print(f"  {m['group']}{m['key']}: diff={m['abs_diff']:.2e}")
```

### Advanced Parity Testing

```python
from equilibria.templates.gtap.gtap_parity_pipeline import (
    GTAPParityRunner,
    GTAPGAMSReference,
    compare_gtap_gams_parity,
)

# Create runner
runner = GTAPParityRunner(
    gdx_file="data/asa7x5.gdx",
    gams_results_gdx="results/gams_baseline.gdx",
    closure="gtap_standard",
    tolerance=1e-6,
)

# Run Python model
py_result = runner.run_python()
print(f"Python: {py_result.status}, Walras={py_result.walras_value:.2e}")

# Run parity check
comparison = runner.run_parity_check()
print(f"Comparison: {comparison.n_mismatches} mismatches")

# Generate detailed report
report = runner.generate_report(comparison)
print(report)
```

### Parity Results

The parity system compares:
- **Activity levels**: xp (production), x (output)
- **Prices**: px, pp, ps, pd, pa, pmt, pet
- **Trade flows**: xe, xw, xmt, xet
- **Factor markets**: xf, xft, pf, pft
- **Demand**: xc, xg, xi
- **Income**: regy, yc, yg, yi
- **Indices**: pnum, pabs, walras

### Exit Codes

```bash
0  # Parity check passed
1  # Parity check failed (mismatches found)
2  # Error (missing files, solve failure, etc.)
```

### Examples

```bash
# Save report to file
python scripts/gtap/run_gtap_parity.py \
    --gdx-file data/asa7x5.gdx \
    --gams-results results/gams_baseline.gdx \
    --output parity_report.txt

# Export JSON results
python scripts/gtap/run_gtap_parity.py \
    --gdx-file data/asa7x5.gdx \
    --gams-results results/gams_baseline.gdx \
    --json-output results.json

# Use different tolerance
python scripts/gtap/run_gtap_parity.py \
    --gdx-file data/asa7x5.gdx \
    --gams-results results/gams_baseline.gdx \
    --tolerance 1e-8
```

## Future Enhancements

- [ ] Melitz module (heterogeneous firms)
- [ ] Dynamic recursive mode
- [ ] Energy/environmental extension (GTAP-E)
- [ ] MyGTAP module (household heterogeneity)
- [ ] Sub-regional modeling (NUTS2)
- [ ] Welfare decomposition
- [ ] Sensitivity analysis
- [x] GAMS parity testing system

## License

Same as equilibria project.
