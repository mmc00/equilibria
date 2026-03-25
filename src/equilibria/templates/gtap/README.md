# GTAP CGE Template (CGEBox Version)

Complete GTAP CGE model implementation following the CGEBox specification.

## Overview

This template provides a full multi-regional CGE model based on the GTAP (Global Trade Analysis Project) database and CGEBox framework. It includes:

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

# Load data
sets = GTAPSets()
sets.load_from_gdx("asa7x5.gdx")

params = GTAPParameters()
params.load_from_gdx("asa7x5.gdx")

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
python scripts/gtap/run_gtap.py info --gdx-file data/asa7x5.gdx
```

### Solve Baseline
```bash
python scripts/gtap/run_gtap.py solve --gdx-file data/asa7x5.gdx --solver ipopt
```

### Apply Shock
```bash
# 10% tariff on agricultural imports from EUR to USA
python scripts/gtap/run_gtap.py shock \
    --gdx-file data/asa7x5.gdx \
    --variable rtms \
    --index '(USA,agr,EUR)' \
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

### CGEBox Full (`cgebox_full`)
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

## References

- CGEBox: /Users/marmol/proyectos2/cge_babel/cgebox/gams/model/model.gms
- GTAP Database: https://www.gtap.agecon.purdue.edu/
- GTAP-E: Energy and environmental extension

## Implementation Notes

1. **GDX Reading**: Uses `equilibria.babel.gdx.reader` for native GDX file reading
2. **Pyomo**: All equations built as Pyomo model (sets, params, vars, constraints)
3. **Warm Start**: Variables initialized at benchmark (1.0) for faster convergence
4. **Scaling**: Automatic scaling of variables and equations
5. **Bounds**: Economic bounds applied (positive variables, etc.)

## Future Enhancements

- [ ] Melitz module (heterogeneous firms)
- [ ] Dynamic recursive mode
- [ ] Energy/environmental extension (GTAP-E)
- [ ] MyGTAP module (household heterogeneity)
- [ ] Sub-regional modeling (NUTS2)
- [ ] Welfare decomposition
- [ ] Sensitivity analysis

## License

Same as equilibria project.
