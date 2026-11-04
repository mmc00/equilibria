# SimpleOpen parity contract

This guide pins the canonical economic contract of `SimpleOpen` for the future parity against GAMS.

The idea is to make three things explicit:

- which contract counts as official
- which is the canonical baseline closure
- which numeric benchmark Python and GAMS must share

## Canonical contract

Today the official parity contract is:

- `contract.name = "simple_open_v1"`

That contract solves a minimal system of three equations:

- `EQ_VA`
- `EQ_INT`
- `EQ_CET`

And it uses this canonical variable order:

- `VA`
- `INT`
- `X`
- `D`
- `E`
- `ER`
- `PFX`
- `CAB`
- `FSAV`

That order is already formalized in:

- [simple_open_parity_spec.py](../../../src/equilibria/templates/simple_open_parity_spec.py)

## Canonical baseline closure

The base closure for parity is:

- `closure.name = "simple_open_default"`

Interpretation:

- numeraire: `PFX`
- `PFX` and `FSAV` fixed
- `ER` and `CAB` endogenous
- capital mobility: `mobile`

There is also a secondary benchmark closure:

- `flexible_external_balance`

But the base parity reference must start from `simple_open_default`.

## Canonical benchmark

Since `SimpleOpen` does not come from a calibrated SAM, the benchmark is defined by a small block of parameters and exogenous levels.

### simple_open_default

- `alpha_va = 0.40`
- `rho_va = 0.75`
- `a_int = 0.50`
- `b_ext = 0.10`
- `theta_cet = 0.60`
- `phi_cet = 1.20`
- `ER = 1.00`
- `PFX = 1.00`
- `D = 1.00`
- `E = 1.00`
- `CAB = 1.00`
- `FSAV = 1.00`

### flexible_external_balance

- `alpha_va = 0.45`
- `rho_va = 0.70`
- `a_int = 0.55`
- `b_ext = 0.08`
- `theta_cet = 0.58`
- `phi_cet = 1.25`
- `ER = 1.08`
- `PFX = 1.00`
- `D = 1.04`
- `E = 0.93`
- `CAB = 0.82`
- `FSAV = 0.82`

The benchmark levels derived from those parameters are already built in:

- [simple_open_parity_spec.py](../../../src/equilibria/templates/simple_open_parity_spec.py)

And the Jacobian harness already consumes exactly that same specification in:

- [simple_open_constraint_jacobian.py](../../../src/equilibria/templates/simple_open_constraint_jacobian.py)

## Why this guide matters before GAMS

Without this explicit specification, the next GAMS step would be ambiguous:

- a `.gms` could use another closure
- another variable order
- or a numeric benchmark different from Python's

With this guide, the parity contract is fixed before the GAMS model is written.

## API and helpers

Useful helpers:

```python
from equilibria.templates import (
    build_simple_open_contract,
    build_simple_open_runtime_config,
)
from equilibria.templates.simple_open_parity_spec import build_simple_open_parity_spec

contract = build_simple_open_contract("simple_open_v1")
runtime = build_simple_open_runtime_config("default_template")
spec = build_simple_open_parity_spec(contract)
```

The resulting `spec` has ready:

- `contract_name`
- `closure_name`
- `equation_names`
- `variable_names`
- `benchmark_parameters`
- `benchmark_levels`

## Criterion for the next cut

The next cut of the epic must use exactly this contract to:

1. write the minimal `.gms`
2. export the benchmark levels
3. compare Python vs GAMS without reinterpreting the system
