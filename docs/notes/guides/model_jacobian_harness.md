# Model Jacobian harness

This guide summarizes the common layer we use today for Jacobian benchmarks and gates in `equilibria`.

The idea is to separate two things:

- the common logic for evaluating, structuring and reporting the Jacobian
- the math specific to each model

## What is already generic

The common layer lives in:

- `equilibria.solver.jacobians`

Main pieces:

- `ConstraintJacobianHarness`
- `ConstraintJacobianStats`
- `ConstraintJacobianSolverStats`
- `ConstraintJacobianStructure`
- `solver_stats_payload(...)`
- `summarize_jacobian_mode_entry(...)`
- `compare_jacobian_modes(...)`
- `evaluate_jacobian_mode_gate(...)`

Put simply:

- the common harness knows how to evaluate residuals, the sparse structure and the values of the Jacobian
- it can mix analytic derivatives with a numeric fallback
- the reporting/gate layer knows how to compare `analytic` vs `numeric` without depending on PEP

## What is still model-specific

Each model still defines:

- its variables
- its equations
- its benchmark/reference point
- its analytic derivatives per row

Today there are two real implementations on top of the base:

- PEP: `equilibria.templates.pep_constraint_jacobian`
- SimpleOpen: `equilibria.templates.simple_open_constraint_jacobian`

## Minimum contract of a child implementation

A child implementation of the harness must provide three things:

1. how to build the context at a point `x`
2. how to return the residual of each equation in a stable order
3. which analytic derivatives it knows for a given equation

That maps to these hooks:

- `_build_context(...)`
- `_calculate_constraint_residual_dict(...)`
- `_analytic_constraint_derivatives(...)`

If a row has no analytic derivative, the common harness falls back to finite differences for that row.

## What the common layer validates today

### pep

Script:

- `scripts/parity/measure_pep_jacobian_modes.py`

It checks:

- convergence of `analytic` and `numeric`
- that `analytic` does not use finite differences above the allowed maximum
- that the parity of `analytic` is no worse than that of `numeric`

### simpleopen

Script:

- `scripts/parity/measure_simple_open_jacobian_modes.py`

It checks:

- exact benchmark residual in two closures
- the same sparse structure in `analytic` and `numeric`
- that `analytic` uses `0` finite differences
- that `numeric` is no better than `analytic` against the analytic reference

## When to use each benchmark

- use `measure_pep_jacobian_modes.py` when the change touches the PEP solver or its parity
- use `measure_simple_open_jacobian_modes.py` when the change touches the common layer of the harness or of the reporting/gate

If you change the common base and only PEP runs, you have not yet checked that the abstraction works outside PEP.

## Recommended pattern for a third model

1. create `<model>_constraint_jacobian.py`
2. inherit from `ConstraintJacobianHarness`
3. expose a small, deterministic benchmark
4. add a `measure_<model>_jacobian_modes.py` script
5. reuse:
   - `solver_stats_payload`
   - `compare_jacobian_modes`
   - `evaluate_jacobian_mode_gate`

## Useful imports

```python
from equilibria.solver import (
    ConstraintJacobianHarness,
    solver_stats_payload,
    compare_jacobian_modes,
    evaluate_jacobian_mode_gate,
)
```

And for the second example model:

```python
from equilibria.templates import SimpleOpenConstraintJacobianHarness
```
