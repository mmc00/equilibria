# GTAP — from zero to a tariff shock

The GTAP template implements the GTAP Standard 7 specification. It loads
sets, parameters and base-year levels directly from a GTAP GDX dataset
(e.g. `basedata-9x10.gdx`) and can be solved with IPOPT or PATH.

This guide covers:

1. Inspecting a GTAP dataset.
2. Building the model and solving the baseline.
3. Running a uniform 10 % tariff shock.
4. Comparing the result against a reference GAMS/NEOS solution.

## Prerequisites

```bash
pip install -e ".[pyomo,ipopt,excel,har]"
```

A 9 × 10 GTAP dataset is required. The repository expects
`basedata-9x10.gdx` (equivalent to `9x10Dat.gdx`); place it under
`src/equilibria/templates/reference/gtap/data/` or pass its path
explicitly.

## Step 1 — Inspect the dataset

The bundled CLI prints set sizes, region/commodity names, and a sanity
summary of base-year flows:

```bash
python scripts/gtap/run_gtap.py info \
    --gdx-file path/to/basedata-9x10.gdx
```

Programmatic equivalent:

```python
from equilibria.templates.gtap import GTAPSets

sets = GTAPSets()
sets.load_from_gdx("path/to/basedata-9x10.gdx")

print(f"Aggregation: {sets.aggregation_name}")
print(f"Regions:      {sets.r}")
print(f"Commodities:  {sets.i}")
print(f"Sectors:      {sets.j}")
```

## Step 2 — Build and solve the baseline

`GTAPParameters` loads elasticities and taxes from the same GDX;
`GTAPModelEquations` assembles the Pyomo model; `GTAPSolver` runs PATH
(default) or IPOPT.

```python
from pathlib import Path

import equilibria
from equilibria.templates.gtap import (
    GTAPSets,
    GTAPParameters,
    GTAPSolver,
    build_gtap_contract,
)
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

equilibria.setup_logging(level="INFO")

GDX = Path("path/to/basedata-9x10.gdx")

sets = GTAPSets()
sets.load_from_gdx(GDX)

params = GTAPParameters()
params.load_from_gdx(GDX)

contract = build_gtap_contract("standard")  # default closure

equations = GTAPModelEquations(sets, params)
model = equations.build_model()

solver = GTAPSolver(
    model,
    closure=contract.closure,
    solver_name="path",   # or "ipopt"
    params=params,
)
result = solver.solve()
print(f"Status: {result.status}, residual: {result.residual:.2e}")
```

## Step 3 — Run a tariff shock

The reference GAMS run shocks the *power* of import tariffs uniformly
by 10 %: `tm_new = (1 + tm_old) * 1.1 − 1`. In Python this is encoded
as `--shock-mode tm_pct` in the CLI, or directly when constructing the
shock:

```bash
python scripts/gtap/run_gtap.py validate-shock \
    --gdx-file path/to/basedata-9x10.gdx \
    --shock-variable tm \
    --shock-mode tm_pct \
    --shock-value 0.10 \
    --output out/shock.json
```

The CLI runs the baseline, applies the shock, re-solves, and writes a
JSON report with both pre- and post-shock variable levels. Internally
this is :func:`scripts.gtap.run_gtap.validate_shock`, which calls the
same `GTAPSolver` you used above.

## Step 4 — GAMS parity check

The `gtap_parity_pipeline` module turns a Python solution into a
side-by-side comparison against a reference GAMS GDX:

```python
from equilibria.templates.gtap.gtap_parity_pipeline import run_gtap_parity_test

comparison = run_gtap_parity_test(
    python_solution=result,
    gams_gdx=Path("reference/out.gdx"),
    rel_tol=1e-4,
)

print(f"Mismatches: {comparison.n_mismatches}")
for mismatch in comparison.top_mismatches(10):
    print(f"  {mismatch.group}{mismatch.key}: diff={mismatch.abs_diff:.2e}")
```

## Closure and shock conventions

A few conventions baked into the template are worth knowing up front:

* **Residual region** — `NAmerica` is the GAMS-equivalent residual
  region (`rres`); the template pins the numeraire there. If you change
  the residual, also update the closure.
* **Solver mode** — for full Standard 7 (10,296 equations), always use
  PATH in *nonlinear full* mode; the linearised block is for diagnostics
  only.
* **Shock formula** — for parity with GAMS reference runs, use
  `--shock-mode tm_pct` (power scaling). The legacy `pct` mode scales
  only the rate and produces a smaller effective shock.
* **`equation_scaling=True`** — strongly recommended for both baseline
  and shocked runs; without it the baseline residual stalls at ~1e-6
  instead of ~1e-9.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `PATH executable was not resolved by Pyomo` | PATH is not on `PATH`; install via `pip install -e ".[pyomo]"` and ensure `pyomo --solvers` lists `path`. |
| Baseline residual ~1e-6 (expected ~1e-9) | `equation_scaling=True` was not passed; the CLI sets it automatically, but custom scripts must opt in. |
| Shocked run shows wrong sign on tariff variables | The shock was applied with `pct` instead of `tm_pct`. Verify the CLI flag or the call to `_apply_shock_to_params`. |
| GAMS parity comparison fails on `gdpmp` only | Known calibration trick in `cal.gms:652` overwrites `yi` deliberately; the Python template intentionally does not replicate it because doing so breaks convergence. See the parity status notes for context. |
