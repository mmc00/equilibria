# PEP — from zero to a solved baseline

The PEP (Politique Économique et Pauvreté) template implements the
PEP-1-1 v2.1 modular CGE model. It is calibrated against the standard
PEP SAM in GDX format and can be solved with IPOPT, PATH or a built-in
homotopy solver.

This guide walks through the minimal end-to-end flow:

1. Locate (or generate) a SAM in GDX format.
2. Calibrate the model.
3. Solve the baseline.
4. Inspect the solution and (optionally) compare against a GAMS reference.

## Prerequisites

```bash
pip install -e ".[pyomo,ipopt,excel]"
```

A reference SAM ships with the repository at
`src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx`. To use your
own SAM, pass the path explicitly.

## Step 1 — Calibrate

`PEPModelCalibrator` runs all calibration phases (income, production,
trade, final demand) in order. The result is a `PEPModelState` object
that contains every calibrated parameter and base-year level.

```python
import logging
from pathlib import Path

import equilibria
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator

equilibria.setup_logging(level="INFO")

REPO = Path(equilibria.__file__).resolve().parents[2]
SAM_FILE = REPO / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"

calibrator = PEPModelCalibrator(sam_file=SAM_FILE)
state = calibrator.calibrate()

print(state.report.summary())
```

The `PEPModelState` carries:

* `state.sets` — every set (`i`, `j`, `h`, `r`, …) with its elements;
* `state.parameters` — calibrated elasticities, taxes, shares;
* `state.variables_init` — base-year levels used as the warm start;
* `state.report` — per-phase pass/fail status and any validation
  warnings that surfaced during calibration.

## Step 2 — Solve the baseline

`PEPModelSolver` consumes the calibrated state and runs the equation
system to convergence. Two initialization modes are supported:

| `init_mode` | Source of the warm start |
|-------------|--------------------------|
| `"excel"` (default) | Calibrated `*O` levels straight from the SAM |
| `"gams"` | Read levels from a reference Results GDX |

```python
from equilibria.templates.pep_model_solver import PEPModelSolver

solver = PEPModelSolver(
    calibrated_state=state,
    tolerance=1e-6,
    max_iterations=200,
    init_mode="excel",
)
solution = solver.solve()

if solution.converged:
    print(f"GDP_BP = {solution.variables.GDP_BP:,.2f}")
    print(f"Walras residual = {solution.walras:.2e}")
else:
    print("Did not converge — inspect solution.diagnostics")
```

## Step 3 — Inspect or persist the solution

`solution.variables` is a `PEPModelVariables` dataclass with named
attributes for every endogenous variable. Convert to a flat dictionary
or a pandas frame for further analysis:

```python
import pandas as pd

flows = pd.Series(
    {key: getattr(solution.variables, key)
     for key in solution.variables.__dataclass_fields__},
    name="value",
)
flows.head(10)
```

To save the full state for later runs:

```python
import json
Path("out/pep_solution.json").write_text(json.dumps(solution.to_dict(), indent=2))
```

## Step 4 — (Optional) GAMS parity check

If you have access to a reference Results GDX produced by the PEP-1-1
GAMS model, you can compare variable-by-variable:

```python
from equilibria.templates.gams_comparison import run_gams_comparison

report = run_gams_comparison(
    python_solution=solution,
    gams_results_gdx=REPO / "src/equilibria/templates/reference/pep2/scripts/Results.gdx",
    rel_tol=1e-4,
)
print(report.summary())
```

## CLI shortcuts

For day-to-day work, the helper scripts under `scripts/cli/` wrap the
same calls:

```bash
# Calibrate + solve with default SAM
python scripts/cli/run_solver.py --verbose

# Calibrate + solve from your own SAM
python scripts/cli/run_solver.py \
    --sam-file data/my_sam.gdx \
    --save-solution out/my_solution.json
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Calibration phase fails immediately | The SAM does not include every set element the calibrator expects. Run with `logging.DEBUG` to see which symbol is missing, then either patch the SAM or pass `sets=...` explicitly. |
| Solver converges but Walras residual is large | The closure may pin the wrong numeraire. Try `contract="standard"` to use the canonical PEP closure or pass an explicit `PEPClosureConfig`. |
| `IPOPT not available` | Install the optional extra: `pip install -e ".[ipopt]"`. The solver falls back to a Newton-style iteration if IPOPT cannot be imported, but it is much slower. |
