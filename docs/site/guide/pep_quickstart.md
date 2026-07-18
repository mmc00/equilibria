# PEP — from zero to a solved baseline

The PEP (Politique Économique et Pauvreté) template implements the
PEP-1-1 v2.1 modular CGE model. It calibrates against the canonical
PEP SAM and can be solved with IPOPT, PATH or a built-in homotopy
solver.

This guide walks through the minimal end-to-end flow:

1. Calibrate from the bundled SAM.
2. Solve the baseline.
3. Inspect the solution and (optionally) compare against a GAMS reference.

## Prerequisites

```bash
pip install -e ".[pyomo,ipopt,excel]"
```

The bundled PEP dataset (`SAM-V2_0.xlsx` + `VAL_PAR.xlsx`) ships inside
the wheel and loads via `load_bundled("pep")`. To use a custom SAM,
build a `PEPModelCalibrator` directly with your own paths.

## Step 1 — Calibrate

`load_bundled("pep")` reads the canonical Excel SAM, materialises the
4D GDX layout the calibrator consumes (cached under
`~/.cache/equilibria/pep/`), and returns a `PEPModelCalibrator` ready
to run. The result of `.calibrate()` is a `PEPModelState` with every
calibrated parameter and base-year level.

```python
import equilibria
from equilibria import load_bundled

equilibria.setup_logging(level="INFO")

calibrator = load_bundled("pep")
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

## Pyomo port — NLP and MCP forms

PEP-1-1 is also available as a **Pyomo** model
(`equilibria.templates.pep_pyomo`), unifying it with the GTAP template on a
single engine (the original PEP solver used cyipopt). `build_pep_model` takes
a `variant` (`"base"` or `"objdef"`, which adds a dummy `OBJDEF: OBJ==0`
objective) and a `form`:

- **`form="nlp"`** — the equality system solved by IPOPT on the raw model
  (`nlp_scaling_method=none`, faithful to GAMS's raw solve).
- **`form="mcp"`** — the same system as a mixed complementarity problem
  (WALRAS ⊥ LEON free-row, `e` fixed as numeraire) solved by PATH. This is the
  first PEP-MCP.

```python
from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep

model = build_pep_model(state, variant="base", form="mcp")
result = solve_pep(model)          # code=1 at the calibrated benchmark
```

To run the reference SIM1 counterfactual (a 25% export-tax cut), apply the
shock to the calibrated state **before** building — this scales the `ttixO`
benchmark exactly as GAMS's `ttix.fx = ttixO*0.75` does:

```python
from equilibria.templates.pep_pyomo.pep_pyomo_scenarios import apply_sim1_export_tax_cut

apply_sim1_export_tax_cut(state)   # ttixO *= 0.75, in place
model = build_pep_model(state, variant="base", form="mcp")
result = solve_pep(model)          # GDP_BP moves 46707 → 46748.2
```

Both forms are validated cell-by-cell against a GAMS reference solved by the
same engine, and the two forms mirror each other exactly — see the
{doc}`PEP coverage matrix <pep_coverage_matrix>`.

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
