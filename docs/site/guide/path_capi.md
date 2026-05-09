# Solving with PATH via the C API

For full GTAP Standard 7 (10,296 equations) and other large MCP models
where Pyomo's PATH driver hits scaling limits, `equilibria` integrates
with [`path-capi-python`](https://github.com/mmc00/path-capi-python),
a thin Python wrapper around the official PATH C API.

The wrapper drives the solver through residual and Jacobian callbacks
instead of writing GMS files to disk, which yields:

- Convergence on the full nonlinear system without first linearising;
- Deterministic residual diagnostics (per-equation, per-iteration);
- Direct interop with the same Pyomo model `equilibria` already builds.

```{important}
PATH itself is third-party software with its own license. The wrapper
is MIT-licensed for the bridge code only — you still need a valid
PATH license (full or restricted) to solve.
```

## When to use it

| Scenario | Recommended driver |
|----------|--------------------|
| Small / medium models, exploratory work | `GTAPSolver(..., solver_name="path")` (Pyomo PATH driver) |
| GTAP 9 × 10 baseline & tariff shocks for GAMS parity | `path-capi-python` via the nonlinear-full helper |
| IPOPT-friendly problems with strict NLP structure | `GTAPSolver(..., solver_name="ipopt")` |

If you don't have `path-capi-python` installed, every example in the
{doc}`GTAP quickstart <gtap_quickstart>` still works — `solver_name="path"`
goes through Pyomo's regular PATH plugin.

## Installation

`path-capi-python` is a separate project. Clone it next to `equilibria`
and install in the same environment:

```bash
git clone https://github.com/mmc00/path-capi-python.git
cd path-capi-python
pip install -e .
```

The wrapper expects the PATH shared library to be findable at runtime.
Two ways to point it at the right binary:

```bash
# Recommended: explicit path to the .dylib / .so / .dll
export PATH_CAPI_LIBPATH=/path/to/libpath50.dylib
export PATH_CAPI_LIBLUSOL=/path/to/liblusol.dylib   # macOS only

# Or rely on the loader's auto-detection (uses Pyomo's PATH location).
```

If you have a license string, set it before solving:

```bash
export PATH_LICENSE_STRING='<your license string>'
```

## Minimal nonlinear MCP solve

The wrapper exposes three entry points that `equilibria` uses:

| Symbol | Purpose |
|--------|---------|
| `PATHLoader` | Discover and validate the PATH runtime |
| `PyomoMCPAdapter` | Build residual/Jacobian callbacks from a Pyomo model |
| `solve_nonlinear_mcp` | Drive PATH through those callbacks |

A minimal hand-rolled solve looks like this:

```python
from path_capi_python import (
    PATHLoader,
    PyomoMCPAdapter,
    solve_nonlinear_mcp,
)

# 1. Load the PATH library once per process.
runtime = PATHLoader().load()

# 2. Wrap a Pyomo MCP model. The adapter pairs every constraint with
#    a complementary variable (uses the model's bounds and `fixed` flags).
adapter = PyomoMCPAdapter(model)
callback_data = adapter.build_nonlinear_callbacks()

# 3. Solve. The result mirrors PATH's status codes (1 = converged) and
#    reports the final residual at the solution.
result = solve_nonlinear_mcp(
    runtime,
    callback_data,
    convergence_tol=1e-8,
    major_iterations=500,
)

if result.status == 1:
    adapter.write_back(result.solution, model)
    print(f"Converged: residual={result.residual:.2e}")
else:
    print(f"PATH status={result.status}: {result.message}")
```

## Calling the GTAP nonlinear-full helper

For the GTAP Standard 7 workflow, `equilibria` ships a higher-level
helper that wires up the closure, conditional fixing, equation
scaling, and warm-start handling required to reach GAMS parity. The
helper currently lives in `scripts/gtap/run_gtap.py` (it depends on
patches that are GTAP-specific, so it has not been promoted to the
public package yet) and is reachable as `_run_path_capi_nonlinear_full`.

```python
import sys
from pathlib import Path

# Add the scripts dir so the GTAP-specific helpers import cleanly.
SCRIPTS = Path(__file__).resolve().parent / "scripts" / "gtap"
sys.path.insert(0, str(SCRIPTS))

from run_gtap import (
    _build_gtap_contract_with_calibration,
    _run_path_capi_nonlinear_full,
)

contract, sets, params, model = _build_gtap_contract_with_calibration(
    gdx_file=Path("path/to/basedata-9x10.gdx"),
)

baseline = _run_path_capi_nonlinear_full(
    model,
    params,
    closure_config=contract.closure,
    equation_scaling=True,            # required for residual ~1e-9
    path_capi_convergence_tol=1e-8,
    jacobian_eval_mode="reverse_numeric",
    solver_output=False,
)

print(f"PATH status: {baseline['status']}, residual: {baseline['residual']:.2e}")
```

Then re-use `baseline["solution_hint"]` as a warm start for the shocked
model:

```python
shocked = _run_path_capi_nonlinear_full(
    shocked_model,
    shocked_params,
    closure_config=contract.closure,
    equation_scaling=True,
    solution_hint=baseline["solution_hint"],
)
```

## Recommended invariants

These are non-negotiable for GAMS parity — silently dropping any of
them puts the solver in a slower / less precise regime:

* **`equation_scaling=True`** for both baseline and shocked runs.
  Without it the baseline residual stalls at ~1e-6 instead of ~1e-9.
* **`jacobian_eval_mode="reverse_numeric"`** unless you have measured a
  better mode for your specific dataset.
* **Convergence tolerance ≤ 1e-8.** The strict gate that compares
  Python ↔ GAMS deltas assumes residuals at this level.
* **Warm start from baseline.** Cold-starting the shocked model from
  GAMS levels works but takes much longer and occasionally lands in a
  different fixed point.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `Unable to import path_capi_python` | The wrapper is not installed in the active environment, or the project lives at a non-default path. Install with `pip install -e .` from the wrapper repo. |
| `PATH license has expired` / `code=11` | Set `PATH_LICENSE_STRING` to a current license, or fall back to the restricted-license demo for tiny problems. |
| Residual stuck at ~1e-6 | `equation_scaling` not enabled. Pass `equation_scaling=True`. |
| Different solution from GAMS even at low residual | Check the closure: residual region (`NAmerica`) must match `rres`, and the shock formula must use the `tm_pct` (power) form, not legacy `pct`. |
