# GTAP — from zero to a tariff shock

The GTAP template implements the GTAP Standard 7 specification. It loads
sets, parameters and base-year levels directly from a GTAP GDX dataset
(e.g. `basedata-9x10.gdx`) and can be solved with IPOPT or PATH.

This guide covers:

1. Inspecting a GTAP dataset.
2. Building the model and solving the baseline.
3. Running a uniform 10 % tariff shock.
4. Comparing the result against a reference GAMS/NEOS solution.
5. Decomposing welfare changes (Huff/RunGTAP, optional WELVIEW.har).

## Prerequisites

```bash
pip install -e ".[pyomo,ipopt,excel]"
```

The HAR reader is native pure-Python (`equilibria.babel.har`) — no extra
needed to load the bundled HAR datasets.

`equilibria` ships two GTAP datasets — the canonical 9×10 GAMS
Standard 7 aggregation and a 3-region NUS333 (GTAPv7/GEMPACK). Both
travel as native HAR/PRM files inside the wheel and load via
`load_bundled("gtap", ...)`. To work with a custom aggregation, point
`load_from_har` (or `load_from_gdx`) at your own files.

## Step 1 — Inspect the dataset

```python
from equilibria import load_bundled

params = load_bundled("gtap", "9x10")  # or "nus333"
sets = params.sets

print(f"Aggregation: {sets.aggregation_name}")
print(f"Regions:      {sets.r}")
print(f"Commodities:  {sets.i}")
print(f"Sectors:      {sets.j}")
```

`load_bundled` reads the native HAR/PRM files (`basedata.har`,
`sets.har`, `default.prm`, plus optional `baserate.har`) and returns a
fully calibrated `GTAPParameters`. The 9×10 aggregation derives every
tax rate from `basedata.har` wedges (no `baserate.har` exists for it
upstream); NUS333 ships its own `baserate.har` from the GEMPACK pack.

## Step 2 — Build and solve the baseline

`GTAPModelEquations` assembles the Pyomo model from the calibrated
parameters; `GTAPSolver` runs PATH (default) or IPOPT.

```python
import equilibria
from equilibria import load_bundled
from equilibria.templates.gtap import GTAPSolver, build_gtap_contract
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

equilibria.setup_logging(level="INFO")

params = load_bundled("gtap", "9x10")
sets = params.sets

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
by 10 %: `tm_new = (1 + tm_old) * 1.1 − 1`. The `tm_pct` mode encodes
exactly that formula. The shock is applied directly on the
`GTAPParameters` containers *before* the model is built, so the
calibration and model assembly only need to be done once per experiment.

```python
import equilibria
from equilibria import load_bundled
from equilibria.templates.gtap import (
    GTAPSolver,
    apply_tariff_shock,
    build_gtap_contract,
)
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

equilibria.setup_logging(level="INFO")

# 1. Load base parameters once.
base_params = load_bundled("gtap", "9x10")
sets = base_params.sets
contract = build_gtap_contract("standard")

# 2. Solve the baseline.
baseline_eq = GTAPModelEquations(sets, base_params)
baseline_model = baseline_eq.build_model()
baseline_solver = GTAPSolver(
    baseline_model, closure=contract.closure, solver_name="path", params=base_params,
)
baseline_result = baseline_solver.solve()

# 3. Apply a uniform 10% tariff shock (GAMS-equivalent power scaling).
shocked_params = apply_tariff_shock(base_params, value=0.10, mode="tm_pct")

# 4. Solve the shocked model.
shock_eq = GTAPModelEquations(sets, shocked_params)
shock_model = shock_eq.build_model()
shock_solver = GTAPSolver(
    shock_model, closure=contract.closure, solver_name="path", params=shocked_params,
)
shocked_result = shock_solver.solve()

print(f"Baseline status:  {baseline_result.status}")
print(f"Shocked status:   {shocked_result.status}")
```

`apply_tariff_shock` deep-copies the parameters, applies the GAMS
formula `tm_new = (1 + tm_old) * (1 + value) - 1` to every
`imptx[source, commodity, dest]` entry (skipping the `source == dest`
diagonal so domestic sales stay untaxed), and keeps the legacy `rtms`
alias in sync. To restrict the shock to a subset, pass any combination
of `commodities=`, `sources=`, `destinations=` — for example,
`apply_tariff_shock(base_params, 0.10, commodities=["c_HeavyMnfc"], sources=["China"])`
only shocks Chinese heavy-manufacturing exports.

### Shocking other parameters

`apply_tariff_shock` is a thin wrapper around the generic
`apply_shock`, which can shock any registered container — not just
import tariffs. List the available targets at runtime:

```python
from equilibria.templates.gtap import apply_shock, list_shock_targets

list_shock_targets()
# ['taxes.imptx', 'taxes.rtf', 'taxes.rtfd', 'taxes.rtfi',
#  'taxes.rtgd', 'taxes.rtgi', 'taxes.rto', 'taxes.rtpd',
#  'taxes.rtpi', 'taxes.rtxs']
```

The signature is:

```python
apply_shock(
    params,
    target: str,           # e.g. "taxes.rto", "taxes.rtf"
    value: float,
    *,
    mode: ShockMode = "pct",   # "pct" | "power" | "set" | "add" | "mul"
    inplace: bool = False,
    **filters,             # commodities=, sources=, regions=, sectors=, factors=, destinations=
)
```

Each target advertises its own filter names (matching its tuple-key
dimensions); passing an unknown filter raises ``TypeError``. Examples:

```python
# +5% to the output-tax rate in every (region, sector) cell
apply_shock(base_params, "taxes.rto", 0.05, mode="pct")

# Set the factor tax on Land in Crops to zero, USA only
apply_shock(
    base_params,
    "taxes.rtf",
    0.0,
    mode="set",
    regions=["USA"],
    factors=["Land"],
    sectors=["c_Crops"],
)

# Power scaling on import tariffs (equivalent to apply_tariff_shock)
apply_shock(base_params, "taxes.imptx", 0.10, mode="power")
```

Modes:

| Mode | Formula | Typical use |
|------|---------|-------------|
| `pct` | `new = old * (1 + value)` | Scale a rate by a percentage |
| `power` | `new = (1 + old) * (1 + value) - 1` | GAMS-style tariff/tax shocks |
| `set` | `new = value` | Replace the rate outright |
| `add` | `new = old + value` | Additive perturbation |
| `mul` | `new = old * value` | Direct multiplicative override |
| `tm_pct` | alias of `power` | Legacy name, kept for `apply_tariff_shock` |

The diagonal-skip rule and the `rtms ↔ imptx` alias-sync that matter
for trade taxes are encoded in the registry, so they apply
automatically whenever `target="taxes.imptx"` or `target="taxes.rtxs"`.

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

## Step 5 — Welfare decomposition

Once you have a baseline and a shocked solution, the
`welfare_decomp` module computes the Huff (1996) / McDougall (2003)
decomposition that RunGTAP reports in `WELVIEW.har`. Total equivalent
variation (USD M) splits additively into allocative efficiency (`A`,
broken into 11 distortion sub-buckets), terms-of-trade (`T`),
investment-savings (`IS`), endowment (`ENDW`) and technical (`TECH`)
contributions:

```python
from equilibria.templates.gtap.welfare_decomp import (
    compute_welfare_decomposition,
    compute_welfare_decomposition_homotopy,
)

# Single-step (1–3 % residual vs RunGTAP — first-order approximation)
welfare = compute_welfare_decomposition(
    base_params=base_params,
    base_model=baseline_model,
    shock_params=shocked_params,
    shock_model=shock_model,
)

for region, comp in welfare.items():
    print(f"{region}: EV={comp.EV:+.1f}  A={comp.A_total:+.1f}  T={comp.T:+.1f}")
```

For RunGTAP-grade exactness (residual <0.01 %), use the homotopy
variant — pass the per-step models/params captured by
`_run_homotopy_shocked` in `scripts/gtap/run_gtap.py`:

```python
welfare = compute_welfare_decomposition_homotopy(
    base_params=base_params, base_model=baseline_model,
    step_params=step_params,           # list of N intermediate states
    step_models=step_models,
)
```

CLI:

```bash
uv run python scripts/gtap/run_gtap.py validate-shock \
    --gdx-file data/9x10/9x10Dat.gdx \
    --variable rtms --index "(Oceania,c_Crops,EastAsia)" --value 0.10 \
    --shock-mode tm_pct \
    --output reports/welfare/ \
    --welfare-decomp \
    --homotopy-steps 4 \
    --welfare-har reports/welfare/WELVIEW.har
```

This writes `welfare_decomposition.csv` (one row per region with all
sub-buckets) and an optional `WELVIEW.har` readable by `harview` /
`ViewHAR` / any GEMPACK tool.

See {doc}`welfare_decomposition` for the formulas, the 11-bucket
table, and an interpretation example.

## Closure and shock conventions

A few conventions baked into the template are worth knowing up front:

* **Residual region** — `NAmerica` is the GAMS-equivalent residual
  region (`rres`); the template pins the numeraire there. If you change
  the residual, also update the closure.
* **Solver mode** — for full Standard 7 (10,296 equations), always use
  PATH in *nonlinear full* mode; the linearised block is for diagnostics
  only.
* **Shock formula** — for parity with GAMS reference runs, call
  `apply_tariff_shock(..., mode="tm_pct")` (power scaling). The legacy
  `pct` mode scales only the rate and produces a smaller effective shock.
* **`equation_scaling=True`** — strongly recommended for both baseline
  and shocked runs; without it the baseline residual stalls at ~1e-6
  instead of ~1e-9.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `PATH executable was not resolved by Pyomo` | PATH is not on `PATH`; install via `pip install -e ".[pyomo]"` and ensure `pyomo --solvers` lists `path`. |
| Baseline residual ~1e-6 (expected ~1e-9) | `equation_scaling=True` was not passed to the PATH-CAPI helper. |
| Shocked run shows wrong sign on tariff variables | The shock was applied with `mode="pct"` instead of `mode="tm_pct"` in `apply_tariff_shock`. |
| GAMS parity comparison fails on `gdpmp` only | Known calibration trick in `cal.gms:652` overwrites `yi` deliberately; the Python template intentionally does not replicate it because doing so breaks convergence. See the parity status notes for context. |
