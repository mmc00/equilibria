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

contract = build_gtap_contract()  # default closure

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

Shocks run on the **multi-period** model, which replicates the GAMS
`loop(tsim)`: it solves `base → check → shock` in one model. The shock
lives in the `ShockBlock`: every policy or technology instrument (`imptx`,
`lambdava`, `aft`, ...) is a fixed variable per period, and a shock is its
`shock`-period cell differing from the `check` one. You write the shock on the
built model with `apply_shock` and the driver reads it from there.

The example uses the `gtap7_3x3` dataset shipped in the repository
(`datasets/gtap7_3x3`):

```python
from pathlib import Path

from pyomo.environ import value

from equilibria.templates.gtap import GTAPParameters, apply_shock
from equilibria.templates.gtap.gtap_block_model import build_block_model
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

data = Path("datasets/gtap7_3x3")
params = GTAPParameters()
params.load_from_har(
    basedata_path=data / "basedata.har",
    sets_path=data / "sets.har",
    default_path=data / "default.prm",
    baserate_path=data / "baserate.har",
)
closure = GTAPClosureConfig(
    name="base",
    closure_type="MCP",
    capital_mobility="sluggish",
    fix_endowments=False,
    fix_taxes=False,
    fix_technology=False,
    if_sub=False,
    numeraire="pnum",
)
m, _ = build_block_model(params, params.sets, closure, "RestofWorld")

# +10% on the tariff power (1 + tms) of one route, as in a GEMPACK .EXP:
route = ("EU_28", "Mnfcs", "USA")  # (exporter, commodity, importer)
apply_shock(m, {"imptx": {route: 10.0}})

# GAMS compStat does not solve the base: its levels are the benchmark.
results = solve_multiperiod(
    m, params, closure, mode="gtap", skip_base_solve=True, solve_check=True
)
print(results["shock"])  # {'code': 1, 'residual': ...}
for t in ("check", "shock"):
    print(t, value(m.imptx[(*route, t)]), value(m.xw[(*route, t)]))
# check 0.0145... 0.382...
# shock 0.1160... 0.198...
```

`apply_shock(m, {instrument: {cell: pct}})` takes each number as the GEMPACK
percentage change and converts it to levels according to the instrument
(`shocks.GEMPACK_KIND`), starting from the `check` value:

| Kind | Instruments | Level in the shock period |
|------|-------------|---------------------------|
| `pct` | `lambdava`, `aft`, `pop`, `lambdaf`, `axp`, `lambdam`, `lambdamg` | `x * (1 + p/100)` |
| `power` | `imptx`, `exptx`, `prdtx_rai`, `dintx_tgt`, `mintx_tgt` | `(1 + t) * (1 + p/100) - 1` |
| `power_kappa` | `kappaf` | `1 - (1 - k) / (1 + p/100)` |
| `power_fct` | `fcttx` | `fcttx` absorbs the change of `1 + fctts + fcttx` |

Several instruments and cells go in one call. To give the level directly,
pass `levels=True`. Applying the same shock twice gives the same model as
applying it once. A cell no live equation reads, a misspelt label or an
out-of-domain value raises `ValueError` before any solve. With `if_sub=True`
the `imptx` cells are rejected for now (the tariff enters through inlined
macros).

Without any shock in the `ShockBlock`, `solve_multiperiod` applies the
reference GAMS run's default: +10% on the power of every import tariff.

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
* **Shock formula** — `apply_shock` reads each number as the GEMPACK %
  change and picks the conversion from the instrument: tax shocks act on
  the power `1 + t`, as in GAMS (see Step 3).
* **`equation_scaling=True`** — strongly recommended for both baseline
  and shocked runs; without it the baseline residual stalls at ~1e-6
  instead of ~1e-9.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `PATH executable was not resolved by Pyomo` | PATH is not on `PATH`; install via `pip install -e ".[pyomo]"` and ensure `pyomo --solvers` lists `path`. |
| Baseline residual ~1e-6 (expected ~1e-9) | `equation_scaling=True` was not passed to the PATH-CAPI helper. |
| GAMS parity comparison fails on `gdpmp` only | Known calibration trick in `cal.gms:652` overwrites `yi` deliberately; the Python template intentionally does not replicate it because doing so breaks convergence. See the parity status notes for context. |
