# GTAP — from zero to a tariff shock

The GTAP template implements the GTAP Standard 7 specification. You build
the model from its blocks (see {doc}`architecture`), write a shock on it, and
solve `base → check → shock` in one multi-period model, as the GAMS
`loop(tsim)` does.

This guide covers:

1. Inspecting a GTAP dataset.
2. Building the model and running a tariff shock.
3. Reading the welfare effect.
4. Checking the result against GAMS.

## Prerequisites

```bash
pip install -e ".[pyomo,ipopt,excel]"
```

The model is solved as an MCP with PATH through its C API. equilibria finds
the `path-capi-python` checkout and the PATH libraries as described in
{doc}`path_capi`.

`equilibria` ships two GTAP datasets as native HAR/PRM files inside the
wheel, loaded with `load_bundled("gtap", ...)`:

- `"9x10"`: the canonical GAMS Standard 7 aggregation, 10 regions × 10
  commodities.
- `"nus333"`: the Burfisher NUS333 database (GTAPv7/GEMPACK), 2 regions
  (`USA`, `ROW`) × 3 commodities.

To work with your own aggregation, point `GTAPParameters.load_from_har` at
your files (Step 4 shows it).

## Step 1 — Inspect the dataset

```python
from equilibria import load_bundled

params = load_bundled("gtap", "nus333")  # or "9x10"
sets = params.sets

print("Regions:    ", sets.r)
print("Commodities:", sets.i)
print("Activities: ", sets.a)
print("Factors:    ", sets.f)
# Regions:     ['USA', 'ROW']
# Commodities: ['AGR', 'MFG', 'SER']
# Activities:  ['AGR', 'MFG', 'SER']
# Factors:     ['LAND', 'LABOR', 'CAPITAL']
```

`load_bundled` reads `basedata.har`, `sets.har`, `default.prm` and, when the
dataset has one, `baserate.har`, and returns a calibrated `GTAPParameters`.

## Step 2 — Build the model and run a tariff shock

`build_block_model` composes the GTAP blocks into one multi-period model.
The shock lives in the `ShockBlock`: every policy or technology instrument
(`imptx`, `lambdava`, `aft`, ...) is a fixed variable per period, and a shock
is its `shock`-period cell differing from the `check` one. You write it on the
built model with `apply_shock`, and `solve_multiperiod` reads it from there.

```python
from pyomo.environ import value

from equilibria.templates.gtap import apply_shock
from equilibria.templates.gtap.gtap_block_model import build_block_model
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

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
m, _ = build_block_model(params, sets, closure, "ROW")  # "ROW": the residual region

# +10% on the tariff power (1 + tms) of one route, as in a GEMPACK .EXP:
route = ("USA", "MFG", "ROW")  # (exporter, commodity, importer)
apply_shock(m, {"imptx": {route: 10.0}})

# GAMS compStat does not solve the base: its levels are the benchmark.
results = solve_multiperiod(
    m, params, closure, mode="gtap", skip_base_solve=True, solve_check=True
)
print(results["shock"])  # {'code': 1, 'residual': ...}
for t in ("check", "shock"):
    print(t, value(m.xw[(*route, t)]))
# check 0.9668...
# shock 0.6811...
```

`check` must replicate `base`: it is the benchmark solved again, and a
`check` that moves means the model does not reproduce its own data. `shock`
is the counterfactual.

`apply_shock(m, {instrument: {cell: pct}})` takes each number as the GEMPACK
percentage change and converts it to levels according to the instrument
(`shocks.GEMPACK_KIND`), starting from the `check` value:

| Kind | Instruments | Level in the shock period |
|------|-------------|---------------------------|
| `pct` | `lambdava`, `aft`, `pop`, `lambdaf`, `axp`, `lambdam`, `lambdamg`, `pnum` | `x * (1 + p/100)` |
| `power` | `imptx`, `exptx`, `prdtx_rai`, `dintx_tgt`, `mintx_tgt` | `(1 + t) * (1 + p/100) - 1` |
| `power_kappa` | `kappaf` | `1 - (1 - k) / (1 + p/100)` |
| `power_fct` | `fcttx` | `fcttx` absorbs the change of `1 + fctts + fcttx` |

Several instruments and cells go in one call. To give the level directly,
pass `levels=True`. Applying the same shock twice gives the same model as
applying it once. A cell no live equation reads, a misspelt label or an
out-of-domain value raises `ValueError` before any solve. With `if_sub=True`
an `imptx` cell is read by `eq_xweq` (the tariff enters through the inlined
import price), so it takes a shock as with `if_sub=False`.

Without any shock in the `ShockBlock`, `solve_multiperiod` writes the reference
GAMS run's default there (`shocks.apply_default_tariff`): in `mode="gtap"`,
+10% on the power of every import tariff, the domestic diagonal included; in
`mode="altertax"`, every tariff rate times 1.1. It is a shock like any other:
the equations are not rebuilt.

An exercise that needs a different closure (a real-wage target, a quantity
target) changes one block in the shock period with `@overwrite`; see
{doc}`architecture`.

## Step 3 — Read the welfare effect

`regional_ev` computes the equivalent variation of each regional household
as GTAP defines it (`gtapv7.tab` §11): the income that, at the prices before
the shock, reaches the utility after it, minus the income before. It works on
the solved model and leaves it as it was.

```python
from equilibria.templates.gtap.welfare_regional import regional_ev

print(regional_ev(m))  # in units of regy
# {'USA': -0.1079..., 'ROW': 0.0885...}
```

The exporter loses and the region that raised the tariff gains, as expected.
`regional_ev` is tested against GEMPACK's EV on five exercises from the
Burfisher book, within 0.1 % of income. The decomposition of EV into
allocative, terms-of-trade and other contributions is in
{doc}`welfare_decomposition`.

## Step 4 — Check the result against GAMS

This step needs a checkout of the repository: it reads the 3×3 dataset in
`data/gtap7/gtap7_3x3` and the GAMS solution of its default shock in
`tests/fixtures/`. `equilibria.compare` lines the two solutions up cell by
cell; the mapping file says which GAMS name matches which equilibria name.

```python
from pathlib import Path

from equilibria.compare import Mapping, compare, from_gdx, from_pyomo
from equilibria.templates.gtap import GTAPParameters

data = Path("data/gtap7/gtap7_3x3")
params3 = GTAPParameters()
params3.load_from_har(
    basedata_path=data / "basedata.har",
    sets_path=data / "sets.har",
    default_path=data / "default.prm",
    baserate_path=data / "baserate.har",
)
m3, _ = build_block_model(params3, params3.sets, closure, "ROW")
solve_multiperiod(m3, params3, closure, mode="gtap", skip_base_solve=True, solve_check=True)

report = compare(
    from_gdx("tests/fixtures/gtap7/gtap7_3x3/out_gtap_shock_ifsub0.gdx", period_dim=-1),
    from_pyomo(m3, period_dim=-1),
    Mapping.from_yaml("tests/compare/fixtures/gtap3x3_minimo.yaml"),
)
for t in ("check", "shock"):
    cells = [c for c in report.cells if c.period == t]
    print(t, sum(c.passed for c in cells), "of", len(cells), "cells within 0.1 %")
# check 21 of 21 cells within 0.1 %
# shock 21 of 21 cells within 0.1 %
```

No shock is applied here, so the driver writes the GAMS default (+10 % on
every tariff power), the same shock GAMS solved. Compare `check` and
`shock`, the periods both sides solve: in `compStat` GAMS does not solve
`base`, and its `base` levels are the initial values from `cal.gms`, not an
equilibrium.

The full set of gates (MCP, NLP and the `.nl` structure against GAMS, levels
against GEMPACK) runs with:

```bash
uv run python scripts/gtap/run_parity_gates.py --quick
```

## Conventions

* **Residual region** — the region whose income equation is replaced by
  Walras's law (GAMS's `rres`). Pass it explicitly, as above. Under the
  `capFix` savings closure `"auto"` picks the region with the largest base
  investment, which helps PATH on large aggregations.
* **Shock formula** — `apply_shock` reads each number as the GEMPACK %
  change and picks the conversion from the instrument: tax shocks act on
  the power `1 + t`, as in GAMS (see Step 2).
* **Blocks only** — `solve_multiperiod` needs a model built by
  `build_block_model`; the old single-file model
  (`gtap_model_equations.py`) is not solved by the driver.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `path_capi_python` cannot be imported | The `path-capi-python` checkout is not where equilibria looks; set `EQUILIBRIA_PATH_CAPI_SRC` (see {doc}`path_capi`). |
| PATH cannot find its library | Set `EQUILIBRIA_PATH_CAPI_LIB_DIR` to the directory with `libpath` and `liblusol` (see {doc}`path_capi`). |
| `check` differs from `base` | The model does not reproduce its benchmark: check the dataset and the closure before reading any shock result. |
