# Simulations API (model-agnostic)

> The PEP examples in this guide run on the old PEP engine, which is deleted in
> stage E1 of the PEP roadmap. The current engine is `pep_pyomo`: see
> `docs/site/guide/pep_quickstart.md`.

The `equilibria.simulations` API runs scenarios in scikit style:

1. create a simulator per model,
2. calibrate once with `fit()`,
3. list the available shocks,
4. run scenarios with `run_scenarios(...)`.

## Basic use

```python
from equilibria.simulations import Scenario, Shock, Simulator

sim = Simulator(
    model="pep",
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
    init_mode="excel",
).fit()

print(sim.available_shocks())
```

## Simplest use (pep)

If you do not want to build `Scenario(...)` by hand:

```python
from equilibria.simulations import PepSimulator

sim = PepSimulator(
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
).fit()

report_export = sim.run_export_tax(multiplier=0.75)
report_import_all = sim.run_import_shock(multiplier=1.25)
report_import_agr = sim.run_import_price(commodity="agr", multiplier=1.25)
report_gov = sim.run_government_spending(multiplier=1.2)
```

You can also build shocks directly without writing `Scenario(...)` by hand:

```python
sim.available_shocks()

scenario = sim.shock(var="ttix", index="*", multiplier=0.75, name="export_tax")
report = sim.run_shock(var="PWM", index="agr", multiplier=1.25)
```

`available_presets()`, `make_preset()` and `run_preset()` still exist for compatibility, but they are deprecated.

If you want to see how to structure a more human project flow
(`preprocess SAM` -> `fit()` -> `shocks` -> `report`), see also:

- `docs/notes/pep/pep_user_flow_examples.md`

## Export tax scenario (ttix)

```python
from equilibria.simulations import Scenario, Shock

report = sim.run_scenarios(
    scenarios=[
        Scenario(
            name="export_tax",
            shocks=[Shock(var="ttix", op="scale", values={"*": 0.75})],
        ),
    ],
    reference_results_gdx="src/equilibria/templates/reference/pep2/scripts/Results.gdx",
)
```

## Import price scenario (PWM)

Sector shock:

```python
Scenario(
    name="import_price_agr",
    shocks=[Shock(var="PWM", op="scale", values={"agr": 1.25})],
)
```

Shock for every good:

```python
Scenario(
    name="import_shock_all",
    shocks=[Shock(var="PWM", op="scale", values={"*": 1.25})],
)
```

## Government spending scenario (G)

```python
Scenario(
    name="government_spending",
    shocks=[Shock(var="G", op="scale", values=1.2)],
)
```

## Closure per scenario (pep)

If a PEP scenario needs a closure different from the base simulator's,
you can pass it as a direct `closure={...}` block in `Scenario(...)`.

```python
from equilibria.simulations import PepSimulator, Scenario, Shock

sim = PepSimulator(
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
    init_mode="excel",
    contract="pep_nlp_v1",
    config="default_ipopt",
).fit()

report = sim.run_scenarios(
    scenarios=[
        Scenario(
            name="government_spending",
            shocks=[Shock(var="G", op="scale", values=1.2)],
            closure={
                "fixed": ["G", "CAB", "KS", "LS", "PWM", "PWX", "CMIN", "VSTK", "TR_SELF"],
                "endogenous": ["IT", "SH", "SF", "SG", "SROW"],
                "numeraire": "e",
                "capital_mobility": "mobile",
            },
        ),
    ],
)
```

Notes:

- `closure` is optional.
- If you do not pass it, the scenario uses the simulator's base contract.
- `closure` is merged over the base contract; it does not replace the whole contract.
- The report returns the requested `closure` and the `effective_contract` the solver actually used.

Important:

- `closure` is part of the public per-scenario API.
- `equations` and `bounds` are not part of the public per-scenario API.
- If you need to change `equations`, `bounds` or the solver `config`, that is done at the level of `PepSimulator(...)`, not inside `Scenario(...)`.
- The detailed guide to this separation is in `docs/notes/pep/pep_contract_api.md`.

### Closure symbols supported today in pep

The names the solver understands today are these:

- `G`
- `CAB`
- `SG`
- `SROW`
- `IT`
- `SH`
- `SF`
- `EXD`
- `LS`
- `PWM`
- `PWX`
- `CMIN`
- `VSTK`
- `TR_SELF`
- `TR_AGD_ROW`
- `TR_ROW_AGNG`
- `KS`
- the `numeraire` you declare, for example `e`

Quick reading:

- `SH`, `SF`, `EXD`, `PWM`, `VSTK` and `CMIN` are indexed families; in the closure they are written by their aggregate name.
- `TR_SELF` stands for `TR(gvt,gvt)` and `TR(row,row)`.
- `TR_AGD_ROW` stands for transfers `row -> AGD`.
- `TR_ROW_AGNG` stands for transfers `AGNG -> row`.
- `LS` controls labor supply when you want to fix or free it explicitly.
- `KS` controls capital supply. If `capital_mobility='mobile'`, it fixes/frees `KS[k]`; if `capital_mobility='sector_specific'`, it applies that closure to `KD[k,j]`.

If you pass an unsupported symbol, the solver does not accept it silently: the run is marked invalid in `validation.closure_validation`, with messages such as `Unsupported fixed closure symbols` or `Unsupported endogenous closure symbols`.

## Design notes

- `Shock` and `Scenario` are immutable pydantic models.
- `Simulator.fit()` caches the calibrated base state.
- `warm_start=True` reuses the levels of the previous run.
- `reference_results_gdx` is optional; when given, a per-scenario comparison is added.
- The API is generic and is extended through adapters (`model="pep"` today, other models later).
- The report includes the adapter's `capabilities`:
  - `has_solver`
  - `has_reference_compare`
  - `mode`

## Models registered today

```python
from equilibria.simulations import available_models

print(available_models())
# ('gtap', 'icio', 'ieem', 'pep')
```

Convenience wrappers:

```python
from equilibria.simulations import GTAPSimulator, ICIOSimulator, IEEMSimulator

ieem = IEEMSimulator(base_state={"x": 1.0}).fit()
gtap = GTAPSimulator(base_state={"x": 1.0}).fit()
icio = ICIOSimulator(base_state={"x": 1.0}).fit()
```

`pep` has full calibration + solver integration.

`ieem`, `gtap` and `icio` are enabled as state-based adapters (no native solver yet):
- they allow `fit`, a shock catalog and running scenarios,
- `solve_state` runs in `no_solver` mode by default,
- comparison against an external reference is not implemented by default.

## Current status per model

- `pep`: full native runtime (calibration + solve + comparison).
- `ieem`: no pre-registered native runtime.
- `gtap`: no pre-registered native runtime.
- `icio`: no pre-registered native runtime.

Note: for `ieem/gtap/icio` no runtime is registered automatically, by design, until a real native implementation of the model exists.

You can also inject native hooks without changing the API:

```python
from equilibria.simulations import Simulator

sim = Simulator(
    model="ieem",
    base_state={"x": 1.0},
    solve_fn=my_ieem_solve_fn,              # opcional
    compare_fn=my_ieem_compare_fn,          # opcional
    key_indicators_fn=my_ieem_indicators_fn # opcional
).fit()
```

When hooks are passed, `capabilities.mode` changes automatically:
- `state_only_no_solver` (no hooks),
- `state_with_solver_hook` (only `solve_fn`),
- `state_with_solver_and_compare_hooks` (`solve_fn` + `compare_fn`).

If you do not want to pass hooks on every instance, you can register them once per model:

```python
from equilibria.simulations import (
    IEEMSimulator,
    register_mapping_runtime,
)

register_mapping_runtime(
    "ieem",
    solve_fn=my_ieem_solve_fn,
    compare_fn=my_ieem_compare_fn,
    key_indicators_fn=my_ieem_indicators_fn,
)

sim = IEEMSimulator(base_state={"x": 1.0}).fit()
report = sim.run_scenarios(...)
```

Same pattern for `gtap` and `icio`:

```python
from equilibria.simulations import GTAPSimulator, ICIOSimulator, register_mapping_runtime

register_mapping_runtime("gtap", solve_fn=my_gtap_solve_fn)
register_mapping_runtime("icio", solve_fn=my_icio_solve_fn)

gtap = GTAPSimulator(base_state={"x": 1.0}).fit()
icio = ICIOSimulator(base_state={"x": 1.0}).fit()
```

If you pass explicit hooks when building `Simulator(...)`, those hooks take priority over the registered runtime.

Available utilities:
- `register_mapping_runtime(model, ...)`
- `get_mapping_runtime(model)`
- `available_mapping_runtimes()`
- `clear_mapping_runtime(model=None)`

## Legacy compatibility

`equilibria.templates.pep_scenario_parity` is kept for compatibility.

The recommended path for new development is `equilibria.simulations` (`Simulator` / `PepSimulator`).

### Recommended migration

Before (legacy):

```python
from equilibria.templates.pep_scenario_parity import PEPScenarioParityRunner

runner = PEPScenarioParityRunner(...)
report = runner.run()
```

Now (recommended):

```python
from equilibria.simulations import PepSimulator

sim = PepSimulator(...).fit()
report = sim.run_export_tax(multiplier=0.75)
```
