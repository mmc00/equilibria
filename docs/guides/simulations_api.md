# simulations api (modelo-agnostica)

La API `equilibria.simulations` permite correr escenarios en estilo scikit:

1. crear simulador por modelo,
2. calibrar una sola vez con `fit()`,
3. listar choques disponibles,
4. ejecutar escenarios con `run_scenarios(...)`.

## uso base

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

## uso ultra simple (pep)

Si no quieres armar `Scenario(...)` manualmente:

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

Tambien puedes correr presets por nombre:

```python
sim.available_presets()
# ('export_tax', 'import_price', 'import_shock', 'government_spending')

report = sim.run_preset("export_tax", multiplier=0.75)
```

## escenario export tax (ttix)

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

## escenario import price (PWM)

Choque sectorial:

```python
Scenario(
    name="import_price_agr",
    shocks=[Shock(var="PWM", op="scale", values={"agr": 1.25})],
)
```

Choque para todos los bienes:

```python
Scenario(
    name="import_shock_all",
    shocks=[Shock(var="PWM", op="scale", values={"*": 1.25})],
)
```

## escenario government spending (G)

```python
Scenario(
    name="government_spending",
    shocks=[Shock(var="G", op="scale", values=1.2)],
)
```

Tambien puedes usar presets:

```python
from equilibria.simulations import (
    export_tax,
    government_spending,
    import_price,
    import_shock,
)

scenarios = [
    export_tax(multiplier=0.75),
    import_shock(multiplier=1.25),
    import_price(commodity="agr", multiplier=1.25),
    government_spending(multiplier=1.2),
]
```

## notas de diseño

- `Shock` y `Scenario` son modelos pydantic inmutables.
- `Simulator.fit()` cachea el estado base calibrado.
- `warm_start=True` reutiliza niveles de la corrida anterior.
- `reference_results_gdx` es opcional; si se incluye, se agrega comparación por escenario.
- La API es genérica y se extiende por adapters (`model="pep"` hoy, otros modelos después).
- El reporte incluye `capabilities` del adapter:
  - `has_solver`
  - `has_reference_compare`
  - `mode`

## modelos registrados actualmente

```python
from equilibria.simulations import available_models

print(available_models())
# ('gtap', 'icio', 'ieem', 'pep')
```

Wrappers convenientes:

```python
from equilibria.simulations import GTAPSimulator, ICIOSimulator, IEEMSimulator

ieem = IEEMSimulator(base_state={"x": 1.0}).fit()
gtap = GTAPSimulator(base_state={"x": 1.0}).fit()
icio = ICIOSimulator(base_state={"x": 1.0}).fit()
```

`pep` tiene integración completa calibración + solver.

`ieem`, `gtap` e `icio` están habilitados como adapters state-based (sin solver nativo aún):
- permiten `fit`, catálogo de choques y ejecución de escenarios,
- `solve_state` corre en modo `no_solver` por defecto,
- comparación de referencia externa no está implementada por defecto.

## estado actual por modelo

- `pep`: runtime nativo completo (calibración + solve + comparación).
- `ieem`: sin runtime nativo pre-registrado.
- `gtap`: sin runtime nativo pre-registrado.
- `icio`: sin runtime nativo pre-registrado.

Nota: para `ieem/gtap/icio` no se registra runtime automático por diseño, hasta que exista implementación nativa real del modelo.

Tambien puedes inyectar hooks nativos sin cambiar la API:

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

Cuando se pasan hooks, `capabilities.mode` cambia automaticamente:
- `state_only_no_solver` (sin hooks),
- `state_with_solver_hook` (solo `solve_fn`),
- `state_with_solver_and_compare_hooks` (`solve_fn` + `compare_fn`).

Si no quieres pasar hooks en cada instancia, puedes registrarlos una vez por modelo:

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

Mismo patrón para `gtap` e `icio`:

```python
from equilibria.simulations import GTAPSimulator, ICIOSimulator, register_mapping_runtime

register_mapping_runtime("gtap", solve_fn=my_gtap_solve_fn)
register_mapping_runtime("icio", solve_fn=my_icio_solve_fn)

gtap = GTAPSimulator(base_state={"x": 1.0}).fit()
icio = ICIOSimulator(base_state={"x": 1.0}).fit()
```

Si pasas hooks explícitos al construir `Simulator(...)`, esos hooks tienen prioridad sobre el runtime registrado.

Utilidades disponibles:
- `register_mapping_runtime(model, ...)`
- `get_mapping_runtime(model)`
- `available_mapping_runtimes()`
- `clear_mapping_runtime(model=None)`

## compatibilidad legacy

`equilibria.templates.pep_scenario_parity` se mantiene por compatibilidad.

La ruta recomendada para desarrollo nuevo es `equilibria.simulations` (`Simulator` / `PepSimulator`).

### migracion recomendada

Antes (legacy):

```python
from equilibria.templates.pep_scenario_parity import PEPScenarioParityRunner

runner = PEPScenarioParityRunner(...)
report = runner.run()
```

Ahora (recomendado):

```python
from equilibria.simulations import PepSimulator

sim = PepSimulator(...).fit()
report = sim.run_export_tax(multiplier=0.75)
```
