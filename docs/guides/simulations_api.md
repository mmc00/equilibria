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

## modelos registrados actualmente

```python
from equilibria.simulations import available_models

print(available_models())
# ('gtap', 'icio', 'ieem', 'pep')
```

`pep` tiene integración completa calibración + solver.

`ieem`, `gtap` e `icio` están habilitados como adapters state-based (sin solver nativo aún):
- permiten `fit`, catálogo de choques y ejecución de escenarios,
- `solve_state` corre en modo `no_solver`,
- comparación de referencia externa aún no implementada para esos tres.
