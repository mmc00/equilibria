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

Tambien puedes construir shocks directos sin escribir `Scenario(...)` a mano:

```python
sim.available_shocks()

scenario = sim.shock(var="ttix", index="*", multiplier=0.75, name="export_tax")
report = sim.run_shock(var="PWM", index="agr", multiplier=1.25)
```

`available_presets()`, `make_preset()` y `run_preset()` siguen existiendo por compatibilidad, pero estan deprecados.

Si quieres ver como estructurar un flujo de proyecto mas humano
(`preprocesar SAM` -> `fit()` -> `shocks` -> `report`), revisa tambien:

- `docs/guides/pep_user_flow_examples.md`

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

## closure por escenario (pep)

Si un escenario PEP necesita una closure distinta a la del simulador base,
puedes pasarla como un bloque directo `closure={...}` en `Scenario(...)`.

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

Notas:

- `closure` es opcional.
- Si no lo pasas, el escenario usa el contrato base del simulador.
- `closure` se mergea sobre el contrato base, no reemplaza todo el contrato.
- El reporte devuelve la `closure` pedida y la `effective_contract` realmente usada por el solver.

Importante:

- `closure` si forma parte de la API publica por escenario.
- `equations` y `bounds` no forman parte de la API publica por escenario.
- Si necesitas cambiar `equations`, `bounds` o la `config` del solver, eso se hace al nivel de `PepSimulator(...)`, no dentro de `Scenario(...)`.
- La guia detallada de esta separacion esta en `docs/guides/pep_contract_api.md`.

### simbolos de closure soportados hoy en pep

Los nombres que hoy entiende el solver son estos:

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
- el `numeraire` que declares, por ejemplo `e`

Interpretacion rapida:

- `SH`, `SF`, `EXD`, `PWM`, `VSTK` y `CMIN` son familias indexadas; en la closure se escriben por nombre agregado.
- `TR_SELF` corresponde a `TR(gvt,gvt)` y `TR(row,row)`.
- `TR_AGD_ROW` corresponde a transferencias `row -> AGD`.
- `TR_ROW_AGNG` corresponde a transferencias `AGNG -> row`.
- `LS` controla la oferta laboral cuando la quieras fijar o liberar explicitamente.
- `KS` controla la oferta de capital. Si `capital_mobility='mobile'`, fija/libera `KS[k]`; si `capital_mobility='sector_specific'`, aplica ese cierre sobre `KD[k,j]`.

Si pasas un simbolo no soportado, el solver no lo acepta en silencio: la corrida queda marcada como invalida en `validation.closure_validation`, con mensajes como `Unsupported fixed closure symbols` o `Unsupported endogenous closure symbols`.

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
