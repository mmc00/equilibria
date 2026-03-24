# pep user flow examples

Esta guia muestra ejemplos de estructura para proyectos que usan la API publica
de `equilibria.simulations` con PEP.

La idea central es simple:

1. tu proyecto puede transformar o limpiar una SAM como quiera,
2. cuando ya tengas una SAM compatible con PEP, entras a `PepSimulator`,
3. los escenarios se construyen con `sim.shock(...)` o con `Scenario(...)`.

Importante:

- `equilibria` no obliga una arquitectura para tu preprocesamiento.
- la API publica de simulacion empieza realmente cuando ya tienes `sam_file` + `val_par_file`.
- si tu proyecto necesita reglas de negocio propias para Excel, mapping o QA estructural, eso vive mejor fuera de `equilibria`.

## ejemplo 1: flujo minimo con la api publica

```python
from equilibria.simulations import PepSimulator


sim = PepSimulator(
    sam_file="data/SAM-CRI-pep.xlsx",
    val_par_file="data/VAL_PAR-CRI.xlsx",
    method="ipopt",
    init_mode="excel",
).fit()

report = sim.run_scenarios(
    scenarios=[
        sim.shock(var="ttix", index="*", multiplier=0.75, name="export_tax"),
        sim.shock(var="PWM", index="agr", multiplier=1.25, name="import_price_agr"),
        sim.shock(var="PWM", index="*", multiplier=1.25, name="import_shock"),
        sim.shock(var="G", multiplier=1.2, name="government_spending"),
    ],
    include_base=True,
    warm_start=True,
)
```

Ese es el flujo recomendado si tu SAM ya esta en formato PEP.

## ejemplo 2: separar preprocesamiento y corrida

En proyectos reales, normalmente conviene no mezclar:

- transformacion de la SAM,
- construccion del `PepSimulator`,
- definicion de escenarios.

Una estructura mas legible suele verse asi:

```python
from pathlib import Path

from equilibria.simulations import PepSimulator


def prepare_pep_sam(raw_excel: Path, mapping: Path) -> Path:
    # Logica propia del proyecto:
    # leer Excel, aplicar mapping, mover cuentas, QA, guardar salida
    ...


def load_fitted_pep_simulator(sam_file: Path, val_par_file: Path) -> PepSimulator:
    return PepSimulator(
        sam_file=sam_file,
        val_par_file=val_par_file,
        method="ipopt",
        init_mode="excel",
    ).fit()


def build_core_scenarios(sim: PepSimulator) -> list:
    return [
        sim.shock(var="ttix", index="*", multiplier=0.75, name="export_tax"),
        sim.shock(var="PWM", index="agr", multiplier=1.25, name="import_price_agr"),
        sim.shock(var="PWM", index="*", multiplier=1.25, name="import_shock"),
        sim.shock(var="G", multiplier=1.2, name="government_spending"),
    ]


def run_project_flow(raw_excel: Path, mapping: Path, val_par_file: Path) -> dict:
    sam_file = prepare_pep_sam(raw_excel, mapping)
    sim = load_fitted_pep_simulator(sam_file, val_par_file)
    scenarios = build_core_scenarios(sim)
    return sim.run_scenarios(scenarios=scenarios, include_base=True, warm_start=True)
```

Ese patron mantiene clara la frontera:

- tu codigo prepara datos,
- `equilibria` calibra y resuelve,
- los shocks viven en la API publica.

## ejemplo 3: estructura de archivos legible para humanos

Cuando el flujo empieza a crecer, esta separacion suele funcionar bien:

- `prepare_pep_sam.py`: solo transformacion y QA de la SAM.
- `run_pep_scenarios.py`: solo construccion del simulador y escenarios.
- `project_rules.py`: mappings, nombres de cuentas y reglas de negocio.

Evita un archivo unico que mezcle:

- mappings raw -> PEP,
- mutaciones estructurales sobre la SAM,
- logging,
- guardado de reportes,
- construccion de shocks,
- solve del modelo.

## ejemplo 4: cuando usar `sim.shock(...)` y cuando usar `Scenario(...)`

Usa `sim.shock(...)` cuando:

- el escenario tiene un solo shock,
- quieres un flujo corto y legible,
- no necesitas configurar `closure` por escenario.

Usa `Scenario(...)` cuando:

- el escenario junta varios shocks,
- necesitas `closure={...}` especifica,
- quieres describir un paquete de politica mas complejo.

```python
from equilibria.simulations import Scenario


scenario = Scenario(
    name="trade_and_fiscal",
    shocks=[
        sim.shock(var="PWM", index="*", multiplier=1.10).shocks[0],
        sim.shock(var="G", multiplier=1.05).shocks[0],
    ],
)
```

## recomendacion final

La API publica de `equilibria` queda mas estable y mas facil de explicar si:

- el proyecto prepara una SAM PEP-compatible afuera,
- `PepSimulator(...).fit()` marca el inicio del runtime publico,
- `sim.shock(...)` y `run_scenarios(...)` muestran los casos de uso mas comunes.

Ver tambien:

- `docs/guides/simulations_api.md`
- `docs/guides/pep_contract_api.md`
- `docs/guides/pep_core_scenarios_gate.md`
