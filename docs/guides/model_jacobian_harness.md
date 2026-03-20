# model jacobian harness

Esta guia resume la capa comun que hoy usamos para benchmarks y gates de Jacobiano en `equilibria`.

La idea es separar dos cosas:

- la logica comun de evaluacion/estructura/reporting del Jacobiano
- la matematica especifica de cada modelo

## que ya es generico

La capa comun vive en:

- `equilibria.solver.jacobians`

Piezas principales:

- `ConstraintJacobianHarness`
- `ConstraintJacobianStats`
- `ConstraintJacobianSolverStats`
- `ConstraintJacobianStructure`
- `solver_stats_payload(...)`
- `summarize_jacobian_mode_entry(...)`
- `compare_jacobian_modes(...)`
- `evaluate_jacobian_mode_gate(...)`

En simple:

- el harness comun sabe evaluar residuales, estructura esparsa y valores del Jacobiano
- puede mezclar derivadas analiticas con fallback numerico
- la capa de reporting/gate sabe comparar `analytic` vs `numeric` sin depender de PEP

## que sigue siendo especifico por modelo

Cada modelo sigue definiendo:

- sus variables
- sus ecuaciones
- su benchmark/punto de referencia
- sus derivadas analiticas por fila

Hoy hay dos implementaciones reales encima de la base:

- PEP: `equilibria.templates.pep_constraint_jacobian`
- SimpleOpen: `equilibria.templates.simple_open_constraint_jacobian`

## contrato minimo de una implementacion hija

Una implementacion hija del harness debe proveer tres cosas:

1. como construir el contexto en un punto `x`
2. como devolver el residual de cada ecuacion en orden estable
3. que derivadas analiticas conoce para una ecuacion dada

Eso corresponde a estos hooks:

- `_build_context(...)`
- `_calculate_constraint_residual_dict(...)`
- `_analytic_constraint_derivatives(...)`

Si una fila no tiene derivada analitica, el harness comun cae a diferencias finitas para esa fila.

## que valida hoy la capa comun

### pep

Script:

- `scripts/parity/measure_pep_jacobian_modes.py`

Valida:

- convergencia de `analytic` y `numeric`
- que `analytic` no use diferencias finitas por encima del maximo permitido
- que la paridad de `analytic` no sea peor que la de `numeric`

### simpleopen

Script:

- `scripts/parity/measure_simple_open_jacobian_modes.py`

Valida:

- benchmark residual exacto en dos closures
- misma estructura esparsa en `analytic` y `numeric`
- que `analytic` use `0` diferencias finitas
- que `numeric` no sea mejor que `analytic` contra la referencia analitica

## cuando usar cada benchmark

- usar `measure_pep_jacobian_modes.py` cuando el cambio toca el solver PEP o su paridad
- usar `measure_simple_open_jacobian_modes.py` cuando el cambio toca la capa comun del harness o del reporting/gate

Si cambias la base comun y solo corre PEP, todavia no validaste que la abstraccion sirva fuera de PEP.

## patron recomendado para un tercer modelo

1. crear `<modelo>_constraint_jacobian.py`
2. heredar de `ConstraintJacobianHarness`
3. exponer un benchmark pequeno y determinista
4. agregar un script `measure_<modelo>_jacobian_modes.py`
5. reusar:
   - `solver_stats_payload`
   - `compare_jacobian_modes`
   - `evaluate_jacobian_mode_gate`

## importaciones utiles

```python
from equilibria.solver import (
    ConstraintJacobianHarness,
    solver_stats_payload,
    compare_jacobian_modes,
    evaluate_jacobian_mode_gate,
)
```

Y para el segundo modelo de ejemplo:

```python
from equilibria.templates import SimpleOpenConstraintJacobianHarness
```
