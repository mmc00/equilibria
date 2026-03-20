# simpleopen parity contract

Esta guia fija el contrato economico canonico de `SimpleOpen` para la futura paridad contra GAMS.

La idea es dejar explicitas tres cosas:

- cual es el contrato que se considera oficial
- cual es la closure canonica de baseline
- cual es el benchmark numerico que debe compartir Python y GAMS

## contrato canonico

Hoy el contrato oficial de paridad es:

- `contract.name = "simple_open_v1"`

Ese contrato resuelve un sistema minimo de tres ecuaciones:

- `EQ_VA`
- `EQ_INT`
- `EQ_CET`

Y usa este orden canonico de variables:

- `VA`
- `INT`
- `X`
- `D`
- `E`
- `ER`
- `PFX`
- `CAB`
- `FSAV`

Ese orden ya esta formalizado en:

- [simple_open_parity_spec.py](/Users/marmol/proyectos/equilibria/src/equilibria/templates/simple_open_parity_spec.py)

## closure canonica de baseline

La closure base para paridad es:

- `closure.name = "simple_open_default"`

Interpretacion:

- numeraire: `PFX`
- `PFX` y `FSAV` fijos
- `ER` y `CAB` endogenos
- movilidad de capital: `mobile`

Tambien existe una closure secundaria de benchmark:

- `flexible_external_balance`

Pero la referencia base de paridad debe arrancar por `simple_open_default`.

## benchmark canonico

Como `SimpleOpen` no viene de una SAM calibrada, el benchmark esta definido por un bloque pequeno de parametros y niveles exogenos.

### simple_open_default

- `alpha_va = 0.40`
- `rho_va = 0.75`
- `a_int = 0.50`
- `b_ext = 0.10`
- `theta_cet = 0.60`
- `phi_cet = 1.20`
- `ER = 1.00`
- `PFX = 1.00`
- `D = 1.00`
- `E = 1.00`
- `CAB = 1.00`
- `FSAV = 1.00`

### flexible_external_balance

- `alpha_va = 0.45`
- `rho_va = 0.70`
- `a_int = 0.55`
- `b_ext = 0.08`
- `theta_cet = 0.58`
- `phi_cet = 1.25`
- `ER = 1.08`
- `PFX = 1.00`
- `D = 1.04`
- `E = 0.93`
- `CAB = 0.82`
- `FSAV = 0.82`

Los niveles benchmark derivados de esos parametros ya se construyen en:

- [simple_open_parity_spec.py](/Users/marmol/proyectos/equilibria/src/equilibria/templates/simple_open_parity_spec.py)

Y el harness del Jacobiano ya consume exactamente esa misma especificacion en:

- [simple_open_constraint_jacobian.py](/Users/marmol/proyectos/equilibria/src/equilibria/templates/simple_open_constraint_jacobian.py)

## por que esta guia importa antes de GAMS

Sin esta especificacion explicita, el siguiente paso de GAMS seria ambiguo:

- un `.gms` podria usar otra closure
- otro orden de variables
- o un benchmark numerico distinto al de Python

Con esta guia, el contrato de paridad ya queda fijo antes de escribir el modelo GAMS.

## api y helpers

Helpers utiles:

```python
from equilibria.templates import (
    build_simple_open_contract,
    build_simple_open_runtime_config,
)
from equilibria.templates.simple_open_parity_spec import build_simple_open_parity_spec

contract = build_simple_open_contract("simple_open_v1")
runtime = build_simple_open_runtime_config("default_template")
spec = build_simple_open_parity_spec(contract)
```

El `spec` resultante deja listos:

- `contract_name`
- `closure_name`
- `equation_names`
- `variable_names`
- `benchmark_parameters`
- `benchmark_levels`

## criterio para el siguiente corte

El siguiente corte de la epica debe usar este contrato exactamente para:

1. escribir el `.gms` minimo
2. exportar niveles benchmark
3. comparar Python vs GAMS sin reinterpretar el sistema
