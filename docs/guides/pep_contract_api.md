# pep contract api

Esta guia aclara como se separan hoy las capas de configuracion de PEP en `equilibria`.

La idea es simple:

- `closure` cambia el cierre economico de un escenario.
- `equations` cambia el sistema de ecuaciones activo.
- `bounds` cambia el dominio numerico del problema.
- `config` cambia como se resuelve o compara el problema.

## decision actual de api

Hoy la decision recomendada es esta:

- `closure`: si se expone en la API publica por escenario.
- `equations`: no se expone por escenario.
- `bounds`: no se expone por escenario.
- `config`: no se expone por escenario.

En otras palabras:

- `Scenario(...)` puede cambiar `closure`.
- `PepSimulator(...)` puede cambiar `contract` y `config`.

Eso deja una API publica mas simple y evita que cada escenario redefina el problema matematico completo.

## por que closure si, pero equations y bounds no

`closure` es parte natural de un escenario CGE.

Ejemplos tipicos:

- fijar `G`
- liberar `SG`
- fijar `KS` y `LS`
- elegir `capital_mobility`

Eso cambia el cierre economico, pero sigue siendo el mismo problema PEP.

En cambio, `equations` y `bounds` cambian algo mas basal:

- `equations` decide que sistema se esta resolviendo
- `bounds` decide que dominios numericos tiene ese sistema

Si eso cambia por escenario, se vuelve dificil comparar:

- `base` vs shock
- un shock vs otro
- Python vs GAMS

Por eso, por ahora, `equations` y `bounds` viven en el `contract` del simulador, no en cada escenario.

## capas actuales

### 1. scenario

Nivel mas publico y mas simple.

Sirve para decir:

- nombre del escenario
- shocks
- `closure` opcional

Ejemplo:

```python
from equilibria.simulations import Scenario, Shock

scenario = Scenario(
    name="government_spending",
    shocks=[Shock(var="G", op="scale", values=1.2)],
    closure={
        "fixed": ["G", "CAB", "KS", "LS", "PWM", "PWX", "CMIN", "VSTK", "TR_SELF"],
        "endogenous": ["IT", "SH", "SF", "SG", "SROW"],
        "numeraire": "e",
        "capital_mobility": "mobile",
    },
)
```

### 2. contract

Nivel del simulador.

Define el problema economico que se va a resolver:

- `closure`
- `equations`
- `bounds`

Ejemplo:

```python
from equilibria.simulations import PepSimulator

sim = PepSimulator(
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
    init_mode="excel",
    contract={
        "name": "pep_nlp_v1",
        "equations": {
            "activation_masks": "gams_parity",
        },
        "bounds": {
            "fixed_from_closure": True,
        },
    },
).fit()
```

### 3. config

Tambien vive al nivel del simulador.

No cambia el modelo economico; cambia la ejecucion:

- `solver`
- `problem_type`
- `tolerance`
- `max_iterations`
- `reference`

Ejemplo:

```python
sim = PepSimulator(
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    contract="pep_nlp_v1",
    config={
        "solver": "ipopt",
        "problem_type": "nlp",
        "tolerance": 1e-8,
        "max_iterations": 300,
        "reference": {
            "enabled": False,
            "source": "none",
        },
    },
).fit()
```

## que significa cada bloque

### closure

Responde:

- que queda fijo
- que queda endogeno
- cual es el numeraire
- como se trata la movilidad de capital

### equations

Responde:

- que ecuaciones entran al sistema
- como se activan sus mascaras

Hoy soporta:

- `include`
- `activation_masks = "gams_parity" | "all_active"`

### bounds

Responde:

- que variables quedan fijas por closure
- que simbolos se fuerzan a quedar libres
- que politica general de dominios se usa

Hoy soporta:

- `fixed_from_closure`
- `free`

## ejemplos utiles

### usar el contrato canonico recomendado

```python
sim = PepSimulator(
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
    init_mode="excel",
    contract="pep_nlp_v1",
    config="default_ipopt",
).fit()
```

### cambiar la mascara de activacion para una corrida experimental

Esto no se recomienda como API por escenario. Se hace al nivel del simulador:

```python
sim = PepSimulator(
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
    init_mode="excel",
    contract={
        "equations": {
            "activation_masks": "all_active",
        }
    },
    config="default_ipopt",
).fit()
```

### liberar un simbolo por bounds

Tambien se hace al nivel del simulador:

```python
sim = PepSimulator(
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
    init_mode="excel",
    contract={
        "bounds": {
            "fixed_from_closure": False,
            "free": ["PWM"],
        }
    },
    config="default_ipopt",
).fit()
```

## resumen practico

Si la pregunta es "donde pongo esto", la regla es:

- si es un cambio de cierre economico del escenario: `Scenario.closure`
- si cambia el sistema o los dominios del problema: `PepSimulator(contract=...)`
- si cambia solver, tolerancia o referencia: `PepSimulator(config=...)`

Esa es la separacion que hoy conviene mantener en la API publica.
