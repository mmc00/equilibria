# Cómo generar la referencia GEMPACK de altertax (RunGTAP, Windows)

**Por qué hace falta.** Las 10 filas `reference="gempack"` de
`scripts/gtap/coverage_matrix.py` son **todas `mode="pure"`**. No existe ninguna
referencia GEMPACK de `altertax` — verificado sobre los 222 fixtures de
`tests/fixtures/gtap7_gempack/`: todos son shocks de aranceles (`tm10`, `tm1`,
`tm0p3`…), ninguno lleva elasticidades altertax.

Consecuencia concreta: el **residual de importaciones agrícolas** (10 pares
`(región importadora, sector)`, ~1-6% sobre GAMS, ver
`docs/architecture/monolito_vs_bloques.md`) vive en `altertax` y hoy **no tiene
árbitro**. Sólo se puede medir contra GAMS — y en el residual `qxs`
(`memory/project_gtap_qxs_bilateral_trade_residual_2026_08_21`) GAMS resultó ser
**el raro**: GEMPACK le daba la razón a equilibria. Atacar el residual sin esta
referencia arriesga "arreglar" algo que ya está bien.

---

## Lo que cambia altertax: SOLO elasticidades

No es otro shock ni otro closure. `apply_altertax_elasticities`
(`src/equilibria/templates/gtap/altertax/parameter_overrides.py`) reemplaza
elasticidades y **no toca** shares calibrados, benchmarks SAM ni tasas de
impuestos. En GEMPACK eso se hace escribiendo headers del `.prm`, exactamente
como `_force_rdlt` ya reescribe `RDLT`.

| override (Python) | valor | header `.prm` |
|---|---|---|
| `esubva` | **1.0** | `ESBV` |
| `esubd`  | **0.95** | `ESBD` |
| `esubm`  | **0.95** | `ESBM` |
| `etrae`  | **1.0** | `ETRE` |
| `esubg`  | **1.0** | `ESBG` |
| `esubi`  | **1.0** | `ESBI` |

`sigmav`/`sigmap`/`sigmand` (=1.0) son del nido CES de producción aguas abajo de
`esubva`; en GTAPv7.tab no tienen header propio separado — quedan cubiertos por
`ESBV=1` (Cobb-Douglas). `omegaf=1.0` es movilidad de factores: si el dataset
expone un header propio, ponerlo en 1.0; si no, se hereda.

**Lo que NO se toca** (queda como está en el `.prm`): `ESBQ`, `ESBS`, `ESBT`,
`ESBC`, `ETRQ`, `SUBP`, `INCP`, y todos los `*R` (`SBVR`, `SBDR`, `SBMR`, `SBCR`).

---

## Procedimiento

En la máquina Windows con RunGTAP instalado, desde la raíz del repo.

### 1. Generar el `.cmf` base

```powershell
uv run python scripts/gtap/run_gempack_matrix.py --no-solve
```

Eso escribe `runs/gempack_matrix/<ds>/tm10.cmf` y copia los inputs del dataset.
**No lo resuelvas todavía.**

### 2. Reescribir las elasticidades en el `.prm` de la corrida

El `.prm` que hay que editar es el de `runs/gempack_matrix/<ds>/`, **no** el de
`datasets/` (ese es la fuente y no se toca). Mismo patrón que `_force_rdlt`:

```python
import numpy as np
from equilibria.babel.har import write_har
from equilibria.babel.har.reader import read_har

ALTERTAX = {"ESBV": 1.0, "ESBD": 0.95, "ESBM": 0.95,
            "ETRE": 1.0, "ESBG": 1.0, "ESBI": 1.0}

prm = "runs/gempack_matrix/gtap7_10x7/default_capfix.prm"   # el que nombre el .cmf
hars = read_har(prm)
for name, val in ALTERTAX.items():
    key = next((k for k in hars if str(k).upper() == name), None)
    if key is None:
        print(f"AVISO: header {name} ausente — se hereda"); continue
    a = np.asarray(hars[key].array, dtype=float)
    a[...] = val
    hars[key].array = a
write_har(prm, hars)
```

**Verificá que reescribió** releyendo el `.prm` antes de resolver: un header
ausente se salta con aviso, y una corrida con elasticidades a medio aplicar es
peor que ninguna referencia.

### 3. Resolver

```powershell
C:\runGTAP375\gtapv7\gtapv7.exe -cmf tm10.cmf
```

El shock sigue siendo `Shock tm = uniform 10` y el closure el estándar
(`dpsave` exógeno, sin swaps). **Lo único que cambia es el `.prm`.** Así el
experimento queda comparable con las filas `pure` existentes: mismo shock, misma
clausura, distintas elasticidades — que es justo la definición de altertax.

### 4. Nombrar y ubicar el fixture

El gate resuelve el fixture con este orden
(`tests/templates/gtap/test_gtap7_gempack_parity.py::_capfix_fixture_for`):

1. `<row.ref sin .har>_capfix.har`
2. `sl4dump_<ds>_tm10_capfix.har`
3. glob `sl4dump_<ds>_*_capfix.har`

Copiá el `sl4dump` resultante a:

```
tests/fixtures/gtap7_gempack/sl4dump_<ds>_tm10_altertax_capfix.har
```

⚠️ **Ese nombre NO lo encuentra el resolver actual** — el glob del paso 3
matchearía y confundiría la fila `pure` con la de `altertax`. Hay que agregar el
eje al resolver (o a `Row`) **antes** de commitear el fixture. Avisame cuando lo
tengas y lo hago; es cambio de test, no de modelo.

### 5. Empezá por `gtap7_10x7`

Es donde están medidas las 107 celdas del residual. Con esa sola referencia ya
se puede decidir si el residual agrícola es infidelidad nuestra o rareza de GAMS.

---

## Lo que NO se puede hacer, y por qué

**`gtap7_20x41` contra GEMPACK: imposible.** No es un hueco por falta de tiempo.
`scripts/gtap/run_gempack_matrix.py:22` lo documenta: *"gtap7_20x41 does not
solve in GEMPACK (loge-of-negative in E_u for Caribbean)"* — GEMPACK mismo no
produce la referencia, consistente con su estado `blocked` en la matriz.

El lado **Python** del 20x41, en cambio, **sí resuelve** — y localmente. El
comentario de `test_gtap7_gempack_parity.py:186` ("OOM >32GB") habla del camino
**IPOPT**, que es el único que ese gate sabe usar; no del modelo. Con el solver
libre (Newton-TR + MUMPS, sin Hessiano) el 20x41 corre en **~67 min** en un Mac
M3 Pro de 18 GB y está EN PRODUCCIÓN — perfil por capas y catálogo de palancas en
`dev-tools/equilibria-tools/memory/project_gtap_20x41_catalogo_palancas_2026_09_02.md`
(el solve son ~64 de esos 67 min: 18 solves de continuación, 655 factorizaciones
MUMPS). La validación da 94.7% within 1pp / 98.5% en niveles.

O sea: el hueco del 20x41 en F5 es **sólo del lado GEMPACK**, y es infranqueable
por el `loge`-of-negative de arriba. Si algún día se quisiera cablear el 20x41 a
este gate habría que enseñarle `EQUILIBRIA_GTAP_SOLVER=scipy_newton_tr` +
`TR_LINSOLVE=mumps` (hoy sólo togglea `EQUILIBRIA_GTAP_SOLVE_NLP`), pero sin
referencia GEMPACK no habría contra qué medir.
