# El modo `asl` del Jacobiano no resuelve la via del DRIVER

**Fecha:** 2026-09-24 (medicion original 2026-09-22 en `35b0316`; re-verificado
en `main` tras el merge de #80)
**Estado:** ABIERTO — hay un workaround en dos tests, no un arreglo
**Alcance:** `EQUILIBRIA_GTAP_JAC_MODE` = `asl` (el DEFAULT) contra
`reverse_numeric`, en la via multiperiodo del driver

## Resumen

`asl` es el default desde `9ef28cb` ("perf(gtap): evaluate the Jacobian through
ASL", que baja el 20x41 de 28,8 a 11,7 min). En la via de BLOQUES funciona y es
mas rapido. En la via del DRIVER **aborta**: `success=False`, los valores nunca
se escriben al modelo, y lo que se observa despues es el residuo del modelo SIN
RESOLVER — no un solve impreciso.

Ningun gate lo ve. Los gates van con `skip_base_solve=True` y semilla del GDX,
donde la condicion no se da.

## La causa (ya diagnosticada)

Esta escrita en `tests/templates/gtap/test_multiperiod_driver.py:59-80`, y es
mas profunda que "ASL falla":

`asl` valida que el sistema sea una biyeccion exacta var<->fila antes de armar
el Jacobiano (`pyomo_adapter.py:333`). Bajo ifSUB este modelo **no puede darla**:
`pfaeq` no se genera (`model.gms:1117`, `$(... and not ifSUB)`) y GAMS deja `pfa`
LIBRE donde hay flujo de factor — solo la fija donde `not xfFlag`
(`iterloop.gms:144`). Python es FIEL a eso, asi que quedan **14 columnas sin fila
en base y 65 en shock**.

GAMS las absorbe con `gtap.holdfixed=1` (`model.gms:1424`) mas filas libres
declaradas (`pfteq` sin `.var`, `model.gms:1413`). Pyomo/PATH no tiene ese
estado, de modo que ASL aborta.

**Fijar esas 14 NO es la salida:** ya se midio y baja el CHECK de ~93% a ~80% de
paridad. Fidelidad sobre velocidad.

## Medicion (gtap7_3x3, unica diferencia = `EQUILIBRIA_GTAP_JAC_MODE`)

### Via DRIVER (`GTAPMultiPeriodModel` + altertax, closure=None)

|                          | `asl` (DEFAULT) | `reverse_numeric` |
|--------------------------|-----------------|-------------------|
| restricciones > 1e-6     | **607 / 845**   | **8 / 845**       |
| residuo peor             | **8.6e+10**     | **0.032**         |
| codigos base/check/shock | 0 / **4** / 0   | 2 / **1** / 9     |
| resid `eq_rgdpmp[USA]`   | 0.103968        | 3.55e-15          |
| `rgdpmp[USA]`            | 10.0969         | **19.5320**       |

Los codigos aceptados como exito son `{1, 2}`. `asl` devuelve 0/4/0 — ninguno lo
es.

Las tres peores filas bajo `asl` son `_eq_xi_recal_ROW_*_shock` (8.6e10, 5.0e8,
5.6e6). No son decorativas: el driver las ANADE desactivando `eq_xi`
(`gtap_multiperiod_driver.py:877`), asi que son el sistema real.

Reproducible: dos corridas por modo, identicas hasta el ultimo digito.

### Via BLOQUES (`build_block_model`, `base_calibrated=True`) — NO afectada

|                       | `asl` | `reverse_numeric` |
|-----------------------|-------|-------------------|
| within_1pp vs GEMPACK | 73.2% | 73.2%             |
| mediana pp            | 0.305 | 0.305             |
| land response         | -3.033% | -3.033%         |
| restricciones > 1e-6  | 0 / 1103 (peor 3.2e-11) | — |

**Identico hasta el ultimo digito.** Por eso los gates pasan en verde: miden la
via de bloques, no la del driver.

## Validacion del instrumento

El mismo chequeo de residuos sobre la via de bloques da 0/1103 violaciones (peor
3.2e-11), coherente con los gates verdes. El contador no esta roto: cuando dice
607/845 en la via del driver, es real.

## Bisect

`9ef28cb` es el primer commit malo (puso `asl` como default).

- `eacc53f` (padre): PASS
- `9ef28cb` en adelante: FAIL con residuo **identico** `0.103968` en los 10
  commits medidos. Un switch, no una deriva.

El mensaje de `9ef28cb` dice "Tests: 42 passed — MCP parity, the .nl coefficient
gate, GEMPACK, and the three multiperiod Fisher suites". El test que lo habria
cazado no estaba en esa lista.

## Estado en `main` hoy (2026-09-24, re-verificado)

- El default sigue siendo `asl` (`solver/path_capi.py:84`).
- El bug SIGUE VIVO: forzar `asl` en
  `test_solve_multiperiod_solves_m_not_slices` lo hace FALLAR; con
  `reverse_numeric` pasa.
- Hay workaround —no arreglo— en dos tests, que fuerzan `reverse_numeric`:
  `test_multiperiod_driver.py:81` y `test_wheel_smoke.py:75`.

## Que NO se midio

- **Cual punto es el fiel contra GEMPACK en la via del driver.** No hay fixture
  GEMPACK para esa configuracion (altertax + closure=None). Lo que si se midio
  es que `asl` NO RESUELVE el sistema que se le da, y eso no depende de cual sea
  el punto economicamente correcto.
- Otros datasets (solo 3x3) y otros modos.
- Si `reverse_numeric` es lento en la via del driver. El 2,5x de `9ef28cb` se
  midio en 20x41 por la via de BLOQUES, que aqui no cambia.

## Recomendacion

Dos opciones; la segunda es la recomendada:

1. Volver el default a `reverse_numeric` — cuesta el 2,5x en 20x41 por una via
   que NO esta rota.
2. Usar `reverse_numeric` SOLO en la via del driver (multiperiodo con
   recalibracion), dejar `asl` en bloques, y **anadir un gate que mire los
   residuos de esa via**. Eso ultimo es lo que falta de verdad: el bug vivio
   semanas porque no habia nada midiendo ahi, y hoy sobrevive como dos
   `monkeypatch.setenv` sueltos que cualquiera puede borrar sin enterarse.
