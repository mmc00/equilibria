# Roadmap de equilibria

Estado real, medido, del proyecto — no aspiracional. Cada cifra que aparece aquí sale de una
corrida registrada en `docs/findings/`, de un gate de `scripts/gtap/run_parity_gates.py` o de
`scripts/paper/run_tables.py`.

Última actualización: 2026-10-27.

Cada tema lleva un ID (`R-…`) que se usa igual en el roadmap interno del autor, para que los
dos documentos no se separen.

---

## Prioridades

| Prioridad | ID | Qué | Estado |
|---|---|---|---|
| P1 | `R-PYPI` | Publicar en PyPI | Pendiente: el paquete **todavía no está en PyPI**; hoy se instala desde el repositorio |
| P1 | `R-BOUND-WARN` | Avisar cuando un solve termina con `code=1` y un precio apoyado en su piso de 1e-3 | Pendiente (ver Deuda técnica) |
| P1 | `R-NL-GATE` | Que el gate `.nl` compare de verdad (#23) | Pendiente (ver Bloqueantes) |
| P2 | `R-MONOLITH` | Retirar el monolito `gtap_model_equations.py` | Pendiente (ver Bloqueantes) |
| P2 | `R-DRIVER` | Un driver que sólo orquesta períodos, con una receta por modelo, usable también por PEP | Pendiente |
| P2 | `R-DYN` | Períodos como lista → GTAP recursivo dinámico | Pendiente (Siguiente 1) |
| P2 | `R-CALIB` | Cada bloque calibra lo suyo desde una interfaz de datos común | Pendiente (Siguiente 2) |
| P2 | `R-PMUV` | Portar `eq_pmuv` a bloques | Pendiente (ver Bloqueantes) |
| P2 | `R-BOOK` | Comparar contra las cifras impresas del libro de Burfisher | Pendiente (ver Validación) |
| P3 | `R-PERF` | Backend de evaluación alternativo (`PERF-EVAL-1`) | Diseño, sin código |
| P3 | `R-HAR-VIEW` | Visor de HAR/`.sl4` para cualquier sistema | Pendiente |
| P4 | `R-ENV` | Un bloque ambiental como prueba de modularidad | Pendiente (Siguiente 3) |

---

## Qué está cerrado

### Validación GTAP contra GEMPACK

Arancel uniforme de importación del 10 %, comparado en niveles celda por celda contra GEMPACK
(Gragg 8-16-32), con el closure por defecto de GEMPACK (`capFlex`, RORDELTA = 1). Medido con
`scripts/gtap/compare_gempack_levels.py` (lo corre `scripts/paper/run_tables.py`); todas las
corridas terminan con `code=1`:

| Dataset | Solver | Celdas | ≤0,1 % | ≤0,5 % | Máx |
|---|---|---|---|---|---|
| 3x3 | PATH (MCP) | 776 | 100 % | 100 % | 0,099 % |
| 3x4 | PATH (MCP) | 1.058 | 100 % | 100 % | 0,099 % |
| 5x5 | PATH (MCP) | 2.472 | 100 % | 100 % | 0,097 % |
| 10x7 | PATH (MCP) | 9.130 | 97,85 % | 100 % | 0,35 % |
| 15x10 | PATH (MCP) | 24.842 | 97,96 % | 100 % | 0,42 % |
| 20x41 | Newton-TR (MUMPS) | 210.906 | 99,28 % | 100 % | 0,40 % |

Las 249.184 celdas quedan dentro de 0,5 %.

**Dos condiciones que no son negociables** y que costaron meses descubrir:

1. **`base_calibrated=True`.** Sin esto el match cae a ~56 %. El residuo que durante mucho tiempo
   se atribuyó a linearización era el *seed*.
2. **El closure debe coincidir.** `capFlex` contra un fixture `capFix` da 70-80 % de falso
   desajuste. `capFlex` ↔ default y `capFix` ↔ swap `dpsave`/`del_tbalry` no son intercambiables.

El residuo que queda en `qxs` (comercio bilateral) **no es un bug**: equilibria coincide con
GEMPACK (0,076 pp) mejor que GAMS en niveles (0,97 pp). Ver
`docs/findings/` para la descomposición término a término.

### Validación contra GAMS

Los 5 gates de `run_parity_gates.py` cubren ambos caminos: el monolito es el oráculo contra GAMS
(mcp, nlp, nl) y los bloques contra GEMPACK. Ver `docs/architecture/monolito_vs_bloques.md`.

Matriz MCP + NLP (18 + 14 filas, 5 datasets, CES y Cobb–Douglas `altertax`, ifSUB 0 y 1): todas
las celdas de los tres períodos dentro de 1 % de GAMS, `code=1` en las 96 etapas (medido
2026-10-27, después de #174; antes el `shock` de las filas `altertax` daba 99,08–99,91 % porque
su arancel no entraba al solve).

### Ejercicios del libro de Burfisher (3ª ed.)

`scripts/gtap/run_burfisher.py`, dataset `nus333`:

- Contra GAMS: 35 de 41 ejercicios con toda celda dentro de 0,1 %.
- Contra GEMPACK (Gragg): los 34 que resuelven ambos programas, toda celda dentro de 0,5 %.
- ME4A/ME4B no tienen solución no lineal en ningún programa (GAMS infactible, GEMPACK aborta).

### Infraestructura

- **Lectores nativos** HAR (GEMPACK) y GDX (GAMS), con escritura y round-trip semántico.
- **Solver libre**: Newton-TR + MUMPS resuelve el 20x41 (3,43 M variables) sin CONOPT ni IPOPT.
- **Pipeline MIP → SAM** con cierre estructural (ver README).
- **Cascada de paridad**: ocho herramientas de diagnóstico Python↔GAMS.

---

## Plan de arquitectura

Decisiones tomadas con el autor (octubre de 2026) y su estado. A diferencia del resto del
documento, esta sección no lleva cifras: dice qué se va a hacer y en qué orden.

### Hecho o en curso

| Qué | Estado |
|---|---|
| Una sola clave de caché para el modelo y la semilla | Hecho (#101) |
| Preparar cada periodo con una receta (`PeriodPreparer`) | Hecho (#103, #108) |
| Un nombre, una declaración: `add_block` rechaza una variable, parámetro o ecuación declarada distinto por dos bloques, y cada bloque lista en `Block.uses` lo que lee de otros | Hecho (#139) |
| Una sola forma de aplicar shocks: el shock vive en el `ShockBlock` y se escribe con `apply_shock` | Hecho (#113, #174). El arancel +10 % por defecto también se escribe en el `ShockBlock` (`shocks.apply_default_tariff`) y salieron ~900 líneas de reconstrucción del driver; `solve_multiperiod` rechaza un modelo sin `ShockBlock` |
| La clave de caché de seed y modelo incluye Python y el stack numérico; `uv.lock` fija las versiones con que se midió el paper | Hecho (#172) |
| Reglas escritas: ADR 0001-0008, `GLOSSARY.md` y especificaciones por área en `specs/` (propuesta → sync → archivo), verificadas en CI por `spec-check` | Hecho (#173) |
| Sacar el monolito del test de forma | Pendiente (`R-MONOLITH`). Incluye generar desde `Block.uses` la lista de variables prestadas de `test_gtap_blocks_form.py` |
| Leer GEMPACK en cualquier sistema: la solución `.sl4` sin `sltoht` | Hecho (#148): `read_sl4_levels`, y `equilibria compare` acepta `.sl4`. Falta un visor tipo ViewHAR (`R-HAR-VIEW`) |

### Siguiente, en este orden

1. **Periodos como lista, con una receta de transición entre uno y otro** (`R-DYN`). Hoy los periodos
   son fijos (`PERIODS = (base, check, shock)`) y el driver asume esa secuencia. Los bloques,
   en cambio, ya no saben de periodos: escriben sus ecuaciones para uno solo y el índice de
   tiempo lo agrega la copia del SP. Con la lista, un **modelo recursivo dinámico** es
   resolver un año a la vez y, entre años, actualizar los stocks (capital, población,
   productividad) con un bloque de dinámica.
2. **Cada bloque calibra lo suyo desde una interfaz de datos común** (SAM o HAR; `R-CALIB`). Hoy GTAP
   calibra en un archivo central (`blocks/gtap/_derived_params.py`) a partir de la base GTAP,
   y PEP calibra desde la SAM con código propio. `Block` ya trae `CalibrationMixin`, pero sólo
   lo usan los bloques genéricos.
3. **Un bloque ambiental (IEEM)** (`R-ENV`) como primera prueba de que agregar un módulo es fácil:
   emisiones y capital natural como bloque propio que declara lo que usa de producción.
   `sam_tools` ya lee las SAM de IEEM, pero sólo su parte económica.

---

## Deuda técnica

Lo que sigue está medido y pendiente. Ninguna entrada es especulativa.

### Rendimiento del 20x41 — `PERF-EVAL-1` (`R-PERF`)

Medido con `scripts/gtap/phase_times.py` (3 periodos, 3 solves, `code=1`, iteraciones 3/3/22):

| Máquina | Total |
|---|---|
| macOS, M3 Pro | 7,2 min |
| GitHub Actions, Linux (AMD EPYC 7763) | 14,5 min |
| GitHub Actions, Windows (mismo EPYC) | 23,7 min |
| Laptop Windows, i7-1365U | 34,9 min |
| GEMPACK, mismo laptop | 4,3 min |

- **Resolver es el 11 %** (Newton-TR + MUMPS). El 89 % es construir y preparar el modelo
  Pyomo antes de cada solve.
- **Windows es 1,64× más lento que Linux con el mismo hardware**, de forma pareja en todas las
  fases: no es una fase concreta, es el trabajo de Python.
- **Intentos anteriores.** El backend POI (`archive/perf/poi-fase0`) no escaló: la generación
  de IR no escala. El traductor JAX (`archive/feat/pyomo-jax-translator`) es correcto (96/96
  familias de ecuaciones coinciden con Pyomo a <1e-7, Jacobiano idéntico) pero **nunca se midió
  activo en el 20x41**, y compilar ya cuesta ~8,4 min en el 3x3. No está descartado: falta medirlo.

Próximo paso: un **backend de evaluación alternativo** (`PERF-EVAL-1`), con Pyomo como oráculo
de paridad. Antes de avanzar tiene que pasar un prototipo con dos gates: igualdad exacta con
Pyomo y escala medida en 3x3, 10x7 y 15x10. Estado: **diseño, sin código**.

### Validación contra la literatura — pendiente (`R-BOOK`)

Los gates actuales miden a equilibria contra GAMS y contra GEMPACK: dos
implementaciones de referencia del mismo modelo. Falta la tercera comprobación, la
que no depende de ninguna de las dos: **reproducir los ejercicios publicados del
libro de texto estándar** (Burfisher, *Introduction to Computable General
Equilibrium Models*, 2ª ed.), cada uno como un test sobre la matriz.

Criterio: cada ejercicio ≥99 % con `code=1`, o `N/A` con el motivo escrito.

Estado: los 41 ejercicios ya se comparan contra GAMS y GEMPACK (ver "Qué está cerrado").
Falta lo que pide este apartado: comparar contra las cifras **publicadas del libro**, no
contra otra implementación.

La deuda de `nus333` para gtap7 (`gdpmp[USA]` −97,93 % contra +4,58 %) está resuelta:
`tests/templates/gtap/test_nus333_neos_parity.py` pasa en `main` (verificado 2026-10-27).

### Bloqueantes conocidos

| Deuda | Detalle | Primer paso |
|---|---|---|
| `R-NL-GATE`: gate `.nl` pasa por vacuidad ([#23](https://github.com/mmc00/equilibria/issues/23)) | El emparejamiento es por nombre y las fixtures de GAMS no traen sus sidecars `.row`/`.col`, así que sus filas se parsean anónimas. **Medido 2026-09-24: los 5 datasets del gate comparan 0 constraints**, no sólo `gtap7_3x3` como decía el issue. Ningún `.row`/`.col` existió nunca en el repo | Regenerar las fixtures con sidecars: el `.gms` debe estar en forma NLP (CONVERT no escribe `.nl` desde un MCP) y la opción es `AmplNL <archivo>`. Probado local con GAMS v53, sin NEOS |
| `R-MONOLITH`: `gtap_model_equations.py` no es retirable | Desde #174 `solve_multiperiod` rechaza un modelo sin `ShockBlock`, así que siete scripts de `scripts/gtap/` que lo resolvían quedaron sin uso hasta migrarlos: `measure_gate_tols`, `measure_gtap_pure_tols`, `measure_nlp_vs_nlp`, `measure_nlp_vs_nlp_altertax`, `diag_mp_3x3`, `bench_nlp_timing`, `gen_linearization_study`. Fuera de eso sigue sin ser retirable sólo por su papel de **oráculo GAMS en el gate `nl`** — medido 2026-09-24: los otros cuatro (`mcp`, `nlp`, `gempack`, `coverage_matrix`) no lo leen, y `docs/architecture/monolito_vs_bloques.md:95` ya lo decía. Ya NO es dependencia de runtime del camino de bloques: el escalado se extrajo a `gtap_benchmark_scaling.py` y la reflexión multiperiodo pide su SP por método, así que `gtap_block_model.py` no lo importa (2026-09-16). Desde 2026-09-24 además hay que **pedirlo**: un modelo sin `_sp_source` va a bloques, y un valor no reconocido levanta | Migrar el gate `nl` a medir bloques, comprobando que los números no se mueven |
| `R-BOUND-WARN`: `code=1` con un precio en su piso | Un solve puede terminar con `code=1` y un precio apoyado en su cota inferior de 1e-3 (p. ej. `pft[USA,LAND]` en ME4A; también visto en 15x10 ifSUB=1). Es un punto de complementariedad en la cota, no un equilibrio económico, y hoy se reporta como convergido | Que el driver o el solver avisen (o devuelvan otro código) cuando al terminar haya precios en su cota |
| `R-PMUV`: `eq_pmuv` sin portar a bloques | Declarado en `blocks/gtap/__init__.py:47-53`, no implementado en el composer. Sólo muerde con `rmuv`/`imuv` no vacíos, que ningún dataset del gate usa | Voltear `pmuv` de Param a Var cuando el closure lo pida |

### Tests que estaban rotos en main

| Test | Síntoma |
|---|---|
| ~~`test_gtap_blocks_form[ClosureBlock]`~~ | **Resuelto 2026-09-16.** No era un defecto del test: el split de los agregados de Fisher condensó `eq_pfact`/`eq_pwfact` a 4 variables, así que su esqueleto no puede coincidir con el del monolito. `_SPLIT_AUX_EQS` las exenta de la comparación de forma —con conteo de exenciones— y la equivalencia se mide sobre los valores |
| ~~`test_multiperiod_driver::test_solve_multiperiod_solves_m_not_slices`~~ | **Pasa en `main`** (verificado 2026-10-27, `d356595`) |
| ~~`test_cascade_layers`, `test_cascade_run`, `test_probe` (×2)~~ | **Pasan en `main`** (verificado 2026-10-27, `d356595`; 4 casos de `test_probe` se saltan sin el GDX de referencia) |

Los 2 de `test_writer` pasaron a `xfail(strict=True)`: el escritor GDX declara
soportar sólo `Set` y `Parameter`, así que exigirle `Variable`/`Equation` es pedir
funcionalidad sin implementar. Cuando se implemente, el test avisará de que sobra la marca.

### Snapshot Fisher: arreglado, y el lado POI sigue archivado

`blocks/gtap/__init__.py` item 3 exige que el composer sobrescriba `pf0`/`xf0`/`mqfactr_bb`/
`mqfactw_bb` con el snapshot post-escalado. Nadie lo hacía: `xf0` quedaba desviado por
exactamente `xscale` (10× en 26 de 45 celdas de `gtap7_3x3`). En el benchmark el sesgo se
cancela, pero fuera de él `eq_pfact` divergía del monolito hasta **1,15 %**.

Arreglado en el camino Pyomo (`apply_fisher_snapshot_overwrite`), portado desde
`archive/poi/fase0-fixes`. Medido: divergencia 1,15 % → **0 exacto**; gates en verde.

**El lado POI sigue sólo en el tag.** Si se retoma esa línea, hay que portarlo junto: con un solo
lado arreglado, una comparación POI-vs-Pyomo pasa de tener un sesgo simétrico —que se cancelaba—
a una asimetría real.

### Solvers: qué hace falta para correr la suite entera

`uv sync` no instala los solvers. Los tests que resuelven modelos se saltan sin ellos,
y lo declaran con un marcador por pieza (ver `tests/conftest.py`):

| Marcador | Necesita |
|---|---|
| `needs_mumps` | `pymumps` (`conda install -c conda-forge pymumps`) |
| `needs_ipopt` | el ejecutable `ipopt`, o `cyipopt` |
| `needs_path` | la librería PATH C-API + el checkout de path-capi-python |

`EQUILIBRIA_REQUIRE_SOLVERS=1` desactiva los guards, para que en local un solver ausente
se note en vez de esconderse tras un skip.

### Los tests GTAP no toleran paralelismo

Compiten por la librería PATH y por las cachés de modelo. Dos corridas simultáneas producen
decenas de fallos espurios: una medición dio 40 fallos que, en aislamiento, se redujeron a 1 real.
**No usar `pytest -n` ni lanzar dos suites a la vez.**

### Código latente o sin consumidores

| Elemento | Estado |
|---|---|
| `blocks/{production,trade,demand,institutions,equilibrium}/` | 4,1 k líneas; consumidores sólo en `simple_open` y ejemplos. Decidir si son API pública |
| `pep_cri_*.py` | Latente: 0 tests, última edición sustantiva 2026-04-22 |
| `pep_levels`, `pep_parity_pipeline`, `pep_scenario_parity` | Zombis: vivos sólo por sus propios tests |
| `templates/gtap_loglevels/` | Marginal: 0 importadores en `src`, 1 test |
| 44 símbolos exportados vía `__init__` | Sin consumidor real (API especulativa) |

---

## Trabajo archivado

Diez ramas con trabajo jamás publicado se conservan como tags `archive/*`. Recuperar una:
`git checkout archive/<nombre>`.

| Tag | Contenido |
|---|---|
| `archive/perf/poi-fase0`, `archive/poi/fase0-fixes` | Backend POI (PyOptInterface) con IR y LLVM. Lo más reciente del repo: la Fase 0 concluye que la premisa se sostiene pero la generación de IR no escala |
| `archive/gtap6-template-impl` | Template GTAP 6: cierra un hueco de 269 DOF con 7 identidades de precio |
| `archive/feat/pyomo-jax-translator` | Traductor Pyomo → JAX |
| `archive/gtap/v62-rollback`, `archive/worktree-runMacos` | Datasets GTAP v7 y reproducción en macOS |
| Otros 5 | Specs, spikes y experimentos de rendimiento |

---

## Convenciones que conviene no reaprender por las malas

- **La fidelidad manda sobre el porcentaje de match.** Nunca revertir un cambio fiel porque baje
  una cifra; la única razón para retirarlo es que sea infiel.
- **GAMS y GEMPACK son la fuente de verdad.** No excluir celdas divergentes para inflar el match.
- **Antes de culpar al modelo**, verificar el closure, el seed y el fixture. El grueso de los
  «bugs» históricos fueron cruces de closure o seeds no calibrados.
- **Las referencias externas se resuelven por variable de entorno** (`EQUILIBRIA_REFS_DIR`,
  `EQUILIBRIA_NUS333_DIR`, `EQUILIBRIA_PATH_CAPI_SRC`, `EQUILIBRIA_PATH_CAPI_LIB_DIR`,
  `EQUILIBRIA_CGE_BABEL_DIR`). Ver `src/equilibria/_local_refs.py`.
- **Los gates de paridad son obligatorios** antes de un push que toque GTAP; el hook
  `scripts/gtap/claude_hooks/block_push_without_gates.py` lo exige.
