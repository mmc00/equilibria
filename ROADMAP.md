# Roadmap de equilibria

Estado real, medido, del proyecto — no aspiracional. Cada cifra que aparece aquí sale de una
corrida registrada en `docs/findings/` o de un gate de `scripts/gtap/run_parity_gates.py`.

Última actualización: 2026-09-16.

---

## Qué está cerrado

### Validación GTAP contra GEMPACK

El modelo GTAP 7 en Python reproduce a GEMPACK dentro de tolerancia en toda la rejilla de
agregaciones, con `base_calibrated=True` y el closure emparejado:

| Dataset | %-change | Niveles |
|---|---|---|
| 3x3 | 99,5 % | — |
| 15x10 | 92,3 % | 97,6 % |
| 20x41 | 94,7 % | 98,5 % |

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

### Infraestructura

- **Lectores nativos** HAR (GEMPACK) y GDX (GAMS), con escritura y round-trip semántico.
- **Solver libre**: Newton-TR + MUMPS resuelve el 20x41 (3,43 M variables) sin CONOPT ni IPOPT.
- **Pipeline MIP → SAM** con cierre estructural (ver README).
- **Cascada de paridad**: ocho herramientas de diagnóstico Python↔GAMS.

---

## Deuda técnica

Lo que sigue está medido y pendiente. Ninguna entrada es especulativa.

### Bloqueantes conocidos

| Deuda | Detalle | Primer paso |
|---|---|---|
| El Jacobiano `asl` no resuelve la vía del **driver** | `asl` es el default desde `9ef28cb` y funciona en bloques, pero en la vía multiperiodo del driver **aborta**: `success=False`, los valores nunca se escriben y lo que se observa es el residuo del modelo SIN resolver — 607 de 845 restricciones violadas contra 8 con `reverse_numeric` (gtap7_3x3). Ningún gate lo mira: van con `skip_base_solve=True`. Hoy sobrevive como dos `monkeypatch.setenv` en tests. Medido en `docs/findings/gtap_asl_jacobian_driver_2026-09-24.md` | Usar `reverse_numeric` sólo en la vía del driver y **añadir un gate que mire los residuos de esa vía** |
| `gtap_model_equations.py` no es retirable | Sólo por su papel de **oráculo GAMS en 3 de 5 gates**. Ya NO es dependencia de runtime del camino de bloques: el escalado se extrajo a `gtap_benchmark_scaling.py` y la reflexión multiperiodo pide su SP por método, así que `gtap_block_model.py` no lo importa (2026-09-16) | Migrar los 3 gates GAMS a medir bloques, comprobando que los números no se mueven |
| `eq_pmuv` sin portar a bloques | Declarado en `blocks/gtap/__init__.py:47-53`, no implementado en el composer. Sólo muerde con `rmuv`/`imuv` no vacíos, que ningún dataset del gate usa | Voltear `pmuv` de Param a Var cuando el closure lo pida |

### Tests rotos en main (anteriores a la limpieza de 2026-09-16)

| Test | Síntoma |
|---|---|
| ~~`test_gtap_blocks_form[ClosureBlock]`~~ | **Resuelto 2026-09-16.** No era un defecto del test: el split de los agregados de Fisher condensó `eq_pfact`/`eq_pwfact` a 4 variables, así que su esqueleto no puede coincidir con el del monolito. `_SPLIT_AUX_EQS` las exenta de la comparación de forma —con conteo de exenciones— y la equivalencia se mide sobre los valores |
| `test_multiperiod_driver::test_solve_multiperiod_solves_m_not_slices` | Falla aislado y de forma reproducible |
| `test_cascade_layers`, `test_cascade_run`, `test_probe` (×2) | Fallos preexistentes en parity |

Verificado: fallan igual en `a3490e4`, el commit anterior a la limpieza.

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
