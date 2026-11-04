# Spec — Limpieza del repositorio equilibria

**Fecha:** 2026-09-16
**Rama:** `cleanup-equilibria`
**Objetivo declarado:** eliminar del repo lo que confunde a un lector nuevo — humano o IA — sin
tocar la fidelidad de los modelos contra GAMS y GEMPACK.

---

## 1. Diagnóstico

Todo lo que sigue está medido en el árbol de trabajo, no inferido.

### 1.1 Lo que está roto de verdad

| Problema | Magnitud | Consecuencia |
|---|---|---|
| Rutas absolutas `/Users/marmol` en archivos versionados | 65 archivos | El repo no corre en otra máquina |
| Rutas funcionales (no comentario) en `src/` | `templates/pep_pyomo/pep_pyomo_solver.py:31,32,52` | Código publicado en PyPI busca `.dylib` en el disco del autor |
| Rutas a scratchpads de sesión muertos | `scripts/pep/mcp_diff_eq_families.py`, `scripts/gtap/compare_nus333_vs_neos.py` | Rotos hoy mismo |
| Symlink `superpowers` versionado apuntando fuera del repo | 1 entrada en el índice git | Roto para quien clone |
| Código citando `CLAUDE.md` como autoridad | ~6 archivos en `src/`, `tests/`, `scripts/` | Cita un archivo ausente del repo (es symlink a `dev-tools`) |
| **CI corre 6 de 172 archivos de test (3,5 %)** | 166 sin cubrir, incluido todo `babel/` | La red de seguridad está desconectada |
| **El gate obligatorio no vigila `blocks/gtap`** | `scripts/gtap/check_parity_gates_stamp.py:28-38` | Se puede empujar un cambio de ecuación sin re-correr gates |

El origen de la lista blanca de CI es el commit `2f5c128` ("ci: stabilize tests ... for linux
runner"): se enumeró lo que pasaba y el resto nunca se reconectó. Es deuda acumulada, no una
decisión vigente.

### 1.2 Monolito GTAP vs bloques — la premisa inicial era incorrecta

La instrucción original era «borrar los monolitos que ya tengan contraparte en bloques». La
medición refuta que eso sea posible hoy, por dos razones independientes:

**(a) El camino de bloques depende del monolito en runtime.** Los módulos `blocks/gtap/*.py`
sólo importan `gtap_parameters` — las 18 menciones a `gtap_model_equations` en ellos son
comentarios de procedencia (`VERBATIM from ... líneas X-Y`). Pero el *composer* sí lo importa:

- `templates/gtap/gtap_block_model.py:42` — import a nivel de módulo
- `:242-246` — instancia un shim `GTAPModelEquations(...)` para `apply_production_scaling()` y
  `_align_xi_xaa_post_scaling()`
- `:340-341`, `:359-360` — monkey-patch de `GTAPModelEquations.build_model`
- `templates/gtap/__init__.py:58` — cualquier `import equilibria.templates.gtap` carga el monolito

**(b) El monolito es oráculo de fidelidad en 3 de los 5 gates.** Según
`scripts/gtap/run_parity_gates.py:41-51`:

| Gate | Camino | Referencia |
|---|---|---|
| `test_gtap7_mcp_parity.py` | monolito | GAMS/PATH |
| `test_gtap7_nlp_parity.py` | monolito | GAMS/IPOPT |
| `test_gtap7_nl_parity.py` | monolito (vía `nl_compare`) | `.nl` de GAMS |
| `test_gtap7_gempack_parity.py` | **bloques** | GEMPACK SL4 |
| `tests/templates/gtap_logvalue/` | bloques | Julia/port |

Borrar `gtap_model_equations.py` provocaría *collection error* en ~34 archivos de test, incluidos
los 9 que ejercitan **solo** bloques.

**Cobertura de ecuaciones:** 101 familias `Constraint(...)` en el monolito frente a 100 `eq_*` en
bloques. De las 7 exclusivas del monolito, **6 se desactivan en la línea siguiente a su
declaración** y nunca entran al `.nl`:

```
prf_y        gtap_model_equations.py:5604 → .deactivate() en :5605
eq_ps                              :5920 → .deactivate() en :5921
eq_pe                              :5981 → .deactivate() en :5982
eq_pe_route                        :5996 → .deactivate() en :5999
eq_xet_agg                         :6013 → .deactivate() en :6014
eq_xe_xw                           :6026 → .deactivate() en :6027
eq_pmuv                            :7983 ← único gap real
```

**La asimetría de modos va al revés de lo supuesto:** `capFlex` (`demand_utility.py:584`),
`capFixDp` (`:116,166`), `base_calibrated=True` (`gtap_block_model.py:372,440-446`) y el basis
`gempack` (`gtap_contract.py:309-321`) existen **sólo en bloques**. El monolito implementa 2 de los
5 `savf_flag` declarados en `gtap_contract.py:357`.

**Conclusión:** los bloques son un superconjunto funcional, pero el monolito no es retirable
mientras sea dependencia del composer y oráculo contra GAMS. Pasa a deuda documentada.

### 1.3 Ruido de proceso

- **Ramas remotas:** 41. Borrables 39 (14 mergeadas por fast-forward + 23 con PR MERGED por squash
  + 2 con PR CLOSED). Se conservan 4 hasta cerrar sus PRs.
- **PRs abiertos:** #1 (HAR loader), #8 y #9 (altertax), #14 (v62 multiperiodo). Los tres temas
  ya viven en `main`, reimplementados: `babel/har/` completo, `templates/gtap/altertax/` con 2
  tests, multiperiodo con 9 tests.
- **Ramas con trabajo único jamás publicado (10 sin PR):** destacan `perf/poi-fase0` y
  `poi/fase0-fixes` (backend POI, septiembre — lo más reciente del repo), `gtap6-template-impl`
  (+17, cierra 269 DOF), `feat/pyomo-jax-translator` (+23).
- **Roadmap:** no existe versionado. Vive fuera del repo, como symlinks a `dev-tools`, invisible
  para quien clone.
- **Docs:** 99 markdown en 10 subdirectorios, sin índice de entrada.

---

## 2. Principio rector

**Nada se borra sin una red de seguridad que lo verifique.**

Línea base medida: **1.145 tests recolectan sin error** (`uv run --frozen python -m pytest tests
--collect-only`). Esa es la red; hoy CI sólo ejercita el 3,5 % de ella. Por eso reconectar CI
precede al borrado: si destapa fallos ocultos, quiero saberlo antes de eliminar código, no después.

---

## 3. Plan por fases

Riesgo creciente. Cada fase termina con la suite en verde antes de pasar a la siguiente.

### Fase 0 — Línea base y cierre del agujero del gate
1. Ejecutar la suite completa y registrar el resultado como referencia.
2. Añadir `src/equilibria/blocks/gtap` a `INPUT_TREES` en
   `scripts/gtap/check_parity_gates_stamp.py:28-38`.

*Riesgo: nulo. El punto 2 corrige un agujero por el que hoy pasa cualquier cambio de ecuación en
bloques sin re-correr gates.*

### Fase 1 — Portabilidad
1. `templates/pep_pyomo/pep_pyomo_solver.py`: sustituir las 3 rutas absolutas funcionales por
   resolución vía variable de entorno con fallback.
2. Los 2 scripts con rutas a scratchpads muertos: parametrizar.
3. Comentarios `Reference: /Users/marmol/...` (~62 archivos): reescribir a rutas relativas o a la
   referencia bibliográfica del modelo.
4. Eliminar del índice git el symlink `superpowers`; añadirlo a `.gitignore`.
5. Reemplazar las citas a `CLAUDE.md` por la sección correspondiente de la documentación del repo.

*Riesgo: bajo. Reversible.*

### Fase 2 — Higiene sin código
1. Cerrar los PRs #1, #8, #9 y #14 con comentario indicando dónde vive hoy la funcionalidad.
2. Borrar las 39 ramas remotas cerradas. Antes, `git tag archive/<rama>` sobre cada una de las 10
   sin PR, para no perder el trabajo no publicado.
3. Sacar del control de versiones los artefactos de corridas (`runs/`, `reports/`, `results/`;
   43 archivos) y añadirlos a `.gitignore`.

*Riesgo: bajo. Las ramas quedan recuperables por tag.*

### Fase 3 — Reconectar CI
1. Cambiar el job `python-tests` de lista blanca de 6 archivos a la suite completa con
   `-m "not gams"`.
2. Corregir o marcar explícitamente lo que falle. Un test que se salta debe decir por qué.

*Riesgo: medio — puede destapar fallos ocultos. Ese es justamente el objetivo.*

### Fase 4 — Borrado

**Borrado inmediato (cero referencias verificadas):**

```
src/equilibria/templates/gtap/gtap_model_full.py        687 líneas, 0 en src, 0 en tests
src/equilibria/babel/gdx/reader_new_decoder.py
src/equilibria/sam_tools/config_loader.py
src/equilibria/simulations/adapters/pep_co2.py          arrastra los 3 pep_co2_*.py
src/equilibria/templates/data/pep/generate_gdx.py
src/equilibria/templates/data/pep/generate_val_par.py
```

Más ~20 funciones/clases públicas sin referencia alguna (`babel/gdx/decoder.py:233`,
`backends/pyomo_equations.py:74,109,123`, `baseline/compatibility.py:679`,
`core/calibration_data.py:400`, `core/calibration_mixin.py:277,308,325`,
`sam_tools/{ieem_raw_excel.py:291, mip_raw_excel.py:347, selectors.py:73,86}`,
`templates/gtap/altertax/calibration_sequence.py:223`,
`templates/gtap/gtap_std7_mapping.py:292,317`, `templates/gtap_julia/solution.py:116`) y 2 privados
muertos (`_logging.py:29`, `sam_tools/mip_to_sam_transforms.py:40`).

**Las 6 ecuaciones `.deactivate()`** de `gtap_model_equations.py` (listadas en §1.2). Decisión
explícita del autor: son inertes, nunca entran al `.nl`, y su eliminación no afecta la fidelidad
que los gates miden. Se verifica con los 5 gates en verde tras el borrado.

**Deuda documentada — NO se borra:**

| Elemento | Razón | Primer corte cuando se aborde |
|---|---|---|
| `gtap_model_equations.py` | Dependencia runtime del composer + oráculo GAMS en 3/5 gates | Extraer `apply_production_scaling` y `_align_xi_xaa_post_scaling` a módulo propio |
| `eq_pmuv` | Gap declarado en `blocks/gtap/__init__.py:47-53`, no implementado en el composer | Voltear `pmuv` de Param a Var cuando `rmuv`/`imuv` no estén vacíos |
| `blocks/{production,trade,demand,institutions,equilibrium}/` | 4,1k líneas; consumidores sólo en `simple_open` y ejemplos | Decidir si son API pública o se retiran |
| `pep_cri_*.py` | Latente: 0 tests, última edición sustantiva 2026-04-22 | — |
| `pep_levels`, `pep_parity_pipeline`, `pep_scenario_parity` | Zombis: vivos sólo por sus propios tests | — |
| Ramas POI, gtap6, pyomo-jax | Trabajo único sin publicar | Preservadas como tags `archive/*` |

*Riesgo: alto. Mitigación: suite completa verde + 5 gates de paridad verdes tras cada borrado.*

### Fase 5 — Historial
1. `git branch backup/pre-filter-repo` y tag con el SHA actual de `main`.
2. `git filter-repo` eliminando el trailer `Co-Authored-By: Claude` de los 487 commits afectados
   (de 786 totales; el autor es `mmc00` en los 786 — no cambia).
3. Force-push a `main`.

*Riesgo: irreversible en el remoto. Invalida clones existentes y los SHAs de los PRs mergeados.
Autorizado explícitamente por el autor. El backup local permite reconstruir.*

---

## 4. Entregables de documentación

1. **`ROADMAP.md` versionado en la raíz** — estado real medido (F0→F12), qué está cerrado y con
   qué números, y el registro de deuda técnica de §3 fase 4.
2. **Índice de `docs/`** — un punto de entrada; archivar lo histórico; eliminar lo que contradiga
   hallazgos ya establecidos (p. ej. documentos que atribuyan a linearización gaps que resultaron
   ser cruces de closure o el seed).
3. **Mapa monolito↔bloques** — documento corto que explique la división real: bloques como
   superconjunto funcional, monolito como oráculo GAMS y dependencia del composer. Es la pieza que
   evita que un lector futuro repita el error de creer que son alternativas intercambiables.

---

## 5. Criterios de aceptación

- [ ] `grep -rn "/Users/marmol" --include="*.py" src tests scripts` no devuelve nada
- [ ] `git ls-files -s | awk '$1=="120000"'` no lista `superpowers`
- [ ] CI ejercita la suite completa, no una lista blanca de 6 archivos
- [ ] `INPUT_TREES` incluye `src/equilibria/blocks/gtap`
- [ ] Los 5 gates de `run_parity_gates.py` en verde
- [ ] La suite recolecta ≥ 1.145 tests y pasa igual o mejor que la línea base
- [ ] 0 PRs abiertos obsoletos; ≤ 6 ramas remotas vivas; trabajo no publicado preservado en tags
- [ ] `ROADMAP.md` existe en la raíz y refleja el estado medido
- [ ] Ningún archivo versionado cita `CLAUDE.md` como fuente de verdad
