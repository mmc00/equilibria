# Limpieza del repositorio equilibria — Plan de implementación

> **Para ejecutores agénticos:** SUB-SKILL REQUERIDO: usar `superpowers:subagent-driven-development`
> (recomendado) o `superpowers:executing-plans` para implementar tarea por tarea. Los pasos usan
> sintaxis de casilla (`- [ ]`) para seguimiento.

**Objetivo:** dejar el repo clonable y ejecutable en cualquier máquina, con CI que ejercite la suite
real, sin código muerto ni ruido de proceso, y con el historial libre del trailer de co-autoría.

**Arquitectura:** seis fases de riesgo creciente. Cada fase termina con la suite de tests en el
mismo estado o mejor que la línea base. El borrado de código va **después** de reconectar CI, no
antes: la red de seguridad existe (1.145 tests) pero hoy sólo se ejercita el 3,5 %.

**Stack:** Python 3.10+, `uv` como runner, pytest, pre-commit (`prek`), GitHub Actions, `gh` CLI,
`git filter-repo`.

**Spec:** `docs/findings/repo_cleanup_spec_2026-09-16.md`

## Restricciones globales

- **La fidelidad manda sobre la limpieza.** Ningún cambio que altere resultados numéricos de los
  modelos es aceptable, por más que mejore la estética del repo. Si un borrado mueve un número,
  se revierte.
- **El intérprete es `uv run --frozen python`.** No hay `.venv` en los worktrees; `python` a secas
  no existe en el PATH de este entorno.
- **Línea base medida:** 1.145 tests recolectados sin error, 1 skip esperado
  (`test_harpy_interop.py`: `harpy3` no instalado).
- **Los 5 gates de paridad** (`scripts/gtap/run_parity_gates.py`) son el criterio de aceptación
  de cualquier cambio que toque `src/equilibria/templates/gtap` o `src/equilibria/blocks/gtap`.
- **Commits sin trailer de co-autoría.** Es precisamente lo que la fase 5 elimina; no reintroducirlo.
- **Mensajes de commit en español**, siguiendo la convención del repo (`tipo(ámbito): descripción`).
- **No tocar `gtap_model_equations.py`** salvo en la Tarea 12 (las 6 ecuaciones inertes), y sólo
  con los 5 gates verdes antes y después.

---

## Estructura de archivos

| Archivo | Responsabilidad | Tarea |
|---|---|---|
| `scripts/gtap/check_parity_gates_stamp.py` | Añadir `blocks/gtap` a `INPUT_TREES` | 1 |
| `src/equilibria/templates/pep_pyomo/pep_pyomo_solver.py` | Quitar rutas personales del descubrimiento de PATH | 2 |
| `scripts/pep/mcp_diff_eq_families.py` | Parametrizar la ruta al JSON de GAMS | 3 |
| ~62 archivos con `Reference: /Users/marmol/...` | Reescribir comentarios de procedencia | 4 |
| `.gitignore`, índice git | Retirar el symlink `superpowers` | 5 |
| 5 archivos que citan `CLAUDE.md` | Apuntar a documentación del repo | 6 |
| `docs/architecture/monolito_vs_bloques.md` | **Nuevo.** El mapa que evita el malentendido | 7 |
| `.github/workflows/tests.yml` | Suite completa en vez de lista blanca | 10 |
| `ROADMAP.md` | **Nuevo.** Estado medido + deuda técnica | 15 |

---

## Fase 0 — Línea base y cierre del agujero del gate

### Tarea 1: Cerrar el agujero de `INPUT_TREES`

Hoy `check_parity_gates_stamp.py` no vigila `src/equilibria/blocks/gtap`. Editar cualquiera de los
10 módulos de bloques GTAP **no invalida el stamp**, así que el hook `block_push_without_gates.py`
deja pasar el push sin re-correr los gates. Los bloques son el oráculo contra GEMPACK: el agujero
es real.

**Archivos:**
- Modificar: `scripts/gtap/check_parity_gates_stamp.py:28-38`
- Test: `tests/scripts/test_check_parity_gates_stamp.py` (crear si no existe)

**Interfaces:**
- Produce: `INPUT_TREES` incluyendo `"src/equilibria/blocks/gtap"`.

- [ ] **Paso 1: Escribir el test que falla**

```python
def test_input_trees_cubre_los_bloques_gtap():
    """Editar blocks/gtap debe invalidar el stamp: es el oráculo contra GEMPACK."""
    from scripts.gtap.check_parity_gates_stamp import INPUT_TREES

    assert "src/equilibria/blocks/gtap" in INPUT_TREES, (
        "blocks/gtap no está vigilado: un cambio de ecuación en bloques "
        "no invalidaría el stamp y el hook dejaría pasar el push sin gates"
    )
```

- [ ] **Paso 2: Verificar que falla**

Ejecutar: `uv run --frozen python -m pytest tests/scripts/test_check_parity_gates_stamp.py -v`
Esperado: FAIL — `blocks/gtap no está vigilado`

- [ ] **Paso 3: Añadir la ruta**

En `scripts/gtap/check_parity_gates_stamp.py`, dentro de `INPUT_TREES`, tras
`"src/equilibria/templates/gtap"`:

```python
    "src/equilibria/blocks/gtap",
```

- [ ] **Paso 4: Verificar que pasa**

Ejecutar: `uv run --frozen python -m pytest tests/scripts/test_check_parity_gates_stamp.py -v`
Esperado: PASS

- [ ] **Paso 5: Commit**

```bash
git add scripts/gtap/check_parity_gates_stamp.py tests/scripts/test_check_parity_gates_stamp.py
git commit -m "fix(gates): vigilar blocks/gtap en INPUT_TREES

Editar los 10 modulos de bloques GTAP no invalidaba el stamp, asi que el
hook block_push_without_gates dejaba pasar el push sin re-correr gates.
Los bloques son el oraculo contra GEMPACK (test_gtap7_gempack_parity)."
```

### Tarea 2: Registrar la línea base

**Archivos:**
- Crear: `docs/findings/cleanup_baseline_2026-09-16.txt`

- [ ] **Paso 1: Ejecutar la suite completa y guardar el resultado**

```bash
uv run --frozen python -m pytest tests -q -m "not gams" \
  2>&1 | tail -30 | tee docs/findings/cleanup_baseline_2026-09-16.txt
```

- [ ] **Paso 2: Anotar el recuento de recolección**

```bash
uv run --frozen python -m pytest tests --collect-only -q 2>&1 \
  | awk -F': ' '/^tests\/.*: [0-9]+$/{s+=$2} END{print "total recolectado:", s}' \
  >> docs/findings/cleanup_baseline_2026-09-16.txt
```

Esperado: `total recolectado: 1145`. Si difiere, **detenerse** y reportar: la línea base cambió
respecto a la medición del spec y hay que entender por qué antes de borrar nada.

- [ ] **Paso 3: Commit**

```bash
git add docs/findings/cleanup_baseline_2026-09-16.txt
git commit -m "docs(cleanup): linea base de la suite antes de la limpieza"
```

---

## Fase 1 — Portabilidad

### Tarea 3: Quitar las rutas personales de `pep_pyomo_solver.py`

Es el caso más grave: código en `src/`, publicado en PyPI, que busca un `.dylib` en el disco del
autor. La función ya respeta `PATH_CAPI_LIBPATH` si está definida (`:28-29`), así que basta con
retirar las rutas personales de la lista de candidatos y dejar la ubicación estándar de GAMS.

**Archivos:**
- Modificar: `src/equilibria/templates/pep_pyomo/pep_pyomo_solver.py:22-56`
- Test: `tests/templates/test_pep_pyomo_paths.py` (crear)

**Interfaces:**
- Consume: la variable de entorno `PATH_CAPI_LIBPATH`, y `EQUILIBRIA_PATH_CAPI_SRC` (nueva).
- Produce: `_ensure_path_lib()` y `_ensure_path_module()` sin rutas absolutas personales.

- [ ] **Paso 1: Escribir el test que falla**

```python
from pathlib import Path


def test_pep_pyomo_solver_sin_rutas_personales():
    """src/ es codigo publicado: no puede referirse al disco del autor."""
    fuente = Path("src/equilibria/templates/pep_pyomo/pep_pyomo_solver.py").read_text()

    assert "/Users/marmol" not in fuente, (
        "pep_pyomo_solver.py busca librerias en el disco del autor; "
        "en cualquier otra maquina la busqueda falla en silencio"
    )
```

- [ ] **Paso 2: Verificar que falla**

Ejecutar: `uv run --frozen python -m pytest tests/templates/test_pep_pyomo_paths.py -v`
Esperado: FAIL — 3 coincidencias de `/Users/marmol`

- [ ] **Paso 3: Reescribir ambas funciones**

Sustituir el cuerpo de `_ensure_path_lib` y `_ensure_path_module` (líneas 22-56) por:

```python
def _ensure_path_lib() -> None:
    """Point PATH_CAPI_LIBPATH at the PATH C-API dylib (like GTAP's run_gtap), so the
    MCP solve is self-contained. Searches the known locations; no-op if already set."""
    import os
    from pathlib import Path

    if os.environ.get("PATH_CAPI_LIBPATH"):
        return
    candidates = [
        Path.cwd() / ".cache" / "path_capi" / "libpath50.silicon.dylib",
        Path("/Library/Frameworks/GAMS.framework/Versions/53/Resources/libpath52.dylib"),
    ]
    for cand in candidates:
        if cand.exists():
            os.environ["PATH_CAPI_LIBPATH"] = str(cand)
            return


def _ensure_path_module() -> None:
    """Make `import path_capi_python` work even when the package isn't pip-installed in the
    active interpreter (e.g. a fresh `uv run` subprocess launched by the parity skill).
    Set EQUILIBRIA_PATH_CAPI_SRC to the checkout's src dir to opt in — no-op otherwise."""
    import importlib.util
    import os
    import sys
    from pathlib import Path

    if importlib.util.find_spec("path_capi_python") is not None:
        return
    src = os.environ.get("EQUILIBRIA_PATH_CAPI_SRC")
    if src and (Path(src) / "path_capi_python").exists() and src not in sys.path:
        sys.path.insert(0, src)
```

- [ ] **Paso 4: Verificar que pasa y que PEP sigue funcionando**

```bash
uv run --frozen python -m pytest tests/templates/test_pep_pyomo_paths.py -v
uv run --frozen python -m pytest tests/templates/pep_pyomo -v -m "not gams"
```
Esperado: ambos PASS. El segundo comando protege contra una regresión en el solver PEP.

- [ ] **Paso 5: Commit**

```bash
git add src/equilibria/templates/pep_pyomo/pep_pyomo_solver.py tests/templates/test_pep_pyomo_paths.py
git commit -m "fix(pep): quitar rutas personales del descubrimiento de PATH

pep_pyomo_solver.py buscaba libpath en /Users/marmol: en cualquier otra
maquina fallaba en silencio. Ahora usa .cache/path_capi del proyecto, la
ruta estandar de GAMS, y EQUILIBRIA_PATH_CAPI_SRC para el checkout."
```

### Tarea 4: Parametrizar el scratchpad muerto

`scripts/pep/mcp_diff_eq_families.py:11` abre un JSON en un scratchpad de una sesión ya terminada.
El script está roto hoy.

**Archivos:**
- Modificar: `scripts/pep/mcp_diff_eq_families.py:11`

- [ ] **Paso 1: Comprobar que está roto**

```bash
uv run --frozen python scripts/pep/mcp_diff_eq_families.py 2>&1 | tail -3
```
Esperado: `FileNotFoundError` sobre `/private/tmp/claude-501/...`

- [ ] **Paso 2: Sustituir la ruta fija por un argumento**

Reemplazar la línea 11 por:

```python
import sys

_ruta_gams = sys.argv[1] if len(sys.argv) > 1 else "gams_inst.json"
gams = {k.upper(): v for k, v in json.load(open(_ruta_gams)).items()}
```

- [ ] **Paso 3: Verificar el mensaje de error útil**

```bash
uv run --frozen python scripts/pep/mcp_diff_eq_families.py 2>&1 | tail -3
```
Esperado: `FileNotFoundError` sobre `gams_inst.json` — un fallo que el usuario puede corregir
pasando la ruta, no una referencia a una sesión muerta.

- [ ] **Paso 4: Commit**

```bash
git add scripts/pep/mcp_diff_eq_families.py
git commit -m "fix(pep): el JSON de GAMS llega por argumento, no por ruta fija

Apuntaba al scratchpad de una sesion terminada: el script estaba roto."
```

### Tarea 5: Reescribir los comentarios `Reference:`

~62 archivos llevan comentarios del tipo `Reference: /Users/marmol/proyectos2/cge_babel/
standard_gtap_7/model.gms`. No rompen nada, pero delatan la máquina y no le sirven a nadie más.
Apuntan a los fuentes GAMS de referencia del modelo estándar GTAP 7.

**Archivos:**
- Modificar: todos los que contengan `Reference: /Users/marmol`

- [ ] **Paso 1: Inventariar**

```bash
grep -rn "Reference: /Users/marmol" --include="*.py" src tests scripts | tee /tmp/refs.txt | wc -l
```

- [ ] **Paso 2: Reescribir con `sed`**

```bash
grep -rl "/Users/marmol/proyectos2/cge_babel/" --include="*.py" src tests scripts \
  | xargs sed -i '' 's|/Users/marmol/proyectos2/cge_babel/|GTAP reference sources: |g'
```

- [ ] **Paso 3: Revisar los residuos a mano**

```bash
grep -rn "/Users/marmol" --include="*.py" src tests scripts
```
Cada coincidencia restante se edita individualmente. Criterio: si es un comentario, se reescribe
como referencia relativa o bibliográfica; si es funcional, se parametriza como en la Tarea 3.

- [ ] **Paso 4: Verificar que nada se rompió**

```bash
uv run --frozen python -m pytest tests -q -m "not gams" 2>&1 | tail -5
```
Esperado: mismo resultado que la línea base de la Tarea 2.

- [ ] **Paso 5: Commit**

```bash
git add -A
git commit -m "docs: rutas de referencia relativas en vez del disco del autor

62 archivos citaban /Users/marmol/proyectos2/cge_babel como fuente de
los .gms de referencia GTAP 7."
```

### Tarea 6: Retirar el symlink `superpowers`

Está versionado (modo `120000`) y apunta a `../../../../proyectos/dev-tools/...`, fuera del repo:
roto para cualquiera que clone.

**Archivos:**
- Eliminar del índice: `superpowers`
- Modificar: `.gitignore`

- [ ] **Paso 1: Confirmar que es un symlink versionado**

```bash
git ls-files -s superpowers
```
Esperado: `120000 ... superpowers`

- [ ] **Paso 2: Sacarlo del índice sin borrar el archivo local**

```bash
git rm --cached superpowers
```

- [ ] **Paso 3: Ignorarlo**

Añadir al final de `.gitignore`:

```
# Herramientas personales montadas por stow (dev-tools), fuera del repo
superpowers
```

- [ ] **Paso 4: Verificar**

```bash
git ls-files -s | awk '$1=="120000"{print $4}'
```
Esperado: los 3 `.gdx`/`.xlsx` de `templates/reference`, **sin** `superpowers`.

- [ ] **Paso 5: Commit**

```bash
git add .gitignore
git commit -m "chore: sacar del control de versiones el symlink superpowers

Apuntaba fuera del repo (dev-tools via stow): roto para quien clone."
```

### Tarea 7: Sustituir las citas a `CLAUDE.md` y escribir el mapa monolito↔bloques

Cinco archivos citan `CLAUDE.md` como fuente de verdad. Ese archivo no está en el repo (es un
symlink a `dev-tools`), así que quien clone lee «por diseño, ver CLAUDE.md» y no encuentra nada.
Dos de las citas explican decisiones de modelado reales que merecen vivir en el repo.

**Archivos:**
- Crear: `docs/architecture/monolito_vs_bloques.md`
- Modificar: `src/equilibria/templates/gtap/gtap_multiperiod_driver.py:753`,
  `src/equilibria/templates/gtap/gtap_model_equations.py:2785`,
  `tests/templates/gtap/test_gtap_baseline_mirror.py:96`,
  `scripts/gtap/compare_nus333_vs_neos.py:196`, `scripts/gtap/_parity_json.py:4`

**Interfaces:**
- Produce: `docs/architecture/monolito_vs_bloques.md`, destino de las citas y entregable 3 del spec.

- [ ] **Paso 1: Escribir el documento de arquitectura**

Crear `docs/architecture/monolito_vs_bloques.md` con este contenido:

````markdown
# GTAP: monolito y bloques

Quien llega nuevo al repo suele suponer que `templates/gtap/gtap_model_equations.py` (el
«monolito») y `blocks/gtap/` (los «bloques») son dos implementaciones alternativas, y que la
primera es legado a la espera de ser borrada. **No lo son.** Este documento existe para evitar
ese error, que ya costó tiempo una vez.

## Quién depende de quién

Los módulos de `blocks/gtap/*.py` sólo importan `gtap_parameters`. Las menciones a
`gtap_model_equations` en sus docstrings son comentarios de procedencia
(`VERBATIM from ... líneas X-Y`), no imports.

Pero el **composer** sí depende del monolito:

- `templates/gtap/gtap_block_model.py:42` — import a nivel de módulo
- `:242-246` — instancia un shim `GTAPModelEquations(...)` para reutilizar
  `apply_production_scaling()` y `_align_xi_xaa_post_scaling()`
- `:340-341`, `:359-360` — monkey-patch de `GTAPModelEquations.build_model`

Borrar el monolito provoca *collection error* en ~34 archivos de test, **incluidos los 9 que
ejercitan sólo bloques**.

## Cobertura

101 familias `Constraint(...)` en el monolito frente a 100 `eq_*` en bloques. De las 7 exclusivas
del monolito, 6 se desactivan en la línea siguiente a su declaración y nunca entran al `.nl`.
`eq_pmuv` es el único gap real, y sólo cuando `closure.rmuv` e `imuv` son ambos no vacíos —lo que
no ocurre en ningún dataset del gate.

## Los bloques son un superconjunto funcional

| Modo | Monolito | Bloques |
|---|---|---|
| `capFix`, `capSFix` | sí | sí |
| `capFlex` | **no** (cae a un `else` genérico) | sí (`demand_utility.py:584`) |
| `capFixDp` | **no** | sí (`demand_utility.py:116,166`) |
| `base_calibrated=True` | **no existe** | sí (`gtap_block_model.py:372,440-446`) |
| basis `gempack` | parcial | sí (`gtap_contract.py:309-321`) |

`gtap_contract.py:357` declara 5 `savf_flag`; el monolito implementa 2.

## Por qué se conserva el monolito

Es el oráculo de fidelidad contra **GAMS** en 3 de los 5 gates de
`scripts/gtap/run_parity_gates.py` (`mcp`, `nlp`, `nl`). El camino de bloques es el oráculo contra
**GEMPACK** (`test_gtap7_gempack_parity.py`) — y sólo él puede serlo, porque ese gate necesita
`capFlex` + `base_calibrated=True`, que el monolito no implementa.

## Deuda: cómo se retiraría

El primer corte real es extraer `apply_production_scaling` y `_align_xi_xaa_post_scaling` de
`GTAPModelEquations` a un módulo propio. Mientras vivan ahí, el monolito es dependencia de runtime
del camino de bloques y no puede retirarse.
````

- [ ] **Paso 2: Reescribir las 5 citas**

En `gtap_multiperiod_driver.py:753` y `gtap_model_equations.py:2785`, sustituir
`CLAUDE.md` por `docs/architecture/monolito_vs_bloques.md`. En
`test_gtap_baseline_mirror.py:96`, `compare_nus333_vs_neos.py:196` y `_parity_json.py:4`,
sustituir `CLAUDE.md` por `CONTRIBUTING.md`.

- [ ] **Paso 3: Verificar que no queda ninguna**

```bash
grep -rn "CLAUDE.md" --include="*.py" src tests scripts
```
Esperado: sin resultados.

- [ ] **Paso 4: Verificar la suite**

```bash
uv run --frozen python -m pytest tests -q -m "not gams" 2>&1 | tail -5
```
Esperado: igual que la línea base.

- [ ] **Paso 5: Commit**

```bash
git add docs/architecture/monolito_vs_bloques.md src tests scripts
git commit -m "docs(arquitectura): mapa monolito<->bloques, y quitar citas a CLAUDE.md

Cinco archivos citaban como autoridad un archivo ausente del repo. El
mapa documenta lo que mas confunde: los bloques son un superconjunto
funcional, pero el monolito es dependencia runtime del composer y
oraculo contra GAMS en 3 de los 5 gates."
```

---

## Fase 2 — Higiene sin código

### Tarea 8: Cerrar los 4 PRs obsoletos

Los tres temas ya viven en `main`, reimplementados tras el PR #44 (que reescribió GTAP como
bloques): HAR nativo en `babel/har/`, altertax en `templates/gtap/altertax/` con 2 tests,
multiperiodo con 9 tests.

- [ ] **Paso 1: Verificar que la funcionalidad está en main**

```bash
ls src/equilibria/babel/har/ src/equilibria/templates/gtap/altertax/
ls tests/templates/gtap/ | grep -E "altertax|multiperiod"
```
Esperado: paquetes completos y los tests citados. Si falta algo, **detenerse** y anotarlo como
deuda antes de cerrar el PR correspondiente.

- [ ] **Paso 2: Cerrar cada PR con su explicación**

```bash
gh pr close 1 --comment "Cerrado por obsoleto. El lector HAR nativo vive hoy en src/equilibria/babel/har/ (reader, writer, symbols, wire), reimplementado tras el PR #44; el fix de los headers multi-record REFULL entro por el PR #46."
gh pr close 8 --comment "Cerrado por obsoleto. altertax vive hoy en src/equilibria/templates/gtap/altertax/, con tests en tests/templates/gtap/test_altertax.py y test_altertax_multiperiod_parity.py."
gh pr close 9 --comment "Cerrado por obsoleto. La paridad de altertax se cubre hoy en los gates de run_parity_gates.py; el emparejamiento MCP se resolvio en la rama f3.5."
gh pr close 14 --comment "Cerrado por obsoleto. El multiperiodo vive hoy en gtap_model_multiperiod.py y gtap_multiperiod_driver.py, con 9 tests en tests/templates/gtap/."
```

- [ ] **Paso 3: Verificar**

```bash
gh pr list --state open
```
Esperado: ninguno.

### Tarea 9: Podar ramas preservando el trabajo no publicado

39 de 41 ramas remotas están cerradas (14 mergeadas por fast-forward, 23 con PR MERGED por squash,
2 con PR CLOSED). Diez no tienen PR y algunas llevan trabajo real jamás publicado.

- [ ] **Paso 1: Etiquetar las 10 ramas sin PR antes de tocar nada**

```bash
for b in feat/pyomo-jax-translator gtap6-template-impl perf/poi-fase0 poi/fase0-fixes \
         gtap/f3-blocks-extraction pep-co2-template-v1 perf/mumps-symbolic-reuse \
         perf/scaffold-reuse gtap/v62-rollback worktree-runMacos; do
  git tag "archive/$b" "origin/$b"
done
git push origin --tags
```

- [ ] **Paso 2: Verificar que los tags existen en el remoto**

```bash
git ls-remote --tags origin | grep archive/ | wc -l
```
Esperado: 10.

- [ ] **Paso 3: Borrar las ramas cuyo PR está MERGED o CLOSED**

```bash
for b in gtap/welfare-shadow-rungtap-parity worktree-harwriter fix/har-writer-set-descriptor \
         debug-gtap7-warmstart booming-message pep-phase23 fdocs fix-docs-debts prek \
         gtap/close-15x10-pure-ifsub1-code2 gtap/validation-parity-pages gray-carver \
         fix/refull-multirecord-block-headers gtap/gempack-check-denominator-qga \
         gtap/qxs-qga-check-to-main gtap/sam-filtering gtap7-gempack-template \
         perf/cudss-linsolve doc/gempack-10x7-post-merge-stat doc/gempack-substantive-column \
         doc/gempack-what-not-compared perf/model-disk-cache fix/seed-cache-key-types \
         review-repo-steps perf/skip-validation-on-fix; do
  git push origin --delete "$b"
done
```

- [ ] **Paso 4: Borrar las ya mergeadas por fast-forward y las 10 archivadas**

```bash
git branch -r --merged origin/main | grep -v 'HEAD\|origin/main$' | sed 's|origin/||' \
  | tr -d ' ' | xargs -I{} git push origin --delete {}

for b in feat/pyomo-jax-translator gtap6-template-impl perf/poi-fase0 poi/fase0-fixes \
         gtap/f3-blocks-extraction pep-co2-template-v1 perf/mumps-symbolic-reuse \
         perf/scaffold-reuse gtap/v62-rollback worktree-runMacos \
         feat/gtap-har-native-loader feat/gtap-altertax fix/gtap-altertax-ev-pairing \
         gtap/v62-multiperiod; do
  git push origin --delete "$b" 2>/dev/null || true
done
```

- [ ] **Paso 5: Verificar el resultado**

```bash
git remote prune origin
git branch -r | grep -v HEAD
```
Esperado: sólo `origin/main`. El trabajo no publicado sigue accesible vía `archive/*`.

### Tarea 10: Consolidar los patrones de `runs/` en `.gitignore`

El `.gitignore` tiene 448 líneas, de las cuales **93 son patrones de `runs/`** enumerados por
extensión y por subdirectorio. Los 43 archivos versionados bajo `runs/`, `reports/` y `results/`
son evidencia deliberada de validación (el resultado 20x41 vs GEMPACK, el caso Horridge SIMPLE,
los comparadores) y **se conservan**.

**Archivos:**
- Modificar: `.gitignore:305-411`

- [ ] **Paso 1: Anotar qué está versionado hoy**

```bash
git ls-files runs reports results | tee /tmp/versionados.txt | wc -l
```
Esperado: 43. Esta lista es la referencia del paso 4.

- [ ] **Paso 2: Sustituir los 93 patrones por reglas genéricas**

Reemplazar el bloque `.gitignore:305-411` por:

```
# Salidas de corridas GEMPACK/GAMS: se ignora todo binario o de consola,
# y se versiona a mano la evidencia de validacion (.md, .cmf, .py, .json).
runs/**/*.har
runs/**/*.sl4
runs/**/*.slc
runs/**/*.sti
runs/**/*.sol
runs/**/*.log
runs/**/*.lst
runs/**/*.gdx
runs/**/*.xls
runs/**/*.prm
runs/**/*.UPD
runs/**/*.GEN
runs/**/*_console*.txt
runs/**/sl4dump*
runs/**/sltoht*
runs/**/sl4_levels*
runs/**/sl4map.txt
runs/**/*assert-arith-fail*
runs/**/wmiccap.tmpwmic
runs/**/__pycache__/
```

- [ ] **Paso 3: Conservar las excepciones deliberadas**

Justo después del bloque anterior, añadir:

```
# Evidencia de validacion que se versiona pese a las reglas de arriba
!runs/gempack_20x41_validation/*_console.txt
!runs/gempack_20x41_validation/tm10-assert-arith-fail.har
!runs/horridge_simple/simdata.har
!runs/horridge_simple/input.gdx
```

- [ ] **Paso 4: Verificar que no cambió lo versionado**

```bash
git ls-files runs reports results > /tmp/despues.txt
diff /tmp/versionados.txt /tmp/despues.txt && echo "OK: los 43 siguen versionados"
git status --porcelain runs reports results
```
Esperado: `OK` y sin archivos nuevos sin seguimiento. Si el `diff` no está vacío, ajustar las
excepciones del paso 3 hasta que lo esté.

- [ ] **Paso 5: Commit**

```bash
git add .gitignore
git commit -m "chore(gitignore): reglas genericas para runs/ en vez de 93 patrones

Se enumeraban extensiones subdirectorio por subdirectorio. La evidencia
de validacion versionada (43 archivos) se conserva igual."
```

---

## Fase 3 — Reconectar CI

### Tarea 11: Ejercitar la suite completa

CI corre 6 de 172 archivos de test (3,5 %). La lista blanca viene del commit `2f5c128`
(«stabilize tests for linux runner»): se enumeró lo que pasaba y el resto nunca se reconectó.
Todo `babel/` queda fuera — incluidos los lectores HAR/GDX donde vivía el bug REFULL.

**Archivos:**
- Modificar: `.github/workflows/tests.yml:52-60`

- [ ] **Paso 1: Medir localmente qué falla al correr todo**

```bash
uv run --frozen python -m pytest tests -q -m "not gams" 2>&1 | tail -25
```
Anotar cada fallo. Éste es el momento de descubrirlos: con la red desconectada, cualquiera pudo
entrar sin que nadie se enterara.

- [ ] **Paso 2: Reemplazar la lista blanca**

En `.github/workflows/tests.yml`, sustituir el paso `Run tests (excluding GAMS-marked)` y su lista
de 6 archivos por:

```yaml
      - name: Run tests (excluding GAMS-marked)
        run: |
          uv run pytest -m "not gams" tests
```

- [ ] **Paso 3: Tratar cada fallo del paso 1**

Para cada uno, elegir una de dos salidas —nunca borrar el test sin más:
- Si el test es correcto y el código está mal: arreglar el código.
- Si el test requiere algo que CI no tiene: marcarlo con un motivo explícito, por ejemplo
  `@pytest.mark.skipif(not shutil.which("gams"), reason="requiere GAMS local")`.

- [ ] **Paso 4: Verificar en verde**

```bash
uv run --frozen python -m pytest tests -q -m "not gams" 2>&1 | tail -5
```
Esperado: 0 fallos; los saltos, todos con motivo declarado.

- [ ] **Paso 5: Commit**

```bash
git add .github/workflows/tests.yml tests
git commit -m "ci: ejercitar la suite completa en vez de 6 archivos

CI corria 6 de 172 archivos (3,5%), herencia de 2f5c128 cuando se
estabilizo el runner de linux. Todo babel/ quedaba fuera, incluidos los
lectores HAR/GDX."
```

---

## Fase 4 — Borrado

### Tarea 12: Borrar las 6 ecuaciones inertes del monolito

Se declaran y se desactivan en la línea siguiente: nunca entran al `.nl`. Está documentado en
`blocks/gtap/__init__.py:35-37`. Autorizado explícitamente por el autor.

**Archivos:**
- Modificar: `src/equilibria/templates/gtap/gtap_model_equations.py` (líneas 5604, 5920, 5981,
  5996, 6013, 6026 y sus `.deactivate()` inmediatos)

- [ ] **Paso 1: Correr los gates ANTES de tocar nada**

```bash
uv run --frozen python scripts/gtap/run_parity_gates.py 2>&1 | tail -20
```
Esperado: los 5 en verde. Si alguno falla ya, **detenerse**: hay que resolverlo antes, o el
borrado quedará bajo sospecha.

- [ ] **Paso 2: Eliminar los 6 bloques declaración+desactivación**

Borrar, para cada una de `prf_y` (:5604-5605), `eq_ps` (:5920-5921), `eq_pe` (:5981-5982),
`eq_pe_route` (:5996-5999), `eq_xet_agg` (:6013-6014) y `eq_xe_xw` (:6026-6027), tanto la
declaración `Constraint(...)` como su `.deactivate()`. Trabajar de abajo arriba (de :6026 a :5604)
para que los números de línea no se desplacen.

- [ ] **Paso 3: Verificar que el modelo construye igual**

```bash
uv run --frozen python -m pytest tests/templates/gtap -q -m "not gams" 2>&1 | tail -5
```
Esperado: igual que la línea base.

- [ ] **Paso 4: Correr los gates DESPUÉS**

```bash
uv run --frozen python scripts/gtap/run_parity_gates.py 2>&1 | tail -20
```
Esperado: los 5 en verde, con los mismos porcentajes de coincidencia que el paso 1. **Cualquier
movimiento en un número es motivo de revertir**: la fidelidad manda sobre la limpieza.

- [ ] **Paso 5: Commit**

```bash
git add src/equilibria/templates/gtap/gtap_model_equations.py
git commit -m "refactor(gtap): borrar 6 ecuaciones desactivadas por construccion

prf_y, eq_ps, eq_pe, eq_pe_route, eq_xet_agg y eq_xe_xw se declaraban y
se desactivaban en la linea siguiente: nunca entraban al .nl. Los 5
gates de paridad dan los mismos numeros antes y despues."
```

### Tarea 13: Borrar los módulos huérfanos

**Archivos:**
- Eliminar: `src/equilibria/templates/gtap/gtap_model_full.py` (687 líneas),
  `scripts/gtap/test_full_gtap_realistic.py` (su único consumidor),
  `src/equilibria/babel/gdx/reader_new_decoder.py`,
  `src/equilibria/sam_tools/config_loader.py`,
  `src/equilibria/simulations/adapters/pep_co2.py`,
  `src/equilibria/templates/pep_co2_data.py`,
  `src/equilibria/templates/pep_co2_model_equations.py`,
  `src/equilibria/templates/pep_co2_model_solver.py`,
  `src/equilibria/templates/data/pep/generate_gdx.py`,
  `src/equilibria/templates/data/pep/generate_val_par.py`

- [ ] **Paso 1: Confirmar que ninguno tiene consumidores**

```bash
for m in gtap_model_full reader_new_decoder config_loader pep_co2 generate_gdx generate_val_par; do
  echo "--- $m ---"
  grep -rn "$m" --include="*.py" src tests scripts | grep -v "/$m.py:"
done
```
Esperado: para `gtap_model_full`, sólo `scripts/gtap/test_full_gtap_realistic.py` (que también se
borra); para `pep_co2`, sólo referencias entre ellos mismos. Cualquier otra coincidencia
**detiene** el borrado de ese módulo.

- [ ] **Paso 2: Borrar**

```bash
git rm src/equilibria/templates/gtap/gtap_model_full.py \
       scripts/gtap/test_full_gtap_realistic.py \
       src/equilibria/babel/gdx/reader_new_decoder.py \
       src/equilibria/sam_tools/config_loader.py \
       src/equilibria/simulations/adapters/pep_co2.py \
       src/equilibria/templates/pep_co2_data.py \
       src/equilibria/templates/pep_co2_model_equations.py \
       src/equilibria/templates/pep_co2_model_solver.py \
       src/equilibria/templates/data/pep/generate_gdx.py \
       src/equilibria/templates/data/pep/generate_val_par.py
```

- [ ] **Paso 3: Limpiar los `__init__` que los exporten**

```bash
grep -rn "pep_co2\|config_loader\|reader_new_decoder\|gtap_model_full" \
  --include="__init__.py" src
```
Eliminar cada import o entrada de `__all__` que aparezca.

- [ ] **Paso 4: Verificar recolección y suite**

```bash
uv run --frozen python -m pytest tests --collect-only -q 2>&1 | tail -3
uv run --frozen python -m pytest tests -q -m "not gams" 2>&1 | tail -5
```
Esperado: recolección sin errores de import; la suite igual que la línea base.

- [ ] **Paso 5: Commit**

```bash
git add -A
git commit -m "refactor: borrar modulos sin consumidores

gtap_model_full.py (687 lineas, abandonado desde abril), el decoder GDX
alternativo, config_loader, el adaptador PEP CO2 con sus 3 modulos y los
dos generadores de datos PEP. Ninguno tenia referencias en src ni tests."
```

### Tarea 14: Borrar las funciones sin referencias

**Archivos:**
- Modificar: los 18 archivos listados en el spec §3 fase 4

- [ ] **Paso 1: Re-verificar cada una antes de borrar**

```bash
for f in validate_against_csv CESAggregationEquation LeontiefEquation MarketClearingEquation \
         save_baseline_compatibility_report create_calibration_data compute_ces_shares \
         compute_armington_shares compute_les_parameters load_workflow_config \
         aggregate_table_with_mapping load_mip_raw_excel_table indices_for_selector \
         index_for_key compute_share_recalibration get_set_name map_index_names \
         build_sets_on _install_null_handler _get_fd_columns; do
  n=$(grep -rn "\b$f\b" --include="*.py" src tests scripts | wc -l | tr -d ' ')
  echo "$n	$f"
done
```
Sólo se borra lo que tenga recuento **1** (su propia definición). Todo lo demás se deja y se anota
como deuda: el informe pudo no ver un uso dinámico.

- [ ] **Paso 2: Borrar las confirmadas**

Eliminar cada función o clase junto con sus imports ya sin uso. Si al quitarla un módulo queda
vacío, borrar el módulo.

- [ ] **Paso 3: Pasar el linter**

```bash
uv run --frozen ruff check src --select F401,F811 --fix
uv run --frozen ruff format src
```

- [ ] **Paso 4: Verificar la suite**

```bash
uv run --frozen python -m pytest tests -q -m "not gams" 2>&1 | tail -5
```
Esperado: igual que la línea base.

- [ ] **Paso 5: Commit**

```bash
git add -A
git commit -m "refactor: borrar funciones publicas sin referencias

18 funciones y clases top-level sin uso en src, tests ni scripts, mas 2
privadas muertas dentro de su propio archivo."
```

---

## Fase 5 — Roadmap, documentación e historial

### Tarea 15: Escribir `ROADMAP.md` versionado

Hoy el roadmap vive fuera del repo, como symlinks a `dev-tools`: invisible para quien clone.

**Archivos:**
- Crear: `ROADMAP.md`

- [ ] **Paso 1: Redactar el roadmap con el estado medido**

Crear `ROADMAP.md` con estas secciones, todas con números verificables, no adjetivos:

1. **Qué es equilibria y qué está cerrado hoy** — validación GTAP vs GEMPACK (20x41 al 94,7 % en
   %-change y 98,5 % en niveles; 3x3 al 99,5 %), y la advertencia de que esas cifras exigen
   `base_calibrated=True`.
2. **Fases F0→F12** — cuáles están cerradas, cuál está en curso, cuáles pendientes.
3. **Deuda técnica** — la tabla del spec §3 fase 4, con `gtap_model_equations.py` y su primer
   corte (extraer `apply_production_scaling` y `_align_xi_xaa_post_scaling`), `eq_pmuv`, los
   bloques genéricos sin consumidores, los templates PEP zombis y latentes.
4. **Trabajo archivado** — los 10 tags `archive/*`, señalando POI Fase 0, el template gtap6
   (+17 commits, cierra 269 DOF) y el traductor pyomo-jax como lo más sustancioso.

- [ ] **Paso 2: Enlazarlo desde el README**

Añadir en la tabla de navegación del encabezado del `README.md`, junto a los enlaces existentes:

```markdown
[Roadmap](ROADMAP.md) •
```

- [ ] **Paso 3: Commit**

```bash
git add ROADMAP.md README.md
git commit -m "docs: ROADMAP.md versionado con el estado medido y la deuda

Vivia fuera del repo como symlinks a dev-tools: invisible al clonar."
```

### Tarea 16: Índice de `docs/`

99 markdown en 10 subdirectorios, sin punto de entrada.

**Archivos:**
- Crear: `docs/README.md`

- [ ] **Paso 1: Inventariar y detectar contradicciones**

```bash
find docs -name '*.md' | sort
```
Marcar los documentos que atribuyan a linearización gaps que resultaron ser cruces de closure o el
seed. **No borrar nada todavía:** presentar la lista al autor antes de tocarla (entregable 2 del
spec, el punto con más juicio involucrado).

- [ ] **Paso 2: Escribir el índice**

Crear `docs/README.md` con una sección por subdirectorio, explicando en una línea qué contiene y
si es material vigente o histórico: `architecture/` (cómo encaja el sistema — empezar por
`monolito_vs_bloques.md`), `guides/` (cómo usarlo), `findings/` (resultados de validación con
fecha), `analysis/`, `reference/`, `technical/`, `plans/`, `archive/` (histórico, puede contradecir
hallazgos posteriores), `site/` (fuente de la doc publicada).

- [ ] **Paso 3: Commit**

```bash
git add docs/README.md
git commit -m "docs: indice de docs/ con 99 documentos en 10 directorios"
```

### Tarea 17: Reescribir el historial

**Irreversible en el remoto.** Autorizado explícitamente por el autor. 487 de 786 commits llevan
el trailer `Co-Authored-By: Claude`; el autor es `mmc00` en los 786 y no cambia.

- [ ] **Paso 1: Respaldar antes de nada**

```bash
git branch backup/pre-filter-repo main
git tag backup/pre-filter-repo-$(date +%Y%m%d) main
git push origin backup/pre-filter-repo --tags
git rev-parse main | tee /tmp/main_sha_antes.txt
```

- [ ] **Paso 2: Confirmar que `git filter-repo` está disponible**

```bash
uv tool run git-filter-repo --version || pipx run git-filter-repo --version
```

- [ ] **Paso 3: Quitar el trailer**

```bash
uv tool run git-filter-repo --force --commit-callback '
import re
msg = commit.message.decode("utf-8", errors="replace")
msg = re.sub(r"\n*Co-[Aa]uthored-[Bb]y: Claude[^\n]*\n?", "\n", msg)
msg = re.sub(r"\n*🤖 Generated with \[Claude Code\][^\n]*\n?", "\n", msg)
commit.message = msg.rstrip().encode("utf-8") + b"\n"
'
```

- [ ] **Paso 4: Verificar antes de publicar**

```bash
git log --grep='Co-Authored-By: Claude' --oneline | wc -l   # esperado: 0
git rev-list --count HEAD                                    # esperado: 786
git log --format='%an' | sort -u                             # esperado: solo mmc00
uv run --frozen python -m pytest tests -q -m "not gams" 2>&1 | tail -5
```
Esperado: 0 commits con el trailer, los 786 intactos, un solo autor, y la suite igual que la línea
base. **Si el recuento de commits no es 786, detenerse**: `filter-repo` hizo algo no previsto y el
respaldo del paso 1 es la salida.

- [ ] **Paso 5: Publicar**

```bash
git remote add origin https://github.com/mmc00/equilibria.git  # filter-repo lo elimina por seguridad
git push --force origin main
```

- [ ] **Paso 6: Avisar del impacto**

Los clones existentes quedan invalidados y hay que rehacerlos (`git fetch && git reset --hard
origin/main`). Los SHAs citados en PRs mergeados ya no resuelven; el respaldo
`backup/pre-filter-repo` los preserva.

---

## Autorrevisión

**Cobertura del spec:**

| Requisito del spec | Tarea |
|---|---|
| §3 F0.1 línea base | 2 |
| §3 F0.2 `INPUT_TREES` | 1 |
| §3 F1.1 rutas funcionales en `src/` | 3 |
| §3 F1.2 scratchpads muertos | 4 |
| §3 F1.3 comentarios `Reference:` | 5 |
| §3 F1.4 symlink `superpowers` | 6 |
| §3 F1.5 citas a `CLAUDE.md` | 7 |
| §3 F2.1 cerrar PRs | 8 |
| §3 F2.2 podar ramas + tags | 9 |
| §3 F2.3 artefactos de corridas | 10 (ajustado: ya estaban ignorados; se consolidan los 93 patrones y se conservan los 43 versionados) |
| §3 F3 reconectar CI | 11 |
| §3 F4 las 6 ecuaciones inertes | 12 |
| §3 F4 módulos huérfanos | 13 |
| §3 F4 funciones sin referencias | 14 |
| §3 F5 historial | 17 |
| §4.1 ROADMAP versionado | 15 |
| §4.2 índice de docs | 16 |
| §4.3 mapa monolito↔bloques | 7 |

Sin huecos. La única desviación es la Tarea 10, documentada arriba: el spec proponía sacar
`runs/`, `reports/` y `results/` del control de versiones, pero la medición mostró que `runs/` ya
estaba ignorado por 93 patrones y que los 43 archivos versionados son evidencia deliberada de
validación. Se conservan; se consolidan los patrones.

**Placeholders:** ninguno. Cada paso lleva el comando o el código que hay que ejecutar.

**Consistencia de nombres:** `INPUT_TREES`, `_ensure_path_lib`, `_ensure_path_module`,
`PATH_CAPI_LIBPATH`, `EQUILIBRIA_PATH_CAPI_SRC`, `apply_production_scaling`,
`_align_xi_xaa_post_scaling`, `build_block_model` y los nombres de las 6 ecuaciones se usan igual
en el spec, en el plan y en `docs/architecture/monolito_vs_bloques.md`.
