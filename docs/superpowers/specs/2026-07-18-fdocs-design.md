# F-docs — mejorar la documentación (diseño)

**Fecha:** 2026-07-18 · **Fase:** F-docs del roadmap equilibria 1.0
(`2026-06-24-equilibria-1.0-roadmap-design.md` §3.5) · **Rama:** `fdocs` (desde `origin/main` @ `dd68304`)
**Tipo:** docs-only. No toca `src/equilibria/` (cero riesgo de regresión del modelo).

---

## Objetivo

Ejecutar los quick-wins de §3.5 del roadmap más tres pedidos explícitos del usuario:

1. Las páginas de comparación (matrices de cobertura y benchmarks) pasan al **formato
   artifact** (cards, legend, celdas con chips `✓ code 1` y tints de match%, dark mode) —
   referencia visual: el artifact "GTAP7 NLP-vs-NLP Convergence Matrix", cuyo diseño ya
   vive en `docs/site/_static/gtap7_{nlp,mcp}_matrix.html`.
2. La página GTAP se organiza bajo dos headers — **NLP vs NLP** y **MCP vs MCP** — y lo
   mismo la de PEP. La tabla del `.nl` coefficient gate y las dos tablas SOLVE viejas
   (`altertax`/`gtap_solve` vs refs locales) **salen del doc** (sus tests no cambian).
3. El sitio deja de ser un único "User guide": sidebar de **6 secciones por función**.

## Decisiones tomadas (brainstorm 2026-07-18)

| Decisión | Elección |
|---|---|
| Estructura del sitio | 6 toctrees con caption en `index.md`: Start here / Templates / Data & solvers / Validation & parity / Examples / Reference |
| Tabla `.nl` en la página GTAP | Se elimina por completo; una frase en el intro menciona el canario de CI |
| Gates SOLVE viejos (PATH vs refs locales) | Se eliminan del doc; la sección MCP vs MCP queda solo con el fidelity gate (refs NEOS) |
| Formato artifact | Aplica a las 2 matrices de cobertura **y** a benchmarks |
| Implementación del formato | Generadores emiten MyST + bloques `{raw} html`; CSS único compartido; gate de sync intacto |
| CHANGELOG | Keep-a-Changelog, hitos curados (no historial exhaustivo) |

---

## 1. Estructura del sitio

`docs/site/index.md` pasa de 3 toctrees a 6, con captions:

```
Start here:           guide/installation, guide/architecture        ← architecture NUEVA
Templates:            guide/templates,                              ← NUEVA
                      guide/gtap_quickstart, guide/pep_quickstart,
                      guide/welfare_decomposition
Data & solvers:       guide/mip_to_sam, guide/har_io, guide/path_capi
Validation & parity:  guide/benchmarks, guide/gtap7_coverage_matrix,
                      guide/pep_coverage_matrix
Examples:             gallery/index
Reference:            api/index, changelog                          ← changelog NUEVA
```

- Las páginas existentes **no se mueven de archivo** — cero churn de URLs; solo cambia el
  toctree al que pertenecen.
- `guide/index.md` queda como página-mapa breve (la card del landing sigue apuntándole);
  enlaza las secciones con links normales, **no** toctree, para evitar warnings de
  documentos en múltiples toctrees. Como deja de pertenecer a un toctree, lleva
  `orphan: true` en el frontmatter — sin eso `sphinx-build -W` falla por
  "document isn't included in any toctree".
- Las cards del landing se actualizan si algún destino cambió de rol.

## 2. `guide/architecture.md` (nueva)

Página conceptual "cómo encaja el framework":

- El flujo: **SAM/HAR/GDX I/O → calibración → templates (gtap · pep_pyomo · simple_open)
  sobre `blocks/` → solvers (PATH MCP · IPOPT NLP) → validación (paridad vs GAMS,
  matrices de cobertura)**.
- Diagrama **SVG commiteado** (sin dependencias nuevas de build — no graphviz/mermaid).
- Mapa del paquete `src/equilibria/` (qué hace cada subpaquete).
- Filosofía de paridad: GAMS es la fuente de verdad; comparaciones mismo-engine
  (NLP-vs-NLP, MCP-vs-MCP) para que la tolerancia del solver se cancele.
- Absorbe lo útil de `docs/architecture/gams_parity_matrix.md` (hoy huérfano); ese
  archivo se deja donde está (fuera del sitio) o se referencia — no se borra en F-docs.

## 3. `guide/templates.md` (nueva)

Overview de templates con estado, una sección corta cada uno (qué es, módulos, cómo
correrlo, links a quickstart/matriz):

- **gtap** — GTAP Standard 7: multi-período, NLP+MCP, 6 datasets.
- **pep_pyomo** — port PEP-1-1 v2.1 a Pyomo: NLP 100% / MCP 100% vs GAMS.
- **simple_open** — modelo didáctico de economía abierta.
- **pep** legacy (cyipopt) — marcado como superseded por pep_pyomo.

No se inventan quickstarts nuevos; esta página es la referencia de existencia/estado.

## 4. Matrices en formato artifact

**Infra compartida:**

- `docs/site/_scripts/matrix_html.py` — helper que emite el markup del formato artifact
  (cards, legend, filas con celdas `valor + chip`). Importable por los dos generadores
  (via path insert) y por `render_benchmarks.py` (que ya corre en build desde `conf.py`).
- `docs/site/_static/matrix.css` — **un solo** CSS con el lenguaje visual del artifact:
  variables de color good/warn/bad/accent, tablecard, chips, legend, eyebrow; dark mode
  vía `@media (prefers-color-scheme: dark)` **y** el toggle de furo (`body[data-theme]`).
- Los `.md` generados embeben el markup con bloques MyST ` ```{raw} html `.

**`gtap7_coverage_matrix.md`** (regenerada por `scripts/gtap/gen_coverage_doc.py`):

- Dos headers: **NLP vs NLP** (IPOPT ambos lados, refs `ifMCP=0`) y **MCP vs MCP**
  (PATH ambos lados, refs NEOS limpias), cada uno con subsecciones *Pure-gtap (real-CES)*
  y *Altertax (CD)*.
- **Celdas = el floor que el gate pytest asserta** (`base ≥ / check ≥ / shock ≥`), con
  tint por floor + chip de contrato `code=1` + chip `ci/local/blocked`. El snapshot
  medido NO se copia a la matriz (principio del roadmap: sin copias muertas); los live
  views `_static/gtap7_{nlp,mcp}_matrix.html` quedan enlazados desde cada sección como
  la vista medida.
- Fuera del doc: tabla `.nl` (queda 1 frase en el intro sobre el canario de CI) y las dos
  tablas SOLVE viejas. `coverage_matrix.py` (los ROWS y los tests) **no cambia**; solo
  cambia qué secciones renderiza `gen_coverage_doc.py`.

**`pep_coverage_matrix.md`** (regenerada por `scripts/pep/gen_pep_coverage_doc.py`):

- Mismos dos headers **NLP vs NLP** / **MCP vs MCP**, más *NLP↔MCP mirror* y *objdef*
  como secciones menores. La matriz PEP sí guarda `match` medido → las celdas muestran
  match + chip.

**Benchmarks:** `docs/site/_scripts/render_benchmarks.py` restylado al mismo lenguaje
visual reutilizando `matrix.css` y el helper.

**Gates de sync:** `test_coverage_doc_in_sync` (gtap) y su equivalente PEP no cambian de
mecanismo (igualdad de string vs `render()`); se regeneran los `.md` y se commitean.

## 5. API reference expandida

`api/index.md` pasa a toctree con:

| Página | Contenido (`automodule`) |
|---|---|
| `core.md` | `equilibria.model`, `equilibria.datasets`, `baseline`, `simulations`, `solver`, `calibration`, `contracts` |
| `blocks.md` | `blocks.base` + `production` / `trade` / `demand` / `institutions` / `equilibrium` |
| `sam_tools.md` | existente, se conserva |
| `templates_gtap.md` | superficie pública: `gtap_contract`, `gtap_parameters`, `gtap_sets`, `gtap_solver`, `gtap_multiperiod_driver`, `shocks`, `welfare_decomp`. **Excluye** el monolito `gtap_model_equations` (6151 líneas — F3 lo partirá en blocks); nota explicándolo |
| `templates_pep_pyomo.md` | los 6 módulos `pep_pyomo_*` |

- Si un import opcional pesado rompe autodoc → `autodoc_mock_imports` en `conf.py`.
  **No se toca código del modelo**; un docstring roto se anota como deuda, no se arregla.
- El alcance exacto módulo-a-módulo puede ajustarse en el plan si un `automodule`
  resulta vacío o irrelevante (criterio: superficie pública útil, no exhaustividad).

## 6. `CHANGELOG.md` (raíz, nuevo)

- Formato **Keep-a-Changelog** + SemVer, hitos curados.
- `[Unreleased]` + `0.5.1` y los hitos previos agrupados por versión/fecha (del git log):
  paridad GTAP7 NLP+MCP+multi-período, port PEP→Pyomo, HAR I/O, PATH C API, matrices de
  cobertura, sam_tools/MIP→SAM, etc.
- `pyproject.toml` ya apunta a `blob/main/CHANGELOG.md` — no se toca.
- El sitio lo expone en `docs/site/changelog.md` vía `{include}` del archivo raíz
  (una sola fuente).

## 7. Revisión de quickstarts

- `gtap_quickstart` y `pep_quickstart`: verificar comando por comando contra los CLIs
  actuales (`scripts/gtap/run_gtap.py`, flujo pep_pyomo). Corregir **solo texto**.
- Si algo requiere tocar código → se anota como deuda en la sección "Deudas anotadas"
  del PR y NO se toca (regla F-docs).

## 8. Gate y CI

- Job nuevo **`docs-build`** en `.github/workflows/tests.yml`: instala el extra `docs` y
  corre `sphinx-build -W` (warnings = error) sobre `docs/site`. Es el gate del roadmap
  ("Sphinx buildea limpio, corre en CI").
- RTD (`.readthedocs.yaml`) queda como está (`fail_on_warning: false`) — el enforcement
  vive en CI, el deploy no se arriesga.
- Canario `.nl`: `uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py` debe seguir
  verde (no se toca modelo).

## Criterio de terminado

1. `sphinx-build -W` limpio (local y en el job `docs-build`).
2. `test_coverage_doc_in_sync` (gtap y pep) verdes con los `.md` regenerados.
3. Canario `.nl` 5/5 verde.
4. PR docs-only a `main` (docs + generadores de docs + `tests.yml`; nada de
   `src/equilibria/`).

## Fuera de alcance

- Tocar ecuaciones, parámetros o solver del modelo (cualquier hallazgo → deuda anotada).
- Restylar los live views `_static/gtap7_{nlp,mcp}_matrix.html` (ya están en formato
  artifact; se regeneran con solver, fuera de F-docs).
- Cambiar el esquema de `coverage_matrix.py` / `pep_coverage_matrix.py` (eso es F0).
- Borrar/migrar el árbol `docs/` legacy (guides/, findings/, plans/…) — F-docs solo
  cablea el sitio; una limpieza general de `docs/` es trabajo aparte.
