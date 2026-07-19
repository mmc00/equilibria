# equilibria 1.0 — Roadmap a una versión estable en PyPI

**Fecha:** 2026-06-24 · **Reconciliado con el estado real:** 2026-07-19
**Estado:** documento vivo. Buena parte de F0–F5 ya avanzó desde el diseño original (ver §0).
**Tipo:** documento paraguas (umbrella). Cada fase F1–F8 genera su propio spec→plan a su turno.
**Memoria relacionada:** `project_equilibria_roadmap_1.0` (devtools)

---

## 0. Estado real (reconciliación 2026-07-19)

El roadmap se diseñó el 2026-06-24. Desde entonces el repo avanzó ~130 commits (GTAP7
paridad NLP+MCP, el port PEP→Pyomo, la matriz de cobertura con gates por etapa). Esta
sección reconcilia el plan con lo que EXISTE hoy en `main` — la tabla de fases (§4) marca
cada una con este estado. **La verdad de detalle sigue viviendo en la matriz de cobertura**
(`scripts/gtap/coverage_matrix.py`); esto es solo el resumen ejecutivo por fase.

| Fase | Estado real | Evidencia en `main` |
|------|-------------|---------------------|
| **F-docs** *(nuevo primer step)* | ⬜ **NO empezado** | 11 guide pages en `docs/site/guide/` pero sin overview/arquitectura, sin doc de `blocks/`, API autogen mínima (`docs/site/api/` = index + sam_tools). Ver §3.5 |
| **F0** — esquema de la matriz | 🟡 **PARCIAL** | `Row` tiene `kind`/`ifsub`/`phases` + gates NLP y MCP + **matriz PEP nueva** (`scripts/pep/pep_coverage_matrix.py`). FALTA: ejes explícitos `model`/`period`/`solver` separados y columnas GEMPACK |
| **F1** — multi-período GTAP7 (MCP) | 🟢 **HECHO** | `gtap_multiperiod_driver.py` (3365 líneas) + `gtap_model_multiperiod.py`; tests `test_*_multiperiod_*`; 6 datasets en `altertax`/`gtap_solve`/`mcp`/`nlp` (3x3…20x41, 20x41 blocked por ref) |
| **F2** — gaps a 99% (MCP) | 🟢 **CASI** | solo `gtap7_15x10` sigue en 94–96% (eq_paa Armington micro-cell, conocido) + `20x41` blocked por ref insana; resto ≥99% |
| **F3** — modularización a bloques | 🟡 **PARCIAL** | `src/equilibria/blocks/` existe y POBLADO (production/trade/demand/institutions/equilibrium, ~4000 líneas reales) — pero es el framework GENÉRICO; el monolito GTAP `gtap_model_equations.py` (6151 líneas) sigue **intacto**, la extracción 0-diff de GTAP a bloques NO empezó |
| **F4** — solver NLP (ifMCP=0) | 🟢 **HECHO** | `EQUILIBRIA_GTAP_SOLVE_NLP` cableado; 14 filas `kind="nlp"` en la matriz; gate NLP-vs-NLP (`test_gtap7_nlp_parity.py`) + vista viva `gtap7_nlp_matrix.html` |
| **F5** — co-validación GEMPACK/RunGTAP | 🟡 **EMPEZADO** | `scripts/gtap/compare_{nus333,9x10}_rungtap.py` existen; falta llenar `gap_gempack` en la matriz y cubrir el resto de datasets |
| **F6** — empaquetado PyPI 1.0 | ⬜ **NO empezado** | `pyproject.toml` v0.5.1 con extras, pero sin CHANGELOG, sin API pública congelada, sin publicación |

**Lectura:** el corazón técnico (multi-período, NLP+MCP, paridad) está mayormente hecho; lo
que falta para 1.0 es **terminar** (F0 ejes, F3 extracción completa, F5 GEMPACK) y **pulir
para release** (documentación, empaquetado). Por eso el nuevo primer step es documentación.

---

## 1. Visión y corte de release

Llevar `equilibria` de **v0.5.1** a un **1.0 estable en PyPI** centrado en **GTAP7 completo y
modular**. GTAP6 y las extensiones (MyGTAP / GTAP-Energy) son fases posteriores (1.x / 2.0) del
mismo roadmap; reutilizan toda la infraestructura construida para 1.0.

### Definición de "1.0" (el corte)

GTAP7, cumpliendo TODO lo siguiente sobre la matriz de cobertura:

- variantes **core + altertax**
- períodos **single-period + multi-período**
- etapas **base / check / shock**
- por **ifSUB 0 y 1**
- por **solver MCP (PATH) y NLP (walras / ifMCP=0)**
- sobre los **6 datasets** (3x3, 3x4, 5x5, 10x7, 15x10, 20x41)
- **convergencia `code=1` y gap ≥ 99%** en toda celda alcanzable
- **co-validado contra GAMS y GEMPACK/RunGTAP** (criterio co-igual, ver §6)
- **modularización completa a bloques** (`src/equilibria/blocks/`), con el gate `.nl` garantizando
  **0-diff vs el monolítico actual** — esto es lo que habilita MyGTAP/GTAP-Energy a futuro

### Fuera de 1.0 (post-release)

- **GTAP6** completo (mismos ejes; los datasets `gtap6_*` ya existen en disco).
- **MyGTAP / GTAP-Energy** como prueba de que la modularidad funciona (cambiar un bloque).

---

## 2. Principio rector: la matriz de cobertura es el single source of truth

El roadmap NO inventa una estructura nueva: **extiende la matriz que ya existe**
(`scripts/gtap/coverage_matrix.py` → `docs/site/guide/gtap7_coverage_matrix.md`, con
`test_coverage_doc_in_sync` en CI). Esa matriz pasa a ser el **tablero de control de todo el
proyecto**: es a la vez el plan (qué falta), el reporte (qué se logró) y el gate (CI verifica que
el doc renderizado coincide con la fuente).

**No hay estado del proyecto fuera de la matriz.** El progreso se lee siempre de ella, nunca se
copia a prosa en dos lados.

### Ejes de cada celda

| Eje | Valores | Estado hoy |
|-----|---------|-----------|
| `model` | gtap7 *(1.0)*, gtap6 *(1.x)* | **falta** (todo asume gtap7) |
| `variant` | core, altertax | existe como `kind` |
| `period` | single, multi | implícito en `kind` hoy; pasa a eje propio |
| `solver` | mcp (PATH), nlp (walras/ifMCP=0) | **falta** (mcp implícito) |
| `ifsub` | 0, 1, None | existe |
| `phases` | base, check, shock | existe |
| `dataset` | 3x3, 3x4, 5x5, 10x7, 15x10, 20x41 | existe |

### Estado medible por celda

- convergencia: `code=1` / `code=2` / no-converge
- `gap_min` / `gap_note`: floor + snapshot **vs GAMS** (ref correspondiente: MCP→ifMCP=1,
  NLP→ifMCP=0)
- `gap_gempack` / `note_gempack`: floor + snapshot **vs GEMPACK/RunGTAP** *(nuevo)*
- `ci_status`: `ci` / `local` / `blocked`

### Celdas N/A explícitas

Combinaciones que no aplican (p. ej. una ref GEMPACK inexistente para cierto dataset) se marcan
`— N/A` **con motivo**, nunca se omiten en silencio. Consistente con la regla del proyecto
*"GAMS is source of truth / nunca inflar match excluyendo celdas"*.

---

## 3. Extensión del esquema de la matriz (trabajo de F0)

Extender el dataclass `Row` en `coverage_matrix.py`:

```python
Row(
    model:        "gtap7" | "gtap6"        # NUEVO
    variant:      "core" | "altertax"       # renombra "kind"
    period:       "single" | "multi"        # NUEVO (hoy implícito en kind)
    solver:       "mcp" | "nlp"             # NUEVO
    ifsub:        0 | 1 | None
    phases:       tuple[str, ...]           # base / check / shock
    gap_min:      float | None              # floor vs GAMS
    gap_note:     str                       # snapshot vs GAMS
    gap_gempack:  float | None              # NUEVO: floor vs RunGTAP
    note_gempack: str                       # NUEVO: snapshot vs RunGTAP
    ci_status:    "ci" | "local" | "blocked"
    ref:          str                       # ref GAMS (MCP→ifMCP=1, NLP→ifMCP=0)
    ref_gempack:  str | None                # NUEVO: provenance RunGTAP
)
```

**Migración no destructiva.** Las filas actuales se mapean automáticamente:

- `kind="gtap"`     → `variant="core",     period="single", solver="mcp"`
- `kind="altertax"` → `variant="altertax", period="multi",  solver="mcp"`

La validación import-time (`_validate()`) se extiende con los invariantes nuevos
(p. ej. `solver in {"mcp","nlp"}`, `model in {"gtap6","gtap7"}`,
`gap_gempack is None or 0 < gap_gempack < 100`). Los consumidores actuales
(`nl_rows()`, `altertax_rows()`, los dos tests, `gen_coverage_doc.py`) siguen funcionando o se
adaptan en la misma fase.

El doc generado pasa a tener **vistas por sección** (model × solver) más una **tabla de progreso
global** (% de celdas `done` / `partial` / `blocked` / `N/A`).

### Tabla de modularización (eje ortogonal, misma fuente)

El estado del refactor a bloques NO encaja en la matriz de paridad (es ortogonal a
dataset/solver). Vive como una **segunda tabla declarativa** en la misma fuente: una lista de
bloques objetivo, cada uno con estado.

| bloque (objetivo) | estado |
|---|---|
| `production_ces_nest` | extraído / testeado / 0-diff vs monolítico |
| `armington_trade` | … |
| `cde_demand` | … |
| `institutions_tax` | … |
| `closure_fisher` | … |

---

## 3.5. F-docs — mejorar la documentación *(nuevo PRIMER step, 2026-07-19)*

Con el corazón técnico mayormente hecho (F1/F4 done, F2 casi), el mayor déficit para un 1.0
presentable es la **documentación**: alguien que llega al repo no tiene un mapa conceptual, ni
referencia de API más allá de `sam_tools`, ni páginas para varios templates ya funcionando.
Antes de terminar F0/F3/F5 y empaquetar, se pule la documentación — es barato, no toca el
modelo (cero riesgo de regresión), y hace el resto del roadmap legible para revisores.

**Entrega (quick-wins concretos, ordenados por valor):**

1. **Overview / arquitectura en el sitio.** Existe `docs/architecture/` pero NO está cableado a
   `docs/site/`. Traer (o escribir) una página conceptual "cómo encaja el framework"
   (SAM tools → calibración → templates → solver → validación) y meterla en el toctree del
   guide. Es lo primero que un lector nuevo necesita.
2. **Expandir el API autodoc.** Hoy `docs/site/api/index.md` solo lista `sam_tools`. Autodoc ya
   está configurado (`sphinx.ext.autodoc` en `conf.py`); agregar páginas `automodule` para el
   core, `blocks/`, y los templates (gtap, pep_pyomo) — la mayor parte de la superficie pública
   no tiene referencia generada.
3. **Páginas de guía faltantes por template.** `pep_pyomo` (el port PEP→Pyomo, 6 módulos) solo
   se menciona dentro de `pep_quickstart`; `simple_open` no tiene página. Darles una página cada
   uno (o al menos una "templates overview" que los liste con su estado).
4. **`CHANGELOG.md`.** El `pyproject.toml` ya apunta a `blob/main/CHANGELOG.md` pero el archivo
   **no existe** — crearlo (Keep-a-Changelog), sembrado con el historial de 0.x hasta hoy. Es un
   prerequisito real de F6 (release).
5. **Revisar los quickstarts existentes** (gtap/pep) contra el estado actual del código — el
   `pep_quickstart` ya se actualizó con el flujo Pyomo NLP/MCP; verificar que `gtap_quickstart`
   siga vigente.

**Gate:** el sitio Sphinx **buildea limpio** (0 warnings de toctree/autodoc rotos) + el canario
`.nl` 5/5 sigue verde (no debería tocarse, es solo docs). No requiere solver ni GAMS → **corre
en CI**. Genera su propio spec→plan vía writing-plans cuando se ejecute.

---

## 4. Fases del roadmap

Cada fase = llenar/verificar un bloque de celdas de la matriz. Orden por dependencia.
**Invariante global:** el gate `.nl` single-period (canario, hoy 5/5) debe pasar después de CADA
fase (regla de `CLAUDE.md`).

| Fase | Nombre | Qué llena / entrega | Gate |
|------|--------|---------------------|------|
| **F-docs** ⬜ **← PRIMER STEP** | Mejorar la documentación | Overview/arquitectura en el sitio, expandir API autodoc más allá de sam_tools, páginas por template (pep_pyomo/simple_open), `CHANGELOG.md`, revisar quickstarts (ver §3.5) | **Sphinx buildea limpio** + canario 5/5 (CI, sin solver) |
| **F0** 🟡 PARCIAL | Esquema de la matriz *(habilitador)* | Extender `Row` (model/variant/period/solver/gempack), migración no destructiva, validación, doc con vistas + % global, tabla de modularización. **HECHO:** gates NLP+MCP, mode, stage_floors, matriz PEP. **FALTA:** ejes model/period/solver explícitos + columnas GEMPACK | ✅ `test_coverage_doc_in_sync` + canario 5/5 |
| **F1** 🟢 HECHO | Multi-período GTAP7 (MCP) | Plan `2026-06-20-gtap-multiperiodo-plan.md` (eje `t` nativo, Fisher como filas, driver loop(tsim)). Llena `gtap7 × {core,altertax} × multi × mcp × ifSUB × 6 ds` | ✅ code=1 3x3 + canario |
| **F2** 🟢 CASI | Cierre de gaps a 99% (MCP) | Subir cada celda `partial` a ≥99% vs GAMS vía la cascada de 8–10 tools. **Resta:** 15x10 en 94–96% (eq_paa Armington); 20x41 `blocked` hasta ref sana | celdas ≥99% + canario |
| **F3** 🟡 PARCIAL | Modularización a bloques | Refactor de `gtap_model_equations.py` (6151 líneas, monolito intacto) a `blocks/` (production CES anidada, Armington, CDE, instituciones/impuestos, cierre/Fisher). El framework `blocks/` genérico ya existe; falta la extracción GTAP | **0-diff `.nl` vs monolítico por extracción** + canario |
| **F4** 🟢 HECHO | Solver NLP (ifMCP=0) | Formulación NLP paralela + refs GAMS ifMCP=0. `EQUILIBRIA_GTAP_SOLVE_NLP` cableado; 14 filas `nlp` en la matriz; gate `test_gtap7_nlp_parity.py` | ✅ gate propio + canario |
| **F5** 🟡 PARCIAL | Co-validación GEMPACK/RunGTAP | Llenar `gap_gempack` en la matriz. **HECHO:** compare RunGTAP EV/welfare para nus333 y 9x10. **FALTA:** las columnas gempack en `Row` + cubrir el resto de datasets | `gap_gempack` ≥99% donde haya ref |
| **F6** ⬜ | Empaquetado y release 1.0 | API pública estable, `pyproject` (extras `[mcp]`/`[nlp]`/`[gtap]` — hoy faltan), docs RTD, ejemplos, CHANGELOG (F-docs lo crea), publicación a PyPI, tag `v1.0.0` | build limpio + smoke install |
| **F7** *(post-1.0)* | GTAP6 | Mismos ejes que GTAP7, reusando toda la infra. Llena las celdas `model="gtap6"` | mismos gates por celda |
| **F8** *(post-1.0)* | MyGTAP / GTAP-Energy | Prueba de la modularidad: cambiar/agregar un bloque sin tocar el resto | composición de bloques funciona |

**Orden recomendado desde hoy (2026-07-19):** **F-docs** (barato, cero riesgo, hace el resto
legible) → **F0-resto** (ejes model/period/solver + columnas GEMPACK, habilita reportar F5/F7)
→ **F3** (extracción GTAP a bloques, el trabajo pesado restante) → **F5** (llenar gempack) →
**F2-resto** (los 15x10 sub-99) → **F6** (release). F1 y F4 ya están hechos.

Cada fase será su **propio spec→plan** cuando llegue su turno. Este documento es el paraguas.

---

## 5. Manejo del eje NLP (F4) — aclaración *(HECHO 2026-07; se conserva como registro de diseño)*

> **Estado:** F4 ya está implementado — el eje NLP existe en la matriz (`kind="nlp"`, gate
> `test_gtap7_nlp_parity.py`, `EQUILIBRIA_GTAP_SOLVE_NLP`). Lo de abajo era la aclaración del
> diseño original (cuando el repo era MCP-only); se conserva para contexto.

Cuando se escribió el diseño el repo era **MCP-only** (PATH), y NLP era **trabajo nuevo no
trivial**, no un flag:

- NLP = el modo `solve using nlp` de GAMS (numéraire walras, `ifMCP=0`) como **solver-mode paralelo**
  en Python.
- Tiene su **propia referencia GAMS ifMCP=0** (distinta de la MCP). Cada celda de la matriz existe en
  variante `solver="mcp"` (vs ref ifMCP=1) y `solver="nlp"` (vs ref ifMCP=0).
- Requiere: (a) la formulación NLP, (b) generar/conseguir las refs ifMCP=0 por dataset. De ahí el
  **gate de decisión propio** de F4 antes de invertir en todos los datasets.

---

## 6. Criterio GEMPACK/RunGTAP (F5) — co-igual a GAMS

El gap del 99% debe cumplirse **contra GEMPACK/RunGTAP tanto como contra GAMS**. GAMS sigue siendo
el source of truth para depuración (la cascada de tools compara vs GAMS), pero RunGTAP es un gate
medible co-igual en la matriz (`gap_gempack`).

- Sub-tarea de datos: **conseguir/generar outputs RunGTAP** para los datasets (es trabajo en sí).
- Donde no exista ref RunGTAP para un dataset, la celda `gap_gempack` se marca `— N/A` con motivo
  (no se omite).

---

## 7. Ubicación, sincronización y mantenimiento

**El roadmap vive en el repo (fuente):**

- `docs/superpowers/specs/2026-06-24-equilibria-1.0-roadmap-design.md` — este documento paraguas
  (visión, ejes, fases, criterios). Estático: describe el plan, NO el estado celda-a-celda.
- La **matriz extendida** (`coverage_matrix.py` → `gtap7_coverage_matrix.md`) es el **tablero vivo
  de estado**. El roadmap apunta a ella; no duplica el estado en prosa.

**En devtools (espejo para agentes):**

- Symlink de este roadmap a `~/proyectos/dev-tools/equilibria-tools/plans/` (mismo patrón que los
  dos archivos multiperiodo ya symlinkeados al repo).
- Memoria `project_equilibria_roadmap_1.0.md` en `equilibria-tools/memory/` con el hook (corte 1.0,
  fases F0–F8, dónde vive la matriz), enlazada desde `MEMORY.md`. Así cualquier sesión futura sabe
  que el roadmap existe y dónde mirar el estado.

**Regla anti-divergencia:**

- **Única fuente de estado = la matriz.** CI ya verifica que el doc renderizado coincide
  (`test_coverage_doc_in_sync`).
- Cada fase, al cerrarse, actualiza filas de la matriz (gap, ci_status) y marca la fase en este
  roadmap. La memoria devtools se actualiza solo en **hitos** (cierre de fase), no celda a celda.
- El roadmap-design NO se renderiza en el sitio RTD: copia en `docs/superpowers/specs/` + symlink a
  devtools alcanza (decisión del usuario).

---

## 8. Riesgos

| # | Riesgo | Mitigación |
|---|--------|-----------|
| R1 | Triplicar el MCP (multi-período) rompe/ralentiza PATH | **Gate de decisión 3x3 en F1**: si no da code=1, enfoque A refutado, se para antes de escalar |
| R2 | El refactor a bloques (F3) diverge del modelo validado | **0-diff `.nl` vs monolítico por cada extracción**; canario 5/5 siempre |
| R3 | NLP requiere refs nuevas que pueden no converger (F4) | **Gate de decisión propio en F4**; NLP está después del corte mínimo, no bloquea releases tempranos |
| R4 | Refs RunGTAP inexistentes para algún dataset (F5) | Celda `gap_gempack = N/A` con motivo; no se omite ni se infla |
| R5 | 20x41 ref NEOS Infeasible (ya conocido) | Permanece `blocked` en la matriz hasta conseguir ref sana; no bloquea 1.0 si las demás celdas cumplen |
| R6 | Divergencia entre roadmap (repo) y memoria (devtools) | Estado SOLO en la matriz; roadmap y memoria son estáticos, se tocan en hitos |

---

## 9. Criterio de "Terminado" (1.0)

- Toda celda alcanzable de `gtap7 × {core,altertax} × {single,multi} × {mcp,nlp} × {ifSUB 0,1} ×
  6 datasets` en `code=1` y **≥99% vs GAMS y vs GEMPACK** (o `N/A`/`blocked` con motivo honesto).
- `gtap_model_equations.py` refactorizado a `blocks/`, **0-diff `.nl`** confirmado.
- Canario `.nl` single-period 5/5 verde.
- Paquete publicado a PyPI, `v1.0.0` tag, docs RTD al día, matriz renderizada en sync (CI verde).

---

## 10. Próximo paso *(actualizado 2026-07-19)*

Generar el **plan de implementación de F-docs** (mejorar la documentación, §3.5) vía la skill
writing-plans. Es el primer step desde hoy: barato, cero riesgo de regresión (no toca el modelo),
corre en CI, y hace legible el resto del roadmap para revisores. Luego F0-resto (ejes
model/period/solver + columnas GEMPACK) → F3 (extracción GTAP a bloques). F1 y F4 ya están hechos.
