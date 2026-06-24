# equilibria 1.0 — Roadmap a una versión estable en PyPI

**Fecha:** 2026-06-24
**Estado:** diseño aprobado, pendiente de plan de implementación de F0
**Tipo:** documento paraguas (umbrella). Cada fase F1–F8 genera su propio spec→plan a su turno.
**Memoria relacionada:** `project_equilibria_roadmap_1.0` (devtools)

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

## 4. Fases del roadmap

Cada fase = llenar/verificar un bloque de celdas de la matriz. Orden por dependencia.
**Invariante global:** el gate `.nl` single-period (canario, hoy 5/5) debe pasar después de CADA
fase (regla de `CLAUDE.md`).

| Fase | Nombre | Qué llena / entrega | Gate |
|------|--------|---------------------|------|
| **F0** ✅ | Esquema de la matriz *(habilitador)* | Extender `Row` (model/variant/period/solver/gempack), migración no destructiva, validación, doc con vistas + % global, tabla de modularización | ✅ `test_coverage_doc_in_sync` + canario 5/5 |
| **F1** | Multi-período GTAP7 (MCP) | Ejecutar el plan `2026-06-20-gtap-multiperiodo-plan.md` (eje `t` nativo, Fisher como filas, driver loop(tsim)). Llena `gtap7 × {core,altertax} × multi × mcp × ifSUB × 6 ds` | **Gate de decisión 3x3: code=1** + canario |
| **F2** | Cierre de gaps a 99% (MCP) | Subir cada celda `partial` a ≥99% vs GAMS vía la cascada de 8–10 tools (EU_28 basin, Armington 15x10, etc.). 20x41 sigue `blocked` hasta ref sana | celdas ≥99% + canario |
| **F3** | Modularización a bloques | Refactor de `gtap_model_equations.py` a `blocks/` (production CES anidada, Armington, CDE, instituciones/impuestos, cierre/Fisher). Llena la tabla de modularización | **0-diff `.nl` vs monolítico por extracción** + canario |
| **F4** | Solver NLP (ifMCP=0) | Formulación NLP paralela + refs GAMS ifMCP=0 nuevas. Duplica celdas resueltas en `solver="nlp"`, cada una validada vs su ref ifMCP=0 | **Gate de decisión propio** (refs nuevas, riesgo alto) + canario |
| **F5** | Co-validación GEMPACK/RunGTAP | Conseguir/generar outputs RunGTAP; llenar `gap_gempack` en celdas alcanzables. Criterio co-igual: ≥99% vs GEMPACK además de vs GAMS | `gap_gempack` ≥99% donde haya ref |
| **F6** | Empaquetado y release 1.0 | API pública estable, `pyproject` (extras `[mcp]`/`[nlp]`/`[gtap]`), docs RTD, ejemplos, CHANGELOG, publicación a PyPI, tag `v1.0.0` | build limpio + smoke install |
| **F7** *(post-1.0)* | GTAP6 | Mismos ejes que GTAP7, reusando toda la infra. Llena las celdas `model="gtap6"` | mismos gates por celda |
| **F8** *(post-1.0)* | MyGTAP / GTAP-Energy | Prueba de la modularidad: cambiar/agregar un bloque sin tocar el resto | composición de bloques funciona |

**Decisión de orden:** F3 (modularización) **antes** de F4 (NLP) — el refactor a bloques facilita
formular la variante NLP sobre bloques limpios en vez de sobre el monolítico.

Cada fase F1–F8 será su **propio spec→plan** cuando llegue su turno. Este documento es el paraguas;
**F1 ya tiene plan listo** (`2026-06-20-gtap-multiperiodo-plan.md`).

---

## 5. Manejo del eje NLP (F4) — aclaración

Hoy el repo es **MCP-only** (PATH). `CLAUDE.md` afirma explícitamente que *"Python no soporta NLP
en modo nonlinear full"* y que *ifMCP=0/NLP cambiaría los valores de referencia*. Por eso NLP es
**trabajo nuevo no trivial**, no un flag:

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

## 10. Próximo paso

Generar el **plan de implementación de F0** (extensión del esquema de la matriz) vía la skill
writing-plans. F0 es el primer trabajo ejecutable y habilita reportar el estado de todas las fases
siguientes. F1 ya tiene su plan (`2026-06-20-gtap-multiperiodo-plan.md`).
