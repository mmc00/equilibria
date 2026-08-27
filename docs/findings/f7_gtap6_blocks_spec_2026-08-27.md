# F7 — GTAP 6.2 Template on Symbolic Blocks — Design Spec

**Date:** 2026-08-27
**Status:** Design (approved for planning)
**Roadmap:** F7 (GTAP6.2), adelantado fuera de orden a pedido explícito del usuario —
normalmente post-F5 (GEMPACK)/F6 (release 1.0). Ver
`project_equilibria_roadmap_1.0` (dev-tools memory) y
[`f3_blocks_extraction_spec_2026-07-25.md`](f3_blocks_extraction_spec_2026-07-25.md)
(que ya reservaba F7 = GTAP6, post-F3/F5, reusando `blocks/gtap/`).

## Goal

Implementar un template GTAP versión 6.2 (Hertel/Itakura/McDougall 2003,
formato de datos GTAP Data Base pre-v9 / pre-make-matrix, documentado como
predecesor de v7 en van der Mensbrugghe TP/19/01 nota 4) sobre el framework de
**Block simbólicos** (`equilibria.blocks` / `Block` de pydantic) que F3 ya
probó y dejó en ≥99% de paridad vs GAMS para GTAP7. Éxito = `gtap6_3x3` →
`gtap6_15x10` resuelven vía bloques y matchean GEMPACK con gap ≤1%
(replicando el resultado ya alcanzado por un prototipo previo, nunca
mergeado).

## Contexto — qué existe ya

### Los datos
`datasets/gtap6_{3x3,3x4,5x5,10x7,15x10,20x41}/` ya existen (`basedata.har`,
`baserate.har`, `default.prm`, `sets.har`). Formato v6.2a confirmado leyendo
los HAR directamente:

- **Sets** (`sets.har`): headers cortos GEMPACK clásicos — `H1`=REG,
  `H2`=TRAD_COMM, `H6`=ENDW_COMM, `H9`=CGDS_COMM (alias de los nombres
  largos, no sets nuevos). `COMM==ACTS` siempre — no hay make-matrix, no hay
  split actividad/commodity. `ENDL=['UnSkLab','SkLab']` en ambos v6/v7 (el
  labor set no cambió).
- **Benchmark** (`basedata.har`): arrays **planos** — `VDFA/VDFM` (firm
  intermediate demand agent/market), `VIFA/VIFM` (ídem import), `VDPA/VDPM`,
  `VDGA/VDGM`, `VIPA/VIPM`, `VIGA/VIGM`, `VXMD`/`VXWD` (exports),
  `VIWS/VIMS` (imports bilateral), `EVFA/EVOA` (factores), `VKB`, `VST`,
  `VTWR`, `FBEP/FTRV/MFRV/TFRV/XTRV/ADRV/PTAX/PURV/CSEP/ISEP/DPSM/SAVE/POP/
  VDEP`. **Sin** `MAKB/MAKS/VMSB/VCIF/VFOB/VXSB/VRRV` (los de v7).
- Mobilidad de factores viene de `SLUG` en `default.prm` (binario
  mobile/sluggish), no de la matriz `ENDOWFLAG(e,t)` de v7.

### El framework de bloques (F3, GTAP7)
`src/equilibria/blocks/gtap/` — 7 unidades (`TradeCET`, `ProductionSupply`,
`Factor`, `ArmingtonBilateral`, `DemandUtility`, `Income`, `Closure`), cada
una subclase de `equilibria.blocks.base.Block`, compuestas por un composer
que las registra en `GTAP_BLOCK_ORDER` y las traduce a Pyomo vía el bridge
reparado (`backends/pyomo_backend.py`, ya no traga excepciones ni inyecta
`dummy_constraint`). Este framework hoy da ≥99% NLP-vs-NLP y MCP-vs-MCP
contra GAMS para GTAP7 en 3x3→15x10 (`docs/site/guide/gtap7_coverage_matrix.md`).
GTAP6 **reusará el contrato `Block`/composer/bridge**, pero con sus **propias**
subclases (los nests de v6.2 son estructuralmente distintos — ver abajo) bajo
`blocks/gtap6/`, no las 7 de v7.

### El prototipo previo (nunca mergeado)
Ramas huérfanas `gtap/v62-multiperiod` / `gtap/v62-rollback` (última
actividad 2026-06-10) contienen un template v6.2 **imperativo** completo:
`src/equilibria/templates/gtap_v62/` (~4400 líneas: sets, contract,
calibration, model_equations monolítico de 2055 líneas, solver), con 30+
docs de findings fase-por-fase (`docs/findings/gtap_v62_phase3XX_*.md`) y un
notation crosswalk GAMS-v7↔v6.2↔equilibria (`runs/gtap_v62_vs_v7/`).

**Resultado ya alcanzado (Phase 3.38, `docs/findings/gtap_v62_phase338_*.md`):**
gap vs GEMPACK Gragg-multi de **0.06–0.64%** en gtap6_3x3/5x5/10x7/15x10,
Walras < 2e-8. `gtap6_20x41` no converge (IPOPT/MUMPS toca el límite de
stack de enteros de 32-bit — un límite del solver/tamaño, no del modelo,
análogo al bug de 20x41 que F3 ya resolvió para GTAP7).

Es arquitectónicamente obsoleto (monolito Pyomo escrito antes de que
existiera `blocks/gtap/`), pero el trabajo de datos/calibración/hallazgos es
sólido y se reusa (ver Decisión 2).

## Design decisions (todas aprobadas 2026-08-27)

### Decisión 1 — Arquitectura: bloques nuevos, no monolito revivido
`blocks/gtap6/` con subclases `Block` propias, análogas en espíritu a
`blocks/gtap/` pero NO comparten instancias (v6.2 no tiene make-matrix, no
tiene bundle ND intermedio — Leontief implícito en la CES top-nest, no tiene
CET de output/commodity homogeneity, `cgds` es un sector productor no un
agente, y el tax stream es un agregado único no 10 streams separados). Se
descarta portar `gtap_v62_model_equations.py` tal cual como template
imperativo separado — quedaría inconsistente con la arquitectura F3 y no
compondría con el resto del repo (calibración multi-bloque, introspección).

Unidades propuestas (5, no 7 — v6.2 es más simple):
`TRADE_ARMINGTON` (leaf) → `PRODUCTION` → `FACTOR` → `DEMAND_UTILITY` (CDE) →
`INCOME_CLOSURE` (last).

### Decisión 2 — Reuso del prototipo: puerto directo de datos, reescritura de ecuaciones
- **Puerto directo** (adaptar imports/API, sin reescribir lógica):
  `gtap_v62_sets.py` → `gtap6_sets.py`, `gtap_v62_parameters.py` →
  `gtap6_parameters.py`, `gtap_v62_calibration.py` → `gtap6_calibration.py`,
  `gtap_v62_contract.py` → `gtap6_contract.py`. Estos ya resuelven H1/H2/H6/H9,
  SLUG, lectura HAR v6.2a plana, y calibración SAM-consistente — no dependen
  de arquitectura de bloques.
- **Reescritura guiada por findings**: `gtap_v62_model_equations.py` (2055
  líneas monolíticas) se descompone en las 5 `Block` subclasses de
  `blocks/gtap6/`, usando los 30+ docs de fase como mapa de qué NO repetir
  (ver "Bugs conocidos a evitar" abajo) y como oracle de forma (el monolito
  puede ejecutarse tal cual, sin mergear, como referencia de comparación
  transitoria — igual rol que jugó `gtap_model_equations.py` para F3).

### Decisión 3 — Gate de aceptación y alcance
Gate por dataset, orden F3-style, verde antes de avanzar al siguiente:

| Dataset | Gate |
|---|---|
| `gtap6_3x3` | NLP+MCP vs GEMPACK, gap ≤1%, Walras < 1e-6 |
| `gtap6_5x5` | ídem |
| `gtap6_10x7` | ídem |
| `gtap6_15x10` | ídem |
| `gtap6_20x41` | **fuera de alcance**, documentado como conocido-roto (límite solver, no modelo) |

El gap ≤1% replica (no relaja) el resultado ya demostrado por el prototipo
(0.06–0.64% medido). Si la reimplementación en bloques no alcanza ese mismo
piso, es señal de una regresión de fidelidad respecto al trabajo previo, no
un piso nuevo más laxo.

### Bugs conocidos a evitar (de los findings del prototipo)
1. **`sav` debe ser `Var`, no `Param`** — dejarlo constante rompe la
   identidad de presupuesto regional `y = yp + yg + sav` bajo shock y el
   residuo se lo traga `walras` (Phase 3.38).
2. **VIWS se mide `qxs * pmcif`** (precio CIF, mundial), no `qxs * pms`
   (precio agente, post-arancel) — un error de métrica de test, no de
   modelo, pero corrompía la comparación ~16pp (Phase 3.38).
3. **Shock aplicado como POWER, no rate**, al generar cualquier referencia
   GAMS local nueva (Phase encontrado en `gtap/v62-multiperiod` log,
   consistente con la convención GEMPACK/GAMS ya documentada para GTAP7).
4. Ver el resto de `docs/findings/gtap_v62_phase3XX_*.md` para hallazgos de
   escalado (Phase 3.34/3.36), diagonal trade (Phase 3.15/3.16), y closure
   condicional (Phase 3.28) antes de re-derivar desde cero.

## Architecture

```
src/equilibria/templates/gtap6/
  __init__.py
  gtap6_sets.py          # puerto de gtap_v62_sets.py
  gtap6_parameters.py    # puerto de gtap_v62_parameters.py (HAR v6.2a directo)
  gtap6_calibration.py   # puerto de gtap_v62_calibration.py
  gtap6_contract.py      # puerto de gtap_v62_contract.py (closure v6.2, numeraire pgdpwld)
  gtap6_block_model.py   # NUEVO — composer, análogo a templates/gtap/gtap_block_model.py
  gtap6_solver.py        # puerto de gtap_v62_solver.py

src/equilibria/blocks/gtap6/
  __init__.py             # GTAP6_BLOCK_ORDER + composer checklist (análogo a blocks/gtap/__init__.py)
  trade_armington.py       # Armington 2-nivel, sin MRIO, incluye margins Cobb-Douglas (leaf)
  production.py            # CES top nest, sin make-matrix, sin bundle ND separado
  factor.py                 # mercados a nivel commodity (sin tinc(e,a,r) de v7), mobile/sluggish via SLUG
  demand_utility.py         # CDE real (Hanoch-Hertel) + gov Cobb-Douglas + cgds como sector productor
  income_closure.py         # agregado único de tax streams (6: imptx/exptx/outtx/indtx/facttx/subdy) + closure NLP/MCP, sav como Var
```

Composer: mismo patrón que `blocks/gtap/__init__.py` — dedup de vars
compartidas, orden leaf→closure, checklist explícito de qué hace el
composer vs qué define cada bloque (snapshots post-scaling, shares
recompute, etc., análogo al checklist ya documentado para GTAP7).

El monolito `gtap_v62_model_equations.py` se trae desde la rama huérfana a
`scripts/gtap6/_v62_monolith_oracle.py` (prefijo `_` = no es API pública, no
se importa desde `templates/gtap6/` ni `blocks/gtap6/`; solo lo consumen los
tests de form-diff de la capa 1 del gate). Sirve de **oracle de forma
transitorio** durante la migración y se borra una vez todos los datasets
(3x3→15x10) pasen el gate final contra GEMPACK.

## Per-solve fidelity gate (4 capas, barato → caro; TODAS verdes para aceptar)
1. **Equation-form diff** — expresión Pyomo expandida de cada bloque ==
   la del monolito v6.2 oracle, celda por celda.
2. **Var domain+bounds diff** — cada var mantiene EXACTAMENTE su dominio/bounds.
3. **Canario (`gtap6_3x3`)** — resuelve primero; segundos; no hay full
   sweep si esto rompe.
4. **Full NLP+MCP sweep vs GEMPACK** (patrón `test_gtap7_gempack_parity.py`)
   — gate final por dataset.

## Testing
- `tests/blocks/gtap6/test_gtap6_blocks_form.py` — form-diff vs monolito oracle.
- `tests/blocks/gtap6/test_gtap6_blocks_domain.py` — domain/bounds diff.
- `tests/templates/gtap6/test_gtap6_blocks_solve.py` — canario 3x3 (NLP+MCP, code=1).
- `tests/templates/gtap6/test_gtap6_gempack_parity.py` — gate final, ≥99%
  (gap ≤1%) por dataset, patrón `test_gtap7_gempack_parity.py`.

## Non-goals / YAGNI
- No MRIO, no NTM AVE, no dynamics — v6.2 nunca tuvo esas extensiones
  (son adiciones de v7.1 documentadas en el paper van der Mensbrugghe).
- No unificar `blocks/gtap/` y `blocks/gtap6/` en una jerarquía compartida
  de clases — los nests difieren lo suficiente (make-matrix vs diagonal,
  CDE vs CD) que forzar herencia compartida sería prematuro; YAGNI hasta
  que un tercer template (GTAP-E, MyGTAP) muestre el patrón común real.
- No tocar `blocks/gtap/` (GTAP7) en absoluto — es un template hermano,
  no una dependencia.
- No perseguir `gtap6_20x41` en esta fase — documentado como límite de
  solver conocido, no bloquea el resto.
- No relajar el gate ≤1% aunque cueste más iteración — el prototipo ya
  demostró que es alcanzable (0.06–0.64%); relajarlo sería una regresión.

## Risks (y mitigaciones)
- **Reescribir 2055 líneas de ecuaciones a bloques puede introducir bugs
  nuevos** que el monolito v6.2 ya no tenía → mitigado por el oracle de
  forma (capa 1 del gate) comparando expresión por expresión antes de
  solvear.
- **El puerto directo de `gtap_v62_parameters.py` puede no anticipar algo
  que el composer de bloques necesita** (p.ej. snapshots post-scaling como
  los que `blocks/gtap/__init__.py` documenta para v7) → se descubre en el
  primer solve del canario 3x3; el checklist del composer se escribe
  ANTES de ese primer solve, como hizo F3.
- **Los findings de fase asumen el monolito imperativo** (p.ej. "sav como
  Var" se implementaba en una función `_add_equations` lineal) → el fix
  conceptual se porta (sav vive en `IncomeClosure` como `Var`, no
  `Param`), no el código imperativo literal.
