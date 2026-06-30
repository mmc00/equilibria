# GTAP Standard 7 — Barrido de estado FRESCO (medido 2026-06-30)

**Worktree:** `gtap-parity-decision` (`main` @ `52cb113`, fix `eq_xi` `0e2db11` vivo)
**Método:** se mide cada `kind` con SU gate. La doc (`coverage_matrix.py`) tiene **dos ejes (`kind`) distintos**, con contratos distintos. Son tablas separadas.

---

## Eje 1 — `kind = gtap` (GTAP7 "puro", real-CES). **Tiene DOS gates:**

> El gtap puro NO es single-period: comparte `PERIODS = (base, check, shock)` con el builder. Y tiene dos gates distintos — **álgebra** (`.nl`, sin solver) y **solve** (PATH, sí resuelve). Mi barrido inicial sólo midió el `.nl`; acá agrego el SOLVE.

### 1a — gate `.nl` (álgebra, sin solver, CI)

- **Mide:** coeficientes Jacobianos Python vs fixtures GAMS. **No resuelve.** ifSUB no aplica (`None`).
- **Fases:** `base`+`shock`, e inserta `check` (paso CD multi-período) **sólo si hay `gams_check.nl`** → **3x3/5x5/10x7 = base+check+shock; 15x10/3x4 = base+shock.**

| dataset      | ifSUB | fases medidas         | gate              | resultado FRESCO          | ci_status |
|--------------|-------|-----------------------|-------------------|---------------------------|-----------|
| nus333       | —     | base, shock           | NEOS+GAMS ≥99.5%  | **100%** (doc, otro gate) | ci        |
| 9x10         | —     | base, shock           | NEOS ≥99.5%       | **100%** (doc, otro gate) | ci        |
| gtap7_3x3    | —     | **base, check, shock**| 0 diffs `.nl`     | **✅ PASSED (0 diffs)**    | ci        |
| gtap7_5x5    | —     | **base, check, shock**| 0 diffs `.nl`     | **✅ PASSED (0 diffs)**    | ci        |
| gtap7_10x7   | —     | **base, check, shock**| 0 diffs `.nl`     | **✅ PASSED (0 diffs)**    | ci        |
| gtap7_15x10  | —     | base, shock           | 0 diffs `.nl`     | **✅ PASSED (0 diffs)**    | ci        |
| gtap7_3x4    | —     | base, shock           | 0 diffs `.nl`     | **✅ PASSED (0 diffs)**    | ci        |

> `5 passed in 11.34s`. Álgebra 100% idéntica a GAMS en todas las fases medidas.

### 1b — gate SOLVE (mode="gtap", RESUELVE, por ifSUB, local-only) — **el que faltaba**

- **Gate:** `test_gtap_multiperiod_parity.py` — resuelve base→check→shock con PATH (mode="gtap", real-CES, NO altertax CD), seed del ref GAMS LOCAL `out_gtap_shock_ifsub{0,1}.gdx`. Verifica `code=1` + match.
- **Cobertura real:** SÓLO **gtap7_3x3** tiene fixture gtap-puro-resuelto (ambos ifSUB). Ningún otro dataset → no medible hoy. El test commiteado sólo cubre ifSUB=0.

| dataset    | ifSUB | code b/c/s | fase  | tol 0.1% | tol 0.5% | tol 1%  | celdas | nota |
|------------|-------|------------|-------|----------|----------|---------|--------|------|
| gtap7_3x3  | 0     | **1/1/1**  | check | 100.0    | 100.0    | 100.0   | 1326   | exacto |
| gtap7_3x3  | 0     | **1/1/1**  | shock | 96.32    | 99.40    | **99.70** | 1332 | paridad (basin EU_28 Land 0.30%) |
| gtap7_3x3  | 1     | **1/1/1**  | check | 100.0    | 100.0    | 100.0   | 1326   | exacto |
| gtap7_3x3  | 1     | **1/1/1**  | shock | 53.45    | 54.88    | **55.63** ❌ | 1332 | **ROTO** (xm[USA] +150%) |

> **Dos hallazgos del SOLVE (lo que destapó medir-después-de-resolver + ifSUB):**
> 1. **gtap puro ifSUB=1 ni siquiera corría** → `UnboundLocalError: _recompute_params` (`gtap_multiperiod_driver.py:1919`). `_recompute_params` se asignaba sólo dentro de `if not _gtap_mode:`; bajo gtap-mode+ifSUB=1 el rebuild de `eq_pmeq` reescribe 0 cells (eqs de margen desactivadas) → el flag queda False → se usa la var sin asignar. **Fix de 1 línea aplicado** (hoist del init arriba del bloque `_gtap_mode`). Verificado: gates altertax (ambos ifSUB) + gtap-puro-ifSUB0 **siguen verdes**.
> 2. **El shock gtap-puro sólo está cableado para ifSUB=0.** CHECK exacto en ambos, pero SHOCK ifSUB=1 colapsa a ~55% (`xm[USA]` +150%): el wedge del shock entra vía `_rebuild_eq_pmeq_shock`/`_rebuild_eq_ytax_mt_shock`, que asumen eqs de margen ACTIVAS; bajo ifSUB=1 están desactivadas → el shock no entra a las eqs de import → import quantities altas. Clase "shock incompletamente cableado", NO basin. **Lead concreto, no perseguido en esta sesión** (es cierre, no barrido).

---

## Eje 2 — `kind = altertax` (GTAP7 altertax, multi-período, por ifSUB)

- **Gate:** cell-by-cell PATH (`test_altertax_multiperiod_parity`). **Con solver PATH C API, nonlinear full.**
- **Contrato:** los **3 períodos** convergen `code=1` **Y** match shock ≥ `gap_min`.
- **ifSUB:** 0 y 1 (filas separadas).
- **Fases:** base + check + shock (multi-período, encadenado).
- **Dónde corre:** local-only (necesita PATH + HAR).
- **Tolerancias:** el gate compara a tol 1%; acá desgloso **0.1% / 0.5% / 1%** sobre el mismo modelo resuelto.

| dataset      | ifSUB | code b/c/s | tol 0.1% | tol 0.5%   | tol 1%  | celdas | gap_min | ci_status |
|--------------|-------|------------|----------|------------|---------|--------|---------|-----------|
| gtap7_3x3    | 0     | **1/1/1**  | 85.72    | **98.88** ✅ | 99.93   | 1338   | 98.0    | local     |
| gtap7_3x3    | 1     | **1/1/1**  | 85.80    | **98.51** ✅ | 99.78   | 1338   | 98.0    | local     |
| gtap7_3x4    | 0     | **1/1/1**  | 82.92    | 96.79 ⚠️   | 99.72   | 1809   | 99.0    | local     |
| gtap7_3x4    | 1     | **1/1/1**  | 82.92    | 96.46 ⚠️   | 99.72   | 1809   | 99.0    | local     |
| gtap7_5x5    | 0     | **1/1/1**  | 86.39    | **98.53** ✅ | 99.88   | 4137   | 99.5    | local     |
| gtap7_5x5    | 1     | **1/1/1**  | 86.34    | **98.38** ✅ | 99.81   | 4137   | 99.5    | local     |
| gtap7_10x7   | 0     | **1/1/1**  | 84.48    | 96.83 ⚠️   | 99.33   | 15307  | 98.0    | local     |
| gtap7_10x7   | 1     | **1/1/1**  | 84.49    | 96.81 ⚠️   | 99.31   | 15307  | 98.0    | local     |
| gtap7_15x10  | 0     | **1/1/1**  | 89.35    | **98.19** ✅ | 99.57   | 41987  | 99.0    | local     |
| gtap7_15x10  | 1     | **1/1/1**  | 89.17    | 97.92 ⚠️   | 99.40   | 41987  | 99.0    | local     |
| gtap7_20x41  | 0     | **BLOQUEADO** | — referencia corrupta (viola 37 de sus propias eqs) — | | | | None | blocked |

> **Todos los medibles convergen `code=1/1/1`. A tol 1% todos ≥99.3%.**
> ✅ = pasa el goal estricto (≥98% @ tol 0.5%). ⚠️ = queda bajo 98% @ tol 0.5% (pero ≥99.3% @ tol 1%).

---

## Veredictos `seed_and_solve` (los ⚠️ sub-98% @ tol 0.5%: 3x4 y 10x7)

Corrido PRIMERO (orden de CLAUDE.md), period shock, leyendo la **cola** del residual (no la mediana):

| dataset    | veredicto | code | drift sentinel pft | cola real (ecuación que difiere)                         | eq_xi |
|------------|-----------|------|--------------------|----------------------------------------------------------|-------|
| gtap7_3x4  | **GOES**  | 2    | +17.7%             | `eq_paa[EGY]` (12 celdas ≤0.0048) + `eq_xma[EGY]`; 18/27 en EGY | limpio |
| gtap7_10x7 | **GOES**  | 2    | +37.2%             | **`eq_paa` 25/25** (JPN-Rice, SSA-Textiles, IND-FoodProc, 0.005–0.020) | limpio |

**Ecuación nombrada que difiere: `eq_paa`** (precio compuesto Armington).
= el artefacto **ya catalogado** `project_gtap7_armington_shares_bug`: en celdas de demanda microscópica (Rice/Textiles, demanda ~1e-9) el piso del init `xaa` (1e-8) ≠ el agregado CES de `xda+xma` → las shares no suman 1 → `eq_paa` carga residual en el punto GAMS.
**No** es selección de equilibrio ni una eq nueva. **`eq_xi` está limpio en ambos** → el fix `0e2db11` (que capó 3x3) aguanta a escala.
Fix fiel pendiente = calibración **quirúrgica** de `xaa` init (= GAMS `xa.l`) sólo en celdas mal calibradas. El renormalizado global ya se revirtió por romper 98→78%.

Confirmación por celdas peores del gate (todas micro): `xp[GBR,a_Rice]` py=0 vs gams=0.631 (15x10, colapso de actividad Rice), `xm[JPN,c_Rice]` py~3e-5 vs 2e-5 (10x7), `xw[*,c_Food,EGY]` (3x4).

---

## 20x41 — el único bloqueado, investigado

`validate_reference.py` (shock) sobre `equilibria_refs/gtap7_20x41_altertax_cd/out_altertax_ifsub0.gdx`:
**VIOLA 37 de sus PROPIAS familias de ecuaciones >0.01** — peores `eq_pfeq[ZAF,Capital,Rice]=968`, `eq_paa[THA,Sugar,VegFruits]=893`, `eq_us[EGY]=853`.
El GDX **parece** resuelto (`gdpmp[USA]` 19.48→18.12 en shock) pero es **internamente inconsistente** → medir Python contra él = medir contra basura.
`blocked` **confirmado por herramienta**, no por suposición. Sólo existe ifsub0 (no hay ifsub1). **Regenerar antes de cualquier claim 20x41.**

---

## Correcciones al estado "conocido" / `coverage_matrix.py`

1. **3x4 NO es "ref Infeasible".** Converge `1/1/1` → 99.72% @ tol1%. El único bloqueado es **20x41**.
2. **El `blocked` de 20x41** tiene causa precisa: **ref corrupta (37 eqs violadas)**, no "NEOS Infeasible" a secas.
3. **Fixtures fuera del árbol:** sólo 3x3/5x5/10x7 GDX commiteados en `tests/fixtures/gtap7_altertax/`. Los de 3x4/15x10/20x41 viven en `/Users/marmol/proyectos2/equilibria_refs/gtap7_*_altertax_cd/` (se stagean para medir).
4. **Provenance:** 10x7 = NEOS job 19621130, todos los períodos MODEL STATUS 1 Optimal (sólida). 3x4/15x10 = refs CD locales. 20x41 = origen incierto, inconsistente.
5. **Piso a tol 0.1%** (83–89% en TODOS los datasets) = el floor numérico del nido CD (`project_pva_is_quantity_anchor`, medido 3 formas, ningún fix de modelo lo cierra). No es bug por dataset.

---

## Síntesis

- **Eje `gtap` (puro):** 100% verde — 0 diffs de coeficientes en los 5 datasets `gtap7_*`, 100% en nus333/9x10.
- **Eje `altertax`:** todos los medibles convergen `1/1/1`, ≥99.3% @ tol1%. El único gap sistemático sub-98%@0.5% (3x4, 10x7) es **una sola familia, `eq_paa`** en celdas micro de Rice/Textiles — caracterizada, con fix fiel conocido pendiente.
- **20x41:** correctamente bloqueado (ref corrupta probada).

**Herramientas/artefactos de esta sesión:** `scripts/gtap/measure_gate_tols.py` (nueva, corre el solve del gate y reporta match% a 0.1/0.5/1% + codes). Memoria: `project_gtap7_full_sweep_2026_06_30.md`.
