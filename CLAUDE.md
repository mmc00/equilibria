# equilibria — GTAP Standard 7 GAMS/NEOS Parity (CLOSED)

## Estado del branch

**Paridad lograda (base + shock 10%):**

| Dataset | Python vs NEOS | Python vs GAMS local |
|---------|----------------|----------------------|
| NUS333 (3×2) | 100% / 100% | 100% / 100% |
| 9x10        | 100% / 100% | bloqueado por licencia community |
| **gtap7_3x3 (altertax CD shock)** | — | **96.80% (tol 0.1%) / 100% (tol 1%), code=1** |

**gtap7_3x3 shock cerrado (2026-06-17): 78.69% → 96.80%/100%, fiel a GAMS, sin hardcodeo.** El cap 78.69% eran TRES errores que se tapaban entre sí: (1) `eq_pvaeq` bajo CD era la tautología `exp(log(pva))==pva` → pva libre (FALTABA la ecuación, no era una distinta); (2) `eq_pwfact` en forma cuadrática (raíz espuria) compensaba parcialmente el error de pva; (3) el warm-start del shock usaba un seeder incompleto. Fix fiel: eq_pvaeq → identidad de valor `pva·va = Σ_f(pfa·xf)` (verificada =GAMS exacto, diff=0); eq_pwfact → forma sqrt de GAMS; seeder completo en [3/3]. Commits `caf16a0`, `08e7c60`. Las herramientas de comparación (tools 3/4/6) NO lo veían porque una tautología es IGUAL en ambos modelos (no hay diferencia que comparar); lo destaparon el drift test (tool 7, síntoma) y `diff_tautology.py` (tool 9, causa — perturbar la var y ver si su ecuación reacciona).

Detalles por sesión en `GTAP_VALIDATION_STATUS.md`. Plan/diagnóstico en `plan_gtap7_3x3_shock_close.md`. Trabajo previo del branch: PR #3, commit `28a9b93` en `main`.

## Objetivo original

Lograr **paridad exacta** entre el template Python `equilibria` GTAP Standard 7 (9×10 y 3×2 NUS333) y la referencia GAMS/NEOS para:

1. **Baseline** (pre-shock)
2. **Shock uniforme de 10% en aranceles de importación** (`tm.fx = tm.l * 1.1`)

**Criterio:** signo y magnitud de los deltas endógenos coinciden con GAMS.

## Reglas de trabajo (no negociables)

- **El modelo altertax ES MULTI-PERÍODO y NO se resuelve de un solo tiro (hecho, no opinión).** `diff_altertax.py` resuelve en 3 etapas SECUENCIALES: `[1/3]` betaCal/base → `[2/3]` check (sin shock) → `[3/3]` shock (+10% imptx), cada etapa warm-started de la anterior y encadenada vía `t0_snapshot` (igual que el `loop(tsim)` base→check→shock de GAMS). `gtap_parameters.py` reconoce el eje temporal (`time_order={'base':1,'check':2,'shock':3}`). Lo ÚNICO single-period es la representación INTERNA de cada slice (las Vars son `pva[r,a]`, sin índice `t`) — eso NO significa "el modelo es single-period" ni "resuelve de un golpe". NO confundir las dos cosas. La diferencia real con GAMS: GAMS CONGELA el período previo (`var.fx(tsim-1)` + `holdfixed=1`); Python lo pasa como `t0_snapshot` (lectura para Fisher) pero NO lo congela → el DOF libre (pva) se desliza. Fix fiel probado (2026-06-17): congelar el base (pva=1.0) durante el check → pva=0.8536 EXACTO GAMS, check 63%→97%. Ver `feedback_gtap_IS_multiperiod` y tool 8 `diff_holdfixed.py`.
- **Solver:** siempre usar PATH C API en modo **nonlinear full** (10,296 ecuaciones). Nunca el bloque linear (1,370). IPOPT no aplica (degrees-of-freedom).
- **ifMCP (GAMS) = 1:** el run de referencia NEOS (job 18737509) usa `ifMCP=1` → `solve using mcp` → PATH. Python también usa PATH. Ambos están alineados — NO cambiar a `ifMCP=0` (NLP/walras) porque cambiaría los valores de referencia y Python no soporta NLP en modo nonlinear full.
- **equation_scaling=True:** siempre pasar `equation_scaling=True` a `_run_path_capi_nonlinear_full` (baseline Y shocked). Sin esto el baseline queda en code=2/res~1e-6 en vez de code=1/res~1e-9. Tanto `validate-shock` como `_run_homotopy_shocked` lo usan — `validate_gams_parity.py` también debe usarlo.
- **Modo de shock:** usar siempre `--shock-mode tm_pct` para shocks tipo GAMS (multiplica la *power* del arancel: `imptx_new = (1+imptx_old)*(1+v) − 1`). El modo `pct` viejo escala sólo la tasa y produce shocks ~10× menores en bienes con arancel bajo.
- **Región residual:** `NAmerica` (coincide con `rres` de GAMS). NO usar `RestofWorld` aunque exista como 10ª región real en el dataset 9x10.
- **GAMS license expirada (Oct 2024):** sólo `gdxdump` CLI funciona para leer GDX (no `gdxpds`, no correr modelos GAMS).
- **Dataset:** `basedata-9x10.gdx` ≡ `9x10Dat.gdx` (verificados idénticos).

## Gate de regresión (obligatorio antes de cualquier PR)

Antes de hacer merge de cualquier cambio que toque ecuaciones, parámetros, o carga de datos GTAP, correr:

```bash
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -v
```

Este test construye el `.nl` de Python para cada dataset `gtap7_*` y compara coeficientes familia-por-familia contra los fixtures GAMS commiteados en `tests/fixtures/gtap7/`. No requiere solver ni NEOS — corre en <1s por dataset.

- **0 diffs = OK.** Cualquier diff indica regresión en ecuaciones o parámetros.
- **Para agregar un nuevo dataset:** generar fixtures con `nl_compare.py --dataset <name> --phase base shock --out-dir /tmp/nl_<name>`, copiar los `gams_*.nl` a `tests/fixtures/gtap7/<name>/`, commitear.
- **`gtap7_20x41`** solo se corre localmente (fixture ~159MB, no está en git).
- El CI corre este gate automáticamente en cada push/PR (`gtap7-nl-parity` job en `.github/workflows/tests.yml`).

## Disciplina de git (obligatoria)

Cada corrida de validación debe tener un commit limpio asociado. Flujo obligatorio:

1. **Antes de correr:** hacer commit de todos los cambios en curso con mensaje descriptivo del experimento.
2. **Al reportar resultado:** incluir el SHA del commit en el reporte (`git rev-parse --short HEAD`).
3. **Si el resultado es FAIL:** hacer commit igualmente (con nota del resultado en el mensaje) para preservar el estado exacto del experimento.
4. **Nunca correr `validate_gams_parity.py` con working directory sucio** — así cada run es reproducible.

Formato de mensaje de commit para experimentos:
```
gtap: <descripción del cambio> → <resultado esperado o motivo>

EastAsia regy_delta: <valor si conocido>
Residual: <valor si conocido>
```

## Estado actual (ver `GTAP_VALIDATION_STATUS.md`)

| Item | Estado |
|------|--------|
| Region residual fix (`NAmerica`) | ✅ aplicado |
| Shock formula fix (`tm_pct`) | ✅ aplicado |
| `pdp/pmp` postsim recalc para alpha=0 | ✅ aplicado |
| CDE/chiInv elasticities frozen como Param | ✅ aplicado |
| `pmuv` Tornqvist Var+eq con `pefob0=(1+exptx)` | ✅ aplicado |
| `pwmg=0` donde `tmarg=0` (NEOS bundle) | ✅ aplicado |
| `ytax(r,gy)` con 10 streams canónicos | ✅ PR #3 |
| Paridad 9x10 (base + shock vs NEOS) | ✅ 100% / 100% |
| Paridad NUS333 (base + shock vs NEOS) | ✅ 100% / 100% |
| Paridad NUS333 (base + shock vs GAMS local) | ✅ 100% / 100% |
| Paridad 9x10 vs GAMS local | ⛔ bloqueado: licencia GAMS community (2500 rows) |

> **Matriz de cobertura (fuente única):** ver [`docs/site/guide/gtap7_coverage_matrix.md`](docs/site/guide/gtap7_coverage_matrix.md), generada de `scripts/gtap/coverage_matrix.py`. NO editar a mano (CI `test_coverage_doc_in_sync` lo verifica).

## Herramientas de debug parity (cascade de 11)

> **ORDEN DE USO (lo más importante — el error de la saga gtap7_3x3 fue el ORDEN, no las herramientas).** Cuando el gate cell-by-cell (`test_altertax_multiperiod_parity`) no llega al objetivo, la secuencia es:
> 1. **`seed_and_solve.py` PRIMERO (tool 11).** Siembra el punto GAMS exacto, resuelve, y responde la pregunta de raíz en UNA corrida: **STAYS** (resid~0, ninguna eq real con residual → es SELECCIÓN DE EQUILIBRIO, no hay ecuación mala, **parar**) vs **GOES** (drifta/falla → una ecuación DIFIERE; seguir). Es el discriminador definitivo; la saga lo usó **octavo** y eso costó dos cierres falsos.
> 2. **Leer la COLA del residual** que `seed_and_solve` reporta (las peores N ecuaciones en el punto GAMS), **NO la mediana** — con ~1100 ecuaciones la mediana es ~1e-13 (parece fixed-point limpio) mientras un puñado de >1e-4 lleva TODA la señal. La eq real (no-leaf, no-ROW) del tope es la candidata.
> 3. **`convert_gams.py` (CONVERT)** sobre esa ecuación: emite GAMS como Pyomo canónico (niveles, `scaleopt=0`) y se compara la ecuación candidata escrita vs equilibria → confirma si difiere DE VERDAD o es equivalente (dos formas pueden diferir en escritura y dar lo mismo). CONVERT **confirma**, no descubre.
> 4. **El banco** (`bench.py` / `diff_altertax` contra el gate cell-by-cell) confirma que el fix sube el gate sin romper.
>
> El **orquestador** localiza la CAPA (barre todo, no se le escapa una, pero deja varios sospechosos); **seed_and_solve** discrimina la CAUSA (selección vs ecuación-difiere, en un sí/no); **CONVERT** la confirma nombrable. Los tres, en ese orden. Solos, ninguno cierra.
>
> **Las cuatro lecciones de método (cada una costó un cierre falso en gtap7_3x3 — no repetir):**
> 1. **Un punto que el modelo EMPUJA no es un equilibrio alternativo — es una ecuación que difiere.** Si el punto GAMS fuera fixed-point (mismas ecuaciones → misma solución), el solve se quedaría. Que lo empuje ⇒ hay una ecuación nombrable. NO etiquetar "selección de equilibrio irreducible" mientras el punto GAMS no sea fixed-point probado.
> 2. **Leer la COLA del residual, no la mediana.** La mediana ~1e-13 esconde el puñado de >1e-4 que ES la señal. Mirar la mediana produjo dos "fixed-point a 1e-10, selección" falsos.
> 3. **`seed_and_solve` es el discriminador y va PRIMERO.** Drifta si y solo si una ecuación difiere. Octavo en la saga = ocho leads y dos cierres falsos antes de la respuesta que daba en una corrida.
> 4. **Disciplinas de setup del residual-en-el-punto-GAMS** (sin ellas se fabrican falsos leads): (a) **cascadear el seed de variables DERIVADAS** (xd/xc/xg/xi/xiagg/xigbl/pigbl/kapEnd no están en el GDX → quedan en init → residual espurio 1-2 que cae a ~0 al sembrarlas de su identidad; una eq que difiere NO cae); (b) **construir el modelo shock con el shock de impuesto APLICADO ANTES del build** (imptx·1.10 antes de `build_model`, no sembrar valores shock sobre un modelo base → eso deja `model.imptx` en base y fabrica un 0.014 falso en eq_pmeq). Ambas ya horneadas en `seed_and_solve.py`.

Cada herramienta ve una capa distinta. Nunca concluir de una sola herramienta — un sesgo de calibración bajo tolerancia se ve idéntico a "ruido de basin CD" hasta que la herramienta 4 lo aísla; y una diferencia de forma de ecuación que coincide numéricamente en el punto sembrado pasa el .nl pero la atrapa la herramienta 5. **Las tools 3-6 comparan Python vs GAMS asumiendo que el gap es una DIFERENCIA entre ellos; son ciegas a un gap que NO es diferencia: una variable que es un DOF libre/degenerado IDÉNTICAMENTE en ambos (tautología CD) — eso solo lo atrapa la tool 7 (drift test).**

> **Salida JSON pura (estandarización 2026-06-27):** seis tools de la cascada — `probe.py`, `drift_test.py`, `nl_compare.py`, `diff_mcp_pairing.py`, `validate_reference.py`, `diff_calibration.py` — emiten a **stdout únicamente un objeto JSON** (sin modo texto, sin flag `--json`). Esquema común `{tool, dataset, period(base/check/shock/null), status(clean|dirty|error), headline, violations[], meta{}}`; `violations` ordenadas por `|value|` desc, cada una con `metric` propio (`residual`/`drift_rel`/`jac_coef_diff`/`mcp_pairing`/`calib_rel`). `status` es la señal binaria del orquestador (Fase 2): `clean`=esta capa no explica el gap, `dirty`=sí, `error`=no pudo correr. Exit codes `clean→0/dirty→1/error→2`; errores también son JSON (`meta.error_kind` discrimina `no_convergence`/`no_common_constraints`/`gdx_not_found`/`exception`). Todo log/solver/progreso va a **stderr** vía captura total. El contrato vive en `scripts/gtap/_parity_json.py` (`run_tool`/`emit`/`make_violation`/`stdout_to_stderr`) — **mismo módulo en las seis, no copias**. Los flags de ENTRADA de cada tool quedaron intactos; solo cambió la salida. Formateadores humanos viejos conservados como `_debug_print` a stderr.

| # | Herramienta | Script | Ve | Ruta de uso |
|---|-------------|--------|----|-------------|
| 0 | **check-warmstart** | `triage.py --check-warmstart` | Residuales de ecuaciones + gaps de variables antes del solver | Solver converge a equilibrio equivocado |
| 1 | **Value/residual diff** | `triage.py` (locate→isolate→trace) | Dónde divergen valores resueltos | Variable específica diverge |
| 2 | **Closure diff** | `diff_closure.py` | Variables fijas/libres, ecuaciones activas | Shock diverge ampliamente (nivel de precios entero) |
| 3 | **.nl diff** | `nl_compare.py` | Coeficientes Jacobiano, forma algebraica | Pregunta sobre álgebra/coeficientes |
| 4 | **Calibration diff** | `diff_calibration.py` | Insumos de calibración en el benchmark vs GAMS: betaP/betaG/betaS, factY/yTaxInd/ytaxTot/phi/phiP, cada stream de `ytax` (pt/fc/pc/gc/ic/dt/mt/et), CDE (eh/bh/xcshr/zcons), bloque factor (gf/aft/xscale), por región/celda a tol 1e-4 | Una var de ingreso/impuesto (yc, ytax, regY) diverge ~1-2% y parece "ruido de basin", O validate_reference culpa a la referencia por eq_yc/eq_yg/eq_rsav |
| 5 | **Equation-form diff** | `diff_equation_form.py` | Forma simbólica expandida de una ecuación Python (coeficientes sustituidos, ej. 1/xscale → 10.0/100.0) lado a lado con la definición GAMS del `.gms` | Inputs (tool 4) y coeficientes (.nl tool 3) coinciden pero una familia sigue divergiendo → diferencia estructural de forma (1/xscale de más, factor extra, dominio de suma) que el .nl no muestra |
| 6 | **MCP-pairing diff** | `diff_mcp_pairing.py` | Emparejamiento ecuación↔variable: parsea `model gtap /eq.var, .../` de GAMS (eq.var = emparejado, eq solo = FREE-ROW) y lo contrasta con qué ecuaciones Python desactiva (`--apply-closure`). Numéricamente invisible a tools 3-5 | El solve converge (code=1) pero a OTRA raíz de un bloque multivaluado (ej. pft≈3.6 vs 1.0). Causa raíz del factor-2: GAMS deja `pfteq` free-row, Python la resolvía para pft → raíz espuria |
| 7 | **Drift test (free-DOF detector)** | `drift_test.py` | Siembra Python en el punto GAMS, RESUELVE, y rankea qué variables se ALEJAN de la semilla. Marca con ⚑ las que driftan mientras su ecuación pareada tiene residual≈0 = DOF libre/tautológico. Invisible a tools 3/4/6 porque NO hay diferencia Python-vs-GAMS — la var es libre igual en ambos | El solve converge (code=1) pero a OTRA rama de equilibrio y NINGUNA tool estática ve nada (coeficientes/pairing/calibración idénticos). Causa raíz del gap gtap7_3x3 78.69→97.68%: pva/pnd libres bajo CD (eq_pvaeq/eq_pndeq tautológicas), PATH los desliza; GAMS los pincha vía holdfix multi-período (externo a las ecuaciones) |
| 8 | **Holdfixed/sequence diff** | `diff_holdfixed.py` | Parsea el bloque `var.fx(...tsim-1)=var.l(...tsim-1)` del `loop(tsim)` de GAMS → la lista de variables que GAMS CONGELA del período anterior (holdfixed=1), y contrasta con qué congela Python entre etapas [1/3]→[2/3]→[3/3]. Es la disciplina de SECUENCIA, no del solve estático — invisible a tools 0-7 (que miran UN solve). Da además la receta del fix fiel | La tool 7 ve el SÍNTOMA (pva se escapa) pero ninguna ve la CAUSA del lado GAMS: GAMS congela pf/xf/pa/pe/pabs/pfact/pwfact del período previo (ancla pva/pnd indirectamente); Python warm-startea pero NO congela → el DOF libre re-desliza. gtap7_3x3: GAMS congela 25 vars, Python 0 |
| 9 | **Tautology / unanchored-var** | `diff_tautology.py` | Para cada par ecuación↔variable, PERTURBA la variable y mide ∂resid/∂var. Sensibilidad ≈0 → la ecuación NO restringe su propia variable pareada = es TAUTOLOGÍA = variable libre. Detecta la CAUSA (ecuación vacía) que la tool 7 solo ve como síntoma | El gap es una ecuación FALTANTE/tautológica (no una diferente). Las tools 3/4/6 son ciegas: ven coeficientes/pairing IDÉNTICOS (la tautología es igual en ambos modelos), residual ~0 en el punto GAMS (una tautología la satisface cualquier valor). Causa del cap 78.69%: eq_pvaeq bajo CD era `exp(log(pva))==pva` → pva libre. Fix: darle el determinante económico real `pva·va=Σ(pfa·xf)` |
| **11** | **Seed-and-solve (selección vs eq-difiere)** — **CORRER PRIMERO** | `seed_and_solve.py` | Siembra el punto GAMS EXACTO (seed derivado cascadeado, shock construido antes del build), RESUELVE, y da el sí/no de raíz: **STAYS** (resid~0, 0 eqs reales con residual = SELECCIÓN, parar) vs **GOES** (drifta = una eq DIFIERE). Reporta la **COLA** del residual (peores N eqs, NO mediana) que nombra la eq candidata; clasifica leaf (rgdpmp/pgdpmp) + ROW (ref corrupta) como benignos | El gate cell-by-cell no llega al objetivo y hay que saber si es selección (no hay ecuación mala) o una eq que difiere (nombrable). Discriminador definitivo en UNA corrida. Precedente gtap7_3x3: con el fix esubi revertido nombra `eq_xi` (0.006, drift 57.7%) de primera — la eq que costó 8 leads a mano. Causa: esubi key-shape, sigmai=0 Leontief vs GAMS 1.01 CES (commit 0e2db11) |

**Pitfall clave (herramienta 5):** el .nl compara coeficientes en UN punto; una diferencia de forma que coincide numéricamente en la semilla pasa el gate. La tool 5 imprime ambas formas pareadas para inspección dirigida (no auto-diff: la igualdad simbólica cross-lenguaje es frágil).

**Pitfall clave (herramienta 6):** un emparejamiento MCP distinto (Python resuelve una ecuación que GAMS deja free-row) hace que PATH converja limpio (code=1, residual ~0) a una raíz DISTINTA de un bloque multivaluado. Invisible a inputs/coeficientes/formas — solo el bloque `model gtap` lo revela. Free-rows de gtap7_3x3: xseq, pfteq, savfeq, capAccteq, pnumeq, walraseq, eveq, cveq (los últimos 4 son benignos: numéraire/walras/bienestar).

**Pitfall clave (herramienta 7 — la que cierra el agujero de las tools 3/4/6):** cuando el gap NO es una diferencia Python-vs-GAMS sino un DOF libre/degenerado IDÉNTICO en ambos (ej. bajo CD `eq_pvaeq`/`eq_pndeq` son la tautología `exp(log(pva))==pva` en los DOS), ninguna comparación estática lo ve: el .nl ve coeficientes idénticos (tautología==tautología → gate verde), el MCP-pairing ve el mismo emparejamiento (`pvaeq.pva` en ambos → "pairing matches"), el residual en el punto GAMS es ~0 (una tautología la satisface cualquier valor). Solo el **drift test** lo atrapa: siembra en GAMS, resuelve, y la var libre se ALEJA aunque su residual sea ~0 (⚑). Precedente gtap7_3x3 (78.69→97.68%): pva/pnd se deslizan 4.7%, marcadas ⚑ por drift_test; GAMS las pincha vía holdfix multi-período (mecánica de solver/secuencia EXTERNA a las ecuaciones — por eso invisible a todo compare de modelo). El fix no es tocar ecuaciones: es HOLD (fijar var + desactivar la eq tautológica pareada = `gtap.holdfixed=1`), expuesto como `--holdfix-pva` en `diff_altertax.py`.

**Pitfall clave (herramientas 0-3):** warm-start con keys GAMS (`a_Food`, `c_Agr`) falla silenciosamente en Pyomo porque los elementos del set son `Food`, `Agr`. Siempre normalizar prefijos `a_`/`c_`/`f_`/`r_` antes del lookup. Y: **más seeding ≠ mejor warm-start** — sembrar GAMS-derived vars (xa→xaa, p→p_rai, piGbl) que chocan con el init de Python aterriza un basin code=2 PEOR; el subset inline (xd/xm+camelCase) reproduce el code=1 baseline. El drift test usa el subset inline a propósito.

**Pitfall clave (herramienta 4):** un sesgo de insumo de calibración de ~0.04% es **invisible** a la herramienta 1 (sale "ok" bajo `tol_rel=1e-3`) y mal atribuido por `validate_reference` (ve el residual downstream, culpa al GDX). Precedente: `ytax[ROW,'mt']` usaba `imptx·vmsb` (valor de mercado) en vez de `imptx·VCIF` (valor CIF), sesgando yTaxInd→regY→betaP en 0.039% — escondido varias sesiones hasta que `diff_calibration.py` lo marcó por stream (fix en `_compute_ytax_ind_bench`: `vmsb`→`vcif`). `regY` se compara income-side (Python regY == GAMS factY+yTaxInd), no contra el regY que GAMS fija expenditure-side.

## Archivos clave

| Archivo | Propósito |
|---------|-----------|
| `src/equilibria/templates/gtap/gtap_model_equations.py` | Ecuaciones del modelo. Áreas críticas: `get_gdpmp_init`, `get_yi_init`, `get_xiagg_init`, `eq_ytax`, `eq_yc`, `eq_yg`, `eq_pabs`, `eq_gdpmp`. Líneas 1134, 1862, 4510, 4574 ya fijadas a `NAmerica`. |
| `scripts/gtap/run_gtap.py` | CLI. `validate-shock`, `_apply_shock_to_params` (`tm_pct`), `_collect_key_quantities` (emite `ytax(r,gy)` con 10 streams canónicos). |
| `scripts/gtap/diff_altertax.py` | Diff cell-by-cell Python altertax vs NEOS out.gdx. 3 períodos: betaCal → check → shock. Flags: `--compare-gdx` (warm-start de un GDX, comparar contra otro), `--no-gams-warm`, `--use-gams-check`. |
| `scripts/gtap/validate_reference.py` | Siembra un GDX GAMS en Python y reporta qué ecuaciones viola la PROPIA referencia (detecta GDX corrupto/mal convergido). |
| `scripts/gtap/diff_calibration.py` | **Herramienta 4 de la cascada.** Diff de insumos de calibración (betaP/betaG/betaS, factY/yTaxInd, cada stream de `ytax`, CDE eh/bh/xcshr/zcons, bloque factor gf/aft/xscale) Python vs GAMS en el benchmark, por región/celda, a tol 1e-4. Atrapa sesgos de calibración invisibles al comparador de solve. |
| `scripts/gtap/diff_equation_form.py` | **Herramienta 5 de la cascada.** Imprime la forma simbólica expandida de una ecuación Python (1/xscale → literal 10.0/100.0) lado a lado con la definición GAMS del `.gms`. Atrapa diferencias estructurales de forma que el .nl (coeficientes en un punto) no muestra. Uso: `--eq eq_xd_agg --cell ROW,Svces --gams-eq xds`. |
| `scripts/gtap/diff_mcp_pairing.py` | **Herramienta 6 de la cascada.** Parsea el emparejamiento MCP de GAMS (`model gtap /eq.var/`) y marca free-rows de GAMS que Python mantiene activas+emparejadas (clase del bug pfteq factor-2). `--apply-closure` refleja el estado real post-solver. Uso: `--dataset gtap7_3x3 --apply-closure`. |
| `scripts/gtap/drift_test.py` | **Herramienta 7 de la cascada (free-DOF detector).** Siembra Python en el punto GAMS, resuelve, rankea drift por variable; marca ⚑ las que driftan con su eq pareada en residual≈0 (DOF libre/tautológico). Reusa el build+seed de `diff_altertax`. Usa el seed INLINE (no el completo — over-seeding aterriza code=2). Atrapa el gap que tools 3/4/6 no ven: un DOF libre IDÉNTICO en Python y GAMS (ej. pva/pnd bajo CD). Uso: `--dataset gtap7_3x3 --gdx <ref> --period shock`. |
| `scripts/gtap/diff_holdfixed.py` | **Herramienta 8 de la cascada (holdfixed/sequence diff).** Parsea `var.fx(...tsim-1)=var.l(...tsim-1)` del `loop(tsim)` de GAMS (la lista holdfixed del período previo) y reporta qué congela Python entre etapas (hoy: NADA). Es la disciplina de SECUENCIA — invisible a tools 0-7. Da la receta del fix fiel (congelar esas vars en los valores del período previo de PYTHON, no de GAMS). gtap7_3x3: GAMS congela 25 vars (pf/xf/pa/pe/pabs/pfact/pwfact… anclan pva/pnd), Python 0. Uso: `--gms <model_altertax_ifsub0.gms>`. |
| `scripts/gtap/diff_tautology.py` | **Herramienta 9 de la cascada (tautology / unanchored-var detector).** Para cada par eq↔var, perturba la variable y mide ∂resid/∂var; sensibilidad≈0 → la ecuación es tautológica para su variable → variable libre (free DOF). Detecta la CAUSA (ecuación vacía de contenido) que tools 3/4/6 no ven (comparan diferencias; una tautología es IGUAL en ambos modelos) y que tool 7 solo ve como síntoma. Atrapó el bug que capó gtap7_3x3 en 78.69%: eq_pvaeq bajo CD = `exp(log(pva))==pva`. Uso: `--dataset gtap7_3x3 --gdx <ref> --period shock`. |
| `scripts/gtap/seed_and_solve.py` | **Herramienta 11 de la cascada (selección vs eq-difiere) — CORRER PRIMERO.** Siembra el punto GAMS exacto (seed derivado cascadeado + shock construido antes del build, las dos disciplinas horneadas), resuelve, y da el sí/no: **STAYS**=selección (parar) / **GOES**=eq difiere. Reporta la COLA del residual (no mediana) que nombra la eq, clasifica leaf+ROW como benignos. Enganchada al orquestador como primera capa (`cascade_layers.py`). Discriminó `eq_xi` (esubi key-shape) en una corrida. Uso: `--dataset gtap7_3x3 --gdx <ref> --period shock`. |
| `scripts/gtap/convert_gams.py` | **CONVERT como comando reproducible (paso de CONFIRMACIÓN, no de descubrimiento).** Dado el comp `.gms`, lo reescribe a `scaleopt=0` (niveles vs niveles) y rutea el solve MCP por el solver CONVERT de GAMS → emite el modelo como Pyomo canónico (`conv_<period>.py`) + dict `x###/e### → var/eq`. Para check inserta `abort` tras el solve del check. Requiere GAMS local (CONVERT viene incluido, sin licencia). Se usa DESPUÉS de que `seed_and_solve` nombra la eq candidata: diffear esa eq escrita vs equilibria → confirmar difiere-de-verdad vs equivalente. Uso: `--dataset gtap7_3x3 --period shock --out-dir /tmp/gtap_convert`. |
| `scripts/gtap/diff_nus333_full.py` / `diff_9x10_full.py` | Diffs cell-by-cell vs GAMS (NEOS o local). 0 cells diverge en ambos. |
| `scripts/gtap/bench_nus333_dual.py` | Benchmark dual-reference (NEOS + GAMS local) + wall-time N=5. |
| `scripts/parity/triage.py` | CLI de debug parity: locate→isolate→trace→check-warmstart. |
| `scripts/parity/_triage_steps.py` | Implementación de los 4 pasos de triage. |
| `scripts/parity/_adapter_protocol.py` | Protocolo ParityAdapter + registry. |
| `scripts/parity/probe.py` | Probe cacheado (de `main`). `--show <vars>`, `--residuals [--top N]`, `--seed-gams <period> --gdx-ref <gdx>`, `--compare-ref <commit>`, y `--params`/`--params-compare-builds` (Param/calibration diff: detecta constantes horneadas en build según `t0_snapshot` — pf0, base_rgdpmp, p_gf, betap — que la cascada no ve). Acelera la iteración de hipótesis. Pitfall: warm-start con keys GAMS (`a_Food`) falla silenciosamente; normalizar prefijos `a_`/`c_`/`f_`/`r_`. |
| `src/equilibria/templates/gtap/gtap_solver.py` | Wrapper PATH. `apply_closure`, `apply_aggressive_fixing_for_mcp`, fijación de numerario. |
| `src/equilibria/templates/gtap/gtap_parameters.py` | Carga de parámetros, `savf_bar`, splits de demanda final. |
| `GTAP_VALIDATION_STATUS.md` | Status detallado por sesión, hipótesis, hallazgos. |
| `docs/site/benchmarks.md` | Página benchmarks rendered en Read the Docs (parity NEOS + local + wall-time). |

## Referencia GAMS

- `tariff_comp.gms` con `ifSUB=0` (equivalente Python: `if_sub=False`).
- NEOS jobs de referencia: 18737509 (9×10), regen NUS333 vía `build_nus333_neos_bundle.py` (postsim recalc `pdp/pmp` + `pwmg=0` fix incluidos upstream en `postsim.gms`/`iterloop.gms`).
