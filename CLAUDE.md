# equilibria — GTAP Standard 7 GAMS/NEOS Parity (CLOSED)

## Estado del branch

**Paridad 100% lograda en ambos datasets (base + shock 10%):**

| Dataset | Python vs NEOS | Python vs GAMS local |
|---------|----------------|----------------------|
| NUS333 (3×2) | 100% / 100% | 100% / 100% |
| 9x10        | 100% / 100% | bloqueado por licencia community |

Detalles por sesión en `GTAP_VALIDATION_STATUS.md`. Trabajo activo del branch cerrado el 2026-05-12 (PR #3, commit `28a9b93` en `main`).

## Objetivo original

Lograr **paridad exacta** entre el template Python `equilibria` GTAP Standard 7 (9×10 y 3×2 NUS333) y la referencia GAMS/NEOS para:

1. **Baseline** (pre-shock)
2. **Shock uniforme de 10% en aranceles de importación** (`tm.fx = tm.l * 1.1`)

**Criterio:** signo y magnitud de los deltas endógenos coinciden con GAMS.

## Reglas de trabajo (no negociables)

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

## Herramientas de debug parity (cascade de 4)

Cada herramienta ve una capa distinta. Nunca concluir de una sola herramienta.

| # | Herramienta | Script | Ve | Ruta de uso |
|---|-------------|--------|----|-------------|
| 0 | **check-warmstart** | `triage.py --check-warmstart` | Residuales de ecuaciones + gaps de variables antes del solver | Solver converge a equilibrio equivocado |
| 1 | **Value/residual diff** | `triage.py` (locate→isolate→trace) | Dónde divergen valores resueltos | Variable específica diverge |
| 2 | **Closure diff** | `diff_closure.py` | Variables fijas/libres, ecuaciones activas | Shock diverge ampliamente (nivel de precios entero) |
| 3 | **.nl diff** | `nl_compare.py` | Coeficientes Jacobiano, forma algebraica | Pregunta sobre álgebra/coeficientes |

**Pitfall clave:** warm-start con keys GAMS (`a_Food`, `c_Agr`) falla silenciosamente en Pyomo porque los elementos del set son `Food`, `Agr`. Siempre normalizar prefijos `a_`/`c_`/`f_`/`r_` antes del lookup.

**Iteración rápida:** `scripts/parity/probe.py` — probe cacheado. Construye el modelo siempre (refleja el código actual) y cachea sólo el punto resuelto (invalidado por hash de las ecuaciones). Flags: `--show <vars> [--region R]`, `--residuals [--top N] [--family F]`, `--seed-gams <period> --gdx-ref <gdx>` (con gate de cobertura), `--compare-ref <commit>` (A/B automático entre commits). No reemplaza la cascada de 4 — acelera la iteración de hipótesis y la atribución.

## Archivos clave

| Archivo | Propósito |
|---------|-----------|
| `src/equilibria/templates/gtap/gtap_model_equations.py` | Ecuaciones del modelo. Áreas críticas: `get_gdpmp_init`, `get_yi_init`, `get_xiagg_init`, `eq_ytax`, `eq_yc`, `eq_yg`, `eq_pabs`, `eq_gdpmp`. Líneas 1134, 1862, 4510, 4574 ya fijadas a `NAmerica`. |
| `scripts/gtap/run_gtap.py` | CLI. `validate-shock`, `_apply_shock_to_params` (`tm_pct`), `_collect_key_quantities` (emite `ytax(r,gy)` con 10 streams canónicos). |
| `scripts/gtap/diff_altertax.py` | Diff cell-by-cell Python altertax vs NEOS out.gdx. 3 períodos: betaCal → check → shock. |
| `scripts/gtap/diff_nus333_full.py` / `diff_9x10_full.py` | Diffs cell-by-cell vs GAMS (NEOS o local). 0 cells diverge en ambos. |
| `scripts/gtap/bench_nus333_dual.py` | Benchmark dual-reference (NEOS + GAMS local) + wall-time N=5. |
| `scripts/parity/triage.py` | CLI de debug parity: locate→isolate→trace→check-warmstart. |
| `scripts/parity/_triage_steps.py` | Implementación de los 4 pasos de triage. |
| `scripts/parity/_adapter_protocol.py` | Protocolo ParityAdapter + registry. |
| `src/equilibria/templates/gtap/gtap_solver.py` | Wrapper PATH. `apply_closure`, `apply_aggressive_fixing_for_mcp`, fijación de numerario. |
| `src/equilibria/templates/gtap/gtap_parameters.py` | Carga de parámetros, `savf_bar`, splits de demanda final. |
| `GTAP_VALIDATION_STATUS.md` | Status detallado por sesión, hipótesis, hallazgos. |
| `docs/site/benchmarks.md` | Página benchmarks rendered en Read the Docs (parity NEOS + local + wall-time). |

## Referencia GAMS

- `tariff_comp.gms` con `ifSUB=0` (equivalente Python: `if_sub=False`).
- NEOS jobs de referencia: 18737509 (9×10), regen NUS333 vía `build_nus333_neos_bundle.py` (postsim recalc `pdp/pmp` + `pwmg=0` fix incluidos upstream en `postsim.gms`/`iterloop.gms`).
