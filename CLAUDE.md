# equilibria — GTAP Standard 7 (9×10) GAMS/NEOS Parity Branch

## Objetivo del branch

Lograr **paridad exacta** entre el template Python `equilibria` GTAP Standard 7 (9 sectores × 10 regiones) y la referencia GAMS/NEOS (`out.gdx`, job 18737509, `tariff_comp.gms`, `ifSUB=0`) para:

1. **Baseline** (pre-shock)
2. **Shock uniforme de 10% en aranceles de importación** (`tm.fx = tm.l * 1.1`)

**Criterio único de éxito:** signo y magnitud de los deltas endógenos coinciden con GAMS.
La calidad de convergencia (residual) NO es criterio — sólo importa la paridad de deltas.

## Reglas de trabajo (no negociables)

- **Solver:** siempre usar PATH C API en modo **nonlinear full** (10,296 ecuaciones). Nunca el bloque linear (1,370). IPOPT no aplica (degrees-of-freedom).
- **ifMCP (GAMS) = 1:** el run de referencia NEOS (job 18737509) usa `ifMCP=1` → `solve using mcp` → PATH. Python también usa PATH. Ambos están alineados — NO cambiar a `ifMCP=0` (NLP/walras) porque cambiaría los valores de referencia y Python no soporta NLP en modo nonlinear full.
- **equation_scaling=True:** siempre pasar `equation_scaling=True` a `_run_path_capi_nonlinear_full` (baseline Y shocked). Sin esto el baseline queda en code=2/res~1e-6 en vez de code=1/res~1e-9. Tanto `validate-shock` como `_run_homotopy_shocked` lo usan — `validate_gams_parity.py` también debe usarlo.
- **Modo de shock:** usar siempre `--shock-mode tm_pct` para shocks tipo GAMS (multiplica la *power* del arancel: `imptx_new = (1+imptx_old)*(1+v) − 1`). El modo `pct` viejo escala sólo la tasa y produce shocks ~10× menores en bienes con arancel bajo.
- **Región residual:** `NAmerica` (coincide con `rres` de GAMS). NO usar `RestofWorld` aunque exista como 10ª región real en el dataset 9x10.
- **GAMS license expirada (Oct 2024):** sólo `gdxdump` CLI funciona para leer GDX (no `gdxpds`, no correr modelos GAMS).
- **Dataset:** `basedata-9x10.gdx` ≡ `9x10Dat.gdx` (verificados idénticos).

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
| Region residual fix | ✅ aplicado |
| Shock formula fix (`tm_pct`) | ✅ aplicado |
| `EastAsia regy_delta` vs GAMS | ✅ −20,991 vs −20,101 (4% error) |
| Baseline nonlinear residual | ⚠️ 1.04e-06 (gate estricto 1e-8) |
| Shocked nonlinear convergence (1000 pares) | ❌ residual 0.38 |
| `gdpmp` baseline parity | ❌ 15.061 vs 15.220 |
| Parity check variable-por-variable | ❌ pendiente |
| Bug `_collect_key_quantities` ytax | ❌ pendiente (baja prio) |

## Archivos clave

| Archivo | Propósito |
|---------|-----------|
| `src/equilibria/templates/gtap/gtap_model_equations.py` | Ecuaciones del modelo. Áreas críticas: `get_gdpmp_init`, `get_yi_init`, `get_xiagg_init`, `eq_ytax`, `eq_yc`, `eq_yg`, `eq_pabs`, `eq_gdpmp`. Líneas 1134, 1862, 4510, 4574 ya fijadas a `NAmerica`. |
| `scripts/gtap/run_gtap.py` | CLI. `validate-shock`, `_apply_shock_to_params` (`tm_pct`), bug pendiente en `_collect_key_quantities` (583–594). |
| `src/equilibria/templates/gtap/gtap_solver.py` | Wrapper PATH. `apply_closure`, `apply_aggressive_fixing_for_mcp`, fijación de numerario. |
| `src/equilibria/templates/gtap/gtap_parameters.py` | Carga de parámetros, `savf_bar`, splits de demanda final. |
| `GTAP_VALIDATION_STATUS.md` | Status detallado por sesión, hipótesis, hallazgos. |
| `output/gtap_ifsub_false_warmstart.json` | Última corrida shockeada buena (signos invertidos, pre-fix). |
| `output/gtp_baseline_reverted.json` | Última baseline buena (res 1.12e-06). |

## Referencia GAMS

- `tariff_comp.gms` con `ifSUB=0` (equivalente Python: `if_sub=False`).
- Truco crítico GAMS en `cal.gms:652`: sobrescribe `yi = pi*depr*kstock + rsav + savf` dejando `xi` en valor de absorción → residual deliberado en `xieq` que el solver resuelve subiendo `xi` y `gdpmp`. Replicarlo en Python rompió convergencia (residual 21) — revertido.
