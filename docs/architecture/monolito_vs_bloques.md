# GTAP: monolito y bloques

Quien llega nuevo al repo suele suponer que `templates/gtap/gtap_model_equations.py` (el
«monolito») y `blocks/gtap/` (los «bloques») son dos implementaciones alternativas del mismo
modelo, y que la primera es legado a la espera de ser borrada.

**No lo son.** Este documento existe para evitar ese error, que ya costó tiempo una vez.

## Quién depende de quién

Los módulos de `blocks/gtap/*.py` sólo importan `gtap_parameters`. Las menciones a
`gtap_model_equations` en sus docstrings son comentarios de procedencia
(`VERBATIM from ... líneas X-Y`): dicen de dónde se copió cada ecuación, no crean una dependencia.

Eso lleva a concluir —incorrectamente— que el monolito ya no se usa. Pero el **composer** sí
depende de él:

| Sitio | Qué hace |
|---|---|
| `templates/gtap/gtap_block_model.py:42` | Importa `GTAPModelEquations` a nivel de módulo |
| `:242-246` | Instancia un shim para reutilizar `apply_production_scaling()` y `_align_xi_xaa_post_scaling()` |
| `:340-341`, `:359-360` | Monkey-patch de `GTAPModelEquations.build_model` para reaprovechar la reflexión multiperiodo |
| `templates/gtap/__init__.py:58` | Cualquier `import equilibria.templates.gtap` carga el monolito |

Borrar `gtap_model_equations.py` provoca *collection error* en ~34 archivos de test — **incluidos
los 9 que ejercitan sólo bloques**, porque el import falla antes de llegar a ellos.

## Cobertura de ecuaciones

101 familias `Constraint(...)` en el monolito frente a 100 `eq_*` en bloques.

De las 7 que sólo existían en el monolito, 6 se declaraban y se desactivaban en la línea siguiente
—nunca entraban al `.nl`— y ya fueron eliminadas. Queda `eq_pmuv` como único gap real, y sólo
muerde cuando `closure.rmuv` e `imuv` son ambos no vacíos, lo que no ocurre en ningún dataset del
gate. Está documentado en `blocks/gtap/__init__.py:47-53`.

Las 6 exclusivas de bloques (`eq_mfr_{bs,sb,ss}`, `eq_mfw_{bs,sb,ss}` en `closure.py:218-341`) no
son funcionalidad nueva: son la descomposición en constraints de lo que el monolito inlinea como
Expressions dentro de `eq_pfact`/`eq_pwfact`.

## Los bloques son un superconjunto funcional

La asimetría va en el sentido contrario al que sugiere la intuición:

| Modo | Monolito | Bloques |
|---|---|---|
| `capFix`, `capSFix` | sí | sí |
| `capFlex` | **no** (cae a un `else` genérico en `:7276`) | sí (`demand_utility.py:584`) |
| `capFixDp` | **no** | sí (`demand_utility.py:116,166`) |
| `base_calibrated=True` | **no existe** | sí (`gtap_block_model.py:372,440-446`) |
| basis `gempack` | parcial | sí (`gtap_contract.py:309-321`) |

`gtap_contract.py:357` declara 5 `savf_flag`; el monolito implementa 2.

## Por qué se conserva el monolito

Es el oráculo de fidelidad contra **GAMS** en 3 de los 5 gates de
`scripts/gtap/run_parity_gates.py`:

| Gate | Camino | Referencia |
|---|---|---|
| `test_gtap7_mcp_parity.py` | monolito | GAMS/PATH |
| `test_gtap7_nlp_parity.py` | monolito | GAMS/IPOPT |
| `test_gtap7_nl_parity.py` | monolito (vía `nl_compare`) | `.nl` de GAMS |
| `test_gtap7_gempack_parity.py` | **bloques** | GEMPACK SL4 |
| `tests/templates/gtap_logvalue/` | bloques | Julia/port |

El camino de bloques es el oráculo contra **GEMPACK**, y sólo él puede serlo: ese gate necesita
`capFlex` + `base_calibrated=True`, que el monolito no implementa.

## Deuda: cómo se retiraría

El primer corte real es extraer `apply_production_scaling` y `_align_xi_xaa_post_scaling` de
`GTAPModelEquations` a un módulo propio. Mientras esos dos métodos vivan ahí, el monolito es
dependencia de runtime del camino de bloques y no puede retirarse, por más que su cobertura de
ecuaciones ya esté replicada.

Después de ese corte quedaría por resolver el papel de oráculo contra GAMS: los 3 gates que hoy
miden el monolito tendrían que medir bloques, y eso exige comprobar que los números no se mueven.

Ver también: `ROADMAP.md` (registro de deuda técnica) y
`docs/findings/repo_cleanup_spec_2026-09-16.md` (la medición que originó este documento).
