# Documentación de equilibria

Índice de los ~99 documentos de este árbol. Empieza por aquí antes de buscar a ciegas.

## Por dónde empezar

| Si quieres… | Lee |
|---|---|
| Entender qué es el proyecto y cómo se usa | `../README.md` |
| Saber en qué estado está y qué falta | `../ROADMAP.md` |
| Entender la arquitectura GTAP (monolito vs bloques) | `architecture/monolito_vs_bloques.md` |
| Saber por qué un número no coincide con GAMS o GEMPACK | `findings/` (ordenado por fecha) |

## Los directorios

### `architecture/` — cómo encaja el sistema

- **`monolito_vs_bloques.md`** — lectura obligada antes de tocar GTAP. Explica por qué el monolito
  y los bloques no son alternativas intercambiables, y por qué no se puede borrar el primero.
- `gams_parity_matrix.md` — qué mide cada gate de paridad.

### `findings/` — resultados de investigación, con fecha

El registro de lo que se descubrió y cuándo. **Son documentos históricos: un hallazgo posterior
puede refutar a uno anterior.** Cuando eso pasa, el documento refutado lleva un aviso al principio;
si el título y el aviso se contradicen, manda el aviso.

Ejemplo vivo: `gempack_residual_is_linearization_2026-07-24.md` sostiene en el título que el
residuo contra GEMPACK es linearización, y abre con un `⚠️ UPDATE 2026-08-19` que lo desmiente —
la causa real era el *seed* (`base_calibrated=True`). El step-grid de Gragg lo demuestra: GEMPACK
converge a 0,002 pp, así que la linearización no podía explicar un gap de 0,4 pp.

Documentos de cierre destacados: `f3_blocks_done_2026-07-29.md` (GTAP como bloques compuestos),
`f3_5_base_calibrado_done_2026-07-30.md` (base model-consistent),
`gempack_fixture_closure_mislabeled_2026-08-02.md` (el cruce de closures).

### `guides/` — cómo hacer cosas

Conversión MIP → SAM, la API del contrato PEP, el harness del Jacobiano, los gates de escenarios.

### `analysis/` — estudios sobre datos concretos

Sobre todo el caso Bolivia (MIP → SAM). Su subdirectorio `archive/` es material superado.

### `technical/` — métodos numéricos

Balanceo de matrices: GRAS, RAS, SUT-RAS, entropía, y comparación entre variantes.

### `reference/` — formatos de archivo

Estructura binaria de GDX, principalmente.

### `plans/` — planes de implementación, con fecha

Histórico de cómo se abordó cada bloque de trabajo. Consúltalos para entender decisiones pasadas,
no como indicación de lo que está pendiente hoy: para eso está `../ROADMAP.md`.

### `archive/` — material superado

Se conserva por trazabilidad. **Puede contradecir hallazgos posteriores.**

### `site/` — fuente de la documentación publicada

Lo que alimenta la web (mkdocs). Incluye las matrices de cobertura generadas por los gates:
`site/guide/gtap7_coverage_matrix.md` y `site/guide/gtap7_coverage_matrix_gempack.md`. Se
regeneran solas — no se editan a mano.

### `assets/` — logos e imágenes

## Política de nombres de archivo

- Por defecto, nombres en minúsculas.
- Las excepciones legacy o contractuales están listadas en `scripts/check_lowercase_filenames.py`.
- Verificación local: `uv run --frozen python scripts/check_lowercase_filenames.py` (CI lo exige).

## Convención sobre fechas

Un documento con fecha en el nombre es un registro de ese momento, no una verdad permanente.
Si necesitas saber qué es cierto **hoy**, mira `../ROADMAP.md` o corre los gates:

```bash
uv run --frozen python scripts/gtap/run_parity_gates.py
```
