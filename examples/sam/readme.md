# ejemplos sam

Pipeline YAML para transformar SAM entre formatos y aplicar ajustes secuenciales.

Tambien puedes usar la API Python con la clase `SAM`:

```python
from equilibria.sam_tools import SAM

sam = SAM.from_ieem_excel("data/matriz_contabilidad_social_2016.xlsx", sheet_name="MCS2016")
sam.aggregate("data/mapping_template.xlsx").balance_ras()
state = sam.to_raw_state()
```

## Ejemplo CRI -> PEP compatible

```bash
uv run python scripts/sam_tools/run_sam_transform_pipeline.py \
  --config examples/sam/cri_pep_transform.yaml
```

## Ejemplo RAW IEEM -> PEP compatible (una sola corrida)

`input.format: ieem_raw_excel` permite arrancar desde una SAM tipo IEEM (por ejemplo `MCS2016`) y aplicar:
- lectura raw,
- agregación con mapping,
- normalización a cuentas PEP,
- y luego transformaciones del pipeline.

En `input.options`:
- `sheet_name`: nombre de hoja raw (default `MCS2016`).

Ejemplo de config:
- `examples/sam/cri_ieem_raw_to_pep.yaml`

## Operaciones disponibles

Nota de arquitectura:
- `aggregate_mapping` es generica para cualquier SAM agregada por mapping.
- Las operaciones IEEM->PEP (`balance_ras`, `normalize_pep_accounts`, `create_x_block`, `convert_exports_to_x`, `align_ti_to_gvt_j`) viven en `ieem_to_pep_transformations.py`.

- `scale_all`: reescala toda la matriz.
- `scale_slice`: reescala un bloque (`row`/`col`) con selector.
- `aggregate_mapping`: agrega una SAM RAW usando `mapping_path`.
- `balance_ras`: rebalancea matriz RAW con RAS.
  - `ras_type`: `arithmetic` (default), `geometric`, `row`, `column`.
- `normalize_pep_accounts`: convierte etiquetas agregadas a cuentas PEP (`J/I/L/K/AG/MARG/OTH`).
- `create_x_block`: agrega cuentas `X.*` para cada commodity `I.*`.
- `convert_exports_to_x`: mueve exportaciones `I.* -> AG.row` a `X.* -> AG.row` y ajusta `J->I/J->X`.
- `align_ti_to_gvt_j`: mueve `AG.ti -> J.*` a `AG.gvt -> J.*`.
- `shift_row_slice`: mueve una fracción de una fila origen a otra fila destino para columnas seleccionadas.
- `move_cell`: mueve monto entre dos celdas específicas.
- `move_k_to_ji`: mueve flujos `K.* -> I.*` hacia `J.map(i) -> I.*`.
- `move_l_to_ji`: mueve flujos `L.* -> I.*` hacia `J.map(i) -> I.*`.
- `move_margin_to_i_margin`: mueve `MARG.MARG -> I.*` hacia `I.{margin_commodity} -> I.*`.
- `move_tx_to_ti_on_i`: mueve `AG.tx -> I.*` hacia `AG.ti -> I.*`.
- `pep_structural_moves`: atajo compuesto (legacy) que ejecuta las cuatro operaciones anteriores.
- `rebalance_ipfp`: rebalancea con IPFP.
- `enforce_export_balance`: fuerza identidad de valor exportado.

## Selectores

- Exacto: `AG.tx`, `I.agr`
- Wildcards: `AG.*`, `*.agr`, `*.*`
- También acepta `{cat: AG, elem: tx}`

## Transformaciones IEEM -> PEP

- Indice (2 idiomas: espanol e ingles):
  - `docs/ieem_to_pep_transformations.md`
