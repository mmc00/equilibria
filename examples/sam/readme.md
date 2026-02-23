# ejemplos sam

Pipeline YAML para transformar SAM entre formatos y aplicar ajustes secuenciales.

## Ejemplo CRI -> PEP compatible

```bash
uv run python scripts/sam_tools/run_sam_transform_pipeline.py \
  --config examples/sam/cri_pep_transform.yaml
```

## Operaciones disponibles

- `scale_all`: reescala toda la matriz.
- `scale_slice`: reescala un bloque (`row`/`col`) con selector.
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
