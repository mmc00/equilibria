# SAM examples

A manual pipeline to convert a SAM between formats and apply adjustments in sequence.

Recommended simple API (centered on `Sam`):

```python
from equilibria.sam_tools import export_sam, parse_sam, run_ieem_to_pep

# leer/escribir sin exponer metadatos internos
sam = parse_sam("data/sam.xlsx", "excel")
export_sam(sam, "output/sam.gdx", output_format="gdx")

# IEEM raw -> PEP en una llamada
result = run_ieem_to_pep(
    "data/matriz_contabilidad_social_2016.xlsx",
    "data/mapping_template.xlsx",
    output_path="output/sam_cri_pep_compatible.xlsx",
    report_path="output/sam_cri_pep_compatible_report.json",
)
sam_pep = result.sam
```

You can also use the Python API with the `Sam` class (directly or through the manual pipeline):

```python
from equilibria.sam_tools.ieem_raw_excel import IEEMRawSAM
from equilibria.sam_tools.manual_pipeline import run_manual_pipeline

sam = IEEMRawSAM.from_ieem_excel("data/matriz_contabilidad_social_2016.xlsx", sheet_name="MCS2016")
summary = run_manual_pipeline(sam)
print(summary.total_flow)
```

## Example: raw IEEM -> PEP-compatible (a single run)

`input.format: ieem_raw_excel` starts from an IEEM-style SAM (for example `MCS2016`) and applies:
- raw reading,
- aggregation with a mapping,
- normalization to PEP accounts,
- and then the pipeline transformations.

In `input.options`:
- `sheet_name`: name of the raw sheet (default `MCS2016`).

Example config:
- `examples/sam/cri_ieem_raw_to_pep.yaml`

## Available operations

Architecture note:
- `aggregate_mapping` is generic for any SAM aggregated by a mapping.
- The IEEM->PEP operations (`balance_ras`, `normalize_pep_accounts`, `create_x_block`, `convert_exports_to_x`, `align_ti_to_gvt_j`) live in `ieem_to_pep_transformations.py`.

- `scale_all`: rescales the whole matrix.
- `scale_slice`: rescales a block (`row`/`col`) with a selector.
- `aggregate_mapping`: aggregates a RAW SAM using `mapping_path`.
- `balance_ras`: rebalances a RAW matrix with RAS.
  - `ras_type`: `arithmetic` (default), `geometric`, `row`, `column`.
- `normalize_pep_accounts`: turns aggregated labels into PEP accounts (`J/I/L/K/AG/MARG/OTH`).
- `create_x_block`: adds `X.*` accounts for each commodity `I.*`.
- `convert_exports_to_x`: moves exports `I.* -> AG.row` to `X.* -> AG.row` and adjusts `J->I/J->X`.
- `align_ti_to_gvt_j`: moves `AG.ti -> J.*` to `AG.gvt -> J.*`.
- `shift_row_slice`: moves a fraction of a source row to a target row for the selected columns.
- `move_cell`: moves an amount between two given cells.
- `move_k_to_ji`: moves flows `K.* -> I.*` to `J.map(i) -> I.*`.
- `move_l_to_ji`: moves flows `L.* -> I.*` to `J.map(i) -> I.*`.
- `move_margin_to_i_margin`: moves `MARG.MARG -> I.*` to `I.{margin_commodity} -> I.*`.
- `move_tx_to_ti_on_i`: moves `AG.tx -> I.*` to `AG.ti -> I.*`.
- `pep_structural_moves`: a compound shortcut (legacy) that runs the four operations above.
- `rebalance_ipfp`: rebalances with IPFP.
- `enforce_export_balance`: enforces the exported-value identity.

## Selectors

- Exact: `AG.tx`, `I.agr`
- Wildcards: `AG.*`, `*.agr`, `*.*`
- It also accepts `{cat: AG, elem: tx}`

## IEEM -> PEP transformations

- Index (2 languages: Spanish and English):
  - `docs/notes/pep/ieem_to_pep_transformations.md`
