# MIP → SAM pipeline

`equilibria.sam_tools` converts a raw input-output / MIP (Matrix of
Intermediate Production) table into a balanced Social Accounting Matrix.
The conversion is decomposed into ten transformation steps that each
record their own balance statistics, so you can audit exactly where flows
were created or rebalanced.

## Pipeline overview

| Step | Purpose |
|------|---------|
| `normalize_mip_accounts` | Re-classify raw `RAW.*` rows/cols into `I` (commodities), `J` (sectors), `VA`, `IMP`, `FD` (final demand). |
| `disaggregate_va_to_factors` | Split `VA` aggregate into labour `L` and capital `K` using configurable shares. |
| `create_factor_income_distribution` | Route factor income to institutional accounts (`AG.hh`). |
| `create_household_expenditure` | Convert `FD.HH` columns into household consumption rows `AG.hh → I`. |
| `create_government_flows` | Build `gvt`, `ti`, `tm` accounts (taxes, transfers, government consumption). |
| `create_row_account` | Attach a Rest-of-World account from import rows. |
| `create_make_matrix` | Add the diagonal J → I closure for sectoral production. |
| `create_investment_account` | Place investment and savings flows. |
| `create_x_block` | Attach the export/X block. |
| `convert_exports` | Re-route `FD.EXP` columns into the `X` aggregate. |
| `balance_ras` | Final iterated GRAS balance using `(row_sum + col_sum) / 2` targets. |

## One-shot conversion

The simplest entry point is :func:`equilibria.sam_tools.run_mip_to_sam`,
which loads the MIP, runs every step, balances the SAM, and returns a
result object with both the final SAM and the per-step report.

```python
from pathlib import Path
from equilibria.sam_tools import run_mip_to_sam

result = run_mip_to_sam(
    Path("data/my_mip.xlsx"),
    sheet_name="MIP",
    va_factor_shares={"L": 0.65, "K": 0.35},
    ras_max_iter=200,
    output_path=Path("out/my_sam.xlsx"),
    report_path=Path("out/my_sam_report.json"),
)

print(f"Steps run: {len(result.steps)}")
print(f"Final balance: {result.steps[-1]['balance']['max_row_col_abs_diff']:.2e}")
```

A complete runnable version of this snippet (with output) lives in the
{doc}`example gallery <../gallery/example_mip_to_sam>`.

## Calling individual transforms

When you need fine-grained control — e.g. for diagnostic comparisons or
to inject a custom step — call the transforms directly. Each one mutates
the `MIPRawSAM` in place and returns a small report dict.

```python
from equilibria.sam_tools.mip_raw_excel import MIPRawSAM
from equilibria.sam_tools.mip_to_sam_transforms import (
    normalize_mip_accounts,
    disaggregate_va_to_factors,
    create_factor_income_distribution,
)

sam = MIPRawSAM.from_mip_excel("data/my_mip.xlsx", sheet_name="MIP")

normalize_mip_accounts(sam, {})
disaggregate_va_to_factors(sam, {"va_factor_shares": {"L": 0.65, "K": 0.35}})
create_factor_income_distribution(sam, {})

# ... run the rest of the pipeline ...
```

## SAM convention

`equilibria` follows the convention `df.loc[receiver_row, payer_col]` —
**rows receive, columns pay**. Whenever you write your own transforms,
keep this orientation; the post-step balance check compares row sums to
column sums.

## Troubleshooting

* **`max_row_col_abs_diff` does not converge** — the source MIP is
  square-padded with zero rows/columns that should not exist; check the
  fixture against `MIPRawSAM.from_mip_excel(..., debug=True)` and either
  trim the padding or rebalance the inputs.
* **Negative GRAS targets** — `gras_balance` handles negative entries,
  but the iterated outer loop in `_final_balance_normalized_sam` averages
  row and column sums; if any target stays negative the iteration stalls.
  The fix is usually to inspect which sub-step introduced a negative
  flow (look for `total_distributed < 0` in the report).
