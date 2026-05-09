"""
MIP → SAM in one call
=====================

This example takes a tiny 3×3 MIP fixture and runs the full
:func:`equilibria.sam_tools.run_mip_to_sam` pipeline. Each step is recorded
with balance statistics; the final SAM is balanced via iterated GRAS.
"""

# %%
# Run the pipeline
# ----------------
from pathlib import Path

import equilibria
from equilibria.sam_tools import run_mip_to_sam

# Locate the test fixture relative to the installed package.
PKG_ROOT = Path(equilibria.__file__).resolve().parents[2]
FIXTURE = PKG_ROOT / "tests" / "sam_tools" / "fixtures" / "simple_mip.xlsx"

result = run_mip_to_sam(FIXTURE)

# %%
# Inspect the recorded steps
# --------------------------
# Every transformation logs its name and the post-step balance.
import pandas as pd

steps_df = pd.DataFrame(
    [
        {
            "step": s["step"],
            "total": s["balance"]["total"],
            "max_row_col_diff": s["balance"]["max_row_col_abs_diff"],
        }
        for s in result.steps
    ]
)
print(steps_df.to_string(index=False))

# %%
# Look at the resulting SAM
# -------------------------
df = result.sam.to_dataframe()
print("Shape:", df.shape)
print("Row categories:", sorted({cat for cat, _ in result.sam.row_keys}))
print("Column categories:", sorted({cat for cat, _ in result.sam.col_keys}))
