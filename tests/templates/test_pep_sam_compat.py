from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equilibria.templates.pep_sam_compat import SAMGrid, write_grid_to_excel


def test_write_grid_to_excel_handles_string_typed_cells(tmp_path: Path) -> None:
    raw_df = pd.DataFrame(
        [
            ["", "", ""],
            ["ROW", "COL", "VAL"],
            ["r1", "c1", "0"],
        ],
        dtype="string",
    )
    grid = SAMGrid(
        raw_df=raw_df,
        data_start_row=2,
        data_start_col=2,
        row_keys=[("AG", "ROW")],
        col_keys=[("AG", "COL")],
        matrix=np.array([[0.0]]),
    )
    output = tmp_path / "sam.xlsx"

    write_grid_to_excel(grid, np.array([[1.25]]), output)

    reloaded = pd.read_excel(output, sheet_name="SAM", header=None)
    assert float(reloaded.iat[2, 2]) == 1.25
