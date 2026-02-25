from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IEEM_RAW_GROUPS: list[tuple[str, str]] = [
    ("actividades productivas", "act_agr"),
    ("bienes y servicios", "com_agr"),
    ("margenes", "marg"),
    ("factores", "usk"),
    ("hogares", "hh"),
    ("empresas", "firm"),
    ("gobierno", "gvt"),
    ("resto del mundo", "row"),
    ("ahorro", "s_hh"),
    ("inversion", "inv"),
]


def write_sample_ieem_raw_excel(
    path: Path,
    matrix: np.ndarray,
    sheet_name: str = "MCS2016",
) -> None:
    """Write a minimal IEEM raw SAM Excel file for testing."""

    if matrix.shape != (len(IEEM_RAW_GROUPS), len(IEEM_RAW_GROUPS)):
        raise ValueError("IEEM raw fixture matrix must be square with shape (10, 10)")

    n_rows = 4 + len(IEEM_RAW_GROUPS) + 5
    n_cols = 3 + len(IEEM_RAW_GROUPS) + 5
    raw = np.full((n_rows, n_cols), np.nan, dtype=object)
    start_row = 4

    for i, (group_name, label) in enumerate(IEEM_RAW_GROUPS):
        row = start_row + i
        raw[row, 1] = group_name
        raw[row, 2] = label

    for i in range(len(IEEM_RAW_GROUPS)):
        for j in range(len(IEEM_RAW_GROUPS)):
            raw[start_row + i, 3 + j] = float(matrix[i, j])

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(raw).to_excel(writer, sheet_name=sheet_name, index=False, header=False)
