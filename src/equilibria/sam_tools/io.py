"""Input/output helpers for SAM workflow state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.babel.gdx.symbols import Parameter
from equilibria.babel.gdx.writer import write_gdx
from equilibria.sam_tools.ieem_raw_excel import SAM
from equilibria.sam_tools.models import SAMTransformState
from equilibria.sam_tools.selectors import norm_text
from equilibria.templates.pep_sam_compat import load_sam_grid


def _load_from_excel(path: Path) -> SAMTransformState:
    grid = load_sam_grid(path)
    return SAMTransformState(
        matrix=grid.matrix.copy().astype(float),
        row_keys=grid.row_keys,
        col_keys=grid.col_keys,
        source_path=path,
        source_format="excel",
        raw_df=grid.raw_df,
        data_start_row=grid.data_start_row,
        data_start_col=grid.data_start_col,
    )


def _load_from_gdx(path: Path) -> SAMTransformState:
    gdx = read_gdx(path)
    values = read_parameter_values(gdx, "SAM")
    if not values:
        raise ValueError(f"No SAM records found in GDX: {path}")

    row_order: list[tuple[str, str]] = []
    col_order: list[tuple[str, str]] = []
    row_idx: dict[tuple[str, str], int] = {}
    col_idx: dict[tuple[str, str], int] = {}

    for keys in values:
        if len(keys) != 4:
            continue
        row_key = (norm_text(keys[0]), norm_text(keys[1]))
        col_key = (norm_text(keys[2]), norm_text(keys[3]))
        if row_key not in row_idx:
            row_idx[row_key] = len(row_order)
            row_order.append(row_key)
        if col_key not in col_idx:
            col_idx[col_key] = len(col_order)
            col_order.append(col_key)

    if not row_order or not col_order:
        raise ValueError(f"Could not build row/column support from GDX SAM: {path}")

    matrix = np.zeros((len(row_order), len(col_order)), dtype=float)
    for keys, value in values.items():
        if len(keys) != 4:
            continue
        row_key = (norm_text(keys[0]), norm_text(keys[1]))
        col_key = (norm_text(keys[2]), norm_text(keys[3]))
        matrix[row_idx[row_key], col_idx[col_key]] += float(value)

    return SAMTransformState(
        matrix=matrix,
        row_keys=row_order,
        col_keys=col_order,
        source_path=path,
        source_format="gdx",
    )


def load_state(
    path: Path,
    fmt: str,
    options: dict[str, Any] | None = None,
) -> SAMTransformState:
    if not path.exists():
        raise FileNotFoundError(f"Input SAM not found: {path}")
    opts = options or {}
    if fmt == "excel":
        return _load_from_excel(path)
    if fmt == "gdx":
        return _load_from_gdx(path)
    if fmt == "ieem_raw_excel":
        return SAM.from_ieem_excel(
            path=path,
            sheet_name=str(opts.get("sheet_name", "MCS2016")),
        ).to_raw_state(source_path=path, source_format="ieem_raw_excel")
    raise ValueError(f"Unsupported input format: {fmt}")


def _write_excel_preserving_layout(state: SAMTransformState, output_path: Path) -> None:
    if state.raw_df is None or state.data_start_row is None or state.data_start_col is None:
        raise ValueError("No original Excel layout available")

    out_df = state.raw_df.copy()
    n_rows, n_cols = state.matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            out_df.iat[state.data_start_row + i, state.data_start_col + j] = float(
                state.matrix[i, j]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="SAM", index=False, header=False)


def _write_excel_canonical(state: SAMTransformState, output_path: Path) -> None:
    n_rows, n_cols = state.matrix.shape
    grid = np.full((n_rows + 2, n_cols + 2), "", dtype=object)

    for j, (cat, elem) in enumerate(state.col_keys):
        grid[0, 2 + j] = cat
        grid[1, 2 + j] = elem

    for i, (cat, elem) in enumerate(state.row_keys):
        grid[2 + i, 0] = cat
        grid[2 + i, 1] = elem
        for j in range(n_cols):
            grid[2 + i, 2 + j] = float(state.matrix[i, j])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(grid).to_excel(writer, sheet_name="SAM", index=False, header=False)


def _write_state_to_excel(state: SAMTransformState, output_path: Path) -> None:
    if (
        state.raw_df is not None
        and state.data_start_row is not None
        and state.data_start_col is not None
        and state.data_start_row + state.matrix.shape[0] <= state.raw_df.shape[0]
        and state.data_start_col + state.matrix.shape[1] <= state.raw_df.shape[1]
    ):
        _write_excel_preserving_layout(state, output_path)
    else:
        _write_excel_canonical(state, output_path)


def _write_state_to_gdx(
    state: SAMTransformState,
    output_path: Path,
    symbol_name: str,
    zero_threshold: float = 1e-14,
) -> int:
    records: list[tuple[list[str], float]] = []
    for i, (r_cat, r_elem) in enumerate(state.row_keys):
        for j, (c_cat, c_elem) in enumerate(state.col_keys):
            value = float(state.matrix[i, j])
            if abs(value) <= zero_threshold:
                continue
            records.append(([r_cat, r_elem, c_cat, c_elem], value))

    param = Parameter(
        name=symbol_name,
        dimensions=4,
        domain=["*", "*", "*", "*"],
        records=records,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_gdx(str(output_path), [param])
    return len(records)


def write_state(
    state: SAMTransformState,
    output_path: Path,
    output_format: str,
    output_symbol: str,
) -> dict[str, Any]:
    if output_format == "excel":
        _write_state_to_excel(state, output_path)
        return {"format": "excel"}

    if output_format == "gdx":
        records = _write_state_to_gdx(state, output_path, output_symbol)
        return {"format": "gdx", "records": records, "symbol": output_symbol}

    raise ValueError(f"Unsupported output format: {output_format}")
