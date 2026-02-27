"""Input/output primitives for SAM table objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.babel.gdx.symbols import Parameter
from equilibria.babel.gdx.writer import write_gdx
from equilibria.sam_tools.enums import SAMFormat
from equilibria.sam_tools.ieem_raw_excel import IEEMRawSAM, load_ieem_raw_excel_table
from equilibria.sam_tools.models import Sam, SamTable
from equilibria.sam_tools.selectors import norm_text
from equilibria.templates.pep_sam_compat import load_sam_grid

# Keep module attribute for test monkeypatch usage.
_ = IEEMRawSAM


def _normalize_format(fmt: str | SAMFormat) -> SAMFormat:
    """Normalize any format token into the canonical ``SAMFormat`` enum."""
    if isinstance(fmt, SAMFormat):
        return fmt
    return SAMFormat.from_alias(fmt)


def _load_from_excel(path: Path) -> SamTable:
    """Load a canonical Excel SAM grid into a ``SamTable``."""
    grid = load_sam_grid(path)
    sam = Sam.from_matrix(grid.matrix.copy().astype(float), grid.row_keys, grid.col_keys)
    return SamTable(
        sam=sam,
        source_path=path,
        source_format=SAMFormat.EXCEL.value,
        raw_df=grid.raw_df,
        data_start_row=grid.data_start_row,
        data_start_col=grid.data_start_col,
    )


def _load_from_gdx(path: Path) -> SamTable:
    """Load SAM values from GDX and rebuild the 2D matrix support."""
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

    sam = Sam.from_matrix(matrix, row_order, col_order)
    return SamTable(
        sam=sam,
        source_path=path,
        source_format=SAMFormat.GDX.value,
    )


def load_table(
    path: Path,
    fmt: str | SAMFormat,
    options: dict[str, Any] | None = None,
) -> SamTable:
    """Load one SAM table from file according to the selected format."""
    if not path.exists():
        raise FileNotFoundError(f"Input SAM not found: {path}")

    try:
        format_enum = _normalize_format(fmt)
    except ValueError as exc:
        raise ValueError(f"Unsupported input format: {fmt}") from exc
    opts = options or {}

    if format_enum == SAMFormat.EXCEL:
        return _load_from_excel(path)
    if format_enum == SAMFormat.GDX:
        return _load_from_gdx(path)
    if format_enum == SAMFormat.IEEM_RAW_EXCEL:
        return load_ieem_raw_excel_table(
            input_path=path,
            sheet_name=str(opts.get("sheet_name", "MCS2016")),
            group_order=tuple(opts["group_order"]) if isinstance(opts.get("group_order"), (list, tuple)) else None,
            group_aliases=opts.get("group_aliases") if isinstance(opts.get("group_aliases"), dict) else None,
            group_col=int(opts["group_col"]) if "group_col" in opts else None,
            label_col=int(opts["label_col"]) if "label_col" in opts else None,
            data_start_col=int(opts["data_start_col"]) if "data_start_col" in opts else None,
        )

    raise ValueError(f"Unsupported input format: {fmt}")


def _write_excel_preserving_layout(table: SamTable, output_path: Path) -> None:
    """Write table values into the original Excel layout when available."""
    if table.raw_df is None or table.data_start_row is None or table.data_start_col is None:
        raise ValueError("No original Excel layout available")

    out_df = table.raw_df.copy()
    n_rows, n_cols = table.matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            out_df.iat[table.data_start_row + i, table.data_start_col + j] = float(
                table.matrix[i, j]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="SAM", index=False, header=False)


def _write_excel_canonical(table: SamTable, output_path: Path) -> None:
    """Write table in canonical two-header SAM layout."""
    n_rows, n_cols = table.matrix.shape
    grid = np.full((n_rows + 2, n_cols + 2), "", dtype=object)

    for j, (cat, elem) in enumerate(table.col_keys):
        grid[0, 2 + j] = cat
        grid[1, 2 + j] = elem

    for i, (cat, elem) in enumerate(table.row_keys):
        grid[2 + i, 0] = cat
        grid[2 + i, 1] = elem
        for j in range(n_cols):
            grid[2 + i, 2 + j] = float(table.matrix[i, j])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(grid).to_excel(writer, sheet_name="SAM", index=False, header=False)


def _write_table_to_excel(table: SamTable, output_path: Path) -> None:
    """Dispatch Excel writer based on whether original layout metadata exists."""
    if (
        table.raw_df is not None
        and table.data_start_row is not None
        and table.data_start_col is not None
        and table.data_start_row + table.matrix.shape[0] <= table.raw_df.shape[0]
        and table.data_start_col + table.matrix.shape[1] <= table.raw_df.shape[1]
    ):
        _write_excel_preserving_layout(table, output_path)
    else:
        _write_excel_canonical(table, output_path)


def _write_table_to_gdx(
    table: SamTable,
    output_path: Path,
    symbol_name: str,
    zero_threshold: float = 1e-14,
) -> int:
    """Serialize table matrix as a 4D ``SAM`` parameter in GDX."""
    records: list[tuple[list[str], float]] = []
    for i, (r_cat, r_elem) in enumerate(table.row_keys):
        for j, (c_cat, c_elem) in enumerate(table.col_keys):
            value = float(table.matrix[i, j])
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


def write_table(
    table: SamTable,
    output_path: Path,
    output_format: str | SAMFormat,
    output_symbol: str,
) -> dict[str, Any]:
    """Write one SAM table to disk in Excel or GDX format."""
    try:
        format_enum = _normalize_format(output_format)
    except ValueError as exc:
        raise ValueError(f"Unsupported output format: {output_format}") from exc
    if format_enum == SAMFormat.EXCEL:
        _write_table_to_excel(table, output_path)
        return {"format": SAMFormat.EXCEL.value}

    if format_enum == SAMFormat.GDX:
        records = _write_table_to_gdx(table, output_path, output_symbol)
        return {"format": SAMFormat.GDX.value, "records": records, "symbol": output_symbol}

    raise ValueError(f"Unsupported output format: {output_format}")
