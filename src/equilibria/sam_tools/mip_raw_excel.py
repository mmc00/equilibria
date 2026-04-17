"""MIP raw SAM support for converting Input-Output tables to SAM format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equilibria.sam_tools.aggregation import build_multiindex_labels
from equilibria.sam_tools.models import Sam, SamTable


def _norm_text(value: Any) -> str:
    """Normalize text by stripping and collapsing whitespace."""
    return " ".join(str(value).strip().split())


def _norm_text_lower(value: Any) -> str:
    """Normalize text to lowercase."""
    return _norm_text(value).lower()


def _is_numeric(value: Any) -> bool:
    """Check if value is numeric."""
    return isinstance(value, (int, float, np.integer, np.floating))


def _find_data_start(df: pd.DataFrame) -> tuple[int, int]:
    """
    Find the row and column where the MIP data matrix begins.

    Looks for the first row with multiple numeric values and corresponding labels.
    Returns (data_start_row, data_start_col) where data_start_col is the first
    column with numeric data.
    """
    # Look for first row that has text labels in column 0 and numeric data
    for row_idx in range(min(20, len(df))):
        # Check if there's a label in first column
        label_val = df.iat[row_idx, 0]
        if pd.isna(label_val):
            continue

        # Check if this row has numeric data
        has_numbers = False
        first_num_col = -1

        for col_idx in range(1, min(20, len(df.columns))):
            val = df.iat[row_idx, col_idx]
            if _is_numeric(val) and not pd.isna(val):
                if first_num_col < 0:
                    first_num_col = col_idx
                has_numbers = True
                break

        if has_numbers and first_num_col > 0:
            return row_idx, first_num_col

    raise ValueError("Could not find data start in MIP Excel file")


def _extract_labels_from_mip(
    df: pd.DataFrame,
    *,
    data_start_row: int,
    label_col: int,
    va_row_label: str,
    import_row_label: str,
) -> tuple[list[str], int, int]:
    """
    Extract row labels from MIP table.

    Returns:
        - List of commodity/sector labels
        - Row index of VA (Value Added)
        - Row index of imports
    """
    labels: list[str] = []
    va_row_idx: int | None = None
    import_row_idx: int | None = None

    va_lower = _norm_text_lower(va_row_label)
    import_lower = _norm_text_lower(import_row_label)

    for row_idx in range(data_start_row, len(df)):
        label_value = df.iat[row_idx, label_col]
        if pd.isna(label_value):
            break

        label = _norm_text(label_value)
        label_lower = _norm_text_lower(label)

        # Check for special rows
        if va_lower in label_lower or "valor agregado" in label_lower:
            va_row_idx = row_idx
            continue

        if import_lower in label_lower or "importacion" in label_lower:
            import_row_idx = row_idx
            continue

        # Skip total rows
        if label_lower == "total" or label_lower.startswith("total "):
            continue

        if label:
            labels.append(label)

    if not labels:
        raise ValueError("No commodity/sector labels found in MIP")

    if va_row_idx is None:
        raise ValueError(f"VA row not found (looking for '{va_row_label}')")

    return labels, va_row_idx, import_row_idx or -1


def _extract_final_demand_labels(
    df: pd.DataFrame,
    *,
    data_start_row: int,
    data_start_col: int,
    n_sectors: int,
) -> list[str]:
    """
    Extract final demand column labels (HH, GOV, INV, EXP, etc.).
    """
    header_row = data_start_row - 1
    if header_row < 0:
        return []

    fd_labels: list[str] = []
    start_col = data_start_col + n_sectors

    for col_idx in range(start_col, min(start_col + 10, len(df.columns))):
        if col_idx >= len(df.columns):
            break

        header_value = df.iat[header_row, col_idx]
        if pd.isna(header_value):
            continue

        label = _norm_text(header_value)
        if not label or _norm_text_lower(label) == "total":
            continue

        fd_labels.append(label)

    return fd_labels


def _parse_mip_raw_matrix(
    input_path: Path,
    sheet_name: str,
    *,
    va_row_label: str = "Valor Agregado",
    import_row_label: str = "Importaciones",
) -> tuple[pd.DataFrame, list[str], list[str], int, int]:
    """
    Parse MIP Excel file into raw DataFrame.

    Returns:
        - Raw DataFrame
        - List of commodity/sector labels
        - List of final demand labels
        - VA row index
        - Import row index
    """
    raw_df = pd.read_excel(input_path, sheet_name=sheet_name, header=None)

    # Find where data starts
    data_start_row, data_start_col = _find_data_start(raw_df)
    label_col = data_start_col - 1

    # Extract labels
    sector_labels, va_row_idx, import_row_idx = _extract_labels_from_mip(
        raw_df,
        data_start_row=data_start_row,
        label_col=label_col,
        va_row_label=va_row_label,
        import_row_label=import_row_label,
    )

    fd_labels = _extract_final_demand_labels(
        raw_df,
        data_start_row=data_start_row,
        data_start_col=data_start_col,
        n_sectors=len(sector_labels),
    )

    return raw_df, sector_labels, fd_labels, va_row_idx, import_row_idx


def _build_mip_sam_matrix(
    raw_df: pd.DataFrame,
    sector_labels: list[str],
    fd_labels: list[str],
    va_row_idx: int,
    import_row_idx: int,
    data_start_row: int,
    data_start_col: int,
) -> pd.DataFrame:
    """
    Build SAM matrix from parsed MIP data.

    Structure:
    - Commodities (rows: sector_labels)
    - VA aggregate (row: "VA-aggregate")
    - Imports (row: "IMP-total" if exists)
    - Final demand columns appended
    """
    n_sectors = len(sector_labels)
    n_fd = len(fd_labels)
    has_imports = import_row_idx >= 0

    # All labels: sectors + VA + imports (if exists) + FD
    all_labels = sector_labels.copy()
    all_labels.append("VA-aggregate")
    if has_imports:
        all_labels.append("IMP-total")
    all_labels.extend(fd_labels)

    n_total = len(all_labels)
    matrix = np.zeros((n_total, n_total), dtype=float)

    # Extract intermediate flows (I×J)
    for i, _ in enumerate(sector_labels):
        row_idx = data_start_row + i
        for j, _ in enumerate(sector_labels):
            col_idx = data_start_col + j
            if row_idx < raw_df.shape[0] and col_idx < raw_df.shape[1]:
                value = raw_df.iat[row_idx, col_idx]
                if _is_numeric(value) and not pd.isna(value):
                    matrix[i, j] = float(value)

    # Extract VA row (VA → J)
    va_target_row = n_sectors
    for j, _ in enumerate(sector_labels):
        col_idx = data_start_col + j
        if va_row_idx < raw_df.shape[0] and col_idx < raw_df.shape[1]:
            value = raw_df.iat[va_row_idx, col_idx]
            if _is_numeric(value) and not pd.isna(value):
                matrix[va_target_row, j] = float(value)

    # Extract imports row (IMP → I) if exists
    if has_imports:
        imp_target_row = n_sectors + 1
        for i, _ in enumerate(sector_labels):
            col_idx = data_start_col + i
            if import_row_idx < raw_df.shape[0] and col_idx < raw_df.shape[1]:
                value = raw_df.iat[import_row_idx, col_idx]
                if _is_numeric(value) and not pd.isna(value):
                    matrix[imp_target_row, i] = float(value)

    # Extract final demand (I → FD)
    fd_start_col = n_sectors + 1 + (1 if has_imports else 0)
    for i, _ in enumerate(sector_labels):
        row_idx = data_start_row + i
        for j, _ in enumerate(fd_labels):
            col_idx = data_start_col + n_sectors + j
            if row_idx < raw_df.shape[0] and col_idx < raw_df.shape[1]:
                value = raw_df.iat[row_idx, col_idx]
                if _is_numeric(value) and not pd.isna(value):
                    matrix[i, fd_start_col + j] = float(value)

    return pd.DataFrame(matrix, index=all_labels, columns=all_labels, dtype=float)


class MIPRawSAM(Sam):
    """SAM helper for loading and transforming raw MIP (Input-Output) tables."""

    @classmethod
    def from_mip_excel(
        cls,
        path: Path,
        sheet_name: str = "MIP",
        *,
        va_row_label: str = "Valor Agregado",
        import_row_label: str = "Importaciones",
    ) -> MIPRawSAM:
        """
        Load standard MIP from Excel.

        Expected structure:
        - Rows: Commodities + VA row + Import row (optional)
        - Columns: Sectors + Final demand (HH, GOV, INV, EXP, etc.)

        Args:
            path: Path to Excel file
            sheet_name: Name of sheet containing MIP
            va_row_label: Label to identify Value Added row
            import_row_label: Label to identify imports row

        Returns:
            MIPRawSAM with normalized accounts
        """
        path = Path(path)

        raw_df, sector_labels, fd_labels, va_row_idx, import_row_idx = _parse_mip_raw_matrix(
            path,
            sheet_name,
            va_row_label=va_row_label,
            import_row_label=import_row_label,
        )

        data_start_row, data_start_col = _find_data_start(raw_df)

        matrix_df = _build_mip_sam_matrix(
            raw_df,
            sector_labels,
            fd_labels,
            va_row_idx,
            import_row_idx,
            data_start_row,
            data_start_col,
        )

        # Build multi-index with RAW category
        labels = list(matrix_df.index)
        multi_index, _ = build_multiindex_labels(labels, category="RAW")

        return cls(
            dataframe=pd.DataFrame(
                matrix_df.to_numpy(dtype=float),
                index=multi_index,
                columns=multi_index,
            )
        )

    def to_table(
        self,
        *,
        source_path: Path | None = None,
        source_format: str = "mip_raw_excel",
    ) -> SamTable:
        """Convert to SamTable with metadata."""
        return SamTable(
            sam=self,
            source_path=source_path or Path("<memory>"),
            source_format=source_format,
        )


def load_mip_raw_excel_table(
    input_path: Path,
    *,
    sheet_name: str = "MIP",
    va_row_label: str | None = None,
    import_row_label: str | None = None,
) -> SamTable:
    """Load a MIP raw workbook as ``SamTable``."""
    kwargs: dict[str, Any] = {}
    if va_row_label is not None:
        kwargs["va_row_label"] = va_row_label
    if import_row_label is not None:
        kwargs["import_row_label"] = import_row_label

    sam = MIPRawSAM.from_mip_excel(
        path=input_path,
        sheet_name=sheet_name,
        **kwargs,
    )
    return sam.to_table(source_path=input_path, source_format="mip_raw_excel")
