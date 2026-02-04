"""PEP data loaders and utilities.

This module provides data loading functionality for PEP model data,
including SAM and parameter files.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from equilibria.babel import SAM


def load_pep_sam(filepath: Path | str) -> SAM:
    """Load PEP SAM from Excel file.

    Args:
        filepath: Path to SAM Excel file (e.g., SAM-V2_0.xls)

    Returns:
        SAM object
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"SAM file not found: {filepath}")

    # Load from Excel - PEP SAM has header rows, skip them
    # First read to find the actual data start
    df_raw = pd.read_excel(filepath, header=None)

    # Find where actual data starts (look for 'L' or 'K' or 'AG' in first column)
    data_start_row = 0
    for i, row in df_raw.iterrows():
        first_val = str(row[0]).strip() if pd.notna(row[0]) else ""
        if first_val in ["L", "K", "AG", "J", "I", "MARG", "OTH"]:
            data_start_row = i
            break

    # Find where actual data columns start
    data_start_col = 0
    for j in range(len(df_raw.columns)):
        first_val = (
            str(df_raw.iloc[data_start_row, j]).strip()
            if pd.notna(df_raw.iloc[data_start_row, j])
            else ""
        )
        if first_val in ["L", "K", "AG", "J", "I", "MARG", "OTH"]:
            data_start_col = j
            break

    # Extract the data matrix
    # Get row labels (account names) from first column
    row_labels = []
    for i in range(data_start_row, len(df_raw)):
        label = (
            df_raw.iloc[i, data_start_col - 1]
            if data_start_col > 0
            else df_raw.iloc[i, 0]
        )
        if pd.notna(label):
            row_labels.append(str(label).strip())
        else:
            row_labels.append(f"ROW_{i}")

    # Get column labels (account names) from first row
    col_labels = []
    for j in range(data_start_col, len(df_raw.columns)):
        label = (
            df_raw.iloc[data_start_row - 1, j]
            if data_start_row > 0
            else df_raw.iloc[0, j]
        )
        if pd.notna(label):
            col_labels.append(str(label).strip())
        else:
            col_labels.append(f"COL_{j}")

    # Extract the numeric data
    data = df_raw.iloc[data_start_row:, data_start_col:].values

    # Convert to numeric, replacing non-numeric with 0
    import numpy as np

    data = (
        pd.to_numeric(pd.DataFrame(data).stack(), errors="coerce")
        .unstack()
        .fillna(0)
        .values
    )

    # Create DataFrame
    df = pd.DataFrame(data, index=row_labels, columns=col_labels)

    # Ensure square matrix
    all_accounts = list(set(row_labels) | set(col_labels))
    df = df.reindex(index=all_accounts, columns=all_accounts, fill_value=0)

    # Create SAM
    return SAM.from_dataframe(df, name=filepath.stem)


def load_pep_parameters(filepath: Path | str) -> dict[str, Any]:
    """Load PEP parameters from Excel file.

    Args:
        filepath: Path to VAL_PAR Excel file

    Returns:
        Dictionary of parameter name -> values
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Parameter file not found: {filepath}")

    # Load all sheets
    excel = pd.ExcelFile(filepath)
    parameters = {}

    for sheet_name in excel.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        parameters[sheet_name] = df

    return parameters


def get_default_pep_data_dir() -> Path:
    """Get the default PEP data directory."""
    return Path(__file__).parent


def load_default_pep_sam() -> SAM:
    """Load the default PEP SAM (SAM-V2_0)."""
    data_dir = get_default_pep_data_dir()
    sam_path = data_dir / "SAM-V2_0.xls"
    return load_pep_sam(sam_path)


def load_default_pep_parameters() -> dict[str, Any]:
    """Load the default PEP parameters (VAL_PAR)."""
    data_dir = get_default_pep_data_dir()
    param_path = data_dir / "VAL_PAR.xlsx"
    return load_pep_parameters(param_path)
