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

    # Load from Excel
    df = pd.read_excel(filepath, index_col=0, header=0)

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
