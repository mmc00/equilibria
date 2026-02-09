"""PEP data loaders and utilities.

This module provides data loading functionality for PEP model data,
including SAM and parameter files with multi-dimensional support.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from equilibria.babel import SAM, SAM4D
from equilibria.babel.sam_loader import load_sam_4d


def load_pep_sam(
    filepath: Path | str | None = None,
    rdim: int = 2,
    cdim: int = 2,
    sparse: bool = True,
    separator: str = "_",
    unique_elements: bool = True,
    **kwargs: Any,
) -> SAM4D:
    """Load PEP SAM from Excel file with multi-dimensional support.

    This loader supports GDXXRW-style dimension specification with
    Rdim and Cdim parameters for controlling multi-dimensional indexing.

    Args:
        filepath: Path to SAM Excel file (default: SAM-V2_0.xls in package)
        rdim: Number of row dimensions (default: 2 for PEP)
            - rdim=1: Single row index
            - rdim=2: Category + element (PEP standard)
            - rdim=3: Category + subcategory + element
        cdim: Number of column dimensions (default: 2 for PEP)
            - cdim=1: Single column index
            - cdim=2: Category + element (PEP standard)
            - cdim=3: Category + subcategory + element
        sparse: If True, store only non-zero values (default: True)
        separator: Separator for flattening dimension names (default: "_")
        unique_elements: If True (default), only keep first occurrence of
                        duplicate element names. Results in 191 records for PEP.
                        If False, include all duplicates. Results in 196 records
                        (matching cge_babel behavior).
        **kwargs: Additional arguments for SAM loader

    Returns:
        SAM4D object with both 2D matrix and 4D record access

    Example:
        >>> # Load standard PEP SAM (191 records, unique elements)
        >>> sam = load_pep_sam("SAM-V2_0.xls")
        >>>
        >>> # Load with duplicates included (196 records, like cge_babel)
        >>> sam = load_pep_sam("SAM-V2_0.xls", unique_elements=False)
        >>>
        >>> # Access 2D matrix
        >>> matrix = sam.matrix
        >>>
        >>> # Access 4D records for GAMS
        >>> records = sam.to_gdx_records()
        >>>
        >>> # Get specific value
        >>> value = sam.get_value("L", "USK", "AG", "HRP")

        >>> # Load with custom settings
        >>> sam = load_pep_sam(
        ...     "SAM-V2_0.xls",
        ...     rdim=2,
        ...     cdim=2,
        ...     sparse=True,
        ...     separator="-",
        ... )
    """
    if filepath is None:
        # Use default SAM file from package
        filepath = Path(__file__).parent / "SAM-V2_0.xls"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"SAM file not found: {filepath}")

    # Load using the enhanced 4D loader
    return load_sam_4d(
        filepath=filepath,
        rdim=rdim,
        cdim=cdim,
        sparse=sparse,
        separator=separator,
        unique_elements=unique_elements,
        **kwargs,
    )


def load_default_pep_sam(
    rdim: int = 2,
    cdim: int = 2,
    sparse: bool = True,
    separator: str = "_",
    unique_elements: bool = True,
) -> SAM4D:
    """Load default PEP SAM (SAM-V2_0.xls) from package.

    Convenience function to load the standard PEP SAM file included
    with the package.

    Args:
        rdim: Number of row dimensions (default: 2)
        cdim: Number of column dimensions (default: 2)
        sparse: Store only non-zero values (default: True)
        separator: Separator for dimension names (default: "_")
        unique_elements: If True (default), keep only unique elements (191 records).
                        If False, include duplicates like cge_babel (191-196 records).

    Returns:
        SAM4D object

    Example:
        >>> # Load with unique elements (default, 191 records)
        >>> sam = load_default_pep_sam()
        >>> print(f"SAM shape: {sam.shape}")
        >>> print(f"Non-zero values: {sam.non_zero_count}")
        >>>
        >>> # Load with duplicates included (like cge_babel)
        >>> sam = load_default_pep_sam(unique_elements=False)
    """
    filepath = Path(__file__).parent / "SAM-V2_0.xls"
    return load_pep_sam(
        filepath=filepath,
        rdim=rdim,
        cdim=cdim,
        sparse=sparse,
        separator=separator,
        unique_elements=unique_elements,
    )


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


def get_sam_indices_mapping(filepath: Path | str | None = None) -> dict[str, list[str]]:
    """Extract dimension indices mapping from SAM Excel file.

    This function extracts the ordered element lists for each dimension
    from the SAM Excel file, which can be used to remap GDX indices.

    The mapping follows GDXXRW format with Rdim=2 Cdim=2:
    - dim1: Row dimension 1 (categories: L, K, AG, J, I, X, OTH)
    - dim2: Row dimension 2 (elements: USK, SK, CAP, LAND, etc.)
    - dim3: Column dimension 1 (categories: L, K, AG, J, I, X, OTH)
    - dim4: Column dimension 2 (elements: USK, SK, CAP, LAND, etc.)

    Args:
        filepath: Path to SAM Excel file (default: SAM-V2_0.xls in package)

    Returns:
        Dictionary mapping dimension names to ordered element lists.
        Example:
        {
            "dim1": ["L", "K", "AG", "J", "I", "X", "OTH"],
            "dim2": ["USK", "SK", "CAP", "LAND", "HRP", ...],
            "dim3": ["L", "K", "AG", "J", "I", "X", "OTH"],
            "dim4": ["USK", "SK", "CAP", "LAND", "HRP", ...],
        }

    Example:
        >>> mapping = get_sam_indices_mapping("SAM-V2_0.xls")
        >>> print(mapping["dim1"])
        ['L', 'K', 'AG', 'J', 'I', 'X', 'OTH']
        >>>
        >>> # Use with GDX reader
        >>> gdx_data = read_gdx("SAM-V2_0.gdx")
        >>> sam = read_parameter_values(
        ...     gdx_data, "SAM",
        ...     rdim=2, cdim=2,
        ...     indices_mapping=mapping
        ... )
    """
    if filepath is None:
        filepath = Path(__file__).parent / "SAM-V2_0.xls"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"SAM file not found: {filepath}")

    # Read Excel
    df = pd.read_excel(filepath, header=None)

    # Extract dimensions based on GDXXRW Rdim=2 Cdim=2
    # Row dim 1: row 3 (index 3), starting from column 2
    row_dim1 = []
    for col in range(2, len(df.columns)):
        val = df.iloc[3, col]
        if pd.notna(val) and val not in row_dim1:
            row_dim1.append(str(val))

    # Row dim 2: row 4 (index 4), starting from column 2
    row_dim2 = []
    for col in range(2, len(df.columns)):
        val = df.iloc[4, col]
        if pd.notna(val) and val not in row_dim2:
            row_dim2.append(str(val))

    # Col dim 1: column 0, starting from row 5
    col_dim1 = []
    for row in range(5, len(df)):
        val = df.iloc[row, 0]
        if pd.notna(val) and val not in col_dim1:
            col_dim1.append(str(val))

    # Col dim 2: column 1, starting from row 5
    col_dim2 = []
    for row in range(5, len(df)):
        val = df.iloc[row, 1]
        if pd.notna(val) and val not in col_dim2:
            col_dim2.append(str(val))

    return {
        "dim1": row_dim1,
        "dim2": row_dim2,
        "dim3": col_dim1,
        "dim4": col_dim2,
    }


def get_default_pep_data_dir() -> Path:
    """Get default PEP data directory.
    
    Returns:
        Path to default PEP data directory
    """
    return Path(__file__).parent


# Backward compatibility - but now returns SAM4D instead of SAM
__all__ = [
    "load_pep_sam",
    "load_default_pep_sam",
    "load_pep_parameters",
    "get_sam_indices_mapping",
    "get_default_pep_data_dir",
]
