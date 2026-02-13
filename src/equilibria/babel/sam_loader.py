"""Enhanced SAM loader with GDXXRW-style dimension handling.

This module provides a SAM loader that supports:
- Rdim/Cdim parameters (like GDXXRW)
- Sparse and dense storage
- Both 2D matrix and 4D record formats
- Configurable separators for dimension flattening
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class SAM4D(BaseModel):
    """SAM with 4D multi-dimensional support.

    Stores SAM data in both 2D matrix format (for analysis) and
    4D record format (for GAMS compatibility).

    Attributes:
        matrix: 2D DataFrame with flattened element names
        records: List of (keys, value) tuples for 4D access
        row_categories: Set of category names for rows
        col_categories: Set of category names for columns
        row_elements: Dict mapping category -> list of elements
        col_elements: Dict mapping category -> list of elements
        rdim: Number of row dimensions (1-3)
        cdim: Number of column dimensions (1-3)
        separator: String used to join dimension names
        sparse: Whether sparse storage is used
    """

    model_config = {"arbitrary_types_allowed": True}

    # Core data storage
    matrix: pd.DataFrame = Field(..., description="2D matrix view")
    records: list[tuple[list[str], float]] = Field(
        default_factory=list, description="4D records: ([cat1,elem1,cat2,elem2], value)"
    )

    # Dimension configuration
    rdim: int = Field(2, ge=1, le=3, description="Row dimensions")
    cdim: int = Field(2, ge=1, le=3, description="Column dimensions")
    separator: str = Field("_", description="Separator for dimension names")
    sparse: bool = Field(True, description="Sparse storage mode")

    # Element tracking
    row_categories: set[str] = Field(default_factory=set)
    col_categories: set[str] = Field(default_factory=set)
    row_elements: dict[str, list[str]] = Field(default_factory=dict)
    col_elements: dict[str, list[str]] = Field(default_factory=dict)

    # Metadata
    filepath: str = Field("", description="Source file path")
    name: str = Field("SAM", description="SAM name")

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, v: pd.DataFrame) -> pd.DataFrame:
        """Ensure matrix is a DataFrame."""
        if not isinstance(v, pd.DataFrame):
            raise ValueError("matrix must be a pandas DataFrame")
        return v

    def to_2d(self) -> pd.DataFrame:
        """Get 2D matrix view."""
        return self.matrix

    def to_gdx_records(self) -> list[tuple[list[str], float]]:
        """Get records in GDX format.

        Returns list of ([dim1, dim2, dim3, dim4], value) tuples.
        """
        return self.records

    def get_value(self, *keys: str) -> float | None:
        """Get value by full dimension keys.

        Args:
            *keys: Full dimension keys (rdim + cdim keys)

        Returns:
            Value if found, None otherwise
        """
        if len(keys) != self.rdim + self.cdim:
            raise ValueError(f"Expected {self.rdim + self.cdim} keys, got {len(keys)}")

        key_list = list(keys)
        for record_keys, value in self.records:
            if record_keys == key_list:
                return value
        return None

    def get_submatrix(
        self,
        row_cat: str | None = None,
        col_cat: str | None = None,
    ) -> pd.DataFrame:
        """Extract submatrix by category.

        Args:
            row_cat: Filter rows by this category
            col_cat: Filter columns by this category

        Returns:
            Filtered DataFrame
        """
        result = self.matrix.copy()

        if row_cat and row_cat in self.row_elements:
            # Filter rows
            row_names = self.row_elements[row_cat]
            mask = result.index.isin(row_names)
            result = result[mask]

        if col_cat and col_cat in self.col_elements:
            # Filter columns
            col_names = self.col_elements[col_cat]
            mask = result.columns.isin(col_names)
            result = result.loc[:, mask]

        return result

    @property
    def shape(self) -> tuple[int, int]:
        """Get matrix shape."""
        return self.matrix.shape

    @property
    def non_zero_count(self) -> int:
        """Count non-zero values."""
        return (self.matrix != 0).sum().sum()

    def save_to_gdx(self, output_path: str | Path) -> None:
        """Save SAM to GDX file.

        Args:
            output_path: Path for output GDX file
        """
        from equilibria.babel.gdx.writer import write_gdx
        from equilibria.babel.gdx.symbols import Parameter

        # Create parameter with correct dimensions
        domains = ["*"] * (self.rdim + self.cdim)

        param = Parameter(
            name="SAM",
            dimensions=self.rdim + self.cdim,
            domain=domains,
            records=self.records,
        )

        write_gdx(str(output_path), [param])


class SAM4DLoader:
    """GDXXRW-style SAM loader with multi-dimensional support.

    Supports Rdim/Cdim parameters like GDXXRW for controlling
    how many dimensions come from row/column headers.
    """

    def __init__(
        self,
        rdim: int = 2,
        cdim: int = 2,
        sparse: bool = True,
        separator: str = "_",
        zero_threshold: float = 1e-10,
        unique_elements: bool = True,
    ):
        """Initialize loader.

        Args:
            rdim: Number of row dimensions (1-3)
            cdim: Number of column dimensions (1-3)
            sparse: Store only non-zero values
            separator: Separator for flattening dimensions
            zero_threshold: Values below this are treated as zero
            unique_elements: If True, only keep first occurrence of duplicate
                           element names. If False, include all occurrences.
        """
        self.rdim = rdim
        self.cdim = cdim
        self.sparse = sparse
        self.separator = separator
        self.zero_threshold = zero_threshold
        self.unique_elements = unique_elements

    def load(
        self,
        filepath: str | Path,
        sheet_name: str = "SAM",
        data_start_row: int | None = None,
        data_start_col: int | None = None,
        **pandas_kwargs: Any,
    ) -> SAM4D:
        """Load SAM from Excel file.

        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name (default: "SAM")
            data_start_row: First data row (auto-detect if None)
            data_start_col: First data column (auto-detect if None)
            **pandas_kwargs: Additional arguments for pd.read_excel

        Returns:
            SAM4D object with both 2D and 4D formats
        """
        filepath = Path(filepath)

        # Read raw Excel data
        df = self._read_excel(filepath, sheet_name, **pandas_kwargs)

        # Detect boundaries if not specified
        if data_start_row is None or data_start_col is None:
            data_start_row, data_start_col = self._detect_boundaries(df)

        # Extract multi-dimensional indices
        row_data = self._extract_row_indices(df, data_start_row, data_start_col)
        col_data = self._extract_col_indices(df, data_start_row, data_start_col)

        # Extract data matrix
        data_matrix = self._extract_data(
            df,
            data_start_row,
            data_start_col,
            len(row_data["tuples"]),
            len(col_data["tuples"]),
        )

        # Build 2D matrix
        matrix = self._build_2d_matrix(
            data_matrix,
            row_data["flat_names"],
            col_data["flat_names"],
        )

        # Build 4D records
        records = self._build_4d_records(
            data_matrix,
            row_data["tuples"],
            col_data["tuples"],
        )

        # Create SAM4D object
        return SAM4D(
            matrix=matrix,
            records=records,
            rdim=self.rdim,
            cdim=self.cdim,
            separator=self.separator,
            sparse=self.sparse,
            row_categories=set(row_data["categories"].values()),
            col_categories=set(col_data["categories"].values()),
            row_elements=row_data["by_category"],
            col_elements=col_data["by_category"],
            filepath=str(filepath),
        )

    def _read_excel(
        self,
        filepath: Path,
        sheet_name: str,
        **pandas_kwargs: Any,
    ) -> pd.DataFrame:
        """Read Excel file without headers."""
        return pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            header=None,
            **pandas_kwargs,
        )

    def _detect_boundaries(self, df: pd.DataFrame) -> tuple[int, int]:
        """Detect where data starts in the Excel file.

        Returns:
            (data_start_row, data_start_col)
        """
        # Find first row with 'L' (Labor category marker)
        data_start_row = 0
        for i in range(len(df)):
            first_val = str(df.iloc[i, 0]).strip() if pd.notna(df.iloc[i, 0]) else ""
            if first_val == "L":
                data_start_row = i
                break

        # Data starts after header rows (cdim rows for column headers)
        data_start_row = max(data_start_row, self.cdim)

        # Data columns start immediately after row index dimensions.
        # For PEP (rdim=2), row headers are in columns 0..1 and data starts at 2.
        data_start_col = self.rdim

        return data_start_row, data_start_col

    def _extract_row_indices(
        self,
        df: pd.DataFrame,
        data_start_row: int,
        data_start_col: int,
    ) -> dict:
        """Extract row indices with multi-dimensional structure.

        Returns dict with:
            - tuples: List of tuples (category, element, ...)
            - flat_names: List of flattened names
            - categories: Dict of position -> category
            - by_category: Dict of category -> list of elements
        """
        tuples = []
        flat_names = []
        categories = {}
        by_category: dict[str, list[str]] = {}
        seen_elements: set[str] = set()  # Track seen elements for deduplication

        for i in range(data_start_row, len(df)):
            # Extract row tuple (category, element1, element2, ...)
            row_tuple = []
            category = ""

            for dim in range(self.rdim):
                col_idx = dim
                if col_idx < len(df.columns):
                    val = (
                        str(df.iloc[i, col_idx]).strip()
                        if pd.notna(df.iloc[i, col_idx])
                        else ""
                    )
                    if dim == 0:
                        category = val
                    row_tuple.append(val)

            if row_tuple and any(row_tuple):  # Skip empty rows
                # Get the element name (last item in tuple)
                element_name = row_tuple[-1] if row_tuple else ""

                # Check for duplicates if unique_elements is enabled
                if self.unique_elements and element_name:
                    if element_name in seen_elements:
                        # Skip this duplicate row
                        continue
                    seen_elements.add(element_name)

                categories[len(tuples)] = category
                tuples.append(tuple(row_tuple))

                # Build flat name
                flat_name = self.separator.join(x for x in row_tuple if x and x != "*")
                flat_names.append(flat_name)

                # Track by category
                if category:
                    if category not in by_category:
                        by_category[category] = []
                    if element_name and element_name not in by_category[category]:
                        by_category[category].append(element_name)

        return {
            "tuples": tuples,
            "flat_names": flat_names,
            "categories": categories,
            "by_category": by_category,
        }

    def _extract_col_indices(
        self,
        df: pd.DataFrame,
        data_start_row: int,
        data_start_col: int,
    ) -> dict:
        """Extract column indices with multi-dimensional structure."""
        tuples = []
        flat_names = []
        categories = {}
        by_category: dict[str, list[str]] = {}
        seen_elements: set[str] = set()  # Track seen elements for deduplication

        # Column headers are in rows before data_start_row
        for j in range(data_start_col, len(df.columns)):
            # Extract column tuple from header rows
            col_tuple = []
            category = ""

            for dim in range(self.cdim):
                # Header rows are just before data_start_row
                # For cdim=2: look at rows (data_start_row-2) and (data_start_row-1)
                row_idx = data_start_row - self.cdim + dim
                if row_idx >= 0 and j < len(df.columns):
                    val = (
                        str(df.iloc[row_idx, j]).strip()
                        if pd.notna(df.iloc[row_idx, j])
                        else ""
                    )
                    if dim == 0:
                        category = val
                    col_tuple.append(val)

            if col_tuple and any(col_tuple):
                # Get the element name (last item in tuple)
                element_name = col_tuple[-1] if col_tuple else ""

                # Check for duplicates if unique_elements is enabled
                if self.unique_elements and element_name:
                    if element_name in seen_elements:
                        # Skip this duplicate
                        continue
                    seen_elements.add(element_name)

                categories[len(tuples)] = category
                tuples.append(tuple(col_tuple))

                # Build flat name
                flat_name = self.separator.join(x for x in col_tuple if x and x != "*")
                flat_names.append(flat_name)

                # Track by category
                if category:
                    if category not in by_category:
                        by_category[category] = []
                    if element_name and element_name not in by_category[category]:
                        by_category[category].append(element_name)

        return {
            "tuples": tuples,
            "flat_names": flat_names,
            "categories": categories,
            "by_category": by_category,
        }

    def _extract_data(
        self,
        df: pd.DataFrame,
        data_start_row: int,
        data_start_col: int,
        n_rows: int,
        n_cols: int,
    ) -> np.ndarray:
        """Extract data matrix from Excel."""
        end_row = min(data_start_row + n_rows, len(df))
        end_col = min(data_start_col + n_cols, len(df.columns))

        data = df.iloc[data_start_row:end_row, data_start_col:end_col]
        return data.fillna(0).values

    def _build_2d_matrix(
        self,
        data: np.ndarray,
        row_names: list[str],
        col_names: list[str],
    ) -> pd.DataFrame:
        """Build 2D DataFrame from data matrix."""
        return pd.DataFrame(
            data,
            index=row_names[: data.shape[0]],
            columns=col_names[: data.shape[1]],
        )

    def _build_4d_records(
        self,
        data: np.ndarray,
        row_tuples: list[tuple],
        col_tuples: list[tuple],
    ) -> list[tuple[list[str], float]]:
        """Build 4D records from data matrix."""
        records = []

        for i in range(min(data.shape[0], len(row_tuples))):
            for j in range(min(data.shape[1], len(col_tuples))):
                value = float(data[i, j])

                # Skip if sparse and value is near zero
                if self.sparse and abs(value) < self.zero_threshold:
                    continue

                # Build 4D key: (row_cat, row_elem, col_cat, col_elem)
                row_tuple = row_tuples[i]
                col_tuple = col_tuples[j]

                # Pad to match rdim and cdim
                row_padded = list(row_tuple) + ["*"] * (self.rdim - len(row_tuple))
                col_padded = list(col_tuple) + ["*"] * (self.cdim - len(col_tuple))

                keys = row_padded + col_padded
                records.append((keys, value))

        return records


# Convenience function
def load_sam_4d(
    filepath: str | Path,
    rdim: int = 2,
    cdim: int = 2,
    sparse: bool = True,
    separator: str = "_",
    unique_elements: bool = True,
    **kwargs: Any,
) -> SAM4D:
    """Load SAM with 4D multi-dimensional support.

    This is a convenience function that creates a SAM4DLoader
    and loads the SAM file.

    Args:
        filepath: Path to Excel file
        rdim: Number of row dimensions (default: 2)
        cdim: Number of column dimensions (default: 2)
        sparse: Store only non-zero values (default: True)
        separator: Separator for dimension names (default: "_")
        unique_elements: If True, only keep first occurrence of duplicate
                        element names. If False, include all duplicates.
        **kwargs: Additional arguments for loader

    Returns:
        SAM4D object

    Example:
        >>> # Default: unique elements (191 records for PEP)
        >>> sam = load_sam_4d("SAM-V2_0.xls", rdim=2, cdim=2)
        >>>
        >>> # Include duplicates (196 records for PEP, like cge_babel)
        >>> sam = load_sam_4d("SAM-V2_0.xls", rdim=2, cdim=2, unique_elements=False)
    """
    loader = SAM4DLoader(
        rdim=rdim,
        cdim=cdim,
        sparse=sparse,
        separator=separator,
        unique_elements=unique_elements,
    )
    return loader.load(filepath, **kwargs)
