"""SAM (Social Accounting Matrix) class for CGE models.

The SAM is the data foundation of any CGE model. This module provides
a powerful SAM class that can load from various formats, validate
balance, and extract sets automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class SAM(BaseModel):
    """Social Accounting Matrix for CGE modeling.

    The SAM represents the circular flow of income in an economy,
    organized as a square matrix where rows represent receipts and
    columns represent expenditures.

    Attributes:
        data: DataFrame containing the SAM matrix
        sets: Dictionary of sets extracted from SAM
        name: Name of the SAM
        description: Description of the SAM

    Example:
        >>> sam = SAM.from_excel("data/sam.xlsx")
        >>> sam.validate()  # Check row/column balance
        >>> print(sam.summary())
    """

    data: pd.DataFrame = Field(..., description="SAM matrix data")
    sets: dict[str, list[str]] = Field(
        default_factory=dict, description="Sets from SAM"
    )
    name: str = Field(default="SAM", description="SAM name")
    description: str = Field(default="", description="SAM description")

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_excel(
        cls,
        filepath: str | Path,
        sheet_name: str = "SAM",
        index_col: int = 0,
        header: int = 0,
    ) -> SAM:
        """Load SAM from Excel file.

        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name containing SAM
            index_col: Column to use as index
            header: Row to use as header

        Returns:
            SAM object
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"File not found: {filepath}"
            raise FileNotFoundError(msg)

        df = pd.read_excel(
            filepath, sheet_name=sheet_name, index_col=index_col, header=header
        )

        # Ensure square matrix
        if df.shape[0] != df.shape[1]:
            msg = f"SAM must be square, got {df.shape}"
            raise ValueError(msg)

        # Extract sets from index/columns
        accounts = df.index.tolist()
        sets = {"AC": accounts}  # Accounts set

        return cls(data=df, sets=sets, name=filepath.stem)

    @classmethod
    def from_gdx(cls, filepath: str | Path, symbol_name: str = "SAM") -> SAM:
        """Load SAM from GDX file.

        Args:
            filepath: Path to GDX file
            symbol_name: Name of SAM parameter in GDX

        Returns:
            SAM object
        """
        from equilibria.babel.gdx import read_gdx

        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"File not found: {filepath}"
            raise FileNotFoundError(msg)

        gdx_data = read_gdx(filepath)

        # Find SAM parameter
        sam_symbol = None
        for sym in gdx_data.get("symbols", []):
            if sym.get("name") == symbol_name:
                sam_symbol = sym
                break

        if sam_symbol is None:
            msg = f"Symbol '{symbol_name}' not found in GDX file"
            raise ValueError(msg)

        # Convert to DataFrame (simplified - assumes 2D)
        # In practice, need to handle multi-dimensional parameters
        data = sam_symbol.get("data", [])

        # Extract unique elements for each dimension
        dim1_elements = sorted({row[0] for row in data})
        dim2_elements = sorted({row[1] for row in data})

        # Create matrix
        matrix = np.zeros((len(dim1_elements), len(dim2_elements)))
        for row in data:
            i = dim1_elements.index(row[0])
            j = dim2_elements.index(row[1])
            matrix[i, j] = row[2]

        df = pd.DataFrame(matrix, index=dim1_elements, columns=dim2_elements)

        sets = {
            "AC": dim1_elements,
        }

        return cls(data=df, sets=sets, name=filepath.stem)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = "SAM") -> SAM:
        """Create SAM from DataFrame.

        Args:
            df: DataFrame with SAM data
            name: Name for the SAM

        Returns:
            SAM object
        """
        if df.shape[0] != df.shape[1]:
            msg = f"SAM must be square, got {df.shape}"
            raise ValueError(msg)

        accounts = df.index.tolist()
        sets = {"AC": accounts}

        return cls(data=df, sets=sets, name=name)

    def check_balance(self, tolerance: float = 1e-6) -> dict[str, Any]:
        """Validate SAM row/column balance.

        Args:
            tolerance: Tolerance for balance check

        Returns:
            Validation results dictionary
        """
        row_sums = self.data.sum(axis=1)
        col_sums = self.data.sum(axis=0)
        differences = row_sums - col_sums

        max_diff = differences.abs().max()
        is_balanced = max_diff <= tolerance

        # Find unbalanced accounts
        unbalanced = differences[differences.abs() > tolerance].to_dict()

        return {
            "is_balanced": is_balanced,
            "max_difference": max_diff,
            "tolerance": tolerance,
            "unbalanced_accounts": unbalanced,
            "total_row_sum": row_sums.sum(),
            "total_col_sum": col_sums.sum(),
        }

    def balance(self, method: str = "ras") -> SAM:
        """Balance the SAM using RAS method.

        Args:
            method: Balancing method ("ras" only for now)

        Returns:
            New balanced SAM
        """
        if method != "ras":
            msg = f"Unknown balancing method: {method}"
            raise ValueError(msg)

        # Simple RAS implementation
        target_rows = self.data.sum(axis=1)
        target_cols = self.data.sum(axis=0)

        # Initialize with current values
        balanced = self.data.copy().values

        # Iterate RAS
        for _ in range(100):  # Max iterations
            # Row scaling
            row_sums = balanced.sum(axis=1)
            row_factors = np.where(row_sums > 0, target_rows / row_sums, 1)
            balanced = balanced * row_factors[:, np.newaxis]

            # Column scaling
            col_sums = balanced.sum(axis=0)
            col_factors = np.where(col_sums > 0, target_cols / col_sums, 1)
            balanced = balanced * col_factors

            # Check convergence
            max_error = max(
                np.abs(balanced.sum(axis=1) - target_rows).max(),
                np.abs(balanced.sum(axis=0) - target_cols).max(),
            )
            if max_error < 1e-10:
                break

        df = pd.DataFrame(balanced, index=self.data.index, columns=self.data.columns)
        return SAM(data=df, sets=self.sets.copy(), name=f"{self.name}_balanced")

    def get_submatrix(self, rows: list[str], cols: list[str]) -> pd.DataFrame:
        """Extract submatrix for specific accounts.

        Args:
            rows: Row account names
            cols: Column account names

        Returns:
            Submatrix DataFrame
        """
        return self.data.loc[rows, cols]

    def get_set(self, name: str) -> list[str] | None:
        """Get a set by name.

        Args:
            name: Set name

        Returns:
            List of elements or None if not found
        """
        return self.sets.get(name)

    def extract_sets(self, mapping: dict[str, list[str]]) -> None:
        """Extract sets from accounts using mapping.

        Args:
            mapping: Dictionary of set name to list of account prefixes

        Example:
            >>> sam.extract_sets({
            ...     "J": ["sec_"],  # Sectors
            ...     "I": ["fac_"],  # Factors
            ...     "H": ["hh_"],   # Households
            ... })
        """
        accounts = self.data.index.tolist()

        for set_name, prefixes in mapping.items():
            elements = []
            for prefix in prefixes:
                elements.extend([a for a in accounts if a.startswith(prefix)])
            self.sets[set_name] = elements

    def summary(self) -> dict[str, Any]:
        """Return summary statistics of the SAM."""
        validation = self.check_balance()

        return {
            "name": self.name,
            "description": self.description,
            "shape": self.data.shape,
            "accounts": len(self.data),
            "total_value": self.data.sum().sum(),
            "is_balanced": validation["is_balanced"],
            "max_difference": validation["max_difference"],
            "sets": {k: len(v) for k, v in self.sets.items()},
        }

    def to_excel(self, filepath: str | Path, sheet_name: str = "SAM") -> None:
        """Export SAM to Excel.

        Args:
            filepath: Output file path
            sheet_name: Sheet name
        """
        filepath = Path(filepath)
        self.data.to_excel(filepath, sheet_name=sheet_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert SAM to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "data": self.data.to_dict(),
            "sets": self.sets,
        }

    def __repr__(self) -> str:
        """String representation."""
        shape = self.data.shape
        sets_info = ", ".join(f"{k}({len(v)})" for k, v in self.sets.items())
        return f"SAM '{self.name}': {shape[0]}x{shape[1]}, sets: {sets_info}"
