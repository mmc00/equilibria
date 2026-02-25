"""Calibration data extraction and caching for equilibria CGE framework.

This module provides the CalibrationData class for extracting data from SAM,
caching results, and managing dummy data generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import re

if TYPE_CHECKING:
    from equilibria.babel import SAM
    from equilibria.core.parameters import Parameter
    from equilibria.core.sets import Set, SetManager


class CalibrationError(Exception):
    """Raised when calibration data extraction fails."""

    pass


class SAMDataMissingError(CalibrationError):
    """Raised when required SAM data is missing."""

    pass


class CalibrationData:
    """Extracts, caches, and provides calibration data from SAM.

    This class manages the extraction of data from the Social Accounting
    Matrix (SAM) for use in calibrating CGE model parameters. It caches
    extracted matrices to avoid repeated processing and provides a uniform
    interface for both SAM and dummy data modes.

    The SAM is organized with:
    - Rows = sources (who receives payment)
    - Columns = uses (who makes payment)

    Example:
        >>> sam = load_sam("SAM-V2_0.xls")
        >>> data = CalibrationData(sam, mode="sam")
        >>> factor_payments = data.get_matrix("F", "J")
        >>> # Returns matrix where element [f, j] is payment from sector j to factor f

    Attributes:
        sam: The Social Accounting Matrix
        mode: "sam" or "dummy"
        _cache: Cache of extracted matrices
        _params: Storage for calibrated "0" parameters by block
    """

    def __init__(
        self,
        sam: SAM | None,
        mode: str = "sam",
        dummy_value: float = 1.0,
    ):
        """Initialize calibration data manager.

        Args:
            sam: Social Accounting Matrix (None for dummy mode)
            mode: "sam" to use SAM data, "dummy" for uniform values
            dummy_value: Default value for dummy mode (default: 1.0)

        Raises:
            CalibrationError: If mode="sam" but sam is None
        """
        if mode == "sam" and sam is None:
            raise CalibrationError("Mode is 'sam' but no SAM data provided")

        self.sam = sam
        self.mode = mode
        self.dummy_value = dummy_value
        self._cache: dict[str, np.ndarray] = {}
        self._params: dict[str, dict[str, np.ndarray]] = {}
        self._set_mappings: dict[str, list[str]] = {}

    def register_set_mapping(self, set_name: str, sam_accounts: list[str]) -> None:
        """Register mapping from set name to SAM account names.

        This allows sets to be mapped to specific SAM accounts.
        For example, set "F" might map to accounts ["usk", "sk", "cap", "land"].

        Args:
            set_name: Name of the set (e.g., "F", "J", "I")
            sam_accounts: List of SAM account names for this set
        """
        self._set_mappings[set_name] = sam_accounts

    def get_matrix(
        self,
        row_set: str,
        col_set: str,
        row_accounts: list[str] | None = None,
        col_accounts: list[str] | None = None,
    ) -> np.ndarray:
        """Extract a sub-matrix from SAM or create dummy data.

        Args:
            row_set: Name of row set (e.g., "F", "J", "I")
            col_set: Name of column set (e.g., "F", "J", "I")
            row_accounts: Specific SAM accounts for rows (overrides mapping)
            col_accounts: Specific SAM accounts for columns (overrides mapping)

        Returns:
            Matrix of shape (len(row_set), len(col_set))

        Raises:
            SAMDataMissingError: If mode="sam" and data not found in SAM
        """
        cache_key = f"{row_set}_{col_set}"

        if cache_key not in self._cache:
            if self.mode == "sam":
                self._cache[cache_key] = self._extract_from_sam(
                    row_set, col_set, row_accounts, col_accounts
                )
            else:
                self._cache[cache_key] = self._create_dummy(
                    row_set, col_set, row_accounts, col_accounts
                )

        return self._cache[cache_key]

    def _extract_from_sam(
        self,
        row_set: str,
        col_set: str,
        row_accounts: list[str] | None,
        col_accounts: list[str] | None,
    ) -> np.ndarray:
        """Extract sub-matrix from SAM.

        Args:
            row_set: Row set name
            col_set: Column set name
            row_accounts: Specific row accounts
            col_accounts: Specific column accounts

        Returns:
            Extracted matrix

        Raises:
            SAMDataMissingError: If accounts not found in SAM
        """
        if self.sam is None:
            raise SAMDataMissingError("No SAM data available")

        # Get account names
        rows = row_accounts or self._set_mappings.get(row_set, [row_set])
        cols = col_accounts or self._set_mappings.get(col_set, [col_set])

        # Initialize result matrix
        result = np.zeros((len(rows), len(cols)))

        # Extract from SAM
        sam_data = self.sam.data

        for i, row_acc in enumerate(rows):
            for j, col_acc in enumerate(cols):
                try:
                    # Try exact match first
                    if row_acc in sam_data.index and col_acc in sam_data.columns:
                        result[i, j] = sam_data.loc[row_acc, col_acc]
                    else:
                        # Try case-insensitive match
                        row_match = self._find_case_insensitive(row_acc, sam_data.index)
                        col_match = self._find_case_insensitive(
                            col_acc, sam_data.columns
                        )

                        if row_match and col_match:
                            result[i, j] = sam_data.loc[row_match, col_match]
                        else:
                            # Account not found - this is an error in strict mode
                            raise SAMDataMissingError(
                                f"SAM data missing: row '{row_acc}', col '{col_acc}' "
                                f"for matrix {row_set}x{col_set}"
                            )
                except KeyError as e:
                    raise SAMDataMissingError(
                        f"SAM data missing: {e} for matrix {row_set}x{col_set}"
                    )

        return result

    def _find_case_insensitive(self, value: str, candidates: pd.Index) -> str | None:
        """Find a case-insensitive match for value in candidates.

        Args:
            value: The value to search for
            candidates: Index of possible matches

        Returns:
            The matching string from candidates, or None if not found
        """
        value_upper = value.upper()
        for candidate in candidates:
            if str(candidate).upper() == value_upper:
                return candidate

        # Try matching suffix (e.g., J_AGR -> AGR)
        for candidate in candidates:
            candidate_str = str(candidate)
            parts = re.split(r"[._]", candidate_str)
            if parts and parts[-1].upper() == value_upper:
                return candidate

        return None

    def _create_dummy(
        self,
        row_set: str,
        col_set: str,
        row_accounts: list[str] | None,
        col_accounts: list[str] | None,
    ) -> np.ndarray:
        """Create dummy matrix with uniform values.

        Args:
            row_set: Row set name (for determining shape)
            col_set: Column set name (for determining shape)
            row_accounts: Specific row accounts (for shape)
            col_accounts: Specific column accounts (for shape)

        Returns:
            Matrix filled with dummy_value
        """
        rows = row_accounts or self._set_mappings.get(row_set, [row_set])
        cols = col_accounts or self._set_mappings.get(col_set, [col_set])

        n_rows = len(rows) if isinstance(rows, list) else 1
        n_cols = len(cols) if isinstance(cols, list) else 1

        return np.full((n_rows, n_cols), self.dummy_value)

    def get_row_sum(
        self,
        row_set: str,
        col_set: str,
        row_idx: int | None = None,
    ) -> float | np.ndarray:
        """Get sum of a row from a matrix.

        Args:
            row_set: Row set name
            col_set: Column set name
            row_idx: Specific row index (None for all rows)

        Returns:
            Sum value(s)
        """
        matrix = self.get_matrix(row_set, col_set)

        if row_idx is not None:
            return matrix[row_idx, :].sum()
        else:
            return matrix.sum(axis=1)

    def get_col_sum(
        self,
        row_set: str,
        col_set: str,
        col_idx: int | None = None,
    ) -> float | np.ndarray:
        """Get sum of a column from a matrix.

        Args:
            row_set: Row set name
            col_set: Column set name
            col_idx: Specific column index (None for all columns)

        Returns:
            Sum value(s)
        """
        matrix = self.get_matrix(row_set, col_set)

        if col_idx is not None:
            return matrix[:, col_idx].sum()
        else:
            return matrix.sum(axis=0)

    def get_element(
        self,
        row_set: str,
        col_set: str,
        row_idx: int,
        col_idx: int,
    ) -> float:
        """Get single element from a matrix.

        Args:
            row_set: Row set name
            col_set: Column set name
            row_idx: Row index
            col_idx: Column index

        Returns:
            Matrix element value
        """
        matrix = self.get_matrix(row_set, col_set)
        return float(matrix[row_idx, col_idx])

    def set_block_params(self, block_name: str, params: dict[str, np.ndarray]) -> None:
        """Store calibrated parameters for a block.

        These parameters can be accessed by other blocks during calibration.

        Args:
            block_name: Name of the block
            params: Dictionary of parameter names to values
        """
        self._params[block_name] = params

    def get_block_params(self, block_name: str) -> dict[str, np.ndarray]:
        """Get calibrated parameters from another block.

        Args:
            block_name: Name of the block to get params from

        Returns:
            Dictionary of parameter names to values (empty if not found)
        """
        return self._params.get(block_name, {})

    def get_block_param(self, block_name: str, param_name: str) -> np.ndarray | None:
        """Get a specific parameter from another block.

        Args:
            block_name: Name of the block
            param_name: Name of the parameter

        Returns:
            Parameter value or None if not found
        """
        block_params = self._params.get(block_name, {})
        return block_params.get(param_name)

    def list_calibrated_blocks(self) -> list[str]:
        """List names of blocks that have been calibrated.

        Returns:
            List of block names
        """
        return list(self._params.keys())

    def clear_cache(self) -> None:
        """Clear the matrix cache.

        Useful if SAM data has been modified.
        """
        self._cache.clear()

    def clear_block_params(self, block_name: str | None = None) -> None:
        """Clear stored block parameters.

        Args:
            block_name: Specific block to clear (None for all)
        """
        if block_name is None:
            self._params.clear()
        elif block_name in self._params:
            del self._params[block_name]

    def to_dict(self) -> dict[str, Any]:
        """Convert calibration data to dictionary for serialization.

        Returns:
            Dictionary with mode, dummy_value, and cached data
        """
        return {
            "mode": self.mode,
            "dummy_value": self.dummy_value,
            "cached_matrices": list(self._cache.keys()),
            "calibrated_blocks": list(self._params.keys()),
            "set_mappings": self._set_mappings,
        }


class DummyCalibrationData(CalibrationData):
    """Convenience class for dummy calibration mode.

    Automatically sets mode to "dummy" and provides sensible defaults.
    """

    def __init__(self, dummy_value: float = 1.0):
        """Initialize dummy calibration data.

        Args:
            dummy_value: Value to use for all dummy matrices (default: 1.0)
        """
        super().__init__(sam=None, mode="dummy", dummy_value=dummy_value)


# Convenience functions
def create_calibration_data(
    sam: SAM | None,
    mode: str = "sam",
    dummy_value: float = 1.0,
) -> CalibrationData:
    """Create CalibrationData instance.

    Args:
        sam: Social Accounting Matrix (None for dummy mode)
        mode: "sam" or "dummy"
        dummy_value: Default value for dummy mode

    Returns:
        CalibrationData instance
    """
    return CalibrationData(sam, mode, dummy_value)
