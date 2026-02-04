"""Calibration base classes for equilibria CGE framework.

This module provides the foundation for calibrating CGE models
from SAM data, including elasticity estimation and parameter
computation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from equilibria.babel import SAM
from equilibria.model import Model


class CalibrationResult(BaseModel):
    """Results from a calibration operation.

    Stores calibrated parameters, statistics, and diagnostic
    information from the calibration process.

    Attributes:
        success: Whether calibration succeeded
        parameters: Dictionary of calibrated parameter values
        statistics: Calibration statistics
        messages: List of diagnostic messages
        warnings: List of warning messages
    """

    success: bool = Field(default=True, description="Calibration success status")
    parameters: dict[str, np.ndarray] = Field(
        default_factory=dict, description="Calibrated parameter values"
    )
    statistics: dict[str, Any] = Field(
        default_factory=dict, description="Calibration statistics"
    )
    messages: list[str] = Field(default_factory=list, description="Diagnostic messages")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")

    model_config = {"arbitrary_types_allowed": True}

    def add_message(self, message: str) -> None:
        """Add a diagnostic message."""
        self.messages.append(message)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "parameters": {k: v.tolist() for k, v in self.parameters.items()},
            "statistics": self.statistics,
            "messages": self.messages,
            "warnings": self.warnings,
        }


class Calibrator(ABC, BaseModel):
    """Abstract base class for model calibrators.

    Calibrators compute model parameters from SAM data and
    user-provided elasticities.

    Attributes:
        name: Calibrator name
        description: Calibrator description
    """

    name: str = Field(..., description="Calibrator name")
    description: str = Field(default="", description="Calibrator description")

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def calibrate(
        self,
        model: Model,
        sam: SAM,
        elasticities: dict[str, float] | None = None,
    ) -> CalibrationResult:
        """Calibrate model parameters from SAM data.

        Args:
            model: Model to calibrate
            sam: Social Accounting Matrix with base year data
            elasticities: Optional user-provided elasticities

        Returns:
            CalibrationResult with calibrated parameters
        """
        ...

    def validate_sam(self, sam: SAM, tolerance: float = 1e-6) -> bool:
        """Validate that SAM is balanced.

        Args:
            sam: SAM to validate
            tolerance: Balance tolerance

        Returns:
            True if valid

        Raises:
            ValueError: If SAM is not balanced
        """
        validation = sam.check_balance(tolerance)
        if not validation["is_balanced"]:
            msg = f"SAM is not balanced. Max difference: {validation['max_difference']}"
            raise ValueError(msg)
        return True

    def get_sam_value(
        self,
        sam: SAM,
        row_account: str,
        col_account: str,
    ) -> float:
        """Get value from SAM matrix.

        Args:
            sam: SAM data
            row_account: Row account name
            col_account: Column account name

        Returns:
            Matrix value
        """
        return float(sam.data.loc[row_account, col_account])

    def compute_io_coefficients(
        self,
        sam: SAM,
        sectors: list[str],
    ) -> dict[tuple[str, str], float]:
        """Compute input-output coefficients.

        Args:
            sam: SAM data
            sectors: List of sector names

        Returns:
            Dictionary of (input_sector, output_sector) -> coefficient
        """
        coefficients = {}

        for output_sector in sectors:
            # Get total output for this sector
            total_output = sum(
                self.get_sam_value(sam, output_sector, s) for s in sectors
            )
            total_output += sum(
                self.get_sam_value(sam, output_sector, f)
                for f in sam.data.index
                if f not in sectors
            )

            if total_output > 0:
                for input_sector in sectors:
                    value = self.get_sam_value(sam, input_sector, output_sector)
                    coefficients[(input_sector, output_sector)] = value / total_output

        return coefficients

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.name})"


class ModelCalibrator:
    """Orchestrates calibration of multiple model components.

    Manages multiple calibrators and coordinates the full
    model calibration process.

    Attributes:
        calibrators: List of calibrators to apply
    """

    def __init__(self) -> None:
        """Initialize empty calibrator."""
        self._calibrators: list[Calibrator] = []

    def add_calibrator(self, calibrator: Calibrator) -> None:
        """Add a calibrator.

        Args:
            calibrator: Calibrator to add
        """
        self._calibrators.append(calibrator)

    def calibrate(
        self,
        model: Model,
        sam: SAM,
        elasticities: dict[str, float] | None = None,
    ) -> dict[str, CalibrationResult]:
        """Run all calibrators on model.

        Args:
            model: Model to calibrate
            sam: SAM data
            elasticities: Optional elasticities

        Returns:
            Dictionary of calibrator name -> results
        """
        results = {}

        for calibrator in self._calibrators:
            result = calibrator.calibrate(model, sam, elasticities)
            results[calibrator.name] = result

        return results

    def apply_results(
        self,
        model: Model,
        results: dict[str, CalibrationResult],
    ) -> None:
        """Apply calibration results to model.

        Args:
            model: Model to update
            results: Calibration results
        """
        for calibrator_name, result in results.items():
            if not result.success:
                continue

            for param_name, values in result.parameters.items():
                if param_name in model.parameter_manager.list_params():
                    param = model.get_parameter(param_name)
                    param.value = values

    def __repr__(self) -> str:
        """String representation."""
        return f"ModelCalibrator({len(self._calibrators)} calibrators)"
