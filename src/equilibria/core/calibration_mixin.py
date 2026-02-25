"""Calibration mixin for Block classes in equilibria CGE framework.

This module provides the CalibrationMixin class that adds calibration
functionality to the Block base class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import model_validator

if TYPE_CHECKING:
    from equilibria.core.calibration_data import CalibrationData
    from equilibria.core.calibration_phase import CalibrationPhase
    from equilibria.core.parameters import Parameter, ParameterManager
    from equilibria.core.sets import SetManager
    from equilibria.core.variables import Variable, VariableManager


class CalibrationMixin:
    """Mixin class adding calibration functionality to Blocks.

    This mixin provides the calibration infrastructure for CGE model blocks.
    Blocks inherit from this mixin and override calibration methods to
    implement their specific calibration logic.

    Calibration follows the GAMS pattern:
    1. Extract base year data from SAM into "0" parameters (e.g., QD0, VA0)
    2. Compute derived parameters (e.g., beta_VA from factor payments)
    3. Initialize variables from "0" parameters

    Example:
        >>> class MyBlock(Block, CalibrationMixin):
        ...     def get_calibration_phases(self):
        ...         return [CalibrationPhase.PRODUCTION]
        ...
        ...     def _extract_calibration(self, phase, data, mode, set_manager):
        ...         # Extract from SAM or create dummy data
        ...         if mode == "sam":
        ...             FD0 = data.get_matrix("F", "J")
        ...         else:
        ...             FD0 = np.ones((4, 5))
        ...         return {"FD0": FD0, "beta_VA": FD0 / FD0.sum()}
        ...
        ...     def _initialize_variables(self, calibrated, set_manager):
        ...         self.variables["FD"].value = calibrated["FD0"].copy()
    """

    # User-specified dummy defaults - class attribute, can be overridden per instance
    dummy_defaults: dict[str, Any] = {}

    @model_validator(mode="after")
    def _init_calibration_attrs(self):
        """Initialize calibration attributes after Pydantic validation."""
        # Initialize instance attributes that must not be shared between blocks
        object.__setattr__(self, "_calibrated_params", {})
        object.__setattr__(self, "_calibrated_data", {})
        return self

    def get_calibration_phases(self):
        """Return the calibration phases this block participates in.

        Override in subclasses to declare which phases this block
        needs to be calibrated in.

        Returns:
            List of CalibrationPhase values
        """
        return []

    def calibrate(
        self,
        phase,
        data,
        mode,
        set_manager,
        param_manager,
        var_manager,
        dependency_validator=None,
    ):
        """Calibrate this block in the given phase.

        This is the main entry point for calibration. It checks if the
        block participates in the given phase, extracts calibration data,
        stores "0" parameters, and initializes variables.

        Args:
            phase: The current calibration phase
            data: CalibrationData instance with SAM or dummy data
            mode: "sam" or "dummy"
            set_manager: SetManager for accessing sets
            param_manager: ParameterManager for storing "0" parameters
            var_manager: VariableManager for initializing variables
            dependency_validator: Optional validator for checking dependencies

        Raises:
            CalibrationError: If calibration fails
        """
        # Check if this block participates in this phase
        phases = self.get_calibration_phases()
        if phase not in phases:
            return

        # Extract calibration data
        calibrated = self._extract_calibration(phase, data, mode, set_manager)

        if not calibrated:
            return

        # Store the calibrated data internally
        self._calibrated_params.update(calibrated)
        object.__setattr__(self, "_calibrated_data", dict(self._calibrated_params))

        # Store "0" parameters permanently in ParameterManager
        for name, value in calibrated.items():
            if name.endswith("0"):  # It's a base year parameter
                # Check if parameter already exists
                if name in param_manager:
                    # Update existing parameter
                    existing = param_manager.get(name)
                    existing.value = value
                else:
                    # Create new parameter
                    from equilibria.core.parameters import Parameter

                    param = Parameter(
                        name=name,
                        value=value,
                        description=f"Base year value for {name[:-1]}",
                    )
                    param_manager.add(param)

        # Register with dependency validator
        if dependency_validator is not None:
            dependency_validator.register_calibration(self, phase)

        # Store in CalibrationData for other blocks to access
        data.set_block_params(self.name, self._calibrated_params)

        # Initialize variables from calibrated data
        self._initialize_variables(calibrated, set_manager, var_manager)

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for this block.

        Override this method in subclasses to implement block-specific
        calibration logic. Extract data from SAM or create dummy data.

        Args:
            phase: Current calibration phase
            data: CalibrationData instance
            mode: "sam" or "dummy"
            set_manager: SetManager for accessing sets

        Returns:
            Dictionary mapping parameter names to numpy arrays.
            Should include "0" parameters (e.g., "FD0", "VA0") and
            derived parameters (e.g., "beta_VA").

        Raises:
            NotImplementedError: If not overridden in subclass
        """
        raise NotImplementedError(
            f"Block '{self.name}' must implement _extract_calibration() "
            f"to support calibration mode '{mode}'"
        )

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize block variables from calibrated parameters.

        Override this method to set variable initial values from
        the calibrated "0" parameters.

        Args:
            calibrated: Dictionary of calibrated parameters
            set_manager: SetManager
            var_manager: VariableManager
        """
        # Default: try to match variable names to "0" parameters
        # e.g., variable "FD" gets initialized from parameter "FD0"
        for var_name in self.get_variable_names():
            zero_param_name = f"{var_name}0"
            if zero_param_name in calibrated:
                var = var_manager.get(var_name)
                if var is not None:
                    var.value = calibrated[zero_param_name].copy()

    def _get_dummy_value(self, param_name, shape, default=1.0):
        """Get dummy value for a parameter.

        Checks dummy_defaults first, then falls back to uniform value.

        Args:
            param_name: Name of the parameter
            shape: Shape of the array
            default: Default value if not in dummy_defaults

        Returns:
            Numpy array with dummy values
        """
        if param_name in self.dummy_defaults:
            spec = self.dummy_defaults[param_name]
            if isinstance(spec, dict):
                # Advanced spec: {"value": val, "distribution": "uniform"}
                val = spec.get("value", default)
            else:
                # Simple spec: just the value
                val = spec
            return np.full(shape, val)
        else:
            return np.full(shape, default)

    def _compute_shares(self, values, axis=None, epsilon=1e-10):
        """Compute shares that sum to 1.0.

        Helper method for computing CES/Cobb-Douglas shares, budget shares, etc.

        Args:
            values: Array of values
            axis: Axis along which to compute shares (None for total)
            epsilon: Small value to avoid division by zero

        Returns:
            Shares that sum to 1.0 along specified axis
        """
        total = values.sum(axis=axis, keepdims=True)
        # Avoid division by zero
        total = np.where(total < epsilon, epsilon, total)
        return values / total

    def get_calibrated_params(self):
        """Get the calibrated parameters for this block.

        Returns:
            Dictionary of parameter names to values
        """
        return self._calibrated_params.copy()

    def get_calibrated_param(self, name):
        """Get a specific calibrated parameter.

        Args:
            name: Parameter name

        Returns:
            Parameter value or None if not found
        """
        return self._calibrated_params.get(name)

    def clear_calibration(self):
        """Clear calibrated parameters.

        Useful for recalibrating with different data.
        """
        self._calibrated_params.clear()

    def has_calibration(self):
        """Check if this block has been calibrated.

        Returns:
            True if calibrated parameters exist
        """
        return len(self._calibrated_params) > 0

    def get_variable_names(self):
        """Get list of variable names in this block.

        Override in Block subclass.

        Returns:
            List of variable names
        """
        return list(self.variables.keys()) if hasattr(self, "variables") else []


# Helper functions for common calibration operations


def compute_ces_shares(factor_payments, axis=0):
    """Compute CES share parameters from factor payments.

    Args:
        factor_payments: Matrix of factor payments [factors, sectors]
        axis: Axis along which to compute shares (0=factors, 1=sectors)

    Returns:
        Share parameters that sum to 1.0
    """
    total = factor_payments.sum(axis=axis, keepdims=True)
    total = np.where(total < 1e-10, 1e-10, total)
    return factor_payments / total


def compute_io_coefficients(intermediate_inputs, total_output):
    """Compute input-output coefficients.

    Args:
        intermediate_inputs: Matrix [commodities, sectors]
        total_output: Vector [sectors]

    Returns:
        IO coefficients matrix [commodities, sectors]
    """
    # Broadcast total_output to match intermediate_inputs shape
    output_expanded = total_output[np.newaxis, :]
    output_expanded = np.where(output_expanded < 1e-10, 1e-10, output_expanded)
    return intermediate_inputs / output_expanded


def compute_armington_shares(domestic_supply, imports):
    """Compute Armington domestic and import shares.

    Args:
        domestic_supply: Vector [commodities]
        imports: Vector [commodities]

    Returns:
        Tuple of (alpha_D, alpha_M) share vectors
    """
    total = domestic_supply + imports
    total = np.where(total < 1e-10, 1e-10, total)
    alpha_D = domestic_supply / total
    alpha_M = imports / total
    return alpha_D, alpha_M


def compute_les_parameters(consumption, prices, income, subsistence_ratio=0.1):
    """Compute LES parameters from consumption data.

    Args:
        consumption: Vector of consumption quantities [commodities]
        prices: Vector of prices [commodities]
        income: Total income
        subsistence_ratio: Fraction of income for subsistence

    Returns:
        Tuple of (gamma, beta) - subsistence and marginal budget shares
    """
    # Subsistence consumption (minimum requirements)
    total_subsistence = income * subsistence_ratio
    gamma = consumption * 0.5  # Assume 50% is subsistence (simplified)

    # Marginal budget shares
    expenditure = consumption * prices
    beta = (
        (expenditure / expenditure.sum())
        if expenditure.sum() > 0
        else np.ones_like(expenditure) / len(expenditure)
    )

    return gamma, beta
