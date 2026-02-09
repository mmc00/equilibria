"""Core data structures for equilibria CGE modeling framework.

This module provides the fundamental building blocks for CGE models:
- Sets: Index definitions for multi-dimensional data
- Parameters: Constant values (calibrated from SAM)
- Variables: Endogenous model variables
- Equations: Mathematical relationships
- Calibration: SAM-based parameter calibration
"""

from equilibria.core.calibration_data import CalibrationData, DummyCalibrationData
from equilibria.core.calibration_mixin import CalibrationMixin
from equilibria.core.calibration_phase import (
    CalibrationDependencyError,
    CalibrationPhase,
    DependencyValidator,
    PhaseRegistry,
    get_calibration_phase,
    list_calibration_phases,
    register_calibration_phase,
)
from equilibria.core.equations import Equation
from equilibria.core.parameters import Parameter
from equilibria.core.sets import Set, SetManager
from equilibria.core.variables import Variable

__all__ = [
    # Core data structures
    "Set",
    "SetManager",
    "Parameter",
    "Variable",
    "Equation",
    # Calibration
    "CalibrationData",
    "DummyCalibrationData",
    "CalibrationMixin",
    "CalibrationPhase",
    "CalibrationDependencyError",
    "DependencyValidator",
    "PhaseRegistry",
    "get_calibration_phase",
    "list_calibration_phases",
    "register_calibration_phase",
]
