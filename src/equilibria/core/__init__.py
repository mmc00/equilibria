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
from equilibria.core.recipe import PERIOD_KINDS, KindPlan, Period, Recipe
from equilibria.core.sets import Set, SetManager
from equilibria.core.variables import Variable

__all__ = [
    # Runs: recipe, driver, exercise (ADR-0011)
    "Exercise",
    "KindPlan",
    "PERIOD_KINDS",
    "Period",
    "Recipe",
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


def __getattr__(name: str):
    # Exercise pulls in the driver and Pyomo; load it on first use so that
    # importing the core primitives stays light.
    if name == "Exercise":
        from equilibria.core.exercise import Exercise

        return Exercise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
