"""Backend base classes for equilibria CGE framework.

This module defines the abstract base classes for solver backends
and the solution container for model results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class Solution(BaseModel):
    """Container for model solution results.

    Stores the solved values of all variables and provides
    methods for accessing and comparing results.

    Attributes:
        model_name: Name of the solved model
        status: Solver status (e.g., 'optimal', 'infeasible')
        objective_value: Value of the objective function (if any)
        variables: Dictionary of variable names to their solved values
        solve_time: Time taken to solve (seconds)
        iterations: Number of solver iterations
    """

    model_name: str = Field(..., description="Model name")
    status: str = Field(default="unknown", description="Solver status")
    objective_value: float | None = Field(default=None, description="Objective value")
    variables: dict[str, np.ndarray] = Field(
        default_factory=dict, description="Variable values"
    )
    solve_time: float = Field(default=0.0, description="Solve time in seconds")
    iterations: int = Field(default=0, description="Number of iterations")

    model_config = {"arbitrary_types_allowed": True}

    def get_variable(self, name: str) -> np.ndarray | None:
        """Get the value of a variable by name.

        Args:
            name: Variable name

        Returns:
            Variable value array or None if not found
        """
        return self.variables.get(name)

    def compare(self, other: Solution, tolerance: float = 1e-6) -> dict[str, Any]:
        """Compare this solution with another.

        Args:
            other: Another solution to compare with
            tolerance: Tolerance for considering values different

        Returns:
            Dictionary with comparison results
        """
        differences = {}

        all_vars = set(self.variables.keys()) | set(other.variables.keys())

        for var_name in all_vars:
            val1 = self.variables.get(var_name)
            val2 = other.variables.get(var_name)

            if val1 is None or val2 is None:
                differences[var_name] = {
                    "status": "missing",
                    "in_self": val1 is not None,
                    "in_other": val2 is not None,
                }
            elif val1.shape != val2.shape:
                differences[var_name] = {
                    "status": "shape_mismatch",
                    "shape_self": val1.shape,
                    "shape_other": val2.shape,
                }
            else:
                diff = np.abs(val1 - val2)
                max_diff = np.max(diff)
                if max_diff > tolerance:
                    differences[var_name] = {
                        "status": "different",
                        "max_difference": max_diff,
                        "mean_difference": np.mean(diff),
                    }

        return {
            "is_equal": len(differences) == 0,
            "differences": differences,
            "tolerance": tolerance,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert solution to dictionary (with numpy arrays as lists)."""
        return {
            "model_name": self.model_name,
            "status": self.status,
            "objective_value": self.objective_value,
            "variables": {k: v.tolist() for k, v in self.variables.items()},
            "solve_time": self.solve_time,
            "iterations": self.iterations,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Solution({self.model_name}): "
            f"status={self.status}, "
            f"vars={len(self.variables)}, "
            f"time={self.solve_time:.2f}s"
        )


class Backend(ABC):
    """Abstract base class for solver backends.

    Backends translate equilibria models into solver-specific
    representations and handle the solution process.
    """

    def __init__(self, solver: str | None = None) -> None:
        """Initialize backend.

        Args:
            solver: Solver name (backend-specific)
        """
        self.solver = solver
        self._model = None

    @abstractmethod
    def build(self, model: Any) -> None:
        """Build the solver model from equilibria model.

        Args:
            model: equilibria Model instance
        """
        ...

    @abstractmethod
    def solve(self, options: dict[str, Any] | None = None) -> Solution:
        """Solve the model.

        Args:
            options: Solver-specific options

        Returns:
            Solution object with results
        """
        ...

    @abstractmethod
    def get_solver_status(self) -> dict[str, Any]:
        """Get detailed solver status information.

        Returns:
            Dictionary with solver status details
        """
        ...

    def list_available_solvers(self) -> list[str]:
        """List available solvers for this backend.

        Returns:
            List of solver names
        """
        return []

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(solver={self.solver})"
