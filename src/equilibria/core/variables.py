"""Variable definitions for CGE models.

Variables represent endogenous quantities in the model that are
determined by the solver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from equilibria.core.sets import SetManager


class Variable(BaseModel):
    """A model variable with multi-dimensional indexing.

    Variables represent endogenous quantities determined by the solver.
    They support n-dimensional indexing and can have lower/upper bounds.

    Attributes:
        name: Variable identifier
        value: Current value (initial guess or solution)
        domains: Tuple of set names defining dimensions
        lower: Lower bound (default: 0 for non-negative variables)
        upper: Upper bound (default: inf)
        description: Human-readable description
        initial_value: Starting value for solver

    Example:
        >>> price = Variable(
        ...     name="P",
        ...     value=np.ones((2, 3)),
        ...     domains=("J", "I"),
        ...     lower=0.0,
        ...     description="Price level"
        ... )
    """

    name: str = Field(..., min_length=1, description="Variable identifier")
    value: np.ndarray = Field(..., description="Variable values")
    domains: tuple[str, ...] = Field(
        default_factory=tuple, description="Dimension set names"
    )
    lower: float | np.ndarray = Field(default=0.0, description="Lower bound")
    upper: float | np.ndarray = Field(default=float("inf"), description="Upper bound")
    description: str = Field(default="", description="Human-readable description")

    model_config = {"arbitrary_types_allowed": True, "frozen": False}

    @field_validator("value", mode="before")
    @classmethod
    def ensure_numpy_array(cls, v: Any) -> np.ndarray:  # noqa: N805
        """Convert input to numpy array."""
        if isinstance(v, (list, tuple)):
            return np.array(v, dtype=float)
        if isinstance(v, np.ndarray):
            return v.astype(float)
        return np.array([v], dtype=float)

    def __post_init__(self) -> None:
        """Validate dimensions match domains."""
        if len(self.domains) != self.value.ndim:
            msg = (
                f"Variable '{self.name}': {len(self.domains)} domains "
                f"but {self.value.ndim} dimensions in value"
            )
            raise ValueError(msg)

    def __getitem__(self, key: str | tuple[str, ...]) -> float:
        """Get variable value by index.

        Args:
            key: Element name(s) matching domains

        Returns:
            Variable value
        """
        if isinstance(key, str):
            key = (key,)

        if len(key) != len(self.domains):
            msg = (
                f"Variable '{self.name}': expected {len(self.domains)} "
                f"indices, got {len(key)}"
            )
            raise IndexError(msg)

        indices = tuple(self._get_index(i, k) for i, k in enumerate(key))
        return float(self.value[indices])

    def __setitem__(self, key: str | tuple[str, ...], value: float) -> None:
        """Set variable value by index.

        Args:
            key: Element name(s) matching domains
            value: New value
        """
        if isinstance(key, str):
            key = (key,)

        if len(key) != len(self.domains):
            msg = (
                f"Variable '{self.name}': expected {len(self.domains)} "
                f"indices, got {len(key)}"
            )
            raise IndexError(msg)

        indices = tuple(self._get_index(i, k) for i, k in enumerate(key))
        self.value[indices] = value

    def _get_index(self, dim: int, element: str) -> int:
        """Convert element name to array index."""
        return hash(element) % self.value.shape[dim]

    def set_initial_value(self, value: float | np.ndarray) -> None:
        """Set initial value for solver.

        Args:
            value: Initial guess (scalar or array matching shape)
        """
        if np.isscalar(value):
            self.value = np.full_like(self.value, value)
        else:
            self.value = np.array(value, dtype=float).reshape(self.value.shape)

    def fix(self, value: float | None = None) -> None:
        """Fix variable to current or specified value.

        Args:
            value: Optional value to fix to (uses current if None)
        """
        if value is not None:
            self.value = np.full_like(self.value, value)
        self.lower = self.value.copy()
        self.upper = self.value.copy()

    def unfix(self, lower: float = 0.0, upper: float = float("inf")) -> None:
        """Unfix variable and restore bounds.

        Args:
            lower: New lower bound
            upper: New upper bound
        """
        self.lower = lower
        self.upper = upper

    def is_fixed(self) -> bool:
        """Check if variable is fixed."""
        return np.allclose(self.lower, self.upper)

    def shape(self) -> tuple[int, ...]:
        """Return shape of variable array."""
        return tuple(self.value.shape)

    def ndim(self) -> int:
        """Return number of dimensions."""
        return self.value.ndim

    def to_dict(self) -> dict[str, Any]:
        """Convert variable to dictionary."""
        lower_val: float | list[float]
        upper_val: float | list[float]

        if isinstance(self.lower, np.ndarray):
            lower_val = self.lower.tolist()
        elif hasattr(self.lower, "__float__"):
            lower_val = float(self.lower)
        else:
            lower_val = 0.0

        if isinstance(self.upper, np.ndarray):
            upper_val = self.upper.tolist()
        elif hasattr(self.upper, "__float__"):
            upper_val = float(self.upper)
        else:
            upper_val = float("inf")

        return {
            "name": self.name,
            "value": self.value.tolist(),
            "domains": self.domains,
            "lower": lower_val,
            "upper": upper_val,
            "description": self.description,
            "fixed": self.is_fixed(),
        }

    def __repr__(self) -> str:
        """String representation."""
        domain_str = f"[{', '.join(self.domains)}]" if self.domains else "scalar"
        fixed_str = " [FIXED]" if self.is_fixed() else ""
        return f"Variable {self.name}{domain_str}: shape {self.shape()}{fixed_str}"


class VariableManager:
    """Manages all variables in a model.

    Provides centralized access to variables with set-based validation.
    """

    def __init__(self, set_manager: SetManager | None = None) -> None:
        """Initialize variable manager.

        Args:
            set_manager: Optional set manager for validation
        """
        self._vars: dict[str, Variable] = {}
        self._set_manager = set_manager

    def add(self, var: Variable) -> None:
        """Add a variable.

        Args:
            var: Variable to add

        Raises:
            ValueError: If variable already exists
        """
        if var.name in self._vars:
            msg = f"Variable '{var.name}' already exists"
            raise ValueError(msg)

        # Validate domains exist in set manager
        if self._set_manager is not None:
            for domain in var.domains:
                if domain not in self._set_manager:
                    msg = f"Domain set '{domain}' not found for variable '{var.name}'"
                    raise ValueError(msg)

        self._vars[var.name] = var

    def get(self, name: str) -> Variable:
        """Get variable by name.

        Args:
            name: Variable name

        Returns:
            The Variable object

        Raises:
            KeyError: If variable not found
        """
        if name not in self._vars:
            msg = f"Variable '{name}' not found"
            raise KeyError(msg)
        return self._vars[name]

    def __getitem__(self, name: str) -> Variable:
        """Get variable by name using bracket notation."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if variable exists."""
        return name in self._vars

    def list_vars(self) -> list[str]:
        """Return list of all variable names."""
        return list(self._vars.keys())

    def get_total_count(self) -> int:
        """Return total number of scalar variables."""
        total: int = 0
        for v in self._vars.values():
            shape = v.shape()
            prod = 1
            for dim in shape:
                prod *= dim
            total += prod
        return total

    def summary(self) -> dict[str, Any]:
        """Return summary of all variables."""
        return {
            "total_variables": len(self._vars),
            "total_scalar_vars": self.get_total_count(),
            "variables": {
                name: {
                    "shape": v.shape(),
                    "domains": v.domains,
                    "fixed": v.is_fixed(),
                    "description": v.description,
                }
                for name, v in self._vars.items()
            },
        }
