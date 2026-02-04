"""Parameter containers for CGE models.

Parameters are constant values that are typically calibrated from
SAM data. They support multi-dimensional indexing using sets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from equilibria.core.sets import SetManager


class Parameter(BaseModel):
    """A model parameter with multi-dimensional indexing.

    Parameters represent constant values in the model (calibrated from
    SAM data). They support n-dimensional indexing using sets.

    Attributes:
        name: Parameter identifier
        value: Numpy array of values
        domains: Tuple of set names defining dimensions
        description: Human-readable description

    Example:
        >>> sigma = Parameter(
        ...     name="sigma_VA",
        ...     value=np.array([[0.8, 0.9], [0.7, 1.0]]),
        ...     domains=("J", "I"),
        ...     description="CES elasticity of substitution"
        ... )
        >>> print(sigma["agr", "labor"])
        0.8
    """

    name: str = Field(..., min_length=1, description="Parameter identifier")
    value: np.ndarray = Field(..., description="Parameter values")
    domains: tuple[str, ...] = Field(
        default_factory=tuple, description="Dimension set names"
    )
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
                f"Parameter '{self.name}': {len(self.domains)} domains "
                f"but {self.value.ndim} dimensions in value"
            )
            raise ValueError(msg)

    def __getitem__(self, key: str | tuple[str, ...]) -> float:
        """Get parameter value by index.

        Args:
            key: Element name(s) matching domains

        Returns:
            Parameter value

        Example:
            >>> param["agr"]  # 1D
            >>> param["agr", "labor"]  # 2D
        """
        if isinstance(key, str):
            key = (key,)

        if len(key) != len(self.domains):
            msg = (
                f"Parameter '{self.name}': expected {len(self.domains)} "
                f"indices, got {len(key)}"
            )
            raise IndexError(msg)

        # For now, return based on position (set manager validates indices)
        indices = tuple(self._get_index(i, k) for i, k in enumerate(key))
        return float(self.value[indices])

    def __setitem__(self, key: str | tuple[str, ...], value: float) -> None:
        """Set parameter value by index.

        Args:
            key: Element name(s) matching domains
            value: New value
        """
        if isinstance(key, str):
            key = (key,)

        if len(key) != len(self.domains):
            msg = (
                f"Parameter '{self.name}': expected {len(self.domains)} "
                f"indices, got {len(key)}"
            )
            raise IndexError(msg)

        indices = tuple(self._get_index(i, k) for i, k in enumerate(key))
        self.value[indices] = value

    def _get_index(self, dim: int, element: str) -> int:
        """Convert element name to array index.

        Args:
            dim: Dimension index
            element: Element name

        Returns:
            Array index position

        Note: This is a placeholder. Actual validation requires SetManager.
        """
        # For now, assume element names map to positions
        # In practice, SetManager validates and converts
        return hash(element) % self.value.shape[dim]

    def get_value(self, *indices: int) -> float:
        """Get value by integer indices.

        Args:
            *indices: Integer positions for each dimension

        Returns:
            Parameter value
        """
        return float(self.value[indices])

    def set_value(self, value: float, *indices: int) -> None:
        """Set value by integer indices.

        Args:
            value: New value
            *indices: Integer positions for each dimension
        """
        self.value[indices] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert parameter to dictionary."""
        return {
            "name": self.name,
            "value": self.value.tolist(),
            "domains": self.domains,
            "description": self.description,
        }

    def shape(self) -> tuple[int, ...]:
        """Return shape of parameter array."""
        return tuple(self.value.shape)

    def ndim(self) -> int:
        """Return number of dimensions."""
        return self.value.ndim

    def __repr__(self) -> str:
        """String representation."""
        domain_str = f"[{', '.join(self.domains)}]" if self.domains else "scalar"
        return f"Parameter {self.name}{domain_str}: shape {self.shape()}"


class ParameterManager:
    """Manages all parameters in a model.

    Provides centralized access to parameters with set-based validation.

    Attributes:
        parameters: Dictionary of parameter name to Parameter objects
        set_manager: Reference to SetManager for validation
    """

    def __init__(self, set_manager: SetManager | None = None) -> None:
        """Initialize parameter manager.

        Args:
            set_manager: Optional set manager for validation
        """
        self._params: dict[str, Parameter] = {}
        self._set_manager = set_manager

    def add(self, param: Parameter) -> None:
        """Add a parameter.

        Args:
            param: Parameter to add

        Raises:
            ValueError: If parameter already exists
        """
        if param.name in self._params:
            msg = f"Parameter '{param.name}' already exists"
            raise ValueError(msg)

        # Validate domains exist in set manager
        if self._set_manager is not None:
            for domain in param.domains:
                if domain not in self._set_manager:
                    msg = (
                        f"Domain set '{domain}' not found for parameter '{param.name}'"
                    )
                    raise ValueError(msg)

        self._params[param.name] = param

    def get(self, name: str) -> Parameter:
        """Get parameter by name.

        Args:
            name: Parameter name

        Returns:
            The Parameter object

        Raises:
            KeyError: If parameter not found
        """
        if name not in self._params:
            msg = f"Parameter '{name}' not found"
            raise KeyError(msg)
        return self._params[name]

    def __getitem__(self, name: str) -> Parameter:
        """Get parameter by name using bracket notation."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if parameter exists."""
        return name in self._params

    def list_params(self) -> list[str]:
        """Return list of all parameter names."""
        return list(self._params.keys())

    def summary(self) -> dict[str, Any]:
        """Return summary of all parameters."""
        return {
            "total_parameters": len(self._params),
            "parameters": {
                name: {
                    "shape": p.shape(),
                    "domains": p.domains,
                    "description": p.description,
                }
                for name, p in self._params.items()
            },
        }
