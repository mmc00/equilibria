"""Block base classes for equilibria CGE modeling.

Blocks are self-contained equation modules that define economic behavior.
Each block declares its required sets, parameters, variables, and equations
using Pydantic for validation and introspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from equilibria.core.calibration_mixin import CalibrationMixin
from equilibria.core.parameters import Parameter
from equilibria.core.sets import SetManager
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

if TYPE_CHECKING:
    pass


class ParameterSpec(BaseModel):
    """Specification for a block parameter.

    Defines a parameter that the block requires, including its
    name, domains, and default value.

    Attributes:
        name: Parameter identifier
        domains: Tuple of set names defining dimensions
        default: Default value (optional)
        description: Human-readable description
    """

    name: str = Field(..., description="Parameter identifier")
    domains: tuple[str, ...] = Field(
        default_factory=tuple, description="Dimension set names"
    )
    default: float | None = Field(default=None, description="Default value")
    description: str = Field(default="", description="Parameter description")

    model_config = {"frozen": True}


class VariableSpec(BaseModel):
    """Specification for a block variable.

    Defines a variable that the block declares, including its
    name, domains, and bounds.

    Attributes:
        name: Variable identifier
        domains: Tuple of set names defining dimensions
        lower: Lower bound (default: 0)
        upper: Upper bound (default: inf)
        description: Human-readable description
    """

    name: str = Field(..., description="Variable identifier")
    domains: tuple[str, ...] = Field(
        default_factory=tuple, description="Dimension set names"
    )
    lower: float = Field(default=0.0, description="Lower bound")
    upper: float = Field(default=float("inf"), description="Upper bound")
    description: str = Field(default="", description="Variable description")

    model_config = {"frozen": True}


class EquationSpec(BaseModel):
    """Specification for a block equation.

    Defines an equation that the block contributes to the model.

    Attributes:
        name: Equation identifier
        domains: Tuple of set names defining equation indices
        description: Human-readable description
    """

    name: str = Field(..., description="Equation identifier")
    domains: tuple[str, ...] = Field(
        default_factory=tuple, description="Dimension set names"
    )
    description: str = Field(default="", description="Equation description")

    model_config = {"frozen": True}


class Block(BaseModel, CalibrationMixin, ABC):
    """Base class for CGE model blocks.

    Blocks are modular components that define economic behavior through
    equations. Each block declares its required sets, parameters, variables,
    and equations using Pydantic fields for validation.

    Blocks also support calibration from SAM data via the CalibrationMixin.

    Attributes:
        name: Block identifier
        description: Human-readable description
        required_sets: List of set names required by this block
        parameters: Dictionary of parameter specifications
        variables: Dictionary of variable specifications
        equations: List of equation specifications
        dummy_defaults: User-specified dummy values for calibration

    Example:
        >>> class CESValueAdded(Block):
        ...     name: str = "CES_VA"
        ...     description: str = "CES value-added production"
        ...     required_sets: list[str] = ["J", "I"]
        ...     sigma: float = Field(default=0.8, description="Elasticity")
        ...
        ...     def get_calibration_phases(self):
        ...         return [CalibrationPhase.PRODUCTION]
        ...
        ...     def _extract_calibration(self, phase, data, mode, set_manager):
        ...         # Extract from SAM
        ...         FD0 = data.get_matrix("F", "J")
        ...         VA0 = FD0.sum(axis=0)
        ...         beta_VA = self._compute_shares(FD0, axis=0)
        ...         return {"FD0": FD0, "VA0": VA0, "beta_VA": beta_VA}
    """

    name: str = Field(..., description="Block identifier")
    description: str = Field(default="", description="Block description")
    required_sets: list[str] = Field(
        default_factory=list, description="Required set names"
    )
    parameters: dict[str, ParameterSpec] = Field(
        default_factory=dict, description="Parameter specifications"
    )
    variables: dict[str, VariableSpec] = Field(
        default_factory=dict, description="Variable specifications"
    )
    equations: list[EquationSpec] = Field(
        default_factory=list, description="Equation specifications"
    )

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @field_validator("required_sets")
    @classmethod
    def validate_unique_sets(cls, v: list[str]) -> list[str]:  # noqa: N805
        """Ensure required sets are unique."""
        if len(v) != len(set(v)):
            msg = "Required sets must be unique"
            raise ValueError(msg)
        return v

    @abstractmethod
    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        """Set up the block in the model.

        This method is called when the block is added to a model.
        It should create and return the actual equation objects.

        Args:
            set_manager: Set manager for index validation
            parameters: Dictionary to add parameters to
            variables: Dictionary to add variables to

        Returns:
            List of SymbolicEquation objects contributed by this block
        """
        ...

    def validate_sets(self, set_manager: SetManager) -> bool:
        """Validate that all required sets exist.

        Args:
            set_manager: Set manager to check against

        Returns:
            True if all sets exist

        Raises:
            ValueError: If a required set is missing
        """
        for set_name in self.required_sets:
            if set_name not in set_manager:
                msg = f"Block '{self.name}' requires set '{set_name}' which is not defined"
                raise ValueError(msg)
        return True

    def get_info(self) -> dict[str, Any]:
        """Get block metadata as dictionary.

        Returns:
            Dictionary with block information
        """
        return {
            "name": self.name,
            "description": self.description,
            "required_sets": self.required_sets,
            "parameters": {k: v.model_dump() for k, v in self.parameters.items()},
            "variables": {k: v.model_dump() for k, v in self.variables.items()},
            "equations": [eq.model_dump() for eq in self.equations],
        }

    def initialize_levels(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
        mode: str = "gams_blockwise",
    ) -> None:
        """Initialize or update variable levels for this block.

        This hook is intentionally optional. Concrete blocks can override it to
        implement GAMS-style blockwise initialization logic without embedding the
        logic directly in a solver.

        Args:
            set_manager: Set manager with resolved model sets.
            parameters: Calibrated parameter values/symbols.
            variables: Mutable variable-level container to update in-place.
            mode: Initialization mode label (e.g. ``gams_blockwise``).
        """
        _ = (set_manager, parameters, variables, mode)

    def validate_initialization(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
    ) -> dict[str, float]:
        """Return block residual diagnostics after initialization.

        Blocks can override this to report equation-level residuals immediately
        after ``initialize_levels``. Default implementation returns an empty map.

        Args:
            set_manager: Set manager with resolved model sets.
            parameters: Calibrated parameter values/symbols.
            variables: Current initialized variable levels.

        Returns:
            Mapping ``equation_name -> residual``.
        """
        _ = (set_manager, parameters, variables)
        return {}

    def __repr__(self) -> str:
        """String representation."""
        sets_str = f"[{', '.join(self.required_sets)}]" if self.required_sets else "[]"
        return (
            f"Block {self.name}{sets_str}: "
            f"{len(self.parameters)} params, "
            f"{len(self.variables)} vars, "
            f"{len(self.equations)} eqs"
        )
