"""Main Model class for equilibria CGE framework.

The Model class is the central component that assembles blocks,
manages sets, parameters, variables, and equations, and provides
interfaces for calibration and solving.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from equilibria.blocks.base import Block
from equilibria.core.equations import Equation, EquationManager
from equilibria.core.parameters import Parameter, ParameterManager
from equilibria.core.sets import Set, SetManager
from equilibria.core.variables import Variable, VariableManager


class ModelStatistics(BaseModel):
    """Statistics for a CGE model.

    Provides counts of variables, equations, degrees of freedom,
    and other model metrics.

    Attributes:
        variables: Total number of scalar variables
        equations: Total number of scalar equations
        degrees_of_freedom: DOF (variables - equations)
        blocks: Number of blocks
        sparsity: Sparsity ratio (0-1)
    """

    variables: int = Field(default=0, description="Total scalar variables")
    equations: int = Field(default=0, description="Total scalar equations")
    degrees_of_freedom: int = Field(default=0, description="Degrees of freedom")
    blocks: int = Field(default=0, description="Number of blocks")
    sparsity: float = Field(default=0.0, description="Sparsity ratio")

    model_config = {"frozen": True}


class Model(BaseModel):
    """CGE Model class.

    The Model class assembles equation blocks, manages all model
    components (sets, parameters, variables, equations), and provides
    interfaces for calibration and solving.

    Attributes:
        name: Model identifier
        description: Model description
        set_manager: Manager for all sets
        parameter_manager: Manager for all parameters
        variable_manager: Manager for all variables
        equation_manager: Manager for all equations
        blocks: List of blocks in the model

    Example:
        >>> model = Model(name="MyCGE")
        >>> model.add_sets([
        ...     Set(name="J", elements=["agr", "mfg", "svc"]),
        ... ])
        >>> model.add_block(CESValueAdded(sigma=0.8))
        >>> print(model.statistics)
    """

    name: str = Field(..., description="Model identifier")
    description: str = Field(default="", description="Model description")
    set_manager: SetManager = Field(
        default_factory=SetManager, description="Set manager"
    )
    parameter_manager: ParameterManager = Field(
        default_factory=ParameterManager, description="Parameter manager"
    )
    variable_manager: VariableManager = Field(
        default_factory=VariableManager, description="Variable manager"
    )
    equation_manager: EquationManager = Field(
        default_factory=EquationManager, description="Equation manager"
    )
    blocks: list[Block] = Field(default_factory=list, description="Model blocks")

    model_config = {"arbitrary_types_allowed": True}

    # (kind, name) -> name of the block that declared it first.
    _owner: dict[tuple[str, str], str] = PrivateAttr(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        """Initialize model with proper manager linkage."""
        super().__init__(**data)
        # Link managers to set_manager
        self.parameter_manager._set_manager = self.set_manager
        self.variable_manager._set_manager = self.set_manager
        self.equation_manager._set_manager = self.set_manager

    def add_set(self, set_obj: Set) -> None:
        """Add a set to the model.

        Args:
            set_obj: Set to add
        """
        self.set_manager.add(set_obj)

    def add_sets(self, sets: list[Set]) -> None:
        """Add multiple sets to the model.

        Args:
            sets: List of sets to add
        """
        for set_obj in sets:
            self.add_set(set_obj)

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the model.

        Args:
            param: Parameter to add
        """
        self.parameter_manager.add(param)

    def add_variable(self, var: Variable) -> None:
        """Add a variable to the model.

        Args:
            var: Variable to add
        """
        self.variable_manager.add(var)

    def add_equation(self, eq: Equation) -> None:
        """Add an equation to the model.

        Args:
            eq: Equation to add
        """
        self.equation_manager.add(eq)

    def add_block(self, block: Block) -> None:
        """Add a block to the model.

        This validates that required sets exist, then calls the block's
        setup method to add parameters, variables, and equations.

        Two blocks may declare the same parameter or variable only if both
        declarations are identical (value, domains, bounds...): a second block
        cannot silently change what the first one declared. An equation has one
        owner. The block is checked whole before anything is registered, so a
        rejected block leaves the model untouched.

        Args:
            block: Block to add

        Raises:
            ValueError: If required sets are missing, or if the block declares a
                parameter or variable differently from an earlier block, or an
                equation an earlier block already declared.
        """
        # Validate required sets exist
        block.validate_sets(self.set_manager)

        # Create parameter and variable dicts for the block to populate
        block_params: dict[str, Parameter] = {}
        block_vars: dict[str, Variable] = {}

        # Call block setup to get equations and populate params/vars
        equations = block.setup(self.set_manager, block_params, block_vars)

        new_params = [
            p
            for p in block_params.values()
            if not self._check_redeclaration("parameter", p, block)
        ]
        new_vars = [
            v
            for v in block_vars.values()
            if not self._check_redeclaration("variable", v, block)
        ]
        for eq in equations:
            if eq.name in self.equation_manager:
                raise self._redeclared(
                    "equation", eq.name, block, "an equation has one owner"
                )

        for kind, items in (
            ("parameter", new_params),
            ("variable", new_vars),
            ("equation", equations),
        ):
            for item in items:
                _MANAGERS[kind](self).add(item)
                self._owner[(kind, item.name)] = block.name

        # Store the block
        self.blocks.append(block)

    def owner_of(self, kind: str, name: str) -> str | None:
        """Name of the block that declared ``name`` first (``kind`` is
        "parameter", "variable" or "equation"); None if no block did."""
        return self._owner.get((kind, name))

    def _check_redeclaration(self, kind: str, new: Any, block: Block) -> bool:
        """Whether ``new`` is already registered with the same declaration.

        Returns False if the name is new and True if an earlier block declared it
        identically. Raises ValueError if an earlier block declared it differently.
        """
        manager = _MANAGERS[kind](self)
        if new.name not in manager:
            return False
        old = manager.get(new.name)
        for field in _DECLARATION_FIELDS[kind]:
            if not _same(getattr(old, field), getattr(new, field)):
                raise self._redeclared(
                    kind,
                    new.name,
                    block,
                    f"with a different {field}: one name, one declaration",
                )
        return True

    def _redeclared(self, kind: str, name: str, block: Block, why: str) -> ValueError:
        return ValueError(
            f"{kind} {name} is declared by block {self.owner_of(kind, name)!r} "
            f"and again by {block.name!r} {why}"
        )

    def check_uses(self) -> None:
        """Check that every variable a block uses (``Block.uses``) is declared.

        Call it once every block is added: a block may use what a later block
        declares.

        Raises:
            ValueError: If a used variable is declared by no block.
        """
        missing = [
            f"block {b.name!r} uses {v}"
            for b in self.blocks
            for v in b.uses
            if v not in self.variable_manager
        ]
        if missing:
            them = "it" if len(missing) == 1 else "them"
            raise ValueError("; ".join(missing) + f", but no block declares {them}")

    def add_blocks(self, blocks: list[Block]) -> None:
        """Add multiple blocks to the model.

        Args:
            blocks: List of blocks to add
        """
        for block in blocks:
            self.add_block(block)

    def get_parameter(self, name: str) -> Parameter:
        """Get a parameter by name.

        Args:
            name: Parameter name

        Returns:
            Parameter object
        """
        return self.parameter_manager.get(name)

    def get_variable(self, name: str) -> Variable:
        """Get a variable by name.

        Args:
            name: Variable name

        Returns:
            Variable object
        """
        return self.variable_manager.get(name)

    def get_equation(self, name: str) -> Equation:
        """Get an equation by name.

        Args:
            name: Equation name

        Returns:
            Equation object
        """
        return self.equation_manager.get(name)

    @property
    def statistics(self) -> ModelStatistics:
        """Calculate model statistics.

        Returns:
            ModelStatistics with counts and metrics
        """
        n_vars = self.variable_manager.get_total_count()
        n_eqs = self.equation_manager.get_total_count()
        dof = n_vars - n_eqs

        # Calculate sparsity (simplified - would need Jacobian analysis)
        sparsity = 0.0
        if n_vars > 0 and n_eqs > 0:
            # Placeholder - real sparsity requires analyzing Jacobian
            sparsity = 1.0 - min(n_vars, n_eqs) / (n_vars * n_eqs)

        return ModelStatistics(
            variables=n_vars,
            equations=n_eqs,
            degrees_of_freedom=dof,
            blocks=len(self.blocks),
            sparsity=sparsity,
        )

    def summary(self) -> dict[str, Any]:
        """Return comprehensive model summary.

        Returns:
            Dictionary with model information
        """
        stats = self.statistics

        return {
            "name": self.name,
            "description": self.description,
            "statistics": stats.model_dump(),
            "sets": self.set_manager.summary(),
            "parameters": self.parameter_manager.summary(),
            "variables": self.variable_manager.summary(),
            "equations": self.equation_manager.summary(),
            "blocks": [block.get_info() for block in self.blocks],
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.statistics
        return (
            f"Model '{self.name}': "
            f"{stats.variables} vars, "
            f"{stats.equations} eqs, "
            f"DOF={stats.degrees_of_freedom}, "
            f"{stats.blocks} blocks"
        )


# The fields that make up a declaration: two blocks sharing a name must agree on
# all of them. Parameters have no bounds.
_DECLARATION_FIELDS: dict[str, tuple[str, ...]] = {
    "parameter": ("value", "domains", "mutable"),
    "variable": ("value", "domains", "domain", "lower", "upper"),
}

_MANAGERS: dict[str, Any] = {
    "parameter": lambda m: m.parameter_manager,
    "variable": lambda m: m.variable_manager,
    "equation": lambda m: m.equation_manager,
}


def _same(a: Any, b: Any) -> bool:
    """Equality of a declaration field: arrays cell by cell (NaN == NaN), and a
    scalar equals an array holding that value in every cell (``lower=0.0`` is
    ``lower=np.zeros(n)``)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            if a.ndim and b.ndim:
                return False
            a, b = np.broadcast_arrays(a, b)
        if a.dtype.kind in "fc" and b.dtype.kind in "fc":
            return bool(np.array_equal(a, b, equal_nan=True))
        return bool(np.array_equal(a, b))
    return a == b
