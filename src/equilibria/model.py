"""Main Model class for equilibria CGE framework.

The Model class is the central component that assembles blocks,
manages sets, parameters, variables, and equations, and provides
interfaces for calibration and solving.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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

        Args:
            block: Block to add

        Raises:
            ValueError: If required sets are missing
        """
        # Validate required sets exist
        block.validate_sets(self.set_manager)

        # Create parameter and variable dicts for the block to populate
        block_params: dict[str, Parameter] = {}
        block_vars: dict[str, Variable] = {}

        # Call block setup to get equations and populate params/vars
        equations = block.setup(self.set_manager, block_params, block_vars)

        # Add block's parameters, variables, and equations to model
        for param in block_params.values():
            if param.name not in self.parameter_manager:
                self.add_parameter(param)

        for var in block_vars.values():
            if var.name not in self.variable_manager:
                self.add_variable(var)

        for eq in equations:
            if eq.name not in self.equation_manager:
                self.add_equation(eq)

        # Store the block
        self.blocks.append(block)

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
