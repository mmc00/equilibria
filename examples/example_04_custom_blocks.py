"""Example 4: Creating Custom Blocks

This example demonstrates how to:
1. Create a custom block by inheriting from Block
2. Define required sets, parameters, and variables
3. Implement the setup method
4. Register and use the custom block
"""

from typing import Any

import numpy as np
from pydantic import Field

from equilibria.blocks import Block, ParameterSpec, VariableSpec, register_block
from equilibria.core import Equation, Parameter, Variable
from equilibria.core.sets import SetManager


@register_block
class SimpleDemand(Block):
    """Simple demand block with Cobb-Douglas utility.

    This is a custom block that demonstrates how to create
    your own economic behavior blocks.

    Required sets:
        - I: Commodities

    Equations:
        1. Demand: QD[i] = alpha[i] * Y / P[i]
        where alpha are expenditure shares, Y is income, P are prices
    """

    name: str = Field(default="SimpleDemand", description="Block name")
    description: str = Field(
        default="Simple Cobb-Douglas demand", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        # Define required sets
        self.required_sets = ["I"]

        # Define parameters
        self.parameters = {
            "alpha": ParameterSpec(
                name="alpha",
                domains=("I",),
                description="Expenditure share parameter",
            ),
        }

        # Define variables
        self.variables = {
            "QD": VariableSpec(
                name="QD",
                domains=("I",),
                lower=0.0,
                description="Commodity demand",
            ),
            "Y": VariableSpec(
                name="Y",
                lower=0.0,
                description="Household income",
            ),
            "P": VariableSpec(
                name="P",
                domains=("I",),
                lower=0.0,
                description="Commodity prices",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the demand block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Create parameters
        # Equal expenditure shares
        alpha_vals = np.ones((n_comm,)) / n_comm
        parameters["alpha"] = Parameter(
            name="alpha",
            value=alpha_vals,
            domains=("I",),
            description="Expenditure shares",
        )

        # Create variables
        qd_vals = np.ones((n_comm,))
        variables["QD"] = Variable(
            name="QD",
            value=qd_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity demand",
        )

        y_val = np.array([100.0])
        variables["Y"] = Variable(
            name="Y",
            value=y_val,
            lower=0.0,
            description="Household income",
        )

        p_vals = np.ones((n_comm,))
        variables["P"] = Variable(
            name="P",
            value=p_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity prices",
        )

        # Return empty list (equations would be defined here in a full implementation)
        return []


def main():
    """Demonstrate custom block creation."""
    print("=" * 70)
    print("Example 4: Creating Custom Blocks")
    print("=" * 70)

    # Show the registered block
    print("\n" + "-" * 70)
    print("Step 1: Block Registration")
    print("-" * 70)

    from equilibria.blocks import get_registry

    registry = get_registry()
    print(f"\nRegistered blocks:")
    for block_name in registry.list_blocks():
        print(f"  - {block_name}")

    # Create instance of custom block
    print("\n" + "-" * 70)
    print("Step 2: Creating Custom Block Instance")
    print("-" * 70)

    demand_block = SimpleDemand(name="SimpleDemand")
    print(f"\nCreated block: {demand_block.name}")
    print(f"Description: {demand_block.description}")
    print(f"Required sets: {demand_block.required_sets}")

    # Get block info
    print("\n" + "-" * 70)
    print("Step 3: Block Metadata")
    print("-" * 70)

    info = demand_block.get_info()
    print(f"\nBlock Information:")
    print(f"  Name: {info['name']}")
    print(f"  Description: {info['description']}")
    print(f"  Required sets: {info['required_sets']}")

    print(f"\nParameters:")
    for param_name, param_spec in info["parameters"].items():
        print(f"  {param_name}:")
        print(f"    Domains: {param_spec['domains']}")
        print(f"    Description: {param_spec['description']}")

    print(f"\nVariables:")
    for var_name, var_spec in info["variables"].items():
        print(f"  {var_name}:")
        print(f"    Domains: {var_spec['domains']}")
        print(f"    Description: {var_spec['description']}")

    # Use the block in a model
    print("\n" + "-" * 70)
    print("Step 4: Using Custom Block in Model")
    print("-" * 70)

    from equilibria import Model
    from equilibria.core import Set

    model = Model(name="CustomModel", description="Model with custom block")

    # Add sets
    commodities = Set(
        name="I",
        elements=("food", "manuf", "serv"),
        description="Commodities",
    )
    model.add_set(commodities)
    print(f"\nAdded set: {commodities}")

    # Add custom block
    print(f"\nAdding custom block to model...")
    model.add_block(demand_block)
    print(f"  ✓ Block added successfully")

    # Show model statistics
    print("\n" + "-" * 70)
    print("Step 5: Model Statistics")
    print("-" * 70)

    stats = model.statistics
    print(f"\nModel Statistics:")
    print(f"  Variables: {stats.variables}")
    print(f"  Parameters: {len(model.parameter_manager.list_params())}")
    print(f"  Blocks: {stats.blocks}")

    print(f"\nParameters in model:")
    for param_name in model.parameter_manager.list_params():
        param = model.get_parameter(param_name)
        print(f"  {param_name}: shape {param.shape()}")

    print(f"\nVariables in model:")
    for var_name in model.variable_manager.list_vars():
        var = model.get_variable(var_name)
        print(f"  {var_name}: shape {var.shape()}")

    # Create block using registry
    print("\n" + "-" * 70)
    print("Step 6: Creating Block from Registry")
    print("-" * 70)

    block_from_registry = registry.create("SimpleDemand")
    print(f"\nCreated block from registry: {block_from_registry.name}")
    print(f"  Same type: {type(block_from_registry) == type(demand_block)}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nTo create your own block:")
    print("  1. Inherit from Block")
    print("  2. Define required_sets, parameters, variables")
    print("  3. Implement setup() method")
    print("  4. Use @register_block decorator (optional)")
    print("=" * 70)


if __name__ == "__main__":
    main()
