"""Example 1: Basic CGE Model Setup

This example demonstrates how to:
1. Create a simple CGE model with sets
2. Add production and trade blocks
3. View model statistics
"""

from equilibria import Model
from equilibria.blocks import (
    ArmingtonCES,
    CETExports,
    CETTransformation,
    CESValueAdded,
    LeontiefIntermediate,
)
from equilibria.core import Set


def main():
    """Create and configure a basic CGE model."""
    print("=" * 70)
    print("Example 1: Basic CGE Model Setup")
    print("=" * 70)

    # Create a new model
    model = Model(
        name="SimpleCGE",
        description="A simple 3-sector CGE model",
    )
    print(f"\nCreated model: {model.name}")
    print(f"Description: {model.description}")

    # Define sets
    print("\n" + "-" * 70)
    print("Step 1: Define Sets")
    print("-" * 70)

    sectors = Set(
        name="J",
        elements=("agr", "mfg", "svc"),
        description="Production sectors",
    )
    print(f"\nSectors (J): {sectors}")

    factors = Set(
        name="I",
        elements=("labor", "capital"),
        description="Factors of production",
    )
    print(f"Factors (I): {factors}")

    commodities = Set(
        name="COMM",
        elements=("agr", "mfg", "svc"),
        description="Commodities",
    )
    print(f"Commodities (COMM): {commodities}")

    # Add sets to model
    model.add_set(sectors)
    model.add_set(factors)
    model.add_set(commodities)
    print("\nSets added to model successfully!")

    # Add production blocks
    print("\n" + "-" * 70)
    print("Step 2: Add Production Blocks")
    print("-" * 70)

    # CES Value-Added block
    ces_va = CESValueAdded(
        sigma=0.8,
        name="CES_VA",
        description="CES value-added production",
    )
    print(f"\nAdding block: {ces_va.name}")
    print(f"  Required sets: {ces_va.required_sets}")
    print(f"  Parameters: {list(ces_va.parameters.keys())}")
    print(f"  Variables: {list(ces_va.variables.keys())}")

    model.add_block(ces_va)
    print("  ✓ Block added successfully")

    # Leontief Intermediate block
    leontief = LeontiefIntermediate(
        name="Leontief_INT",
        description="Leontief intermediate inputs",
    )
    print(f"\nAdding block: {leontief.name}")
    print(f"  Required sets: {leontief.required_sets}")
    print(f"  Parameters: {list(leontief.parameters.keys())}")
    print(f"  Variables: {list(leontief.variables.keys())}")

    model.add_block(leontief)
    print("  ✓ Block added successfully")

    # CET Transformation block
    cet = CETTransformation(
        omega=2.0,
        name="CET",
        description="CET output transformation",
    )
    print(f"\nAdding block: {cet.name}")
    print(f"  Required sets: {cet.required_sets}")
    print(f"  Parameters: {list(cet.parameters.keys())}")
    print(f"  Variables: {list(cet.variables.keys())}")

    model.add_block(cet)
    print("  ✓ Block added successfully")

    # Display model statistics
    print("\n" + "-" * 70)
    print("Step 3: Model Statistics")
    print("-" * 70)

    stats = model.statistics
    print("\nModel Statistics:")
    print(f"  Total scalar variables: {stats.variables}")
    print(f"  Total scalar equations: {stats.equations}")
    print(f"  Degrees of freedom: {stats.degrees_of_freedom}")
    print(f"  Number of blocks: {stats.blocks}")
    print(f"  Sparsity: {stats.sparsity:.2%}")

    # Display model summary
    print("\n" + "-" * 70)
    print("Step 4: Model Summary")
    print("-" * 70)

    summary = model.summary()
    print(f"\nModel: {summary['name']}")
    print(f"Description: {summary['description']}")

    print("\nSets:")
    for set_name, set_info in summary["sets"]["sets"].items():
        print(f"  {set_name}: {set_info['elements']} elements")

    print("\nParameters:")
    for param_name, param_info in summary["parameters"]["parameters"].items():
        print(f"  {param_name}: shape {param_info['shape']}")

    print("\nVariables:")
    for var_name, var_info in summary["variables"]["variables"].items():
        print(f"  {var_name}: shape {var_info['shape']}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)

    return model


if __name__ == "__main__":
    model = main()
