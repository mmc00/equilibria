"""Example 6: Using the Pyomo Backend

This example demonstrates how to:
1. Build a model and translate it to Pyomo
2. Solve the model using IPOPT
3. Access and analyze the solution
"""

import numpy as np

from equilibria import Model
from equilibria.backends import PyomoBackend, Solution
from equilibria.blocks import CESValueAdded, LeontiefIntermediate
from equilibria.core import Set


def main():
    """Demonstrate Pyomo backend usage."""
    print("=" * 70)
    print("Example 6: Using the Pyomo Backend")
    print("=" * 70)

    # Create a simple model
    print("\n" + "-" * 70)
    print("Step 1: Create Model")
    print("-" * 70)

    model = Model(
        name="PyomoDemo",
        description="Simple model for Pyomo backend demo",
    )

    # Add sets (must match block requirements: J for sectors, I for factors)
    sectors = Set(
        name="J",
        elements=("sec1", "sec2"),
        description="Sectors",
    )
    factors = Set(
        name="I",
        elements=("labor", "capital"),
        description="Factors",
    )

    model.add_sets([sectors, factors])
    print(f"\nCreated model with {len(sectors)} sectors and {len(factors)} factors")
    print(f"\nCreated model with {len(sectors)} sectors and {len(factors)} factors")

    # Add blocks
    print("\n" + "-" * 70)
    print("Step 2: Add Blocks")
    print("-" * 70)

    ces_block = CESValueAdded(sigma=0.8, name="CES_VA")
    model.add_block(ces_block)
    print(f"\nAdded CES block with {len(ces_block.parameters)} parameters")

    leontief_block = LeontiefIntermediate(name="Leontief_INT")
    model.add_block(leontief_block)
    print(f"Added Leontief block with {len(leontief_block.parameters)} parameters")

    # Show model stats
    stats = model.statistics
    param_count = len(model.parameter_manager.list_params())
    print(f"\nModel has {stats.variables} variables and {param_count} parameters")

    # Create Pyomo backend
    print("\n" + "-" * 70)
    print("Step 3: Create Pyomo Backend")
    print("-" * 70)

    try:
        backend = PyomoBackend(solver="ipopt")
        print(f"\nCreated backend: {backend}")

        # List available solvers
        available = backend.list_available_solvers()
        print(f"Available solvers: {available}")

    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install IPOPT to run this example")
        return

    # Build the model
    print("\n" + "-" * 70)
    print("Step 4: Build Pyomo Model")
    print("-" * 70)

    backend.build(model)
    print("\n✓ Model built successfully")
    print(f"Pyomo model has {len(backend.pyomo_model.component_map())} components")

    # Show Pyomo model structure
    print("\nPyomo model components:")
    for name in backend.pyomo_model.component_map():
        comp = getattr(backend.pyomo_model, name)
        comp_type = type(comp).__name__
        print(f"  {name}: {comp_type}")

    # Note about solving
    print("\n" + "-" * 70)
    print("Step 5: Solving (Note)")
    print("-" * 70)

    print("\nNote: This model doesn't have equations defined yet.")
    print("To solve, you would need to:")
    print("  1. Define equations in your blocks")
    print("  2. Add constraints to the Pyomo model")
    print("  3. Call backend.solve()")
    print("\nExample:")
    print("  solution = backend.solve()")
    print("  print(solution.status)")
    print("  print(solution.variables)")

    # Demonstrate Solution class
    print("\n" + "-" * 70)
    print("Step 6: Solution Object Demo")
    print("-" * 70)

    # Create a mock solution
    mock_solution = Solution(
        model_name="Demo",
        status="optimal",
        objective_value=0.0,
        variables={
            "VA": np.array([100.0, 150.0]),
            "FD": np.array([[50.0, 30.0], [50.0, 120.0]]),
        },
        solve_time=1.23,
        iterations=42,
    )

    print(f"\nMock solution: {mock_solution}")
    print(f"\nVariable values:")
    for var_name, values in mock_solution.variables.items():
        print(f"  {var_name}: {values}")

    # Compare solutions
    print("\n" + "-" * 70)
    print("Step 7: Comparing Solutions")
    print("-" * 70)

    # Create another solution with slight differences
    solution2 = Solution(
        model_name="Demo",
        status="optimal",
        variables={
            "VA": np.array([105.0, 150.0]),  # 5% difference
            "FD": np.array([[50.0, 30.0], [50.0, 120.0]]),
        },
    )

    comparison = mock_solution.compare(solution2, tolerance=1e-6)
    print(f"\nComparison results:")
    print(f"  Solutions are equal: {comparison['is_equal']}")
    print(f"  Tolerance: {comparison['tolerance']}")

    if comparison["differences"]:
        print(f"\n  Differences found:")
        for var_name, diff_info in comparison["differences"].items():
            print(f"    {var_name}: {diff_info}")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)
    print("\nTo use the Pyomo backend in practice:")
    print("  1. Install IPOPT: brew install ipopt (macOS)")
    print("  2. Define equations in your blocks")
    print("  3. Build and solve: backend.build(model); solution = backend.solve()")
    print("=" * 70)


if __name__ == "__main__":
    main()
