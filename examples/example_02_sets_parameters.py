"""Example 2: Working with Sets and Parameters

This example demonstrates how to:
1. Create and manipulate sets
2. Work with parameters and multi-dimensional indexing
3. Use SetManager for complex operations
"""

import numpy as np

from equilibria.core import Parameter, Set, SetManager


def main():
    """Demonstrate set and parameter operations."""
    print("=" * 70)
    print("Example 2: Working with Sets and Parameters")
    print("=" * 70)

    # Create sets
    print("\n" + "-" * 70)
    print("Step 1: Creating Sets")
    print("-" * 70)

    sectors = Set(
        name="J",
        elements=("agr", "mfg", "svc", "ene"),
        description="Production sectors",
    )
    print(f"\nCreated set: {sectors}")

    factors = Set(
        name="I",
        elements=("labor", "capital", "land"),
        description="Factors of production",
    )
    print(f"Created set: {factors}")

    regions = Set(
        name="R",
        elements=("USA", "EUR", "CHN", "ROW"),
        description="Regions",
    )
    print(f"Created set: {regions}")

    # Create SetManager
    print("\n" + "-" * 70)
    print("Step 2: SetManager Operations")
    print("-" * 70)

    manager = SetManager()
    manager.add(sectors)
    manager.add(factors)
    manager.add(regions)

    print(f"\nAdded {len(list(manager.list_sets()))} sets to manager:")
    for set_name in manager.list_sets():
        print(f"  - {set_name}")

    # Cartesian product
    print("\n" + "-" * 70)
    print("Step 3: Cartesian Products")
    print("-" * 70)

    print("\nCartesian product of J × I (first 10 combinations):")
    count = 0
    for combo in manager.product("J", "I"):
        print(f"  {combo}")
        count += 1
        if count >= 10:
            print(f"  ... ({len(sectors) * len(factors) - 10} more)")
            break

    print(
        f"\nTotal combinations: {len(sectors)} × {len(factors)} = {len(sectors) * len(factors)}"
    )

    # Create parameters
    print("\n" + "-" * 70)
    print("Step 4: Creating Parameters")
    print("-" * 70)

    # 1D parameter
    sigma = Parameter(
        name="sigma_VA",
        value=np.array([0.8, 0.9, 0.7, 0.85]),
        domains=("J",),
        description="CES elasticity by sector",
    )
    print(f"\n1D Parameter: {sigma}")
    print(f"  Shape: {sigma.shape()}")
    print(f"  Values: {sigma.value}")

    # 2D parameter
    io_coeffs = Parameter(
        name="a_io",
        value=np.array(
            [
                [0.15, 0.20, 0.10, 0.25],  # agr
                [0.10, 0.30, 0.15, 0.20],  # mfg
                [0.05, 0.10, 0.20, 0.15],  # svc
                [0.20, 0.15, 0.10, 0.30],  # ene
            ]
        ),
        domains=("J", "J"),
        description="Input-output coefficients",
    )
    print(f"\n2D Parameter: {io_coeffs}")
    print(f"  Shape: {io_coeffs.shape()}")
    print(f"  Domains: {io_coeffs.domains}")

    # Access parameter values
    print("\n" + "-" * 70)
    print("Step 5: Accessing Parameter Values")
    print("-" * 70)

    print(f"\nAccessing sigma_VA values:")
    for i, sector in enumerate(sectors):
        val = sigma.get_value(i)
        print(f"  sigma_VA[{sector}] = {val}")

    print(f"\nAccessing IO coefficients:")
    print(f"  a_io[agr, mfg] = {io_coeffs.get_value(0, 1):.2f}")
    print(f"  a_io[mfg, mfg] = {io_coeffs.get_value(1, 1):.2f}")
    print(f"  a_io[svc, ene] = {io_coeffs.get_value(2, 3):.2f}")

    # Modify parameter values
    print("\n" + "-" * 70)
    print("Step 6: Modifying Parameters")
    print("-" * 70)

    print(f"\nOriginal sigma_VA[agr] = {sigma.get_value(0)}")
    sigma.set_value(0.85, 0)
    print(f"Modified sigma_VA[agr] = {sigma.get_value(0)}")

    # SetManager summary
    print("\n" + "-" * 70)
    print("Step 7: SetManager Summary")
    print("-" * 70)

    summary = manager.summary()
    print(f"\nTotal sets: {summary['total_sets']}")
    print("\nSet details:")
    for set_name, info in summary["sets"].items():
        print(f"  {set_name}: {info['elements']} elements - {info['description']}")

    # Subset example
    print("\n" + "-" * 70)
    print("Step 8: Creating Subsets")
    print("-" * 70)

    industrial = Set(
        name="J_IND",
        elements=("mfg", "ene"),
        description="Industrial sectors",
        domain="J",
    )
    print(f"\nCreated subset: {industrial}")
    print(f"  Domain: {industrial.domain}")
    print(f"  Elements: {industrial.elements}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
