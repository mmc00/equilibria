"""Example 5: Complete CGE Model with Multiple Blocks

This example demonstrates how to:
1. Build a complete CGE model with multiple interacting blocks
2. View detailed model structure
3. Access and modify parameters and variables
"""


from equilibria import Model
from equilibria.blocks import (
    CESValueAdded,
    CETTransformation,
    LeontiefIntermediate,
)
from equilibria.core import Set


def main():
    """Build a complete CGE model."""
    print("=" * 70)
    print("Example 5: Complete CGE Model")
    print("=" * 70)

    # Create model
    model = Model(
        name="OpenEconomyCGE",
        description="Open economy CGE with production",
    )

    # Define sets
    print("\n" + "-" * 70)
    print("Step 1: Define Sets")
    print("-" * 70)

    sectors = Set(
        name="J",
        elements=("AGR", "MFG", "SRV"),
        description="Production sectors",
    )

    factors = Set(
        name="F",
        elements=("LAB", "CAP"),
        description="Factors of production",
    )

    commodities = Set(
        name="I",
        elements=("AGR", "MFG", "SRV"),
        description="Commodities",
    )

    model.add_sets([sectors, factors, commodities])

    print("\nSets defined:")
    for set_name in model.set_manager.list_sets():
        s = model.set_manager.get(set_name)
        print(f"  {set_name}: {len(s)} elements - {s.description}")

    # Add all blocks
    print("\n" + "-" * 70)
    print("Step 2: Add Production Blocks")
    print("-" * 70)

    blocks = [
        CESValueAdded(sigma=0.8, name="CES_VA"),
        LeontiefIntermediate(name="Leontief_INT"),
        CETTransformation(omega=2.0, name="CET"),
    ]

    for block in blocks:
        print(f"\nAdding: {block.name}")
        print(f"  Sets: {block.required_sets}")
        print(f"  Params: {len(block.parameters)}")
        print(f"  Vars: {len(block.variables)}")
        model.add_block(block)
        print("  ✓ Added")

    # Model statistics
    print("\n" + "-" * 70)
    print("Step 3: Model Statistics")
    print("-" * 70)

    stats = model.statistics
    print(f"\n{model}")
    print("\nDetailed Statistics:")
    print(f"  Total scalar variables: {stats.variables}")
    print(f"  Total scalar equations: {stats.equations}")
    print(f"  Degrees of freedom: {stats.degrees_of_freedom}")
    print(f"  Number of blocks: {stats.blocks}")

    # Access parameters
    print("\n" + "-" * 70)
    print("Step 4: Access and Modify Parameters")
    print("-" * 70)

    print("\nAll parameters in model:")
    for param_name in sorted(model.parameter_manager.list_params()):
        param = model.get_parameter(param_name)
        print(f"  {param_name:15s} shape={param.shape()}")

    # Modify a parameter
    print("\nModifying CES elasticity parameter:")
    sigma_va = model.get_parameter("sigma_VA")
    print(f"  Original values: {sigma_va.value}")

    # Change elasticity for manufacturing
    sigma_va.value[1] = 1.0  # MFG sector
    print(f"  Modified values: {sigma_va.value}")
    print("  ✓ sigma_VA[MFG] changed from 0.8 to 1.0")

    # Access variables
    print("\n" + "-" * 70)
    print("Step 5: Access and Modify Variables")
    print("-" * 70)

    print("\nAll variables in model:")
    for var_name in sorted(model.variable_manager.list_vars()):
        var = model.get_variable(var_name)
        fixed_str = " [FIXED]" if var.is_fixed() else ""
        print(f"  {var_name:10s} shape={var.shape()}{fixed_str}")

    # Fix a variable
    print("\nFixing variable 'WF' (factor prices):")
    wf_var = model.get_variable("WF")
    print(f"  Before: is_fixed={wf_var.is_fixed()}")
    wf_var.fix(1.0)
    print(f"  After:  is_fixed={wf_var.is_fixed()}, value={wf_var.value}")

    # Model summary
    print("\n" + "-" * 70)
    print("Step 6: Full Model Summary")
    print("-" * 70)

    summary = model.summary()

    print(f"\nModel: {summary['name']}")
    print(f"Description: {summary['description']}")

    print(f"\nSets ({summary['sets']['total_sets']}):")
    for set_name, set_info in summary["sets"]["sets"].items():
        print(f"  {set_name}: {set_info['elements']} elements")

    print(f"\nParameters ({summary['parameters']['total_parameters']}):")
    for param_name, param_info in summary["parameters"]["parameters"].items():
        print(f"  {param_name}: {param_info['shape']}")

    print(f"\nVariables ({summary['variables']['total_variables']}):")
    for var_name, var_info in summary["variables"]["variables"].items():
        fixed_str = " [FIXED]" if var_info["fixed"] else ""
        print(f"  {var_name}: {var_info['shape']}{fixed_str}")

    print(f"\nBlocks ({len(summary['blocks'])}):")
    for block_info in summary["blocks"]:
        print(f"  {block_info['name']}: {block_info['description']}")

    # Block details
    print("\n" + "-" * 70)
    print("Step 7: Detailed Block Information")
    print("-" * 70)

    for block in model.blocks:
        print(f"\n{block.name}:")
        print(f"  Description: {block.description}")
        print(f"  Required sets: {block.required_sets}")

        if block.parameters:
            print("  Parameters:")
            for param_name, param_spec in block.parameters.items():
                domains_str = (
                    f"[{', '.join(param_spec.domains)}]"
                    if param_spec.domains
                    else "scalar"
                )
                print(f"    - {param_name} {domains_str}")

        if block.variables:
            print("  Variables:")
            for var_name, var_spec in block.variables.items():
                domains_str = (
                    f"[{', '.join(var_spec.domains)}]" if var_spec.domains else "scalar"
                )
                bounds_str = f"[{var_spec.lower}, {var_spec.upper}]"
                print(f"    - {var_name} {domains_str} {bounds_str}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nThis model demonstrates:")
    print("  ✓ Multiple interacting blocks")
    print("  ✓ Production (CES + Leontief)")
    print("  ✓ CET transformation")
    print("  ✓ Parameter and variable management")
    print("  ✓ Model introspection and statistics")
    print("=" * 70)


if __name__ == "__main__":
    main()
