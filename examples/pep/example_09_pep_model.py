"""Example 9: PEP Model Template

This example demonstrates how to:
1. Create a PEP model using the template
2. Extract sets from SAM
3. View model structure
"""

from equilibria.templates import PEP1R


def main():
    """Demonstrate PEP template usage."""
    print("=" * 70)
    print("Example 9: PEP Model Template")
    print("=" * 70)

    # Create PEP template
    print("\n" + "-" * 70)
    print("Step 1: Create PEP Template")
    print("-" * 70)

    template = PEP1R()
    print(f"\nCreated template: {template.name}")
    print(f"Description: {template.description}")

    # Extract sets from SAM
    print("\n" + "-" * 70)
    print("Step 2: Extract Sets from SAM")
    print("-" * 70)

    # Use default PEP sets (hardcoded for standard PEP model)
    template.use_default_pep_sets()

    print("\nExtracted sets:")
    print(f"  Sectors: {template.sectors}")
    print(f"  Commodities: {template.commodities}")
    print(f"  Labor types: {template.labor_types}")
    print(f"  Capital types: {template.capital_types}")
    print(f"  Households: {template.households}")

    # Create model
    print("\n" + "-" * 70)
    print("Step 3: Create Model")
    print("-" * 70)

    model = template.create_model()
    print(f"\nCreated model: {model.name}")

    # Show model statistics
    stats = model.statistics
    print("\nModel Statistics:")
    print(f"  Variables: {stats.variables}")
    print(f"  Equations: {stats.equations}")
    print(f"  Blocks: {stats.blocks}")

    # Show sets
    print("\nSets in model:")
    for set_name in model.set_manager.list_sets():
        s = model.set_manager.get(set_name)
        print(f"  {set_name}: {len(s)} elements")

    # Show blocks
    print("\nBlocks in model:")
    for block in model.blocks:
        print(f"  {block.name}: {block.description}")

    # Show template info
    print("\n" + "-" * 70)
    print("Step 4: Template Info")
    print("-" * 70)

    info = template.get_info()
    print("\nTemplate configuration:")
    print(f"  Variant: {info.get('variant')}")
    print("  Features:")
    for feature, enabled in info.get("features", {}).items():
        print(f"    {feature}: {enabled}")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)
    print("\nThe PEP template provides:")
    print("  ✓ Complete CGE model structure")
    print("  ✓ 4 household types")
    print("  ✓ Full tax system")
    print("  ✓ Trade with margins")
    print("  ✓ Ready for calibration")
    print("=" * 70)


if __name__ == "__main__":
    main()
