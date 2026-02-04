"""Example 7: Using Model Templates

This example demonstrates how to:
1. Use the SimpleOpenEconomy template
2. Customize template parameters
3. Create models with different configurations
"""

from equilibria.templates import SimpleOpenEconomy


def main():
    """Demonstrate model templates."""
    print("=" * 70)
    print("Example 7: Using Model Templates")
    print("=" * 70)

    # Create template with default settings
    print("\n" + "-" * 70)
    print("Step 1: Create Template with Defaults")
    print("-" * 70)

    template = SimpleOpenEconomy()
    print(f"\nCreated template: {template}")
    print(f"Description: {template.description}")

    # Show template info
    info = template.get_info()
    print(f"\nTemplate Configuration:")
    print(f"  Sectors: {info['num_sectors']} - {info['sector_names']}")
    print(f"  Factors: {info['num_factors']} - {info['factor_names']}")
    print(f"  Elasticities:")
    print(f"    CES (VA): {info['sigma_va']}")
    print(f"    Armington: {info['sigma_m']}")
    print(f"    CET: {info['sigma_e']}")
    print(f"  Blocks: {', '.join(info['blocks'])}")

    # Create model from template
    print("\n" + "-" * 70)
    print("Step 2: Create Model from Template")
    print("-" * 70)

    model = template.create_model()
    print(f"\nCreated model: {model.name}")

    stats = model.statistics
    print(f"\nModel Statistics:")
    print(f"  Variables: {stats.variables}")
    print(f"  Equations: {stats.equations}")
    print(f"  Degrees of freedom: {stats.degrees_of_freedom}")
    print(f"  Blocks: {stats.blocks}")

    print(f"\nSets in model:")
    for set_name in model.set_manager.list_sets():
        s = model.set_manager.get(set_name)
        print(f"  {set_name}: {len(s)} elements")

    # Create customized template
    print("\n" + "-" * 70)
    print("Step 3: Customize Template")
    print("-" * 70)

    custom_template = SimpleOpenEconomy(
        num_sectors=5,
        num_factors=3,
        sigma_va=1.0,
        sigma_m=2.0,
        sigma_e=3.0,
    )

    print(f"\nCreated custom template: {custom_template}")

    custom_info = custom_template.get_info()
    print(f"\nCustom Configuration:")
    print(f"  Sectors: {custom_info['num_sectors']} - {custom_info['sector_names']}")
    print(f"  Factors: {custom_info['num_factors']} - {custom_info['factor_names']}")
    print(f"  Elasticities:")
    print(f"    CES (VA): {custom_info['sigma_va']}")
    print(f"    Armington: {custom_info['sigma_m']}")
    print(f"    CET: {custom_info['sigma_e']}")

    # Create model from custom template
    custom_model = custom_template.create_model()
    print(f"\nCreated custom model: {custom_model.name}")

    custom_stats = custom_model.statistics
    print(f"\nCustom Model Statistics:")
    print(f"  Variables: {custom_stats.variables}")
    print(f"  Blocks: {custom_stats.blocks}")

    # Compare models
    print("\n" + "-" * 70)
    print("Step 4: Compare Models")
    print("-" * 70)

    print(f"\nComparison:")
    print(f"  Default model: {stats.variables} variables")
    print(f"  Custom model:  {custom_stats.variables} variables")
    print(f"  Difference:    {custom_stats.variables - stats.variables} more variables")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)
    print("\nTemplates provide:")
    print("  ✓ Pre-configured models")
    print("  ✓ Sensible defaults")
    print("  ✓ Easy customization")
    print("  ✓ Consistent structure")
    print("=" * 70)


if __name__ == "__main__":
    main()
