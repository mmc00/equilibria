"""
Execute PEP model and save results for comparison with GAMS.

This script runs the PEP model in Python and exports results to GDX format
for comparison with GAMS results.
"""

import sys
from pathlib import Path

# Add equilibria to path
sys.path.insert(0, '/Users/marmol/proyectos/equilibria/src')

from equilibria.templates import PEP1R
from equilibria.babel.gdx.reader import read_gdx
from equilibria.babel.gdx.writer import write_gdx


def run_pep_model():
    """Run PEP model and return results."""
    print("=" * 70)
    print("Running PEP Model in Python (equilibria)")
    print("=" * 70)
    
    # Load data from pep_static_clean
    data_path = Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original")
    
    # Read SAM data
    sam_file = data_path / "SAM-V2_0.gdx"
    sam_data = read_gdx(sam_file)
    print(f"✓ Loaded SAM: {len(sam_data['symbols'])} symbols")
    
    # Create template and model
    template = PEP1R()
    template.extract_sets_from_gdx_data(sam_data)
    
    print(f"\nExtracted sets:")
    print(f"  Sectors: {template.sectors}")
    print(f"  Commodities: {template.commodities}")
    print(f"  Labor types: {template.labor_types}")
    print(f"  Capital types: {template.capital_types}")
    print(f"  Households: {template.households}")
    
    # Create model (without calibration for now)
    print("\nCreating model...")
    model = template.create_model(calibrate=False)
    
    # Print statistics
    stats = model.statistics
    print(f"\nModel Statistics:")
    print(f"  Variables: {stats.variables}")
    print(f"  Equations: {stats.equations}")
    print(f"  Degrees of Freedom: {stats.degrees_of_freedom}")
    print(f"  Blocks: {stats.blocks}")
    
    # TODO: Solve the model
    # For now, we'll export the initialized model state
    print("\n⚠️  Model solving not yet implemented")
    print("   Exporting initialized model state for comparison")
    
    return model, template


def export_results_to_gdx(model, template, output_path):
    """Export model results to GDX file.
    
    Args:
        model: Model instance
        template: PEP1R template
        output_path: Path to save GDX file
    """
    print(f"\nExporting results to: {output_path}")
    
    # Prepare data for export
    symbols = []
    
    # Add sets using proper Symbol objects
    from equilibria.babel.gdx.symbols import Set as GDXSet
    
    sets_data = {
        'J': template.sectors,
        'I': template.commodities,
        'L': template.labor_types,
        'K': template.capital_types,
        'H': template.households,
    }
    
    for set_name, elements in sets_data.items():
        gdx_set = GDXSet(
            name=set_name,
            sym_type="set",
            dimensions=1,
            description=f"Set {set_name}",
            records=[([elem], 1.0) for elem in elements]
        )
        symbols.append(gdx_set)
    
    # TODO: Add parameters and variables from solved model
    # For now, just export sets
    
    # Write GDX
    write_gdx(output_path, symbols)
    print(f"✓ Results exported to: {output_path}")


def main():
    """Main execution."""
    # Run model
    model, template = run_pep_model()
    
    # Export results
    output_path = Path("/Users/marmol/proyectos/equilibria/results/python_pep_results.gdx")
    output_path.parent.mkdir(exist_ok=True)
    export_results_to_gdx(model, template, output_path)
    
    print("\n" + "=" * 70)
    print("Python Model Execution Complete")
    print("=" * 70)
    print(f"\nResults saved to: {output_path}")
    print("\nNext steps:")
    print("1. Run GAMS model: cd /Users/marmol/proyectos/cge_babel/pep_static_clean/gams && gams PEP-1-1_v2_1_modular.gms")
    print("2. Compare results with: python compare_results.py")


if __name__ == "__main__":
    main()
