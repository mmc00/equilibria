"""
PEP Static Clean Model Example

This example demonstrates how to load data from pep_static_clean
and create a complete PEP CGE model using equilibria.

The pep_static_clean dataset contains:
- SAM-V2_0.gdx: Social Accounting Matrix
- VAL_PAR.gdx: Model parameters
"""

from pathlib import Path

from equilibria.templates import PEP1R
from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def load_pep_static_clean_data(data_path: Path | None = None):
    """Load data from pep_static_clean dataset.
    
    Args:
        data_path: Path to pep_static_clean/data/original directory
                  (uses default location if not provided)
    
    Returns:
        Tuple of (sam_data, param_data)
    """
    if data_path is None:
        # Try to find pep_static_clean in common locations
        possible_paths = [
            Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original"),
            Path.home() / "proyectos" / "cge_babel" / "pep_static_clean" / "data" / "original",
            Path("../cge_babel/pep_static_clean/data/original"),
            Path("../../cge_babel/pep_static_clean/data/original"),
        ]
        
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                "Could not find pep_static_clean data. Please provide data_path."
            )
    
    print(f"Loading data from: {data_path}")
    
    # Load SAM
    sam_file = data_path / "SAM-V2_0.gdx"
    if not sam_file.exists():
        raise FileNotFoundError(f"SAM file not found: {sam_file}")
    
    sam_data = read_gdx(sam_file)
    print(f"✓ Loaded SAM: {len(sam_data['symbols'])} symbols")
    
    # Load parameters
    param_file = data_path / "VAL_PAR.gdx"
    if param_file.exists():
        param_data = read_gdx(param_file)
        print(f"✓ Loaded parameters: {len(param_data['symbols'])} symbols")
    else:
        param_data = None
        print("⚠ Parameter file not found, using defaults")
    
    return sam_data, param_data


def create_pep_model_from_gdx(sam_data, param_data=None, calibrate: bool = True):
    """Create PEP model from GDX data.
    
    Args:
        sam_data: SAM data from read_gdx()
        param_data: Parameter data from read_gdx() (optional)
        calibrate: Whether to run calibration
    
    Returns:
        Configured Model instance
    """
    # Create template
    template = PEP1R()
    
    # Extract sets from GDX data
    template.extract_sets_from_gdx_data(sam_data)
    
    print(f"\nExtracted sets:")
    print(f"  Sectors: {template.sectors}")
    print(f"  Commodities: {template.commodities}")
    print(f"  Labor types: {template.labor_types}")
    print(f"  Capital types: {template.capital_types}")
    print(f"  Households: {template.households}")
    
    # Create model (skip calibration for now due to SAM4D compatibility)
    # TODO: Fix calibration to work with SAM4D
    model = template.create_model(calibrate=False)
    
    return model


def main():
    """Main execution."""
    print("=" * 60)
    print("PEP Static Clean Model - equilibria Implementation")
    print("=" * 60)
    
    # Load data
    try:
        sam_data, param_data = load_pep_static_clean_data()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure pep_static_clean is available at:")
        print("  /Users/marmol/proyectos/cge_babel/pep_static_clean/")
        return
    
    # Create model
    print("\n" + "-" * 60)
    print("Creating PEP model...")
    print("-" * 60)
    
    model = create_pep_model_from_gdx(sam_data, param_data, calibrate=True)
    
    # Print model statistics
    print("\n" + "=" * 60)
    print("Model Statistics")
    print("=" * 60)
    print(f"Name: {model.name}")
    print(f"Description: {model.description}")
    
    stats = model.statistics
    print(f"Variables: {stats.variables}")
    print(f"Equations: {stats.equations}")
    print(f"Degrees of Freedom: {stats.degrees_of_freedom}")
    print(f"Blocks: {stats.blocks}")
    print(f"Sparsity: {stats.sparsity:.2%}")
    
    # Validate against GAMS if available
    print("\n" + "-" * 60)
    print("Validation")
    print("-" * 60)
    
    gams_results = Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/gams/pep_results.gdx")
    if gams_results.exists():
        print(f"✓ GAMS results found: {gams_results}")
        print("  Run model.validate(gams_results) to compare")
    else:
        print("⚠ GAMS results not found, skipping validation")
    
    print("\n" + "=" * 60)
    print("Model ready for simulation!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    model = main()
