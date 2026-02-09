"""
Execute PEP model with calibration from GAMS baseline.

This script loads the calibrated baseline from GAMS results and uses it
as starting point for the Python model.
"""

import sys
from pathlib import Path

sys.path.insert(0, '/Users/marmol/proyectos/equilibria/src')

import pandas as pd
from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def load_gams_baseline():
    """Load GAMS baseline results."""
    gams_results = Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/gams/results/all_data_baseline.gdx")
    
    if not gams_results.exists():
        print(f"❌ GAMS baseline not found: {gams_results}")
        return None
    
    print(f"Loading GAMS baseline: {gams_results}")
    data = read_gdx(gams_results)
    
    print(f"✓ Loaded GAMS baseline:")
    print(f"  Symbols: {len(data['symbols'])}")
    print(f"  Elements: {len(data['elements'])}")
    
    # List all symbols
    print(f"\nAvailable symbols:")
    for sym in data['symbols'][:20]:
        print(f"  - {sym['name']} ({sym.get('type_name', 'unknown')})")
    if len(data['symbols']) > 20:
        print(f"  ... and {len(data['symbols']) - 20} more")
    
    return data


def extract_key_variables(gams_data):
    """Extract key variables from GAMS results."""
    print("\n" + "=" * 70)
    print("Extracting Key Variables")
    print("=" * 70)
    
    # Key variables to extract
    key_vars = [
        'Y',      # Production
        'C',      # Consumption
        'I',      # Investment
        'G',      # Government spending
        'X',      # Exports
        'M',      # Imports
        'P',      # Prices
        'W',      # Wages
        'R',      # Rental rate
        'Q',      # Armington composite
        'D',      # Domestic demand
        'E',      # Energy/Exports
        'DS',     # Domestic supply
    ]
    
    results = {}
    
    for var_name in key_vars:
        try:
            values = read_parameter_values(gams_data, var_name)
            if values:
                results[var_name] = values
                print(f"✓ {var_name}: {len(values)} values")
        except Exception as e:
            print(f"⚠ {var_name}: not found ({e})")
    
    return results


def compare_with_sam(gams_data, sam_data):
    """Compare GAMS results with original SAM."""
    print("\n" + "=" * 70)
    print("Comparing GAMS Results with SAM")
    print("=" * 70)
    
    # Load SAM values
    sam_values = read_parameter_values(sam_data, 'SAM')
    
    if not sam_values:
        print("⚠ Could not read SAM values")
        return
    
    print(f"SAM entries: {len(sam_values)}")
    
    # Sample comparison
    print("\nSample SAM entries:")
    for i, (key, value) in enumerate(list(sam_values.items())[:5]):
        print(f"  {key}: {value}")


def main():
    """Main execution."""
    print("=" * 70)
    print("PEP Model - GAMS Baseline Analysis")
    print("=" * 70)
    
    # Load GAMS baseline
    gams_data = load_gams_baseline()
    if not gams_data:
        return
    
    # Extract key variables
    key_vars = extract_key_variables(gams_data)
    
    # Load SAM for comparison
    sam_file = Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.gdx")
    sam_data = read_gdx(sam_file)
    
    compare_with_sam(gams_data, sam_data)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"GAMS baseline variables extracted: {len(key_vars)}")
    print("\nTo implement full comparison:")
    print("1. Fix calibration in Python model")
    print("2. Use GAMS values as initial guess")
    print("3. Solve and compare variable by variable")


if __name__ == "__main__":
    main()
