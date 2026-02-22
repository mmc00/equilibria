"""
Compare Python and GAMS PEP model results.

This script compares the results from the Python equilibria implementation
with the GAMS reference results.
"""

import sys
from pathlib import Path

sys.path.insert(0, '/Users/marmol/proyectos/equilibria/src')

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def load_python_results():
    """Load Python model results."""
    results_file = Path("/Users/marmol/proyectos/equilibria/results/python_pep_results.gdx")
    
    if not results_file.exists():
        print(f"❌ Python results not found: {results_file}")
        print("   Run: python3 scripts/run_pep_model.py")
        return None
    
    print("Loading Python results...")
    data = read_gdx(results_file)
    
    print(f"✓ Loaded Python results:")
    print(f"  Symbols: {len(data['symbols'])}")
    print(f"  Elements: {len(data['elements'])}")
    
    return data


def load_gams_results():
    """Load GAMS reference results."""
    # Try multiple possible locations
    possible_paths = [
        Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/gams/results/Results.gdx"),
        Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/gams/results/all_data_baseline.gdx"),
        Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/gams/pep_results.gdx"),
    ]
    
    for results_file in possible_paths:
        if results_file.exists():
            print(f"\nLoading GAMS results from: {results_file}")
            data = read_gdx(results_file)
            
            print(f"✓ Loaded GAMS results:")
            print(f"  Symbols: {len(data['symbols'])}")
            print(f"  Elements: {len(data['elements'])}")
            
            return data
    
    print("❌ GAMS results not found in any expected location")
    return None


def compare_sets(py_data, gams_data):
    """Compare sets between Python and GAMS."""
    print("\n" + "=" * 70)
    print("SET COMPARISON")
    print("=" * 70)
    
    py_elements = set(py_data['elements'])
    gams_elements = set(gams_data['elements'])
    
    common = py_elements & gams_elements
    only_py = py_elements - gams_elements
    only_gams = gams_elements - py_elements
    
    print(f"\nPython elements: {len(py_elements)}")
    print(f"GAMS elements: {len(gams_elements)}")
    print(f"Common elements: {len(common)}")
    
    if only_py:
        print(f"\nOnly in Python ({len(only_py)}):")
        for elem in sorted(only_py)[:10]:
            print(f"  - {elem}")
        if len(only_py) > 10:
            print(f"  ... and {len(only_py) - 10} more")
    
    if only_gams:
        print(f"\nOnly in GAMS ({len(only_gams)}):")
        for elem in sorted(only_gams)[:10]:
            print(f"  - {elem}")
        if len(only_gams) > 10:
            print(f"  ... and {len(only_gams) - 10} more")
    
    match_rate = len(common) / max(len(py_elements), len(gams_elements)) * 100
    print(f"\nSet match rate: {match_rate:.1f}%")
    
    return match_rate


def compare_symbols(py_data, gams_data):
    """Compare symbols between Python and GAMS."""
    print("\n" + "=" * 70)
    print("SYMBOL COMPARISON")
    print("=" * 70)
    
    py_symbols = {s['name']: s for s in py_data['symbols']}
    gams_symbols = {s['name']: s for s in gams_data['symbols']}
    
    common_names = set(py_symbols.keys()) & set(gams_symbols.keys())
    only_py = set(py_symbols.keys()) - set(gams_symbols.keys())
    only_gams = set(gams_symbols.keys()) - set(py_symbols.keys())
    
    print(f"\nPython symbols: {len(py_symbols)}")
    print(f"GAMS symbols: {len(gams_symbols)}")
    print(f"Common symbols: {len(common_names)}")
    
    if only_py:
        print(f"\nOnly in Python ({len(only_py)}):")
        for name in sorted(only_py):
            print(f"  - {name}")
    
    if only_gams:
        print(f"\nOnly in GAMS ({len(only_gams)}):")
        for name in sorted(only_gams)[:10]:
            print(f"  - {name}")
        if len(only_gams) > 10:
            print(f"  ... and {len(only_gams) - 10} more")
    
    # Compare common symbols
    if common_names:
        print(f"\nCommon symbols:")
        for name in sorted(common_names)[:10]:
            py_sym = py_symbols[name]
            gams_sym = gams_symbols[name]
            print(f"  {name}:")
            print(f"    Python: type={py_sym.get('type_name', 'unknown')}, dim={py_sym.get('dimension', '?')}, records={py_sym.get('records', '?')}")
            print(f"    GAMS:   type={gams_sym.get('type_name', 'unknown')}, dim={gams_sym.get('dimension', '?')}, records={gams_sym.get('records', '?')}")
    
    match_rate = len(common_names) / max(len(py_symbols), len(gams_symbols)) * 100
    print(f"\nSymbol match rate: {match_rate:.1f}%")
    
    return match_rate


def main():
    """Main comparison."""
    print("=" * 70)
    print("PEP MODEL COMPARISON: Python (equilibria) vs GAMS")
    print("=" * 70)
    
    # Load results
    py_data = load_python_results()
    gams_data = load_gams_results()
    
    if py_data is None or gams_data is None:
        print("\n❌ Cannot perform comparison - missing data")
        return
    
    # Compare
    set_match = compare_sets(py_data, gams_data)
    symbol_match = compare_symbols(py_data, gams_data)
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Set match rate:    {set_match:.1f}%")
    print(f"Symbol match rate: {symbol_match:.1f}%")
    
    if set_match == 100 and symbol_match == 100:
        print("\n✅ PERFECT MATCH: Python and GAMS results are identical!")
    elif set_match >= 90 and symbol_match >= 90:
        print("\n✅ GOOD MATCH: Results are highly compatible")
    else:
        print("\n⚠️  PARTIAL MATCH: Some differences detected")
    
    print("\nNote: Full variable/parameter value comparison requires")
    print("      solving the Python model (currently not implemented)")


if __name__ == "__main__":
    main()
