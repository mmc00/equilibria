"""
Test script for reading multi-dimensional parameters from GDX files.

This script demonstrates reading parameters with 3 or more dimensions.
"""

from pathlib import Path

from equilibria.babel.gdx.reader import get_symbol, read_gdx, read_parameter_values


def test_multidim_parameter_reading():
    """Test reading multi-dimensional parameters."""

    # Look for test GDX files in fixtures
    fixtures_dir = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

    # Try to find a suitable test file
    gdx_files = list(fixtures_dir.glob("*.gdx"))

    if not gdx_files:
        print("No GDX files found in fixtures directory")
        return

    print(f"Found {len(gdx_files)} GDX files to test\n")

    for gdx_file in gdx_files[:5]:  # Test first 5 files
        print(f"\n{'='*60}")
        print(f"Testing: {gdx_file.name}")
        print(f"{'='*60}")

        try:
            # Read GDX file
            data = read_gdx(gdx_file)

            # Find parameters with dimension >= 3
            multidim_params = [
                sym for sym in data["symbols"]
                if sym["type"] == 1 and sym["dimension"] >= 3
            ]

            if not multidim_params:
                print("  No 3+ dimensional parameters found")
                continue

            print(f"\nFound {len(multidim_params)} multi-dimensional parameters:")

            for param in multidim_params:
                print(f"\n  Parameter: {param['name']}")
                print(f"    Dimension: {param['dimension']}")
                print(f"    Records: {param['records']}")
                print(f"    Description: {param['description']}")

                # Try to read the values
                try:
                    values = read_parameter_values(data, param['name'])

                    print(f"    Successfully read {len(values)} values")

                    # Show first few values
                    if values:
                        print("    Sample values:")
                        for i, (key, val) in enumerate(list(values.items())[:5]):
                            print(f"      {key} = {val}")

                        if len(values) > 5:
                            print(f"      ... ({len(values) - 5} more values)")

                except Exception as e:
                    print(f"    Error reading values: {e}")

        except Exception as e:
            print(f"  Error: {e}")


def analyze_parameter_structure(gdx_file: Path, param_name: str):
    """Detailed analysis of a specific parameter's binary structure."""
    print(f"\n{'='*60}")
    print(f"Detailed analysis: {param_name} in {gdx_file.name}")
    print(f"{'='*60}")

    data = read_gdx(gdx_file)
    symbol = get_symbol(data, param_name)

    if not symbol:
        print(f"Parameter '{param_name}' not found")
        return

    print("\nSymbol info:")
    print(f"  Type: {symbol['type_name']}")
    print(f"  Dimension: {symbol['dimension']}")
    print(f"  Records: {symbol['records']}")
    print(f"  Type flag: {symbol['type_flag']:#x}")

    # Read values
    try:
        values = read_parameter_values(data, param_name)
        print(f"\nSuccessfully read {len(values)} values")

        if values:
            # Analyze index structure
            first_key = next(iter(values.keys()))
            print("\nIndex structure:")
            print(f"  Tuple length: {len(first_key)}")
            print(f"  First index: {first_key}")

            # Show all unique elements per dimension
            for dim in range(len(first_key)):
                unique_elems = sorted(set(key[dim] for key in values.keys()))
                print(f"  Dimension {dim+1}: {len(unique_elems)} unique elements")
                if len(unique_elems) <= 10:
                    print(f"    Elements: {unique_elems}")
                else:
                    print(f"    Sample: {unique_elems[:5]} ... {unique_elems[-2:]}")

            # Show value distribution
            all_vals = list(values.values())
            print("\nValue statistics:")
            print(f"  Min: {min(all_vals)}")
            print(f"  Max: {max(all_vals)}")
            print(f"  Mean: {sum(all_vals) / len(all_vals)}")

    except Exception as e:
        print(f"\nError reading values: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run general test
    test_multidim_parameter_reading()

    # If you have a specific file and parameter to analyze, uncomment:
    # analyze_parameter_structure(
    #     Path("tests/fixtures/your_file.gdx"),
    #     "your_parameter_name"
    # )
