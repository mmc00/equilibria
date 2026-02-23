"""Analyze domain information for sets."""

from pathlib import Path

from equilibria.babel.gdx.reader import read_gdx

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

gdx_path = FIXTURES_DIR / "variables_equations_test.gdx"
gdx_data = read_gdx(gdx_path)

print("=" * 70)
print("COMPLETE GDX DATA ANALYSIS")
print("=" * 70)

# Print all symbols with their details
print("\nALL SYMBOLS:")
for idx, sym in enumerate(gdx_data["symbols"]):
    print(f"\n[{idx}] {sym['name']}")
    print(f"    Type: {sym['type']} ({sym['type_name']})")
    print(f"    Dimension: {sym['dimension']}")
    print(f"    Records: {sym['records']}")
    print(f"    Description: {sym.get('description', 'N/A')}")
    if 'domains' in sym:
        print(f"    Domains: {sym['domains']}")

# Print UEL
print("\n\nUEL ELEMENTS:")
for idx, elem in enumerate(gdx_data['elements']):
    print(f"  [{idx}] (1-based:{idx+1}) {elem}")

# Print domains if available
if 'domains' in gdx_data:
    print("\n\nDOMAINS:")
    print(gdx_data['domains'])
