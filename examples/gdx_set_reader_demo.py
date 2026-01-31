"""Demonstration of full set reading functionality in GDX files.

This example shows how to read different types of sets:
- 1D sets (simple lists)
- 2D sets (pairs)
- Multi-dimensional sets
- Sets with explanatory text

Usage:
    python examples/gdx_set_reader_demo.py
"""

from pathlib import Path

from equilibria.babel.gdx.reader import read_gdx, read_set_elements, get_sets

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def demo_1d_sets():
    """Demonstrate reading 1-dimensional sets."""
    print("=" * 70)
    print("1D SETS DEMO")
    print("=" * 70)
    
    # Read GDX file with 1D sets
    gdx_path = FIXTURES_DIR / "variables_equations_test.gdx"
    if not gdx_path.exists():
        print(f"⚠️  File not found: {gdx_path}")
        return
    
    gdx_data = read_gdx(gdx_path)
    
    # Get all sets in the file
    sets = get_sets(gdx_data)
    print(f"\n📋 Found {len(sets)} sets in the file:")
    for s in sets:
        print(f"  - {s['name']}: {s['description']}")
    
    # Read elements of each set
    print("\n📊 Set Elements:")
    for s in sets:
        if s['dimension'] == 1:
            elements = read_set_elements(gdx_data, s['name'])
            print(f"\n  Set '{s['name']}' ({s['description']}):")
            print(f"  Records: {len(elements)}")
            print(f"  Elements: {[e[0] for e in elements]}")


def demo_multidim_sets():
    """Demonstrate reading multi-dimensional sets."""
    print("\n" + "=" * 70)
    print("MULTI-DIMENSIONAL SETS DEMO")
    print("=" * 70)
    
    gdx_path = FIXTURES_DIR / "multidim_test.gdx"
    if not gdx_path.exists():
        print(f"⚠️  File not found: {gdx_path}")
        return
    
    gdx_data = read_gdx(gdx_path)
    
    # Get all sets
    sets = get_sets(gdx_data)
    print(f"\n📋 Found {len(sets)} sets:")
    
    for s in sets:
        elements = read_set_elements(gdx_data, s['name'])
        print(f"\n  Set '{s['name']}' (dimension: {s['dimension']}):")
        print(f"  Records: {len(elements)}")
        print(f"  First 5 elements: {elements[:5]}")


def demo_sparse_sets():
    """Demonstrate reading sparse sets."""
    print("\n" + "=" * 70)
    print("SPARSE SETS DEMO")
    print("=" * 70)
    
    gdx_path = FIXTURES_DIR / "sparse_test.gdx"
    if not gdx_path.exists():
        print(f"⚠️  File not found: {gdx_path}")
        return
    
    gdx_data = read_gdx(gdx_path)
    
    # Get sets
    sets = get_sets(gdx_data)
    print(f"\n📋 Found {len(sets)} set(s)")
    
    for s in sets:
        elements = read_set_elements(gdx_data, s['name'])
        print(f"\n  Set '{s['name']}':")
        print(f"  Description: {s.get('description', 'N/A')}")
        print(f"  Records: {len(elements)}")
        print(f"  Elements: {[e[0] for e in elements if len(e) > 0]}")


def demo_set_statistics():
    """Show statistics about sets in GDX files."""
    print("\n" + "=" * 70)
    print("SET STATISTICS")
    print("=" * 70)
    
    test_files = [
        "variables_equations_test.gdx",
        "multidim_test.gdx",
        "sparse_test.gdx",
    ]
    
    for filename in test_files:
        gdx_path = FIXTURES_DIR / filename
        if not gdx_path.exists():
            continue
        
        print(f"\n📁 {filename}:")
        gdx_data = read_gdx(gdx_path)
        sets = get_sets(gdx_data)
        
        if not sets:
            print("  No sets found")
            continue
        
        for s in sets:
            elements = read_set_elements(gdx_data, s['name'])
            print(f"  - {s['name']:12s} dim={s['dimension']} records={len(elements):3d}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("GDX SET READER - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\nThis demo shows the full functionality of reading sets from GDX files.")
    print("Sets can be 1D, 2D, or multi-dimensional.")
    
    demo_1d_sets()
    demo_multidim_sets()
    demo_sparse_sets()
    demo_set_statistics()
    
    print("\n" + "=" * 70)
    print("✅ Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
