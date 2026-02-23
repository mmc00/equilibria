"""Demonstration of GDX writer functionality.

This example shows how to write GDX files from Python without GAMS.

Usage:
    python examples/gdx_writer_demo.py
"""

from pathlib import Path

from equilibria.babel.gdx.reader import read_gdx
from equilibria.babel.gdx.symbols import Parameter, Set, Variable
from equilibria.babel.gdx.writer import write_gdx


def demo_write_simple_parameter():
    """Write a simple 1D parameter to GDX."""
    print("=" * 70)
    print("DEMO 1: Writing Simple Parameter")
    print("=" * 70)

    # Create a parameter
    price = Parameter(
        name="price",
        sym_type="parameter",
        dimensions=1,
        description="Commodity prices",
        domain=["i"],
        records=[
            (["agr"], 1.5),
            (["mfg"], 2.0),
            (["srv"], 2.5),
        ]
    )

    # Write to GDX
    output_file = Path("test_output.gdx")
    write_gdx(output_file, [price])

    print(f"\n✅ Wrote parameter to: {output_file}")
    print(f"   File size: {output_file.stat().st_size} bytes")

    # Read back and verify
    gdx_data = read_gdx(output_file)
    print("\n📖 Read back from file:")
    print(f"   Symbols: {len(gdx_data['symbols'])}")
    print(f"   Elements: {gdx_data['elements']}")

    # Clean up
    output_file.unlink()
    print("\n🗑️  Cleaned up test file")


def demo_write_set_and_parameter():
    """Write a set and parameter to GDX."""
    print("\n" + "=" * 70)
    print("DEMO 2: Writing Set and Parameter")
    print("=" * 70)

    # Create a set
    industries = Set(
        name="i",
        sym_type="set",
        dimensions=1,
        description="Industries",
        domain=["*"],
        elements=[["agr"], ["mfg"], ["srv"]]
    )

    # Create a parameter using the set
    output = Parameter(
        name="Q",
        sym_type="parameter",
        dimensions=1,
        description="Output by industry",
        domain=["i"],
        records=[
            (["agr"], 100.0),
            (["mfg"], 200.0),
            (["srv"], 300.0),
        ]
    )

    # Write both to GDX
    output_file = Path("test_set_param.gdx")
    write_gdx(output_file, [industries, output])

    print(f"\n✅ Wrote set and parameter to: {output_file}")
    print(f"   File size: {output_file.stat().st_size} bytes")

    # Read back
    gdx_data = read_gdx(output_file)
    print("\n📖 Read back from file:")
    print(f"   Symbols: {[s['name'] for s in gdx_data['symbols']]}")
    print(f"   UEL: {gdx_data['elements']}")

    # Clean up
    output_file.unlink()
    print("\n🗑️  Cleaned up test file")


def demo_write_2d_parameter():
    """Write a 2-dimensional parameter (matrix) to GDX."""
    print("\n" + "=" * 70)
    print("DEMO 3: Writing 2D Parameter (Matrix)")
    print("=" * 70)

    # Create sets
    industries = Set(
        name="i",
        sym_type="set",
        dimensions=1,
        description="Industries",
        elements=[["agr"], ["mfg"], ["srv"]]
    )

    commodities = Set(
        name="j",
        sym_type="set",
        dimensions=1,
        description="Commodities",
        elements=[["food"], ["goods"], ["services"]]
    )

    # Create 2D parameter (SAM)
    sam = Parameter(
        name="SAM",
        sym_type="parameter",
        dimensions=2,
        description="Social Accounting Matrix",
        domain=["i", "j"],
        records=[
            (["agr", "food"], 100.0),
            (["agr", "goods"], 50.0),
            (["mfg", "food"], 30.0),
            (["mfg", "goods"], 200.0),
            (["srv", "services"], 150.0),
        ]
    )

    # Write to GDX
    output_file = Path("test_2d_param.gdx")
    write_gdx(output_file, [industries, commodities, sam])

    print(f"\n✅ Wrote 2D parameter to: {output_file}")
    print(f"   File size: {output_file.stat().st_size} bytes")

    # Read back
    gdx_data = read_gdx(output_file)
    print("\n📖 Read back from file:")
    print(f"   Symbols: {[s['name'] for s in gdx_data['symbols']]}")
    print(f"   UEL elements: {len(gdx_data['elements'])}")

    # Clean up
    output_file.unlink()
    print("\n🗑️  Cleaned up test file")


def demo_write_variable():
    """Write a variable with all attributes to GDX."""
    print("\n" + "=" * 70)
    print("DEMO 4: Writing Variable")
    print("=" * 70)

    # Create a variable
    X = Variable(
        name="X",
        sym_type="variable",
        dimensions=1,
        description="Output levels",
        domain=["i"],
        records=[
            (["agr"], (100.0, 0.0, 0.0, float("inf"), 1.0)),
            (["mfg"], (200.0, 0.5, 0.0, float("inf"), 1.0)),
            (["srv"], (300.0, 1.2, 50.0, 500.0, 1.0)),
        ]
    )

    # Write to GDX
    output_file = Path("test_variable.gdx")
    write_gdx(output_file, [X])

    print(f"\n✅ Wrote variable to: {output_file}")
    print(f"   File size: {output_file.stat().st_size} bytes")

    # Read back
    gdx_data = read_gdx(output_file)
    print("\n📖 Read back from file:")
    print(f"   Variables: {[s['name'] for s in gdx_data['symbols'] if s['type'] == 2]}")

    # Clean up
    output_file.unlink()
    print("\n🗑️  Cleaned up test file")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("GDX WRITER - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\nDemonstrating GDX file writing capabilities.")

    try:
        demo_write_simple_parameter()
        demo_write_set_and_parameter()
        demo_write_2d_parameter()
        demo_write_variable()

        print("\n" + "=" * 70)
        print("✅ All demos completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
