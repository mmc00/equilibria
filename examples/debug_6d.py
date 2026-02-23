"""Simple debugging script for 6D parameter reading."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from equilibria.babel.gdx.reader import read_gdx


def debug_6d():
    """Debug 6D parameter detection."""
    gdx_path = str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "test_6d.gdx")

    print("Reading GDX file...")
    data = read_gdx(gdx_path)

    print("\nSymbols found:")
    for sym in data['symbols']:
        print(f"  Name: {sym['name']}")
        print(f"    Type: {sym['type']} ({sym.get('type_name', 'unknown')})")
        print(f"    Type flag: 0x{sym.get('type_flag', 0):02X}")
        print(f"    Dimension: {sym['dimension']}")
        print(f"    Records: {sym['records']}")
        print()

if __name__ == "__main__":
    debug_6d()
