#!/usr/bin/env python3
"""Analyze SAM GDX storage differences between original and equilibria formats.

This script analyzes the structural differences in how SAM data is stored
to understand sparse vs dense storage strategies.
"""

from pathlib import Path
from equilibria.babel.gdx import read_gdx
from equilibria.babel.gdx.reader import get_symbol
import pandas as pd


def analyze_gdx_structure(filepath: Path, label: str) -> dict:
    """Analyze the structure of a SAM GDX file."""
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")

    data = read_gdx(filepath)

    result = {"filepath": str(filepath), "symbols": [], "total_records": 0}

    print(f"\nFile: {filepath}")
    print(f"Symbols count: {len(data['symbols'])}")

    for sym in data["symbols"]:
        sym_info = {
            "name": sym["name"],
            "type": sym["type_name"],
            "dimension": sym.get("dimension", 0),
            "records": sym.get("records", 0),
        }
        result["symbols"].append(sym_info)
        result["total_records"] += sym_info["records"]

        print(f"\n  Symbol: {sym['name']}")
        print(f"    Type: {sym['type_name']}")
        print(f"    Dimension: {sym.get('dimension', 'N/A')}")
        print(f"    Records: {sym.get('records', 'N/A')}")

        # For sets, show elements
        if sym["type"] == 0:  # Set
            from equilibria.babel.gdx.reader import read_set_elements

            try:
                elements = read_set_elements(data, sym["name"])
                print(f"    Elements: {len(elements)}")
                print(f"    Sample: {elements[:5]}")
            except Exception as e:
                print(f"    Error reading elements: {e}")

    return result


def analyze_excel_sam(filepath: Path) -> dict:
    """Load and analyze the source Excel SAM."""
    print(f"\n{'=' * 70}")
    print("SOURCE EXCEL SAM")
    print(f"{'=' * 70}")

    df = pd.read_excel(filepath, sheet_name="SAM", index_col=0, header=0)

    print(f"\nFile: {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Total accounts: {len(df)}")
    print(f"Total cells: {df.shape[0] * df.shape[1]}")

    # Count non-zero values
    non_zero = (df != 0).sum().sum()
    print(f"Non-zero values: {non_zero}")

    # Count zero values
    zero_values = (df == 0).sum().sum()
    print(f"Zero values: {zero_values}")

    # Show account names
    accounts = df.index.tolist()
    print(f"\nAccount names ({len(accounts)}):")
    for i, acc in enumerate(accounts):
        print(f"  {i + 1:2d}. {acc}")

    return {
        "shape": df.shape,
        "accounts": accounts,
        "non_zero": non_zero,
        "zero_values": zero_values,
        "total_cells": df.shape[0] * df.shape[1],
    }


def compare_structures(original: dict, equilibria: dict, excel_info: dict):
    """Compare the structures and provide analysis."""
    print(f"\n{'=' * 70}")
    print("COMPARISON ANALYSIS")
    print(f"{'=' * 70}")

    print("\n## 1. Symbol Count")
    print(f"  Original: {len(original['symbols'])} symbol(s)")
    print(f"  Equilibria: {len(equilibria['symbols'])} symbol(s)")

    # Get SAM parameter info from both
    orig_sam = next((s for s in original["symbols"] if s["name"] == "SAM"), None)
    eq_sam = next((s for s in equilibria["symbols"] if s["name"] == "SAM"), None)
    eq_accounts = next(
        (s for s in equilibria["symbols"] if s["name"] == "ACCOUNTS"), None
    )

    print("\n## 2. SAM Parameter Structure")
    if orig_sam:
        print(f"  Original SAM:")
        print(f"    - Dimension: {orig_sam['dimension']}D")
        print(f"    - Records: {orig_sam['records']}")

    if eq_sam:
        print(f"  Equilibria SAM:")
        print(f"    - Dimension: {eq_sam['dimension']}D")
        print(f"    - Records: {eq_sam['records']}")

    if eq_accounts:
        print(f"  Equilibria ACCOUNTS Set:")
        print(f"    - Elements: {eq_accounts['records']}")

    print("\n## 3. Storage Strategy Analysis")

    # Calculate expected values
    n_accounts = excel_info["accounts"]
    total_cells = excel_info["total_cells"]
    non_zero = excel_info["non_zero"]

    print(f"\n  Source Excel SAM:")
    print(f"    - Total accounts: {n_accounts}")
    print(f"    - Matrix size: {total_cells} cells ({n_accounts} × {n_accounts})")
    print(f"    - Non-zero values: {non_zero}")
    print(f"    - Zero values: {excel_info['zero_values']}")

    if orig_sam:
        print(f"\n  Original GDX (4D sparse):")
        print(f"    - Stores only non-zero transactions")
        print(f"    - Records: {orig_sam['records']}")
        print(f"    - If sparse: should store ~{non_zero} records")
        print(f"    - Difference from expected: {orig_sam['records'] - non_zero}")

        if orig_sam["records"] == non_zero:
            print(f"    ✓ Matches expected non-zero count!")
        elif orig_sam["records"] < non_zero:
            print(f"    ⚠ Stores fewer records than non-zero cells (compression?)")
        else:
            print(f"    ⚠ Stores more records than non-zero cells")

    if eq_sam:
        print(f"\n  Equilibria GDX (2D sparse):")
        print(f"    - Stores only non-zero transactions")
        print(f"    - Records: {eq_sam['records']}")
        print(f"    - If sparse: should store ~{non_zero} records")
        print(f"    - Difference from expected: {eq_sam['records'] - non_zero}")

        if eq_sam["records"] == non_zero:
            print(f"    ✓ Matches expected non-zero count!")
        elif eq_sam["records"] < non_zero:
            print(f"    ⚠ Stores fewer records than non-zero cells")
        else:
            print(f"    ⚠ Stores more records than non-zero cells")

    print("\n## 4. Key Differences")

    if orig_sam and eq_sam:
        print(f"\n  Dimensionality:")
        print(
            f"    - Original: {orig_sam['dimension']}D (likely: set×set×account×account)"
        )
        print(f"    - Equilibria: {eq_sam['dimension']}D (account×account)")

        print(f"\n  Records:")
        print(f"    - Original: {orig_sam['records']}")
        print(f"    - Equilibria: {eq_sam['records']}")
        print(f"    - Difference: {eq_sam['records'] - orig_sam['records']}")

        if eq_sam["records"] > orig_sam["records"]:
            print(
                f"    ⚠ Equilibria stores {eq_sam['records'] - orig_sam['records']} more records"
            )
            print(f"      This could mean:")
            print(f"      1. Different accounts (more in equilibria)")
            print(f"      2. Different zero handling (equilibria stores some zeros)")
            print(f"      3. Data processing differences")
        elif eq_sam["records"] < orig_sam["records"]:
            print(
                f"    ⚠ Equilibria stores {orig_sam['records'] - eq_sam['records']} fewer records"
            )
            print(f"      This could mean:")
            print(f"      1. Different accounts (fewer in equilibria)")
            print(f"      2. Original stores some zeros that equilibria doesn't")
        else:
            print(f"    ✓ Same number of records")

    print("\n## 5. Does It Matter for GAMS?")
    print("\n  Short answer: NO, as long as transaction values are identical.")
    print("\n  Why it doesn't matter:")
    print("  • GAMS can read both 2D and 4D parameters")
    print("  • GAMS treats missing values as zero by default")
    print("  • Both formats store the same underlying data (just indexed differently)")
    print("  • Equilibria's 2D format is actually MORE intuitive for SAM")
    print("\n  What matters for GAMS compatibility:")
    print("  ✓ Transaction values must be identical")
    print("  ✓ Account/element names must match (case-insensitive)")
    print("  ✓ Domain references must be valid")
    print("\n  What does NOT matter:")
    print("  • Number of symbols (1 vs 2)")
    print("  • Storage dimension (2D vs 4D)")
    print("  • Record count (as long as it's the non-zero values)")


def main():
    """Run the analysis."""
    print("=" * 70)
    print("SAM GDX STORAGE DIFFERENCE ANALYSIS")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parents[2]

    # Paths
    original_gdx = repo_root / "src" / "equilibria" / "templates" / "reference" / "pep2" / "data" / "SAM-V2_0.gdx"
    equilibria_gdx = repo_root / "src" / "equilibria" / "templates" / "data" / "pep" / "SAM-V2_0.gdx"
    excel_sam = repo_root / "src" / "equilibria" / "templates" / "data" / "pep" / "SAM-V2_0.xls"

    # Check files exist
    if not original_gdx.exists():
        print(f"\n❌ Original GDX not found: {original_gdx}")
        print("   Skipping original comparison")
        original_gdx = None

    if not equilibria_gdx.exists():
        print(f"\n❌ Equilibria GDX not found: {equilibria_gdx}")
        return

    if not excel_sam.exists():
        print(f"\n❌ Excel SAM not found: {excel_sam}")
        excel_info = None
    else:
        excel_info = analyze_excel_sam(excel_sam)

    # Analyze GDX files
    if original_gdx:
        original_info = analyze_gdx_structure(original_gdx, "ORIGINAL GDX (cge_babel)")
    else:
        original_info = None

    equilibria_info = analyze_gdx_structure(equilibria_gdx, "EQUILIBRIA GDX")

    # Compare
    if original_info and excel_info:
        compare_structures(original_info, equilibria_info, excel_info)
    elif excel_info:
        print("\n⚠ Cannot do full comparison - original GDX not available")
        print("\n## Equilibria Analysis:")
        print(f"  Accounts: {len(excel_info['accounts'])}")
        print(f"  Expected non-zero records: {excel_info['non_zero']}")
        eq_sam = next(
            (s for s in equilibria_info["symbols"] if s["name"] == "SAM"), None
        )
        if eq_sam:
            print(f"  Actual records: {eq_sam['records']}")
            if eq_sam["records"] == excel_info["non_zero"]:
                print("  ✓ Matches expected count!")
            else:
                print(f"  ⚠ Difference: {eq_sam['records'] - excel_info['non_zero']}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
