"""Generate SAM GDX in 4D format matching original cge_babel structure.

This script creates SAM-V2_0.gdx with the exact 4D structure that GAMS expects:
SAM(category1, element1, category2, element2) = value
"""

from pathlib import Path
import pandas as pd
import numpy as np

from equilibria.babel.gdx.writer import write_gdx
from equilibria.babel.gdx.symbols import Parameter


def read_sam_excel_4d(filepath: Path) -> dict:
    """Read SAM Excel and organize by 4D categories.

    Returns:
        Dictionary with structure: {(cat1, elem1, cat2, elem2): value}
    """
    df = pd.read_excel(filepath, header=None)

    # Find data start
    data_start_row = None
    for i in range(len(df)):
        if str(df.iloc[i, 0]).strip() == "L":
            data_start_row = i
            break

    # Get row categories and elements
    row_categories = []
    row_elements = []
    for i in range(data_start_row, len(df)):
        cat = str(df.iloc[i, 0]).strip() if pd.notna(df.iloc[i, 0]) else ""
        elem = str(df.iloc[i, 1]).strip() if pd.notna(df.iloc[i, 1]) else ""
        if elem:  # Only include if has element name
            row_categories.append(cat)
            row_elements.append(elem)

    # Get column categories and elements
    header_row = data_start_row - 1
    col_categories = []
    col_elements = []
    for j in range(2, len(df.columns)):
        # Column categories are in header_row, alternate pattern
        val = (
            str(df.iloc[header_row, j]).strip()
            if pd.notna(df.iloc[header_row, j])
            else ""
        )
        if val:
            col_categories.append(val)
            col_elements.append(val)  # Use same for element in columns

    # Read data
    data = df.iloc[
        data_start_row : data_start_row + len(row_elements), 2 : 2 + len(col_elements)
    ]
    data = data.fillna(0)

    # Build 4D dictionary
    sam_4d = {}
    for i, (row_cat, row_elem) in enumerate(zip(row_categories, row_elements)):
        for j, (col_cat, col_elem) in enumerate(zip(col_categories, col_elements)):
            value = float(data.iloc[i, j])
            if value != 0:  # Sparse storage
                sam_4d[(row_cat, row_elem, col_cat, col_elem)] = value

    return sam_4d


def generate_sam_4d_gdx(excel_path: Path, output_path: Path) -> None:
    """Generate SAM-V2_0.gdx in 4D format.

    Args:
        excel_path: Path to SAM-V2_0.xls
        output_path: Path to output GDX file
    """
    print(f"Reading SAM from: {excel_path}")
    sam_data = read_sam_excel_4d(excel_path)

    print(f"Total 4D entries: {len(sam_data)}")

    # Convert to GDX records format: ([cat1, elem1, cat2, elem2], value)
    records = []
    for (cat1, elem1, cat2, elem2), value in sam_data.items():
        records.append(([cat1, elem1, cat2, elem2], float(value)))

    # Create SAM parameter with 4 dimensions
    sam_param = Parameter(
        name="SAM",
        dimensions=4,
        domain=["*", "*", "*", "*"],  # 4 wildcards for 4D
        records=records,
    )

    # Write GDX file
    symbols = [sam_param]
    write_gdx(str(output_path), symbols)

    print(f"✓ Generated 4D SAM GDX: {output_path}")
    print(f"  Records: {len(records)}")
    print(f"  Format: SAM(cat1, elem1, cat2, elem2) = value")

    # Show sample entries
    print("\nSample entries:")
    for i, (keys, value) in enumerate(records[:5]):
        print(f"  SAM({keys[0]!r}, {keys[1]!r}, {keys[2]!r}, {keys[3]!r}) = {value}")


def main():
    """Generate 4D SAM GDX."""
    excel_path = Path(
        "/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.xls"
    )
    output_path = Path(
        "/Users/marmol/proyectos/equilibria/src/equilibria/templates/data/pep/SAM-V2_0_4D.gdx"
    )

    generate_sam_4d_gdx(excel_path, output_path)

    print("\n" + "=" * 70)
    print("4D SAM GDX Generation Complete!")
    print("=" * 70)
    print(f"\nOutput file: {output_path}")
    print("\nThis GDX should be compatible with GAMS code expecting:")
    print("  SAM('I',i,'AG',h), SAM('J',j,'I',i), etc.")


if __name__ == "__main__":
    main()
