"""Generate GDX files for GAMS PEP model from SAM data.

This script converts the PEP SAM Excel file to GDX format for use with GAMS.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from equilibria.babel.gdx.writer import write_gdx
from equilibria.babel.gdx.symbols import Set, Parameter
from equilibria.templates.data.pep import load_default_pep_sam


def generate_pep_gdx_files(output_dir: Path | str | None = None) -> dict[str, Path]:
    """Generate GDX files for GAMS PEP model.

    Args:
        output_dir: Directory to save GDX files (default: reference/pep/)

    Returns:
        Dictionary mapping file types to paths
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "reference" / "pep"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAM data...")
    sam = load_default_pep_sam()

    results = {}

    # Generate SAM GDX
    sam_gdx_path = output_dir / "SAM-V2_0.gdx"
    print(f"Generating {sam_gdx_path}...")
    generate_sam_gdx(sam, sam_gdx_path)
    results["sam"] = sam_gdx_path

    # Generate sets GDX
    sets_gdx_path = output_dir / "pep_sets.gdx"
    print(f"Generating {sets_gdx_path}...")
    generate_sets_gdx(sam, sets_gdx_path)
    results["sets"] = sets_gdx_path

    # Generate calibration data GDX
    calib_gdx_path = output_dir / "pep_calibration.gdx"
    print(f"Generating {calib_gdx_path}...")
    generate_calibration_gdx(sam, calib_gdx_path)
    results["calibration"] = calib_gdx_path

    print("\nGDX files generated successfully:")
    for name, path in results.items():
        print(f"  {name}: {path}")

    return results


def generate_sam_gdx(sam, output_path: Path) -> None:
    """Generate GDX file with SAM matrix."""
    # Get all accounts
    accounts = list(sam.data.index)

    # Create set symbol - elements should be list of lists
    accounts_set = Set(
        name="ACCOUNTS",
        dimensions=1,
        domain=["*"],
        elements=[[acc] for acc in accounts],
    )

    # Create SAM parameter - records should be list of (keys, value) tuples
    sam_values = sam.data.values
    sam_records = []
    for i, row_acc in enumerate(accounts):
        for j, col_acc in enumerate(accounts):
            if sam_values[i, j] != 0:  # Only store non-zero values
                sam_records.append(([row_acc, col_acc], float(sam_values[i, j])))

    sam_param = Parameter(
        name="SAM", dimensions=2, domain=["ACCOUNTS", "ACCOUNTS"], records=sam_records
    )

    # Write GDX file
    symbols = [accounts_set, sam_param]
    write_gdx(str(output_path), symbols)

    print(
        f"  Written {len(accounts)} accounts, {len(sam_records)} non-zero SAM entries"
    )


def generate_sets_gdx(sam, output_path: Path) -> None:
    """Generate GDX file with PEP sets."""
    # Define PEP sets based on standard structure
    # These match the hardcoded sets in PEP1R template

    sets_data = {
        "J": ["agr", "othind", "food", "ser", "adm"],  # Sectors/Industries
        "I": ["agr", "othind", "food", "ser", "adm"],  # Commodities
        "L": ["usk", "sk"],  # Labor types
        "K": ["cap", "land"],  # Capital types
        "F": ["usk", "sk", "cap", "land"],  # All factors
        "H": ["hrp", "hup", "hrr", "hur"],  # Households
        "AG": ["hrp", "hup", "hrr", "hur", "firm", "gvt", "row"],  # Agents
    }

    # Create set symbols - elements should be list of lists
    symbols = []
    for set_name, elements in sets_data.items():
        set_symbol = Set(
            name=set_name,
            dimensions=1,
            domain=["*"],
            elements=[[elem] for elem in elements],
        )
        symbols.append(set_symbol)
        print(f"  Written set {set_name}: {len(elements)} elements")

    # Write GDX file
    write_gdx(str(output_path), symbols)


def generate_calibration_gdx(sam, output_path: Path) -> None:
    """Generate GDX file with calibration data extracted from SAM."""
    from equilibria.core.calibration_data import CalibrationData

    # Create calibration data
    data = CalibrationData(sam, mode="sam")

    # Register mappings (lowercase to match hardcoded sets)
    data.register_set_mapping("F", ["usk", "sk", "cap", "land"])
    data.register_set_mapping("J", ["agr", "othind", "food", "ser", "adm"])
    data.register_set_mapping("I", ["agr", "othind", "food", "ser", "adm"])

    # Define factor and sector names
    factors = ["usk", "sk", "cap", "land"]
    sectors = ["agr", "othind", "food", "ser", "adm"]
    commodities = ["agr", "othind", "food", "ser", "adm"]

    symbols = []

    # Extract and write factor demands (F x J)
    FD0 = data.get_matrix("F", "J")
    fd0_records = []
    for i, f in enumerate(factors):
        for j, s in enumerate(sectors):
            if FD0[i, j] != 0:
                fd0_records.append(([f, s], float(FD0[i, j])))

    fd0_param = Parameter(
        name="FD0", dimensions=2, domain=["F", "J"], records=fd0_records
    )
    symbols.append(fd0_param)
    print(f"  Written FD0: {len(fd0_records)} non-zero entries")

    # Extract value added (sum of FD0 over factors for each sector)
    VA0 = FD0.sum(axis=0)
    va0_records = [([s], float(VA0[j])) for j, s in enumerate(sectors) if VA0[j] != 0]
    va0_param = Parameter(name="VA0", dimensions=1, domain=["J"], records=va0_records)
    symbols.append(va0_param)
    print(f"  Written VA0: {len(va0_records)} entries")

    # Write default/normalized prices (all 1.0 in base year)
    for price_name in ["P0", "PC0", "PD0", "PM0", "PE0"]:
        price_records = [([c], 1.0) for c in commodities]
        price_param = Parameter(
            name=price_name, dimensions=1, domain=["I"], records=price_records
        )
        symbols.append(price_param)
    print(f"  Written base year prices (all normalized to 1.0)")

    # Write factor prices (normalized to 1.0)
    wf0_records = [([f], 1.0) for f in factors]
    wf0_param = Parameter(name="WF0", dimensions=1, domain=["F"], records=wf0_records)
    symbols.append(wf0_param)
    print(f"  Written WF0: factor prices")

    # Write GDX file
    write_gdx(str(output_path), symbols)


def main():
    """Generate all GDX files."""
    print("=" * 70)
    print("Generating GDX Files for GAMS PEP Model")
    print("=" * 70)
    print()

    # Get output directory from templates/data/pep
    data_dir = Path(__file__).parent

    results = generate_pep_gdx_files(data_dir)

    print()
    print("=" * 70)
    print("GDX Generation Complete!")
    print("=" * 70)
    print()
    print("Files generated:")
    for name, path in results.items():
        print(f"  {name}: {path}")
    print()
    print("These files can now be used with GAMS PEP model.")
    print("=" * 70)


if __name__ == "__main__":
    main()
