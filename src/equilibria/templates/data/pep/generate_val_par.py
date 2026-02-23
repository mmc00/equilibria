"""Generate VAL_PAR.gdx for equilibria from original Excel file.

This script reads the original VAL_PAR.xlsx and converts it to GDX format
for comparison with the original VAL_PAR.gdx.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from equilibria.babel.gdx.writer import write_gdx
from equilibria.babel.gdx.symbols import Set, Parameter


def read_val_par_excel(filepath: Path) -> dict:
    """Read VAL_PAR.xlsx and extract all parameters.

    Args:
        filepath: Path to VAL_PAR.xlsx

    Returns:
        Dictionary with all parameters organized by type
    """
    # Read the PAR sheet without headers
    df = pd.read_excel(filepath, sheet_name="PAR", header=None)

    params = {
        "sets": {},
        "sigma_KD": {},  # By sector (j)
        "sigma_LD": {},  # By sector (j)
        "sigma_VA": {},  # By sector (j)
        "sigma_XT": {},  # By sector (j)
        "sigma_M": {},  # By commodity (i)
        "sigma_XD": {},  # By commodity (i)
        "sigma_ij": {},  # By sector x commodity (j,i)
        "frisch": {},  # By household (h)
        "les_elasticities": {},  # By commodity x household (i,h)
    }

    # Extract sectors (j) - rows 5-8
    sectors = ["AGR", "IND", "SER", "ADM"]
    for i, sector in enumerate(sectors):
        row_idx = 5 + i
        params["sigma_KD"][sector] = (
            float(df.iloc[row_idx, 1]) if pd.notna(df.iloc[row_idx, 1]) else 0.8
        )
        params["sigma_LD"][sector] = (
            float(df.iloc[row_idx, 2]) if pd.notna(df.iloc[row_idx, 2]) else 0.8
        )
        params["sigma_VA"][sector] = (
            float(df.iloc[row_idx, 3]) if pd.notna(df.iloc[row_idx, 3]) else 1.5
        )
        params["sigma_XT"][sector] = (
            float(df.iloc[row_idx, 4]) if pd.notna(df.iloc[row_idx, 4]) else 2.0
        )

    # Extract commodities (i) - rows 12-16
    commodities = ["AGR", "FOOD", "OTHIND", "SER", "ADM"]
    for i, comm in enumerate(commodities):
        row_idx = 12 + i
        params["sigma_M"][comm] = (
            float(df.iloc[row_idx, 1]) if pd.notna(df.iloc[row_idx, 1]) else 2.0
        )
        params["sigma_XD"][comm] = (
            float(df.iloc[row_idx, 2]) if pd.notna(df.iloc[row_idx, 2]) else 2.0
        )

    # Extract sigma_ij (j,i matrix) - rows 20-23
    for i, sector in enumerate(sectors):
        row_idx = 20 + i
        params["sigma_ij"][sector] = {}
        for j, comm in enumerate(commodities):
            val = df.iloc[row_idx, 1 + j]
            params["sigma_ij"][sector][comm] = float(val) if pd.notna(val) else 2.0

    # Extract households (ag) - rows 27-35
    households = ["HRP", "HUP", "HRR", "HUR"]

    # Frisch parameter
    for i, hh in enumerate(households):
        val = df.iloc[27, 1 + i]
        params["frisch"][hh] = float(val) if pd.notna(val) else -1.5

    # LES elasticities by commodity and household
    for i, comm in enumerate(commodities):
        row_idx = 28 + i
        params["les_elasticities"][comm] = {}
        for j, hh in enumerate(households):
            val = df.iloc[row_idx, 1 + j]
            params["les_elasticities"][comm][hh] = float(val) if pd.notna(val) else 1.0

    # Store sets
    params["sets"] = {
        "j": sectors,
        "i": commodities,
        "h": households,
    }

    return params


def generate_val_par_gdx(excel_path: Path, output_path: Path) -> None:
    """Generate VAL_PAR.gdx from Excel file.

    Args:
        excel_path: Path to VAL_PAR.xlsx
        output_path: Path to output GDX file
    """
    print(f"Reading VAL_PAR from: {excel_path}")
    params = read_val_par_excel(excel_path)

    symbols = []

    # Create sets
    for set_name, elements in params["sets"].items():
        set_symbol = Set(
            name=set_name.upper(),
            dimensions=1,
            domain=["*"],
            elements=[[elem] for elem in elements],
        )
        symbols.append(set_symbol)
        print(f"  Created set {set_name}: {len(elements)} elements")

    # Create parameters indexed by j (sectors)
    for param_name in ["sigma_KD", "sigma_LD", "sigma_VA", "sigma_XT"]:
        records = []
        for sector in params["sets"]["j"]:
            val = params[param_name].get(sector, 0.0)
            records.append(([sector], float(val)))

        param = Parameter(name=param_name, dimensions=1, domain=["J"], records=records)
        symbols.append(param)
        print(f"  Created parameter {param_name}: {len(records)} records")

    # Create parameters indexed by i (commodities)
    for param_name in ["sigma_M", "sigma_XD"]:
        records = []
        for comm in params["sets"]["i"]:
            val = params[param_name].get(comm, 0.0)
            records.append(([comm], float(val)))

        param = Parameter(name=param_name, dimensions=1, domain=["I"], records=records)
        symbols.append(param)
        print(f"  Created parameter {param_name}: {len(records)} records")

    # Create sigma_ij (j,i matrix)
    records = []
    for sector in params["sets"]["j"]:
        for comm in params["sets"]["i"]:
            val = params["sigma_ij"].get(sector, {}).get(comm, 2.0)
            records.append(([sector, comm], float(val)))

    param = Parameter(name="sigma_ij", dimensions=2, domain=["J", "I"], records=records)
    symbols.append(param)
    print(f"  Created parameter sigma_ij: {len(records)} records")

    # Create frisch parameter (indexed by h)
    records = []
    for hh in params["sets"]["h"]:
        val = params["frisch"].get(hh, -1.5)
        records.append(([hh], float(val)))

    param = Parameter(name="frisch", dimensions=1, domain=["H"], records=records)
    symbols.append(param)
    print(f"  Created parameter frisch: {len(records)} records")

    # Create LES elasticities (i,h matrix)
    records = []
    for comm in params["sets"]["i"]:
        for hh in params["sets"]["h"]:
            val = params["les_elasticities"].get(comm, {}).get(hh, 1.0)
            records.append(([comm, hh], float(val)))

    param = Parameter(
        name="les_elasticities", dimensions=2, domain=["I", "H"], records=records
    )
    symbols.append(param)
    print(f"  Created parameter les_elasticities: {len(records)} records")

    # Write GDX file
    write_gdx(str(output_path), symbols)
    print(f"\n✓ Generated VAL_PAR.gdx: {output_path}")
    print(f"  Total symbols: {len(symbols)}")


def main():
    """Generate VAL_PAR.gdx for equilibria."""
    repo_root = Path(__file__).resolve().parents[5]
    excel_path = repo_root / "src" / "equilibria" / "templates" / "reference" / "pep2" / "data" / "VAL_PAR.xlsx"
    output_path = repo_root / "src" / "equilibria" / "templates" / "data" / "pep" / "VAL_PAR.gdx"

    generate_val_par_gdx(excel_path, output_path)


if __name__ == "__main__":
    main()
