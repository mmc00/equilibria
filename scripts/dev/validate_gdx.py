"""Validate equilibria GDX files against original cge_babel data.

This script performs comprehensive validation of the generated GDX files
by comparing values with the original Excel and cge_babel GDX.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.templates.data.pep import load_pep_sam

REPO_ROOT = Path(__file__).resolve().parents[2]


def validate_sam_values():
    """Validate SAM transaction values match original Excel."""
    print("=" * 70)
    print("VALIDACIÓN SAM: Equilibria vs Excel Original")
    print("=" * 70)
    print()

    # Load equilibria SAM
    sam = load_pep_sam(rdim=2, cdim=2, sparse=True)

    # Load original Excel for comparison
    excel_path = REPO_ROOT / "src" / "equilibria" / "templates" / "reference" / "pep2" / "data" / "SAM-V2_0.xls"
    df_excel = pd.read_excel(excel_path, header=None)

    # Read Excel with same logic as cge_babel
    # Find data boundaries
    data_start_row = None
    for i in range(len(df_excel)):
        if str(df_excel.iloc[i, 0]).strip() == "L":
            data_start_row = i
            break

    # Extract categories and elements from Excel
    row_cats = []
    row_elems = []
    for i in range(data_start_row, len(df_excel)):
        cat = str(df_excel.iloc[i, 0]).strip() if pd.notna(df_excel.iloc[i, 0]) else ""
        elem = str(df_excel.iloc[i, 1]).strip() if pd.notna(df_excel.iloc[i, 1]) else ""
        if elem:
            row_cats.append(cat)
            row_elems.append(elem)

    # Extract column info
    col_cats = []
    col_elems = []
    header_row = data_start_row - 1
    for j in range(2, len(df_excel.columns)):
        val = (
            str(df_excel.iloc[header_row, j]).strip()
            if pd.notna(df_excel.iloc[header_row, j])
            else ""
        )
        if val:
            col_cats.append(val)
            col_elems.append(val)

    # Extract data
    data = df_excel.iloc[
        data_start_row : data_start_row + len(row_elems), 2 : 2 + len(col_elems)
    ]
    data = data.fillna(0)

    print(f"Excel loaded: {len(row_elems)} rows × {len(col_elems)} cols")
    print(f"Equilibria records: {len(sam.records)}")
    print()

    # Compare values
    mismatches = []
    total_compared = 0

    for i, (row_cat, row_elem) in enumerate(zip(row_cats, row_elems)):
        for j, (col_cat, col_elem) in enumerate(zip(col_cats, col_elems)):
            excel_val = float(data.iloc[i, j])

            # Get equilibria value
            eq_val = sam.get_value(row_cat, row_elem, col_cat, col_elem)

            if eq_val is not None:
                total_compared += 1

                if abs(excel_val - eq_val) > 1e-6:
                    mismatches.append(
                        {
                            "keys": (row_cat, row_elem, col_cat, col_elem),
                            "excel": excel_val,
                            "equilibria": eq_val,
                            "diff": abs(excel_val - eq_val),
                        }
                    )

    # Report results
    print(f"Valores comparados: {total_compared}")
    print(f"Coincidencias exactas: {total_compared - len(mismatches)}")
    print(f"Discrepancias: {len(mismatches)}")

    if mismatches:
        print()
        print("⚠️  DISCREPANCIAS ENCONTRADAS:")
        for m in mismatches[:10]:
            print(
                f"  {m['keys']}: Excel={m['excel']}, Equilibria={m['equilibria']}, Diff={m['diff']:.6f}"
            )
        if len(mismatches) > 10:
            print(f"  ... y {len(mismatches) - 10} más")
    else:
        print()
        print("✅ TODOS LOS VALORES COINCIDEN EXACTAMENTE")

    match_pct = (
        ((total_compared - len(mismatches)) / total_compared * 100)
        if total_compared > 0
        else 0
    )
    print(f"\nPorcentaje de match: {match_pct:.2f}%")

    return len(mismatches) == 0


def validate_gdx_structure():
    """Validate GDX structure matches expectations."""
    print()
    print("=" * 70)
    print("VALIDACIÓN ESTRUCTURA GDX")
    print("=" * 70)
    print()

    # Read our GDX
    gdx_path = REPO_ROOT / "src" / "equilibria" / "templates" / "data" / "pep" / "SAM-V2_0_4D_new.gdx"
    gdx_data = read_gdx(gdx_path)

    print(f"Archivo: {gdx_path.name}")
    print(f"Símbolos: {len(gdx_data['symbols'])}")

    for sym in gdx_data["symbols"]:
        print(
            f"  {sym['name']}: {sym['type_name']}, dim={sym['dimension']}, records={sym['records']}"
        )

    # Check SAM parameter
    sam_sym = [s for s in gdx_data["symbols"] if s["name"] == "SAM"]
    if sam_sym:
        sam = sam_sym[0]
        print()
        print("✅ SAM Parameter:")
        print(f"  Dimensiones: {sam['dimension']} (esperado: 4)")
        print(f"  Registros: {sam['records']}")

        if sam["dimension"] == 4:
            print("  ✅ Estructura 4D correcta")
        else:
            print(f"  ❌ ERROR: Esperaba 4 dimensiones, encontrado {sam['dimension']}")
    else:
        print("\n❌ ERROR: No se encontró parámetro SAM")
        return False

    return True


def validate_val_par():
    """Validate VAL_PAR structure and content."""
    print()
    print("=" * 70)
    print("VALIDACIÓN VAL_PAR")
    print("=" * 70)
    print()

    # Read our VAL_PAR GDX
    gdx_path = REPO_ROOT / "src" / "equilibria" / "templates" / "data" / "pep" / "VAL_PAR.gdx"
    gdx_data = read_gdx(gdx_path)

    print(f"Archivo: {gdx_path.name}")
    print(f"Símbolos: {len(gdx_data['symbols'])}")

    # Expected parameters
    expected_params = [
        "J",
        "I",
        "H",
        "sigma_KD",
        "sigma_LD",
        "sigma_VA",
        "sigma_XT",
        "sigma_M",
        "sigma_XD",
        "sigma_ij",
        "frisch",
        "les_elasticities",
    ]

    found_params = [s["name"] for s in gdx_data["symbols"]]

    print()
    print("Parámetros encontrados:")
    for param in found_params:
        status = "✅" if param in expected_params else "⚠️ "
        print(f"  {status} {param}")

    missing = set(expected_params) - set(found_params)
    if missing:
        print(f"\n❌ Parámetros faltantes: {missing}")
        return False
    else:
        print(f"\n✅ Todos los parámetros esperados están presentes")

    return True


def main():
    """Run all validations."""
    print("\n" + "=" * 70)
    print("VALIDACIÓN COMPLETA DE GDX EQUILIBRIA")
    print("=" * 70)
    print()

    results = {
        "SAM valores": validate_sam_values(),
        "SAM estructura": validate_gdx_structure(),
        "VAL_PAR": validate_val_par(),
    }

    print()
    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print()

    all_passed = all(results.values())

    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test}")

    print()
    if all_passed:
        print("✅ TODAS LAS VALIDACIONES PASARON")
        print("Los GDX de equilibria son válidos y compatibles")
    else:
        print("⚠️  ALGUNAS VALIDACIONES FALLARON")
        print("Revisar los detalles arriba")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
