"""
Ejemplo de lectura de variables y ecuaciones desde GDX.
"""

from pathlib import Path

from equilibria.babel.gdx.reader import (
    get_equations,
    get_variables,
    read_equation_values,
    read_gdx,
    read_variable_values,
)


def main():
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    gdx_file = fixtures_dir / "variables_equations_test.gdx"

    if not gdx_file.exists():
        print(f"Archivo no encontrado: {gdx_file}")
        return

    print("Leyendo archivo con variables y ecuaciones...")
    gdx_data = read_gdx(gdx_file)

    print("\nVariables:")
    variables = get_variables(gdx_data)
    for v in variables:
        print(f"  {v['name']}: dim={v['dimension']}, records={v['records']}")

    print("\nEcuaciones:")
    equations = get_equations(gdx_data)
    for eq in equations:
        print(f"  {eq['name']}: dim={eq['dimension']}, records={eq['records']}")

    # Intentar leer valores (puede estar vacío si el formato es complejo)
    if variables:
        var_name = variables[0]["name"]
        print(f"\nIntentando leer variable '{var_name}'...")
        try:
            values = read_variable_values(gdx_data, var_name)
            print(f"  Valores leídos: {len(values)}")
            if values:
                for key, attrs in list(values.items())[:2]:
                    print(f"    {key}: level={attrs['level']}, marginal={attrs['marginal']}")
            else:
                print("  (No se pudieron leer valores - formato complejo)")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
