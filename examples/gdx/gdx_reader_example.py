"""
Ejemplo de uso del lector GDX.

Este script demuestra cómo usar el módulo equilibria.babel.gdx.reader
para leer archivos GDX completos.
"""

from pathlib import Path

from equilibria.babel.gdx.reader import (
    get_equations,
    get_parameters,
    get_sets,
    get_symbol,
    get_variables,
    read_equation_values,
    read_gdx,
    read_parameter_values,
    read_set_elements,
    read_variable_values,
)


def main():
    """Ejemplo de lectura completa de un archivo GDX."""
    # Ruta al archivo GDX de prueba
    fixtures_dir = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    gdx_file = fixtures_dir / "simple_test.gdx"

    if not gdx_file.exists():
        print(f"Archivo no encontrado: {gdx_file}")
        print("Ejecuta primero los tests para generar los fixtures.")
        return

    print("=" * 70)
    print("EJEMPLO: Lectura completa de archivo GDX")
    print("=" * 70)
    print()

    # 1. Leer el archivo GDX completo
    print("1. Leyendo archivo GDX...")
    gdx_data = read_gdx(gdx_file)
    print(f"   Archivo: {gdx_data['filepath']}")
    print(f"   Versión GDX: {gdx_data['header']['version']}")
    print(f"   Plataforma: {gdx_data['header']['platform']}")
    print(f"   Producer: {gdx_data['header']['producer']}")
    print()

    # 2. Listar todos los símbolos
    print("2. Símbolos en el archivo:")
    print(f"   Total: {len(gdx_data['symbols'])} símbolos")
    for sym in gdx_data["symbols"]:
        print(
            f"   - {sym['name']:15s} ({sym['type_name']:10s}) "
            f"dim={sym['dimension']}, records={sym['records']}"
        )
    print()

    # 3. Listar elementos únicos (UEL)
    print("3. Unique Element List (UEL):")
    elements = gdx_data["elements"]
    print(f"   Total elementos: {len(elements)}")
    print(f"   Elementos: {', '.join(elements[:10])}")
    if len(elements) > 10:
        print(f"   ... y {len(elements) - 10} más")
    print()

    # 4. Obtener sets
    print("4. Sets:")
    sets = get_sets(gdx_data)
    for s in sets:
        print(f"   - {s['name']}: {s['description']}")
        try:
            set_elements = read_set_elements(gdx_data, s["name"])
            print(f"     Elementos: {set_elements[:5]}")
            if len(set_elements) > 5:
                print(f"     ... y {len(set_elements) - 5} más")
        except Exception as e:
            print(f"     Error leyendo elementos: {e}")
    print()

    # 5. Obtener parámetros
    print("5. Parámetros:")
    params = get_parameters(gdx_data)
    for p in params:
        print(f"   - {p['name']}: {p['description']}")
        print(f"     Dimensión: {p['dimension']}, Records: {p['records']}")
        try:
            values = read_parameter_values(gdx_data, p["name"])
            if values:
                print(f"     Valores leídos: {len(values)}")
                # Mostrar algunos valores de ejemplo
                sample_items = list(values.items())[:3]
                for key, val in sample_items:
                    print(f"       {key} = {val}")
                if len(values) > 3:
                    print(f"       ... y {len(values) - 3} más")
            else:
                print("     (Sin valores leídos)")
        except Exception as e:
            print(f"     Error leyendo valores: {e}")
    print()

    # 6. Obtener variables
    print("6. Variables:")
    variables = get_variables(gdx_data)
    if variables:
        for v in variables:
            print(f"   - {v['name']}: {v['description']}")
            print(f"     Dimensión: {v['dimension']}, Records: {v['records']}")
            try:
                values = read_variable_values(gdx_data, v["name"])
                if values:
                    print(f"     Valores leídos: {len(values)}")
                    # Mostrar un valor de ejemplo con todos los atributos
                    sample_items = list(values.items())[:2]
                    for key, attrs in sample_items:
                        print(f"       {key}:")
                        print(f"         level:    {attrs['level']}")
                        print(f"         marginal: {attrs['marginal']}")
                        print(f"         lower:    {attrs['lower']}")
                        print(f"         upper:    {attrs['upper']}")
                        print(f"         scale:    {attrs['scale']}")
                else:
                    print("     (Sin valores leídos)")
            except Exception as e:
                print(f"     Error leyendo valores: {e}")
    else:
        print("   (No hay variables en este archivo)")
    print()

    # 7. Obtener ecuaciones
    print("7. Ecuaciones:")
    equations = get_equations(gdx_data)
    if equations:
        for eq in equations:
            print(f"   - {eq['name']}: {eq['description']}")
            print(f"     Dimensión: {eq['dimension']}, Records: {eq['records']}")
            try:
                values = read_equation_values(gdx_data, eq["name"])
                if values:
                    print(f"     Valores leídos: {len(values)}")
                    sample_items = list(values.items())[:2]
                    for key, attrs in sample_items:
                        print(f"       {key}:")
                        print(f"         level:    {attrs['level']}")
                        print(f"         marginal: {attrs['marginal']}")
                else:
                    print("     (Sin valores leídos)")
            except Exception as e:
                print(f"     Error leyendo valores: {e}")
    else:
        print("   (No hay ecuaciones en este archivo)")
    print()

    # 8. Buscar un símbolo específico
    print("8. Búsqueda de símbolo específico:")
    symbol_name = "sam"
    symbol = get_symbol(gdx_data, symbol_name)
    if symbol:
        print(f"   Símbolo encontrado: {symbol_name}")
        print(f"   Tipo: {symbol['type_name']}")
        print(f"   Dimensión: {symbol['dimension']}")
        print(f"   Records: {symbol['records']}")
        print(f"   Descripción: {symbol['description']}")
        if symbol["type_name"] == "parameter":
            values = read_parameter_values(gdx_data, symbol_name)
            print(f"   Valores leídos: {len(values)}")
    else:
        print(f"   Símbolo '{symbol_name}' no encontrado")
    print()

    print("=" * 70)
    print("Lectura completa finalizada exitosamente!")
    print("=" * 70)


if __name__ == "__main__":
    main()
