#!/usr/bin/env python3
"""
Demo completa del lector GDX.

Muestra todas las capacidades del módulo equilibria.babel.gdx.reader.
"""

import sys
from pathlib import Path

from equilibria.babel.gdx.reader import (
    get_equations,
    get_parameters,
    get_sets,
    get_symbol,
    get_variables,
    read_gdx,
    read_parameter_values,
    read_set_elements,
    read_variable_values,
)


def print_header(text: str, char: str = "=") -> None:
    """Imprime un header decorado."""
    print(f"\n{char * 70}")
    print(text)
    print(f"{char * 70}")


def demo_basic_read(gdx_file: Path) -> dict:
    """Demo 1: Lectura básica."""
    print_header("DEMO 1: Lectura Básica de Archivo GDX")

    gdx_data = read_gdx(gdx_file)

    print(f"\n📄 Archivo: {gdx_data['filepath']}")
    print(f"📊 Versión GDX: {gdx_data['header']['version']}")
    print(f"💻 Plataforma: {gdx_data['header']['platform']}")
    print(f"🏭 Producer: {gdx_data['header']['producer'][:60]}...")
    print(f"🔢 Total símbolos: {len(gdx_data['symbols'])}")
    print(f"🔤 Total elementos únicos: {len(gdx_data['elements'])}")

    return gdx_data


def demo_symbol_listing(gdx_data: dict) -> None:
    """Demo 2: Listado de símbolos."""
    print_header("DEMO 2: Listado de Símbolos")

    print("\nTodos los símbolos:")
    print(f"{'Nombre':<15} {'Tipo':<12} {'Dim':<5} {'Records':<8} {'Descripción'}")
    print("-" * 70)

    for sym in gdx_data["symbols"]:
        name = sym["name"][:14]
        type_name = sym["type_name"]
        dim = sym["dimension"]
        records = sym["records"]
        desc = sym["description"][:30]
        print(f"{name:<15} {type_name:<12} {dim:<5} {records:<8} {desc}")


def demo_filter_by_type(gdx_data: dict) -> None:
    """Demo 3: Filtrado por tipo."""
    print_header("DEMO 3: Filtrado de Símbolos por Tipo")

    # Sets
    sets = get_sets(gdx_data)
    print(f"\n📦 Sets ({len(sets)}):")
    for s in sets:
        print(f"  • {s['name']}: {s['description']}")

    # Parameters
    params = get_parameters(gdx_data)
    print(f"\n🔢 Parameters ({len(params)}):")
    for p in params:
        print(f"  • {p['name']}: dim={p['dimension']}, records={p['records']}")

    # Variables
    variables = get_variables(gdx_data)
    if variables:
        print(f"\n📈 Variables ({len(variables)}):")
        for v in variables:
            print(f"  • {v['name']}: dim={v['dimension']}, records={v['records']}")
    else:
        print("\n📈 Variables: (ninguna)")

    # Equations
    equations = get_equations(gdx_data)
    if equations:
        print(f"\n⚖️  Equations ({len(equations)}):")
        for eq in equations:
            print(f"  • {eq['name']}: dim={eq['dimension']}, records={eq['records']}")
    else:
        print("\n⚖️  Equations: (ninguna)")


def demo_uel(gdx_data: dict) -> None:
    """Demo 4: Unique Element List."""
    print_header("DEMO 4: Unique Element List (UEL)")

    elements = gdx_data["elements"]
    print(f"\nTotal elementos: {len(elements)}")
    print("\nPrimeros 20 elementos:")
    for i, elem in enumerate(elements[:20], 1):
        print(f"  {i:2d}. {elem}")

    if len(elements) > 20:
        print(f"  ... y {len(elements) - 20} más")


def demo_read_sets(gdx_data: dict) -> None:
    """Demo 5: Lectura de sets."""
    print_header("DEMO 5: Lectura de Elementos de Sets")

    sets = get_sets(gdx_data)
    for s in sets:
        print(f"\n📦 Set '{s['name']}':")
        try:
            elements = read_set_elements(gdx_data, s["name"])
            print(f"  Elementos: {len(elements)}")
            for elem in elements[:10]:
                print(f"    • {elem}")
            if len(elements) > 10:
                print(f"    ... y {len(elements) - 10} más")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")


def demo_read_parameters(gdx_data: dict) -> None:
    """Demo 6: Lectura de valores de parámetros."""
    print_header("DEMO 6: Lectura de Valores de Parámetros")

    params = get_parameters(gdx_data)
    for p in params[:3]:  # Solo los primeros 3
        print(f"\n🔢 Parameter '{p['name']}' (dim={p['dimension']}):")
        try:
            values = read_parameter_values(gdx_data, p["name"])
            print(f"  Valores leídos: {len(values)} de {p['records']}")

            # Mostrar algunos valores
            items = list(values.items())[:5]
            for key, val in items:
                key_str = str(key).replace("(", "").replace(")", "").replace("'", "")
                print(f"    {p['name']}({key_str}) = {val:.6f}")

            if len(values) > 5:
                print(f"    ... y {len(values) - 5} más")

        except Exception as e:
            print(f"  ⚠️  Error: {e}")


def demo_read_variables(gdx_data: dict) -> None:
    """Demo 7: Lectura de variables."""
    print_header("DEMO 7: Lectura de Valores de Variables")

    variables = get_variables(gdx_data)
    if not variables:
        print("\n(No hay variables en este archivo)")
        return

    for v in variables[:2]:  # Solo las primeras 2
        print(f"\n📈 Variable '{v['name']}' (dim={v['dimension']}):")
        try:
            values = read_variable_values(gdx_data, v["name"])
            if values:
                print(f"  Valores leídos: {len(values)}")
                items = list(values.items())[:2]
                for key, attrs in items:
                    key_str = str(key).replace("(", "").replace(")", "").replace("'", "")
                    print(f"    {v['name']}({key_str}):")
                    print(f"      level:    {attrs['level']:>12.6f}")
                    print(f"      marginal: {attrs['marginal']:>12.6f}")
                    print(f"      lower:    {attrs['lower']:>12.6f}")
                    print(f"      upper:    {attrs['upper']:>12.6f}")
                    print(f"      scale:    {attrs['scale']:>12.6f}")
            else:
                print("  (No se pudieron leer valores)")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")


def demo_search_symbol(gdx_data: dict, symbol_name: str) -> None:
    """Demo 8: Búsqueda de símbolo específico."""
    print_header("DEMO 8: Búsqueda de Símbolo Específico")

    print(f"\n🔍 Buscando símbolo '{symbol_name}'...")

    symbol = get_symbol(gdx_data, symbol_name)
    if symbol:
        print("✅ Símbolo encontrado!")
        print(f"\n  Nombre:      {symbol['name']}")
        print(f"  Tipo:        {symbol['type_name']}")
        print(f"  Tipo code:   {symbol['type']} (flag: {symbol['type_flag']:#x})")
        print(f"  Dimensión:   {symbol['dimension']}")
        print(f"  Records:     {symbol['records']}")
        print(f"  Descripción: {symbol['description']}")

        # Intentar leer valores
        if symbol["type_name"] == "parameter":
            try:
                values = read_parameter_values(gdx_data, symbol_name)
                print(f"\n  📊 Valores: {len(values)} leídos")
            except Exception as e:
                print(f"\n  ⚠️  No se pudieron leer valores: {e}")
    else:
        print(f"❌ Símbolo '{symbol_name}' no encontrado")


def main():
    """Función principal."""
    # Determinar archivo a leer
    fixtures_dir = Path(__file__).resolve().parents[2] / "data" / "gdx_tests" / "specimens" / "simple"
    gdx_file = fixtures_dir / "simple_test.gdx"

    if len(sys.argv) > 1:
        gdx_file = Path(sys.argv[1])

    if not gdx_file.exists():
        print(f"❌ Error: Archivo no encontrado: {gdx_file}")
        print("\nUso:")
        print(f"  {sys.argv[0]} [archivo.gdx]")
        print(f"\nSi no se especifica archivo, se usa: {fixtures_dir / 'simple_test.gdx'}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🚀 DEMOSTRACIÓN COMPLETA DEL LECTOR GDX")
    print("=" * 70)
    print(f"\nArchivo a analizar: {gdx_file}")

    try:
        # Demo 1: Lectura básica
        gdx_data = demo_basic_read(gdx_file)

        # Demo 2: Listado de símbolos
        demo_symbol_listing(gdx_data)

        # Demo 3: Filtrado por tipo
        demo_filter_by_type(gdx_data)

        # Demo 4: UEL
        demo_uel(gdx_data)

        # Demo 5: Lectura de sets
        demo_read_sets(gdx_data)

        # Demo 6: Lectura de parámetros
        demo_read_parameters(gdx_data)

        # Demo 7: Lectura de variables
        demo_read_variables(gdx_data)

        # Demo 8: Búsqueda de símbolo
        if len(gdx_data["symbols"]) > 0:
            search_name = gdx_data["symbols"][0]["name"]
            demo_search_symbol(gdx_data, search_name)

        print_header("✅ DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")

    except Exception as e:
        print(f"\n❌ Error durante la demostración: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
