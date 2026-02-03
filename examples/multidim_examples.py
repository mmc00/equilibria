"""
Ejemplo práctico de lectura de parámetros multidimensionales desde GDX.

Este ejemplo muestra cómo leer y trabajar con parámetros de 3+ dimensiones,
típicos en modelos CGE (Computable General Equilibrium).
"""

from pathlib import Path
from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def example_3d_parameter():
    """Ejemplo con parámetro 3D: trade flows por origen, destino y bien."""
    print("="*60)
    print("Ejemplo 1: Parámetro 3D - Flujos de Comercio")
    print("="*60)
    
    # En un modelo CGE real, podrías tener:
    # PARAMETER TRADE(i,r,rp) "Trade flows from region r to rp for good i"
    
    gdx_file = Path("tests/fixtures/multidim_test.gdx")
    if not gdx_file.exists():
        print("⚠️  Archivo de prueba no encontrado")
        print("Ejecutar: gams tests/fixtures/generate_multidim_test.gms")
        return
    
    data = read_gdx(gdx_file)
    
    # Leer parámetro 3D sparse
    trade_flows = read_parameter_values(data, "p3d_sparse")
    
    print(f"\nTotal de flujos comerciales: {len(trade_flows)}")
    print("\nFlujos por bien:")
    
    # Agrupar por primera dimensión (bien)
    by_good = {}
    for (good, origin, dest), value in trade_flows.items():
        if good not in by_good:
            by_good[good] = []
        by_good[good].append(((origin, dest), value))
    
    for good, flows in sorted(by_good.items()):
        print(f"\n  {good}:")
        for (origin, dest), value in sorted(flows):
            print(f"    {origin} → {dest}: {value:,.0f}")


def example_4d_parameter():
    """Ejemplo con parámetro 4D: IO table por sector, producto, región, tiempo."""
    print("\n" + "="*60)
    print("Ejemplo 2: Parámetro 4D - Tabla Input-Output")
    print("="*60)
    
    gdx_file = Path("tests/fixtures/multidim_test.gdx")
    if not gdx_file.exists():
        print("⚠️  Archivo de prueba no encontrado")
        return
    
    data = read_gdx(gdx_file)
    
    # Leer parámetro 4D sparse
    io_table = read_parameter_values(data, "p4d_sparse")
    
    print(f"\nTotal de coeficientes IO: {len(io_table)}")
    
    # Analizar por región
    by_region = {}
    for (region, sector, product, factor), value in io_table.items():
        if region not in by_region:
            by_region[region] = []
        by_region[region].append(((sector, product, factor), value))
    
    print("\nCoeficientes por región:")
    for region, coeffs in sorted(by_region.items()):
        print(f"\n  {region}: {len(coeffs)} coeficientes")
        for (sector, product, factor), value in sorted(coeffs)[:3]:  # Mostrar primeros 3
            print(f"    {sector}.{product}.{factor}: {value:,.0f}")
        if len(coeffs) > 3:
            print(f"    ... y {len(coeffs)-3} más")


def example_slicing():
    """Ejemplo de slicing: extraer sub-tablas de parámetros multidimensionales."""
    print("\n" + "="*60)
    print("Ejemplo 3: Slicing de Parámetros Multidimensionales")
    print("="*60)
    
    gdx_file = Path("tests/fixtures/multidim_test.gdx")
    if not gdx_file.exists():
        print("⚠️  Archivo de prueba no encontrado")
        return
    
    data = read_gdx(gdx_file)
    io_table = read_parameter_values(data, "p4d_sparse")
    
    # Slice 1: Fijar región = 'i1'
    print("\nSlice para región 'i1':")
    region_i1 = {
        (s, p, f): v 
        for (r, s, p, f), v in io_table.items() 
        if r == 'i1'
    }
    print(f"  {len(region_i1)} coeficientes")
    for key, val in list(region_i1.items())[:3]:
        print(f"    {key}: {val}")
    
    # Slice 2: Fijar sector = 'j1'
    print("\nSlice para sector 'j1':")
    sector_j1 = {
        (r, p, f): v 
        for (r, s, p, f), v in io_table.items() 
        if s == 'j1'
    }
    print(f"  {len(sector_j1)} coeficientes")
    for key, val in list(sector_j1.items())[:3]:
        print(f"    {key}: {val}")
    
    # Slice 3: Matriz 2D (región, factor) para sector='j1', producto='k1'
    print("\nMatriz 2D para sector='j1', producto='k1':")
    matrix_2d = {
        (r, f): v 
        for (r, s, p, f), v in io_table.items() 
        if s == 'j1' and p == 'k1'
    }
    if matrix_2d:
        print("  Región  Factor  Valor")
        print("  " + "-"*30)
        for (r, f), v in sorted(matrix_2d.items()):
            print(f"  {r:6}  {f:6}  {v:,.0f}")
    else:
        print("  (vacía)")


def example_aggregation():
    """Ejemplo de agregación: sumar valores sobre dimensiones."""
    print("\n" + "="*60)
    print("Ejemplo 4: Agregación de Parámetros")
    print("="*60)
    
    gdx_file = Path("tests/fixtures/multidim_test.gdx")
    if not gdx_file.exists():
        print("⚠️  Archivo de prueba no encontrado")
        return
    
    data = read_gdx(gdx_file)
    io_table = read_parameter_values(data, "p4d_sparse")
    
    # Agregación 1: Total por región
    print("\nTotal por región:")
    by_region = {}
    for (region, sector, product, factor), value in io_table.items():
        by_region[region] = by_region.get(region, 0) + value
    
    for region, total in sorted(by_region.items()):
        print(f"  {region}: {total:,.0f}")
    
    # Agregación 2: Total por sector (sumando sobre regiones, productos, factores)
    print("\nTotal por sector:")
    by_sector = {}
    for (region, sector, product, factor), value in io_table.items():
        by_sector[sector] = by_sector.get(sector, 0) + value
    
    for sector, total in sorted(by_sector.items()):
        print(f"  {sector}: {total:,.0f}")
    
    # Agregación 3: Matriz region × factor (sumando sobre sector y producto)
    print("\nMatriz región × factor:")
    matrix = {}
    for (region, sector, product, factor), value in io_table.items():
        key = (region, factor)
        matrix[key] = matrix.get(key, 0) + value
    
    # Get unique regions and factors
    regions = sorted(set(r for r, f in matrix.keys()))
    factors = sorted(set(f for r, f in matrix.keys()))
    
    # Print header
    print(f"  {'':6}", end="")
    for f in factors:
        print(f"  {f:>8}", end="")
    print()
    print("  " + "-" * (8 + len(factors) * 10))
    
    # Print rows
    for r in regions:
        print(f"  {r:6}", end="")
        for f in factors:
            val = matrix.get((r, f), 0)
            print(f"  {val:8,.0f}", end="")
        print()


if __name__ == "__main__":
    example_3d_parameter()
    example_4d_parameter()
    example_slicing()
    example_aggregation()
    
    print("\n" + "="*60)
    print("✅ Ejemplos completados")
    print("="*60)
