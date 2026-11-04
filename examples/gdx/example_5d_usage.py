"""
Ejemplo práctico de uso de parámetros 5D en modelos CGE.

Los parámetros 5D son comunes en modelos CGE avanzados que consideran
múltiples dimensiones como: sector, producto, región origen, región destino, tiempo.
"""

from pathlib import Path

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def example_5d_trade_matrix():
    """
    Ejemplo: Matriz de comercio 5D
    Dimensiones: sector × producto × región_origen × región_destino × año
    """
    print("="*70)
    print("Ejemplo: Matriz de Comercio 5D")
    print("="*70)
    print("\nEn un modelo CGE multi-regional con series de tiempo, podríamos tener:")
    print("PARAMETER TRADE(sector, product, region_from, region_to, year)")
    print("\nDimensiones:")
    print("  1. Sector de producción (i)")
    print("  2. Producto comercializado (j)")
    print("  3. Región origen (k)")
    print("  4. Región destino (m)")
    print("  5. Año/periodo (n)")

    gdx_file = Path("data/gdx_tests/specimens/5d_test/test_5d.gdx")
    if not gdx_file.exists():
        print("\n⚠️  Archivo de prueba no encontrado")
        return

    data = read_gdx(gdx_file)
    trade = read_parameter_values(data, "p5d_sparse")

    print(f"\n📊 Total de flujos comerciales: {len(trade)}")

    # Analizar por región origen
    print("\n🌎 Flujos por región origen:")
    by_origin = {}
    for (sector, product, origin, dest, period), value in trade.items():
        if origin not in by_origin:
            by_origin[origin] = []
        by_origin[origin].append(((sector, product, dest, period), value))

    for origin, flows in sorted(by_origin.items()):
        print(f"\n  Región {origin}: {len(flows)} flujos")
        for (s, p, d, t), v in sorted(flows)[:3]:
            print(f"    {s}.{p} → {d} ({t}): {v:,.0f}")
        if len(flows) > 3:
            print(f"    ... y {len(flows)-3} más")


def example_5d_io_coefficients():
    """
    Ejemplo: Coeficientes Input-Output 5D
    Dimensiones: sector_comprador × sector_vendedor × región × factor × tiempo
    """
    print("\n" + "="*70)
    print("Ejemplo: Coeficientes Input-Output 5D")
    print("="*70)
    print("\nTabla IO multidimensional:")
    print("PARAMETER IO(buyer_sector, seller_sector, region, factor, period)")

    gdx_file = Path("data/gdx_tests/specimens/5d_test/test_5d.gdx")
    if not gdx_file.exists():
        print("\n⚠️  Archivo de prueba no encontrado")
        return

    data = read_gdx(gdx_file)
    io_table = read_parameter_values(data, "p5d_dense")

    print(f"\n📊 Total de coeficientes: {len(io_table)}")

    # Crear matriz 2D para visualización: sector × región (promediando otras dims)
    print("\n📈 Matriz agregada (sector × región):")
    matrix = {}
    for (sector, product, region, factor, period), value in io_table.items():
        key = (sector, region)
        matrix[key] = matrix.get(key, 0) + value

    # Obtener dimensiones únicas
    sectors = sorted(set(s for s, r in matrix))
    regions = sorted(set(r for s, r in matrix))

    # Imprimir cabecera
    print(f"\n  {'Sector':>8}", end="")
    for r in regions:
        print(f"  {r:>10}", end="")
    print(f"  {'Total':>10}")
    print("  " + "-" * (12 + len(regions) * 12 + 12))

    # Imprimir filas
    for s in sectors:
        print(f"  {s:>8}", end="")
        row_total = 0
        for r in regions:
            val = matrix.get((s, r), 0)
            row_total += val
            print(f"  {val:>10,.0f}", end="")
        print(f"  {row_total:>10,.0f}")

    # Totales por columna
    print("  " + "-" * (12 + len(regions) * 12 + 12))
    print(f"  {'Total':>8}", end="")
    grand_total = 0
    for r in regions:
        col_total = sum(matrix.get((s, r), 0) for s in sectors)
        grand_total += col_total
        print(f"  {col_total:>10,.0f}", end="")
    print(f"  {grand_total:>10,.0f}")


def example_5d_advanced_slicing():
    """Operaciones avanzadas de slicing en 5D."""
    print("\n" + "="*70)
    print("Ejemplo: Slicing Avanzado en 5D")
    print("="*70)

    gdx_file = Path("data/gdx_tests/specimens/5d_test/test_5d.gdx")
    if not gdx_file.exists():
        print("\n⚠️  Archivo de prueba no encontrado")
        return

    data = read_gdx(gdx_file)
    full_data = read_parameter_values(data, "p5d_sparse")

    print(f"\n📊 Dataset completo: {len(full_data)} puntos de datos")

    # Slice 1: Fijar 2 dimensiones (región origen y destino)
    print("\n🔍 Slice 1: Región origen='k2', Región destino='m2'")
    slice1 = {
        (s, p, t): v
        for (s, p, o, d, t), v in full_data.items()
        if o == 'k2' and d == 'm2'
    }
    print(f"  Resultados: {len(slice1)} puntos")
    for key, val in list(slice1.items())[:5]:
        print(f"    {key}: {val:,.0f}")

    # Slice 2: Extraer serie temporal (fijar todas menos tiempo)
    print("\n📅 Slice 2: Serie temporal para sector='i1', producto='j2', origen='k1', destino='m1'")
    time_series = {
        t: v
        for (s, p, o, d, t), v in full_data.items()
        if s == 'i1' and p == 'j2' and o == 'k1' and d == 'm1'
    }
    if time_series:
        print(f"  Períodos encontrados: {sorted(time_series.keys())}")
        for t, v in sorted(time_series.items()):
            print(f"    {t}: {v:,.0f}")
    else:
        print("  (no hay datos para esta combinación)")

    # Slice 3: Matriz 2D de cualquier par de dimensiones
    print("\n📊 Slice 3: Matriz (sector × producto) para origen='k2', destino='m2', periodo='n1'")
    matrix_2d = {
        (s, p): v
        for (s, p, o, d, t), v in full_data.items()
        if o == 'k2' and d == 'm2' and t == 'n1'
    }
    if matrix_2d:
        sectors = sorted(set(s for s, p in matrix_2d))
        products = sorted(set(p for s, p in matrix_2d))

        print(f"\n  {'':>8}", end="")
        for p in products:
            print(f"  {p:>8}", end="")
        print()
        print("  " + "-" * (10 + len(products) * 10))

        for s in sectors:
            print(f"  {s:>8}", end="")
            for p in products:
                val = matrix_2d.get((s, p), 0)
                if val > 0:
                    print(f"  {val:>8,.0f}", end="")
                else:
                    print(f"  {'—':>8}", end="")
            print()
    else:
        print("  (no hay datos para esta combinación)")


def example_5d_aggregations():
    """Agregaciones sofisticadas en múltiples dimensiones."""
    print("\n" + "="*70)
    print("Ejemplo: Agregaciones Múltiples en 5D")
    print("="*70)

    gdx_file = Path("data/gdx_tests/specimens/5d_test/test_5d.gdx")
    if not gdx_file.exists():
        print("\n⚠️  Archivo de prueba no encontrado")
        return

    data = read_gdx(gdx_file)
    full_data = read_parameter_values(data, "p5d_dense")

    print(f"\n📊 Dataset: {len(full_data)} valores")

    # Agregación 1: Reducir a 3D (sumar sobre dim 4 y 5)
    print("\n📉 Agregación 1: Reducir 5D → 3D (sumar sobre m y n)")
    agg_3d = {}
    for (s, p, r, f, t), v in full_data.items():
        key = (s, p, r)
        agg_3d[key] = agg_3d.get(key, 0) + v
    print(f"  Resultado: {len(agg_3d)} combinaciones")
    for key, val in list(agg_3d.items())[:5]:
        print(f"    {key}: {val:,.0f}")

    # Agregación 2: Reducir a 2D (diferentes combinaciones)
    print("\n📉 Agregación 2: Reducir 5D → 2D (sector × región)")
    agg_2d_sr = {}
    for (s, p, r, f, t), v in full_data.items():
        key = (s, r)
        agg_2d_sr[key] = agg_2d_sr.get(key, 0) + v

    sectors = sorted(set(s for s, r in agg_2d_sr))
    regions = sorted(set(r for s, r in agg_2d_sr))

    print(f"\n  {'Sector':>8}", end="")
    for r in regions:
        print(f"  {r:>8}", end="")
    print()
    print("  " + "-" * (10 + len(regions) * 10))

    for s in sectors:
        print(f"  {s:>8}", end="")
        for r in regions:
            val = agg_2d_sr.get((s, r), 0)
            print(f"  {val:>8,.0f}", end="")
        print()

    # Agregación 3: Total por dimensión individual
    print("\n📊 Agregación 3: Totales por cada dimensión")

    dimensions = [
        ("Sector", 0),
        ("Producto", 1),
        ("Región", 2),
        ("Factor", 3),
        ("Período", 4),
    ]

    for dim_name, dim_idx in dimensions:
        totals = {}
        for key, v in full_data.items():
            dim_val = key[dim_idx]
            totals[dim_val] = totals.get(dim_val, 0) + v

        print(f"\n  {dim_name}:")
        for k, v in sorted(totals.items()):
            print(f"    {k}: {v:>10,.0f}")


if __name__ == "__main__":
    example_5d_trade_matrix()
    example_5d_io_coefficients()
    example_5d_advanced_slicing()
    example_5d_aggregations()

    print("\n" + "="*70)
    print("✅ Todos los ejemplos de 5D completados")
    print("="*70)
