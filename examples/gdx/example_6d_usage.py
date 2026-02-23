"""
Example demonstrating 6D parameter usage in CGE models.

6D parameters can represent extremely complex relationships, such as:
- Bilateral trade flows by sector, time, mode, and origin/destination
- Input-output coefficients with temporal and spatial dimensions
- Multi-regional, multi-sectoral, multi-factor production data
"""

from pathlib import Path

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def example_trade_flow_6d():
    """
    Example: 6D trade flow matrix.
    
    Dimensions:
    - origin region
    - destination region
    - sector
    - time period
    - transport mode
    - trade type (import/export/transit)
    """
    gdx_file = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "test_6d.gdx"

    if not gdx_file.exists():
        print("GDX file not found. Run generate_6d_test.gms first.")
        return

    print("=" * 80)
    print("6D TRADE FLOW ANALYSIS")
    print("=" * 80)

    data = read_gdx(gdx_file)
    trade = read_parameter_values(data, "p6d_dense")

    print(f"\nTotal trade records: {len(trade)}")

    # Example 1: Sum over all transport modes and trade types
    # Result: 4D matrix (origin, destination, sector, time)
    print("\n1. AGGREGATE BY ORIGIN-DESTINATION-SECTOR-TIME:")
    print("-" * 80)

    agg_4d = {}
    for (i, j, k, l, m, n), value in trade.items():
        key_4d = (i, j, k, l)
        agg_4d[key_4d] = agg_4d.get(key_4d, 0) + value

    print(f"   Aggregated to {len(agg_4d)} combinations")
    for key, val in list(agg_4d.items())[:3]:
        print(f"   {key}: {val:,.0f}")

    # Example 2: Extract specific slice (origin i1, time l1)
    print("\n2. SLICE: ORIGIN=i1, TIME=l1:")
    print("-" * 80)

    slice_i1_l1 = {(j, k, m, n): v for (i, j, k, l, m, n), v in trade.items()
                   if i == 'i1' and l == 'l1'}

    print(f"   {len(slice_i1_l1)} records")
    for key, val in list(slice_i1_l1.items())[:5]:
        print(f"   Dest={key[0]}, Sector={key[1]}, Mode={key[2]}, Type={key[3]}: {val:,.0f}")

    # Example 3: Calculate bilateral trade balance
    print("\n3. BILATERAL TRADE BALANCE (origin i1 vs i2):")
    print("-" * 80)

    i1_exports = sum(v for (i, j, k, l, m, n), v in trade.items() if i == 'i1')
    i2_exports = sum(v for (i, j, k, l, m, n), v in trade.items() if i == 'i2')

    print(f"   i1 total exports: {i1_exports:,.0f}")
    print(f"   i2 total exports: {i2_exports:,.0f}")
    print(f"   Balance: {i1_exports - i2_exports:,.0f}")


def example_io_coefficients_6d():
    """
    Example: 6D Input-Output coefficient matrix.
    
    Dimensions:
    - region
    - sector (output)
    - sector (input)
    - factor
    - time period
    - scenario
    """
    gdx_file = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "test_6d.gdx"

    if not gdx_file.exists():
        return

    print("\n\n" + "=" * 80)
    print("6D INPUT-OUTPUT COEFFICIENT ANALYSIS")
    print("=" * 80)

    data = read_gdx(gdx_file)
    io_coef = read_parameter_values(data, "p6d_dense")

    # Example 1: Extract regional IO table (region i1, time l1, scenario n1)
    print("\n1. REGIONAL IO TABLE (region=i1, time=l1, scenario=n1):")
    print("-" * 80)

    regional_io = {(j, k, m): v for (i, j, k, l, m, n), v in io_coef.items()
                   if i == 'i1' and l == 'l1' and n == 'n1'}

    print(f"   {len(regional_io)} coefficient entries")
    print("   Sample coefficients:")
    for (output_sector, input_sector, factor), coef in list(regional_io.items())[:4]:
        print(f"   Output={output_sector}, Input={input_sector}, Factor={factor}: {coef:.2f}")

    # Example 2: Compare scenarios
    print("\n2. SCENARIO COMPARISON (region=i1, time=l1):")
    print("-" * 80)

    scenario_n1 = {(j, k, m): v for (i, j, k, l, m, n), v in io_coef.items()
                   if i == 'i1' and l == 'l1' and n == 'n1'}
    scenario_n2 = {(j, k, m): v for (i, j, k, l, m, n), v in io_coef.items()
                   if i == 'i1' and l == 'l1' and n == 'n2'}

    print(f"   Scenario n1: {len(scenario_n1)} coefficients, sum={sum(scenario_n1.values()):,.0f}")
    print(f"   Scenario n2: {len(scenario_n2)} coefficients, sum={sum(scenario_n2.values()):,.0f}")

    # Calculate differences
    common_keys = set(scenario_n1.keys()) & set(scenario_n2.keys())
    differences = {k: scenario_n2[k] - scenario_n1[k] for k in common_keys}
    max_diff_key = max(differences, key=lambda k: abs(differences[k]))

    print(f"   Max difference at {max_diff_key}: {differences[max_diff_key]:.2f}")


def example_production_data_6d():
    """
    Example: 6D production data.
    
    Dimensions:
    - region
    - sector
    - technology
    - time period
    - input factor
    - output product
    """
    gdx_file = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "test_6d.gdx"

    if not gdx_file.exists():
        return

    print("\n\n" + "=" * 80)
    print("6D PRODUCTION DATA ANALYSIS")
    print("=" * 80)

    data = read_gdx(gdx_file)
    production = read_parameter_values(data, "p6d_sparse")

    print(f"\nTotal production records: {len(production)}")

    # Example 1: Sum by region
    print("\n1. PRODUCTION BY REGION:")
    print("-" * 80)

    by_region = {}
    for (region, sector, tech, time, factor, product), value in production.items():
        by_region[region] = by_region.get(region, 0) + value

    for region, total in sorted(by_region.items()):
        print(f"   {region}: {total:,.2f}")

    # Example 2: Technology comparison
    print("\n2. PRODUCTION BY TECHNOLOGY:")
    print("-" * 80)

    by_tech = {}
    for (region, sector, tech, time, factor, product), value in production.items():
        by_tech[tech] = by_tech.get(tech, 0) + value

    for tech, total in sorted(by_tech.items()):
        print(f"   {tech}: {total:,.2f}")

    # Example 3: Factor intensity
    print("\n3. FACTOR INTENSITY BY SECTOR:")
    print("-" * 80)

    factor_use = {}
    for (region, sector, tech, time, factor, product), value in production.items():
        key = (sector, factor)
        factor_use[key] = factor_use.get(key, 0) + value

    for (sector, factor), intensity in sorted(factor_use.items()):
        print(f"   Sector {sector}, Factor {factor}: {intensity:,.2f}")


def example_advanced_6d_operations():
    """Advanced operations on 6D parameters."""
    gdx_file = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "test_6d.gdx"

    if not gdx_file.exists():
        return

    print("\n\n" + "=" * 80)
    print("ADVANCED 6D OPERATIONS")
    print("=" * 80)

    data = read_gdx(gdx_file)
    param = read_parameter_values(data, "p6d_dense")

    # 1. Multi-dimensional aggregation
    print("\n1. PROGRESSIVE AGGREGATION:")
    print("-" * 80)

    # 6D -> 5D: Sum over last dimension
    agg_5d = {}
    for (i, j, k, l, m, n), v in param.items():
        key_5d = (i, j, k, l, m)
        agg_5d[key_5d] = agg_5d.get(key_5d, 0) + v
    print(f"   6D ({len(param)} entries) -> 5D ({len(agg_5d)} entries)")

    # 5D -> 4D
    agg_4d = {}
    for (i, j, k, l, m), v in agg_5d.items():
        key_4d = (i, j, k, l)
        agg_4d[key_4d] = agg_4d.get(key_4d, 0) + v
    print(f"   5D ({len(agg_5d)} entries) -> 4D ({len(agg_4d)} entries)")

    # 4D -> 3D
    agg_3d = {}
    for (i, j, k, l), v in agg_4d.items():
        key_3d = (i, j, k)
        agg_3d[key_3d] = agg_3d.get(key_3d, 0) + v
    print(f"   4D ({len(agg_4d)} entries) -> 3D ({len(agg_3d)} entries)")

    # 2. Tensor-like operations
    print("\n2. TENSOR OPERATIONS:")
    print("-" * 80)

    # Calculate marginal totals for each dimension
    dims = [set(), set(), set(), set(), set(), set()]
    for (i, j, k, l, m, n), v in param.items():
        dims[0].add(i)
        dims[1].add(j)
        dims[2].add(k)
        dims[3].add(l)
        dims[4].add(m)
        dims[5].add(n)

    print(f"   Dimension sizes: {[len(d) for d in dims]}")
    print(f"   Theoretical max: {' × '.join(str(len(d)) for d in dims)} = "
          f"{sum(1 for _ in param)}")

    # 3. Conditional filtering
    print("\n3. CONDITIONAL FILTERING:")
    print("-" * 80)

    # Filter values above threshold
    threshold = 150000
    high_values = {k: v for k, v in param.items() if v > threshold}
    print(f"   Values > {threshold}: {len(high_values)} / {len(param)}")

    # Find maximum and minimum
    max_key = max(param.items(), key=lambda x: x[1])
    min_key = min(param.items(), key=lambda x: x[1])
    print(f"   Maximum: {max_key[1]:,.0f} at {max_key[0]}")
    print(f"   Minimum: {min_key[1]:,.0f} at {min_key[0]}")


if __name__ == "__main__":
    example_trade_flow_6d()
    example_io_coefficients_6d()
    example_production_data_6d()
    example_advanced_6d_operations()
