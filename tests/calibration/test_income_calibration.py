"""
Test income calibration for PEP model.

This script tests the income calibration module and compares results with GAMS.
"""

import sys
from pathlib import Path

sys.path.insert(0, '/Users/marmol/proyectos/equilibria/src')

from equilibria.babel.gdx.reader import read_gdx
from equilibria.templates.pep_calibration_income import IncomeCalibrator


def test_income_calibration():
    """Test income calibration with SAM data."""
    print("=" * 70)
    print("Testing PEP Income Calibration (Phase 1)")
    print("=" * 70)
    
    # Load SAM data
    sam_file = Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.gdx")
    print(f"\nLoading SAM: {sam_file}")
    sam_data = read_gdx(sam_file)
    print(f"✓ Loaded SAM with {len(sam_data['symbols'])} symbols")
    
    # Create calibrator
    print("\nCreating income calibrator...")
    calibrator = IncomeCalibrator(sam_data)
    
    # Run calibration
    print("Running calibration (GAMS Section 4.1)...")
    result = calibrator.calibrate()
    
    # Display results
    print("\n" + "=" * 70)
    print("Calibration Results")
    print("=" * 70)
    
    # Household incomes
    print("\n1. Household Incomes:")
    for h in calibrator.sets['H']:
        print(f"  {h}:")
        print(f"    YHKO  = {result.YHKO.get(h, 0):,.2f}  (Capital income)")
        print(f"    YHLO  = {result.YHLO.get(h, 0):,.2f}  (Labor income)")
        print(f"    YHTRO = {result.YHTRO.get(h, 0):,.2f}  (Transfer income)")
        print(f"    YHO   = {result.YHO.get(h, 0):,.2f}  (Total income)")
        print(f"    YDHO  = {result.YDHO.get(h, 0):,.2f}  (Disposable income)")
        print(f"    CTHO  = {result.CTHO.get(h, 0):,.2f}  (Consumption)")
    
    # Firm incomes
    print("\n2. Firm Incomes:")
    for f in calibrator.sets['F']:
        print(f"  {f}:")
        print(f"    YFKO  = {result.YFKO.get(f, 0):,.2f}  (Capital income)")
        print(f"    YFTRO = {result.YFTRO.get(f, 0):,.2f}  (Transfer income)")
        print(f"    YFO   = {result.YFO.get(f, 0):,.2f}  (Total income)")
        print(f"    YDFO  = {result.YDFO.get(f, 0):,.2f}  (Disposable income)")
    
    # Government income
    print("\n3. Government Income:")
    print(f"  YGKO    = {result.YGKO:,.2f}  (Capital income)")
    print(f"  TDHTO   = {result.TDHTO:,.2f}  (Household taxes)")
    print(f"  TDFTO   = {result.TDFTO:,.2f}  (Firm taxes)")
    print(f"  TICTO   = {result.TICTO:,.2f}  (Indirect taxes)")
    print(f"  TIMTO   = {result.TIMTO:,.2f}  (Import taxes)")
    print(f"  TIXTO   = {result.TIXTO:,.2f}  (Export taxes)")
    print(f"  TIWTO   = {result.TIWTO:,.2f}  (Labor taxes)")
    print(f"  TIKTO   = {result.TIKTO:,.2f}  (Capital taxes)")
    print(f"  TIPTO   = {result.TIPTO:,.2f}  (Production taxes)")
    print(f"  TPRODNO = {result.TPRODNO:,.2f}  (Total production taxes)")
    print(f"  TPRCTSO = {result.TPRCTSO:,.2f}  (Total commodity taxes)")
    print(f"  YGTRO   = {result.YGTRO:,.2f}  (Transfer income)")
    print(f"  YGO     = {result.YGO:,.2f}  (Total government income)")
    
    # Rest of world
    print("\n4. Rest of World:")
    print(f"  YROWO   = {result.YROWO:,.2f}  (Income)")
    print(f"  CABO    = {result.CABO:,.2f}  (Current account)")
    
    # Investment
    print("\n5. Investment:")
    print(f"  ITO     = {result.ITO:,.2f}  (Total investment)")
    
    # Shares calibration
    print("\n6. Shares Calibration (Phase 2):")
    print("-" * 70)
    
    # lambda_RK shares
    print("\n  lambda_RK (Capital Income Shares):")
    for (ag, k), value in sorted(result.lambda_RK.items()):
        print(f"    {ag:8s} - {k:8s}: {value:.6f}")
    
    # lambda_WL shares
    print("\n  lambda_WL (Labor Income Shares):")
    for (h, l), value in sorted(result.lambda_WL.items()):
        print(f"    {h:8s} - {l:8s}: {value:.6f}")
    
    # lambda_TR households
    print("\n  lambda_TR to Households:")
    for (agng, h), value in sorted(result.lambda_TR_households.items()):
        print(f"    {agng:8s} -> {h:8s}: {value:.6f}")
    
    # lambda_TR firms
    print("\n  lambda_TR to Firms:")
    for (ag, f), value in sorted(result.lambda_TR_firms.items()):
        print(f"    {ag:8s} -> {f:8s}: {value:.6f}")
    
    # sh1O and tr1O
    print("\n  Marginal Propensities:")
    for h in calibrator.sets['H']:
        sh1 = result.sh1O.get(h, 0)
        tr1 = result.tr1O.get(h, 0)
        print(f"    {h:8s}: sh1O = {sh1:8.4f}, tr1O = {tr1:8.4f}")
    
    # Validation checks
    print("\n" + "=" * 70)
    print("Validation Checks")
    print("=" * 70)
    
    # Check 1: YHO = YHLO + YHKO + YHTRO
    print("\n1. Checking YHO = YHLO + YHKO + YHTRO:")
    for h in calibrator.sets['H']:
        lhs = result.YHO.get(h, 0)
        rhs = result.YHLO.get(h, 0) + result.YHKO.get(h, 0) + result.YHTRO.get(h, 0)
        diff = abs(lhs - rhs)
        status = "✓" if diff < 0.01 else "✗"
        print(f"  {status} {h}: {lhs:,.2f} = {rhs:,.2f} (diff: {diff:.2f})")
    
    # Check 2: Government budget identity
    print("\n2. Government budget identity:")
    gov_revenue = result.YGKO + result.TDHTO + result.TDFTO + result.TPRODNO + result.TPRCTSO + result.YGTRO
    print(f"  Total revenue: {gov_revenue:,.2f}")
    print(f"  YGO: {result.YGO:,.2f}")
    print(f"  Diff: {abs(gov_revenue - result.YGO):.2f}")
    
    # Check 3: Shares sum to 1
    print("\n3. Checking shares sum to 1:")
    
    # lambda_RK should sum to 1 for each k
    print("  lambda_RK by capital type:")
    for k in calibrator.sets['K']:
        total = sum(result.lambda_RK.get((ag, k), 0) for ag in calibrator.sets['AG'])
        status = "✓" if abs(total - 1.0) < 0.01 else "✗"
        print(f"    {status} {k}: sum = {total:.4f}")
    
    # lambda_WL should sum to 1 for each l
    print("  lambda_WL by labor type:")
    for l in calibrator.sets['L']:
        total = sum(result.lambda_WL.get((h, l), 0) for h in calibrator.sets['H'])
        status = "✓" if abs(total - 1.0) < 0.01 else "✗"
        print(f"    {status} {l}: sum = {total:.4f}")
    
    print("\n" + "=" * 70)
    print("Phase 1 Calibration Complete")
    print("=" * 70)
    
    return result


def compare_with_gams(our_result):
    """Compare calibration results with GAMS baseline."""
    print("\n" + "=" * 70)
    print("Comparison with GAMS Baseline")
    print("=" * 70)
    
    # Load GAMS baseline
    gams_file = Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/gams/results/all_data_baseline.gdx")
    if not gams_file.exists():
        print(f"⚠ GAMS baseline not found: {gams_file}")
        print("  Run GAMS first to generate baseline")
        return
    
    print(f"\nLoading GAMS baseline: {gams_file}")
    gams_data = read_gdx(gams_file)
    print(f"✓ Loaded GAMS baseline with {len(gams_data['symbols'])} symbols")
    
    # Helper to extract parameter from GAMS data
    def get_gams_param(param_name, *indices):
        """Extract parameter value from GAMS GDX data."""
        for sym in gams_data.get('symbols', []):
            if sym.get('name') == param_name:
                # Check if it's a scalar parameter
                if sym.get('dimension') == 0:
                    records = sym.get('records', [])
                    if records and len(records) > 0:
                        return float(records[0].get('value', 0))
                    return 0.0
                
                # For indexed parameters, find matching record
                records = sym.get('records', [])
                for record in records:
                    record_indices = tuple(str(i).lower() for i in record.get('indices', []))
                    query_indices = tuple(str(i).lower() for i in indices)
                    if record_indices == query_indices:
                        return float(record.get('value', 0))
                return 0.0
        return None
    
    # Compare household incomes
    print("\n1. Household Income Comparison:")
    print("-" * 70)
    
    households = ['hrp', 'hup', 'hrr', 'hur']
    total_errors = 0
    max_error = 0
    
    for h in households:
        print(f"\n  {h.upper()}:")
        
        # YHKO
        our_yhko = our_result.YHKO.get(h, 0)
        gams_yhko = get_gams_param('YHKO', h)
        if gams_yhko is not None:
            diff = abs(our_yhko - gams_yhko)
            pct_diff = (diff / gams_yhko * 100) if gams_yhko != 0 else 0
            status = "✓" if diff < 0.01 else "✗"
            print(f"    {status} YHKO:  Our={our_yhko:>12,.2f}  GAMS={gams_yhko:>12,.2f}  Diff={diff:>10.2f} ({pct_diff:>5.2f}%)")
            total_errors += diff
            max_error = max(max_error, diff)
        
        # YHLO
        our_yhlo = our_result.YHLO.get(h, 0)
        gams_yhlo = get_gams_param('YHLO', h)
        if gams_yhlo is not None:
            diff = abs(our_yhlo - gams_yhlo)
            pct_diff = (diff / gams_yhlo * 100) if gams_yhlo != 0 else 0
            status = "✓" if diff < 0.01 else "✗"
            print(f"    {status} YHLO:  Our={our_yhlo:>12,.2f}  GAMS={gams_yhlo:>12,.2f}  Diff={diff:>10.2f} ({pct_diff:>5.2f}%)")
            total_errors += diff
            max_error = max(max_error, diff)
        
        # YHO
        our_yho = our_result.YHO.get(h, 0)
        gams_yho = get_gams_param('YHO', h)
        if gams_yho is not None:
            diff = abs(our_yho - gams_yho)
            pct_diff = (diff / gams_yho * 100) if gams_yho != 0 else 0
            status = "✓" if diff < 0.01 else "✗"
            print(f"    {status} YHO:   Our={our_yho:>12,.2f}  GAMS={gams_yho:>12,.2f}  Diff={diff:>10.2f} ({pct_diff:>5.2f}%)")
            total_errors += diff
            max_error = max(max_error, diff)
    
    # Compare government totals
    print("\n2. Government Income Comparison:")
    print("-" * 70)
    
    gov_vars = [
        ('TDHTO', our_result.TDHTO),
        ('TDFTO', our_result.TDFTO),
        ('TICTO', our_result.TICTO),
        ('TIMTO', our_result.TIMTO),
        ('TPRODNO', our_result.TPRODNO),
        ('TPRCTSO', our_result.TPRCTSO),
        ('YGO', our_result.YGO),
    ]
    
    for var_name, our_value in gov_vars:
        gams_value = get_gams_param(var_name)
        if gams_value is not None:
            diff = abs(our_value - gams_value)
            pct_diff = (diff / gams_value * 100) if gams_value != 0 else 0
            status = "✓" if diff < 0.01 else "✗"
            print(f"  {status} {var_name:8s}: Our={our_value:>12,.2f}  GAMS={gams_value:>12,.2f}  Diff={diff:>10.2f} ({pct_diff:>5.2f}%)")
            total_errors += diff
            max_error = max(max_error, diff)
    
    # Compare investment
    print("\n3. Investment Comparison:")
    print("-" * 70)
    gams_ito = get_gams_param('ITO')
    if gams_ito is not None:
        diff = abs(our_result.ITO - gams_ito)
        pct_diff = (diff / gams_ito * 100) if gams_ito != 0 else 0
        status = "✓" if diff < 0.01 else "✗"
        print(f"  {status} ITO:     Our={our_result.ITO:>12,.2f}  GAMS={gams_ito:>12,.2f}  Diff={diff:>10.2f} ({pct_diff:>5.2f}%)")
        total_errors += diff
        max_error = max(max_error, diff)
    
    # Summary
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"  Total absolute error: {total_errors:,.2f}")
    print(f"  Maximum error:        {max_error:,.2f}")
    print(f"  Status: {'✓ PASS' if max_error < 0.01 else '✗ FAIL - Differences detected'}")


def main():
    """Main execution."""
    # Test calibration
    result = test_income_calibration()
    
    # Compare with GAMS
    compare_with_gams(result)
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Review calibration results above")
    print("2. Compare with GAMS baseline when available")
    print("3. Proceed to Phase 2: Shares and Investment")


if __name__ == "__main__":
    main()
