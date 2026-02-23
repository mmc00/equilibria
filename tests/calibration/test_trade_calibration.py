"""
Test trade calibration for PEP model.

This script tests the trade calibration module.
"""

import sys
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM_FILE = REPO_ROOT / "src" / "equilibria" / "templates" / "reference" / "pep2" / "data" / "SAM-V2_0.gdx"

sys.path.insert(0, str(REPO_ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from equilibria.babel.gdx.reader import read_gdx
from equilibria.templates.pep_calibration_income import IncomeCalibrator
from equilibria.templates.pep_calibration_production import ProductionCalibrator
from equilibria.templates.pep_calibration_trade import TradeCalibrator


def test_trade_calibration():
    """Test trade calibration with SAM data."""
    print("=" * 70)
    print("Testing PEP Trade Calibration (Phase 4)")
    print("=" * 70)
    
    # Load SAM data
    sam_file = DEFAULT_SAM_FILE
    print(f"\nLoading SAM: {sam_file}")
    sam_data = read_gdx(sam_file)
    print(f"✓ Loaded SAM with {len(sam_data['symbols'])} symbols")
    
    # Step 1: Run income calibration first (prerequisite)
    print("\n" + "=" * 70)
    print("Step 1: Running Income Calibration (Prerequisite)")
    print("=" * 70)
    income_calibrator = IncomeCalibrator(sam_data)
    income_result = income_calibrator.calibrate()
    print("✓ Income calibration complete")
    
    # Step 2: Run production calibration (prerequisite)
    print("\n" + "=" * 70)
    print("Step 2: Running Production Calibration (Prerequisite)")
    print("=" * 70)
    production_calibrator = ProductionCalibrator(sam_data, income_result)
    production_result = production_calibrator.calibrate()
    print("✓ Production calibration complete")
    
    # Step 3: Run trade calibration
    print("\n" + "=" * 70)
    print("Step 3: Running Trade Calibration")
    print("=" * 70)
    trade_calibrator = TradeCalibrator(sam_data, production_result)
    trade_result = trade_calibrator.calibrate()
    print("✓ Trade calibration complete")
    
    # Display results
    print("\n" + "=" * 70)
    print("Trade Calibration Results")
    print("=" * 70)
    
    # Imports
    print("\n1. Import Demand:")
    print("-" * 70)
    print("\n  IMO (Import Quantities):")
    for i, value in sorted(trade_result.IMO.items()):
        if value != 0:
            print(f"    {i:8s}: {value:12,.2f}")
    
    total_imports = sum(trade_result.IMO.values())
    print(f"\n    Total Imports: {total_imports:12,.2f}")
    
    # Exports
    print("\n2. Export Supply:")
    print("-" * 70)
    print("\n  EXDO (World Export Demand):")
    for i, value in sorted(trade_result.EXDO.items()):
        if value != 0:
            print(f"    {i:8s}: {value:12,.2f}")
    
    total_exports = sum(trade_result.EXDO.values())
    print(f"\n    Total Exports: {total_exports:12,.2f}")
    
    # Domestic demand
    print("\n3. Domestic Demand:")
    print("-" * 70)
    print("\n  DDO (Domestic Demand for Local Products):")
    for i, value in sorted(trade_result.DDO.items()):
        if value != 0:
            print(f"    {i:8s}: {value:12,.2f}")
    
    # Composite demand
    print("\n4. Composite Commodity Demand:")
    print("-" * 70)
    print("\n  QO (Composite Demand):")
    for i, value in sorted(trade_result.QO.items()):
        if value != 0:
            print(f"    {i:8s}: {value:12,.2f}")
    
    # Prices
    print("\n5. Trade Prices:")
    print("-" * 70)
    print("\n  PMO (Import Prices):")
    for i, value in sorted(trade_result.PMO.items()):
        print(f"    {i:8s}: {value:8.4f}")
    
    print("\n  PDO (Domestic Prices):")
    for i, value in sorted(trade_result.PDO.items()):
        print(f"    {i:8s}: {value:8.4f}")
    
    # Tax rates
    print("\n6. Trade Tax Rates:")
    print("-" * 70)
    print("\n  ttimO (Import Tax Rates):")
    for i, value in sorted(trade_result.ttimO.items()):
        print(f"    {i:8s}: {value:8.4f}")
    
    print("\n  tticO (Commodity Tax Rates):")
    for i, value in sorted(trade_result.tticO.items()):
        print(f"    {i:8s}: {value:8.4f}")
    
    # CET Parameters
    print("\n7. CET Parameters for Exports:")
    print("-" * 70)
    print("\n  rho_XT (CET Elasticity between Commodities):")
    for j, value in sorted(trade_result.rho_XT.items()):
        print(f"    {j:8s}: {value:8.4f}")
    
    # CES Parameters
    print("\n8. CES Parameters for Imports:")
    print("-" * 70)
    print("\n  rho_M (CES Elasticity for Composite Goods):")
    for i, value in sorted(trade_result.rho_M.items()):
        print(f"    {i:8s}: {value:8.4f}")
    
    print("\n  beta_M (CES Share Parameters):")
    for i, value in sorted(trade_result.beta_M.items()):
        print(f"    {i:8s}: {value:8.4f}")
    
    # Validation
    print("\n" + "=" * 70)
    print("Validation Checks")
    print("=" * 70)
    
    # Check 1: XSO = DSO + EXO
    print("\n1. Checking XSO = DSO + EXO:")
    all_passed = True
    for j in trade_calibrator.sets['J']:
        for i in trade_calibrator.sets['I']:
            xso = trade_result.XSO.get((j, i), 0)
            dso = trade_result.DSO.get((j, i), 0)
            exo = trade_result.EXO.get((j, i), 0)
            expected = dso + exo
            diff = abs(xso - expected)
            if diff > 0.01:
                all_passed = False
                print(f"  ✗ {j}-{i}: {xso:12,.2f} ≠ {expected:12,.2f} (diff: {diff:.2f})")
    if all_passed:
        print("  ✓ All XSO = DSO + EXO checks passed")
    
    # Check 2: QO calculation
    print("\n2. Checking QO = [PMO*IMO + PDO*DDO]/PCO:")
    all_passed = True
    for i in trade_calibrator.sets['I']:
        qo = trade_result.QO.get(i, 0)
        pmo = trade_result.PMO.get(i, 0)
        imo = trade_result.IMO.get(i, 0)
        pdo = trade_result.PDO.get(i, 0)
        ddo = trade_result.DDO.get(i, 0)
        pc = trade_result.PCO.get(i, 1.0)
        
        if pc != 0:
            expected = (pmo * imo + pdo * ddo) / pc
            diff = abs(qo - expected)
            if diff > 0.01:
                all_passed = False
                print(f"  ✗ {i}: {qo:12,.2f} ≠ {expected:12,.2f} (diff: {diff:.2f})")
    if all_passed:
        print("  ✓ All QO calculation checks passed")
    
    # Check 3: Trade balance
    print("\n3. Trade Balance Check:")
    total_imports = sum(trade_result.IMO.values())
    total_exports = sum(trade_result.EXDO.values())
    trade_balance = total_exports - total_imports
    print(f"  Total Exports:  {total_exports:12,.2f}")
    print(f"  Total Imports:  {total_imports:12,.2f}")
    print(f"  Trade Balance:  {trade_balance:12,.2f}")
    print(f"  ✓ Trade balance computed")
    
    print("\n" + "=" * 70)
    print("Phase 4 Calibration Complete")
    print("=" * 70)
    
    return trade_result


def main():
    """Main execution."""
    try:
        result = test_trade_calibration()
        print("\n✓ All tests passed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
