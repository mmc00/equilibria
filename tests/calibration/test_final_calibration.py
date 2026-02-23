"""
Test final calibration and integration for PEP model.

This script tests the final calibration module which integrates all phases.
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
from equilibria.templates.pep_calibration_final import FinalCalibrator


def test_final_calibration():
    """Test final calibration with SAM data."""
    print("=" * 70)
    print("Testing PEP Final Calibration (Phase 5)")
    print("=" * 70)
    
    # Load SAM data
    sam_file = DEFAULT_SAM_FILE
    print(f"\nLoading SAM: {sam_file}")
    sam_data = read_gdx(sam_file)
    print(f"✓ Loaded SAM with {len(sam_data['symbols'])} symbols")
    
    # Step 1: Income calibration
    print("\n" + "=" * 70)
    print("Step 1: Running Income Calibration")
    print("=" * 70)
    income_calibrator = IncomeCalibrator(sam_data)
    income_result = income_calibrator.calibrate()
    print("✓ Income calibration complete")
    
    # Step 2: Production calibration
    print("\n" + "=" * 70)
    print("Step 2: Running Production Calibration")
    print("=" * 70)
    production_calibrator = ProductionCalibrator(sam_data, income_result)
    production_result = production_calibrator.calibrate()
    print("✓ Production calibration complete")
    
    # Step 3: Trade calibration
    print("\n" + "=" * 70)
    print("Step 3: Running Trade Calibration")
    print("=" * 70)
    trade_calibrator = TradeCalibrator(sam_data, production_result)
    trade_result = trade_calibrator.calibrate()
    print("✓ Trade calibration complete")
    
    # Step 4: Final calibration
    print("\n" + "=" * 70)
    print("Step 4: Running Final Calibration")
    print("=" * 70)
    final_calibrator = FinalCalibrator(sam_data, income_result, production_result, trade_result)
    final_result = final_calibrator.calibrate()
    print("✓ Final calibration complete")
    
    # Display results
    print("\n" + "=" * 70)
    print("Final Calibration Results")
    print("=" * 70)
    
    # Consumption
    print("\n1. Consumption Block:")
    print("-" * 70)
    print("\n  CO (Household Consumption by Commodity):")
    for h in final_calibrator.sets['H']:
        total = sum(final_result.CO.get((i, h), 0) for i in final_calibrator.sets['I'])
        print(f"    {h:8s}: {total:12,.2f}")
    
    print("\n  CGO (Government Consumption):")
    for i, value in sorted(final_result.CGO.items()):
        print(f"    {i:8s}: {value:12,.2f}")
    total_gov = sum(final_result.CGO.values())
    print(f"    {'Total':8s}: {total_gov:12,.2f}")
    
    print("\n  GO (Total Government Expenditure):")
    print(f"    {final_result.GO:15,.2f}")
    
    # Investment
    print("\n2. Investment:")
    print("-" * 70)
    print("\n  INVO (Investment Demand):")
    for i, value in sorted(final_result.INVO.items()):
        if value != 0:
            print(f"    {i:8s}: {value:12,.2f}")
    
    print("\n  GFCFO (Gross Fixed Capital Formation):")
    print(f"    {final_result.GFCFO:15,.2f}")
    
    # LES Parameters
    print("\n3. LES Parameters:")
    print("-" * 70)
    print("\n  gamma_LES (Marginal Budget Shares) - Sample:")
    displayed = 0
    for (i, h), value in sorted(final_result.gamma_LES.items()):
        if displayed < 8:
            print(f"    {i:8s} - {h:8s}: {value:8.4f}")
            displayed += 1
    if len(final_result.gamma_LES) > 8:
        print(f"    ... and {len(final_result.gamma_LES) - 8} more")
    
    print("\n  Frisch Parameters:")
    for h, value in sorted(final_result.frisch.items()):
        print(f"    {h:8s}: {value:8.4f}")
    
    # GDP Measures
    print("\n4. GDP Measures:")
    print("-" * 70)
    print(f"  GDP_BPO (Basic Prices):     {final_result.GDP_BPO:15,.2f}")
    print(f"  GDP_MPO (Market Prices):    {final_result.GDP_MPO:15,.2f}")
    print(f"  GDP_IBO (Income Approach):  {final_result.GDP_IBO:15,.2f}")
    print(f"  GDP_FDO (Expenditure):      {final_result.GDP_FDO:15,.2f}")
    
    # Real Variables
    print("\n5. Real Variables:")
    print("-" * 70)
    print(f"  GDP_BP_REALO: {final_result.GDP_BP_REALO:15,.2f}")
    print(f"  GDP_MP_REALO: {final_result.GDP_MP_REALO:15,.2f}")
    print(f"  G_REALO:      {final_result.G_REALO:15,.2f}")
    print(f"  GFCF_REALO:   {final_result.GFCF_REALO:15,.2f}")
    
    print("\n  CTH_REALO (Real Consumption by Household):")
    for h, value in sorted(final_result.CTH_REALO.items()):
        print(f"    {h:8s}: {value:12,.2f}")
    
    # Price Indices
    print("\n6. Price Indices (Base Year):")
    print("-" * 70)
    print(f"  PIXCONO (Consumer):    {final_result.PIXCONO:8.4f}")
    print(f"  PIXGDPO (GDP):         {final_result.PIXGDPO:8.4f}")
    print(f"  PIXGVTO (Government):  {final_result.PIXGVTO:8.4f}")
    print(f"  PIXINVO (Investment):  {final_result.PIXINVO:8.4f}")
    
    # Validation
    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)
    
    if final_result.validation_passed:
        print("\n✓ All validation checks passed!")
    else:
        print("\n✗ Validation errors:")
        for error in final_result.validation_errors:
            print(f"  - {error}")
    
    # Summary checks
    print("\n" + "=" * 70)
    print("Summary Checks")
    print("=" * 70)
    
    print("\n1. GDP Consistency Check:")
    gdp_diff = abs(final_result.GDP_BPO - final_result.GDP_FDO)
    gdp_pct = (gdp_diff / final_result.GDP_BPO * 100) if final_result.GDP_BPO != 0 else 0
    print(f"  GDP_BPO vs GDP_FDO difference: {gdp_diff:,.2f} ({gdp_pct:.2f}%)")
    if gdp_pct < 1.0:
        print("  ✓ GDP measures are consistent (<1% difference)")
    else:
        print("  ✗ GDP measures differ by >1%")
    
    print("\n2. LES Parameter Check:")
    for h in final_calibrator.sets['H']:
        gamma_sum = sum(final_result.gamma_LES.get((i, h), 0) for i in final_calibrator.sets['I'])
        print(f"  {h}: gamma sum = {gamma_sum:.4f}")
    
    print("\n3. Final Demand Composition:")
    total_cons = sum(sum(final_result.CO.get((i, h), 0) for h in final_calibrator.sets['H']) for i in final_calibrator.sets['I'])
    total_gov = sum(final_result.CGO.values())
    total_inv = sum(final_result.INVO.values())
    total_exports = sum(trade_result.EXDO.values())
    total_imports = sum(trade_result.IMO.values())
    
    print(f"  Private Consumption: {total_cons:15,.2f}")
    print(f"  Government:          {total_gov:15,.2f}")
    print(f"  Investment:          {total_inv:15,.2f}")
    print(f"  Exports:             {total_exports:15,.2f}")
    print(f"  Imports:             {total_imports:15,.2f}")
    print(f"  Net Exports:         {total_exports - total_imports:15,.2f}")
    
    print("\n" + "=" * 70)
    print("Phase 5 Calibration Complete")
    print("=" * 70)
    
    return final_result


def main():
    """Main execution."""
    try:
        result = test_final_calibration()
        print("\n✓ All tests passed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
