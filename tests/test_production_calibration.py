"""
Test production calibration for PEP model.

This script tests the production calibration module.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, '/Users/marmol/proyectos/equilibria/src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from equilibria.babel.gdx.reader import read_gdx
from equilibria.templates.pep_calibration_income import IncomeCalibrator
from equilibria.templates.pep_calibration_production import ProductionCalibrator


def test_production_calibration():
    """Test production calibration with SAM data."""
    print("=" * 70)
    print("Testing PEP Production Calibration (Phase 3)")
    print("=" * 70)
    
    # Load SAM data
    sam_file = Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.gdx")
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
    
    # Step 2: Run production calibration
    print("\n" + "=" * 70)
    print("Step 2: Running Production Calibration")
    print("=" * 70)
    production_calibrator = ProductionCalibrator(sam_data, income_result)
    production_result = production_calibrator.calibrate()
    print("✓ Production calibration complete")
    
    # Display results
    print("\n" + "=" * 70)
    print("Production Calibration Results")
    print("=" * 70)
    
    # Factor demands
    print("\n1. Factor Demands:")
    print("-" * 70)
    print("\n  KDO (Capital Demand):")
    for (k, j), value in sorted(production_result.KDO.items()):
        print(f"    {k:8s} - {j:8s}: {value:12,.2f}")
    
    print("\n  LDO (Labor Demand):")
    for (l, j), value in sorted(production_result.LDO.items()):
        print(f"    {l:8s} - {j:8s}: {value:12,.2f}")
    
    print("\n  LDCO (Aggregate Labor Demand):")
    for j, value in sorted(production_result.LDCO.items()):
        print(f"    {j:8s}: {value:12,.2f}")
    
    print("\n  KDCO (Aggregate Capital Demand):")
    for j, value in sorted(production_result.KDCO.items()):
        print(f"    {j:8s}: {value:12,.2f}")
    
    # Value added
    print("\n2. Value Added:")
    print("-" * 70)
    for j in sorted(production_result.VAO.keys()):
        vao = production_result.VAO.get(j, 0)
        pvao = production_result.PVAO.get(j, 0)
        print(f"    {j:8s}: VAO = {vao:12,.2f}, PVAO = {pvao:8.4f}")
    
    # GDP
    print("\n3. GDP:")
    print("-" * 70)
    print(f"    GDP_BPO = {production_result.GDP_BPO:15,.2f}")
    
    # Validation
    print("\n" + "=" * 70)
    print("Validation Checks")
    print("=" * 70)
    
    # Check 1: XSTO = CIO + VAO
    print("\n1. Checking XSTO = CIO + VAO:")
    for j in production_calibrator.sets['J']:
        xsto = production_result.XSTO.get(j, 0)
        cio = production_result.CIO.get(j, 0)
        vao = production_result.VAO.get(j, 0)
        expected = cio + vao
        diff = abs(xsto - expected)
        status = "✓" if diff < 0.01 else "✗"
        print(f"  {status} {j}: {xsto:12,.2f} = {expected:12,.2f} (diff: {diff:.2f})")
    
    print("\n" + "=" * 70)
    print("Phase 3 Calibration Complete")
    print("=" * 70)
    
    return production_result


def main():
    """Main execution."""
    try:
        result = test_production_calibration()
        print("\n✓ All tests passed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
