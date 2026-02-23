"""Example 8: Model Calibration

This example demonstrates how to:
1. Create a model and SAM
2. Calibrate parameters from SAM data
3. Apply calibration results to the model
"""

import numpy as np
import pandas as pd

from equilibria import Model
from equilibria.babel import SAM
from equilibria.blocks import LeontiefIntermediate
from equilibria.calibration import LeontiefCalibrator, ModelCalibrator
from equilibria.core import Set


def main():
    """Demonstrate model calibration."""
    print("=" * 70)
    print("Example 8: Model Calibration")
    print("=" * 70)

    # Create a simple balanced SAM with sectors and factors
    print("\n" + "-" * 70)
    print("Step 1: Create SAM")
    print("-" * 70)

    # Accounts: 2 sectors (AGR, MFG), 2 factors (LAB, CAP), 1 household (HH)
    accounts = ["AGR", "MFG", "LAB", "CAP", "HH"]

    # Balanced SAM structure:
    # - Intermediate demand (sectors x sectors)
    # - Value added (factors x sectors)
    # - Final demand (sectors x HH)
    # - Factor income (HH x factors)

    sam_data = np.array(
        [
            # AGR   MFG   LAB   CAP   HH
            [10, 20, 0, 0, 70],  # AGR: intermediate + final = 100
            [15, 25, 0, 0, 110],  # MFG: intermediate + final = 150
            [40, 60, 0, 0, 0],  # LAB: value added = 100
            [35, 45, 0, 0, 0],  # CAP: value added = 80
            [0, 0, 100, 80, 0],  # HH:  receives factor income = 180
        ],
        dtype=float,
    )

    sam_df = pd.DataFrame(sam_data, index=accounts, columns=accounts)
    sam = SAM.from_dataframe(sam_df, name="SimpleSAM")

    print(f"\nCreated SAM: {sam.name}")
    print(f"Shape: {sam.data.shape}")

    # Verify balance
    validation = sam.check_balance(tolerance=1e-6)
    print(f"SAM is balanced: {validation['is_balanced']}")

    # Create model
    print("\n" + "-" * 70)
    print("Step 2: Create Model")
    print("-" * 70)

    model = Model(name="CalibratedModel")

    # Add sets
    sectors = Set(name="J", elements=("AGR", "MFG"), description="Sectors")
    factors = Set(name="I", elements=("LAB", "CAP"), description="Factors")
    commodities = Set(name="I_COMM", elements=("AGR", "MFG"), description="Commodities")

    model.add_sets([sectors, factors, commodities])
    print(f"\nAdded {len(model.set_manager.list_sets())} sets")

    # Add blocks (only Leontief for this example)
    model.add_block(LeontiefIntermediate(name="Leontief_INT"))
    print(f"Added {len(model.blocks)} blocks")

    # Show pre-calibration parameters
    print("\n" + "-" * 70)
    print("Step 3: Pre-Calibration Parameters")
    print("-" * 70)

    print("\nBefore calibration:")
    for param_name in model.parameter_manager.list_params():
        param = model.get_parameter(param_name)
        print(f"  {param_name}: shape {param.shape()}")
        if param_name == "a_io":
            print(f"    Initial values:\n{param.value}")

    # Calibrate
    print("\n" + "-" * 70)
    print("Step 4: Calibrate Model")
    print("-" * 70)

    # Create calibrator
    calibrator = ModelCalibrator()
    calibrator.add_calibrator(LeontiefCalibrator())

    print("\nRunning calibration...")

    # Run calibration
    results = calibrator.calibrate(model, sam)

    # Show calibration results
    print("\nCalibration Results:")
    for calibrator_name, result in results.items():
        print(f"\n{calibrator_name}:")
        print(f"  Success: {result.success}")
        print(f"  Messages: {result.messages}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
        print(f"  Statistics: {result.statistics}")

    # Apply results
    print("\n" + "-" * 70)
    print("Step 5: Apply Calibration Results")
    print("-" * 70)

    calibrator.apply_results(model, results)

    print("\nAfter calibration:")
    for param_name in model.parameter_manager.list_params():
        param = model.get_parameter(param_name)
        print(f"\n  {param_name}:")
        print(f"    {param.value}")

    # Show calibrated IO coefficients
    print("\n" + "-" * 70)
    print("Step 6: Calibrated IO Coefficients (a_io)")
    print("-" * 70)

    a_io = model.get_parameter("a_io")
    commodities_list = list(model.set_manager.get("I_COMM").iter_elements())
    sectors_list = list(model.set_manager.get("J").iter_elements())

    print("\nInput-output coefficients:")
    for j_idx, sector in enumerate(sectors_list):
        print(f"\n  {sector} inputs:")
        for i_idx, comm in enumerate(commodities_list):
            coeff = a_io.value[i_idx, j_idx]
            if coeff > 0.001:  # Only show significant coefficients
                print(f"    {comm}: {coeff:.3f}")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)
    print("\nCalibration enables:")
    print("  ✓ Automatic parameter computation from SAM")
    print("  ✓ Consistent base year data")
    print("  ✓ Multiple calibration methods")
    print("=" * 70)


if __name__ == "__main__":
    main()
