"""Example 3: Working with SAM Data

This example demonstrates how to:
1. Create a SAM from a DataFrame
2. Validate and balance the SAM
3. Extract sets from the SAM
4. Export to different formats
"""

import numpy as np
import pandas as pd

from equilibria.babel import SAM


def main():
    """Demonstrate SAM operations."""
    print("=" * 70)
    print("Example 3: Working with SAM Data")
    print("=" * 70)

    # Create a simple SAM matrix
    print("\n" + "-" * 70)
    print("Step 1: Creating a SAM from DataFrame")
    print("-" * 70)

    # Define accounts
    accounts = [
        "agr",
        "mfg",
        "svc",  # Sectors
        "labor",
        "capital",  # Factors
        "hh",
        "gov",  # Institutions
        "row",  # Rest of world
    ]

    # Create a balanced SAM matrix (simplified)
    data = np.array(
        [
            # agr   mfg   svc   labor  capital  hh    gov   row
            [10, 20, 5, 0, 0, 30, 5, 10],  # agr
            [15, 40, 15, 0, 0, 50, 10, 20],  # mfg
            [8, 25, 20, 0, 0, 40, 15, 12],  # svc
            [25, 60, 35, 0, 0, 0, 0, 0],  # labor
            [15, 30, 25, 0, 0, 0, 0, 0],  # capital
            [0, 0, 0, 120, 70, 0, 20, 0],  # hh
            [5, 10, 8, 0, 0, 15, 0, 2],  # gov
            [12, 25, 18, 0, 0, 5, 0, 0],  # row
        ],
        dtype=float,
    )

    # Create DataFrame
    df = pd.DataFrame(data, index=accounts, columns=accounts)

    print("\nSAM Matrix:")
    print(df.to_string())

    # Create SAM object
    sam = SAM.from_dataframe(df, name="SimpleSAM")
    print(f"\nCreated SAM: {sam}")

    # Validate SAM
    print("\n" + "-" * 70)
    print("Step 2: Validating SAM")
    print("-" * 70)

    validation = sam.check_balance(tolerance=1e-6)
    print(f"\nValidation Results:")
    print(f"  Is balanced: {validation['is_balanced']}")
    print(f"  Max difference: {validation['max_difference']:.2e}")
    print(f"  Tolerance: {validation['tolerance']:.2e}")
    print(f"  Total row sum: {validation['total_row_sum']:.2f}")
    print(f"  Total col sum: {validation['total_col_sum']:.2f}")

    if validation["unbalanced_accounts"]:
        print(f"\n  Unbalanced accounts:")
        for acc, diff in validation["unbalanced_accounts"].items():
            print(f"    {acc}: {diff:.2e}")
    else:
        print(f"\n  ✓ All accounts are balanced!")

    # SAM Summary
    print("\n" + "-" * 70)
    print("Step 3: SAM Summary")
    print("-" * 70)

    summary = sam.summary()
    print(f"\nSAM Summary:")
    print(f"  Name: {summary['name']}")
    print(f"  Shape: {summary['shape']}")
    print(f"  Total accounts: {summary['accounts']}")
    print(f"  Total value: {summary['total_value']:.2f}")
    print(f"  Is balanced: {summary['is_balanced']}")

    # Extract submatrix
    print("\n" + "-" * 70)
    print("Step 4: Extracting Submatrices")
    print("-" * 70)

    # Extract intermediate demand matrix (sectors × sectors)
    sectors = ["agr", "mfg", "svc"]
    intermediate = sam.get_submatrix(sectors, sectors)
    print(f"\nIntermediate Demand Matrix (sectors × sectors):")
    print(intermediate.to_string())

    # Extract value added (factors × sectors)
    factors = ["labor", "capital"]
    value_added = sam.get_submatrix(factors, sectors)
    print(f"\nValue Added (factors × sectors):")
    print(value_added.to_string())

    # Extract final demand (sectors × institutions)
    institutions = ["hh", "gov", "row"]
    final_demand = sam.get_submatrix(sectors, institutions)
    print(f"\nFinal Demand (sectors × institutions):")
    print(final_demand.to_string())

    # Extract sets
    print("\n" + "-" * 70)
    print("Step 5: Extracting Sets")
    print("-" * 70)

    # Define set extraction mapping
    set_mapping = {
        "J": ["agr", "mfg", "svc"],  # Sectors
        "I": ["labor", "capital"],  # Factors
        "H": ["hh"],  # Households
        "G": ["gov"],  # Government
        "ROW": ["row"],  # Rest of world
    }

    sam.extract_sets(set_mapping)

    print(f"\nExtracted sets from SAM:")
    for set_name, elements in sam.sets.items():
        print(f"  {set_name}: {elements}")

    # Create an unbalanced SAM and balance it
    print("\n" + "-" * 70)
    print("Step 6: Balancing an Unbalanced SAM")
    print("-" * 70)

    # Create slightly unbalanced SAM
    unbalanced_data = data.copy()
    unbalanced_data[0, 1] += 5  # Add imbalance
    unbalanced_df = pd.DataFrame(unbalanced_data, index=accounts, columns=accounts)
    unbalanced_sam = SAM.from_dataframe(unbalanced_df, name="UnbalancedSAM")

    print(f"\nCreated unbalanced SAM")
    validation_unbal = unbalanced_sam.check_balance(tolerance=1e-6)
    print(f"  Is balanced: {validation_unbal['is_balanced']}")
    print(f"  Max difference: {validation_unbal['max_difference']:.2f}")

    # Balance using RAS
    print(f"\nBalancing using RAS method...")
    balanced_sam = unbalanced_sam.balance(method="ras")

    validation_bal = balanced_sam.check_balance(tolerance=1e-6)
    print(f"  ✓ Balanced SAM created")
    print(f"  Is balanced: {validation_bal['is_balanced']}")
    print(f"  Max difference: {validation_bal['max_difference']:.2e}")

    # Export example (commented out - would create actual files)
    print("\n" + "-" * 70)
    print("Step 7: Export Options")
    print("-" * 70)

    print(f"\nSAM can be exported to:")
    print(f"  - Excel: sam.to_excel('sam.xlsx')")
    print(f"  - Dictionary: sam.to_dict()")
    print(f"  - DataFrame: sam.data")

    # Show dictionary export
    sam_dict = sam.to_dict()
    print(f"\nDictionary export keys: {list(sam_dict.keys())}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
