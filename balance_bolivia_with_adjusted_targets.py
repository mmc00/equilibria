"""
MIP balancing using .npy targets adjusted for consistency.

This approach:
1. Loads the row/column targets from .npy files
2. Adjusts them to be consistent with VA and PIB (most reliable data)
3. Applies GRAS with adjusted targets
"""

import pandas as pd
import numpy as np
from pathlib import Path


def adjust_targets_for_consistency(
    row_targets,
    col_targets,
    va_total,
    pib_target,
    n_products=70,
    n_sectors=70
):
    """
    Adjust targets to be internally consistent.

    Args:
        row_targets: Original row targets (143,)
        col_targets: Original column targets (75,)
        va_total: Fixed VA total (scalar)
        pib_target: Fixed PIB target (scalar)

    Returns:
        Adjusted targets that are consistent
    """
    print("\n" + "="*70)
    print("ADJUSTING TARGETS FOR CONSISTENCY")
    print("="*70)

    # Current target sums
    row_sum = row_targets.sum()
    col_sum = col_targets.sum()

    print(f"\nOriginal targets:")
    print(f"  Row target sum: {row_sum:,.2f}")
    print(f"  Col target sum: {col_sum:,.2f}")
    print(f"  Difference: {abs(row_sum - col_sum):,.2f}")

    # === STEP 1: Adjust VA component of row targets ===
    # VA rows are indices 140, 141, 142 in row_targets
    va_row_indices = [140, 141, 142]

    # Current VA targets
    current_va_targets = row_targets[va_row_indices]
    current_va_sum = current_va_targets.sum()

    print(f"\nVA adjustment:")
    print(f"  Current VA target sum: {current_va_sum:,.2f}")
    print(f"  Required VA (fixed): {va_total:,.2f}")
    print(f"  Adjustment needed: {va_total - current_va_sum:,.2f}")

    # Replace VA targets with fixed values (proportionally)
    if current_va_sum > 1e-6:
        va_proportion = current_va_targets / current_va_sum
        row_targets[va_row_indices] = va_proportion * va_total
    else:
        # If no VA in targets, distribute equally
        row_targets[va_row_indices] = va_total / 3

    # === STEP 2: Calculate implied grand total from PIB ===
    # PIB = Final Demand - Imports to FD
    # We need to figure out what grand total is consistent with this

    # Final demand columns are indices 70-74 in col_targets
    fd_col_indices = list(range(70, 75))
    current_fd_targets = col_targets[fd_col_indices]

    # Import rows for FD are indices 140-144 in row_targets (after products and imports to sectors)
    # Actually, imports are rows 70-139 (70 products)
    # Let's use current import targets
    import_row_indices = list(range(70, 140))
    current_import_targets = row_targets[import_row_indices]

    # Estimate imports to FD as proportion of total imports
    total_imports = current_import_targets.sum()
    # Assume 30% of imports go to FD
    estimated_imp_to_fd = total_imports * 0.3

    # From PIB identity: PIB = FD - IMP_FD
    # => FD = PIB + IMP_FD
    required_fd_total = pib_target + estimated_imp_to_fd

    print(f"\nFinal demand adjustment:")
    print(f"  Current FD target sum: {current_fd_targets.sum():,.2f}")
    print(f"  Required FD (from PIB): {required_fd_total:,.2f}")
    print(f"  Adjustment factor: {required_fd_total / (current_fd_targets.sum() + 1e-10):.4f}")

    # Scale FD targets
    if current_fd_targets.sum() > 1e-6:
        col_targets[fd_col_indices] = current_fd_targets * (required_fd_total / current_fd_targets.sum())

    # === STEP 3: Make row and column totals equal ===
    # After adjusting VA and FD, make row and col sums equal

    new_row_sum = row_targets.sum()
    new_col_sum = col_targets.sum()

    print(f"\nAfter VA and FD adjustment:")
    print(f"  Row sum: {new_row_sum:,.2f}")
    print(f"  Col sum: {new_col_sum:,.2f}")
    print(f"  Difference: {abs(new_row_sum - new_col_sum):,.2f}")

    # Use arithmetic mean as grand total
    grand_total = 0.5 * (new_row_sum + new_col_sum)

    # Scale both to match grand total
    row_scale = grand_total / new_row_sum
    col_scale = grand_total / new_col_sum

    row_targets_adjusted = row_targets * row_scale
    col_targets_adjusted = col_targets * col_scale

    print(f"\nFinal adjustment (to grand total {grand_total:,.2f}):")
    print(f"  Row scale factor: {row_scale:.6f}")
    print(f"  Col scale factor: {col_scale:.6f}")
    print(f"  Final row sum: {row_targets_adjusted.sum():,.2f}")
    print(f"  Final col sum: {col_targets_adjusted.sum():,.2f}")
    print(f"  Difference: {abs(row_targets_adjusted.sum() - col_targets_adjusted.sum()):.6f}")

    # Verify VA is preserved
    final_va_sum = row_targets_adjusted[va_row_indices].sum()
    print(f"\nVA verification:")
    print(f"  Final VA target sum: {final_va_sum:,.2f}")
    print(f"  Required VA: {va_total:,.2f}")
    print(f"  Error: {abs(final_va_sum - va_total):,.2f}")

    return row_targets_adjusted, col_targets_adjusted


def gras_with_targets(matrix, row_targets, col_targets, max_iter=1000, tol=1e-4):
    """GRAS balancing to match row and column targets."""
    M = matrix.copy()

    for iteration in range(max_iter):
        # Row scaling
        row_sums = M.sum(axis=1)
        row_factors = np.where(row_sums > 1e-10, row_targets / row_sums, 1.0)
        M = M * row_factors[:, np.newaxis]

        # Column scaling
        col_sums = M.sum(axis=0)
        col_factors = np.where(col_sums > 1e-10, col_targets / col_sums, 1.0)
        M = M * col_factors[np.newaxis, :]

        # Check convergence
        row_diff = np.abs(M.sum(axis=1) - row_targets).max()
        col_diff = np.abs(M.sum(axis=0) - col_targets).max()

        if max(row_diff, col_diff) < tol:
            return M, iteration + 1, True

    return M, max_iter, False


# === MAIN ===

if __name__ == "__main__":
    input_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_unbalanced.xlsx")
    output_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_balanced_targets.xlsx")
    targets_dir = Path("/Users/marmol/proyectos/cge_babel/playground/bol")

    print("="*70)
    print("MIP BALANCING - ADJUSTED TARGETS METHOD")
    print("="*70)

    # Load MIP
    print("\nLoading MIP...")
    mip_df = pd.read_excel(input_path, sheet_name='mip', header=0, index_col=0)

    if mip_df.index[-1] == 'X':
        mip_df = mip_df.iloc[:-1, :]
    if mip_df.columns[-1] == 'X':
        mip_df = mip_df.iloc[:, :-1]

    n_nans = mip_df.isna().sum().sum()
    if n_nans > 0:
        print(f"Filling {n_nans} NaN values with 0")
        mip_df = mip_df.fillna(0)

    # Load targets
    print("\nLoading targets...")
    row_targets_orig = np.load(targets_dir / "targets_row.npy")
    col_targets_orig = np.load(targets_dir / "targets_col.npy")

    # Get VA and PIB (fixed)
    N = 70
    va_row_names = [
        'Remuneraciones (trabajadores asalariados)',
        'Excedente Bruto de Explotacion',
        'Otros impuestos menos subsidios'
    ]
    va_idx = [mip_df.index.get_loc(name) for name in va_row_names]
    VA_original = mip_df.values[va_idx, :N]
    VA_total = VA_original.sum()
    PIB_target = VA_total  # PIB from VA is most reliable

    print(f"  Fixed values:")
    print(f"    VA total: {VA_total:,.2f}")
    print(f"    PIB target: {PIB_target:,.2f}")

    # Adjust targets
    row_targets, col_targets = adjust_targets_for_consistency(
        row_targets_orig.copy(),
        col_targets_orig.copy(),
        VA_total,
        PIB_target,
        n_products=N,
        n_sectors=N
    )

    # Balance full matrix with adjusted targets
    print("\n" + "="*70)
    print("BALANCING WITH ADJUSTED TARGETS")
    print("="*70)

    M_balanced, iters, converged = gras_with_targets(
        mip_df.values,
        row_targets,
        col_targets,
        max_iter=1000,
        tol=1e-4
    )

    print(f"\nGRAS result:")
    print(f"  Converged: {converged}")
    print(f"  Iterations: {iters}")
    print(f"  Final row diff: {np.abs(M_balanced.sum(axis=1) - row_targets).max():.6f}")
    print(f"  Final col diff: {np.abs(M_balanced.sum(axis=0) - col_targets).max():.6f}")

    # Restore VA exactly
    M_balanced[va_idx, :N] = VA_original

    # Verify
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    Z = M_balanced[:N, :N]
    F = M_balanced[:N, N:N+5]
    IMP_F = M_balanced[N:2*N, N:N+5]
    VA = M_balanced[va_idx, :N]

    PIB_VA = VA.sum()
    PIB_gasto = F.sum() - IMP_F.sum()

    print(f"\nPIB Identity:")
    print(f"  PIB (VA): {PIB_VA:,.2f}")
    print(f"  PIB (gasto): {PIB_gasto:,.2f}")
    print(f"  Difference: {abs(PIB_VA - PIB_gasto):,.2f} ({100*abs(PIB_VA-PIB_gasto)/PIB_VA:.2f}%)")

    print(f"\nVA preserved:")
    print(f"  Target: {VA_total:,.2f}")
    print(f"  Final: {VA.sum():,.2f}")
    print(f"  Error: {abs(VA.sum() - VA_total):.6f}")

    # Save
    balanced_df = pd.DataFrame(M_balanced, index=mip_df.index, columns=mip_df.columns)
    balanced_df['X'] = balanced_df.sum(axis=1)
    balanced_df.loc['X'] = balanced_df.sum(axis=0)

    print(f"\nSaving to: {output_path}")
    balanced_df.to_excel(output_path, sheet_name='mip_balanced')

    print("\n" + "="*70)
    print("✓ BALANCING WITH ADJUSTED TARGETS COMPLETE")
    print("="*70)
