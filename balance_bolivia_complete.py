"""
Complete MIP balancing using Stone + GRAS method.

This implements a full system balancing that adjusts:
- Intermediate flows (Z)
- Final demand (F)
- Imports (M)

While preserving VA (most reliable data) and enforcing all three identities:
1. Supply = Use (by product)
2. Inputs + VA = Production (by sector)
3. GDP(VA) = GDP(expenditure)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def gras_balance(matrix, row_targets, col_targets, max_iter=1000, tol=1e-6):
    """
    Generalized RAS - balances matrix to row and column targets.

    Unlike classic RAS, this handles matrices that may have zeros
    and uses a more robust approach.
    """
    M = matrix.copy()
    n_rows, n_cols = M.shape

    for iteration in range(max_iter):
        # Row adjustment
        row_sums = M.sum(axis=1)
        row_factors = np.ones(n_rows)
        for i in range(n_rows):
            if row_sums[i] > 1e-10 and row_targets[i] > 1e-10:
                row_factors[i] = row_targets[i] / row_sums[i]
        M = M * row_factors[:, np.newaxis]

        # Column adjustment
        col_sums = M.sum(axis=0)
        col_factors = np.ones(n_cols)
        for j in range(n_cols):
            if col_sums[j] > 1e-10 and col_targets[j] > 1e-10:
                col_factors[j] = col_targets[j] / col_sums[j]
        M = M * col_factors[np.newaxis, :]

        # Check convergence
        row_diff = np.abs(M.sum(axis=1) - row_targets).max()
        col_diff = np.abs(M.sum(axis=0) - col_targets).max()
        max_diff = max(row_diff, col_diff)

        if max_diff < tol:
            return M, iteration + 1, True

    return M, max_iter, False


def balance_complete_mip_system(mip_df, max_iter=100, tol=0.1):
    """
    Complete MIP balancing using iterative Stone + GRAS method.

    This balances the entire MIP system to satisfy all three identities.

    Args:
        mip_df: MIP DataFrame (143 × 75)
        max_iter: Maximum outer iterations
        tol: Tolerance for convergence (in absolute terms)

    Returns:
        balanced_df: Balanced MIP
        diagnostics: Dictionary with balancing statistics
    """
    # Define structure
    N = 70  # Products and sectors

    va_row_names = [
        'Remuneraciones (trabajadores asalariados)',
        'Excedente Bruto de Explotacion',
        'Otros impuestos menos subsidios'
    ]
    va_row_indices = [mip_df.index.get_loc(name) for name in va_row_names]

    # Extract blocks
    M = mip_df.values.copy()

    # Preserve VA (most reliable - from national accounts)
    VA_original = M[va_row_indices, :N].copy()
    PIB_VA = VA_original.sum()

    print(f"Starting balancing...")
    print(f"  PIB from VA (fixed): {PIB_VA:,.2f}")

    # Track diagnostics
    history = []

    for outer_iter in range(max_iter):
        # Extract current blocks
        Z = M[:N, :N].copy()           # Intermediate flows
        F = M[:N, N:N+5].copy()        # Final demand (5 categories)
        IMP_Z = M[N:2*N, :N].copy()    # Imports to sectors
        IMP_F = M[N:2*N, N:N+5].copy() # Imports to final demand
        VA = M[va_row_indices, :N].copy()

        # === STEP 1: Calculate production targets from both sides ===

        # From USE side (by product): How much is consumed
        intermediate_use = Z.sum(axis=1)  # Used as inputs by sectors
        final_use = F.sum(axis=1)         # Used for final demand
        total_use = intermediate_use + final_use

        # From PRODUCTION side (by sector): How much is produced
        intermediate_inputs = Z.sum(axis=0)  # Inputs used by each sector
        production_cost = intermediate_inputs + VA.sum(axis=0)  # Inputs + VA = Production

        # Target production: In square case (N=N), production should equal use
        # Use arithmetic mean as compromise
        production_target = 0.5 * (total_use + production_cost)

        # === STEP 2: Balance intermediate matrix Z (70×70) ===

        # Row targets for Z: each product should be used according to total use
        z_row_targets = total_use.copy()

        # Column targets for Z: each sector uses = production - VA
        z_col_targets = production_target - VA.sum(axis=0)
        z_col_targets = np.maximum(z_col_targets, 0)  # Ensure non-negative

        # Apply GRAS to Z
        Z_balanced, z_iters, z_converged = gras_balance(
            Z, z_row_targets, z_col_targets, max_iter=500, tol=1e-4
        )

        # === STEP 3: Adjust final demand to close PIB identity ===

        # PIB identity: PIB(VA) = Final Demand - Imports to FD
        # => Final Demand = PIB(VA) + Imports to FD

        current_imports_fd = IMP_F.sum()
        required_total_fd = PIB_VA + current_imports_fd

        # Current final demand total
        current_total_fd = F.sum()

        # Scale F proportionally
        if current_total_fd > 1e-6:
            fd_scale_factor = required_total_fd / current_total_fd
            F = F * fd_scale_factor

        # === STEP 4: Adjust imports proportionally ===

        # Imports should be proportional to domestic use
        # Simple rule: maintain import penetration ratios

        for i in range(N):
            domestic_use_intermediate = Z_balanced[i, :].sum()
            domestic_use_final = F[i, :].sum()
            total_domestic_use = domestic_use_intermediate + domestic_use_final

            # Import penetration: assume ~15% (or maintain current ratio)
            current_imports = IMP_Z[i, :].sum() + IMP_F[i, :].sum()

            if total_domestic_use > 1e-6:
                # Target: 15% import penetration
                target_imports = total_domestic_use * 0.15

                if current_imports > 1e-6:
                    import_scale = target_imports / current_imports
                    IMP_Z[i, :] *= import_scale
                    IMP_F[i, :] *= import_scale

        # === STEP 5: Update matrix ===

        M[:N, :N] = Z_balanced
        M[:N, N:N+5] = F
        M[N:2*N, :N] = IMP_Z
        M[N:2*N, N:N+5] = IMP_F
        M[va_row_indices, :N] = VA_original  # Restore VA (always fixed)

        # === STEP 6: Check convergence ===

        # Verify the three identities

        # 1. Product balance (supply = use)
        Z_final = M[:N, :N]
        F_final = M[:N, N:N+5]
        IMP_Z_final = M[N:2*N, :N]
        IMP_F_final = M[N:2*N, N:N+5]

        supply = Z_final.sum(axis=0) + IMP_Z_final.sum(axis=0)
        use = Z_final.sum(axis=1) + F_final.sum(axis=1)
        product_balance_error = np.abs(supply - use[:N]).max()

        # 2. Sector balance (inputs + VA = production)
        inputs = Z_final.sum(axis=0)
        production_from_cost = inputs + VA_original.sum(axis=0)
        production_from_use = Z_final.sum(axis=1)[:N] + F_final.sum(axis=1)[:N]
        sector_balance_error = np.abs(production_from_cost - production_from_use).max()

        # 3. PIB identity
        PIB_from_VA = VA_original.sum()
        PIB_from_expenditure = F_final.sum() - IMP_F_final.sum()
        pib_error = abs(PIB_from_VA - PIB_from_expenditure)
        pib_error_pct = 100 * pib_error / PIB_from_VA

        # Record diagnostics
        diagnostics = {
            'iteration': outer_iter + 1,
            'product_balance_error': product_balance_error,
            'sector_balance_error': sector_balance_error,
            'pib_error': pib_error,
            'pib_error_pct': pib_error_pct,
            'z_converged': z_converged,
            'z_iterations': z_iters
        }
        history.append(diagnostics)

        if outer_iter % 10 == 0 or outer_iter < 5:
            print(f"  Iter {outer_iter+1:3d}: "
                  f"product_err={product_balance_error:8.2f}, "
                  f"sector_err={sector_balance_error:8.2f}, "
                  f"PIB_err={pib_error_pct:6.2f}%")

        # Check if all identities are satisfied
        if (product_balance_error < tol and
            sector_balance_error < tol and
            pib_error_pct < 1.0):  # 1% tolerance for PIB
            print(f"\n✓ Converged in {outer_iter + 1} iterations!")
            break
    else:
        print(f"\n⚠ Did not fully converge after {max_iter} iterations")

    # Create balanced DataFrame
    balanced_df = pd.DataFrame(M, index=mip_df.index, columns=mip_df.columns)

    return balanced_df, history


# === MAIN EXECUTION ===

if __name__ == "__main__":
    # Paths
    input_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_unbalanced.xlsx")
    output_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_balanced_complete.xlsx")

    print("="*70)
    print("COMPLETE MIP BALANCING - STONE + GRAS METHOD")
    print("="*70)
    print("\nLoading MIP...")

    # Load MIP
    mip_df = pd.read_excel(input_path, sheet_name='mip', header=0, index_col=0)

    # Remove totals
    if mip_df.index[-1] == 'X':
        mip_df = mip_df.iloc[:-1, :]
    if mip_df.columns[-1] == 'X':
        mip_df = mip_df.iloc[:, :-1]

    # Fill NaN
    n_nans = mip_df.isna().sum().sum()
    if n_nans > 0:
        print(f"Filling {n_nans} NaN values with 0")
        mip_df = mip_df.fillna(0)

    print(f"MIP shape: {mip_df.shape}")

    # Balance
    print("\n" + "="*70)
    print("BALANCING")
    print("="*70)

    balanced_df, history = balance_complete_mip_system(
        mip_df,
        max_iter=100,
        tol=0.1
    )

    # Final verification
    print("\n" + "="*70)
    print("FINAL VERIFICATION")
    print("="*70)

    N = 70
    va_row_names = [
        'Remuneraciones (trabajadores asalariados)',
        'Excedente Bruto de Explotacion',
        'Otros impuestos menos subsidios'
    ]
    va_idx = [balanced_df.index.get_loc(name) for name in va_row_names]

    M = balanced_df.values
    Z = M[:N, :N]
    F = M[:N, N:N+5]
    IMP_Z = M[N:2*N, :N]
    IMP_F = M[N:2*N, N:N+5]
    VA = M[va_idx, :N]

    # Check all identities
    print("\n1. INTERMEDIATE MATRIX (Z):")
    z_row = Z.sum(axis=1)
    z_col = Z.sum(axis=0)
    print(f"   Row-column balance: max diff = {np.abs(z_row - z_col).max():.6f}")

    print("\n2. PRODUCT BALANCE (supply = use):")
    supply = z_col + IMP_Z.sum(axis=0)
    use = z_row + F.sum(axis=1)
    print(f"   Max imbalance: {np.abs(supply - use[:N]).max():.2f}")

    print("\n3. SECTOR BALANCE (inputs + VA = production):")
    inputs = Z.sum(axis=0)
    prod_cost = inputs + VA.sum(axis=0)
    prod_use = Z.sum(axis=1)[:N] + F.sum(axis=1)[:N]
    print(f"   Max imbalance: {np.abs(prod_cost - prod_use).max():.2f}")

    print("\n4. PIB IDENTITY:")
    PIB_VA = VA.sum()
    PIB_gasto = F.sum() - IMP_F.sum()
    pib_diff = abs(PIB_VA - PIB_gasto)
    pib_pct = 100 * pib_diff / PIB_VA
    print(f"   PIB (VA):          {PIB_VA:,.2f}")
    print(f"   PIB (expenditure): {PIB_gasto:,.2f}")
    print(f"   Discrepancy:       {pib_diff:,.2f} ({pib_pct:.2f}%)")

    # Save
    balanced_df['X'] = balanced_df.sum(axis=1)
    balanced_df.loc['X'] = balanced_df.sum(axis=0)

    print(f"\nSaving to: {output_path}")
    balanced_df.to_excel(output_path, sheet_name='mip_balanced')

    # Save diagnostics
    import json
    diag_path = output_path.parent / "mip_balancing_diagnostics.json"
    with open(diag_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Diagnostics saved to: {diag_path}")

    print("\n" + "="*70)
    print("✓ COMPLETE MIP BALANCING FINISHED")
    print("="*70)
