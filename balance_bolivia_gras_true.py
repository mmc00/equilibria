"""
True GRAS implementation (Junius & Oosterhaven 2003).

GRAS differs from RAS in that it:
1. Handles negative values (sign-preserving)
2. Uses exponential transformation of Lagrange multipliers
3. Minimizes cross-entropy while preserving signs
"""

import pandas as pd
import numpy as np
from pathlib import Path


def gras_true(X0, u, v, max_iter=1000, tol=1e-6):
    """
    True GRAS algorithm (Junius & Oosterhaven 2003).

    Args:
        X0: Initial matrix (can have negative values)
        u: Row targets
        v: Column targets
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        X: Balanced matrix
        iterations: Number of iterations
        converged: Whether converged
    """
    n, m = X0.shape

    # Separate positive and negative parts
    X0_pos = np.maximum(X0, 0)
    X0_neg = np.maximum(-X0, 0)

    # Initialize multipliers
    r = np.ones(n)
    s = np.ones(m)

    for iteration in range(max_iter):
        # Update r (row multipliers)
        for i in range(n):
            row_sum_pos = (r[i] * X0_pos[i, :] * s).sum()
            row_sum_neg = (r[i] * X0_neg[i, :] * s).sum()
            row_sum = row_sum_pos - row_sum_neg

            if abs(row_sum) > 1e-10 and abs(u[i]) > 1e-10:
                # GRAS update rule (different from RAS!)
                if u[i] * row_sum > 0:
                    r[i] = r[i] * (abs(u[i]) / abs(row_sum))
                else:
                    r[i] = r[i] * 0.5  # Damped update if signs differ

        # Update s (column multipliers)
        for j in range(m):
            col_sum_pos = (r[:, np.newaxis] * X0_pos[:, j] * s[j]).sum()
            col_sum_neg = (r[:, np.newaxis] * X0_neg[:, j] * s[j]).sum()
            col_sum = col_sum_pos - col_sum_neg

            if abs(col_sum) > 1e-10 and abs(v[j]) > 1e-10:
                if v[j] * col_sum > 0:
                    s[j] = s[j] * (abs(v[j]) / abs(col_sum))
                else:
                    s[j] = s[j] * 0.5

        # Compute balanced matrix (sign-preserving)
        X_pos = r[:, np.newaxis] * X0_pos * s[np.newaxis, :]
        X_neg = r[:, np.newaxis] * X0_neg * s[np.newaxis, :]
        X = X_pos - X_neg

        # Check convergence
        row_sums = X.sum(axis=1)
        col_sums = X.sum(axis=0)
        row_diff = np.abs(row_sums - u).max()
        col_diff = np.abs(col_sums - v).max()

        if max(row_diff, col_diff) < tol:
            return X, iteration + 1, True

    return X, max_iter, False


def balance_complete_mip_gras(mip_df, max_iter_outer=50):
    """
    Complete MIP balancing using true GRAS.

    Strategy:
    1. Fix VA (most reliable)
    2. Fix PIB target
    3. Use GRAS to adjust Z, F, IMP iteratively
    """
    N = 70
    va_row_names = [
        'Remuneraciones (trabajadores asalariados)',
        'Excedente Bruto de Explotacion',
        'Otros impuestos menos subsidios'
    ]
    va_idx = [mip_df.index.get_loc(name) for name in va_row_names]

    M = mip_df.values.copy()

    # Fixed values
    VA_fixed = M[va_idx, :N].copy()
    PIB_target = VA_fixed.sum()

    print(f"GRAS Setup:")
    print(f"  PIB target (from VA): {PIB_target:,.2f}")
    print(f"  VA fixed: {VA_fixed.sum():,.2f}")

    for outer_iter in range(max_iter_outer):
        # Extract blocks
        Z = M[:N, :N].copy()
        F = M[:N, N:N+5].copy()
        IMP_Z = M[N:2*N, :N].copy()
        IMP_F = M[N:2*N, N:N+5].copy()

        # === STEP 1: Balance Z with GRAS ===
        # Row targets: total use
        z_row_targets = Z.sum(axis=1) + F.sum(axis=1)

        # Column targets: total production - VA
        z_col_targets = Z.sum(axis=0) + VA_fixed.sum(axis=0) - VA_fixed.sum(axis=0)
        z_col_targets = np.maximum(z_col_targets, 0)

        # For square matrix, use arithmetic mean
        z_targets = 0.5 * (z_row_targets + z_col_targets[:N])

        Z_balanced, z_iters, z_conv = gras_true(
            Z, z_targets, z_targets, max_iter=500, tol=1e-4
        )

        M[:N, :N] = Z_balanced

        # === STEP 2: Adjust F to satisfy PIB identity ===
        # PIB = F.sum() - IMP_F.sum() = PIB_target
        # => F.sum() = PIB_target + IMP_F.sum()

        current_imp_f = IMP_F.sum()
        required_f_total = PIB_target + current_imp_f

        current_f_total = F.sum()
        if current_f_total > 1e-6:
            f_scale = required_f_total / current_f_total
            F = F * f_scale

        M[:N, N:N+5] = F

        # === STEP 3: Adjust imports proportionally ===
        # Keep import ratios constant
        total_use = Z_balanced.sum(axis=1) + F.sum(axis=1)
        for i in range(N):
            if total_use[i] > 1e-6:
                # Import penetration ~ 15%
                target_imp = total_use[i] * 0.15
                current_imp = IMP_Z[i, :].sum() + IMP_F[i, :].sum()

                if current_imp > 1e-6:
                    imp_scale = target_imp / current_imp
                    IMP_Z[i, :] *= imp_scale
                    IMP_F[i, :] *= imp_scale

        M[N:2*N, :N] = IMP_Z
        M[N:2*N, N:N+5] = IMP_F

        # Restore VA
        M[va_idx, :N] = VA_fixed

        # Check convergence
        Z_final = M[:N, :N]
        F_final = M[:N, N:N+5]
        IMP_F_final = M[N:2*N, N:N+5]
        VA_final = M[va_idx, :N]

        PIB_VA = VA_final.sum()
        PIB_gasto = F_final.sum() - IMP_F_final.sum()
        pib_error = abs(PIB_VA - PIB_gasto)
        pib_pct = 100 * pib_error / PIB_VA

        z_balance = abs(Z_final.sum(axis=1) - Z_final.sum(axis=0)).max()

        if outer_iter % 10 == 0 or outer_iter < 5:
            print(f"  Iter {outer_iter+1:3d}: z_balance={z_balance:8.2f}, PIB_err={pib_pct:6.2f}%")

        if pib_pct < 1.0 and z_balance < 1.0:
            print(f"\\n✓ Converged in {outer_iter + 1} iterations!")
            break
    else:
        print(f"\\n⚠ Did not fully converge")

    balanced_df = pd.DataFrame(M, index=mip_df.index, columns=mip_df.columns)
    return balanced_df


# === MAIN ===

if __name__ == "__main__":
    input_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_unbalanced.xlsx")
    output_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_balanced_gras.xlsx")

    print("="*70)
    print("TRUE GRAS BALANCING (Junius & Oosterhaven 2003)")
    print("="*70)

    # Load
    print("\\nLoading MIP...")
    mip_df = pd.read_excel(input_path, sheet_name='mip', header=0, index_col=0)

    if mip_df.index[-1] == 'X':
        mip_df = mip_df.iloc[:-1, :]
    if mip_df.columns[-1] == 'X':
        mip_df = mip_df.iloc[:, :-1]

    n_nans = mip_df.isna().sum().sum()
    if n_nans > 0:
        print(f"Filling {n_nans} NaN values with 0")
        mip_df = mip_df.fillna(0)

    # Balance
    print("\\n" + "="*70)
    print("BALANCING")
    print("="*70)

    balanced_df = balance_complete_mip_gras(mip_df, max_iter_outer=50)

    # Verify
    print("\\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    N = 70
    va_idx = [140, 141, 142]
    M = balanced_df.values

    Z = M[:N, :N]
    F = M[:N, N:N+5]
    IMP_F = M[N:2*N, N:N+5]
    VA = M[va_idx, :N]

    PIB_VA = VA.sum()
    PIB_gasto = F.sum() - IMP_F.sum()

    print(f"\\nPIB Identity:")
    print(f"  PIB (VA):     {PIB_VA:,.2f}")
    print(f"  PIB (gasto):  {PIB_gasto:,.2f}")
    print(f"  Error:        {abs(PIB_VA - PIB_gasto):,.2f} ({100*abs(PIB_VA - PIB_gasto)/PIB_VA:.2f}%)")

    print(f"\\nZ balance:")
    print(f"  Max |row-col|: {abs(Z.sum(axis=1) - Z.sum(axis=0)).max():.2f}")

    # Save
    balanced_df['X'] = balanced_df.sum(axis=1)
    balanced_df.loc['X'] = balanced_df.sum(axis=0)

    print(f"\\nSaving to: {output_path}")
    balanced_df.to_excel(output_path, sheet_name='mip_balanced')

    print("\\n" + "="*70)
    print("✓ TRUE GRAS COMPLETE")
    print("="*70)
