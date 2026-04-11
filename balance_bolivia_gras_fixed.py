"""
GRAS Fixed - with damping and consistent targets.

Fixes for overflow:
1. Use consistent targets (row_sum = col_sum)
2. Apply damping to prevent oscillation
3. Handle sparse matrices better
"""

import pandas as pd
import numpy as np
from pathlib import Path


def gras_with_damping(X0, u, v, max_iter=1000, tol=1e-4, damping=0.5):
    """
    GRAS with damping to prevent overflow.

    Args:
        X0: Initial matrix
        u: Row targets
        v: Column targets
        damping: Damping factor (0.5 = half old, half new)

    Returns:
        X: Balanced matrix
        iterations: Number of iterations
        converged: Whether converged
    """
    n, m = X0.shape

    # Initialize multipliers
    r = np.ones(n)
    s = np.ones(m)

    # Separate positive and negative
    X0_pos = np.maximum(X0, 0)
    X0_neg = np.maximum(-X0, 0)

    for iteration in range(max_iter):
        r_old = r.copy()
        s_old = s.copy()

        # Row update
        for i in range(n):
            row_sum_pos = (r[i] * X0_pos[i, :] * s).sum()
            row_sum_neg = (r[i] * X0_neg[i, :] * s).sum()
            row_sum = row_sum_pos - row_sum_neg

            if abs(row_sum) > 1e-10 and abs(u[i]) > 1e-10:
                r_new = abs(u[i]) / abs(row_sum)
                # Apply damping
                r[i] = damping * r_old[i] + (1 - damping) * (r_old[i] * r_new)

                # Clamp to prevent overflow
                r[i] = np.clip(r[i], 1e-6, 1e6)

        # Column update
        for j in range(m):
            col_sum_pos = (r * X0_pos[:, j] * s[j]).sum()
            col_sum_neg = (r * X0_neg[:, j] * s[j]).sum()
            col_sum = col_sum_pos - col_sum_neg

            if abs(col_sum) > 1e-10 and abs(v[j]) > 1e-10:
                s_new = abs(v[j]) / abs(col_sum)
                # Apply damping
                s[j] = damping * s_old[j] + (1 - damping) * (s_old[j] * s_new)

                # Clamp
                s[j] = np.clip(s[j], 1e-6, 1e6)

        # Compute balanced matrix
        X_pos = r[:, np.newaxis] * X0_pos * s[np.newaxis, :]
        X_neg = r[:, np.newaxis] * X0_neg * s[np.newaxis, :]
        X = X_pos - X_neg

        # Check convergence
        row_sums = X.sum(axis=1)
        col_sums = X.sum(axis=0)
        row_diff = np.abs(row_sums - u).max()
        col_diff = np.abs(col_sums - v).max()

        if iteration % 100 == 0:
            print(f"  GRAS iter {iteration}: row_diff={row_diff:.4f}, col_diff={col_diff:.4f}, "
                  f"max(r)={r.max():.2e}, max(s)={s.max():.2e}")

        if max(row_diff, col_diff) < tol:
            return X, iteration + 1, True

        # Check for divergence
        if np.any(np.isnan(r)) or np.any(np.isnan(s)):
            print(f"  ⚠ NaN detected at iteration {iteration}")
            return X, iteration, False

    return X, max_iter, False


def balance_mip_gras_fixed(mip_df, max_iter_outer=50):
    """
    Complete MIP balancing using fixed GRAS.

    Key fix: Use CONSISTENT targets (row_sum = col_sum)
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

    print(f"GRAS Fixed Setup:")
    print(f"  PIB target: {PIB_target:,.2f}")
    print(f"  Using damping and consistent targets")

    for outer_iter in range(max_iter_outer):
        # Extract blocks
        Z = M[:N, :N].copy()
        F = M[:N, N:N+5].copy()
        IMP_Z = M[N:2*N, :N].copy()
        IMP_F = M[N:2*N, N:N+5].copy()

        # === STEP 1: Balance Z with GRAS ===
        # Calculate CONSISTENT targets (arithmetic mean)
        z_row_sums = Z.sum(axis=1)
        z_col_sums = Z.sum(axis=0)

        # Use arithmetic mean as target
        z_targets = 0.5 * (z_row_sums + z_col_sums)

        print(f"\nOuter iteration {outer_iter + 1}:")
        print(f"  Z row sum: {z_row_sums.sum():,.2f}")
        print(f"  Z col sum: {z_col_sums.sum():,.2f}")
        print(f"  Target: {z_targets.sum():,.2f}")

        Z_balanced, z_iters, z_conv = gras_with_damping(
            Z, z_targets, z_targets,
            max_iter=500, tol=1e-3, damping=0.3  # Strong damping
        )

        if not z_conv:
            print(f"  ⚠ GRAS did not converge for Z")
        else:
            print(f"  ✓ GRAS converged in {z_iters} iterations")

        M[:N, :N] = Z_balanced

        # === STEP 2: Adjust F for PIB ===
        current_imp_f = IMP_F.sum()
        required_f_total = PIB_target + current_imp_f

        current_f_total = F.sum()
        if current_f_total > 1e-6:
            f_scale = required_f_total / current_f_total
            F = F * f_scale

        # Enforce non-negativity for components that cannot be negative
        # Columns: [C_hh, C_gov, FBKF, Var.Stock, Exports]
        # Only Var.Stock (col 3) can be negative
        F[:, 0] = np.maximum(0, F[:, 0])  # C_hh >= 0
        F[:, 1] = np.maximum(0, F[:, 1])  # C_gov >= 0
        F[:, 2] = np.maximum(0, F[:, 2])  # FBKF >= 0
        # F[:, 3] can be negative (Variación de Stock)
        F[:, 4] = np.maximum(0, F[:, 4])  # Exports >= 0

        M[:N, N:N+5] = F

        # === STEP 3: Adjust imports proportionally ===
        total_use = Z_balanced.sum(axis=1) + F.sum(axis=1)
        for i in range(N):
            if total_use[i] > 1e-6:
                target_imp = total_use[i] * 0.12  # 12% import penetration
                current_imp = IMP_Z[i, :].sum() + IMP_F[i, :].sum()

                if current_imp > 1e-6:
                    imp_scale = target_imp / current_imp
                    # Clamp scaling
                    imp_scale = np.clip(imp_scale, 0.5, 2.0)
                    IMP_Z[i, :] *= imp_scale
                    IMP_F[i, :] *= imp_scale

                    # Enforce non-negativity (imports cannot be negative)
                    IMP_Z[i, :] = np.maximum(0, IMP_Z[i, :])
                    IMP_F[i, :] = np.maximum(0, IMP_F[i, :])

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

        print(f"  Results: z_balance={z_balance:.2f}, PIB_err={pib_pct:.2f}%")

        if pib_pct < 0.5 and z_balance < 1.0:
            print(f"\n✓ Converged in {outer_iter + 1} iterations!")
            break
    else:
        print(f"\n⚠ Reached max iterations")

    balanced_df = pd.DataFrame(M, index=mip_df.index, columns=mip_df.columns)
    return balanced_df


# === MAIN ===

if __name__ == "__main__":
    input_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_unbalanced.xlsx")
    output_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_balanced_gras_fixed.xlsx")

    print("="*70)
    print("GRAS FIXED - WITH DAMPING AND CONSISTENT TARGETS")
    print("="*70)

    # Load
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

    # Balance
    print("\n" + "="*70)
    print("BALANCING")
    print("="*70)

    balanced_df = balance_mip_gras_fixed(mip_df, max_iter_outer=30)

    # Verify
    print("\n" + "="*70)
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

    print(f"\nPIB Identity:")
    print(f"  PIB (VA):     {PIB_VA:,.2f}")
    print(f"  PIB (gasto):  {PIB_gasto:,.2f}")
    print(f"  Error:        {abs(PIB_VA - PIB_gasto):,.2f} ({100*abs(PIB_VA - PIB_gasto)/PIB_VA:.4f}%)")

    print(f"\nZ balance:")
    z_row = Z.sum(axis=1)
    z_col = Z.sum(axis=0)
    print(f"  Max |row-col|: {abs(z_row - z_col).max():.6f}")

    # Check for NaN
    if np.any(np.isnan(M)):
        print(f"  ⚠ WARNING: Matrix contains NaN values!")
    else:
        print(f"  ✓ No NaN values")

    # Save
    balanced_df['X'] = balanced_df.sum(axis=1)
    balanced_df.loc['X'] = balanced_df.sum(axis=0)

    print(f"\nSaving to: {output_path}")
    balanced_df.to_excel(output_path, sheet_name='mip_balanced')

    print("\n" + "="*70)
    print("✓ GRAS FIXED COMPLETE")
    print("="*70)
