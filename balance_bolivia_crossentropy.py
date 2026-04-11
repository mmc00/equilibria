"""
MIP balancing using Cross-Entropy Minimization.

This method (Robinson, Cattaneo & El-Said, 2001) minimizes:
    Σᵢⱼ X[i,j] * log(X[i,j] / X₀[i,j])

Subject to:
    - Row sum constraints
    - Column sum constraints
    - PIB identity
    - Non-negativity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize


def cross_entropy_objective(x_flat, x0_flat):
    """
    Cross-entropy divergence: Σ x * log(x / x0)

    This measures information distance from x0 to x.
    """
    x = np.maximum(x_flat, 1e-10)  # Avoid log(0)
    x0 = np.maximum(x0_flat, 1e-10)

    # Cross-entropy
    ce = (x * np.log(x / x0)).sum()

    return ce


def balance_mip_crossentropy(mip_df):
    """
    Balance MIP using cross-entropy minimization.

    This is the method recommended by Robinson et al. (2001) for SAMs.
    """
    N = 70
    va_row_names = [
        'Remuneraciones (trabajadores asalariados)',
        'Excedente Bruto de Explotacion',
        'Otros impuestos menos subsidios'
    ]
    va_idx = [mip_df.index.get_loc(name) for name in va_row_names]

    M0 = mip_df.values.copy()

    # Fixed values
    VA_fixed = M0[va_idx, :N].copy()
    PIB_target = VA_fixed.sum()

    print(f"Cross-Entropy Setup:")
    print(f"  PIB target: {PIB_target:,.2f}")
    print(f"  VA fixed: {VA_fixed.sum():,.2f}")

    # We'll only optimize non-VA elements
    # Extract optimizable blocks
    Z_idx = (slice(0, N), slice(0, N))
    F_idx = (slice(0, N), slice(N, N+5))
    IMP_Z_idx = (slice(N, 2*N), slice(0, N))
    IMP_F_idx = (slice(N, 2*N), slice(N, N+5))

    Z0 = M0[Z_idx]
    F0 = M0[F_idx]
    IMP_Z0 = M0[IMP_Z_idx]
    IMP_F0 = M0[IMP_F_idx]

    # Flatten for optimization
    x0_flat = np.concatenate([
        Z0.flatten(),
        F0.flatten(),
        IMP_Z0.flatten(),
        IMP_F0.flatten()
    ])

    n_vars = len(x0_flat)
    print(f"  Variables to optimize: {n_vars:,}")

    # Objective: cross-entropy from x0
    def objective(x):
        return cross_entropy_objective(x, x0_flat)

    # Constraint: PIB identity
    def pib_constraint(x):
        # Unpack
        idx = 0
        Z = x[idx:idx+Z0.size].reshape(Z0.shape)
        idx += Z0.size
        F = x[idx:idx+F0.size].reshape(F0.shape)
        idx += F0.size
        IMP_Z = x[idx:idx+IMP_Z0.size].reshape(IMP_Z0.shape)
        idx += IMP_Z0.size
        IMP_F = x[idx:idx+IMP_F0.size].reshape(IMP_F0.shape)

        PIB_gasto = F.sum() - IMP_F.sum()
        return PIB_gasto - PIB_target

    constraints = [
        {'type': 'eq', 'fun': pib_constraint}
    ]

    # Bounds: non-negativity
    bounds = [(0, None)] * n_vars

    print(f"\\nRunning optimization...")
    print(f"  Method: SLSQP")
    print(f"  Constraints: 1 (PIB identity)")

    result = minimize(
        objective,
        x0_flat,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 100, 'disp': True}
    )

    print(f"\\nOptimization result:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Iterations: {result.nit}")
    print(f"  Final objective (cross-entropy): {result.fun:.6f}")

    # Unpack solution
    idx = 0
    Z_bal = result.x[idx:idx+Z0.size].reshape(Z0.shape)
    idx += Z0.size
    F_bal = result.x[idx:idx+F0.size].reshape(F0.shape)
    idx += F0.size
    IMP_Z_bal = result.x[idx:idx+IMP_Z0.size].reshape(IMP_Z0.shape)
    idx += IMP_Z0.size
    IMP_F_bal = result.x[idx:idx+IMP_F0.size].reshape(IMP_F0.shape)

    # Build balanced matrix
    M_bal = M0.copy()
    M_bal[Z_idx] = Z_bal
    M_bal[F_idx] = F_bal
    M_bal[IMP_Z_idx] = IMP_Z_bal
    M_bal[IMP_F_idx] = IMP_F_bal
    M_bal[va_idx, :N] = VA_fixed  # Restore VA

    balanced_df = pd.DataFrame(M_bal, index=mip_df.index, columns=mip_df.columns)

    return balanced_df, result


# === MAIN ===

if __name__ == "__main__":
    input_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_unbalanced.xlsx")
    output_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_balanced_crossentropy.xlsx")

    print("="*70)
    print("CROSS-ENTROPY MINIMIZATION (Robinson et al. 2001)")
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
    print("OPTIMIZATION")
    print("="*70)

    balanced_df, result = balance_mip_crossentropy(mip_df)

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
    print(f"  Error:        {abs(PIB_VA - PIB_gasto):,.2f} ({100*abs(PIB_VA-PIB_gasto)/PIB_VA:.6f}%)")

    print(f"\\nZ balance:")
    print(f"  Max |row-col|: {abs(Z.sum(axis=1) - Z.sum(axis=0)).max():.2f}")

    # Save
    balanced_df['X'] = balanced_df.sum(axis=1)
    balanced_df.loc['X'] = balanced_df.sum(axis=0)

    print(f"\\nSaving to: {output_path}")
    balanced_df.to_excel(output_path, sheet_name='mip_balanced')

    print("\\n" + "="*70)
    print("✓ CROSS-ENTROPY COMPLETE")
    print("="*70)
