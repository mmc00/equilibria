"""
Complete MIP Balancing via Constrained Optimization.

Uses scipy.optimize to balance ALL three identities simultaneously:
1. Product balance (rows): Supply = Demand for each product
2. Sector balance (columns): Inputs = Outputs for each sector
3. PIB identity: PIB_VA = PIB_gasto

Optimization problem:
- Variables: Z, F, IMP_Z, IMP_F (VA is fixed)
- Objective: Minimize weighted sum of squared changes from original
- Constraints: All three balance equations
- Bounds: Non-negativity (except Var.Stock can be negative)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize


def build_full_balance_optimization(Z0, F0, IMP_Z0, IMP_F0, VA):
    """
    Build optimization problem for complete MIP balancing.

    Returns objective function, constraints, and bounds.
    """
    N = Z0.shape[0]

    # Flatten matrices to vectors for optimization
    n_z = N * N
    n_f = N * 5
    n_imp_z = N * N
    n_imp_f = N * 5

    total_vars = n_z + n_f + n_imp_z + n_imp_f

    # Pack original matrices
    x0 = np.concatenate([
        Z0.flatten(),
        F0.flatten(),
        IMP_Z0.flatten(),
        IMP_F0.flatten()
    ])

    # Indices for unpacking
    idx_z_end = n_z
    idx_f_end = idx_z_end + n_f
    idx_imp_z_end = idx_f_end + n_imp_z
    idx_imp_f_end = idx_imp_z_end + n_imp_f

    def unpack(x):
        """Unpack optimization vector to matrices."""
        Z = x[:idx_z_end].reshape(N, N)
        F = x[idx_z_end:idx_f_end].reshape(N, 5)
        IMP_Z = x[idx_f_end:idx_imp_z_end].reshape(N, N)
        IMP_F = x[idx_imp_z_end:idx_imp_f_end].reshape(N, 5)
        return Z, F, IMP_Z, IMP_F

    # === OBJECTIVE: Minimize weighted changes ===

    def objective(x):
        """Minimize sum of squared relative changes, weighted by importance."""
        Z, F, IMP_Z, IMP_F = unpack(x)

        # Weights: penalize changes to Z more than F (Z is more reliable)
        w_z = 10.0
        w_f = 1.0
        w_imp = 5.0

        # Squared relative changes (avoid division by zero)
        rel_change_z = np.sum(((Z - Z0) / (np.abs(Z0) + 1))**2)
        rel_change_f = np.sum(((F - F0) / (np.abs(F0) + 1))**2)
        rel_change_imp_z = np.sum(((IMP_Z - IMP_Z0) / (np.abs(IMP_Z0) + 1))**2)
        rel_change_imp_f = np.sum(((IMP_F - IMP_F0) / (np.abs(IMP_F0) + 1))**2)

        return (w_z * rel_change_z +
                w_f * rel_change_f +
                w_imp * rel_change_imp_z +
                w_imp * rel_change_imp_f)

    # === CONSTRAINTS ===

    constraints = []

    # 1. Product balance (70 constraints): Supply_i = Demand_i
    for i in range(N):
        def product_balance_i(x, i=i):
            Z, F, IMP_Z, IMP_F = unpack(x)

            # Supply of product i
            # Output (assuming product=industry): inputs_i + VA_i
            output_i = Z[:, i].sum() + VA[:, i].sum()
            imports_i = IMP_Z[i, :].sum() + IMP_F[i, :].sum()
            supply_i = output_i + imports_i

            # Demand of product i
            intermediate_use_i = Z[i, :].sum()
            final_demand_i = F[i, :].sum()
            demand_i = intermediate_use_i + final_demand_i

            return supply_i - demand_i

        constraints.append({
            'type': 'eq',
            'fun': product_balance_i
        })

    # 2. Sector balance (70 constraints): Inputs_j = Outputs_j
    for j in range(N):
        def sector_balance_j(x, j=j):
            Z, F, IMP_Z, IMP_F = unpack(x)

            # Inputs to sector j
            intermediate_inputs_j = Z[:, j].sum()
            imports_j = IMP_Z[:, j].sum()
            va_j = VA[:, j].sum()
            inputs_j = intermediate_inputs_j + imports_j + va_j

            # Outputs of sector j (assuming product=industry)
            # = what sector j produces = Z[j,:] + F[j,:]
            outputs_j = Z[j, :].sum() + F[j, :].sum()

            return inputs_j - outputs_j

        constraints.append({
            'type': 'eq',
            'fun': sector_balance_j
        })

    # 3. PIB identity (1 constraint): PIB_VA = PIB_gasto
    def pib_identity(x):
        Z, F, IMP_Z, IMP_F = unpack(x)

        PIB_VA = VA.sum()
        PIB_gasto = F.sum() - IMP_F.sum()

        return PIB_VA - PIB_gasto

    constraints.append({
        'type': 'eq',
        'fun': pib_identity
    })

    # === BOUNDS: Non-negativity ===

    bounds = []

    # Z: all non-negative
    bounds.extend([(0, None)] * n_z)

    # F: columns 0,1,2,4 non-negative; column 3 (Var.Stock) can be negative
    for i in range(N):
        for k in range(5):
            if k == 3:  # Var.Stock
                bounds.append((None, None))
            else:
                bounds.append((0, None))

    # IMP_Z, IMP_F: all non-negative
    bounds.extend([(0, None)] * n_imp_z)
    bounds.extend([(0, None)] * n_imp_f)

    return objective, constraints, bounds, x0, unpack


# === MAIN ===

if __name__ == "__main__":
    input_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_unbalanced.xlsx")
    output_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_balanced_full_opt.xlsx")

    print("="*80)
    print("COMPLETE MIP BALANCING - CONSTRAINED OPTIMIZATION")
    print("="*80)

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

    print(f"MIP shape: {mip_df.shape}")

    # Extract blocks
    N = 70
    va_row_names = [
        'Remuneraciones (trabajadores asalariados)',
        'Excedente Bruto de Explotacion',
        'Otros impuestos menos subsidios'
    ]
    va_idx = [mip_df.index.get_loc(name) for name in va_row_names]

    M_orig = mip_df.values.copy()

    Z0 = M_orig[:N, :N].copy()
    F0 = M_orig[:N, N:N+5].copy()
    IMP_Z0 = M_orig[N:2*N, :N].copy()
    IMP_F0 = M_orig[N:2*N, N:N+5].copy()
    VA = M_orig[va_idx, :N].copy()

    print(f"\nOriginal MIP statistics:")
    print(f"  Z total:    {Z0.sum():,.2f}")
    print(f"  F total:    {F0.sum():,.2f}")
    print(f"  VA total:   {VA.sum():,.2f}")
    print(f"  PIB (VA):   {VA.sum():,.2f}")

    # Build optimization problem
    print("\n" + "="*80)
    print("BUILDING OPTIMIZATION PROBLEM")
    print("="*80)

    objective, constraints, bounds, x0, unpack = build_full_balance_optimization(
        Z0, F0, IMP_Z0, IMP_F0, VA
    )

    print(f"\nProblem size:")
    print(f"  Variables:   {len(x0):,}")
    print(f"  Constraints: {len(constraints)}")
    print(f"    - Product balance:  70")
    print(f"    - Sector balance:   70")
    print(f"    - PIB identity:     1")

    # Run optimization
    print("\n" + "="*80)
    print("RUNNING OPTIMIZATION")
    print("="*80)
    print("\nUsing SLSQP (Sequential Least Squares Programming)...")
    print("This may take several minutes...\n")

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={
            'maxiter': 500,
            'ftol': 1e-6,
            'disp': True
        }
    )

    print(f"\nOptimization finished:")
    print(f"  Success:     {result.success}")
    print(f"  Message:     {result.message}")
    print(f"  Iterations:  {result.nit}")
    print(f"  Objective:   {result.fun:.6f}")

    # Unpack solution
    Z_bal, F_bal, IMP_Z_bal, IMP_F_bal = unpack(result.x)

    # Reconstruct MIP
    M_balanced = M_orig.copy()
    M_balanced[:N, :N] = Z_bal
    M_balanced[:N, N:N+5] = F_bal
    M_balanced[N:2*N, :N] = IMP_Z_bal
    M_balanced[N:2*N, N:N+5] = IMP_F_bal
    M_balanced[va_idx, :N] = VA

    # Verification
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # 1. Product balance
    output = Z_bal.sum(axis=0) + VA.sum(axis=0)
    supply = output + IMP_Z_bal.sum(axis=1) + IMP_F_bal.sum(axis=1)
    demand = Z_bal.sum(axis=1) + F_bal.sum(axis=1)
    product_balance = supply - demand

    print(f"\n1. Product balance (Supply = Demand):")
    print(f"   Max |diff|:  {np.abs(product_balance).max():.6f}")
    print(f"   Mean |diff|: {np.abs(product_balance).mean():.6f}")

    # 2. Sector balance
    inputs = Z_bal.sum(axis=0) + IMP_Z_bal.sum(axis=0) + VA.sum(axis=0)
    outputs = Z_bal.sum(axis=1) + F_bal.sum(axis=1)
    sector_balance = inputs - outputs

    print(f"\n2. Sector balance (Inputs = Outputs):")
    print(f"   Max |diff|:  {np.abs(sector_balance).max():.6f}")
    print(f"   Mean |diff|: {np.abs(sector_balance).mean():.6f}")

    # 3. PIB identity
    PIB_VA = VA.sum()
    PIB_gasto = F_bal.sum() - IMP_F_bal.sum()

    print(f"\n3. PIB identity:")
    print(f"   PIB (VA):    {PIB_VA:,.2f}")
    print(f"   PIB (gasto): {PIB_gasto:,.2f}")
    print(f"   Error:       {100*abs(PIB_VA - PIB_gasto)/PIB_VA:.6f}%")

    # Check for NaN
    if np.any(np.isnan(M_balanced)):
        print(f"\n⚠ WARNING: Contains NaN!")
    else:
        print(f"\n✓ No NaN values")

    # Save
    balanced_df = pd.DataFrame(M_balanced, index=mip_df.index, columns=mip_df.columns)
    balanced_df['X'] = balanced_df.sum(axis=1)
    balanced_df.loc['X'] = balanced_df.sum(axis=0)

    print(f"\nSaving to: {output_path}")
    balanced_df.to_excel(output_path, sheet_name='mip_balanced')

    print("\n" + "="*80)
    print("✓ OPTIMIZATION COMPLETE")
    print("="*80)
