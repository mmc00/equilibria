"""
Triple RAS Balancing - Balances ALL three identities simultaneously.

Balances:
1. Product balance (rows): Supply = Demand for each product
2. Sector balance (columns): Inputs = Outputs for each sector
3. PIB identity: PIB_VA = PIB_gasto

Uses iterative biproportional scaling with three sets of multipliers:
- r_i: Product row multipliers
- s_j: Sector column multipliers
- f_k: Final demand multipliers
"""

import pandas as pd
import numpy as np
from pathlib import Path


def triple_ras_balance(
    Z, F, IMP_Z, IMP_F, VA,
    max_iter=1000,
    tol=1e-3,
    damping=0.5
):
    """
    Triple RAS: Balance product rows, sector columns, and PIB simultaneously.

    Args:
        Z: Intermediate flows (N×N)
        F: Final demand (N×5)
        IMP_Z: Imports to sectors (N×N)
        IMP_F: Imports to final demand (N×5)
        VA: Value added (3×N) - FIXED, not adjusted

    Returns:
        Z_bal, F_bal, IMP_Z_bal, IMP_F_bal, converged
    """
    N = Z.shape[0]

    # Initialize multipliers
    r = np.ones(N)  # Product (row) multipliers
    s = np.ones(N)  # Sector (column) multipliers
    f = np.ones(5)  # Final demand category multipliers

    # VA is fixed
    PIB_target = VA.sum()

    print(f"\nTriple RAS Setup:")
    print(f"  PIB target (from VA): {PIB_target:,.2f}")
    print(f"  Products: {N}")
    print(f"  Sectors: {N}")
    print(f"  FD categories: 5")

    for iteration in range(max_iter):
        r_old = r.copy()
        s_old = s.copy()
        f_old = f.copy()

        # === STEP 1: Balance products (rows) - SUPPLY = DEMAND ===

        # For each product i:
        # Supply_i = Output_i + Imports_i
        # Demand_i = IntermediateUse_i + FinalDemand_i

        # Output of product i (assuming product=industry)
        output = (s * Z).sum(axis=0) + VA.sum(axis=0)

        for i in range(N):
            # Supply of product i
            imports_i = r[i] * IMP_Z[i, :].sum() + r[i] * IMP_F[i, :].sum()
            supply_i = output[i] + imports_i

            # Demand of product i
            intermediate_use_i = (r[i] * Z[i, :] * s).sum()
            final_demand_i = (r[i] * F[i, :] * f).sum()
            demand_i = intermediate_use_i + final_demand_i

            # Update r[i]
            if demand_i > 1e-10:
                r_new = supply_i / demand_i
                r[i] = damping * r_old[i] + (1 - damping) * r_new
                r[i] = np.clip(r[i], 1e-6, 1e6)

        # === STEP 2: Balance sectors (columns) - INPUTS = OUTPUTS ===

        # For each sector j:
        # Inputs_j = IntermediateInputs_j + Imports_j + VA_j
        # Outputs_j = Production_j

        for j in range(N):
            # Inputs to sector j
            intermediate_inputs_j = (r * Z[:, j] * s[j]).sum()
            imports_j = (r * IMP_Z[:, j] * s[j]).sum()
            va_j = VA[:, j].sum()
            inputs_j = intermediate_inputs_j + imports_j + va_j

            # Output of sector j
            output_j = intermediate_inputs_j + imports_j + va_j  # By definition

            # In a balanced MIP: output_j should equal what sector j produces
            # Production goes to intermediate use (column j becomes row j) + final demand
            production_j = (r[j] * Z[j, :] * s).sum() + (r[j] * F[j, :] * f).sum()

            # Update s[j]
            if production_j > 1e-10:
                s_new = output_j / production_j
                s[j] = damping * s_old[j] + (1 - damping) * s_new
                s[j] = np.clip(s[j], 1e-6, 1e6)

        # === STEP 3: Balance PIB - PIB_VA = PIB_gasto ===

        # PIB from VA (fixed)
        PIB_VA = VA.sum()

        # PIB from expenditure
        C = (r * F[:, 0] * f[0]).sum()
        G = (r * F[:, 1] * f[1]).sum()
        I = (r * F[:, 2] * f[2]).sum() + (r * F[:, 3] * f[3]).sum()
        X = (r * F[:, 4] * f[4]).sum()
        M = (r[:, np.newaxis] * IMP_F * f[np.newaxis, :]).sum()

        PIB_gasto = C + G + I + X - M

        # Adjust final demand multipliers proportionally
        if PIB_gasto > 1e-10:
            f_scale = PIB_VA / PIB_gasto
            f = damping * f_old + (1 - damping) * (f_old * f_scale)
            f = np.clip(f, 1e-6, 1e6)

        # === STEP 4: Check convergence ===

        # Apply multipliers
        Z_bal = r[:, np.newaxis] * Z * s[np.newaxis, :]
        F_bal = r[:, np.newaxis] * F * f[np.newaxis, :]
        IMP_Z_bal = r[:, np.newaxis] * IMP_Z * s[np.newaxis, :]
        IMP_F_bal = r[:, np.newaxis] * IMP_F * f[np.newaxis, :]

        # Check product balance
        output_check = (Z_bal.sum(axis=0) + VA.sum(axis=0))
        supply = output_check + IMP_Z_bal.sum(axis=1) + IMP_F_bal.sum(axis=1)
        demand = Z_bal.sum(axis=1) + F_bal.sum(axis=1)
        product_diff = np.abs(supply - demand).max()

        # Check sector balance
        inputs = Z_bal.sum(axis=0) + IMP_Z_bal.sum(axis=0) + VA.sum(axis=0)
        outputs = Z_bal.sum(axis=1) + F_bal.sum(axis=1)
        sector_diff = np.abs(inputs - outputs).max()

        # Check PIB
        PIB_check = F_bal.sum() - IMP_F_bal.sum()
        pib_diff = abs(PIB_VA - PIB_check)
        pib_pct = 100 * pib_diff / PIB_VA

        if iteration % 50 == 0 or iteration < 10:
            print(f"  Iter {iteration:4d}: product_diff={product_diff:8.2f}, "
                  f"sector_diff={sector_diff:8.2f}, pib_err={pib_pct:6.4f}%")

        # Convergence criteria
        if product_diff < tol and sector_diff < tol and pib_pct < 0.5:
            print(f"\n✓ Converged in {iteration + 1} iterations!")
            return Z_bal, F_bal, IMP_Z_bal, IMP_F_bal, True

        # Check for divergence
        if np.any(np.isnan(r)) or np.any(np.isnan(s)) or np.any(np.isnan(f)):
            print(f"\n⚠ NaN detected at iteration {iteration}")
            return Z_bal, F_bal, IMP_Z_bal, IMP_F_bal, False

    print(f"\n⚠ Reached max iterations ({max_iter})")
    return Z_bal, F_bal, IMP_Z_bal, IMP_F_bal, False


# === MAIN ===

if __name__ == "__main__":
    input_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_unbalanced.xlsx")
    output_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_balanced_triple_ras.xlsx")

    print("="*80)
    print("TRIPLE RAS BALANCING - COMPLETE MIP BALANCE")
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

    Z = M_orig[:N, :N].copy()
    F = M_orig[:N, N:N+5].copy()
    IMP_Z = M_orig[N:2*N, :N].copy()
    IMP_F = M_orig[N:2*N, N:N+5].copy()
    VA = M_orig[va_idx, :N].copy()

    print(f"\nOriginal MIP statistics:")
    print(f"  Z total:    {Z.sum():,.2f}")
    print(f"  F total:    {F.sum():,.2f}")
    print(f"  VA total:   {VA.sum():,.2f}")
    print(f"  IMP total:  {(IMP_Z.sum() + IMP_F.sum()):,.2f}")

    # Balance
    print("\n" + "="*80)
    print("BALANCING")
    print("="*80)

    Z_bal, F_bal, IMP_Z_bal, IMP_F_bal, converged = triple_ras_balance(
        Z, F, IMP_Z, IMP_F, VA,
        max_iter=2000,
        tol=1.0,  # Accept 1 unit of difference
        damping=0.3
    )

    # Enforce non-negativity
    print("\nEnforcing non-negativity...")
    F_bal[:, 0] = np.maximum(0, F_bal[:, 0])  # C_hh
    F_bal[:, 1] = np.maximum(0, F_bal[:, 1])  # C_gov
    F_bal[:, 2] = np.maximum(0, F_bal[:, 2])  # FBKF
    # F_bal[:, 3] can be negative (Var.Stock)
    F_bal[:, 4] = np.maximum(0, F_bal[:, 4])  # Exports
    IMP_Z_bal = np.maximum(0, IMP_Z_bal)
    IMP_F_bal = np.maximum(0, IMP_F_bal)

    # Reconstruct MIP
    M_balanced = M_orig.copy()
    M_balanced[:N, :N] = Z_bal
    M_balanced[:N, N:N+5] = F_bal
    M_balanced[N:2*N, :N] = IMP_Z_bal
    M_balanced[N:2*N, N:N+5] = IMP_F_bal
    M_balanced[va_idx, :N] = VA  # VA preserved

    # Verification
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    output = Z_bal.sum(axis=0) + VA.sum(axis=0)
    supply = output + IMP_Z_bal.sum(axis=1) + IMP_F_bal.sum(axis=1)
    demand = Z_bal.sum(axis=1) + F_bal.sum(axis=1)
    product_balance = supply - demand

    inputs = Z_bal.sum(axis=0) + IMP_Z_bal.sum(axis=0) + VA.sum(axis=0)
    outputs = Z_bal.sum(axis=1) + F_bal.sum(axis=1)
    sector_balance = inputs - outputs

    PIB_VA = VA.sum()
    PIB_gasto = F_bal.sum() - IMP_F_bal.sum()

    print(f"\n1. Product balance (Supply = Demand):")
    print(f"   Max |diff|:  {np.abs(product_balance).max():.6f}")
    print(f"   Mean |diff|: {np.abs(product_balance).mean():.6f}")

    print(f"\n2. Sector balance (Inputs = Outputs):")
    print(f"   Max |diff|:  {np.abs(sector_balance).max():.6f}")
    print(f"   Mean |diff|: {np.abs(sector_balance).mean():.6f}")

    print(f"\n3. PIB identity:")
    print(f"   PIB (VA):    {PIB_VA:,.2f}")
    print(f"   PIB (gasto): {PIB_gasto:,.2f}")
    print(f"   Error:       {100*abs(PIB_VA - PIB_gasto)/PIB_VA:.4f}%")

    # Check for NaN and negatives
    if np.any(np.isnan(M_balanced)):
        print(f"\n⚠ WARNING: Contains NaN!")
    else:
        print(f"\n✓ No NaN values")

    n_neg_imports = (IMP_Z_bal < -1e-6).sum() + (IMP_F_bal < -1e-6).sum()
    if n_neg_imports > 0:
        print(f"⚠ WARNING: {n_neg_imports} negative imports")
    else:
        print(f"✓ All imports >= 0")

    # Save
    balanced_df = pd.DataFrame(M_balanced, index=mip_df.index, columns=mip_df.columns)
    balanced_df['X'] = balanced_df.sum(axis=1)
    balanced_df.loc['X'] = balanced_df.sum(axis=0)

    print(f"\nSaving to: {output_path}")
    balanced_df.to_excel(output_path, sheet_name='mip_balanced')

    print("\n" + "="*80)
    print("✓ TRIPLE RAS COMPLETE")
    print("="*80)
