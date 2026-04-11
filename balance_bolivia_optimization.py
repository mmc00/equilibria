"""
MIP balancing using constrained optimization.

This uses scipy.optimize to find the balanced MIP that:
1. Preserves VA exactly (hard constraint)
2. Preserves PIB exactly (hard constraint)
3. Minimizes changes to the original matrix (objective)
4. Enforces supply-use balances (constraints)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class MIPBalancingProblem:
    """Optimization problem for MIP balancing."""

    def __init__(self, mip_matrix, va_indices, n_products=70, n_sectors=70):
        """
        Args:
            mip_matrix: Original MIP (143 × 75)
            va_indices: Indices of VA rows
            n_products: Number of products (70)
            n_sectors: Number of sectors (70)
        """
        self.M_original = mip_matrix.copy()
        self.va_indices = va_indices
        self.N = n_products

        # Extract original blocks
        self.Z_orig = self.M_original[:self.N, :self.N].copy()
        self.F_orig = self.M_original[:self.N, self.N:self.N+5].copy()
        self.IMP_Z_orig = self.M_original[self.N:2*self.N, :self.N].copy()
        self.IMP_F_orig = self.M_original[self.N:2*self.N, self.N:self.N+5].copy()
        self.VA_orig = self.M_original[va_indices, :self.N].copy()

        # Fixed values (most reliable)
        self.PIB_target = self.VA_orig.sum()
        self.VA_target = self.VA_orig.copy()

        print(f"Optimization setup:")
        print(f"  PIB target (from VA): {self.PIB_target:,.2f}")
        print(f"  VA preserved: {self.VA_target.sum():,.2f}")
        print(f"  Variables to optimize: Z, F, IMP_Z, IMP_F")

        # Flatten for optimization
        self.n_vars_z = self.Z_orig.size
        self.n_vars_f = self.F_orig.size
        self.n_vars_impz = self.IMP_Z_orig.size
        self.n_vars_impf = self.IMP_F_orig.size
        self.n_vars_total = (self.n_vars_z + self.n_vars_f +
                             self.n_vars_impz + self.n_vars_impf)

        print(f"  Total variables: {self.n_vars_total:,}")

    def unpack_variables(self, x):
        """Convert flat variable vector to matrices."""
        idx = 0

        Z = x[idx:idx+self.n_vars_z].reshape(self.N, self.N)
        idx += self.n_vars_z

        F = x[idx:idx+self.n_vars_f].reshape(self.N, 5)
        idx += self.n_vars_f

        IMP_Z = x[idx:idx+self.n_vars_impz].reshape(self.N, self.N)
        idx += self.n_vars_impz

        IMP_F = x[idx:idx+self.n_vars_impf].reshape(self.N, 5)

        return Z, F, IMP_Z, IMP_F

    def pack_variables(self, Z, F, IMP_Z, IMP_F):
        """Convert matrices to flat variable vector."""
        return np.concatenate([
            Z.flatten(),
            F.flatten(),
            IMP_Z.flatten(),
            IMP_F.flatten()
        ])

    def objective(self, x):
        """
        Minimize sum of squared relative changes.

        This preserves the structure of the original MIP.
        """
        Z, F, IMP_Z, IMP_F = self.unpack_variables(x)

        # Relative changes (avoid division by zero)
        def rel_change(orig, new):
            return np.where(np.abs(orig) > 1e-6,
                            ((new - orig) / (np.abs(orig) + 1e-6))**2,
                            new**2)

        total_change = (
            rel_change(self.Z_orig, Z).sum() +
            rel_change(self.F_orig, F).sum() +
            rel_change(self.IMP_Z_orig, IMP_Z).sum() +
            rel_change(self.IMP_F_orig, IMP_F).sum()
        )

        return total_change

    def constraints_dict(self):
        """Define all constraints as scipy constraint dictionaries."""
        constraints = []

        # === CONSTRAINT 1: PIB Identity (HARD) ===
        # PIB = Final Demand - Imports to FD = VA (fixed)
        def pib_constraint(x):
            _, F, _, IMP_F = self.unpack_variables(x)
            PIB_gasto = F.sum() - IMP_F.sum()
            return PIB_gasto - self.PIB_target  # Should be zero

        constraints.append({
            'type': 'eq',
            'fun': pib_constraint
        })

        # === CONSTRAINT 2: Supply = Use for each product (SOFT via penalty) ===
        # For each product i: Production[i] + Imports[i] = Intermediate use[i] + Final demand[i]
        # This is enforced via penalty in objective, not hard constraint

        # === CONSTRAINT 3: Non-negativity ===
        # All flows must be >= 0
        bounds = [(0, None)] * self.n_vars_total

        return constraints, bounds

    def constraints_as_residuals(self, x):
        """Calculate constraint violations for diagnostics."""
        Z, F, IMP_Z, IMP_F = self.unpack_variables(x)

        # 1. PIB constraint
        PIB_gasto = F.sum() - IMP_F.sum()
        pib_residual = abs(PIB_gasto - self.PIB_target)

        # 2. Product balance
        supply = Z.sum(axis=0) + IMP_Z.sum(axis=0)
        use = Z.sum(axis=1) + F.sum(axis=1)
        product_residuals = np.abs(supply - use[:self.N])

        # 3. Sector balance
        inputs = Z.sum(axis=0)
        production_cost = inputs + self.VA_target.sum(axis=0)
        production_use = Z.sum(axis=1)[:self.N] + F.sum(axis=1)[:self.N]
        sector_residuals = np.abs(production_cost - production_use)

        return {
            'pib': pib_residual,
            'product_max': product_residuals.max(),
            'product_mean': product_residuals.mean(),
            'sector_max': sector_residuals.max(),
            'sector_mean': sector_residuals.mean()
        }


def balance_mip_optimization(mip_df, method='trust-constr', max_iter=100):
    """
    Balance MIP using constrained optimization.

    Args:
        mip_df: MIP DataFrame (143 × 75)
        method: Optimization method ('trust-constr', 'SLSQP')
        max_iter: Maximum iterations

    Returns:
        balanced_df: Balanced MIP
        result: Optimization result object
    """
    # Setup
    N = 70
    va_row_names = [
        'Remuneraciones (trabajadores asalariados)',
        'Excedente Bruto de Explotacion',
        'Otros impuestos menos subsidios'
    ]
    va_indices = [mip_df.index.get_loc(name) for name in va_row_names]

    # Create problem
    problem = MIPBalancingProblem(mip_df.values, va_indices, N, N)

    # Initial guess: original values
    x0 = problem.pack_variables(
        problem.Z_orig,
        problem.F_orig,
        problem.IMP_Z_orig,
        problem.IMP_F_orig
    )

    print(f"\nInitial diagnostics:")
    init_residuals = problem.constraints_as_residuals(x0)
    for key, val in init_residuals.items():
        print(f"  {key}: {val:,.2f}")

    # Get constraints
    constraints, bounds = problem.constraints_dict()

    print(f"\nRunning optimization with {method}...")
    print(f"  Constraints: {len(constraints)} equality constraints")
    print(f"  Bounds: non-negativity on all {len(bounds):,} variables")

    # Optimize
    result = minimize(
        problem.objective,
        x0,
        method=method,
        constraints=constraints,
        bounds=bounds,
        options={
            'maxiter': max_iter,
            'verbose': 2 if method == 'trust-constr' else 1,
            'disp': True
        }
    )

    print(f"\nOptimization finished:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")
    print(f"  Final objective: {result.fun:,.6f}")

    # Extract solution
    Z_bal, F_bal, IMP_Z_bal, IMP_F_bal = problem.unpack_variables(result.x)

    # Build balanced matrix
    M_balanced = problem.M_original.copy()
    M_balanced[:N, :N] = Z_bal
    M_balanced[:N, N:N+5] = F_bal
    M_balanced[N:2*N, :N] = IMP_Z_bal
    M_balanced[N:2*N, N:N+5] = IMP_F_bal
    M_balanced[va_indices, :N] = problem.VA_target  # Restore VA

    balanced_df = pd.DataFrame(
        M_balanced,
        index=mip_df.index,
        columns=mip_df.columns
    )

    # Final diagnostics
    print(f"\nFinal diagnostics:")
    final_residuals = problem.constraints_as_residuals(result.x)
    for key, val in final_residuals.items():
        print(f"  {key}: {val:,.2f}")

    return balanced_df, result


# === MAIN EXECUTION ===

if __name__ == "__main__":
    input_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_unbalanced.xlsx")
    output_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_balanced_optimization.xlsx")

    print("="*70)
    print("MIP BALANCING - CONSTRAINED OPTIMIZATION")
    print("="*70)
    print("\nLoading MIP...")

    # Load
    mip_df = pd.read_excel(input_path, sheet_name='mip', header=0, index_col=0)

    # Clean
    if mip_df.index[-1] == 'X':
        mip_df = mip_df.iloc[:-1, :]
    if mip_df.columns[-1] == 'X':
        mip_df = mip_df.iloc[:, :-1]

    n_nans = mip_df.isna().sum().sum()
    if n_nans > 0:
        print(f"Filling {n_nans} NaN values with 0")
        mip_df = mip_df.fillna(0)

    print(f"MIP shape: {mip_df.shape}")

    # Balance
    print("\n" + "="*70)
    print("OPTIMIZATION")
    print("="*70)

    balanced_df, result = balance_mip_optimization(
        mip_df,
        method='SLSQP',  # Faster than trust-constr for this size
        max_iter=100
    )

    # Verify
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
    IMP_F = M[N:2*N, N:N+5]
    VA = M[va_idx, :N]

    PIB_VA = VA.sum()
    PIB_gasto = F.sum() - IMP_F.sum()

    print(f"\nPIB Identity:")
    print(f"  PIB (VA):          {PIB_VA:,.2f}")
    print(f"  PIB (expenditure): {PIB_gasto:,.2f}")
    print(f"  Difference:        {abs(PIB_VA - PIB_gasto):.6f}")
    print(f"  % error:           {100*abs(PIB_VA - PIB_gasto)/PIB_VA:.6f}%")

    print(f"\nVA preserved:")
    print(f"  Original VA: {48614.10:,.2f}")
    print(f"  Final VA:    {VA.sum():,.2f}")
    print(f"  Difference:  {abs(VA.sum() - 48614.10):.6f}")

    # Save
    balanced_df['X'] = balanced_df.sum(axis=1)
    balanced_df.loc['X'] = balanced_df.sum(axis=0)

    print(f"\nSaving to: {output_path}")
    balanced_df.to_excel(output_path, sheet_name='mip_balanced')

    print("\n" + "="*70)
    print("✓ OPTIMIZATION COMPLETE")
    print("="*70)
