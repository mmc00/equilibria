"""Balancing primitives for SAM matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from equilibria.sam_tools.enums import RASMode


class RASBalanceResult(BaseModel):
    """Result payload for one RAS balancing run."""

    matrix: pd.DataFrame
    target_totals: np.ndarray
    max_diff_before: float
    max_diff_after: float
    iterations: int
    converged: bool
    ras_type: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RASBalancer(BaseModel):
    """RAS balancer with interchangeable target modes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def resolve_ras_type(self, ras_type: str | None) -> RASMode:
        """Return the normalized ``RASMode`` for a user-provided mode string."""
        return RASMode.from_alias(ras_type)

    def _build_targets(
        self,
        row_totals: np.ndarray,
        col_totals: np.ndarray,
        ras_type: RASMode,
    ) -> np.ndarray:
        """Build per-account target totals according to the selected RAS mode."""
        if ras_type == RASMode.ARITHMETIC:
            return 0.5 * (row_totals + col_totals)
        if ras_type == RASMode.ROW:
            return row_totals.copy()
        if ras_type == RASMode.COLUMN:
            return col_totals.copy()

        raw = np.sqrt(np.maximum(row_totals, 0.0) * np.maximum(col_totals, 0.0))
        raw_sum = float(raw.sum())
        total = float(row_totals.sum())
        if raw_sum <= 0.0 or total <= 0.0:
            return raw
        return raw * (total / raw_sum)

    def balance_dataframe(
        self,
        matrix_df: pd.DataFrame,
        *,
        ras_type: str = RASMode.ARITHMETIC.value,
        max_iterations: int = 200,
        tolerance: float = 1e-9,
    ) -> RASBalanceResult:
        """Run iterative RAS scaling and return the balanced matrix plus diagnostics."""
        sam = matrix_df.to_numpy(copy=True, dtype=float)
        if sam.ndim != 2 or sam.shape[0] != sam.shape[1]:
            raise ValueError("RAS balancing requires a square matrix")

        row_totals = sam.sum(axis=1)
        col_totals = sam.sum(axis=0)
        mode = self.resolve_ras_type(ras_type)
        target = self._build_targets(row_totals, col_totals, mode)

        max_diff_before = float(np.max(np.abs(row_totals - col_totals))) if sam.size else 0.0
        max_diff_after = max_diff_before
        converged = max_diff_before <= tolerance
        iterations = 0

        if not converged:
            for step in range(1, max_iterations + 1):
                current_rows = sam.sum(axis=1)
                for i in range(sam.shape[0]):
                    if target[i] <= 0.0:
                        sam[i, :] = 0.0
                    elif current_rows[i] > 0.0:
                        sam[i, :] *= target[i] / current_rows[i]

                current_cols = sam.sum(axis=0)
                for j in range(sam.shape[1]):
                    if target[j] <= 0.0:
                        sam[:, j] = 0.0
                    elif current_cols[j] > 0.0:
                        sam[:, j] *= target[j] / current_cols[j]

                max_diff_after = float(np.max(np.abs(sam.sum(axis=1) - sam.sum(axis=0))))
                iterations = step
                if max_diff_after <= tolerance:
                    converged = True
                    break

        return RASBalanceResult(
            matrix=pd.DataFrame(sam, index=matrix_df.index, columns=matrix_df.columns),
            target_totals=target,
            max_diff_before=max_diff_before,
            max_diff_after=max_diff_after,
            iterations=iterations,
            converged=converged,
            ras_type=mode.value,
        )


class MIPBalanceResult(BaseModel):
    """Result payload for complete MIP balancing."""

    matrix: pd.DataFrame
    row_balance_max_diff: float
    col_balance_max_diff: float
    pib_production: float
    pib_expenditure: float
    pib_diff: float
    iterations: int
    converged: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)


def balance_complete_mip(
    mip_df: pd.DataFrame,
    *,
    n_products: int,
    n_sectors: int,
    va_row_indices: list[int],
    import_row_indices: list[int],
    fd_col_indices: list[int],
    fix_va: bool = True,
    max_iterations: int = 1000,
    tolerance: float = 1e-4,
) -> MIPBalanceResult:
    """
    Balance a complete MIP using GRAS method.

    This function balances the entire MIP system to satisfy all three identities:
    1. Row balance (supply = demand for each product)
    2. Column balance (inputs = production for each sector)
    3. PIB identity (production = expenditure)

    Args:
        mip_df: Full MIP DataFrame with products, imports, VA in rows and sectors, FD in columns
        n_products: Number of product rows (first n rows)
        n_sectors: Number of sector columns (first n columns)
        va_row_indices: Indices of VA rows (e.g., [140, 141, 142] for Remuneraciones, Excedente, Impuestos)
        import_row_indices: Indices of import rows (e.g., [70:140])
        fd_col_indices: Indices of final demand columns (e.g., [70:75] for HH, GOV, INV, Stock, EXP)
        fix_va: If True, preserve VA values (most reliable data)
        max_iterations: Maximum GRAS iterations
        tolerance: Convergence tolerance

    Returns:
        MIPBalanceResult with balanced matrix and diagnostics

    References:
        - Junius & Oosterhaven (2003): "The Solution of Updating or Regionalizing a Matrix"
        - Robinson, Cattaneo & El-Said (2001): "Updating and Estimating a SAM Using Cross Entropy"
    """
    # Work with numpy for efficiency
    M = mip_df.to_numpy(copy=True, dtype=float)

    # Get VA values (fix these if requested)
    va_original = M[va_row_indices, :n_sectors].copy() if fix_va else None

    # Identify matrix blocks
    Z = M[:n_products, :n_sectors]  # Intermediate flows
    F = M[:n_products, fd_col_indices]  # Final demand
    IMP_Z = M[import_row_indices, :n_sectors]  # Imports to sectors
    IMP_F = M[import_row_indices, fd_col_indices]  # Imports to final demand
    VA = M[va_row_indices, :n_sectors]  # Value added by sector

    for iteration in range(max_iterations):
        # Step 1: Calculate production totals from expenditure side
        # X = Σ(intermediate use) + Σ(final demand)
        X_expenditure = Z.sum(axis=0) + F.sum(axis=1)

        # Step 2: Calculate production totals from cost side
        # X = Σ(intermediate inputs) + VA
        X_cost = Z.sum(axis=1) + VA.sum(axis=0)

        # Step 3: Target production = average of both sides
        X_target = 0.5 * (X_expenditure + X_cost)

        # Step 4: Calculate total supply by product
        # Q = domestic production + imports
        Q = X_target.copy()  # This is domestic production
        total_imports_by_product = IMP_Z.sum(axis=1) + IMP_F.sum(axis=1)

        # Step 5: Balance intermediate flows (columns first)
        # Adjust Z columns to match (X_target - VA)
        col_target = X_target - VA.sum(axis=0)
        col_sums = Z.sum(axis=0)
        col_factors = np.where(col_sums > 0, col_target / col_sums, 1.0)
        Z = Z * col_factors[np.newaxis, :]

        # Step 6: Balance rows (products)
        # Total demand = intermediate use + final demand + exports
        row_target = Q + total_imports_by_product  # Total supply
        row_sums_z = Z.sum(axis=1)
        row_sums_f = F.sum(axis=1)
        row_sums_total = row_sums_z + row_sums_f

        row_factors = np.where(row_sums_total > 0, row_target / row_sums_total, 1.0)

        # Apply row factors to both Z and F
        Z = Z * row_factors[:, np.newaxis]
        F = F * row_factors[:, np.newaxis]

        # Also adjust imports proportionally to maintain import shares
        imp_row_sums_z = IMP_Z.sum(axis=1)
        imp_row_sums_f = IMP_F.sum(axis=1)
        imp_row_sums_total = imp_row_sums_z + imp_row_sums_f

        IMP_Z = IMP_Z * row_factors[:, np.newaxis]
        IMP_F = IMP_F * row_factors[:, np.newaxis]

        # Step 7: Restore VA if fixed
        if fix_va and va_original is not None:
            VA = va_original.copy()

        # Check convergence
        X_new_expenditure = Z.sum(axis=0) + F.sum(axis=1)
        X_new_cost = Z.sum(axis=1) + VA.sum(axis=0)

        max_col_diff = float(np.abs(X_new_cost - X_new_expenditure).max())
        max_row_diff = float(np.abs(row_sums_total - row_target).max())

        if max(max_col_diff, max_row_diff) < tolerance:
            # Write back to matrix
            M[:n_products, :n_sectors] = Z
            M[:n_products, fd_col_indices] = F
            M[import_row_indices, :n_sectors] = IMP_Z
            M[import_row_indices, fd_col_indices] = IMP_F
            M[va_row_indices, :n_sectors] = VA

            # Calculate PIB
            pib_production = float(VA.sum())
            total_final_demand = F.sum()
            total_exports = F[:, -1].sum() if F.shape[1] >= 1 else 0.0  # Last FD column is exports
            total_imports = total_imports_by_product.sum()
            pib_expenditure = float(total_final_demand - total_imports)

            return MIPBalanceResult(
                matrix=pd.DataFrame(M, index=mip_df.index, columns=mip_df.columns),
                row_balance_max_diff=max_row_diff,
                col_balance_max_diff=max_col_diff,
                pib_production=pib_production,
                pib_expenditure=pib_expenditure,
                pib_diff=abs(pib_production - pib_expenditure),
                iterations=iteration + 1,
                converged=True,
            )

    # Did not converge
    M[:n_products, :n_sectors] = Z
    M[:n_products, fd_col_indices] = F
    M[import_row_indices, :n_sectors] = IMP_Z
    M[import_row_indices, fd_col_indices] = IMP_F
    M[va_row_indices, :n_sectors] = VA

    pib_production = float(VA.sum())
    total_final_demand = F.sum()
    total_imports = (IMP_Z.sum() + IMP_F.sum())
    pib_expenditure = float(total_final_demand - total_imports)

    return MIPBalanceResult(
        matrix=pd.DataFrame(M, index=mip_df.index, columns=mip_df.columns),
        row_balance_max_diff=max_row_diff,
        col_balance_max_diff=max_col_diff,
        pib_production=pib_production,
        pib_expenditure=pib_expenditure,
        pib_diff=abs(pib_production - pib_expenditure),
        iterations=max_iterations,
        converged=False,
    )
