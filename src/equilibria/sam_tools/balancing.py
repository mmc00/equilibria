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

    # Convert indices to numpy arrays for proper slicing
    va_idx = np.array(va_row_indices)
    imp_idx = np.array(import_row_indices)
    fd_idx = np.array(fd_col_indices)

    # Get VA values (fix these if requested)
    va_original = M[np.ix_(va_idx, range(n_sectors))].copy() if fix_va else None

    # Identify matrix blocks using proper numpy indexing
    Z = M[:n_products, :n_sectors].copy()  # Intermediate flows
    F = M[:n_products, fd_idx].copy()  # Final demand
    IMP_Z = M[np.ix_(imp_idx, range(n_sectors))].copy()  # Imports to sectors
    IMP_F = M[np.ix_(imp_idx, fd_idx)].copy()  # Imports to final demand
    VA = M[np.ix_(va_idx, range(n_sectors))].copy()  # Value added by sector

    for iteration in range(max_iterations):
        # GRAS balancing algorithm
        # MIP structure: rows = products, columns = sectors

        # Step 1: Calculate sector production from USE side (columns)
        # For each sector j: Production[j] = Σᵢ(Z[i,j]) + Σ(F[i,k] where k uses sector j output)
        # Simplification: Use row sums of Z as intermediate use by sector
        X_use = Z.sum(axis=1)  # How much each product is used intermediately
        FD_use = F.sum(axis=1)  # How much each product goes to final demand
        # Total use of each product (row sums)
        total_product_use = X_use + FD_use

        # Step 2: Calculate sector production from COST side (columns)
        # For each sector j: Production[j] = Σᵢ(Z[i,j]) + VA[j]
        X_cost = Z.sum(axis=0) + VA.sum(axis=0)  # Column sums of Z + VA

        # Step 3: For supply-use balance, we need to match dimensions properly
        # Total supply of each product = domestic production + imports
        total_imports = IMP_Z.sum(axis=1) + IMP_F.sum(axis=1)

        # For each product: supply = domestic + imports = use
        # Domestic supply is approximated from sector production (diagonal assumption)
        # In a full MIP, we need make/use tables. Here we use simpler assumption:
        # Domestic supply ≈ column sums (what sectors produce)

        # Target: balance column sums (production) with row sums (use)
        # Take arithmetic mean as target
        # But columns are sectors (n_sectors) and rows are products (n_products)
        # If square (products = sectors), can balance directly

        if n_products == n_sectors:
            # Square case: can balance row sums = column sums
            row_use_with_imports = total_product_use + total_imports
            target = 0.5 * (X_cost + row_use_with_imports)

            # Balance columns (production side)
            col_sums = Z.sum(axis=0) + VA.sum(axis=0)
            col_target = target - VA.sum(axis=0)  # Target for intermediate inputs
            col_factors = np.where(col_sums > tolerance, col_target / col_sums, 1.0)
            Z = Z * col_factors[np.newaxis, :]

            # Balance rows (use side) - adjust Z and F together
            row_sums = Z.sum(axis=1) + F.sum(axis=1)
            row_target = target - total_imports
            row_factors = np.where(row_sums > tolerance, row_target / row_sums, 1.0)

            Z = Z * row_factors[:, np.newaxis]
            F = F * row_factors[:, np.newaxis]

        else:
            # Non-square case: balance what we can
            # Balance intermediate matrix Z to be internally consistent
            row_sums_z = Z.sum(axis=1)[:n_sectors]  # First n_sectors products only
            col_sums_z = Z.sum(axis=0)
            target_z = 0.5 * (row_sums_z + col_sums_z)

            # Row adjustment for Z
            row_factors = np.where(row_sums_z > tolerance, target_z / row_sums_z, 1.0)
            Z[:n_sectors, :] = Z[:n_sectors, :] * row_factors[:, np.newaxis]

            # Column adjustment for Z
            col_factors = np.where(col_sums_z > tolerance, target_z / col_sums_z, 1.0)
            Z = Z * col_factors[np.newaxis, :]

        # Restore VA if fixed
        if fix_va and va_original is not None:
            VA = va_original.copy()

        # Check convergence
        row_sums = Z.sum(axis=1) + F.sum(axis=1)
        col_sums = Z.sum(axis=0) + VA.sum(axis=0)

        if n_products == n_sectors:
            max_diff = float(np.abs(row_sums[:n_sectors] + total_imports[:n_sectors] - col_sums).max())
            max_col_diff = max_diff
            max_row_diff = max_diff
        else:
            max_row_diff = float(np.abs(Z.sum(axis=1)[:n_sectors] - target_z).max()) if iteration > 0 else 1e10
            max_col_diff = float(np.abs(Z.sum(axis=0) - target_z).max()) if iteration > 0 else 1e10

        if max(max_col_diff, max_row_diff) < tolerance:
            # Write back to matrix
            M[:n_products, :n_sectors] = Z
            M[np.ix_(range(n_products), fd_idx)] = F
            M[np.ix_(imp_idx, range(n_sectors))] = IMP_Z
            M[np.ix_(imp_idx, fd_idx)] = IMP_F
            M[np.ix_(va_idx, range(n_sectors))] = VA

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
    M[np.ix_(range(n_products), fd_idx)] = F
    M[np.ix_(imp_idx, range(n_sectors))] = IMP_Z
    M[np.ix_(imp_idx, fd_idx)] = IMP_F
    M[np.ix_(va_idx, range(n_sectors))] = VA

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
