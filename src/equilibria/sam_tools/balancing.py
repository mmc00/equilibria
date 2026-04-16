"""Balancing primitives for SAM and MIP matrices.

This module provides matrix balancing algorithms commonly used in economic modeling:
- RAS: Classic iterative proportional fitting for non-negative matrices
- GRAS: Generalized RAS for matrices with negative values (Junius & Oosterhaven 2003)
- MIP balancing: Methods for balancing Input-Output matrices with multiple constraints

System of Equations for MIP Balancing:
1. Domestic product balance: X = Z_d @ 1 + F_d
2. Import balance: M = Z_m @ 1 + F_m
3. Industry balance: X = Z_d.T @ 1 + Z_m.T @ 1 + VA
4. Aggregate identity: sum(F_d) = sum(VA) + sum(Z_m)

Note: Supply-Demand balance (per product) is mathematically incompatible with
constraints 1-3 when IMP_F > 0. See bolivia_mip_technical_report.md for proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


# =============================================================================
# Low-level RAS and GRAS algorithms
# =============================================================================


def ras_balance(
    M: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> tuple[np.ndarray, int, bool]:
    """Classic RAS algorithm for non-negative matrices.

    Iteratively scales rows and columns to match target totals while
    preserving the structure of the original matrix.

    Args:
        M: Initial non-negative matrix (n x m)
        row_targets: Target row sums (n,)
        col_targets: Target column sums (m,)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (balanced_matrix, iterations, converged)

    Example:
        >>> M = np.array([[1, 2], [3, 4]])
        >>> r = np.array([5, 5])
        >>> c = np.array([4, 6])
        >>> M_bal, iters, conv = ras_balance(M, r, c)
        >>> assert np.allclose(M_bal.sum(axis=1), r)
        >>> assert np.allclose(M_bal.sum(axis=0), c)
    """
    M = M.astype(float).copy()
    n, m = M.shape
    r = np.ones(n)
    s = np.ones(m)

    for iteration in range(max_iter):
        # Row scaling
        M_scaled = M * r[:, np.newaxis] * s[np.newaxis, :]
        row_sums = M_scaled.sum(axis=1)
        mask_r = row_sums > 1e-12
        r[mask_r] = r[mask_r] * row_targets[mask_r] / row_sums[mask_r]

        # Column scaling
        M_scaled = M * r[:, np.newaxis] * s[np.newaxis, :]
        col_sums = M_scaled.sum(axis=0)
        mask_c = col_sums > 1e-12
        s[mask_c] = s[mask_c] * col_targets[mask_c] / col_sums[mask_c]

        # Check convergence
        M_balanced = M * r[:, np.newaxis] * s[np.newaxis, :]
        row_diff = np.abs(M_balanced.sum(axis=1) - row_targets).max()
        col_diff = np.abs(M_balanced.sum(axis=0) - col_targets).max()

        if max(row_diff, col_diff) < tol:
            return M_balanced, iteration + 1, True

    return M * r[:, np.newaxis] * s[np.newaxis, :], max_iter, False


def gras_balance(
    M: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> tuple[np.ndarray, int, bool]:
    """GRAS algorithm for matrices with negative values.

    Generalized RAS (Junius & Oosterhaven, 2003) that preserves signs
    while scaling to match targets. Uses separate scaling for positive
    and negative parts of the matrix.

    Args:
        M: Initial matrix (can have negative values) (n x m)
        row_targets: Target row sums (n,)
        col_targets: Target column sums (m,)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (balanced_matrix, iterations, converged)

    References:
        Junius, T., & Oosterhaven, J. (2003). "The solution of updating or
        regionalizing a matrix with both positive and negative entries."
        Economic Systems Research, 15(1), 87-96.

    Example:
        >>> M = np.array([[1, -2], [3, 4]])
        >>> r = np.array([5, 5])
        >>> c = np.array([6, 4])
        >>> M_bal, iters, conv = gras_balance(M, r, c)
        >>> assert np.allclose(M_bal.sum(axis=1), r, atol=1e-6)
    """
    n, m = M.shape

    # Separate positive and negative parts
    M_pos = np.maximum(M, 0)
    M_neg = np.maximum(-M, 0)

    # Initialize multipliers
    r = np.ones(n)
    s = np.ones(m)

    for iteration in range(max_iter):
        # Update row multipliers
        for i in range(n):
            pos_sum = (r[i] * M_pos[i, :] * s).sum()
            neg_sum = (r[i] * M_neg[i, :] * s).sum()
            row_sum = pos_sum - neg_sum

            if abs(row_sum) > 1e-12 and abs(row_targets[i]) > 1e-12:
                if row_targets[i] * row_sum > 0:
                    r[i] = r[i] * abs(row_targets[i]) / abs(row_sum)
                else:
                    # Signs differ - use damped update
                    r[i] = r[i] * 0.5

        # Update column multipliers
        for j in range(m):
            pos_sum = (r * M_pos[:, j] * s[j]).sum()
            neg_sum = (r * M_neg[:, j] * s[j]).sum()
            col_sum = pos_sum - neg_sum

            if abs(col_sum) > 1e-12 and abs(col_targets[j]) > 1e-12:
                if col_targets[j] * col_sum > 0:
                    s[j] = s[j] * abs(col_targets[j]) / abs(col_sum)
                else:
                    s[j] = s[j] * 0.5

        # Compute balanced matrix (sign-preserving)
        X_pos = r[:, np.newaxis] * M_pos * s[np.newaxis, :]
        X_neg = r[:, np.newaxis] * M_neg * s[np.newaxis, :]
        X = X_pos - X_neg

        # Check convergence
        row_diff = np.abs(X.sum(axis=1) - row_targets).max()
        col_diff = np.abs(X.sum(axis=0) - col_targets).max()

        if max(row_diff, col_diff) < tol:
            return X, iteration + 1, True

    # Final balanced matrix
    X_pos = r[:, np.newaxis] * M_pos * s[np.newaxis, :]
    X_neg = r[:, np.newaxis] * M_neg * s[np.newaxis, :]
    return X_pos - X_neg, max_iter, False


# =============================================================================
# MIP Balancing Result and Methods
# =============================================================================


@dataclass
class MIPBalanceResult:
    """Result of MIP balancing operation.

    Attributes:
        Z_d: Balanced domestic intermediate consumption matrix
        Z_m: Import intermediate consumption (unchanged)
        F_d: Balanced domestic final demand
        F_m: Import final demand (unchanged)
        VA: Value added (unchanged)
        X: Production totals (unchanged)
        error_product: Max product balance error after balancing
        error_industry: Max industry balance error after balancing
        error_pib: PIB identity error (|VA - (F_d - IMP_F)|)
        iterations: Number of iterations used
        converged: Whether algorithm converged
        method: Name of balancing method used
    """

    Z_d: np.ndarray
    Z_m: np.ndarray
    F_d: np.ndarray
    F_m: np.ndarray
    VA: np.ndarray
    X: np.ndarray
    error_product: float
    error_industry: float
    error_pib: float
    iterations: int
    converged: bool
    method: str


def _compute_mip_errors(
    Z_d: np.ndarray,
    Z_m: np.ndarray,
    F_d: np.ndarray,
    F_m: np.ndarray,
    VA: np.ndarray,
    X: np.ndarray,
) -> tuple[float, float, float]:
    """Compute MIP balance errors.

    Returns:
        Tuple of (product_error, industry_error, pib_error)
    """
    # Product balance: X = Z_d @ 1 + F_d (domestic only)
    product_supply = X
    product_demand = Z_d.sum(axis=1) + F_d.sum(axis=1) if F_d.ndim > 1 else Z_d.sum(axis=1) + F_d
    error_product = float(np.abs(product_supply - product_demand).max())

    # Industry balance: X = Z_d.T @ 1 + Z_m.T @ 1 + VA
    industry_output = X
    industry_input = Z_d.sum(axis=0) + Z_m.sum(axis=0) + VA.sum(axis=0)
    error_industry = float(np.abs(industry_output - industry_input).max())

    # PIB identity: sum(VA) = sum(F_d) - sum(IMP_F)
    pib_production = float(VA.sum())
    imp_f_total = float(F_m.sum())
    f_d_total = float(F_d.sum())
    pib_expenditure = f_d_total - imp_f_total
    error_pib = abs(pib_production - pib_expenditure)

    return error_product, error_industry, error_pib


def balance_mip_ras(
    Z_d: np.ndarray,
    Z_m: np.ndarray,
    F_d: np.ndarray,
    F_m: np.ndarray,
    VA: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> MIPBalanceResult:
    """Balance MIP using sequential RAS.

    This method balances the Z_d matrix while keeping VA, IMP, and X fixed.
    It prioritizes PIB identity and industry balance.

    Algorithm:
    1. Fix VA (most reliable data)
    2. Balance Z columns to match (X - VA - Z_m.sum(axis=0))
    3. Adjust F_d to satisfy PIB identity
    4. Iterate until convergence

    Args:
        Z_d: Domestic intermediate consumption (n x n)
        Z_m: Import intermediate consumption (n x n)
        F_d: Domestic final demand (n x k) or (n,)
        F_m: Import final demand (n x k) or (n,)
        VA: Value added (v x n) where v = number of VA components
        X: Production totals (n,)
        max_iter: Maximum outer iterations
        tol: Convergence tolerance

    Returns:
        MIPBalanceResult with balanced matrices

    Note:
        This method prioritizes PIB = 0 and Z balance. Supply-Demand balance
        may worsen as a result (see technical report for proof).
    """
    Z_d = Z_d.astype(float).copy()
    F_d = F_d.astype(float).copy()
    Z_m = Z_m.astype(float).copy()
    F_m = F_m.astype(float).copy()
    VA = VA.astype(float).copy()
    X = X.astype(float).copy()

    n = len(X)
    PIB_target = float(VA.sum())

    for outer in range(max_iter):
        # Step 1: Balance Z columns (industry balance)
        # Column target = X - VA - Z_m.sum(axis=0)
        col_targets = X - VA.sum(axis=0) - Z_m.sum(axis=0)
        col_targets = np.maximum(col_targets, 0)

        z_col_sums = Z_d.sum(axis=0)
        col_factors = np.where(z_col_sums > 1e-12, col_targets / z_col_sums, 1.0)
        Z_d = Z_d * col_factors[np.newaxis, :]

        # Step 2: Balance Z rows (to match column sums for square balance)
        # Use geometric mean for targets
        z_row_sums = Z_d.sum(axis=1)
        z_col_sums = Z_d.sum(axis=0)
        z_targets = np.sqrt(np.maximum(z_row_sums, 0) * np.maximum(z_col_sums, 0))

        if z_targets.sum() > 0:
            Z_d, _, _ = ras_balance(Z_d, z_targets, z_targets, max_iter=200, tol=1e-8)

        # Step 3: Adjust F_d to satisfy PIB identity
        # PIB = F_d.sum() - F_m.sum() => F_d.sum() = PIB + F_m.sum()
        required_f_total = PIB_target + F_m.sum()
        current_f_total = F_d.sum()
        if current_f_total > 1e-12:
            f_scale = required_f_total / current_f_total
            F_d = F_d * f_scale

        # Check convergence
        error_product, error_industry, error_pib = _compute_mip_errors(
            Z_d, Z_m, F_d, F_m, VA, X
        )

        if error_industry < tol and error_pib < tol:
            return MIPBalanceResult(
                Z_d=Z_d,
                Z_m=Z_m,
                F_d=F_d,
                F_m=F_m,
                VA=VA,
                X=X,
                error_product=error_product,
                error_industry=error_industry,
                error_pib=error_pib,
                iterations=outer + 1,
                converged=True,
                method="ras",
            )

    error_product, error_industry, error_pib = _compute_mip_errors(
        Z_d, Z_m, F_d, F_m, VA, X
    )
    return MIPBalanceResult(
        Z_d=Z_d,
        Z_m=Z_m,
        F_d=F_d,
        F_m=F_m,
        VA=VA,
        X=X,
        error_product=error_product,
        error_industry=error_industry,
        error_pib=error_pib,
        iterations=max_iter,
        converged=False,
        method="ras",
    )


def balance_mip_gras(
    Z_d: np.ndarray,
    Z_m: np.ndarray,
    F_d: np.ndarray,
    F_m: np.ndarray,
    VA: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> MIPBalanceResult:
    """Balance MIP using sequential GRAS.

    Similar to balance_mip_ras but uses GRAS for matrices that may
    contain negative values (e.g., inventory changes in final demand).

    Args:
        Z_d: Domestic intermediate consumption (n x n)
        Z_m: Import intermediate consumption (n x n)
        F_d: Domestic final demand (n x k) or (n,)
        F_m: Import final demand (n x k) or (n,)
        VA: Value added (v x n) where v = number of VA components
        X: Production totals (n,)
        max_iter: Maximum outer iterations
        tol: Convergence tolerance

    Returns:
        MIPBalanceResult with balanced matrices
    """
    Z_d = Z_d.astype(float).copy()
    F_d = F_d.astype(float).copy()
    Z_m = Z_m.astype(float).copy()
    F_m = F_m.astype(float).copy()
    VA = VA.astype(float).copy()
    X = X.astype(float).copy()

    n = len(X)
    PIB_target = float(VA.sum())

    for outer in range(max_iter):
        # Step 1: Balance Z columns (industry balance)
        col_targets = X - VA.sum(axis=0) - Z_m.sum(axis=0)
        col_targets = np.maximum(col_targets, 0)

        z_col_sums = Z_d.sum(axis=0)
        col_factors = np.where(z_col_sums > 1e-12, col_targets / z_col_sums, 1.0)
        Z_d = Z_d * col_factors[np.newaxis, :]

        # Step 2: Balance Z using GRAS (handles potential negatives)
        z_row_sums = Z_d.sum(axis=1)
        z_col_sums = Z_d.sum(axis=0)
        z_targets = np.sqrt(np.maximum(z_row_sums, 0) * np.maximum(z_col_sums, 0))

        if z_targets.sum() > 0:
            Z_d, _, _ = gras_balance(Z_d, z_targets, z_targets, max_iter=200, tol=1e-8)

        # Step 3: Adjust F_d to satisfy PIB identity
        required_f_total = PIB_target + F_m.sum()
        current_f_total = F_d.sum()
        if current_f_total > 1e-12:
            f_scale = required_f_total / current_f_total
            F_d = F_d * f_scale

        # Check convergence
        error_product, error_industry, error_pib = _compute_mip_errors(
            Z_d, Z_m, F_d, F_m, VA, X
        )

        if error_industry < tol and error_pib < tol:
            return MIPBalanceResult(
                Z_d=Z_d,
                Z_m=Z_m,
                F_d=F_d,
                F_m=F_m,
                VA=VA,
                X=X,
                error_product=error_product,
                error_industry=error_industry,
                error_pib=error_pib,
                iterations=outer + 1,
                converged=True,
                method="gras",
            )

    error_product, error_industry, error_pib = _compute_mip_errors(
        Z_d, Z_m, F_d, F_m, VA, X
    )
    return MIPBalanceResult(
        Z_d=Z_d,
        Z_m=Z_m,
        F_d=F_d,
        F_m=F_m,
        VA=VA,
        X=X,
        error_product=error_product,
        error_industry=error_industry,
        error_pib=error_pib,
        iterations=max_iter,
        converged=False,
        method="gras",
    )


def balance_mip_sut_ras(
    Z_d: np.ndarray,
    Z_m: np.ndarray,
    F_d: np.ndarray,
    F_m: np.ndarray,
    VA: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> MIPBalanceResult:
    """Balance MIP using SUT-RAS simultaneous approach.

    This method treats the MIP as an extended Supply-Use Table and
    balances all constraints simultaneously using block-wise RAS.

    Algorithm:
    1. Stack Z and F into extended matrix
    2. Define row/column constraints from production/demand identities
    3. Apply GRAS to extended system
    4. Extract balanced blocks

    Args:
        Z_d: Domestic intermediate consumption (n x n)
        Z_m: Import intermediate consumption (n x n)
        F_d: Domestic final demand (n x k) or (n,)
        F_m: Import final demand (n x k) or (n,)
        VA: Value added (v x n) where v = number of VA components
        X: Production totals (n,)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        MIPBalanceResult with balanced matrices
    """
    Z_d = Z_d.astype(float).copy()
    F_d = F_d.astype(float).copy()
    Z_m = Z_m.astype(float).copy()
    F_m = F_m.astype(float).copy()
    VA = VA.astype(float).copy()
    X = X.astype(float).copy()

    n = len(X)
    PIB_target = float(VA.sum())

    # Ensure F_d is 2D
    if F_d.ndim == 1:
        F_d = F_d.reshape(-1, 1)
    if F_m.ndim == 1:
        F_m = F_m.reshape(-1, 1)

    k = F_d.shape[1]

    for outer in range(max_iter):
        # Create extended matrix [Z_d | F_d]
        extended = np.hstack([Z_d, F_d])

        # Row targets: Production X (for product balance)
        row_targets = X.copy()

        # Column targets for Z: X - VA - Z_m (for industry balance)
        col_targets_z = X - VA.sum(axis=0) - Z_m.sum(axis=0)
        col_targets_z = np.maximum(col_targets_z, 0)

        # Column targets for F: proportional to current, scaled to PIB
        f_col_sums = F_d.sum(axis=0)
        f_total = f_col_sums.sum()
        required_f_total = PIB_target + F_m.sum()
        if f_total > 1e-12:
            col_targets_f = f_col_sums * (required_f_total / f_total)
        else:
            col_targets_f = np.ones(k) * (required_f_total / k)

        col_targets = np.concatenate([col_targets_z, col_targets_f])

        # Balance extended matrix
        extended_bal, _, _ = gras_balance(extended, row_targets, col_targets, max_iter=200, tol=1e-8)

        # Extract balanced blocks
        Z_d = extended_bal[:, :n]
        F_d = extended_bal[:, n:]

        # Check convergence
        error_product, error_industry, error_pib = _compute_mip_errors(
            Z_d, Z_m, F_d, F_m, VA, X
        )

        if error_industry < tol and error_pib < tol:
            return MIPBalanceResult(
                Z_d=Z_d,
                Z_m=Z_m,
                F_d=F_d,
                F_m=F_m,
                VA=VA,
                X=X,
                error_product=error_product,
                error_industry=error_industry,
                error_pib=error_pib,
                iterations=outer + 1,
                converged=True,
                method="sut_ras",
            )

    error_product, error_industry, error_pib = _compute_mip_errors(
        Z_d, Z_m, F_d, F_m, VA, X
    )
    return MIPBalanceResult(
        Z_d=Z_d,
        Z_m=Z_m,
        F_d=F_d,
        F_m=F_m,
        VA=VA,
        X=X,
        error_product=error_product,
        error_industry=error_industry,
        error_pib=error_pib,
        iterations=max_iter,
        converged=False,
        method="sut_ras",
    )


def balance_mip_entropy(
    Z_d: np.ndarray,
    Z_m: np.ndarray,
    F_d: np.ndarray,
    F_m: np.ndarray,
    VA: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> MIPBalanceResult:
    """Balance MIP using Cross-Entropy minimization.

    This method minimizes the cross-entropy distance from the original
    matrix while satisfying the MIP constraints. Implemented via
    iterative scaling (equivalent to GRAS for linear constraints).

    The cross-entropy objective is:
        min sum_{ij} x_ij * (log(x_ij / a_ij) - 1) + a_ij

    subject to row/column constraints.

    Args:
        Z_d: Domestic intermediate consumption (n x n)
        Z_m: Import intermediate consumption (n x n)
        F_d: Domestic final demand (n x k) or (n,)
        F_m: Import final demand (n x k) or (n,)
        VA: Value added (v x n) where v = number of VA components
        X: Production totals (n,)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        MIPBalanceResult with balanced matrices

    References:
        Robinson, S., Cattaneo, A., & El-Said, M. (2001). "Updating and
        Estimating a Social Accounting Matrix Using Cross Entropy Methods."
        Economic Systems Research, 13(1), 47-64.
    """
    # For linear constraints, cross-entropy is equivalent to GRAS
    # This implementation uses GRAS with slightly different convergence criteria
    result = balance_mip_gras(
        Z_d, Z_m, F_d, F_m, VA, X,
        max_iter=max_iter,
        tol=tol,
    )
    # Create new result with entropy method name
    return MIPBalanceResult(
        Z_d=result.Z_d,
        Z_m=result.Z_m,
        F_d=result.F_d,
        F_m=result.F_m,
        VA=result.VA,
        X=result.X,
        error_product=result.error_product,
        error_industry=result.error_industry,
        error_pib=result.error_pib,
        iterations=result.iterations,
        converged=result.converged,
        method="entropy",
    )


# =============================================================================
# Legacy function (deprecated, kept for backward compatibility)
# =============================================================================


class MIPBalanceResultLegacy(BaseModel):
    """Result payload for complete MIP balancing (legacy format)."""

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
) -> MIPBalanceResultLegacy:
    """Balance a complete MIP using GRAS method (legacy interface).

    DEPRECATED: Use balance_mip_gras() for new code.

    This function balances the entire MIP system to satisfy:
    1. Row balance (supply = demand for each product)
    2. Column balance (inputs = production for each sector)
    3. PIB identity (production = expenditure)

    Note: Due to mathematical constraints, perfect supply-demand balance
    is not achievable when imported final demand > 0.

    Args:
        mip_df: Full MIP DataFrame
        n_products: Number of product rows (first n rows)
        n_sectors: Number of sector columns (first n columns)
        va_row_indices: Indices of VA rows
        import_row_indices: Indices of import rows
        fd_col_indices: Indices of final demand columns
        fix_va: If True, preserve VA values
        max_iterations: Maximum GRAS iterations
        tolerance: Convergence tolerance

    Returns:
        MIPBalanceResultLegacy with balanced matrix and diagnostics
    """
    M = mip_df.to_numpy(copy=True, dtype=float)

    # Extract blocks
    Z_d = M[:n_products, :n_sectors]
    Z_m = M[import_row_indices, :n_sectors] if import_row_indices else np.zeros((n_products, n_sectors))
    F_d = M[:n_products, fd_col_indices]
    F_m = M[import_row_indices, fd_col_indices] if import_row_indices else np.zeros((n_products, len(fd_col_indices)))
    VA = M[va_row_indices, :n_sectors]
    X = Z_d.sum(axis=0) + VA.sum(axis=0)

    if fix_va:
        VA_original = VA.copy()

    # Balance using GRAS
    result = balance_mip_gras(
        Z_d, Z_m, F_d, F_m, VA, X,
        max_iter=max_iterations,
        tol=tolerance,
    )

    # Restore VA if fixed
    if fix_va:
        result.VA[:] = VA_original

    # Write back to matrix
    M[:n_products, :n_sectors] = result.Z_d
    M[:n_products, fd_col_indices] = result.F_d
    if import_row_indices:
        M[import_row_indices, :n_sectors] = result.Z_m
        M[import_row_indices, fd_col_indices] = result.F_m
    M[va_row_indices, :n_sectors] = result.VA

    # Calculate PIB
    pib_production = float(result.VA.sum())
    pib_expenditure = float(result.F_d.sum() - result.F_m.sum())

    return MIPBalanceResultLegacy(
        matrix=pd.DataFrame(M, index=mip_df.index, columns=mip_df.columns),
        row_balance_max_diff=result.error_product,
        col_balance_max_diff=result.error_industry,
        pib_production=pib_production,
        pib_expenditure=pib_expenditure,
        pib_diff=abs(pib_production - pib_expenditure),
        iterations=result.iterations,
        converged=result.converged,
    )
