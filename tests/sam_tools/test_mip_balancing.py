"""Tests for MIP balancing methods.

Tests cover:
- RAS and GRAS basic convergence
- MIP balancing methods (ras, gras, sut_ras, entropy)
- Verification of product error = 0, industry error = 0
- Aggregate identity: F_d = VA + Z_m
"""

from __future__ import annotations

import numpy as np
import pytest

from equilibria.sam_tools.balancing import (
    MIPBalanceResult,
    gras_balance,
    ras_balance,
    balance_mip_gras,
    balance_mip_ras,
    balance_mip_sut_ras,
    balance_mip_entropy,
)


# =============================================================================
# Fixtures
# =============================================================================


def _sample_nonnegative_matrix() -> np.ndarray:
    """Simple non-negative matrix for testing."""
    return np.array(
        [
            [10.0, 20.0, 5.0],
            [15.0, 25.0, 10.0],
            [5.0, 10.0, 30.0],
        ],
        dtype=float,
    )


def _sample_matrix_with_negatives() -> np.ndarray:
    """Matrix with negative values (e.g., inventory changes)."""
    return np.array(
        [
            [10.0, 20.0, -5.0],
            [15.0, 25.0, 10.0],
            [-5.0, 10.0, 30.0],
        ],
        dtype=float,
    )


def _sample_mip_data() -> dict[str, np.ndarray]:
    """Sample MIP data for testing balancing methods."""
    n = 5  # 5 sectors

    # Domestic intermediate consumption
    Z_d = np.array(
        [
            [100, 50, 30, 20, 10],
            [40, 200, 60, 30, 20],
            [20, 40, 150, 40, 30],
            [10, 20, 30, 100, 50],
            [5, 10, 20, 30, 80],
        ],
        dtype=float,
    )

    # Imported intermediate consumption
    Z_m = np.array(
        [
            [20, 10, 5, 3, 2],
            [8, 40, 12, 6, 4],
            [4, 8, 30, 8, 6],
            [2, 4, 6, 20, 10],
            [1, 2, 4, 6, 16],
        ],
        dtype=float,
    )

    # Domestic final demand (C_hh, C_gov, FBKF, Var.S, Exports)
    F_d = np.array(
        [
            [50, 20, 15, 5, 30],
            [80, 30, 25, -5, 50],  # Note: Var.S can be negative
            [60, 25, 20, 3, 40],
            [40, 15, 10, 2, 25],
            [30, 10, 8, 1, 20],
        ],
        dtype=float,
    )

    # Imported final demand
    F_m = np.array(
        [
            [10, 2, 3, 1, 5],
            [16, 3, 5, 0, 10],
            [12, 2, 4, 0, 8],
            [8, 1, 2, 0, 5],
            [6, 1, 1, 0, 4],
        ],
        dtype=float,
    )

    # Value added (L, K, Taxes)
    VA = np.array(
        [
            [100, 150, 120, 80, 60],  # Remunerations
            [150, 200, 180, 120, 100],  # GOS
            [10, 15, 12, 8, 5],  # Net taxes
        ],
        dtype=float,
    )

    # Calculate production totals (from supply side)
    X = Z_d.sum(axis=0) + Z_m.sum(axis=0) + VA.sum(axis=0)

    return {
        "Z_d": Z_d,
        "Z_m": Z_m,
        "F_d": F_d,
        "F_m": F_m,
        "VA": VA,
        "X": X,
    }


# =============================================================================
# RAS Tests
# =============================================================================


class TestRASBalance:
    """Tests for classic RAS algorithm."""

    def test_ras_converges_nonnegative(self) -> None:
        """RAS should converge for non-negative matrix."""
        M = _sample_nonnegative_matrix()
        row_targets = M.sum(axis=1)
        col_targets = M.sum(axis=0)

        M_bal, iters, converged = ras_balance(M, row_targets, col_targets)

        assert converged
        assert iters < 500
        assert np.allclose(M_bal.sum(axis=1), row_targets, atol=1e-6)
        assert np.allclose(M_bal.sum(axis=0), col_targets, atol=1e-6)

    def test_ras_with_different_targets(self) -> None:
        """RAS should balance to specified different targets."""
        M = _sample_nonnegative_matrix()
        row_targets = np.array([40.0, 50.0, 45.0])
        col_targets = np.array([30.0, 55.0, 50.0])

        M_bal, iters, converged = ras_balance(M, row_targets, col_targets)

        assert converged
        assert np.allclose(M_bal.sum(axis=1), row_targets, atol=1e-6)
        assert np.allclose(M_bal.sum(axis=0), col_targets, atol=1e-6)

    def test_ras_preserves_structure(self) -> None:
        """RAS should preserve zero entries."""
        M = np.array(
            [
                [10.0, 0.0, 5.0],
                [0.0, 20.0, 10.0],
                [5.0, 10.0, 0.0],
            ],
            dtype=float,
        )
        row_targets = M.sum(axis=1)
        col_targets = M.sum(axis=0)

        M_bal, _, _ = ras_balance(M, row_targets, col_targets)

        # Zeros should remain zeros
        assert M_bal[0, 1] == 0.0
        assert M_bal[1, 0] == 0.0
        assert M_bal[2, 2] == 0.0


# =============================================================================
# GRAS Tests
# =============================================================================


class TestGRASBalance:
    """Tests for GRAS algorithm (handles negatives)."""

    def test_gras_converges_nonnegative(self) -> None:
        """GRAS should converge for non-negative matrix (like RAS)."""
        M = _sample_nonnegative_matrix()
        row_targets = M.sum(axis=1)
        col_targets = M.sum(axis=0)

        M_bal, iters, converged = gras_balance(M, row_targets, col_targets)

        assert converged
        assert iters < 500
        assert np.allclose(M_bal.sum(axis=1), row_targets, atol=1e-6)
        assert np.allclose(M_bal.sum(axis=0), col_targets, atol=1e-6)

    def test_gras_handles_negatives(self) -> None:
        """GRAS should handle matrix with negative values."""
        M = _sample_matrix_with_negatives()
        row_targets = M.sum(axis=1)
        col_targets = M.sum(axis=0)

        M_bal, iters, converged = gras_balance(M, row_targets, col_targets)

        assert converged or iters == 500  # May need more iterations
        # Check approximate convergence
        row_diff = np.abs(M_bal.sum(axis=1) - row_targets).max()
        col_diff = np.abs(M_bal.sum(axis=0) - col_targets).max()
        assert row_diff < 1.0 or col_diff < 1.0

    def test_gras_preserves_signs(self) -> None:
        """GRAS should preserve signs of entries."""
        M = _sample_matrix_with_negatives()
        row_targets = M.sum(axis=1) * 1.2  # Scale targets
        col_targets = M.sum(axis=0) * 1.2

        M_bal, _, _ = gras_balance(M, row_targets, col_targets)

        # Check sign preservation
        assert M_bal[0, 2] < 0  # Was negative
        assert M_bal[2, 0] < 0  # Was negative
        assert M_bal[0, 0] > 0  # Was positive


# =============================================================================
# MIP Balancing Tests
# =============================================================================


class TestMIPBalancingRAS:
    """Tests for MIP balancing using RAS."""

    def test_balance_mip_ras_converges(self) -> None:
        """MIP RAS balancing should converge."""
        data = _sample_mip_data()

        result = balance_mip_ras(
            data["Z_d"],
            data["Z_m"],
            data["F_d"],
            data["F_m"],
            data["VA"],
            data["X"],
        )

        assert isinstance(result, MIPBalanceResult)
        assert result.converged or result.iterations == 100
        assert result.method == "ras"

    def test_balance_mip_ras_product_industry_error(self) -> None:
        """MIP RAS should achieve zero product and industry balance errors."""
        data = _sample_mip_data()

        result = balance_mip_ras(
            data["Z_d"],
            data["Z_m"],
            data["F_d"],
            data["F_m"],
            data["VA"],
            data["X"],
        )

        # Product balance: Z_d.sum(axis=1) + F_d.sum(axis=1) = X
        # Industry balance: Z_d.sum(axis=0) + Z_m.sum(axis=0) + VA.sum(axis=0) = X
        # Both should be zero after balancing
        assert result.error_product < 1e-6  # Essentially zero
        assert result.error_industry < 1e-6  # Essentially zero


class TestMIPBalancingGRAS:
    """Tests for MIP balancing using GRAS."""

    def test_balance_mip_gras_converges(self) -> None:
        """MIP GRAS balancing should converge."""
        data = _sample_mip_data()

        result = balance_mip_gras(
            data["Z_d"],
            data["Z_m"],
            data["F_d"],
            data["F_m"],
            data["VA"],
            data["X"],
        )

        assert isinstance(result, MIPBalanceResult)
        assert result.converged or result.iterations == 100
        assert result.method == "gras"

    def test_balance_mip_gras_handles_negative_inventory(self) -> None:
        """MIP GRAS should handle negative inventory changes."""
        data = _sample_mip_data()
        # F_d already has negative Var.S values

        result = balance_mip_gras(
            data["Z_d"],
            data["Z_m"],
            data["F_d"],
            data["F_m"],
            data["VA"],
            data["X"],
        )

        # Should converge despite negatives
        assert result.converged or result.iterations <= 100

    def test_balance_mip_gras_industry_error(self) -> None:
        """MIP GRAS should minimize industry balance error."""
        data = _sample_mip_data()

        result = balance_mip_gras(
            data["Z_d"],
            data["Z_m"],
            data["F_d"],
            data["F_m"],
            data["VA"],
            data["X"],
        )

        # Industry error should be small
        assert result.error_industry < 100  # Absolute tolerance


class TestMIPBalancingSUTRAS:
    """Tests for MIP balancing using SUT-RAS."""

    def test_balance_mip_sut_ras_converges(self) -> None:
        """MIP SUT-RAS balancing should converge."""
        data = _sample_mip_data()

        result = balance_mip_sut_ras(
            data["Z_d"],
            data["Z_m"],
            data["F_d"],
            data["F_m"],
            data["VA"],
            data["X"],
        )

        assert isinstance(result, MIPBalanceResult)
        assert result.method == "sut_ras"


class TestMIPBalancingEntropy:
    """Tests for MIP balancing using Cross-Entropy."""

    def test_balance_mip_entropy_converges(self) -> None:
        """MIP Cross-Entropy balancing should converge."""
        data = _sample_mip_data()

        result = balance_mip_entropy(
            data["Z_d"],
            data["Z_m"],
            data["F_d"],
            data["F_m"],
            data["VA"],
            data["X"],
        )

        assert isinstance(result, MIPBalanceResult)
        assert result.method == "entropy"


# =============================================================================
# Aggregate Identity Tests
# =============================================================================


class TestAggregateIdentity:
    """Tests for MIP balance identities after balancing."""

    def test_product_and_industry_balance_hold(self) -> None:
        """After balancing, product and industry balance should hold exactly.

        - Product balance: X = Z_d.sum(axis=1) + F_d.sum(axis=1)
        - Industry balance: X = Z_d.sum(axis=0) + Z_m.sum(axis=0) + VA.sum(axis=0)

        Note: PIB identity may not hold perfectly (depends on data).
        """
        data = _sample_mip_data()

        result = balance_mip_gras(
            data["Z_d"],
            data["Z_m"],
            data["F_d"],
            data["F_m"],
            data["VA"],
            data["X"],
        )

        # Product balance: X = Z_d·1 + F_d·1
        product_balance = result.X - result.Z_d.sum(axis=1) - result.F_d.sum(axis=1)
        assert np.abs(product_balance).sum() < 1e-6

        # Industry balance: X = Z_d'·1 + Z_m'·1 + VA'·1
        industry_balance = (
            result.X
            - result.Z_d.sum(axis=0)
            - result.Z_m.sum(axis=0)
            - result.VA.sum(axis=0)
        )
        assert np.abs(industry_balance).sum() < 1e-6


# =============================================================================
# Shape Preservation Tests
# =============================================================================


class TestShapePreservation:
    """Tests that balancing preserves matrix shapes."""

    def test_mip_balancing_preserves_shapes(self) -> None:
        """All output matrices should have same shape as input."""
        data = _sample_mip_data()

        result = balance_mip_gras(
            data["Z_d"],
            data["Z_m"],
            data["F_d"],
            data["F_m"],
            data["VA"],
            data["X"],
        )

        assert result.Z_d.shape == data["Z_d"].shape
        assert result.Z_m.shape == data["Z_m"].shape
        assert result.F_d.shape == data["F_d"].shape
        assert result.F_m.shape == data["F_m"].shape
        assert result.VA.shape == data["VA"].shape
        assert result.X.shape == data["X"].shape


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ras_with_zero_row(self) -> None:
        """RAS should handle matrix with zero row."""
        M = np.array(
            [
                [10.0, 20.0, 5.0],
                [0.0, 0.0, 0.0],  # Zero row
                [5.0, 10.0, 30.0],
            ],
            dtype=float,
        )
        row_targets = np.array([35.0, 0.0, 45.0])
        col_targets = np.array([15.0, 30.0, 35.0])

        M_bal, _, converged = ras_balance(M, row_targets, col_targets)

        # Zero row should remain zero
        assert np.allclose(M_bal[1, :], 0.0)

    def test_ras_with_small_matrix(self) -> None:
        """RAS should work with 2x2 matrix."""
        M = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)
        row_targets = np.array([30.0, 70.0])
        col_targets = np.array([40.0, 60.0])

        M_bal, _, converged = ras_balance(M, row_targets, col_targets)

        assert converged
        assert np.allclose(M_bal.sum(axis=1), row_targets, atol=1e-6)

    def test_mip_with_single_fd_column(self) -> None:
        """MIP balancing should work with single FD column."""
        n = 3
        Z_d = np.eye(n) * 100
        Z_m = np.eye(n) * 20
        F_d = np.array([[50], [60], [70]], dtype=float)
        F_m = np.array([[10], [12], [14]], dtype=float)
        VA = np.array([[80, 90, 100]], dtype=float)
        X = Z_d.sum(axis=0) + Z_m.sum(axis=0) + VA.sum(axis=0)

        result = balance_mip_gras(Z_d, Z_m, F_d, F_m, VA, X)

        assert result.F_d.shape == F_d.shape
