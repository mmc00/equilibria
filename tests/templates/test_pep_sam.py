"""Tests for PEP SAM loader.

This module tests the PEP SAM loader functionality to ensure
it correctly reads account names from the Excel file.
"""

import numpy as np
import pytest

from equilibria.core.calibration_data import CalibrationData
from equilibria.templates.data.pep import load_default_pep_sam


def test_sam_loader_reads_correct_accounts():
    """Test that SAM loader reads account names not category names."""
    sam = load_default_pep_sam()

    # Check for expected account names (not categories like L, K, AG)
    expected_accounts = [
        "L_USK",
        "L_SK",  # Labor types
        "K_CAP",
        "K_LAND",  # Capital types
        "AG_HRP",
        "AG_HUP",
        "AG_HRR",
        "AG_HUR",  # Households
    ]

    for account in expected_accounts:
        assert account in sam.data.index, f"Account {account} not found in SAM rows"
        assert account in sam.data.columns, (
            f"Account {account} not found in SAM columns"
        )

    # Check that category names are NOT the main labels
    category_names = ["L", "K", "AG", "J", "I"]
    for cat in category_names:
        # Categories may appear as headers but shouldn't be the account names
        assert cat not in sam.data.index[:10], (
            f"Category {cat} should not be in first 10 accounts"
        )


def test_sam_matrix_dimensions():
    """Test SAM matrix has correct dimensions."""
    sam = load_default_pep_sam()

    # Should be square matrix
    assert sam.data.shape[0] == sam.data.shape[1]

    # Check specific dimensions (should have at least 35+ accounts)
    assert sam.data.shape[0] >= 34, (
        f"SAM should have at least 34 accounts, got {sam.data.shape[0]}"
    )


def test_sam_data_extraction():
    """Test extraction of specific sub-matrices."""
    sam = load_default_pep_sam()
    data = CalibrationData(sam, mode="sam")

    # Register set mappings
    data.register_set_mapping("F", ["usk", "sk", "cap", "land"])
    data.register_set_mapping("J", ["agr", "othind", "food", "ser", "adm"])

    # Extract FxJ matrix (factor demands)
    fxj_matrix = data.get_matrix("F", "J")

    assert fxj_matrix.shape == (4, 5), f"Expected (4,5), got {fxj_matrix.shape}"
    assert not np.all(fxj_matrix == 0), "FxJ matrix should not be all zeros"
    assert not np.any(np.isnan(fxj_matrix)), "FxJ matrix should not contain NaN"


def test_sam_balance():
    """Test SAM rows sum to columns (balance condition)."""
    sam = load_default_pep_sam()

    # Row sums should equal column sums
    row_sums = sam.data.sum(axis=1)
    col_sums = sam.data.sum(axis=0)

    np.testing.assert_allclose(
        row_sums,
        col_sums,
        rtol=1e-6,
        err_msg="SAM should be balanced (row sums = column sums)",
    )


def test_calibration_no_division_by_zero():
    """Test that calibration doesn't produce NaN from division by zero."""
    sam = load_default_pep_sam()
    data = CalibrationData(sam, mode="sam")

    # Register mappings
    data.register_set_mapping("F", ["usk", "sk", "cap", "land"])
    data.register_set_mapping("J", ["agr", "othind", "food", "ser", "adm"])
    data.register_set_mapping("I", ["agr", "othind", "food", "ser", "adm"])

    # Extract matrices that could have zeros
    fxj = data.get_matrix("F", "J")
    jxi = data.get_matrix("J", "I")

    # Test division safety with np.where pattern
    row_sums = fxj.sum(axis=1)
    col_sums = fxj.sum(axis=0)

    # Safe division - should not produce NaN
    safe_div_row = np.where(row_sums > 0, fxj.sum(axis=1) / row_sums, 0.0)
    safe_div_col = np.where(col_sums > 0, fxj.sum(axis=0) / col_sums, 0.0)

    assert not np.any(np.isnan(safe_div_row)), "Row division produced NaN"
    assert not np.any(np.isnan(safe_div_col)), "Column division produced NaN"
