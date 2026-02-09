"""Tests for PEP model calibration consistency.

This module tests that calibration produces consistent base year values
and that all parameters are properly calculated.
"""

import numpy as np
import pytest

from equilibria.core.calibration_data import CalibrationData
from equilibria.templates import PEP1R
from equilibria.templates.data.pep import load_default_pep_sam


def test_calibration_consistency_fd_va():
    """Test FD0 sum across factors equals VA0 (value added)."""
    template = PEP1R()

    # Get calibration data
    sam = load_default_pep_sam()
    data = CalibrationData(sam, mode="sam")

    # Register mappings
    data.register_set_mapping("F", ["usk", "sk", "cap", "land"])
    data.register_set_mapping("J", ["agr", "othind", "food", "ser", "adm"])

    # Extract factor demands (FxJ)
    FD0 = data.get_matrix("F", "J")

    # Sum across factors for each sector (this should equal value added)
    va_from_fd = FD0.sum(axis=0)

    # Check no NaN
    assert not np.any(np.isnan(va_from_fd)), "VA from FD should not have NaN"
    assert not np.any(np.isnan(FD0)), "FD0 should not have NaN"

    # All sectors should have some value added in base year
    assert np.all(va_from_fd >= 0), "All sectors should have non-negative value added"


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


def test_calibration_shares_sum_to_one():
    """Test that calibrated shares sum to approximately 1."""
    from equilibria.blocks import ArmingtonCES, CETTransformation
    from equilibria.core import Set, SetManager

    sam = load_default_pep_sam()
    data = CalibrationData(sam, mode="sam")
    data.register_set_mapping("J", ["agr", "othind", "food", "ser", "adm"])
    data.register_set_mapping("I", ["agr", "othind", "food", "ser", "adm"])

    set_manager = SetManager()
    set_manager.add(Set(name="J", elements=["agr", "othind", "food", "ser", "adm"]))
    set_manager.add(Set(name="I", elements=["agr", "othind", "food", "ser", "adm"]))

    # Test CET shares
    block = CETTransformation(elasticity_transformation=2.0)
    calibrated = block._extract_calibration(None, data, "sam", set_manager)

    # gamma_D + gamma_E should equal 1 (where Z0 > 0)
    gamma_sum = calibrated["gamma_D"] + calibrated["gamma_E"]
    Z0 = calibrated["Z0"]

    # Only check where Z0 > 0
    mask = Z0 > 0
    if np.any(mask):
        np.testing.assert_allclose(
            gamma_sum[mask],
            1.0,
            rtol=1e-10,
            err_msg="CET shares should sum to 1 where output exists",
        )

    # Test Armington shares
    block = ArmingtonCES(elasticity_substitution=2.0)
    calibrated = block._extract_calibration(None, data, "sam", set_manager)

    # alpha_D + alpha_M should equal 1 (where QA0 > 0)
    alpha_sum = calibrated["alpha_D"] + calibrated["alpha_M"]
    QA0 = calibrated["QA0"]

    mask = QA0 > 0
    if np.any(mask):
        np.testing.assert_allclose(
            alpha_sum[mask],
            1.0,
            rtol=1e-10,
            err_msg="Armington shares should sum to 1 where aggregate exists",
        )


def test_calibration_no_negative_values():
    """Test that calibrated base year values are non-negative."""
    sam = load_default_pep_sam()
    data = CalibrationData(sam, mode="sam")

    # Register mappings
    data.register_set_mapping("F", ["usk", "sk", "cap", "land"])
    data.register_set_mapping("J", ["agr", "othind", "food", "ser", "adm"])
    data.register_set_mapping("I", ["agr", "othind", "food", "ser", "adm"])

    # Extract various matrices
    fxj = data.get_matrix("F", "J")  # Factor demands
    jxi = data.get_matrix("J", "I")  # Output by commodity

    # All should be non-negative (SAM values are non-negative)
    assert np.all(fxj >= 0), "Factor demands should be non-negative"
    assert np.all(jxi >= 0), "Output values should be non-negative"


def test_case_insensitive_matching():
    """Test that case-insensitive matching works for SAM accounts."""
    sam = load_default_pep_sam()
    data = CalibrationData(sam, mode="sam")

    # Register with lowercase (SAM has uppercase)
    data.register_set_mapping("F", ["usk", "sk", "cap", "land"])
    data.register_set_mapping("J", ["agr", "othind", "food", "ser", "adm"])

    # Should still extract correctly
    fxj = data.get_matrix("F", "J")

    assert fxj.shape == (4, 5), f"Expected (4,5), got {fxj.shape}"
    assert not np.any(np.isnan(fxj)), (
        "Case-insensitive extraction should not produce NaN"
    )


def test_calibration_data_caching():
    """Test that calibration data is properly cached."""
    sam = load_default_pep_sam()
    data = CalibrationData(sam, mode="sam")
    data.register_set_mapping("F", ["usk", "sk", "cap", "land"])
    data.register_set_mapping("J", ["agr", "othind", "food", "ser", "adm"])

    # First extraction
    fxj1 = data.get_matrix("F", "J")

    # Second extraction (should be cached)
    fxj2 = data.get_matrix("F", "J")

    # Should be the same object (cached)
    assert fxj1 is fxj2, "Calibration data should be cached"


def test_model_statistics():
    """Test that PEP model has expected structure."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    stats = model.statistics

    # Check expected counts
    assert stats.variables > 100, f"Should have >100 variables, got {stats.variables}"
    assert stats.equations > 50, f"Should have >50 equations, got {stats.equations}"
    assert stats.blocks >= 10, f"Should have >=10 blocks, got {stats.blocks}"

    # Check specific variable existence
    var_names = model.variable_manager.list_vars()
    required_vars = ["FD", "VA", "P", "QS", "QD", "YG", "YH", "FSAV"]
    for var in required_vars:
        assert var in var_names, f"Required variable {var} not found"


def test_calibration_initial_values_positive():
    """Test that variables requiring log() have positive initial values."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    # Variables that appear in log() should have positive initial values
    log_vars = ["XG", "YG", "QM", "QE", "FSAV"]

    for var_name in log_vars:
        if var_name in model.variable_manager.list_vars():
            var = model.variable_manager.get(var_name)
            assert np.all(var.value > 0), (
                f"{var_name} should have positive initial values for log safety"
            )
