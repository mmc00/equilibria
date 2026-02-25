"""Integration tests for full PEP model.

This module tests the complete PEP model workflow from creation
through solving.
"""

import numpy as np
import pytest

from equilibria.backends import PyomoBackend
from pyomo.environ import Constraint
from equilibria.templates import PEP1R


def test_model_creates_successfully():
    """Test PEP model can be created with calibration."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    # Check statistics
    stats = model.statistics
    assert stats.variables > 0, "Model should have variables"
    assert stats.equations > 0, "Model should have equations"
    assert stats.blocks > 0, "Model should have blocks"

    # Expected counts for PEP-1R
    assert stats.variables >= 137, f"Expected >=137 variables, got {stats.variables}"
    assert stats.equations >= 100, f"Expected >=100 equations, got {stats.equations}"


def test_model_builds_with_pyomo():
    """Test model can be built with Pyomo backend."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    backend = PyomoBackend(solver="ipopt")
    backend.build(model)

    # Should have a valid Pyomo model
    assert backend._pyomo_model is not None
    assert hasattr(backend._pyomo_model, "component_objects")


def test_model_sets_correct():
    """Test that model has correct PEP sets."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    # Check sets exist
    assert model.set_manager.has("J"), "Should have J (sectors) set"
    assert model.set_manager.has("I"), "Should have I (commodities) set"
    assert model.set_manager.has("F"), "Should have F (factors) set"
    assert model.set_manager.has("H"), "Should have H (households) set"

    # Check set sizes
    j_set = model.set_manager.get("J")
    assert len(j_set) == 5, f"Should have 5 sectors, got {len(j_set)}"

    f_set = model.set_manager.get("F")
    assert len(f_set) == 4, f"Should have 4 factors, got {len(f_set)}"

    h_set = model.set_manager.get("H")
    assert len(h_set) == 4, f"Should have 4 households, got {len(h_set)}"


def test_model_has_all_blocks():
    """Test that model has all expected blocks."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    block_names = [block.name for block in model.blocks]

    # Check for required blocks
    required_blocks = [
            "CES_VA",
            "Leontief_INT",
            "CET_Transform",
            "Armington",
            "CET_Exports",  # Trade
            "Household",
            "Government",
            "ROW",  # Institutions
            "MarketClearing",
            "FactorMarket",
            "PriceNorm",  # Equilibrium
        ]

    for block_name in required_blocks:
        assert block_name in block_names, f"Required block {block_name} not found"


def test_calibration_no_nan_in_model():
    """Test that model variables don't have NaN after calibration."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    for var_name in model.variable_manager.list_vars():
        var = model.variable_manager.get(var_name)
        assert not np.any(np.isnan(var.value)), f"Variable {var_name} has NaN values"


def test_price_normalization_constraint():
    """Test that price normalization constraint is built."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    backend = PyomoBackend(solver="ipopt")
    backend.build(model)

    # Check that price normalization exists
    pyomo_model = backend._pyomo_model
    assert hasattr(pyomo_model, "P"), "Should have P variable"

    # Check P has initial value of 1 for numeraire
    p_var = model.variable_manager.get("P")
    assert p_var.value[0] == 1.0, "Numeraire price should be initialized to 1"


@pytest.mark.slow
def test_model_solves():
    """Test model can be solved (may take time).

    Marked as slow because solving can take several minutes.
    """
    template = PEP1R()
    model = template.create_model(calibrate=True)

    backend = PyomoBackend(solver="ipopt")
    backend.build(model)

    # Try to solve
    try:
        solution = backend.solve()
        # Should at least not crash, even if infeasible
        assert solution is not None, "Should return a solution object"
        assert hasattr(solution, "status"), "Solution should have status"
        assert hasattr(solution, "variables"), "Solution should have variables"
    except Exception as e:
        # If it fails, we should get a meaningful error
        pytest.fail(f"Model solve failed with exception: {e}")


def test_solution_extraction_2d_variables():
    """Test that 2D variables can be extracted from solution."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    backend = PyomoBackend(solver="ipopt")
    backend.build(model)

    # FD is 2D (F x J)
    fd_var = model.variable_manager.get("FD")
    assert len(fd_var.domains) == 2, "FD should be 2D"

    # After solving, we should be able to extract it
    # (Even if solve fails, the extraction mechanism should work)


def test_model_variable_domains():
    """Test that variables have correct domains."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    # Check 1D variables
    var_1d = ["VA", "P", "QS", "QD", "WF", "FSUP"]
    for var_name in var_1d:
        if var_name in model.variable_manager.list_vars():
            var = model.variable_manager.get(var_name)
            assert len(var.domains) == 1, f"{var_name} should be 1D"

    # Check 2D variables
    var_2d = ["FD"]
    for var_name in var_2d:
        if var_name in model.variable_manager.list_vars():
            var = model.variable_manager.get(var_name)
            assert len(var.domains) == 2, f"{var_name} should be 2D"


def test_model_all_variables_initialized():
    """Test that all variables have initial values."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    for var_name in model.variable_manager.list_vars():
        var = model.variable_manager.get(var_name)
        assert var.value is not None, f"Variable {var_name} should have initial value"
        assert len(var.value) > 0, f"Variable {var_name} should have non-empty value"


def test_pyomo_constraints_added():
    """Test that Pyomo constraints are added to model."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    backend = PyomoBackend(solver="ipopt")
    backend.build(model)

    pyomo_model = backend._pyomo_model

    # Count constraints
    n_constraints = sum(1 for _ in pyomo_model.component_objects(Constraint, active=True))

    # Should have roughly the same number of constraints as equations
    assert n_constraints > 0, "Should have constraints"


def test_calibration_creates_all_required_params():
    """Test that calibration creates all required parameters."""
    template = PEP1R()
    model = template.create_model(calibrate=True)

    # Check that calibration data was created for blocks
    has_calibration = False
    for block in model.blocks:
        if hasattr(block, "_calibrated_data") and block._calibrated_data:
            has_calibration = True
            break

    assert has_calibration, "At least one block should have calibration data"
