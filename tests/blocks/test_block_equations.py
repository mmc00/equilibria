"""Tests for block equation building.

This module tests that all blocks can build their equations without errors
and that variables are properly defined.
"""

import numpy as np
import pytest

from equilibria.blocks import (
    ArmingtonCES,
    CETExports,
    CETTransformation,
    CESValueAdded,
    CobbDouglasConsumer,
    FactorMarketClearing,
    Government,
    Household,
    LeontiefIntermediate,
    LESConsumer,
    MarketClearing,
    PriceNormalization,
    RestOfWorld,
)
from equilibria.core import Set, SetManager
from equilibria.core.parameters import Parameter
from equilibria.core.variables import Variable


def create_test_sets():
    """Create minimal test sets."""
    set_manager = SetManager()
    set_manager.add(Set(name="J", elements=["agr", "manu"]))  # Sectors
    set_manager.add(Set(name="I", elements=["agr", "manu"]))  # Commodities
    set_manager.add(Set(name="F", elements=["labor", "cap"]))  # Factors
    set_manager.add(Set(name="H", elements=["hh1"]))  # Households
    return set_manager


def test_ces_value_added_equations():
    """Test CESValueAdded block equations build correctly."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = CESValueAdded(elasticity_substitution=1.0)
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) > 0, "Should have equations"

    # Check that required variables are created
    assert "VA" in variables, "VA variable should be created"
    assert "FD" in variables, "FD variable should be created"
    assert "WF" in variables, "WF variable should be created"

    # Check FD is 2D (factors x sectors)
    fd_var = variables["FD"]
    assert len(fd_var.domains) == 2, "FD should be 2D"
    assert fd_var.domains[0] == "F", "FD first dimension should be F"
    assert fd_var.domains[1] == "J", "FD second dimension should be J"


def test_leontief_intermediate_equations():
    """Test LeontiefIntermediate block equations build correctly."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = LeontiefIntermediate()
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) > 0, "Should have equations"
    assert "XST" in variables, "XST variable should be created"
    assert "Z" in variables, "Z variable should be created"


def test_cet_transformation_equations():
    """Test CETTransformation block equations build correctly."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = CETTransformation(elasticity_transformation=2.0)
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) > 0, "Should have equations"
    assert "XD" in variables, "XD variable should be created"
    assert "XE" in variables, "XE variable should be created"


def test_armington_ces_equations():
    """Test ArmingtonCES block equations build correctly."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = ArmingtonCES(elasticity_substitution=2.0)
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) > 0, "Should have equations"
    assert "QD" in variables, "QD variable should be created"
    assert "QM" in variables, "QM variable should be created"
    assert "QA" in variables, "QA variable should be created"


def test_cet_exports_equations():
    """Test CETExports block equations build correctly."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = CETExports(elasticity_transformation=2.0)
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) > 0, "Should have equations"
    assert "XE" in variables, "XE variable should be created"


def test_les_consumer_equations():
    """Test LESConsumer block equations build correctly."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = LESConsumer(household="hh1")
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) > 0, "Should have equations"
    assert "QD" in variables, "QD variable should be created"
    assert "PA" in variables, "PA variable should be created"


def test_household_equations():
    """Test Household block equations build correctly."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = Household()
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) > 0, "Should have equations"
    assert "YH" in variables, "YH variable should be created"
    assert "WF" in variables, "WF variable should be created"
    assert "FSUP" in variables, "FSUP variable should be created"


def test_government_equations():
    """Test Government block equations build correctly."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = Government()
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) > 0, "Should have equations"
    assert "YG" in variables, "YG variable should be created"
    assert "XG" in variables, "XG variable should be created"

    # Check XG has positive initial values (not zeros for log safety)
    xg_var = variables["XG"]
    assert np.all(xg_var.value > 0), "XG should have positive initial values"


def test_rest_of_world_equations():
    """Test RestOfWorld block equations build correctly."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = RestOfWorld()
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) > 0, "Should have equations"
    assert "QM" in variables, "QM variable should be created"
    assert "QE" in variables, "QE variable should be created"
    assert "FSAV" in variables, "FSAV variable should be created"

    # Check QM, QE, FSAV have positive initial values
    assert np.all(variables["QM"].value > 0), "QM should have positive initial values"
    assert np.all(variables["QE"].value > 0), "QE should have positive initial values"
    assert variables["FSAV"].value[0] > 0, "FSAV should have positive initial value"


def test_market_clearing_equations():
    """Test MarketClearing block equations build correctly."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = MarketClearing()
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) > 0, "Should have equations"
    assert "QS" in variables, "QS variable should be created"
    assert "QD" in variables, "QD variable should be created"
    assert "P" in variables, "P variable should be created"


def test_price_normalization_scope():
    """Test PriceNormalization numeraire is captured in closure."""
    set_manager = create_test_sets()
    parameters = {}
    variables = {}

    block = PriceNormalization(numeraire="agr")
    equations = block.setup(set_manager, parameters, variables)

    assert len(equations) == 1, "Should have one equation"
    assert "P" in variables, "P variable should be created"

    # The equation should have captured numeraire in closure
    # (actual test would require building with Pyomo)


def test_factor_market_clearing_no_fd_redefinition():
    """Test FactorMarketClearing doesn't redefine FD (it's from CESValueAdded)."""
    set_manager = create_test_sets()

    # First create FD in CESValueAdded
    ces_block = CESValueAdded(elasticity_substitution=1.0)
    parameters = {}
    variables = {}
    ces_block.setup(set_manager, parameters, variables)

    assert "FD" in variables, "CESValueAdded should create FD"
    fd_var = variables["FD"]

    # Now create FactorMarketClearing
    fmc_block = FactorMarketClearing()
    fmc_block.setup(set_manager, parameters, variables)

    # FD should still be the same 2D variable
    assert "FD" in variables, "FD should still exist"
    assert variables["FD"] is fd_var, "FD should not be redefined"
    assert len(variables["FD"].domains) == 2, "FD should still be 2D"


def test_block_calibration_no_nan():
    """Test that block calibration doesn't produce NaN values."""
    from equilibria.core.calibration_data import CalibrationData
    from equilibria.templates.data.pep import load_default_pep_sam

    sam = load_default_pep_sam()
    data = CalibrationData(sam, mode="sam")
    data.register_set_mapping("J", ["agr", "othind", "food", "ser", "adm"])
    data.register_set_mapping("I", ["agr", "othind", "food", "ser", "adm"])

    set_manager = create_test_sets()

    # Test CET transformation calibration
    block = CETTransformation(elasticity_transformation=2.0)
    calibrated = block._extract_calibration(
        phase=None, data=data, mode="sam", set_manager=set_manager
    )

    # Check no NaN in shares
    assert "gamma_D" in calibrated, "Should have gamma_D"
    assert "gamma_E" in calibrated, "Should have gamma_E"
    assert not np.any(np.isnan(calibrated["gamma_D"])), "gamma_D should not have NaN"
    assert not np.any(np.isnan(calibrated["gamma_E"])), "gamma_E should not have NaN"

    # Test Armington calibration
    block = ArmingtonCES(elasticity_substitution=2.0)
    calibrated = block._extract_calibration(
        phase=None, data=data, mode="sam", set_manager=set_manager
    )

    assert not np.any(np.isnan(calibrated["alpha_D"])), "alpha_D should not have NaN"
    assert not np.any(np.isnan(calibrated["alpha_M"])), "alpha_M should not have NaN"
