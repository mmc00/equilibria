"""Tests for Model class and block system."""

import numpy as np
import pytest

from equilibria import Model
from equilibria.blocks import CESValueAdded, LeontiefIntermediate
from equilibria.core import Parameter, Set, Variable
from equilibria.core.sets import SetManager

# Mark test_missing_set_raises as expected to fail since we need a custom block
# that requires a missing set to properly test this
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TestModel:
    """Tests for Model class."""

    @pytest.fixture
    def basic_model(self):
        """Create a basic model with sets."""
        model = Model(name="TestModel")
        model.add_set(Set(name="J", elements=("sec1", "sec2")))
        model.add_set(Set(name="I", elements=("labor", "capital")))
        model.add_set(Set(name="F", elements=("labor", "capital")))
        return model

    def test_model_creation(self):
        """Test basic model creation."""
        model = Model(name="Test", description="A test model")
        assert model.name == "Test"
        assert model.description == "A test model"

    def test_add_set(self, basic_model):
        """Test adding sets."""
        assert "J" in basic_model.set_manager
        assert "I" in basic_model.set_manager
        assert "F" in basic_model.set_manager

    def test_add_sets(self, basic_model):
        """Test adding multiple sets."""
        new_sets = [
            Set(name="R", elements=("USA", "EUR")),
            Set(name="T", elements=("t1", "t2")),
        ]
        basic_model.add_sets(new_sets)
        assert "R" in basic_model.set_manager
        assert "T" in basic_model.set_manager

    def test_add_parameter(self, basic_model):
        """Test adding parameters."""
        param = Parameter(
            name="sigma",
            value=np.array([0.8, 0.9]),
            domains=("J",),
        )
        basic_model.add_parameter(param)
        assert "sigma" in basic_model.parameter_manager

    def test_add_variable(self, basic_model):
        """Test adding variables."""
        var = Variable(
            name="VA",
            value=np.array([100.0, 150.0]),
            domains=("J",),
            lower=0.0,
        )
        basic_model.add_variable(var)
        assert "VA" in basic_model.variable_manager

    def test_add_block(self, basic_model):
        """Test adding blocks."""
        block = CESValueAdded(sigma=0.8, name="CES_VA")
        basic_model.add_block(block)

        assert len(basic_model.blocks) == 1
        assert basic_model.blocks[0].name == "CES_VA"

    def test_add_multiple_blocks(self, basic_model):
        """Test adding multiple blocks."""
        blocks = [
            CESValueAdded(sigma=0.8, name="CES_VA"),
            LeontiefIntermediate(name="Leontief_INT"),
        ]
        basic_model.add_blocks(blocks)

        assert len(basic_model.blocks) == 2

    def test_get_parameter(self, basic_model):
        """Test getting parameters."""
        param = Parameter(name="sigma", value=np.array([0.8]))
        basic_model.add_parameter(param)

        retrieved = basic_model.get_parameter("sigma")
        assert retrieved.name == "sigma"

    def test_get_variable(self, basic_model):
        """Test getting variables."""
        var = Variable(name="VA", value=np.array([100.0]))
        basic_model.add_variable(var)

        retrieved = basic_model.get_variable("VA")
        assert retrieved.name == "VA"

    def test_model_statistics(self, basic_model):
        """Test model statistics."""
        basic_model.add_block(CESValueAdded(sigma=0.8))

        stats = basic_model.statistics
        assert stats.blocks == 1
        assert stats.variables > 0

    def test_model_summary(self, basic_model):
        """Test model summary."""
        basic_model.add_block(CESValueAdded(sigma=0.8))

        summary = basic_model.summary()
        assert summary["name"] == "TestModel"
        assert "sets" in summary
        assert "parameters" in summary
        assert "variables" in summary

    def test_missing_set_raises(self):
        """Test that missing set raises error."""
        # Create a model without the required "F" set
        model = Model(name="IncompleteModel")
        model.add_set(Set(name="J", elements=("sec1", "sec2")))
        # Try to add CES block which requires both "J" and "F"
        with pytest.raises(ValueError):
            model.add_block(CESValueAdded(sigma=0.8))


class TestBlockBase:
    """Tests for Block base class."""

    def test_block_creation(self):
        """Test basic block creation."""
        block = CESValueAdded(sigma=0.8, name="TestCES")
        assert block.name == "TestCES"
        assert block.sigma == 0.8

    def test_block_required_sets(self):
        """Test required sets."""
        block = CESValueAdded()
        assert "J" in block.required_sets
        assert "F" in block.required_sets

    def test_block_parameters(self):
        """Test block parameter specs."""
        block = CESValueAdded()
        assert "sigma_VA" in block.parameters
        assert "beta_VA" in block.parameters

    def test_block_variables(self):
        """Test block variable specs."""
        block = CESValueAdded()
        assert "VA" in block.variables
        assert "FD" in block.variables

    def test_block_validate_sets(self):
        """Test set validation."""
        block = CESValueAdded()
        set_manager = SetManager()
        set_manager.add(Set(name="J", elements=("a", "b")))
        set_manager.add(Set(name="F", elements=("x", "y")))

        # Should not raise
        assert block.validate_sets(set_manager) is True

    def test_block_validate_sets_missing(self):
        """Test validation with missing sets."""
        block = CESValueAdded()
        set_manager = SetManager()
        set_manager.add(Set(name="J", elements=("a", "b")))
        # Missing "F"

        with pytest.raises(ValueError):
            block.validate_sets(set_manager)

    def test_block_get_info(self):
        """Test block info."""
        block = CESValueAdded(sigma=0.8)
        info = block.get_info()

        assert info["name"] == "CES_VA"
        assert "parameters" in info
        assert "variables" in info


class TestProductionBlocks:
    """Tests for production blocks."""

    def test_ces_va_setup(self):
        """Test CES VA block setup."""
        set_manager = SetManager()
        set_manager.add(Set(name="J", elements=("sec1", "sec2")))
        set_manager.add(Set(name="F", elements=("labor", "capital")))

        block = CESValueAdded(sigma=0.8)
        params = {}
        vars = {}

        block.setup(set_manager, params, vars)

        assert "sigma_VA" in params
        assert "VA" in vars
        assert "FD" in vars

    def test_leontief_setup(self):
        """Test Leontief block setup."""
        set_manager = SetManager()
        set_manager.add(Set(name="J", elements=("sec1", "sec2")))
        set_manager.add(Set(name="I", elements=("labor", "capital")))

        block = LeontiefIntermediate()
        params = {}
        vars = {}

        block.setup(set_manager, params, vars)

        assert "a_io" in params
        assert "XST" in vars
        assert "Z" in vars
