"""Tests for model templates."""

import pytest

from equilibria.templates import ModelTemplate, SimpleOpenEconomy


class TestModelTemplate:
    """Tests for ModelTemplate base class."""

    def test_template_creation(self):
        """Test that base template cannot be instantiated."""
        with pytest.raises(TypeError):
            ModelTemplate(name="Test")


class TestSimpleOpenEconomy:
    """Tests for SimpleOpenEconomy template."""

    def test_template_default_creation(self):
        """Test template creation with defaults."""
        template = SimpleOpenEconomy()
        assert template.name == "SimpleOpenEconomy"
        assert template.num_sectors == 3
        assert template.num_factors == 2

    def test_template_custom_creation(self):
        """Test template creation with custom values."""
        template = SimpleOpenEconomy(
            num_sectors=5,
            num_factors=3,
            sigma_va=1.0,
            sigma_m=2.0,
            sigma_e=3.0,
        )
        assert template.num_sectors == 5
        assert template.num_factors == 3
        assert template.sigma_va == 1.0

    def test_default_sector_names(self):
        """Test default sector name generation."""
        template = SimpleOpenEconomy(num_sectors=3)
        names = template.get_default_sector_names()
        assert names == ["AGR", "MFG", "SRV"]

    def test_default_factor_names(self):
        """Test default factor name generation."""
        template = SimpleOpenEconomy(num_factors=2)
        names = template.get_default_factor_names()
        assert names == ["LAB", "CAP"]

    def test_many_sectors(self):
        """Test sector naming with many sectors."""
        template = SimpleOpenEconomy(num_sectors=10)
        names = template.get_default_sector_names()
        assert len(names) == 10
        assert names[0] == "SEC1"
        assert names[9] == "SEC10"

    def test_create_model(self):
        """Test model creation from template."""
        template = SimpleOpenEconomy()
        model = template.create_model()

        assert model.name == "SimpleOpenEconomy"
        assert "J" in model.set_manager
        assert "I" in model.set_manager

    def test_model_has_blocks(self):
        """Test that created model has blocks."""
        template = SimpleOpenEconomy()
        model = template.create_model()

        assert len(model.blocks) > 0

    def test_model_has_variables(self):
        """Test that created model has variables."""
        template = SimpleOpenEconomy()
        model = template.create_model()

        stats = model.statistics
        assert stats.variables > 0

    def test_model_has_parameters(self):
        """Test that created model has parameters."""
        template = SimpleOpenEconomy()
        model = template.create_model()

        param_count = len(model.parameter_manager.list_params())
        assert param_count > 0

    def test_template_info(self):
        """Test template info."""
        template = SimpleOpenEconomy()
        info = template.get_info()

        assert info["name"] == "SimpleOpenEconomy"
        assert info["num_sectors"] == 3
        assert info["num_factors"] == 2
        assert "sigma_va" in info
        assert "blocks" in info

    def test_template_repr(self):
        """Test string representation."""
        template = SimpleOpenEconomy()
        repr_str = repr(template)
        assert "SimpleOpenEconomy" in repr_str
        assert "3 sectors" in repr_str
        assert "2 factors" in repr_str
