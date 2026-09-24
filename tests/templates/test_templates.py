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


class TestSimpleOpenFactorSet:
    """El value-added debe construirse sobre los factores reales (issue #15).

    `CESValueAdded` declara `required_sets = ["J", "F"]`, donde F son los
    FACTORES (beta_VA/FD indexados (F,J); WF indexado (F,)). El template
    llamaba "F" al set de firmas -- un unico elemento -- y "I" al de
    factores, de modo que el bloque se armaba sobre 1 factor ficticio:
    FD (1,3) en vez de (2,3) y solo 3 de las 6 CES_FOC (`WF = PVA*dVA/dFD`)
    llegaban a existir. Construia y corria; nadie lo veia porque el gate de
    paridad compara 9 variables y ninguna es FD/WF.
    """

    def test_factor_set_F_holds_the_factors(self):
        template = SimpleOpenEconomy(num_factors=2)
        model = template.create_model()

        assert tuple(model.set_manager.get("F")) == ("LAB", "CAP")

    def test_value_added_spans_every_factor(self):
        template = SimpleOpenEconomy(num_sectors=3, num_factors=2)
        model = template.create_model()

        fd = model.get_variable("FD")
        wf = model.get_variable("WF")
        beta = model.get_parameter("beta_VA")

        assert fd.value.shape == (2, 3), "FD debe cubrir 2 factores x 3 sectores"
        assert wf.value.shape == (2,), "WF debe tener un precio por factor"
        assert beta.value.shape == (2, 3)

    def test_one_foc_per_factor_and_sector(self):
        """CES_FOC (WF = PVA*dVA/dFD) es una por (factor, sector)."""
        template = SimpleOpenEconomy(num_sectors=3, num_factors=2)
        model = template.create_model()

        focs = model.equation_manager.summary()["equations"]["CES_FOC"]
        assert focs["scalar_count"] == 6, (
            f"esperadas 6 CES_FOC (2 factores x 3 sectores), hay {focs['scalar_count']}"
        )
