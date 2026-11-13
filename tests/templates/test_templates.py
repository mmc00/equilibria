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
    """SimpleOpenEconomy is Hosoe's stdcge; its sets come from the SAM."""

    def test_template_default_creation(self):
        template = SimpleOpenEconomy()
        assert template.name == "SimpleOpenEconomy"
        assert (template.sigma, template.psi) == (2.0, 2.0)
        assert template.numeraire_factor() == "LAB"

    def test_create_model_takes_sets_from_the_sam(self):
        model = SimpleOpenEconomy().create_model()
        assert tuple(model.set_manager.get("I")) == ("BRD", "MLK")
        assert tuple(model.set_manager.get("J")) == ("BRD", "MLK")
        assert tuple(model.set_manager.get("F")) == ("CAP", "LAB")

    def test_model_has_the_eleven_blocks(self):
        model = SimpleOpenEconomy().create_model()
        assert len(model.blocks) == 11

    def test_value_added_spans_every_factor(self):
        """Issue #15: value added is built on the real factors."""
        model = SimpleOpenEconomy().create_model()
        assert model.get_variable("FD").value.shape == (2, 2)
        assert model.get_variable("WF").value.shape == (2,)
        assert model.get_parameter("beta_VA").value.shape == (2, 2)

    def test_template_info_reports_the_sam(self):
        info = SimpleOpenEconomy().get_info()
        assert info["name"] == "SimpleOpenEconomy"
        assert info["goods"] == ["BRD", "MLK"]
        assert info["factors"] == ["CAP", "LAB"]
        assert info["sigma"] == 2.0 and info["psi"] == 2.0
        assert len(info["blocks"]) == 11

    def test_template_repr(self):
        assert repr(SimpleOpenEconomy()) == "SimpleOpenEconomy(2 goods, 2 factors)"
