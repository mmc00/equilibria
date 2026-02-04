"""Tests for core data structures.

Tests for Set, Parameter, Variable, and Equation classes.
"""

import numpy as np
import pytest

from equilibria.core import Parameter, Set, SetManager, Variable


class TestSet:
    """Tests for Set class."""

    def test_set_creation(self):
        """Test basic set creation."""
        s = Set(
            name="J",
            elements=("agr", "mfg", "svc"),
            description="Sectors",
        )
        assert s.name == "J"
        assert s.elements == ("agr", "mfg", "svc")
        assert s.description == "Sectors"
        assert s.domain is None

    def test_set_length(self):
        """Test set length."""
        s = Set(name="J", elements=("a", "b", "c"))
        assert len(s) == 3

    def test_set_iteration(self):
        """Test set element iteration."""
        s = Set(name="J", elements=("a", "b", "c"))
        elements = list(s.iter_elements())
        assert elements == ["a", "b", "c"]

    def test_set_contains(self):
        """Test element membership."""
        s = Set(name="J", elements=("a", "b", "c"))
        assert "a" in s
        assert "d" not in s

    def test_set_indexing(self):
        """Test element indexing."""
        s = Set(name="J", elements=("a", "b", "c"))
        assert s[0] == "a"
        assert s[1] == "b"
        assert s[2] == "c"

    def test_set_index_method(self):
        """Test index lookup."""
        s = Set(name="J", elements=("a", "b", "c"))
        assert s.index("a") == 0
        assert s.index("b") == 1
        assert s.index("c") == 2

        with pytest.raises(ValueError):
            s.index("d")

    def test_set_to_list(self):
        """Test conversion to list."""
        s = Set(name="J", elements=("a", "b", "c"))
        assert s.to_list() == ["a", "b", "c"]

    def test_set_immutable(self):
        """Test that sets are immutable."""
        s = Set(name="J", elements=("a", "b"))
        with pytest.raises(Exception):
            s.elements = ("c", "d")


class TestSetManager:
    """Tests for SetManager class."""

    def test_add_set(self):
        """Test adding sets."""
        manager = SetManager()
        s = Set(name="J", elements=("a", "b"))
        manager.add(s)
        assert "J" in manager

    def test_add_duplicate_raises(self):
        """Test that adding duplicate raises error."""
        manager = SetManager()
        s = Set(name="J", elements=("a", "b"))
        manager.add(s)

        with pytest.raises(ValueError):
            manager.add(Set(name="J", elements=("c", "d")))

    def test_get_set(self):
        """Test retrieving sets."""
        manager = SetManager()
        s = Set(name="J", elements=("a", "b"))
        manager.add(s)

        retrieved = manager.get("J")
        assert retrieved.name == "J"
        assert retrieved.elements == ("a", "b")

    def test_get_missing_raises(self):
        """Test that getting missing set raises error."""
        manager = SetManager()
        with pytest.raises(KeyError):
            manager.get("J")

    def test_subset_validation(self):
        """Test subset domain validation."""
        manager = SetManager()
        parent = Set(name="J", elements=("a", "b", "c"))
        manager.add(parent)

        # Valid subset
        subset = Set(name="J_SUB", elements=("a", "b"), domain="J")
        manager.add(subset)
        assert "J_SUB" in manager

        # Invalid subset (element not in domain)
        with pytest.raises(ValueError):
            invalid = Set(name="J_BAD", elements=("x",), domain="J")
            manager.add(invalid)

    def test_cartesian_product(self):
        """Test cartesian product generation."""
        manager = SetManager()
        manager.add(Set(name="J", elements=("a", "b")))
        manager.add(Set(name="I", elements=("x", "y")))

        product = list(manager.product("J", "I"))
        expected = [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")]
        assert product == expected

    def test_list_sets(self):
        """Test listing all sets."""
        manager = SetManager()
        manager.add(Set(name="J", elements=("a",)))
        manager.add(Set(name="I", elements=("x",)))

        sets = manager.list_sets()
        assert sorted(sets) == ["I", "J"]

    def test_summary(self):
        """Test summary generation."""
        manager = SetManager()
        manager.add(Set(name="J", elements=("a", "b"), description="Sectors"))

        summary = manager.summary()
        assert summary["total_sets"] == 1
        assert "J" in summary["sets"]


class TestParameter:
    """Tests for Parameter class."""

    def test_scalar_parameter(self):
        """Test scalar parameter creation."""
        p = Parameter(
            name="sigma",
            value=np.array([0.8]),
            description="Elasticity",
        )
        assert p.name == "sigma"
        assert p.shape() == (1,)
        assert p.domains == ()

    def test_1d_parameter(self):
        """Test 1D parameter creation."""
        p = Parameter(
            name="sigma_VA",
            value=np.array([0.8, 0.9, 0.7]),
            domains=("J",),
            description="By sector",
        )
        assert p.shape() == (3,)
        assert p.domains == ("J",)

    def test_2d_parameter(self):
        """Test 2D parameter creation."""
        p = Parameter(
            name="a_io",
            value=np.array([[0.1, 0.2], [0.3, 0.4]]),
            domains=("I", "J"),
            description="IO coefficients",
        )
        assert p.shape() == (2, 2)
        assert p.domains == ("I", "J")

    def test_parameter_get_value(self):
        """Test getting parameter values."""
        p = Parameter(
            name="sigma",
            value=np.array([0.8, 0.9, 1.0]),
            domains=("J",),
        )
        assert p.get_value(0) == 0.8
        assert p.get_value(1) == 0.9
        assert p.get_value(2) == 1.0

    def test_parameter_set_value(self):
        """Test setting parameter values."""
        p = Parameter(
            name="sigma",
            value=np.array([0.8, 0.9, 1.0]),
            domains=("J",),
        )
        p.set_value(0.85, 0)
        assert p.get_value(0) == 0.85

    def test_parameter_to_dict(self):
        """Test parameter serialization."""
        p = Parameter(
            name="sigma",
            value=np.array([0.8, 0.9]),
            domains=("J",),
            description="Elasticity",
        )
        d = p.to_dict()
        assert d["name"] == "sigma"
        assert d["value"] == [0.8, 0.9]
        assert d["domains"] == ("J",)


class TestVariable:
    """Tests for Variable class."""

    def test_scalar_variable(self):
        """Test scalar variable creation."""
        v = Variable(
            name="Y",
            value=np.array([100.0]),
            lower=0.0,
            description="Income",
        )
        assert v.name == "Y"
        assert v.shape() == (1,)
        assert v.lower == 0.0

    def test_1d_variable(self):
        """Test 1D variable creation."""
        v = Variable(
            name="VA",
            value=np.array([100.0, 150.0, 200.0]),
            domains=("J",),
            lower=0.0,
            description="Value added",
        )
        assert v.shape() == (3,)
        assert v.domains == ("J",)

    def test_variable_bounds(self):
        """Test variable bounds."""
        v = Variable(
            name="P",
            value=np.array([1.0]),
            lower=0.0,
            upper=10.0,
        )
        assert v.lower == 0.0
        assert v.upper == 10.0

    def test_variable_fix_unfix(self):
        """Test fixing and unfixing variables."""
        v = Variable(name="P", value=np.array([1.0]))

        assert not v.is_fixed()

        v.fix(5.0)
        assert v.is_fixed()
        assert v.value[0] == 5.0

        v.unfix()
        assert not v.is_fixed()

    def test_variable_to_dict(self):
        """Test variable serialization."""
        v = Variable(
            name="VA",
            value=np.array([100.0, 150.0]),
            domains=("J",),
            lower=0.0,
            description="Value added",
        )
        d = v.to_dict()
        assert d["name"] == "VA"
        assert d["value"] == [100.0, 150.0]
        assert d["fixed"] is False
