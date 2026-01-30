"""
equilibria/tests/babel/gdx/test_symbols.py - Tests for GDX symbol models.

Unit tests for equilibria.babel.gdx.symbols module.

Tests cover:
- SymbolType enum
- SymbolBase model (base class)
- Set model
- Parameter model
- Variable model
- Equation model
- Validation and serialization
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from equilibria.babel.gdx.symbols import (
    Equation,
    Parameter,
    Set,
    SymbolBase,
    SymbolType,
    Variable,
)


class TestSymbolType:
    """Tests for SymbolType enum."""

    def test_symbol_type_values(self) -> None:
        """Should have correct string values."""
        assert SymbolType.set.value == "set"
        assert SymbolType.parameter.value == "parameter"
        assert SymbolType.variable.value == "variable"
        assert SymbolType.equation.value == "equation"
        assert SymbolType.alias.value == "alias"

    def test_symbol_type_from_string(self) -> None:
        """Should create from string value."""
        assert SymbolType("set") == SymbolType.set
        assert SymbolType("parameter") == SymbolType.parameter

    def test_symbol_type_is_str_enum(self) -> None:
        """SymbolType should be usable as string."""
        assert SymbolType.set == "set"
        assert (
            SymbolType.set.value == "parameter"
            or SymbolType.parameter.value == "parameter"
        )


class TestSymbolBase:
    """Tests for SymbolBase model."""

    def test_create_minimal_symbol(self) -> None:
        """Should create symbol with minimal required fields."""
        sym: SymbolBase = SymbolBase(name="i", sym_type=SymbolType.set)

        assert sym.name == "i"
        assert sym.sym_type == "set"  # use_enum_values=True
        assert sym.dimensions == 0
        assert sym.description == ""
        assert sym.domain == []

    def test_create_full_symbol(self) -> None:
        """Should create symbol with all fields."""
        sym: SymbolBase = SymbolBase(
            name="price",
            sym_type=SymbolType.parameter,
            dimensions=2,
            description="Market prices",
            domain=["i", "j"],
        )

        assert sym.name == "price"
        assert sym.sym_type == "parameter"
        assert sym.dimensions == 2
        assert sym.description == "Market prices"
        assert sym.domain == ["i", "j"]

    def test_dimension_validation_min(self) -> None:
        """Dimensions should be >= 0."""
        with pytest.raises(ValidationError):
            SymbolBase(name="x", sym_type=SymbolType.set, dimensions=-1)

    def test_dimension_validation_max(self) -> None:
        """Dimensions should be <= 20."""
        with pytest.raises(ValidationError):
            SymbolBase(name="x", sym_type=SymbolType.set, dimensions=21)

    def test_serialization_to_dict(self) -> None:
        """Should serialize to dictionary."""
        sym: SymbolBase = SymbolBase(name="i", sym_type=SymbolType.set)
        data: dict = sym.model_dump()

        assert data["name"] == "i"
        assert data["sym_type"] == "set"
        assert "dimensions" in data
        assert "description" in data
        assert "domain" in data

    def test_create_from_string_type(self) -> None:
        """Should accept string for sym_type."""
        sym: SymbolBase = SymbolBase(name="i", sym_type="set")
        assert sym.sym_type == "set"


class TestSet:
    """Tests for Set model."""

    def test_create_empty_set(self) -> None:
        """Should create set with no elements."""
        s: Set = Set(name="i")

        assert s.name == "i"
        assert s.sym_type == "set"
        assert s.elements == []

    def test_create_set_with_elements(self) -> None:
        """Should create set with elements."""
        s: Set = Set(
            name="i",
            dimensions=1,
            description="Industries",
            domain=["*"],
            elements=[["agr"], ["mfg"], ["srv"]],
        )

        assert s.name == "i"
        assert s.sym_type == "set"
        assert s.dimensions == 1
        assert len(s.elements) == 3
        assert s.elements[0] == ["agr"]
        assert s.elements[2] == ["srv"]

    def test_set_default_type(self) -> None:
        """Set should default to SymbolType.set."""
        s: Set = Set(name="i")
        assert s.sym_type == "set"

    def test_multidimensional_set(self) -> None:
        """Should handle multi-dimensional sets."""
        s: Set = Set(
            name="map_ij",
            dimensions=2,
            domain=["i", "j"],
            elements=[["agr", "food"], ["mfg", "goods"]],
        )

        assert s.dimensions == 2
        assert len(s.elements) == 2
        assert s.elements[0] == ["agr", "food"]


class TestParameter:
    """Tests for Parameter model."""

    def test_create_empty_parameter(self) -> None:
        """Should create parameter with no records."""
        p: Parameter = Parameter(name="price")

        assert p.name == "price"
        assert p.sym_type == "parameter"
        assert p.records == []

    def test_create_parameter_with_records(self) -> None:
        """Should create parameter with records."""
        p: Parameter = Parameter(
            name="price",
            dimensions=1,
            description="Commodity prices",
            domain=["i"],
            records=[(["agr"], 1.0), (["mfg"], 1.5), (["srv"], 2.0)],
        )

        assert p.name == "price"
        assert p.sym_type == "parameter"
        assert len(p.records) == 3
        assert p.records[0] == (["agr"], 1.0)
        assert p.records[1][1] == 1.5

    def test_parameter_default_type(self) -> None:
        """Parameter should default to SymbolType.parameter."""
        p: Parameter = Parameter(name="x")
        assert p.sym_type == "parameter"

    def test_multidimensional_parameter(self) -> None:
        """Should handle multi-dimensional parameters."""
        p: Parameter = Parameter(
            name="sam",
            dimensions=2,
            domain=["i", "j"],
            records=[
                (["agr", "food"], 100.0),
                (["mfg", "goods"], 200.0),
            ],
        )

        assert p.dimensions == 2
        assert len(p.records) == 2
        assert p.records[0][0] == ["agr", "food"]
        assert p.records[0][1] == 100.0


class TestVariable:
    """Tests for Variable model."""

    def test_create_empty_variable(self) -> None:
        """Should create variable with no records."""
        v: Variable = Variable(name="X")

        assert v.name == "X"
        assert v.sym_type == "variable"
        assert v.records == []

    def test_create_variable_with_records(self) -> None:
        """Should create variable with 5-tuple records."""
        v: Variable = Variable(
            name="X",
            dimensions=1,
            description="Output",
            domain=["j"],
            records=[
                (["agr"], (100.0, 0.0, 0.0, float("inf"), 1.0)),
                (["mfg"], (150.0, 0.5, 0.0, float("inf"), 1.0)),
            ],
        )

        assert v.name == "X"
        assert v.sym_type == "variable"
        assert len(v.records) == 2

        # Check first record structure
        keys, values = v.records[0]
        assert keys == ["agr"]
        assert values[0] == 100.0  # level
        assert values[1] == 0.0  # marginal
        assert values[2] == 0.0  # lower
        assert values[3] == float("inf")  # upper
        assert values[4] == 1.0  # scale

    def test_variable_default_type(self) -> None:
        """Variable should default to SymbolType.variable."""
        v: Variable = Variable(name="Y")
        assert v.sym_type == "variable"


class TestEquation:
    """Tests for Equation model."""

    def test_create_empty_equation(self) -> None:
        """Should create equation with no records."""
        e: Equation = Equation(name="market_clearing")

        assert e.name == "market_clearing"
        assert e.sym_type == "equation"
        assert e.records == []

    def test_create_equation_with_records(self) -> None:
        """Should create equation with 5-tuple records."""
        e: Equation = Equation(
            name="market",
            dimensions=1,
            description="Market clearing",
            domain=["i"],
            records=[
                (["agr"], (0.0, 1.5, 0.0, 0.0, 1.0)),
                (["mfg"], (0.0, 2.0, 0.0, 0.0, 1.0)),
            ],
        )

        assert e.name == "market"
        assert e.sym_type == "equation"
        assert len(e.records) == 2

        # Check record values
        keys, values = e.records[0]
        assert keys == ["agr"]
        assert values[1] == 1.5  # marginal (shadow price)

    def test_equation_default_type(self) -> None:
        """Equation should default to SymbolType.equation."""
        e: Equation = Equation(name="eq1")
        assert e.sym_type == "equation"


class TestSymbolSerialization:
    """Tests for symbol serialization/deserialization."""

    def test_set_to_json_and_back(self) -> None:
        """Should serialize and deserialize Set."""
        original: Set = Set(
            name="i",
            dimensions=1,
            description="Industries",
            elements=[["agr"], ["mfg"]],
        )

        json_str: str = original.model_dump_json()
        restored: Set = Set.model_validate_json(json_str)

        assert restored.name == original.name
        assert restored.elements == original.elements

    def test_parameter_to_dict_and_back(self) -> None:
        """Should serialize and deserialize Parameter."""
        original: Parameter = Parameter(
            name="price",
            dimensions=1,
            records=[(["agr"], 1.0), (["mfg"], 2.0)],
        )

        data: dict = original.model_dump()
        restored: Parameter = Parameter.model_validate(data)

        assert restored.name == original.name
        assert restored.records == original.records

    def test_variable_with_infinity(self) -> None:
        """Should handle infinity values in serialization."""
        v: Variable = Variable(
            name="X",
            records=[(["a"], (10.0, 0.0, 0.0, float("inf"), 1.0))],
        )

        data: dict = v.model_dump()
        assert data["records"][0][1][3] == float("inf")
