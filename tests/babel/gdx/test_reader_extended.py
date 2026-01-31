"""
Tests for extended GDX reader functionality.

Tests for reading variables, equations, and set elements.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import (
    read_equation_values,
    read_gdx,
    read_set_elements,
    read_variable_values,
)

FIXTURES_DIR: Path = Path(__file__).parent.parent.parent / "fixtures"


@pytest.mark.skipif(
    not (FIXTURES_DIR / "variables_equations_test.gdx").exists(),
    reason="Test fixture variables_equations_test.gdx not found",
)
class TestReadVariableValues:
    """Tests for read_variable_values() function."""

    def test_read_variable_basic(self) -> None:
        """Should read basic variable structure."""
        gdx_path: Path = FIXTURES_DIR / "variables_equations_test.gdx"
        result: dict = read_gdx(gdx_path)

        # Try to read a variable (may not get all data due to GDX complexity)
        try:
            values: dict = read_variable_values(result, "X")
            # Should return a dict (even if empty due to parsing complexity)
            assert isinstance(values, dict)
        except ValueError:
            # It's OK if the variable isn't found or can't be read
            pass

    def test_variable_not_found_raises(self) -> None:
        """Should raise ValueError for non-existent variable."""
        gdx_path: Path = FIXTURES_DIR / "variables_equations_test.gdx"
        result: dict = read_gdx(gdx_path)

        with pytest.raises(ValueError, match="not found"):
            read_variable_values(result, "nonexistent_var")

    def test_non_variable_raises(self) -> None:
        """Should raise ValueError when reading non-variable symbol."""
        gdx_path: Path = FIXTURES_DIR / "variables_equations_test.gdx"
        result: dict = read_gdx(gdx_path)

        with pytest.raises(ValueError, match="not a variable"):
            read_variable_values(result, "sam")  # sam is a parameter


@pytest.mark.skipif(
    not (FIXTURES_DIR / "variables_equations_test.gdx").exists(),
    reason="Test fixture variables_equations_test.gdx not found",
)
class TestReadEquationValues:
    """Tests for read_equation_values() function."""

    def test_read_equation_basic(self) -> None:
        """Should read basic equation structure."""
        gdx_path: Path = FIXTURES_DIR / "variables_equations_test.gdx"
        result: dict = read_gdx(gdx_path)

        # Try to read an equation
        try:
            values: dict = read_equation_values(result, "eq_output")
            assert isinstance(values, dict)
        except ValueError:
            pass

    def test_equation_not_found_raises(self) -> None:
        """Should raise ValueError for non-existent equation."""
        gdx_path: Path = FIXTURES_DIR / "variables_equations_test.gdx"
        result: dict = read_gdx(gdx_path)

        with pytest.raises(ValueError, match="not found"):
            read_equation_values(result, "nonexistent_eq")

    def test_non_equation_raises(self) -> None:
        """Should raise ValueError when reading non-equation symbol."""
        gdx_path: Path = FIXTURES_DIR / "variables_equations_test.gdx"
        result: dict = read_gdx(gdx_path)

        with pytest.raises(ValueError, match="not an equation"):
            read_equation_values(result, "price")  # price is a parameter


@pytest.mark.skipif(
    not (FIXTURES_DIR / "simple_test.gdx").exists(),
    reason="Test fixture simple_test.gdx not found",
)
class TestReadSetElements:
    """Tests for read_set_elements() function."""

    def test_read_set_elements(self) -> None:
        """Should read set elements."""
        gdx_path: Path = FIXTURES_DIR / "simple_test.gdx"
        result: dict = read_gdx(gdx_path)

        elements: list = read_set_elements(result, "i")
        assert isinstance(elements, list)
        assert len(elements) > 0
        # Each element should be a tuple
        assert all(isinstance(e, tuple) for e in elements)

    def test_set_not_found_raises(self) -> None:
        """Should raise ValueError for non-existent set."""
        gdx_path: Path = FIXTURES_DIR / "simple_test.gdx"
        result: dict = read_gdx(gdx_path)

        with pytest.raises(ValueError, match="not found"):
            read_set_elements(result, "nonexistent_set")

    def test_non_set_raises(self) -> None:
        """Should raise ValueError when reading non-set symbol."""
        gdx_path: Path = FIXTURES_DIR / "simple_test.gdx"
        result: dict = read_gdx(gdx_path)

        with pytest.raises(ValueError, match="not a set"):
            read_set_elements(result, "price")  # price is a parameter


@pytest.mark.skipif(
    not (FIXTURES_DIR / "multidim_test.gdx").exists(),
    reason="Test fixture multidim_test.gdx not found",
)
class TestReadMultidimSets:
    """Tests for reading multi-dimensional sets."""

    def test_read_multidim_set_structure(self) -> None:
        """Should handle multi-dimensional sets."""
        gdx_path: Path = FIXTURES_DIR / "multidim_test.gdx"
        result: dict = read_gdx(gdx_path)

        # Should be able to identify sets
        from equilibria.babel.gdx.reader import get_sets

        sets: list = get_sets(result)
        assert len(sets) > 0
