"""
Tests for GDX writer module.

Tests for equilibria.babel.gdx.writer module covering:
- Basic parameter writing
- Set writing
- 2D parameter writing
- Variable/equation writing
- Round-trip read/write consistency
"""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import read_gdx
from equilibria.babel.gdx.symbols import Equation, Parameter, Set, Variable
from equilibria.babel.gdx.writer import write_gdx


class TestWriteSimpleParameter:
    """Tests for writing simple 1D parameters."""

    def test_write_1d_parameter(self, tmp_path: Path) -> None:
        """Should write a 1D parameter to GDX."""
        # Create parameter
        param = Parameter(
            name="price",
            sym_type="parameter",
            dimensions=1,
            description="Prices",
            domain=["i"],
            records=[(["agr"], 1.5), (["mfg"], 2.0)]
        )

        # Write to file
        output_file = tmp_path / "test.gdx"
        write_gdx(output_file, [param])

        # Verify file exists
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_roundtrip_parameter(self, tmp_path: Path) -> None:
        """Should read back what was written."""
        # Create parameter
        param = Parameter(
            name="X",
            sym_type="parameter",
            dimensions=1,
            records=[(["a"], 10.0), (["b"], 20.0), (["c"], 30.0)]
        )

        # Write
        output_file = tmp_path / "roundtrip.gdx"
        write_gdx(output_file, [param])

        # Read back
        gdx_data = read_gdx(output_file)

        # Verify
        assert len(gdx_data["symbols"]) == 1
        assert gdx_data["symbols"][0]["name"] == "X"
        assert gdx_data["symbols"][0]["type_name"] == "parameter"
        assert set(gdx_data["elements"]) == {"a", "b", "c"}


class TestWriteSet:
    """Tests for writing sets."""

    def test_write_simple_set(self, tmp_path: Path) -> None:
        """Should write a 1D set."""
        s = Set(
            name="i",
            sym_type="set",
            dimensions=1,
            description="Industries",
            elements=[["agr"], ["mfg"], ["srv"]]
        )

        output_file = tmp_path / "test_set.gdx"
        write_gdx(output_file, [s])

        assert output_file.exists()

    def test_roundtrip_set(self, tmp_path: Path) -> None:
        """Should preserve set elements in roundtrip."""
        s = Set(
            name="regions",
            sym_type="set",
            dimensions=1,
            elements=[["north"], ["south"], ["east"], ["west"]]
        )

        output_file = tmp_path / "test_set.gdx"
        write_gdx(output_file, [s])

        # Read back
        gdx_data = read_gdx(output_file)

        # Verify
        assert len(gdx_data["symbols"]) == 1
        assert gdx_data["symbols"][0]["type_name"] == "set"
        assert set(gdx_data["elements"]) == {"north", "south", "east", "west"}


class TestWrite2DParameter:
    """Tests for writing 2D parameters."""

    def test_write_matrix(self, tmp_path: Path) -> None:
        """Should write a 2D parameter (matrix)."""
        param = Parameter(
            name="matrix",
            sym_type="parameter",
            dimensions=2,
            records=[
                (["a", "x"], 1.0),
                (["a", "y"], 2.0),
                (["b", "x"], 3.0),
                (["b", "y"], 4.0),
            ]
        )

        output_file = tmp_path / "matrix.gdx"
        write_gdx(output_file, [param])

        assert output_file.exists()

        # Read back
        gdx_data = read_gdx(output_file)
        assert gdx_data["symbols"][0]["dimension"] == 2


class TestWriteVariable:
    """Tests for writing variables."""

    def test_write_variable_with_attributes(self, tmp_path: Path) -> None:
        """Should write variable with all 5 attributes."""
        var = Variable(
            name="X",
            sym_type="variable",
            dimensions=1,
            records=[
                (["agr"], (100.0, 0.0, 0.0, float("inf"), 1.0)),
                (["mfg"], (200.0, 0.5, 0.0, float("inf"), 1.0)),
            ]
        )

        output_file = tmp_path / "var.gdx"
        write_gdx(output_file, [var])

        assert output_file.exists()

        # Read back
        gdx_data = read_gdx(output_file)
        assert gdx_data["symbols"][0]["type_name"] == "variable"


class TestWriteEquation:
    """Tests for writing equations."""

    def test_write_equation(self, tmp_path: Path) -> None:
        """Should write equation with attributes."""
        eq = Equation(
            name="balance",
            sym_type="equation",
            dimensions=1,
            records=[
                (["agr"], (0.0, 1.5, 0.0, 0.0, 1.0)),
                (["mfg"], (0.0, 2.0, 0.0, 0.0, 1.0)),
            ]
        )

        output_file = tmp_path / "eq.gdx"
        write_gdx(output_file, [eq])

        assert output_file.exists()

        # Read back
        gdx_data = read_gdx(output_file)
        assert gdx_data["symbols"][0]["type_name"] == "equation"


class TestWriteMultipleSymbols:
    """Tests for writing multiple symbols."""

    def test_write_set_and_parameter(self, tmp_path: Path) -> None:
        """Should write multiple symbols to one file."""
        s = Set(
            name="i",
            sym_type="set",
            dimensions=1,
            elements=[["a"], ["b"], ["c"]]
        )

        p = Parameter(
            name="X",
            sym_type="parameter",
            dimensions=1,
            records=[(["a"], 1.0), (["b"], 2.0), (["c"], 3.0)]
        )

        output_file = tmp_path / "multi.gdx"
        write_gdx(output_file, [s, p])

        # Read back
        gdx_data = read_gdx(output_file)

        assert len(gdx_data["symbols"]) == 2
        assert gdx_data["symbols"][0]["name"] == "i"
        assert gdx_data["symbols"][1]["name"] == "X"


class TestWriteEdgeCases:
    """Tests for edge cases and error handling."""

    def test_write_empty_parameter(self, tmp_path: Path) -> None:
        """Should handle parameter with no records."""
        p = Parameter(
            name="empty",
            sym_type="parameter",
            dimensions=1,
            records=[]
        )

        output_file = tmp_path / "empty.gdx"
        write_gdx(output_file, [p])

        assert output_file.exists()

    def test_unsupported_version_raises(self, tmp_path: Path) -> None:
        """Should raise error for unsupported GDX version."""
        p = Parameter(name="X", sym_type="parameter", dimensions=1, records=[])
        output_file = tmp_path / "test.gdx"

        with pytest.raises(ValueError, match="Only GDX version 7"):
            write_gdx(output_file, [p], version=6)

    def test_write_to_nonexistent_directory(self, tmp_path: Path) -> None:
        """Should create directory if needed."""
        p = Parameter(
            name="X",
            sym_type="parameter",
            dimensions=1,
            records=[(["a"], 1.0)]
        )

        # Write to non-existent subdirectory
        output_file = tmp_path / "subdir" / "test.gdx"
        output_file.parent.mkdir(exist_ok=True)

        write_gdx(output_file, [p])
        assert output_file.exists()


class TestWriteSpecialValues:
    """Tests for special values (inf, nan, etc.)."""

    def test_write_infinity(self, tmp_path: Path) -> None:
        """Should handle infinity values."""
        p = Parameter(
            name="big",
            sym_type="parameter",
            dimensions=1,
            records=[
                (["a"], float("inf")),
                (["b"], float("-inf")),
            ]
        )

        output_file = tmp_path / "inf.gdx"
        write_gdx(output_file, [p])

        assert output_file.exists()

    def test_write_zero(self, tmp_path: Path) -> None:
        """Should handle zero values."""
        p = Parameter(
            name="zeros",
            sym_type="parameter",
            dimensions=1,
            records=[(["a"], 0.0), (["b"], 0.0)]
        )

        output_file = tmp_path / "zeros.gdx"
        write_gdx(output_file, [p])

        assert output_file.exists()
