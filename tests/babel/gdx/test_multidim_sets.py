"""
Comprehensive tests for multidimensional sets in GDX reader.

Tests cover 2D, 3D, 4D sets with various patterns:
- Sparse sets
- Full/dense sets  
- Cartesian products
- Empty sets
- Sets with parameters
"""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import (
    get_sets,
    get_symbol,
    read_gdx,
    read_set_elements,
)


class Test2DSets:
    """Tests for 2D sets."""

    def test_2d_sparse_parameter_metadata(self) -> None:
        """Should read sparse 2D parameter metadata correctly."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_sparse.gdx"
        gdx_data = read_gdx(fixture)

        # Check symbols exist
        assert len(gdx_data["symbols"]) == 3

        # Check 1D sets
        i_sym = get_symbol(gdx_data, "i")
        assert i_sym is not None
        assert i_sym["dimension"] == 1
        assert i_sym["records"] == 3

        j_sym = get_symbol(gdx_data, "j")
        assert j_sym is not None
        assert j_sym["dimension"] == 1
        assert j_sym["records"] == 3

        # Check 2D parameter
        map_sym = get_symbol(gdx_data, "map")
        assert map_sym is not None
        assert map_sym["type_name"] == "parameter"
        assert map_sym["dimension"] == 2
        assert map_sym["records"] == 5  # 5 mappings

    def test_2d_full_set(self) -> None:
        """Should read 2D set with explicit pairs."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_full.gdx"
        gdx_data = read_gdx(fixture)

        rr_sym = get_symbol(gdx_data, "rr")
        assert rr_sym is not None
        assert rr_sym["dimension"] == 2
        assert rr_sym["records"] == 4

        elements = read_set_elements(gdx_data, "rr")
        assert len(elements) == 4

        elements_set = set(elements)
        expected = {
            ("north", "south"),
            ("north", "east"),
            ("south", "east"),
            ("east", "west"),
        }
        assert elements_set == expected

    def test_2d_cartesian_product(self) -> None:
        """Should read 2D set representing Cartesian product."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_cartesian.gdx"
        gdx_data = read_gdx(fixture)

        cart_sym = get_symbol(gdx_data, "cart")
        assert cart_sym is not None
        assert cart_sym["dimension"] == 2
        assert cart_sym["records"] == 6  # 3 x 2 = 6

        elements = read_set_elements(gdx_data, "cart")
        assert len(elements) == 6

        # Check all combinations exist
        elements_set = set(elements)
        expected = {
            ("a1", "b1"), ("a1", "b2"),
            ("a2", "b1"), ("a2", "b2"),
            ("a3", "b1"), ("a3", "b2"),
        }
        assert elements_set == expected

    def test_2d_empty_set(self) -> None:
        """Should handle empty 2D set."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_empty.gdx"
        gdx_data = read_gdx(fixture)

        empty_sym = get_symbol(gdx_data, "empty")
        assert empty_sym is not None
        assert empty_sym["dimension"] == 2
        assert empty_sym["records"] == 0

        elements = read_set_elements(gdx_data, "empty")
        assert elements == []

    def test_2d_set_with_parameter(self) -> None:
        """Should read 2D set when file also contains parameters."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_with_param.gdx"
        gdx_data = read_gdx(fixture)

        # Should have 4 symbols: src, dst, route, cost
        assert len(gdx_data["symbols"]) == 4

        route_sym = get_symbol(gdx_data, "route")
        assert route_sym is not None
        assert route_sym["type_name"] == "set"
        assert route_sym["dimension"] == 2
        assert route_sym["records"] == 4

        elements = read_set_elements(gdx_data, "route")
        assert len(elements) == 4

        elements_set = set(elements)
        expected = {
            ("s1", "d1"),
            ("s1", "d2"),
            ("s2", "d1"),
            ("s3", "d2"),
        }
        assert elements_set == expected

    def test_2d_set_preserves_order(self) -> None:
        """Should preserve insertion order of set elements."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_with_param.gdx"
        gdx_data = read_gdx(fixture)

        elements = read_set_elements(gdx_data, "route")

        # Elements should be in consistent order
        assert len(elements) == 4
        assert elements[0] == ("s1", "d1")


class Test3DSets:
    """Tests for 3D sets."""

    def test_3d_sparse_set(self) -> None:
        """Should read sparse 3D set correctly."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_3d.gdx"
        gdx_data = read_gdx(fixture)

        cube_sym = get_symbol(gdx_data, "cube")
        assert cube_sym is not None
        assert cube_sym["type_name"] == "set"
        assert cube_sym["dimension"] == 3
        assert cube_sym["records"] == 7

        elements = read_set_elements(gdx_data, "cube")

        # Should have 7 3-tuples
        assert len(elements) == 7

        for elem in elements:
            assert len(elem) == 3
            assert isinstance(elem, tuple)

        elements_set = set(elements)
        expected = {
            ("a", "x", "p"),
            ("a", "x", "q"),
            ("a", "y", "p"),
            ("b", "x", "p"),
            ("b", "y", "q"),
            ("c", "y", "p"),
            ("c", "y", "q"),
        }
        assert elements_set == expected

    def test_3d_set_dimensions(self) -> None:
        """Should correctly identify 3D set metadata."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_3d.gdx"
        gdx_data = read_gdx(fixture)

        # Check all 1D sets exist
        i3 = get_symbol(gdx_data, "i3")
        j3 = get_symbol(gdx_data, "j3")
        k3 = get_symbol(gdx_data, "k3")

        assert i3["dimension"] == 1
        assert j3["dimension"] == 1
        assert k3["dimension"] == 1

        # Check 3D set references them
        cube = get_symbol(gdx_data, "cube")
        assert cube["dimension"] == 3


class Test4DSets:
    """Tests for 4D sets (stress test for high dimensions)."""

    def test_4d_sparse_set(self) -> None:
        """Should read sparse 4D set."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_4d.gdx"
        gdx_data = read_gdx(fixture)

        hyper_sym = get_symbol(gdx_data, "hypercube")
        assert hyper_sym is not None
        assert hyper_sym["dimension"] == 4
        assert hyper_sym["records"] == 4

        elements = read_set_elements(gdx_data, "hypercube")

        # Should have 4 4-tuples
        assert len(elements) == 4

        for elem in elements:
            assert len(elem) == 4
            assert isinstance(elem, tuple)

        elements_set = set(elements)
        expected = {
            ("x1", "y1", "z1", "w1"),
            ("x1", "y2", "z1", "w2"),
            ("x2", "y1", "z2", "w1"),
            ("x2", "y2", "z2", "w2"),
        }
        assert elements_set == expected


class TestSetEdgeCases:
    """Tests for edge cases in multidimensional sets."""

    def test_read_nonexistent_set_raises(self) -> None:
        """Should raise error for non-existent set."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_with_param.gdx"
        gdx_data = read_gdx(fixture)

        with pytest.raises(ValueError, match="Symbol 'nonexistent' not found"):
            read_set_elements(gdx_data, "nonexistent")

    def test_read_parameter_as_set_raises(self) -> None:
        """Should raise error when trying to read parameter as set."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_with_param.gdx"
        gdx_data = read_gdx(fixture)

        with pytest.raises(ValueError, match="not a set"):
            read_set_elements(gdx_data, "cost")

    def test_get_all_sets_filters_correctly(self) -> None:
        """Should filter only sets when getting all sets."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_with_param.gdx"
        gdx_data = read_gdx(fixture)

        sets = get_sets(gdx_data)

        # Should have 2 sets: src, route (dst is an alias, cost is parameter)
        assert len(sets) == 2

        set_names = {s["name"] for s in sets}
        assert set_names == {"src", "route"}

        # All should be sets
        for s in sets:
            assert s["type_name"] == "set"

    def test_2d_set_element_types(self) -> None:
        """Should return tuples of strings for 2D sets."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_with_param.gdx"
        gdx_data = read_gdx(fixture)

        elements = read_set_elements(gdx_data, "route")

        for elem in elements:
            assert isinstance(elem, tuple)
            assert len(elem) == 2
            assert isinstance(elem[0], str)
            assert isinstance(elem[1], str)

    def test_set_with_different_dimensions(self) -> None:
        """Should handle multiple sets with different dimensions in same file."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_3d.gdx"
        gdx_data = read_gdx(fixture)

        # 1D sets
        i3_elem = read_set_elements(gdx_data, "i3")
        assert len(i3_elem) == 3
        assert all(len(e) == 1 for e in i3_elem)

        k3_elem = read_set_elements(gdx_data, "k3")
        assert len(k3_elem) == 2
        assert all(len(e) == 1 for e in k3_elem)

        # j3 is an alias, not a set - should raise error
        with pytest.raises(ValueError, match="not a set"):
            read_set_elements(gdx_data, "j3")

        k3_elem = read_set_elements(gdx_data, "k3")
        assert len(k3_elem) == 2
        assert all(len(e) == 1 for e in k3_elem)

        # 3D set
        cube_elem = read_set_elements(gdx_data, "cube")
        assert len(cube_elem) == 7
        assert all(len(e) == 3 for e in cube_elem)


class TestSetPerformance:
    """Performance and stress tests for set reading."""

    def test_multiple_set_reads_consistent(self) -> None:
        """Should return same results when reading set multiple times."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_2d_with_param.gdx"
        gdx_data = read_gdx(fixture)

        elements1 = read_set_elements(gdx_data, "route")
        elements2 = read_set_elements(gdx_data, "route")
        elements3 = read_set_elements(gdx_data, "route")

        assert elements1 == elements2
        assert elements2 == elements3

    def test_read_all_sets_in_file(self) -> None:
        """Should be able to read all sets from a file."""
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "set_3d.gdx"
        gdx_data = read_gdx(fixture)

        sets = get_sets(gdx_data)

        # Should be able to read elements from all sets
        for s in sets:
            elements = read_set_elements(gdx_data, s["name"])
            assert isinstance(elements, list)
            assert len(elements) == s["records"]
