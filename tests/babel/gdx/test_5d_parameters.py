"""Test reading 5D parameters from GDX files."""

from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def test_read_5d_sparse_parameter():
    """Test reading a 5D sparse parameter."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "test_5d.gdx"

    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_5d_test.gms first")

    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p5d_sparse")

    # Expected values from GAMS
    expected = {
        ('i1','j1','k1','m1','n1'): 11111.0,
        ('i1','j2','k1','m1','n2'): 12112.0,
        ('i1','j2','k2','m2','n1'): 12221.0,
        ('i2','j3','k1','m2','n2'): 23122.0,
        ('i2','j3','k2','m1','n1'): 23211.0,
        ('i2','j3','k2','m2','n2'): 23222.0,
    }

    assert len(values) == 6, f"Expected 6 values, got {len(values)}"

    # Check each value
    for key, expected_val in expected.items():
        assert key in values, f"Missing key {key}"
        assert abs(values[key] - expected_val) < 0.001, \
            f"Value mismatch for {key}: expected {expected_val}, got {values[key]}"


def test_read_5d_dense_parameter():
    """Test reading a 5D dense parameter."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "test_5d.gdx"

    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_5d_test.gms first")

    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p5d_dense")

    # Expected: 2*3*2*2*2 = 48 values
    # Values follow pattern: ord(i)*10000 + ord(j)*1000 + ord(k)*100 + ord(m)*10 + ord(n)
    expected_count = 2 * 3 * 2 * 2 * 2

    assert len(values) == expected_count, \
        f"Expected {expected_count} values, got {len(values)}"

    # Check that all tuples have 5 dimensions
    for key in values.keys():
        assert len(key) == 5, f"Key should have 5 dimensions, got {len(key)}: {key}"

    # Spot check a few values
    test_cases = [
        (('i1', 'j1', 'k1', 'm1', 'n1'), 11111.0),
        (('i1', 'j2', 'k1', 'm1', 'n2'), 12112.0),
        (('i2', 'j3', 'k2', 'm2', 'n2'), 23222.0),
    ]

    for key, expected_val in test_cases:
        assert key in values, f"Missing key {key}"
        assert abs(values[key] - expected_val) < 0.001, \
            f"Value mismatch for {key}: expected {expected_val}, got {values[key]}"


def test_5d_parameter_slicing():
    """Test slicing operations on 5D parameters."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "test_5d.gdx"

    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_5d_test.gms first")

    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p5d_sparse")

    # Slice: fix first dimension to 'i2'
    slice_i2 = {
        (j, k, m, n): v
        for (i, j, k, m, n), v in values.items()
        if i == 'i2'
    }
    assert len(slice_i2) == 3, "Should have 3 values for i2"

    # Slice: fix last dimension to 'n1'
    slice_n1 = {
        (i, j, k, m): v
        for (i, j, k, m, n), v in values.items()
        if n == 'n1'
    }
    assert len(slice_n1) == 3, "Should have 3 values for n1"

    # Slice: fix middle dimensions
    slice_middle = {
        (i, n): v
        for (i, j, k, m, n), v in values.items()
        if j == 'j3' and k == 'k2'
    }
    assert len(slice_middle) == 2, "Should have 2 values for j3,k2"


def test_5d_parameter_aggregation():
    """Test aggregation operations on 5D parameters."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "test_5d.gdx"

    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_5d_test.gms first")

    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p5d_sparse")

    # Aggregate by first dimension
    by_i = {}
    for (i, j, k, m, n), v in values.items():
        by_i[i] = by_i.get(i, 0) + v

    assert len(by_i) == 2, "Should have 2 unique values for i"
    assert 'i1' in by_i and 'i2' in by_i

    # Aggregate to 2D matrix (i, n)
    matrix_i_n = {}
    for (i, j, k, m, n), v in values.items():
        key = (i, n)
        matrix_i_n[key] = matrix_i_n.get(key, 0) + v

    assert len(matrix_i_n) <= 4, "Should have at most 4 combinations"

    # All aggregated values should be positive
    for v in matrix_i_n.values():
        assert v > 0, "Aggregated values should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
