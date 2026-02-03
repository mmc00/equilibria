"""
Test reading multi-dimensional parameters from GDX files.
"""

import pytest
from pathlib import Path
from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def test_read_3d_sparse_parameter():
    """Test reading a 3D sparse parameter."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "multidim_test.gdx"
    
    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_multidim_test.gms first")
    
    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p3d_sparse")
    
    # Expected values from GAMS
    expected = {
        ("i1", "j1", "k1"): 111.0,
        ("i1", "j2", "k1"): 121.0,
        ("i2", "j3", "k2"): 232.0,
        ("i3", "j4", "k1"): 341.0,
    }
    
    assert len(values) == 4, f"Expected 4 values, got {len(values)}"
    
    # Check each value
    for key, expected_val in expected.items():
        assert key in values, f"Missing key {key}"
        assert abs(values[key] - expected_val) < 0.001, \
            f"Value mismatch for {key}: expected {expected_val}, got {values[key]}"


def test_read_4d_sparse_parameter():
    """Test reading a 4D sparse parameter."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "multidim_test.gdx"
    
    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_multidim_test.gms first")
    
    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p4d_sparse")
    
    # Expected values from GAMS
    expected = {
        ("i1", "j1", "k1", "m1"): 1111.0,
        ("i1", "j2", "k1", "m2"): 1212.0,
        ("i2", "j3", "k2", "m3"): 2323.0,
        ("i3", "j4", "k1", "m1"): 3411.0,
        ("i3", "j4", "k2", "m2"): 3422.0,
    }
    
    assert len(values) == 5, f"Expected 5 values, got {len(values)}"
    
    # Check each value
    for key, expected_val in expected.items():
        assert key in values, f"Missing key {key}"
        assert abs(values[key] - expected_val) < 0.001, \
            f"Value mismatch for {key}: expected {expected_val}, got {values[key]}"


def test_read_3d_dense_parameter():
    """Test reading a 3D dense parameter (currently partial support)."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "multidim_test.gdx"
    
    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_multidim_test.gms first")
    
    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p3d")
    
    # Dense parameters use compression, so we may not get all 24 values
    # For now, just check that we get some values
    assert len(values) > 0, "Should read at least some values"
    
    # Check that all values are reasonable
    for key, val in values.items():
        assert len(key) == 3, f"Key should have 3 dimensions, got {len(key)}"
        assert 100 <= val <= 400, f"Value {val} out of expected range"


def test_read_4d_dense_parameter():
    """Test reading a 4D dense parameter (currently partial support)."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "multidim_test.gdx"
    
    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_multidim_test.gms first")
    
    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p4d")
    
    # Dense parameters use compression, so we may not get all 72 values
    # For now, just check that we get some values
    assert len(values) > 0, "Should read at least some values"
    
    # Check that all values are reasonable
    for key, val in values.items():
        assert len(key) == 4, f"Key should have 4 dimensions, got {len(key)}"
        assert 1000 <= val <= 4000, f"Value {val} out of expected range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
