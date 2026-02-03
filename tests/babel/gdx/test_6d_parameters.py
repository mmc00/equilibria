"""Test reading 6D parameters from GDX files."""

import pytest
from pathlib import Path
from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def test_read_6d_sparse_parameter():
    """Test reading a 6D sparse parameter."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "test_6d.gdx"
    
    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_6d_test.gms first")
    
    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p6d_sparse")
    
    # Expected values from GAMS (8 corners of 6D hypercube)
    expected = {
        ('i1','j1','k1','l1','m1','n1'): 111.111,
        ('i2','j1','k1','l1','m1','n1'): 211.111,
        ('i1','j2','k1','l1','m1','n1'): 121.111,
        ('i1','j1','k2','l1','m1','n1'): 112.111,
        ('i1','j1','k1','l2','m1','n1'): 111.211,
        ('i1','j1','k1','l1','m2','n1'): 111.121,
        ('i1','j1','k1','l1','m1','n2'): 111.112,
        ('i2','j2','k2','l2','m2','n2'): 222.222,
    }
    
    assert len(values) == 8, f"Expected 8 values, got {len(values)}"
    
    # Check each value
    for key, expected_val in expected.items():
        assert key in values, f"Missing key {key}"
        assert abs(values[key] - expected_val) < 0.001, \
            f"Value mismatch for {key}: expected {expected_val}, got {values[key]}"


def test_read_6d_dense_parameter():
    """Test reading a 6D dense parameter."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "test_6d.gdx"
    
    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_6d_test.gms first")
    
    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p6d_dense")
    
    # Expected: 2*2*2*2*2*2 = 64 values (full 6D hypercube)
    # Values follow pattern: ord(i)*100000 + ord(j)*10000 + ord(k)*1000 + ord(l)*100 + ord(m)*10 + ord(n)
    expected_count = 2 * 2 * 2 * 2 * 2 * 2
    
    assert len(values) == expected_count, \
        f"Expected {expected_count} values, got {len(values)}"
    
    # Verify specific values
    assert values[('i1','j1','k1','l1','m1','n1')] == 111111
    assert values[('i2','j1','k1','l1','m1','n1')] == 211111
    assert values[('i1','j2','k1','l1','m1','n1')] == 121111
    assert values[('i1','j1','k2','l1','m1','n1')] == 112111
    assert values[('i1','j1','k1','l2','m1','n1')] == 111211
    assert values[('i1','j1','k1','l1','m2','n1')] == 111121
    assert values[('i1','j1','k1','l1','m1','n2')] == 111112
    assert values[('i2','j2','k2','l2','m2','n2')] == 222222


def test_6d_parameter_slicing():
    """Test slicing operations on 6D parameters."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "test_6d.gdx"
    
    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_6d_test.gms first")
    
    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p6d_dense")
    
    # Extract 5D slice for i1
    slice_i1 = {k[1:]: v for k, v in values.items() if k[0] == 'i1'}
    assert len(slice_i1) == 32  # Half of the hypercube
    
    # Extract 4D slice for i1, j1
    slice_i1_j1 = {k[2:]: v for k, v in values.items() 
                   if k[0] == 'i1' and k[1] == 'j1'}
    assert len(slice_i1_j1) == 16  # Quarter of the hypercube
    
    # Extract 3D slice for i1, j1, k1
    slice_i1_j1_k1 = {k[3:]: v for k, v in values.items() 
                       if k[0] == 'i1' and k[1] == 'j1' and k[2] == 'k1'}
    assert len(slice_i1_j1_k1) == 8  # Eighth of the hypercube


def test_6d_parameter_aggregation():
    """Test aggregation operations on 6D parameters."""
    gdx_file = Path(__file__).parent.parent.parent / "fixtures" / "test_6d.gdx"
    
    if not gdx_file.exists():
        pytest.skip("Test file not found - run generate_6d_test.gms first")
    
    data = read_gdx(gdx_file)
    values = read_parameter_values(data, "p6d_dense")
    
    # Sum over last dimension (n)
    sum_over_n = {}
    for (i, j, k, l, m, n), val in values.items():
        key = (i, j, k, l, m)
        sum_over_n[key] = sum_over_n.get(key, 0) + val
    
    assert len(sum_over_n) == 32  # 2^5 combinations
    
    # Sum over last two dimensions (m, n)
    sum_over_mn = {}
    for (i, j, k, l, m, n), val in values.items():
        key = (i, j, k, l)
        sum_over_mn[key] = sum_over_mn.get(key, 0) + val
    
    assert len(sum_over_mn) == 16  # 2^4 combinations
