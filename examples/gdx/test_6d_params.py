"""Test script to analyze 6D parameter structure in GDX files."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def test_6d_reading():
    """Test reading 6D parameters."""
    gdx_path = str(Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "test_6d.gdx")

    print("=" * 80)
    print("TESTING 6D PARAMETER READING")
    print("=" * 80)

    # Read GDX file
    data = read_gdx(gdx_path)

    # Test 6D sparse parameter
    print("\n6D SPARSE PARAMETER (p6d_sparse):")
    print("-" * 80)
    try:
        p6d_sparse = read_parameter_values(data, 'p6d_sparse')
        print("✅ Successfully read parameter")
        print(f"   Values read: {len(p6d_sparse)}")
        print("\nExpected 8 values:")
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

        correct = 0
        for indices, expected_val in expected.items():
            actual_val = p6d_sparse[indices]
            status = "✅" if abs(actual_val - expected_val) < 0.001 else "❌"
            print(f"   {status} {indices}: expected={expected_val:.3f}, got={actual_val:.3f}")
            if abs(actual_val - expected_val) < 0.001:
                correct += 1

        print(f"\n{'✅' if correct == 8 else '❌'} {correct}/8 values correct")

    except Exception as e:
        print(f"❌ Error reading parameter: {e}")
        import traceback
        traceback.print_exc()

    # Test 6D dense parameter
    print("\n\n6D DENSE PARAMETER (p6d_dense):")
    print("-" * 80)
    try:
        p6d_dense = read_parameter_values(data, 'p6d_dense')
        print("✅ Successfully read parameter")
        print(f"   Values read: {len(p6d_dense)}")
        print("   Expected: 64 values (2^6 full hypercube)")

        # Check a few sample values
        samples = [
            (('i1','j1','k1','l1','m1','n1'), 111111),
            (('i2','j1','k1','l1','m1','n1'), 211111),
            (('i1','j2','k1','l1','m1','n1'), 121111),
            (('i2','j2','k2','l2','m2','n2'), 222222),
        ]

        print("\nSample values:")
        correct_samples = 0
        for indices, expected_val in samples:
            actual_val = p6d_dense[indices]
            status = "✅" if abs(actual_val - expected_val) < 0.1 else "❌"
            print(f"   {status} {indices}: expected={expected_val}, got={actual_val:.0f}")
            if abs(actual_val - expected_val) < 0.1:
                correct_samples += 1

        print(f"\n{'✅' if correct_samples == len(samples) else '❌'} {correct_samples}/{len(samples)} samples correct")
        print(f"{'✅' if len(p6d_dense) == 64 else '❌'} Count: {len(p6d_dense)}/64 values")

    except Exception as e:
        print(f"❌ Error reading parameter: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_6d_reading()
