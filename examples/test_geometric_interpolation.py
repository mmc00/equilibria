"""
Test geometric interpolation in GDX reader.

This script tests the enhanced GDX reader that now supports both
arithmetic and geometric sequence interpolation.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


def test_geometric_sequence():
    """Test reading geometric sequence from GDX."""
    print("=" * 80)
    print("TEST: Reading geometric sequence from GDX")
    print("=" * 80)

    gdx_file = Path(__file__).parent.parent / "tests" / "fixtures" / "test_geometric.gdx"

    if not gdx_file.exists():
        print(f"❌ Test file not found: {gdx_file}")
        return False

    print(f"\n📁 Reading: {gdx_file.name}")

    try:
        # Read the GDX file first
        gdx_data = read_gdx(str(gdx_file))

        # Then read the parameter values
        data = read_parameter_values(gdx_data, "geom")

        print("\n✓ Successfully read parameter 'geom'")
        print(f"  Records read: {len(data)}")

        # Expected values for geometric sequence (1,2,4,8,16,32,64,128,256,512)
        expected = {
            ("t1",): 1.0,
            ("t2",): 2.0,
            ("t3",): 4.0,
            ("t4",): 8.0,
            ("t5",): 16.0,
            ("t6",): 32.0,
            ("t7",): 64.0,
            ("t8",): 128.0,
            ("t9",): 256.0,
            ("t10",): 512.0,
        }

        # Compare values
        print("\n" + "=" * 80)
        print("VALIDATION:")
        print("=" * 80)

        all_correct = True
        max_error = 0.0

        for key, expected_val in expected.items():
            if key in data:
                actual_val = data[key]
                error = abs(actual_val - expected_val)
                rel_error = error / expected_val if expected_val != 0 else 0.0

                status = "✓" if error < 0.01 else "✗"
                if error >= 0.01:
                    all_correct = False

                max_error = max(max_error, rel_error)

                print(f"{status} {key[0]:5s}: expected={expected_val:10.2f}, "
                      f"actual={actual_val:10.2f}, error={rel_error:8.4%}")
            else:
                print(f"✗ {key[0]:5s}: MISSING in output")
                all_correct = False

        print("\n" + "=" * 80)
        if all_correct:
            print(f"✅ All values correct! Max relative error: {max_error:.6%}")
            return True
        else:
            print(f"❌ Some values incorrect. Max relative error: {max_error:.6%}")
            return False

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arithmetic_sequence():
    """Test that arithmetic sequences still work correctly."""
    print("\n" + "=" * 80)
    print("TEST: Reading arithmetic sequence from GDX")
    print("=" * 80)

    gdx_file = Path(__file__).parent.parent / "tests" / "fixtures" / "test_arithmetic.gdx"

    if not gdx_file.exists():
        print(f"❌ Test file not found: {gdx_file}")
        return False

    print(f"\n📁 Reading: {gdx_file.name}")

    try:
        # Read the GDX file first
        gdx_data = read_gdx(str(gdx_file))

        # Then read the parameter values
        data = read_parameter_values(gdx_data, "arith")

        print("\n✓ Successfully read parameter 'arith'")
        print(f"  Records read: {len(data)}")

        # Expected arithmetic sequence (10,20,30,40,50,60,70,80,90,100)
        expected = {
            ("t1",): 10.0,
            ("t2",): 20.0,
            ("t3",): 30.0,
            ("t4",): 40.0,
            ("t5",): 50.0,
            ("t6",): 60.0,
            ("t7",): 70.0,
            ("t8",): 80.0,
            ("t9",): 90.0,
            ("t10",): 100.0,
        }

        print("\n" + "=" * 80)
        print("VALIDATION:")
        print("=" * 80)

        all_correct = True
        max_error = 0.0

        for key, expected_val in expected.items():
            if key in data:
                actual_val = data[key]
                error = abs(actual_val - expected_val)
                rel_error = error / expected_val if expected_val != 0 else 0.0

                status = "✓" if error < 0.01 else "✗"
                if error >= 0.01:
                    all_correct = False

                max_error = max(max_error, rel_error)

                print(f"{status} {key[0]:5s}: expected={expected_val:10.2f}, "
                      f"actual={actual_val:10.2f}, error={rel_error:8.4%}")
            else:
                print(f"✗ {key[0]:5s}: MISSING in output")
                all_correct = False

        print("\n" + "=" * 80)
        if all_correct:
            print(f"✅ All values correct! Max relative error: {max_error:.6%}")
            return True
        else:
            print(f"❌ Some values incorrect. Max relative error: {max_error:.6%}")
            return False

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    geom_ok = test_geometric_sequence()
    arith_ok = test_arithmetic_sequence()

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Geometric interpolation: {'✅ PASS' if geom_ok else '❌ FAIL'}")
    print(f"Arithmetic interpolation: {'✅ PASS' if arith_ok else '❌ FAIL'}")
    print("=" * 80)

    sys.exit(0 if (geom_ok and arith_ok) else 1)
