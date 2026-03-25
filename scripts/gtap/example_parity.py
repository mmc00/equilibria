#!/usr/bin/env python3
"""
Example: GTAP Parity Testing

This example demonstrates how to:
1. Run a GTAP model in Python
2. Load GAMS baseline results
3. Compare and validate parity

Requirements:
- GTAP data GDX file (e.g., asa7x5.gdx)
- GAMS baseline results GDX file
- Both models should use the same data/aggregation
"""

from pathlib import Path
import sys

# Add equilibria to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from equilibria.templates.gtap import (
    GTAPSets,
    GTAPParameters,
    GTAPModelEquations,
    GTAPSolver,
    build_gtap_contract,
)
from equilibria.templates.gtap.gtap_parity_pipeline import (
    GTAPParityRunner,
    GTAPGAMSReference,
    compare_gtap_gams_parity,
)


def example_basic_parity():
    """Basic example of running a parity check."""
    print("=" * 70)
    print("Example 1: Basic Parity Check")
    print("=" * 70)
    
    # File paths
    gdx_file = Path("data/asa7x5.gdx")
    gams_results = Path("results/gams_baseline.gdx")
    
    if not gdx_file.exists() or not gams_results.exists():
        print("⚠  GDX files not found. This example requires:")
        print(f"   - Data file: {gdx_file}")
        print(f"   - GAMS results: {gams_results}")
        print("\n  Creating mock example instead...")
        example_mock_parity()
        return
    
    # Method 1: Use the convenience function
    print("\nMethod 1: Using run_gtap_parity_test()")
    print("-" * 70)
    
    from equilibria.templates.gtap import run_gtap_parity_test
    
    result = run_gtap_parity_test(
        gdx_file=gdx_file,
        gams_results_gdx=gams_results,
        tolerance=1e-6,
    )
    
    print(f"\nResult: {'✓ PASSED' if result.passed else '✗ FAILED'}")
    print(f"Variables compared: {result.n_variables_compared}")
    print(f"Mismatches: {result.n_mismatches}")
    

def example_manual_parity():
    """Manual parity check with full control."""
    print("\n" + "=" * 70)
    print("Example 2: Manual Parity Check")
    print("=" * 70)
    
    gdx_file = Path("data/asa7x5.gdx")
    gams_results = Path("results/gams_baseline.gdx")
    
    if not gdx_file.exists() or not gams_results.exists():
        print("⚠  GDX files not found.")
        return
    
    # Step 1: Create runner
    print("\n1. Creating parity runner...")
    runner = GTAPParityRunner(
        gdx_file=gdx_file,
        gams_results_gdx=gams_results,
        closure="gtap_standard",
        solver="ipopt",
        tolerance=1e-6,
    )
    
    # Step 2: Run Python model
    print("\n2. Running Python GTAP model...")
    py_result = runner.run_python()
    
    if py_result.success:
        print(f"   ✓ Converged in {py_result.iterations} iterations")
        print(f"   ✓ Walras check: {py_result.walras_value:.2e}")
    else:
        print(f"   ✗ Failed: {py_result.message}")
        return
    
    # Step 3: Run parity comparison
    print("\n3. Comparing with GAMS...")
    comparison = runner.run_parity_check()
    
    # Step 4: Analyze results
    print("\n4. Results:")
    print(f"   Status: {'✓ PASSED' if comparison.passed else '✗ FAILED'}")
    print(f"   Variables compared: {comparison.n_variables_compared}")
    print(f"   Mismatches: {comparison.n_mismatches}")
    print(f"   Max absolute diff: {comparison.max_abs_diff:.2e}")
    print(f"   Max relative diff: {comparison.max_rel_diff:.2e}")
    
    # Step 5: Show top mismatches
    if comparison.mismatches:
        print("\n5. Top 10 Mismatches:")
        for i, m in enumerate(comparison.mismatches[:10], 1):
            key_str = str(m['key'])
            print(f"   {i}. {m['group']}{key_str}")
            print(f"      Python: {m['python']:.6f}")
            print(f"      GAMS:   {m['gams']:.6f}")
            print(f"      Diff:   {m['abs_diff']:.6e}")


def example_mock_parity():
    """Example with mock data (no GDX files needed)."""
    print("\n" + "=" * 70)
    print("Example 3: Mock Parity Check (No GDX files)")
    print("=" * 70)
    
    # Create mock Python results
    print("\nCreating mock Python snapshot...")
    py_snap = GTAPVariableSnapshot(
        xp={
            ("USA", "agr"): 100.0,
            ("USA", "mfg"): 200.0,
            ("EUR", "agr"): 150.0,
            ("EUR", "mfg"): 180.0,
        },
        ps={
            ("USA", "agr"): 1.0,
            ("USA", "mfg"): 1.0,
            ("EUR", "agr"): 1.0,
            ("EUR", "mfg"): 1.0,
        },
        regy={
            "USA": 1000.0,
            "EUR": 900.0,
        },
        pnum=1.0,
        walras=1e-12,
    )
    
    # Create mock GAMS results (slightly different)
    print("Creating mock GAMS snapshot...")
    gams_snap = GTAPVariableSnapshot(
        xp={
            ("USA", "agr"): 100.000001,  # Tiny difference
            ("USA", "mfg"): 200.0,
            ("EUR", "agr"): 150.0,
            ("EUR", "mfg"): 180.0001,  # Small difference
        },
        ps={
            ("USA", "agr"): 1.0,
            ("USA", "mfg"): 1.0,
            ("EUR", "agr"): 1.0,
            ("EUR", "mfg"): 1.0,
        },
        regy={
            "USA": 1000.0,
            "EUR": 900.0,
        },
        pnum=1.0,
        walras=1e-10,
    )
    
    # Create GAMS reference
    gams_ref = GTAPGAMSReference(
        gdx_path=Path("mock.gdx"),
        sets=None,
        snapshot=gams_snap,
        modelstat=1.0,
        solvestat=1.0,
        solve_time=1.0,
    )
    
    # Compare with loose tolerance
    print("\nComparing with tolerance=1e-3...")
    result_loose = compare_gtap_gams_parity(py_snap, gams_ref, tolerance=1e-3)
    print(f"Result: {'✓ PASSED' if result_loose.passed else '✗ FAILED'}")
    print(f"Mismatches: {result_loose.n_mismatches}")
    
    # Compare with tight tolerance
    print("\nComparing with tolerance=1e-8...")
    result_tight = compare_gtap_gams_parity(py_snap, gams_ref, tolerance=1e-8)
    print(f"Result: {'✓ PASSED' if result_tight.passed else '✗ FAILED'}")
    print(f"Mismatches: {result_tight.n_mismatches}")
    
    if result_tight.mismatches:
        print("\nMismatches found:")
        for m in result_tight.mismatches:
            print(f"  - {m['group']}{m['key']}: diff={m['abs_diff']:.2e}")


def example_advanced_analysis():
    """Advanced analysis of parity results."""
    print("\n" + "=" * 70)
    print("Example 4: Advanced Analysis")
    print("=" * 70)
    
    gdx_file = Path("data/asa7x5.gdx")
    gams_results = Path("results/gams_baseline.gdx")
    
    if not gdx_file.exists() or not gams_results.exists():
        print("⚠  GDX files not found.")
        return
    
    # Run with different tolerances
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    
    print("\nSensitivity analysis with different tolerances:")
    print("-" * 70)
    print(f"{'Tolerance':>12} {'Status':>10} {'Mismatches':>12} {'Max Diff':>12}")
    print("-" * 70)
    
    for tol in tolerances:
        runner = GTAPParityRunner(
            gdx_file=gdx_file,
            gams_results_gdx=gams_results,
            tolerance=tol,
        )
        
        # Just compare, don't re-solve
        if runner.gams_reference:
            from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot
            py_snap = GTAPVariableSnapshot.from_python_model(runner.model)
            result = compare_gtap_gams_parity(py_snap, runner.gams_reference, tolerance=tol)
            
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{tol:>12.0e} {status:>10} {result.n_mismatches:>12} {result.max_abs_diff:>12.2e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("GTAP Parity Testing Examples")
    print("=" * 70)
    print("\nThis script demonstrates GTAP Python vs GAMS parity testing.")
    print("Some examples require GDX files (will skip if not available).")
    print("")
    
    # Run examples
    example_basic_parity()
    example_manual_parity()
    example_mock_parity()
    example_advanced_analysis()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
    print("\nFor CLI usage:")
    print("  python scripts/gtap/run_gtap_parity.py --help")
    print("\nFor more information, see:")
    print("  - src/equilibria/templates/gtap/README.md")
    print("  - tests/templates/gtap/test_gtap_parity_pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
