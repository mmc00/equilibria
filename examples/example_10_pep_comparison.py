"""Example 10: GAMS Comparison Testing

This example demonstrates how to compare equilibria model results
with GAMS reference solutions using the comparison framework.
"""

from pathlib import Path

from equilibria import Model
from equilibria.templates import PEP1R
from equilibria.templates.gams_comparison import (
    GAMSRunner,
    PEPGAMSComparator,
    SolutionComparator,
)


def main():
    """Run GAMS comparison example."""
    print("=" * 70)
    print("Example 10: GAMS Comparison Testing")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Step 1: Create equilibria PEP model
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Step 1: Create equilibria PEP Model")
    print("-" * 70)
    print()

    template = PEP1R()
    model = template.create_model(calibrate=True)

    print(f"Created model: {model.name}")
    stats = model.statistics
    print(f"Variables: {stats.variables}")
    print(f"Equations: {stats.equations}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Demonstrate comparison framework components
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Step 2: Comparison Framework Components")
    print("-" * 70)
    print()

    # Show available components
    print("Available components:")
    print("  - GAMSRunner: Execute GAMS models and load GDX results")
    print("  - SolutionComparator: Compare two solutions")
    print("  - PEPGAMSComparator: PEP-specific comparison with variable mappings")
    print("  - GAMSComparisonReport: Detailed comparison results")
    print()

    # ------------------------------------------------------------------
    # Step 3: Demonstrate manual comparison
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Step 3: Manual Solution Comparison")
    print("-" * 70)
    print()

    # Create comparator with 0.01% tolerance
    comparator = SolutionComparator(tolerance=1e-4)

    # Example: Compare scalar values
    print("Example comparison of scalar values:")
    result = comparator.compare_scalar(
        equilibria_value=100.0,
        gams_value=100.001,
        variable_name="Price_index",
    )

    print(f"  Variable: {result.variable_name}")
    print(f"  Equilibria: {result.equilibria_value:.6f}")
    print(f"  GAMS: {result.gams_value:.6f}")
    print(f"  Absolute diff: {result.absolute_diff:.6f}")
    print(f"  Relative diff: {result.relative_diff:.4f}%")
    print(f"  Passed: {result.passed}")
    print()

    # Example: Comparison that fails
    result2 = comparator.compare_scalar(
        equilibria_value=100.0,
        gams_value=105.0,
        variable_name="GDP",
    )

    print("Example comparison with larger difference:")
    print(f"  Variable: {result2.variable_name}")
    print(f"  Equilibria: {result2.equilibria_value:.6f}")
    print(f"  GAMS: {result2.gams_value:.6f}")
    print(f"  Relative diff: {result2.relative_diff:.2f}%")
    print(f"  Passed: {result2.passed}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Show PEP-specific comparator
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Step 4: PEP-Specific Variable Mappings")
    print("-" * 70)
    print()

    pep_comparator = PEPGAMSComparator(tolerance=1e-4)

    print("Standard PEP variable mappings (equilibria -> GAMS):")
    for equi_name, gams_name in list(pep_comparator.PEP_VARIABLE_MAP.items())[:10]:
        print(f"  {equi_name} -> {gams_name}")
    print("  ... (and more)")
    print()

    # ------------------------------------------------------------------
    # Step 5: Show how to run full comparison (if GAMS available)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Step 5: Full GAMS Comparison Workflow")
    print("-" * 70)
    print()

    print("To run a full comparison with GAMS:")
    print()
    print("  # 1. Create equilibria model and solve it")
    print("  model = template.create_model()")
    print("  backend = PyomoBackend()")
    print("  solution_equi = backend.solve(model)")
    print()
    print("  # 2. Run GAMS model")
    print("  runner = GAMSRunner(gams_file='PEP-1-1_v2_1_modular.gms')")
    print("  runner.run(output_gdx='results.gdx')")
    print("  gams_results = runner.load_results()")
    print()
    print("  # 3. Compare results")
    print("  comparator = PEPGAMSComparator(tolerance=1e-4)")
    print("  report = comparator.compare_with_gams(")
    print("      equilibria_solution=solution_equi,")
    print("      gams_gdx_file='results.gdx'")
    print("  )")
    print()
    print("  # 4. Review report")
    print("  print(report.summary)")
    print("  print(f'Passed: {report.passed_variables}/{report.total_variables}')")
    print()

    # ------------------------------------------------------------------
    # Step 6: Check if GAMS is available
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Step 6: Check GAMS Availability")
    print("-" * 70)
    print()

    # Try to detect GAMS
    import shutil

    gams_available = shutil.which("gams") is not None

    if gams_available:
        print("✓ GAMS is available on this system")
        print()
        print("You can run full comparisons using:")
        print("  from equilibria.templates import run_gams_comparison")
        print("  report = run_gams_comparison(")
        print("      gams_file='path/to/model.gms',")
        print("      equilibria_solution=solution,")
        print("      tolerance=1e-4")
        print("  )")
    else:
        print("✗ GAMS not found on this system")
        print()
        print("To use GAMS comparison:")
        print("  1. Install GAMS from https://www.gams.com/")
        print("  2. Add GAMS to your system PATH")
        print("  3. Verify with: gams --version")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("The GAMS comparison framework provides:")
    print("  ✓ Automated GAMS execution")
    print("  ✓ GDX result loading")
    print("  ✓ Variable name mapping (equilibria <-> GAMS)")
    print("  ✓ Statistical comparison with configurable tolerance")
    print("  ✓ Detailed reports showing passed/failed variables")
    print()
    print("Default tolerance: 1e-4 (0.01% relative difference)")
    print()
    print("Framework components:")
    print("  - GAMSRunner: Execute GAMS and load GDX")
    print("  - SolutionComparator: Generic comparison")
    print("  - PEPGAMSComparator: PEP-specific with variable mappings")
    print("  - GAMSComparisonReport: Detailed results")
    print()


if __name__ == "__main__":
    main()
