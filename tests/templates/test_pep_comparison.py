"""Run GAMS vs equilibria comparison test.

This script runs the GAMS PEP model and compares results with equilibria.
"""

import sys
from pathlib import Path

# Add GAMS to PATH
import os

os.environ["PATH"] = (
    "/Library/Frameworks/GAMS.framework/Versions/48/Resources:"
    + os.environ.get("PATH", "")
)

from equilibria import Model
from equilibria.backends import PyomoBackend
from equilibria.templates import PEP1R
from equilibria.templates.gams_comparison import (
    GAMSRunner,
    PEPGAMSComparator,
    GAMSComparisonReport,
)


def main():
    """Run comparison between GAMS and equilibria PEP models."""
    print("=" * 70)
    print("GAMS vs equilibria Comparison Test")
    print("=" * 70)
    print()

    # Check GAMS availability
    gams_path = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams")
    if not gams_path.exists():
        print(f"ERROR: GAMS not found at {gams_path}")
        print("Please verify GAMS installation.")
        sys.exit(1)

    print(f"✓ GAMS found: {gams_path}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Create and solve equilibria model
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

    # Solve with Pyomo
    print("Solving equilibria model...")
    backend = PyomoBackend(solver="ipopt")
    backend.build(model)

    try:
        solution_equi = backend.solve()
        print(f"✓ Solved: {solution_equi.status}")
        print(f"  Solve time: {solution_equi.solve_time:.2f}s")
        print(f"  Iterations: {solution_equi.iterations}")
        print()
    except Exception as e:
        print(f"✗ Solve failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Run GAMS model
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Step 2: Run GAMS PEP Model")
    print("-" * 70)
    print()

    gams_file = Path("src/equilibria/templates/reference/pep/PEP-1-1_v2_1_modular.gms")
    if not gams_file.exists():
        print(f"ERROR: GAMS file not found: {gams_file}")
        sys.exit(1)

    print(f"GAMS file: {gams_file}")
    print()

    runner = GAMSRunner(gams_file=gams_file, gams_executable=str(gams_path))

    print("Running GAMS...")
    try:
        returncode = runner.run(output_gdx="pep_results.gdx", timeout=300)

        if returncode != 0:
            print(f"✗ GAMS failed with return code: {returncode}")
            sys.exit(1)

        print("✓ GAMS completed successfully")
        print()
    except Exception as e:
        print(f"✗ GAMS execution failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Load GAMS results
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Step 3: Load GAMS Results")
    print("-" * 70)
    print()

    try:
        gams_results = runner.load_results()
        print(f"✓ Loaded {len(gams_results)} variables from GAMS")
        print()
    except Exception as e:
        print(f"✗ Failed to load GAMS results: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Compare results
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Step 4: Compare Results")
    print("-" * 70)
    print()

    comparator = PEPGAMSComparator(tolerance=1e-4)

    try:
        report = comparator.compare_with_gams(
            equilibria_solution=solution_equi,
            gams_gdx_file=runner.output_file or Path("pep_results.gdx"),
            model_name="PEP-1R",
        )

        print(report.summary)
        print()

        # Show some variable comparisons
        print("Sample variable comparisons:")
        for result in report.results[:10]:
            status = "✓" if result.passed else "✗"
            print(
                f"  {status} {result.variable_name:20s} "
                f"equi={result.equilibria_value:12.6f}  "
                f"gams={result.gams_value:12.6f}  "
                f"diff={result.relative_diff:8.4f}%"
            )

        if len(report.results) > 10:
            print(f"  ... and {len(report.results) - 10} more variables")

        print()

    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print()

    if report.failed_variables == 0:
        print("✓ ALL TESTS PASSED")
        print(
            f"  All {report.total_variables} variables match within tolerance ({report.tolerance})"
        )
        sys.exit(0)
    else:
        print(f"✗ {report.failed_variables} variables failed")
        print(f"  {report.passed_variables}/{report.total_variables} passed")
        print()
        print("Failed variables:")
        for result in report.results:
            if not result.passed:
                print(
                    f"  - {result.variable_name}: {result.relative_diff:.4f}% difference"
                )
        sys.exit(1)


if __name__ == "__main__":
    main()
