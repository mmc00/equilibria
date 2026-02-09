"""Test GAMS comparison framework with mock data.

This test validates the comparison framework without requiring
a full working GAMS model or equilibria solver.
"""

import numpy as np
from pathlib import Path

from equilibria.backends.base import Solution
from equilibria.templates.gams_comparison import (
    GAMSComparisonResult,
    GAMSComparisonReport,
    SolutionComparator,
    PEPGAMSComparator,
)


def test_comparison_result():
    """Test GAMSComparisonResult creation."""
    result = GAMSComparisonResult(
        variable_name="PD",
        equilibria_value=100.0,
        gams_value=100.001,
        absolute_diff=0.001,
        relative_diff=0.001,
        passed=True,
    )

    assert result.variable_name == "PD"
    assert result.passed is True
    print("✓ GAMSComparisonResult test passed")


def test_comparison_report():
    """Test GAMSComparisonReport generation."""
    report = GAMSComparisonReport(
        model_name="TestModel",
        tolerance=1e-4,
        total_variables=10,
        passed_variables=8,
        failed_variables=2,
        results=[
            GAMSComparisonResult(
                variable_name="var1",
                equilibria_value=100.0,
                gams_value=100.0,
                absolute_diff=0.0,
                relative_diff=0.0,
                passed=True,
            ),
            GAMSComparisonResult(
                variable_name="var2",
                equilibria_value=100.0,
                gams_value=105.0,
                absolute_diff=5.0,
                relative_diff=5.0,
                passed=False,
            ),
        ],
    )

    summary = report.generate_summary()
    assert "TestModel" in summary
    assert "8" in summary  # passed count
    assert "2" in summary  # failed count
    print("✓ GAMSComparisonReport test passed")


def test_scalar_comparison():
    """Test scalar value comparison."""
    comparator = SolutionComparator(tolerance=1e-4)

    # Test exact match
    result = comparator.compare_scalar(100.0, 100.0, "exact")
    assert result.passed is True
    assert result.relative_diff == 0.0

    # Test within tolerance
    result = comparator.compare_scalar(100.0, 100.0001, "within_tol")
    assert result.passed is True
    assert result.relative_diff <= 0.01  # 0.01%

    # Test outside tolerance
    result = comparator.compare_scalar(100.0, 105.0, "outside_tol")
    assert result.passed is False
    assert result.relative_diff > 0.01

    print("✓ Scalar comparison test passed")


def test_solution_comparison():
    """Test full solution comparison."""
    # Create mock equilibria solution
    equi_solution = Solution(
        model_name="PEP",
        status="optimal",
        variables={
            "PD": np.array([1.0, 1.0, 1.0]),
            "PX": np.array([1.1, 1.2, 1.3]),
            "PM": np.array([0.9, 0.95, 1.0]),
        },
    )

    # Create mock GAMS results
    gams_results = {
        "pd": [1.0, 1.0, 1.0],
        "px": [1.1, 1.2, 1.3],
        "pm": [0.9, 0.95, 1.0],
    }

    # Compare
    comparator = SolutionComparator(tolerance=1e-4)
    report = comparator.compare(
        equilibria_solution=equi_solution,
        gams_solution=gams_results,
        variable_mapping={"PD": "pd", "PX": "px", "PM": "pm"},
        model_name="PEP-Test",
    )

    assert report.total_variables == 3
    assert report.passed_variables == 3
    assert report.failed_variables == 0
    print("✓ Solution comparison test passed")


def test_pep_variable_mappings():
    """Test PEP variable mappings are defined."""
    comparator = PEPGAMSComparator(tolerance=1e-4)

    # Check key mappings exist
    assert "PD" in comparator.PEP_VARIABLE_MAP
    assert "PX" in comparator.PEP_VARIABLE_MAP
    assert "PM" in comparator.PEP_VARIABLE_MAP
    assert comparator.PEP_VARIABLE_MAP["PD"] == "pd"

    print("✓ PEP variable mappings test passed")


def test_comparison_with_differences():
    """Test comparison that finds differences."""
    equi_solution = Solution(
        model_name="PEP",
        status="optimal",
        variables={
            "PD": np.array([100.0]),
            "PX": np.array([105.0]),  # Different from GAMS
        },
    )

    gams_results = {
        "pd": [100.0],
        "px": [100.0],  # Different from equilibria
    }

    comparator = SolutionComparator(tolerance=1e-4)
    report = comparator.compare(
        equilibria_solution=equi_solution,
        gams_solution=gams_results,
        variable_mapping={"PD": "pd", "PX": "px"},
        model_name="PEP-Diff",
    )

    assert report.total_variables == 2
    assert report.passed_variables == 1  # PD matches
    assert report.failed_variables == 1  # PX differs

    # Check the failed result
    failed_results = [r for r in report.results if not r.passed]
    assert len(failed_results) == 1
    assert failed_results[0].variable_name == "PX"
    assert failed_results[0].relative_diff == 5.0  # 5% difference

    print("✓ Comparison with differences test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("GAMS Comparison Framework Tests")
    print("=" * 70)
    print()

    test_comparison_result()
    test_comparison_report()
    test_scalar_comparison()
    test_solution_comparison()
    test_pep_variable_mappings()
    test_comparison_with_differences()

    print()
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)
