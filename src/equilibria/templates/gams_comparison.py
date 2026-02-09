"""GAMS comparison testing framework for equilibria.

This module provides utilities to compare equilibria model results
with GAMS reference solutions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from equilibria.backends.base import Solution


class GAMSComparisonResult(BaseModel):
    """Result of comparing equilibria solution with GAMS.

    Attributes:
        variable_name: Name of the variable compared
        equilibria_value: Value from equilibria
        gams_value: Value from GAMS
        absolute_diff: Absolute difference
        relative_diff: Relative difference (percentage)
        passed: Whether difference is within tolerance
    """

    variable_name: str = Field(..., description="Variable name")
    equilibria_value: float = Field(..., description="Value from equilibria")
    gams_value: float = Field(..., description="Value from GAMS")
    absolute_diff: float = Field(..., description="Absolute difference")
    relative_diff: float = Field(..., description="Relative difference (%)")
    passed: bool = Field(..., description="Within tolerance")


class GAMSComparisonReport(BaseModel):
    """Full comparison report between equilibria and GAMS.

    Attributes:
        model_name: Name of the model being compared
        tolerance: Tolerance used for comparison
        total_variables: Total number of variables compared
        passed_variables: Number of variables within tolerance
        failed_variables: Number of variables outside tolerance
        results: List of individual comparison results
        summary: Text summary of the comparison
    """

    model_name: str = Field(..., description="Model name")
    tolerance: float = Field(default=1e-4, description="Comparison tolerance")
    total_variables: int = Field(default=0, description="Total variables compared")
    passed_variables: int = Field(default=0, description="Variables within tolerance")
    failed_variables: int = Field(default=0, description="Variables outside tolerance")
    results: list[GAMSComparisonResult] = Field(default_factory=list)
    summary: str = Field(default="", description="Text summary")

    def generate_summary(self) -> str:
        """Generate a text summary of the comparison."""
        lines = [
            f"GAMS Comparison Report: {self.model_name}",
            f"Tolerance: {self.tolerance}",
            f"Total variables: {self.total_variables}",
            f"Passed: {self.passed_variables} ({100 * self.passed_variables / max(1, self.total_variables):.1f}%)",
            f"Failed: {self.failed_variables}",
            "",
        ]

        if self.failed_variables > 0:
            lines.append("Failed variables:")
            for result in self.results:
                if not result.passed:
                    lines.append(
                        f"  {result.variable_name}: "
                        f"equi={result.equilibria_value:.6f}, "
                        f"gams={result.gams_value:.6f}, "
                        f"rel_diff={result.relative_diff:.2f}%"
                    )

        self.summary = "\n".join(lines)
        return self.summary


class GAMSRunner:
    """Runner for GAMS models.

    Handles execution of GAMS models and loading of results from GDX files.

    Example:
        >>> runner = GAMSRunner(gams_file="model.gms")
        >>> runner.run()
        >>> results = runner.load_results("results.gdx")
    """

    def __init__(
        self,
        gams_file: Path | str,
        working_dir: Path | str | None = None,
        gams_executable: str = "gams",
    ):
        """Initialize GAMS runner.

        Args:
            gams_file: Path to the main GAMS file
            working_dir: Working directory for GAMS execution
            gams_executable: GAMS executable name/path
        """
        self.gams_file = Path(gams_file)
        self.working_dir = Path(working_dir) if working_dir else self.gams_file.parent
        self.gams_executable = gams_executable
        self.output_file: Path | None = None
        self.last_returncode: int | None = None

    def run(
        self,
        output_gdx: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: int = 300,
    ) -> int:
        """Run the GAMS model.

        Args:
            output_gdx: Name of output GDX file (optional)
            options: Additional GAMS options
            timeout: Timeout in seconds

        Returns:
            Return code from GAMS

        Raises:
            FileNotFoundError: If GAMS executable not found
            subprocess.TimeoutExpired: If GAMS times out
        """
        cmd = [self.gams_executable, str(self.gams_file)]

        if output_gdx:
            cmd.extend([f"gdx={output_gdx}"])
            self.output_file = self.working_dir / output_gdx

        if options:
            for key, value in options.items():
                cmd.append(f"{key}={value}")

        # Run GAMS
        result = subprocess.run(
            cmd,
            cwd=self.working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        self.last_returncode = result.returncode
        return result.returncode

    def load_results(self, gdx_file: Path | str | None = None) -> dict[str, Any]:
        """Load results from GDX file.

        Args:
            gdx_file: Path to GDX file (uses output_file if None)

        Returns:
            Dictionary of variable name -> values

        Raises:
            FileNotFoundError: If GDX file not found
        """
        gdx_path = Path(gdx_file) if gdx_file else self.output_file
        if not gdx_path:
            raise ValueError("No GDX file specified")

        if not gdx_path.exists():
            raise FileNotFoundError(f"GDX file not found: {gdx_path}")

        # Use equilibria's GDX reader
        from equilibria.babel.gdx.reader import read_gdx

        gdx_data = read_gdx(gdx_path)

        # Extract variables (parameters in GAMS terms)
        results = {}
        for symbol in gdx_data.get("symbols", []):
            if symbol.get("type") == "parameter":
                name = symbol.get("name", "").lower()
                values = symbol.get("values", [])
                results[name] = values

        return results


class SolutionComparator:
    """Compare equilibria solutions with GAMS reference solutions.

    Example:
        >>> comparator = SolutionComparator(tolerance=1e-4)
        >>> report = comparator.compare(
        ...     equilibria_solution=sol_equi,
        ...     gams_solution=sol_gams,
        ...     variable_mapping={"PD": "PD", "PX": "PX"}
        ... )
    """

    def __init__(self, tolerance: float = 1e-4):
        """Initialize comparator.

        Args:
            tolerance: Maximum allowed relative difference (default: 1e-4 = 0.01%)
        """
        self.tolerance = tolerance

    def compare(
        self,
        equilibria_solution: Solution,
        gams_solution: dict[str, Any],
        variable_mapping: dict[str, str] | None = None,
        model_name: str = "Model",
    ) -> GAMSComparisonReport:
        """Compare two solutions.

        Args:
            equilibria_solution: Solution from equilibria
            gams_solution: Solution from GAMS (dictionary of variables)
            variable_mapping: Mapping of equilibria names to GAMS names
            model_name: Name of the model

        Returns:
            Comparison report
        """
        report = GAMSComparisonReport(
            model_name=model_name,
            tolerance=self.tolerance,
        )

        # Get equilibria variables
        equi_vars = equilibria_solution.variables

        # Compare each variable
        for equi_name, equi_value in equi_vars.items():
            # Map name if needed
            gams_name = (
                variable_mapping.get(equi_name, equi_name)
                if variable_mapping
                else equi_name
            )
            gams_name = gams_name.lower()

            if gams_name not in gams_solution:
                continue

            gams_value = gams_solution[gams_name]

            # Handle scalar vs array values
            if isinstance(equi_value, np.ndarray):
                equi_scalar = float(np.mean(equi_value))
            else:
                equi_scalar = float(equi_value)

            if isinstance(gams_value, (list, np.ndarray)):
                gams_scalar = float(np.mean(gams_value))
            else:
                gams_scalar = float(gams_value)

            # Calculate differences
            abs_diff = abs(equi_scalar - gams_scalar)
            if abs(gams_scalar) > 1e-10:
                rel_diff = abs_diff / abs(gams_scalar) * 100
            else:
                rel_diff = abs_diff * 100

            passed = rel_diff <= self.tolerance * 100

            result = GAMSComparisonResult(
                variable_name=equi_name,
                equilibria_value=equi_scalar,
                gams_value=gams_scalar,
                absolute_diff=abs_diff,
                relative_diff=rel_diff,
                passed=passed,
            )

            report.results.append(result)
            report.total_variables += 1
            if passed:
                report.passed_variables += 1
            else:
                report.failed_variables += 1

        report.generate_summary()
        return report

    def compare_scalar(
        self,
        equilibria_value: float,
        gams_value: float,
        variable_name: str = "variable",
    ) -> GAMSComparisonResult:
        """Compare a single scalar value.

        Args:
            equilibria_value: Value from equilibria
            gams_value: Value from GAMS
            variable_name: Name of the variable

        Returns:
            Comparison result
        """
        abs_diff = abs(equilibria_value - gams_value)
        if abs(gams_value) > 1e-10:
            rel_diff = abs_diff / abs(gams_value) * 100
        else:
            rel_diff = abs_diff * 100

        passed = rel_diff <= self.tolerance * 100

        return GAMSComparisonResult(
            variable_name=variable_name,
            equilibria_value=equilibria_value,
            gams_value=gams_value,
            absolute_diff=abs_diff,
            relative_diff=rel_diff,
            passed=passed,
        )


class PEPGAMSComparator:
    """Specialized comparator for PEP models.

    Provides PEP-specific variable mappings and comparison logic.
    """

    # Standard PEP variable name mappings (equilibria -> GAMS)
    PEP_VARIABLE_MAP = {
        "PD": "pd",  # Domestic price
        "PX": "px",  # Export price
        "PM": "pm",  # Import price
        "PE": "pe",  # World export price
        "PWE": "pwe",  # World export price FOB
        "PWM": "pwm",  # World import price CIF
        "XD": "xd",  # Domestic output
        "XE": "xe",  # Exports
        "XM": "xm",  # Imports
        "QA": "qa",  # Armington composite
        "VA": "va",  # Value added
        "Y": "y",  # Income
        "YD": "yd",  # Disposable income
        "C": "c",  # Consumption
        "I": "i",  # Investment
        "G": "g",  # Government spending
        "FSAV": "fsav",  # Foreign savings
        "EXR": "exr",  # Exchange rate
        "WF": "wf",  # Factor price
        "QF": "qf",  # Factor quantity
        "YH": "yh",  # Household income
        "YG": "yg",  # Government revenue
        "TINS": "tins",  # Income tax rate
        "TAU_T": "tau_t",  # Production tax
        "TAU_M": "tau_m",  # Import tariff
        "TAU_X": "tau_x",  # Export tax
    }

    def __init__(self, tolerance: float = 1e-4):
        """Initialize PEP comparator.

        Args:
            tolerance: Maximum allowed relative difference
        """
        self.comparator = SolutionComparator(tolerance=tolerance)

    def compare_with_gams(
        self,
        equilibria_solution: Solution,
        gams_gdx_file: Path | str,
        model_name: str = "PEP-1R",
    ) -> GAMSComparisonReport:
        """Compare equilibria PEP solution with GAMS results.

        Args:
            equilibria_solution: Solution from equilibria PEP model
            gams_gdx_file: Path to GAMS GDX results file
            model_name: Name of the model

        Returns:
            Comparison report
        """
        # Load GAMS results
        runner = GAMSRunner(gams_file=Path(gams_gdx_file))
        gams_results = runner.load_results(gams_gdx_file)

        # Compare using PEP variable mappings
        return self.comparator.compare(
            equilibria_solution=equilibria_solution,
            gams_solution=gams_results,
            variable_mapping=self.PEP_VARIABLE_MAP,
            model_name=model_name,
        )


def run_gams_comparison(
    gams_file: Path | str,
    equilibria_solution: Solution,
    output_gdx: str = "results.gdx",
    tolerance: float = 1e-4,
) -> GAMSComparisonReport:
    """Convenience function to run GAMS and compare results.

    Args:
        gams_file: Path to GAMS model file
        equilibria_solution: Solution from equilibria
        output_gdx: Name of GDX output file
        tolerance: Comparison tolerance

    Returns:
        Comparison report
    """
    # Run GAMS
    runner = GAMSRunner(gams_file=gams_file)
    returncode = runner.run(output_gdx=output_gdx)

    if returncode != 0:
        raise RuntimeError(f"GAMS execution failed with return code {returncode}")

    # Load and compare results
    comparator = SolutionComparator(tolerance=tolerance)
    gams_results = runner.load_results()

    return comparator.compare(
        equilibria_solution=equilibria_solution,
        gams_solution=gams_results,
        model_name=Path(gams_file).stem,
    )
