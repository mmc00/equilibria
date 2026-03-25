"""GTAP GAMS Parity Pipeline

This module provides tools to compare GTAP Python results against
CGEBox GAMS baseline results.

Usage:
    # Load GAMS reference and compare
    from equilibria.templates.gtap.gtap_parity_pipeline import (
        load_gtap_gams_reference,
        compare_gtap_gams_parity,
        GTAPParityRunner,
    )
    
    reference = load_gtap_gams_reference("gams_results.gdx")
    comparison = compare_gtap_gams_parity(
        python_result=python_solution,
        gams_reference=reference,
        tolerance=1e-6,
    )
    
    # Or use the full runner
    runner = GTAPParityRunner(
        gdx_file="asa7x5.gdx",
        gams_results_gdx="gams_results.gdx",
    )
    result = runner.run_parity_check()
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.templates.gtap import (
    GTAPModelEquations,
    GTAPParameters,
    GTAPSets,
    GTAPSolver,
    build_gtap_contract,
)
from equilibria.templates.gtap.gtap_solver import SolverResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GTAPVariableSnapshot:
    """Snapshot of GTAP variable values.
    
    Stores key variables for parity comparison.
    """
    # Activity levels
    xp: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, a)
    x: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, a, i)
    
    # Prices
    px: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, a)
    pp: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, a)
    ps: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    pd: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    pa: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    pmt: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    pet: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    
    # Trade
    xe: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp)
    xw: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp)
    xmt: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    xet: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    
    # Factors
    xf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a)
    xft: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, f)
    pf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a)
    pft: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, f)
    
    # Demand
    xc: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    xg: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    xi: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    
    # Income
    regy: Dict[str, float] = field(default_factory=dict)  # r
    yc: Dict[str, float] = field(default_factory=dict)  # r
    yg: Dict[str, float] = field(default_factory=dict)  # r
    yi: Dict[str, float] = field(default_factory=dict)  # r
    
    # Price indices
    pnum: float = 1.0
    pabs: Dict[str, float] = field(default_factory=dict)  # r
    walras: float = 0.0
    
    @classmethod
    def from_python_model(cls, model) -> "GTAPVariableSnapshot":
        """Extract snapshot from Python Pyomo model."""
        from pyomo.environ import value
        
        def extract_var(var):
            """Extract variable values as dictionary."""
            result = {}
            for idx in var:
                try:
                    result[idx] = float(value(var[idx]))
                except:
                    result[idx] = 0.0
            return result
        
        return cls(
            xp=extract_var(model.xp) if hasattr(model, 'xp') else {},
            x=extract_var(model.x) if hasattr(model, 'x') else {},
            px=extract_var(model.px) if hasattr(model, 'px') else {},
            pp=extract_var(model.pp) if hasattr(model, 'pp') else {},
            ps=extract_var(model.ps) if hasattr(model, 'ps') else {},
            pd=extract_var(model.pd) if hasattr(model, 'pd') else {},
            pa=extract_var(model.pa) if hasattr(model, 'pa') else {},
            pmt=extract_var(model.pmt) if hasattr(model, 'pmt') else {},
            pet=extract_var(model.pet) if hasattr(model, 'pet') else {},
            xe=extract_var(model.xe) if hasattr(model, 'xe') else {},
            xw=extract_var(model.xw) if hasattr(model, 'xw') else {},
            xmt=extract_var(model.xmt) if hasattr(model, 'xmt') else {},
            xet=extract_var(model.xet) if hasattr(model, 'xet') else {},
            xf=extract_var(model.xf) if hasattr(model, 'xf') else {},
            xft=extract_var(model.xft) if hasattr(model, 'xft') else {},
            pf=extract_var(model.pf) if hasattr(model, 'pf') else {},
            pft=extract_var(model.pft) if hasattr(model, 'pft') else {},
            xc=extract_var(model.xc) if hasattr(model, 'xc') else {},
            xg=extract_var(model.xg) if hasattr(model, 'xg') else {},
            xi=extract_var(model.xi) if hasattr(model, 'xi') else {},
            regy=extract_var(model.regy) if hasattr(model, 'regy') else {},
            yc=extract_var(model.yc) if hasattr(model, 'yc') else {},
            yg=extract_var(model.yg) if hasattr(model, 'yg') else {},
            yi=extract_var(model.yi) if hasattr(model, 'yi') else {},
            pnum=float(value(model.pnum)) if hasattr(model, 'pnum') else 1.0,
            pabs=extract_var(model.pabs) if hasattr(model, 'pabs') else {},
            walras=float(value(model.walras)) if hasattr(model, 'walras') else 0.0,
        )
    
    @classmethod
    def from_gdx(cls, gdx_path: Path, sets: GTAPSets) -> "GTAPVariableSnapshot":
        """Extract snapshot from GAMS GDX results file."""
        gdx_data = read_gdx(gdx_path)
        
        def read_var(name: str, ndim: int):
            """Read variable from GDX."""
            try:
                return read_parameter_values(gdx_data, name)
            except:
                return {}
        
        return cls(
            xp=read_var("xp", 2),
            x=read_var("x", 3),
            px=read_var("px", 2),
            pp=read_var("pp", 2),
            ps=read_var("ps", 2),
            pd=read_var("pd", 2),
            pa=read_var("pa", 2),
            pmt=read_var("pmt", 2),
            pet=read_var("pet", 2),
            xe=read_var("xe", 3),
            xw=read_var("xw", 3),
            xmt=read_var("xmt", 2),
            xet=read_var("xet", 2),
            xf=read_var("xf", 3),
            xft=read_var("xft", 2),
            pf=read_var("pf", 3),
            pft=read_var("pft", 2),
            xc=read_var("xc", 2),
            xg=read_var("xg", 2),
            xi=read_var("xi", 2),
            regy=read_var("regy", 1),
            yc=read_var("yc", 1),
            yg=read_var("yg", 1),
            yi=read_var("yi", 1),
            pnum=read_var("pnum", 0).get((), 1.0),
            pabs=read_var("pabs", 1),
            walras=read_var("walras", 0).get((), 0.0),
        )


@dataclass(frozen=True)
class GTAPParityComparison:
    """Parity comparison result between Python and GAMS.
    
    Attributes:
        passed: Whether parity check passed
        tolerance: Tolerance used for comparison
        n_variables_compared: Number of variables compared
        n_mismatches: Number of mismatches found
        max_abs_diff: Maximum absolute difference
        max_rel_diff: Maximum relative difference
        mismatches: Detailed mismatch information
        summary: Summary statistics
    """
    passed: bool
    tolerance: float
    n_variables_compared: int
    n_mismatches: int
    max_abs_diff: float
    max_rel_diff: float
    mismatches: List[Dict[str, Any]]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class GTAPGAMSReference:
    """Reference data from GAMS CGEBox run.
    
    Contains benchmark values and solution from GAMS.
    """
    gdx_path: Path
    sets: GTAPSets
    snapshot: GTAPVariableSnapshot
    modelstat: float
    solvestat: float
    solve_time: float
    
    @classmethod
    def load(cls, gdx_path: Path, sets: Optional[GTAPSets] = None) -> "GTAPGAMSReference":
        """Load GAMS reference from GDX file.
        
        Args:
            gdx_path: Path to GAMS results GDX
            sets: GTAP sets (loaded from data GDX if not provided)
            
        Returns:
            GTAPGAMSReference instance
        """
        gdx_path = Path(gdx_path)
        
        if sets is None:
            # Try to find data GDX in same directory
            data_gdx = gdx_path.parent / f"{gdx_path.stem.split('_')[0]}.gdx"
            if not data_gdx.exists():
                # Try common names
                for name in ["asa7x5.gdx", "gtap.gdx", "data.gdx"]:
                    data_gdx = gdx_path.parent / name
                    if data_gdx.exists():
                        break
            
            if data_gdx.exists():
                sets = GTAPSets()
                sets.load_from_gdx(data_gdx)
            else:
                raise FileNotFoundError(f"Could not find data GDX for {gdx_path}")
        
        # Load snapshot
        snapshot = GTAPVariableSnapshot.from_gdx(gdx_path, sets)
        
        # Read model status
        gdx_data = read_gdx(gdx_path)
        modelstat = 1.0
        solvestat = 1.0
        solve_time = 0.0
        
        try:
            modelstat_data = read_parameter_values(gdx_data, "modelstat")
            if modelstat_data:
                modelstat = list(modelstat_data.values())[0]
        except:
            pass
            
        try:
            solvestat_data = read_parameter_values(gdx_data, "solvestat")
            if solvestat_data:
                solvestat = list(solvestat_data.values())[0]
        except:
            pass
        
        return cls(
            gdx_path=gdx_path,
            sets=sets,
            snapshot=snapshot,
            modelstat=float(modelstat),
            solvestat=float(solvestat),
            solve_time=solve_time,
        )


def compare_variable_groups(
    python: Dict,
    gams: Dict,
    group_name: str,
    tolerance: float = 1e-6,
) -> Tuple[int, int, float, List[Dict]]:
    """Compare a group of variables.
    
    Args:
        python: Python variable values
        gams: GAMS variable values
        group_name: Name of variable group
        tolerance: Tolerance for comparison
        
    Returns:
        (n_compared, n_mismatches, max_diff, mismatch_details)
    """
    n_compared = 0
    n_mismatches = 0
    max_diff = 0.0
    mismatches = []
    
    # Get all keys
    all_keys = set(python.keys()) | set(gams.keys())
    
    for key in all_keys:
        py_val = python.get(key, 0.0)
        gams_val = gams.get(key, 0.0)
        
        # Skip if both are zero or missing
        if py_val == 0.0 and gams_val == 0.0:
            continue
        
        n_compared += 1
        
        # Calculate difference
        abs_diff = abs(py_val - gams_val)
        rel_diff = abs_diff / max(abs(gams_val), 1e-10)
        
        max_diff = max(max_diff, abs_diff)
        
        # Check if mismatch
        if abs_diff > tolerance:
            n_mismatches += 1
            mismatches.append({
                "group": group_name,
                "key": key,
                "python": py_val,
                "gams": gams_val,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            })
    
    return n_compared, n_mismatches, max_diff, mismatches


def compare_gtap_gams_parity(
    python_model,
    gams_reference: GTAPGAMSReference,
    tolerance: float = 1e-6,
) -> GTAPParityComparison:
    """Compare Python model against GAMS reference.
    
    Args:
        python_model: Python Pyomo model (or GTAPVariableSnapshot)
        gams_reference: GAMS reference data
        tolerance: Tolerance for comparison
        
    Returns:
        GTAPParityComparison with results
    """
    # Extract Python snapshot
    if isinstance(python_model, GTAPVariableSnapshot):
        py_snapshot = python_model
    else:
        py_snapshot = GTAPVariableSnapshot.from_python_model(python_model)
    
    gams_snapshot = gams_reference.snapshot
    
    # Compare all variable groups
    all_mismatches = []
    n_compared = 0
    n_mismatches = 0
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    
    variable_groups = [
        ("xp", py_snapshot.xp, gams_snapshot.xp),
        ("px", py_snapshot.px, gams_snapshot.px),
        ("pp", py_snapshot.pp, gams_snapshot.pp),
        ("ps", py_snapshot.ps, gams_snapshot.ps),
        ("pd", py_snapshot.pd, gams_snapshot.pd),
        ("pa", py_snapshot.pa, gams_snapshot.pa),
        ("pmt", py_snapshot.pmt, gams_snapshot.pmt),
        ("pet", py_snapshot.pet, gams_snapshot.pet),
        ("xe", py_snapshot.xe, gams_snapshot.xe),
        ("xw", py_snapshot.xw, gams_snapshot.xw),
        ("xmt", py_snapshot.xmt, gams_snapshot.xmt),
        ("xet", py_snapshot.xet, gams_snapshot.xet),
        ("xf", py_snapshot.xf, gams_snapshot.xf),
        ("xft", py_snapshot.xft, gams_snapshot.xft),
        ("pf", py_snapshot.pf, gams_snapshot.pf),
        ("pft", py_snapshot.pft, gams_snapshot.pft),
        ("xc", py_snapshot.xc, gams_snapshot.xc),
        ("xg", py_snapshot.xg, gams_snapshot.xg),
        ("xi", py_snapshot.xi, gams_snapshot.xi),
        ("regy", py_snapshot.regy, gams_snapshot.regy),
        ("pabs", py_snapshot.pabs, gams_snapshot.pabs),
    ]
    
    for group_name, py_vals, gams_vals in variable_groups:
        comp, mism, max_d, details = compare_variable_groups(
            py_vals, gams_vals, group_name, tolerance
        )
        n_compared += comp
        n_mismatches += mism
        max_abs_diff = max(max_abs_diff, max_d)
        all_mismatches.extend(details)
        
        # Calculate max relative diff
        for detail in details:
            max_rel_diff = max(max_rel_diff, detail.get("rel_diff", 0))
    
    # Check scalar variables
    if abs(py_snapshot.pnum - gams_snapshot.pnum) > tolerance:
        n_mismatches += 1
        all_mismatches.append({
            "group": "pnum",
            "key": (),
            "python": py_snapshot.pnum,
            "gams": gams_snapshot.pnum,
            "abs_diff": abs(py_snapshot.pnum - gams_snapshot.pnum),
            "rel_diff": 0.0,
        })
    n_compared += 1
    
    if abs(py_snapshot.walras - gams_snapshot.walras) > tolerance:
        n_mismatches += 1
        all_mismatches.append({
            "group": "walras",
            "key": (),
            "python": py_snapshot.walras,
            "gams": gams_snapshot.walras,
            "abs_diff": abs(py_snapshot.walras - gams_snapshot.walras),
            "rel_diff": 0.0,
        })
    n_compared += 1
    
    # Sort mismatches by absolute difference
    all_mismatches.sort(key=lambda x: x["abs_diff"], reverse=True)
    
    # Determine pass/fail
    passed = n_mismatches == 0
    
    # Build summary
    summary = {
        "gams_modelstat": gams_reference.modelstat,
        "gams_solvestat": gams_reference.solvestat,
        "n_variables": n_compared,
        "n_mismatches": n_mismatches,
        "mismatch_rate": n_mismatches / max(n_compared, 1) * 100,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
    }
    
    return GTAPParityComparison(
        passed=passed,
        tolerance=tolerance,
        n_variables_compared=n_compared,
        n_mismatches=n_mismatches,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        mismatches=all_mismatches[:50],  # Keep top 50
        summary=summary,
    )


class GTAPParityRunner:
    """Runner for GTAP parity checks.
    
    This class runs both Python and GAMS models and compares results.
    
    Example:
        runner = GTAPParityRunner(
            gdx_file="asa7x5.gdx",
            gams_results_gdx="gams_results.gdx",
        )
        result = runner.run_parity_check()
        
        if result.passed:
            print("✓ Parity check passed!")
        else:
            print(f"✗ {result.n_mismatches} mismatches found")
            for m in result.mismatches[:10]:
                print(f"  {m['group']}{m['key']}: diff={m['abs_diff']:.2e}")
    """
    
    def __init__(
        self,
        gdx_file: Path,
        gams_results_gdx: Optional[Path] = None,
        closure: str = "gtap_standard",
        solver: str = "ipopt",
        tolerance: float = 1e-6,
    ):
        """Initialize parity runner.
        
        Args:
            gdx_file: Path to GTAP data GDX
            gams_results_gdx: Path to GAMS results GDX (optional)
            closure: Closure type
            solver: Solver to use
            tolerance: Tolerance for comparison
        """
        self.gdx_file = Path(gdx_file)
        self.gams_results_gdx = Path(gams_results_gdx) if gams_results_gdx else None
        self.closure = closure
        self.solver = solver
        self.tolerance = tolerance
        
        # Load data
        self.sets = GTAPSets()
        self.sets.load_from_gdx(self.gdx_file)
        
        self.params = GTAPParameters()
        self.params.load_from_gdx(self.gdx_file)
        
        # Build model
        self.contract = build_gtap_contract(closure)
        self.equations = GTAPModelEquations(self.sets, self.params, self.contract.closure)
        self.model = self.equations.build_model()
        
        # Load GAMS reference if available
        self.gams_reference = None
        if self.gams_results_gdx and self.gams_results_gdx.exists():
            self.gams_reference = GTAPGAMSReference.load(self.gams_results_gdx, self.sets)
    
    def run_python(self) -> SolverResult:
        """Run Python model and return result."""
        logger.info("Running Python GTAP model...")
        
        solver = GTAPSolver(self.model, self.contract.closure, solver_name=self.solver)
        result = solver.solve()
        
        logger.info(f"Python solve: {result.status.value}, Walras={result.walras_value:.2e}")
        return result
    
    def run_gams(self, gams_script: Optional[Path] = None) -> GTAPGAMSReference:
        """Run GAMS model and return reference.
        
        Args:
            gams_script: Path to GAMS script (optional)
            
        Returns:
            GTAPGAMSReference
        """
        if gams_script and gams_script.exists():
            logger.info(f"Running GAMS script: {gams_script}")
            # TODO: Implement GAMS execution
            raise NotImplementedError("GAMS execution not yet implemented")
        elif self.gams_reference:
            logger.info("Using existing GAMS results")
            return self.gams_reference
        else:
            raise ValueError("No GAMS results available")
    
    def run_parity_check(self) -> GTAPParityComparison:
        """Run full parity check.
        
        Returns:
            GTAPParityComparison
        """
        # Run Python
        py_result = self.run_python()
        
        if not py_result.success:
            logger.error(f"Python solve failed: {py_result.message}")
            return GTAPParityComparison(
                passed=False,
                tolerance=self.tolerance,
                n_variables_compared=0,
                n_mismatches=0,
                max_abs_diff=0.0,
                max_rel_diff=0.0,
                mismatches=[],
                summary={"error": "Python solve failed", "message": py_result.message},
            )
        
        # Check GAMS reference
        if not self.gams_reference:
            logger.error("No GAMS reference available")
            return GTAPParityComparison(
                passed=False,
                tolerance=self.tolerance,
                n_variables_compared=0,
                n_mismatches=0,
                max_abs_diff=0.0,
                max_rel_diff=0.0,
                mismatches=[],
                summary={"error": "No GAMS reference"},
            )
        
        # Compare
        logger.info("Comparing Python vs GAMS...")
        comparison = compare_gtap_gams_parity(
            self.model,
            self.gams_reference,
            tolerance=self.tolerance,
        )
        
        logger.info(f"Parity check: {comparison.n_mismatches} mismatches, max_diff={comparison.max_abs_diff:.2e}")
        
        return comparison
    
    def generate_report(self, comparison: GTAPParityComparison) -> str:
        """Generate human-readable parity report."""
        lines = []
        lines.append("=" * 70)
        lines.append("GTAP Parity Check Report")
        lines.append("=" * 70)
        lines.append(f"Data file: {self.gdx_file}")
        lines.append(f"GAMS results: {self.gams_results_gdx}")
        lines.append(f"Closure: {self.closure}")
        lines.append(f"Tolerance: {self.tolerance}")
        lines.append("")
        lines.append(f"Status: {'✓ PASSED' if comparison.passed else '✗ FAILED'}")
        lines.append(f"Variables compared: {comparison.n_variables_compared}")
        lines.append(f"Mismatches: {comparison.n_mismatches}")
        lines.append(f"Max absolute diff: {comparison.max_abs_diff:.2e}")
        lines.append(f"Max relative diff: {comparison.max_rel_diff:.2e}")
        lines.append("")
        
        if comparison.mismatches:
            lines.append("Top Mismatches:")
            lines.append("-" * 70)
            for m in comparison.mismatches[:20]:
                key_str = str(m['key']) if isinstance(m['key'], tuple) else m['key']
                lines.append(
                    f"  {m['group']}{key_str:20s} "
                    f"Python={m['python']:12.6f} "
                    f"GAMS={m['gams']:12.6f} "
                    f"Diff={m['abs_diff']:12.6e}"
                )
        
        lines.append("=" * 70)
        return "\n".join(lines)


def load_gtap_gams_reference(gdx_path: Path, sets: Optional[GTAPSets] = None) -> GTAPGAMSReference:
    """Load GAMS reference from GDX file.
    
    Args:
        gdx_path: Path to GAMS results GDX
        sets: GTAP sets (optional)
        
    Returns:
        GTAPGAMSReference
    """
    return GTAPGAMSReference.load(gdx_path, sets)


def run_gtap_parity_test(
    gdx_file: Path,
    gams_results_gdx: Path,
    closure: str = "gtap_standard",
    tolerance: float = 1e-6,
    output_file: Optional[Path] = None,
) -> GTAPParityComparison:
    """Run a complete GTAP parity test.
    
    This is a convenience function for running parity tests.
    
    Args:
        gdx_file: Path to GTAP data GDX
        gams_results_gdx: Path to GAMS results GDX
        closure: Closure type
        tolerance: Tolerance
        output_file: Optional output file for report
        
    Returns:
        GTAPParityComparison
    """
    runner = GTAPParityRunner(
        gdx_file=gdx_file,
        gams_results_gdx=gams_results_gdx,
        closure=closure,
        tolerance=tolerance,
    )
    
    result = runner.run_parity_check()
    
    # Generate report
    report = runner.generate_report(result)
    print(report)
    
    # Save to file if requested
    if output_file:
        output_file = Path(output_file)
        output_file.write_text(report)
        print(f"\nReport saved to: {output_file}")
    
    return result
