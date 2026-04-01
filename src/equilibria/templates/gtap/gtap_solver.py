"""GTAP Solver (Standard GTAP 7)

This module implements the solver for GTAP CGE models.
Supports multiple solvers:
- IPOPT: Interior Point OPTimizer (default, for NLP/CNS)
- PATH: Mixed Complementarity Problem solver (for MCP)

Reference: /Users/marmol/proyectos2/cge_babel/standard_gtap_7/comp.gms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from pyomo.environ import ConcreteModel, SolverFactory, TerminationCondition, value
    from pyomo.opt import SolverStatus
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False

from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

logger = logging.getLogger(__name__)


class SolverStatus(Enum):
    """Solver status enumeration."""
    CONVERGED = "converged"
    ITERATION_LIMIT = "iteration_limit"
    TIME_LIMIT = "time_limit"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NUMERICAL_ISSUES = "numerical_issues"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class SolverResult:
    """Result from solver execution.
    
    Attributes:
        status: Solver status
        termination_condition: Pyomo termination condition
        objective_value: Final objective value (if applicable)
        solve_time: Time taken to solve (seconds)
        iterations: Number of iterations
        residual: Final residual norm
        walras_value: Value of Walras check (should be ~0)
        variables: Dictionary of final variable values
        success: Whether solve was successful
        message: Status message
    """
    status: SolverStatus
    termination_condition: Optional[str] = None
    objective_value: Optional[float] = None
    solve_time: float = 0.0
    iterations: int = 0
    residual: float = float('inf')
    walras_value: float = float('inf')
    variables: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    message: str = ""


class GTAPSolver:
    """Solver for GTAP CGE models.
    
    This class handles the solution of GTAP models using various
    solvers including IPOPT (default) and PATH.
    
    Attributes:
        model: Pyomo ConcreteModel
        closure: Closure configuration
        solver_name: Name of solver to use
        solver_options: Solver-specific options
        
    Example:
        >>> solver = GTAPSolver(model, closure, solver_name="ipopt")
        >>> result = solver.solve()
        >>> if result.success:
        ...     print(f"Converged in {result.iterations} iterations")
    """
    
    def __init__(
        self,
        model: "ConcreteModel",
        closure: Optional[GTAPClosureConfig] = None,
        solver_name: str = "ipopt",
        solver_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize GTAP solver.
        
        Args:
            model: Pyomo ConcreteModel with GTAP equations
            closure: Closure configuration (optional)
            solver_name: Solver to use ("ipopt", "path", "conopt")
            solver_options: Solver-specific options
        """
        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo is required. Install with: pip install pyomo")
        
        self.model = model
        self.closure = closure or GTAPClosureConfig()
        self.solver_name = solver_name.lower()
        
        # Default IPOPT options for handling large-scale SAM values (like GAMS)
        default_options = {
            # Scaling and numerical robustness
            'nlp_scaling_method': 'gradient-based',  # Automatic variable/constraint scaling 
            'nlp_scaling_max_gradient': 1e8,         # Handle large SAM values
            'obj_scaling_factor': 1.0,
            
            # Convergence criteria
            'tol': 1e-6,                             # Optimality tolerance
            'acceptable_tol': 1e-4,                  # Acceptable tolerance
            'acceptable_iter': 15,                   # Accept after 15 iterations
            
            # Iteration limits
            'max_iter': 3000,                        # GAMS often needs many iterations
            
            # Linear solver  
            'linear_solver': 'mumps',                # Robust for ill-conditioned systems
            
            # Output control
            'sb': 'yes',                              # Skip banner
            'print_level': 3,                         # Moderate verbosity
            
            # Initialization
            'warm_start_init_point': 'yes',          # Use provided initialization
            
            # Numerical tolerance (like GAMS - tolerate small errors)
            'acceptable_obj_change_tol': 1.0e-6,
        }
        default_options.update(solver_options or {})
        self.solver_options = default_options
        
        # Initialize variables (preserves SAM levels if already set)
        self._initialize_benchmark()
        
    def _initialize_benchmark(self) -> None:
        """Initialize variables that don't have values yet.
        
        Respects SAM-based initialization (millions) from gtap_model_equations.
        This mirrors GAMS behavior: variables initialized at SAM levels.
        """
        from pyomo.environ import Var
        
        # Don't overwrite SAM-based initialization - only set uninitialized vars
        for var in self.model.component_objects(Var, active=True):
            for idx in var:
                if var[idx].value is None:
                    var[idx].value = 1.0

    def apply_solution_hint(self, hint: Any) -> int:
        """Warm-start variables from a snapshot-like object.

        The hint is typically a `GTAPVariableSnapshot` from the parity pipeline.
        Missing variables or indices are ignored.
        """
        hint_to_model = {
            "xp": "xp",
            "x": "x",
            "xs": "xs",
            "xds": "xds",
            "xd": "xda",
            "px": "px",
            "pp": "pp",
            "ps": "ps",
            "pd": "pd",
            "pa": "pa",
            "paa": "paa",
            "pdp": "pdp",
            "pmt": "pmt",
            "pmcif": "pmcif",
            "pet": "pet",
            "pe": "pe",
            "pefob": "pefob",
            "xe": "xe",
            "xw": "xw",
            "xmt": "xmt",
            "xet": "xet",
            "xaa": "xaa",
            "xwmg": "xwmg",
            "xmgm": "xmgm",
            "pwmg": "pwmg",
            "xtmg": "xtmg",
            "ptmg": "ptmg",
            "xf": "xf",
            "xft": "xft",
            "pf": "pf",
            "pft": "pft",
            "xc": "xc",
            "xg": "xg",
            "xi": "xi",
            "regy": "regy",
            "yc": "yc",
            "yg": "yg",
            "yi": "yi",
            "pabs": "pabs",
            "pnum": "pnum",
        }

        applied = 0
        for hint_name, model_name in hint_to_model.items():
            if not hasattr(hint, hint_name) or not hasattr(self.model, model_name):
                continue
            values = getattr(hint, hint_name)
            target = getattr(self.model, model_name)

            if values is None:
                continue
            if isinstance(values, dict):
                for idx, value in values.items():
                    if value is None or idx not in target:
                        continue
                    target[idx].value = float(value)
                    applied += 1
                continue
            if hasattr(target, "value"):
                target.value = float(values)
                applied += 1

        logger.info("Applied %s warm-start values", applied)
        return applied
                
    def apply_closure(self, closure: Optional[GTAPClosureConfig] = None) -> None:
        """Apply closure rules to fix exogenous variables.
        
        Args:
            closure: Closure configuration (uses self.closure if None)
        """
        closure = closure or self.closure
        
        logger.info(f"Applying closure: {closure.name}")
        
        # Fix variables according to closure
        fixed_count = 0
        
        # Fix numeraire
        if closure.numeraire and hasattr(self.model, closure.numeraire):
            var = getattr(self.model, closure.numeraire)
            if closure.numeraire_mode == "fixed_benchmark":
                var.fix(1.0)
                fixed_count += 1
                logger.debug(f"Fixed numeraire {closure.numeraire} = 1.0")
        
        # Fix technology parameters
        if closure.fix_technology:
            tech_vars = ["axp", "lambdaio", "lambdaf", "lambdaxm", "lambdam"]
            for name in tech_vars:
                if hasattr(self.model, name):
                    var = getattr(self.model, name)
                    for idx in var:
                        var[idx].fix(1.0)
                        fixed_count += 1
            logger.debug(f"Fixed {len(tech_vars)} technology parameters")
        
        # Fix tax rates
        if closure.fix_taxes:
            tax_vars = ["rto", "rtf", "rtms", "rtxs"]
            for name in tax_vars:
                if hasattr(self.model, name):
                    var = getattr(self.model, name)
                    for idx in var:
                        if not hasattr(var[idx], "fix"):
                            continue
                        if var[idx].value is not None:
                            var[idx].fix()
                            fixed_count += 1
            logger.debug(f"Fixed tax rates")
        
        # Fix endowments
        if closure.fix_endowments:
            if hasattr(self.model, "xft"):
                for idx in self.model.xft:
                    self.model.xft[idx].fix(1.0)
                    fixed_count += 1
            logger.debug(f"Fixed factor endowments")
        
        logger.info(f"Closure applied: {fixed_count} variables fixed")
        
    def solve(
        self,
        warm_start: bool = True,
        tee: bool = False,
        keepfiles: bool = False,
        report_timing: bool = False,
    ) -> SolverResult:
        """Solve the GTAP model.
        
        Args:
            warm_start: Use warm start if available
            tee: Display solver output
            keepfiles: Keep temporary files
            report_timing: Report timing information
            
        Returns:
            SolverResult with solution status and values
        """
        import time
        
        start_time = time.time()
        
        # Apply closure
        self.apply_closure()
        
        # Get solver
        solver = SolverFactory(self.solver_name)
        
        if solver is None:
            raise RuntimeError(f"Solver '{self.solver_name}' not available")
        
        # Set defaults first, then override with user options
        if self.solver_name == "ipopt":
            solver.options.setdefault("mu_strategy", "adaptive")
            solver.options.setdefault("output_file", "ipopt.out")
        elif self.solver_name == "path":
            solver.options.setdefault("convergence_tolerance", 1e-6)
            solver.options.setdefault("major_iterations_limit", 500)
        
        # Apply solver options (overrides defaults)
        for key, value in self.solver_options.items():
            solver.options[key] = value
        
        logger.info(f"Solving with {self.solver_name}...")
        
        # Solve
        try:
            results = solver.solve(
                self.model,
                tee=tee,
                keepfiles=keepfiles,
                symbolic_solver_labels=True,
            )
            
            solve_time = time.time() - start_time
            
            # Process results
            return self._process_results(results, solve_time)
            
        except Exception as e:
            logger.error(f"Solve failed: {e}")
            return SolverResult(
                status=SolverStatus.ERROR,
                success=False,
                message=str(e),
                solve_time=time.time() - start_time,
            )
    
    def _process_results(self, results: Any, solve_time: float) -> SolverResult:
        """Process solver results.
        
        Args:
            results: Pyomo solver results
            solve_time: Time taken to solve
            
        Returns:
            SolverResult
        """
        from pyomo.opt import TerminationCondition, SolverStatus as PyomoSolverStatus
        
        # Get termination condition
        term_cond = results.solver.termination_condition
        solver_status = results.solver.status
        
        # Determine success
        success = term_cond in (
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        )
        
        # Map to our status
        if success:
            status = SolverStatus.CONVERGED
        elif term_cond == TerminationCondition.maxIterations:
            status = SolverStatus.ITERATION_LIMIT
        elif term_cond == TerminationCondition.maxTimeLimit:
            status = SolverStatus.TIME_LIMIT
        elif term_cond == TerminationCondition.infeasible:
            status = SolverStatus.INFEASIBLE
        elif term_cond == TerminationCondition.unbounded:
            status = SolverStatus.UNBOUNDED
        else:
            status = SolverStatus.UNKNOWN
        
        # Get objective value
        obj_value = None
        if hasattr(self.model, "OBJ"):
            obj_value = value(self.model.OBJ)
        
        # Get Walras value
        walras_val = float('inf')
        if hasattr(self.model, "walras"):
            walras_val = value(self.model.walras)
        
        # Get iterations
        iterations = results.solver.get("iterations", 0) or 0
        
        # Extract variable values
        variables = self._extract_variable_values()
        
        # Build message
        message = f"Solver {solver_status}, Termination: {term_cond}"
        if success:
            message = f"Converged in {iterations} iterations, Walras = {walras_val:.2e}"
        
        return SolverResult(
            status=status,
            termination_condition=str(term_cond),
            objective_value=obj_value,
            solve_time=solve_time,
            iterations=iterations,
            walras_value=walras_val,
            variables=variables,
            success=success,
            message=message,
        )
    
    def _extract_variable_values(self) -> Dict[str, Any]:
        """Extract variable values from model.
        
        Returns:
            Dictionary of variable values
        """
        from pyomo.environ import Var
        
        variables = {}
        
        for var in self.model.component_objects(Var, active=True):
            var_name = var.name
            var_values = {}
            
            for idx in var:
                try:
                    var_values[idx] = value(var[idx])
                except:
                    var_values[idx] = None
            
            variables[var_name] = var_values
        
        return variables
    
    def apply_shock(self, shock: Dict[str, Any]) -> None:
        """Apply a shock to the model.
        
        Args:
            shock: Dictionary describing the shock
                  Format: {"variable": name, "index": (r, i, ...), "value": new_value}
                  or list of such dictionaries
        """
        if isinstance(shock, dict) and "variable" in shock:
            shock = [shock]
        
        for s in shock:
            var_name = s["variable"]
            index = s.get("index")
            value = s["value"]
            
            if not hasattr(self.model, var_name):
                logger.warning(f"Variable {var_name} not found in model")
                continue
            
            var = getattr(self.model, var_name)
            
            if index is not None:
                if index in var:
                    var[index].value = value
                    logger.info(f"Applied shock: {var_name}[{index}] = {value}")
                else:
                    logger.warning(f"Index {index} not found in {var_name}")
            else:
                # Apply to all indices
                for idx in var:
                    var[idx].value = value
                logger.info(f"Applied shock: {var_name}[*] = {value}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of model and solution.
        
        Returns:
            Dictionary with model summary
        """
        from pyomo.environ import Var, Constraint
        
        n_vars = sum(1 for _ in self.model.component_objects(Var, active=True))
        n_constr = sum(1 for _ in self.model.component_objects(Constraint, active=True))
        
        return {
            "solver": self.solver_name,
            "closure": self.closure.name,
            "n_variables": n_vars,
            "n_constraints": n_constr,
            "has_solution": any(
                v.value is not None 
                for v in self.model.component_objects(Var, active=True)
            ),
        }
