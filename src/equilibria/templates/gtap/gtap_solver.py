"""GTAP Solver (Standard GTAP 7)

This module implements the solver for GTAP CGE models.
Supports multiple solvers:
- IPOPT: Interior Point OPTimizer (default, for NLP/CNS)
- PATH: Mixed Complementarity Problem solver (for MCP)
- PATH-CAPI: Compatibility alias that routes through PATH backend in GTAPSolver

Reference: /Users/marmol/proyectos2/cge_babel/standard_gtap_7/comp.gms
"""

from __future__ import annotations

import logging
import platform
import subprocess
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
from equilibria.templates.gtap.gtap_parameters import GTAPParameters

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
        params: Optional["GTAPParameters"] = None,
    ):
        """Initialize GTAP solver.
        
        Args:
            model: Pyomo ConcreteModel with GTAP equations
            closure: Closure configuration (optional)
            solver_name: Solver to use ("ipopt", "path", "conopt", "path-capi")
            solver_options: Solver-specific options
            params: GTAP parameters for conditional fixing based on data
        """
        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo is required. Install with: pip install pyomo")
        
        self.model = model
        self.closure = closure or GTAPClosureConfig()
        self.params = params  # GTAP parameters for conditional fixing
        self.requested_solver_name = solver_name.lower()
        self.solver_name = self._normalize_solver_name(self.requested_solver_name)

        if self.requested_solver_name != self.solver_name:
            logger.info(
                "GTAPSolver normalized solver '%s' to backend '%s'",
                self.requested_solver_name,
                self.solver_name,
            )
        
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

    @staticmethod
    def _normalize_solver_name(solver_name: str) -> str:
        """Normalize user-facing solver aliases to Pyomo backend names."""
        name = (solver_name or "ipopt").lower().strip()
        aliases = {
            "path-capi": "path",
            "path_capi": "path",
        }
        return aliases.get(name, name)
        
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
            "ytax": "ytax",
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
                    component = target[idx]
                    # Skip already-fixed variables so we don't override values
                    # set by apply_aggressive_fixing_for_mcp at shocked init levels.
                    if getattr(component, "fixed", False):
                        continue
                    if hasattr(component, "set_value"):
                        component.set_value(float(value))
                        applied += 1
                    elif hasattr(component, "value"):
                        component.value = float(value)
                        applied += 1
                continue
            if hasattr(target, "set_value"):
                target.set_value(float(values))
                applied += 1
            elif hasattr(target, "value"):
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
                        component = var[idx]
                        if not hasattr(component, "fix"):
                            continue
                        component.fix(1.0)
                        fixed_count += 1
            logger.debug(f"Fixed {len(tech_vars)} technology parameters")
        
        # Fix tax rates
        if closure.fix_taxes:
            tax_vars = ["rto", "rtf", "rtms", "rtxs", "mtax", "etax", "dintx", "mintx"]
            for name in tax_vars:
                if hasattr(self.model, name):
                    var = getattr(self.model, name)
                    for idx in var:
                        if not hasattr(var[idx], "fix"):
                            continue
                        if var[idx].value is not None:
                            var[idx].fix()
                            fixed_count += 1

            # These wedges are exogenous in the standard GTAP closure. Keep
            # their defining equalities out of the active MCP when the wedges
            # themselves are fixed, otherwise PATH can move them numerically
            # before the equality is enforced.
            for con_name in ("eq_dintxeq", "eq_mintxeq"):
                if hasattr(self.model, con_name):
                    getattr(self.model, con_name).deactivate()
            logger.debug(f"Fixed tax rates")
        
        # Fix endowments
        if closure.fix_endowments:
            if hasattr(self.model, "xft"):
                for idx in self.model.xft:
                    level = self.model.xft[idx].value
                    lb = self.model.xft[idx].lb
                    fixed_level = 1.0 if level is None else float(level)
                    if lb is not None:
                        fixed_level = max(fixed_level, float(lb))
                    self.model.xft[idx].fix(fixed_level)
                    fixed_count += 1
            logger.debug(f"Fixed factor endowments")

        if getattr(closure, "apply_flag_fixing", False):
            fixed_count += self.apply_conditional_fixing()

        if getattr(closure, "close_mcp_gap", False):
            if self.requested_solver_name in {"path", "path-capi", "path_capi"}:
                fixed_count += self.apply_aggressive_fixing_for_mcp()
        
        logger.info(f"Closure applied: {fixed_count} variables fixed")

    def apply_conditional_fixing(self) -> int:
        """Apply conditional fixing based on SAM data (like GAMS).
        
        This fixes variables where no underlying flow exists in the data:
        - Trade variables (xw, pe, pm, pmcif, pefob, etc.) when VXSB=0
        - Factor variables (xf, pf) when VFM/EVFB=0  
        - Output variables (x) when VOA/VOM=0
        
        This mirrors GAMS behavior where variables are fixed at zero/numeraire
        when the corresponding data flow is zero.
        
        Returns:
            Number of variables fixed
        """
        from pyomo.environ import value

        if self.params is None:
            logger.debug("No params provided, skipping conditional fixing")
            return 0
        
        fixed_count = 0
        benchmark = self.params.benchmark
        
        # ===================================================================
        # 1. Create xwFlag: bilateral trade exists if VXSB(r,i,rp) > 0
        #    (including diagonal routes when present, matching GAMS).
        # ===================================================================
        xw_flag = set()
        benchmark_vxsb = getattr(benchmark, 'vxsb', {})
        for (r, i, rp), val in benchmark_vxsb.items():
            if val is not None and float(val) > 0:
                xw_flag.add((r, i, rp))

        # Some GTAP bundles only provide the equilibrium snapshot through
        # `apply_equilibrium_snapshot()`, which populates the import shares
        # (`p_amw`) but leaves `vxsb` empty. In that case, use the snapshot
        # import-share support as the route activation mask instead of
        # fixing every bilateral trade flow to zero.
        if not xw_flag and hasattr(self.params, "shares"):
            snapshot_import_share = getattr(self.params.shares, "p_amw", {})
            for (r, i, rp), share in snapshot_import_share.items():
                if share is not None and float(share) > 0.0:
                    xw_flag.add((r, i, rp))
        
        logger.debug(f"Trade flows (xwFlag): {len(xw_flag)} active route pairs")
        
        def _fix_to_zero_with_lb(var, idx) -> None:
            lb = var[idx].lb
            if lb is not None and float(lb) > 0.0:
                var[idx].setlb(0.0)
            var[idx].fix(0.0)

        # Fix trade variables where NO bilateral trade exists
        # Quantities -> 0, prices -> 1.0 (GAMS-style defaults for inactive routes)
        trade_qty_vars = ["xw"]
        trade_price_vars = ["pe", "pm", "pmcif", "pefob"]

        for var_name in trade_qty_vars:
            if not hasattr(self.model, var_name):
                continue
            var = getattr(self.model, var_name)
            for idx in var:
                # idx expected to be (r, i, rp) or similar
                if len(idx) >= 3:
                    r, i, rp = idx[0], idx[1], idx[2]
                    key = (r, i, rp)
                    if key not in xw_flag:
                        _fix_to_zero_with_lb(var, idx)
                        fixed_count += 1

        for var_name in trade_price_vars:
            if not hasattr(self.model, var_name):
                continue
            var = getattr(self.model, var_name)
            for idx in var:
                if len(idx) >= 3:
                    r, i, rp = idx[0], idx[1], idx[2]
                    key = (r, i, rp)
                    if key not in xw_flag:
                        var[idx].fix(1.0)
                        fixed_count += 1
        
        # Fix bilateral taxes where no trade
        tax_trade_vars = ["imptx", "exptx"]
        for var_name in tax_trade_vars:
            if not hasattr(self.model, var_name):
                continue
            var = getattr(self.model, var_name)
            for idx in var:
                if len(idx) >= 3:
                    r, i, rp = idx[0], idx[1], idx[2]
                    key = (r, i, rp)
                    if key not in xw_flag:
                        # Fix tax rates at current level
                        val = var[idx].value if var[idx].value is not None else 0.0
                        var[idx].fix(float(val))
                        fixed_count += 1

        # xe is a legacy Pyomo-only helper. Keep it fixed at its initialized
        # benchmark level so it does not participate in the active MCP.
        if hasattr(self.model, "xe"):
            for idx in self.model.xe:
                val = self.model.xe[idx].value if self.model.xe[idx].value is not None else 0.0
                self.model.xe[idx].fix(float(val))
                fixed_count += 1

        # Fix trade-margin variables when the route is inactive. This mirrors the
        # GAMS tmgFlag/amgm guards: no intra-regional TT route, no margin demand,
        # and no free margin price on routes with zero benchmark margins.
        if hasattr(self.model, "xwmg"):
            for idx in self.model.xwmg:
                r, i, rp = idx
                tmarg = float(value(self.model.tmarg[r, i, rp])) if hasattr(self.model, "tmarg") else 0.0
                if tmarg <= 0.0:
                    _fix_to_zero_with_lb(self.model.xwmg, idx)
                    fixed_count += 1

        if hasattr(self.model, "pwmg"):
            for idx in self.model.pwmg:
                r, i, rp = idx
                tmarg = float(value(self.model.tmarg[r, i, rp])) if hasattr(self.model, "tmarg") else 0.0
                if tmarg <= 0.0:
                    _fix_to_zero_with_lb(self.model.pwmg, idx)
                    fixed_count += 1

        if hasattr(self.model, "xmgm"):
            for idx in self.model.xmgm:
                m, r, i, rp = idx
                share = float(value(self.model.amgm[m, r, i, rp])) if hasattr(self.model, "amgm") else 0.0
                if share <= 0.0:
                    _fix_to_zero_with_lb(self.model.xmgm, idx)
                    fixed_count += 1
        
        # ===================================================================
        # 2. Create xfFlag: factor usage exists if VFM(r,f,a) > 0
        # ===================================================================
        xf_flag = set()
        for (r, f, a), val in getattr(benchmark, 'vfm', {}).items():
            if val is not None and float(val) > 0:
                xf_flag.add((r, f, a))
        
        logger.debug(f"Factor usage (xfFlag): {len(xf_flag)} active combinations")
        
        # Fix factor demand/price where factor not used
        # Variables: xf, pf, pfa, pfy
        factor_vars = ["xf", "pf", "pfa", "pfy"]
        for var_name in factor_vars:
            if not hasattr(self.model, var_name):
                continue
            var = getattr(self.model, var_name)
            for idx in var:
                # idx expected to be (r, f, a)
                if len(idx) >= 3:
                    key = (idx[0], idx[1], idx[2])
                    if key not in xf_flag:
                        # Fix at zero for quantities, 1.0 for prices
                        if var_name == "xf":
                            _fix_to_zero_with_lb(var, idx)
                        else:
                            var[idx].fix(1.0)
                        fixed_count += 1
        
        # ===================================================================
        # 3. Create xFlag: output exists if VOM(r,a) > 0
        # ===================================================================
        x_flag = set()
        for (r, a), val in getattr(benchmark, 'vom', {}).items():
            if val is not None and float(val) > 0:
                x_flag.add((r, a))
        
        logger.debug(f"Output (xFlag): {len(x_flag)} active sector-region pairs")
        
        # Fix output variables where no production (activity-level)
        output_vars = ["xp", "va", "nd"]
        for var_name in output_vars:
            if not hasattr(self.model, var_name):
                continue
            var = getattr(self.model, var_name)
            for idx in var:
                # idx expected to be (r, a) or (r, a, i)
                if len(idx) >= 2:
                    key = (idx[0], idx[1])
                    if key not in x_flag:
                        _fix_to_zero_with_lb(var, idx)
                        fixed_count += 1

        # Fix output mapping variables by commodity when no make flow exists
        # Variables: x (quantity), p_rai, pp_rai (prices)
        x_rai_flag = set()
        for (r, a, i), val in getattr(benchmark, 'makb', {}).items():
            if val is not None and abs(float(val)) > 1e-12:
                x_rai_flag.add((r, a, i))

        if hasattr(self.model, "x"):
            for idx in self.model.x:
                if len(idx) >= 3:
                    key = (idx[0], idx[1], idx[2])
                    if key not in x_rai_flag:
                        _fix_to_zero_with_lb(self.model.x, idx)
                        fixed_count += 1

        # GAMS fixes pre-tax make-route prices outside xFlag to their benchmark
        # level. In this template those inactive routes initialize at 1.0.
        if hasattr(self.model, "p_rai"):
            for idx in self.model.p_rai:
                if len(idx) >= 3:
                    key = (idx[0], idx[1], idx[2])
                    if key not in x_rai_flag:
                        val = self.model.p_rai[idx].value if self.model.p_rai[idx].value is not None else 1.0
                        self.model.p_rai[idx].fix(float(val))
                        fixed_count += 1

        # Post-tax make-route prices are also inactive outside xFlag in GAMS.
        if hasattr(self.model, "pp_rai"):
            for idx in self.model.pp_rai:
                if len(idx) >= 3:
                    key = (idx[0], idx[1], idx[2])
                    if key not in x_rai_flag:
                        val = self.model.pp_rai[idx].value if self.model.pp_rai[idx].value is not None else 1.0
                        self.model.pp_rai[idx].fix(float(val))
                        fixed_count += 1

        # The aggregate pp(r,a) variable is a Pyomo-only helper. Keep it fixed
        # at its initialized benchmark level so it does not participate in the MCP.
        if hasattr(self.model, "pp"):
            for idx in self.model.pp:
                val = self.model.pp[idx].value if self.model.pp[idx].value is not None else 1.0
                self.model.pp[idx].fix(float(val))
                fixed_count += 1

        # Welfare variables are kept as Pyomo-only auxiliaries until the CDE
        # block is ported with a numerically safe formulation for PATH.
        for var_name in ("ev", "cv"):
            if hasattr(self.model, var_name):
                var = getattr(self.model, var_name)
                for idx in var:
                    val = var[idx].value if var[idx].value is not None else 1.0
                    var[idx].fix(float(val))
                    fixed_count += 1

        # ===================================================================
        # 4. Fix PABS for regions without production (numeraire consistency)
        # ===================================================================
        if hasattr(self.model, "pabs"):
            # pabs should be fixed where region has no output
            regions_with_output = {r for (r, a) in x_flag}
            for idx in self.model.pabs:
                r = idx if isinstance(idx, str) else idx[0]
                # If region has output, pabs should be fixed (part of closure)
                # If no output, fix at numeraire (1.0)
                val = self.model.pabs[idx].value if self.model.pabs[idx].value is not None else 1.0
                if r not in regions_with_output:
                    self.model.pabs[idx].fix(float(val))
                    fixed_count += 1
        
        logger.info(f"Conditional fixing applied: {fixed_count} variables fixed based on SAM data")
        return fixed_count

    def apply_aggressive_fixing_for_mcp(self) -> int:
        """Apply aggressive fixing to make MCP square.
        
        This fixes additional variables that GAMS fixes but the standard
        conditional fixing doesn't cover. Used specifically for PATH-CAPI
        nonlinear mode.
        
        Returns:
            Number of additional variables fixed
        """
        from collections import deque

        from pyomo.environ import Constraint as Con, Var, value
        from pyomo.core.expr.visitor import identify_variables
        
        fixed_count = 0
        
        # Count current state
        constraints = sum(1 for c in self.model.component_objects(Con, active=True) 
                        for _ in c)
        free_vars = sum(1 for var in self.model.component_objects(Var, active=True)
                       for idx in var if not var[idx].fixed)
        
        gap = free_vars - constraints
        logger.info(f"MCP gap before aggressive fixing: {gap} (free={free_vars}, cons={constraints})")
        
        if gap <= 0:
            logger.info("MCP is already square or over-determined")
            return 0
        
        def is_pyomo_var(obj):
            """Check if object is a Pyomo Var."""
            return isinstance(obj, Var) or (hasattr(obj, 'ctype') and obj.ctype == Var)
        
        def fix_var_list(
            var_names: list,
            default_val: float = 1.0,
            max_to_fix: int = None,
            sort_by_abs_value: bool = False,
        ) -> int:
            """Fix free variables in the given list, up to max_to_fix.

            When `sort_by_abs_value` is True, variables are fixed from the
            smallest absolute current level first. This minimizes distortion by
            preferring near-zero variables when closing a residual MCP gap.
            """
            count = 0
            for var_name in var_names:
                if max_to_fix is not None and count >= max_to_fix:
                    break
                if not hasattr(self.model, var_name):
                    continue
                var = getattr(self.model, var_name)
                if not is_pyomo_var(var):
                    continue

                free_indices: list = [idx for idx in var if not var[idx].fixed]
                if sort_by_abs_value:
                    def _abs_level(i):
                        val = var[i].value
                        if val is None:
                            return abs(default_val)
                        try:
                            return abs(float(val))
                        except Exception:
                            return abs(default_val)
                    free_indices.sort(key=_abs_level)

                for idx in free_indices:
                    if max_to_fix is not None and count >= max_to_fix:
                        break
                    val = var[idx].value if var[idx].value is not None else default_val
                    var[idx].fix(float(val))
                    count += 1
            return count
        
        def get_current_gap():
            """Calculate current gap."""
            free = sum(1 for var in self.model.component_objects(Var, active=True)
                      for idx in var if not var[idx].fixed)
            return free - constraints

        current_gap = gap

        # Strategy 0 (preferred): structural bipartite matching between active
        # equations and free variables. Fix unmatched variables first, because
        # they are the precise DOF excess reported by the Jacobian incidence.
        # This is deterministic and lower impact than heuristic component lists.
        constraints_data = [
            con_data
            for con_data in self.model.component_data_objects(Con, active=True, descend_into=True)
            if con_data.active
        ]
        free_var_data = [
            var_data
            for var_data in self.model.component_data_objects(Var, active=True, descend_into=True)
            if not var_data.fixed
        ]

        if constraints_data and free_var_data and current_gap > 0:
            var_id_to_col = {id(var_data): col for col, var_data in enumerate(free_var_data)}
            adjacency: list[list[int]] = []
            for con_data in constraints_data:
                neighbors: list[int] = []
                seen_cols: set[int] = set()
                for var_data in identify_variables(con_data.body, include_fixed=False):
                    if var_data.fixed:
                        continue
                    col = var_id_to_col.get(id(var_data))
                    if col is None or col in seen_cols:
                        continue
                    seen_cols.add(col)
                    neighbors.append(col)
                adjacency.append(neighbors)

            n_left = len(constraints_data)
            n_right = len(free_var_data)
            pair_left = [-1] * n_left   # equation row -> variable column
            pair_right = [-1] * n_right  # variable column -> equation row
            distance = [0] * n_left
            inf = 10 ** 9

            def _bfs() -> bool:
                queue: deque[int] = deque()
                found_augmenting = False
                for u in range(n_left):
                    if pair_left[u] == -1:
                        distance[u] = 0
                        queue.append(u)
                    else:
                        distance[u] = inf
                while queue:
                    u = queue.popleft()
                    for v in adjacency[u]:
                        matched_u = pair_right[v]
                        if matched_u == -1:
                            found_augmenting = True
                        elif distance[matched_u] == inf:
                            distance[matched_u] = distance[u] + 1
                            queue.append(matched_u)
                return found_augmenting

            def _dfs(u: int) -> bool:
                for v in adjacency[u]:
                    matched_u = pair_right[v]
                    if matched_u == -1 or (
                        distance[matched_u] == distance[u] + 1 and _dfs(matched_u)
                    ):
                        pair_left[u] = v
                        pair_right[v] = u
                        return True
                distance[u] = inf
                return False

            while _bfs():
                for u in range(n_left):
                    if pair_left[u] == -1:
                        _dfs(u)

            unmatched_vars = [
                free_var_data[col]
                for col, matched_row in enumerate(pair_right)
                if matched_row == -1
            ]

            # Fix unmatched first by smallest absolute level.
            unmatched_vars.sort(
                key=lambda var_data: abs(float(var_data.value or 0.0))
            )
            to_fix = min(current_gap, len(unmatched_vars))
            for var_data in unmatched_vars[:to_fix]:
                val = var_data.value if var_data.value is not None else 0.0
                var_data.fix(float(val))
                fixed_count += 1

            current_gap = get_current_gap()
            if to_fix > 0:
                logger.info(
                    "MCP structural matching fixed %s unmatched vars; gap now %s",
                    to_fix,
                    current_gap,
                )

            if current_gap <= 0:
                logger.info("MCP square after structural fixing: %s variables fixed", fixed_count)
                return fixed_count

        # Strategy 1+: fallback heuristic close-out when structural unmatched
        # is insufficient (e.g., numerical dependence beyond structural graph).
        # Prefer near-zero Armington detail variables before touching
        # aggregate production/value variables.

        # Phase 1: near-zero Armington detail quantities
        phase1_vars = ["xda", "xma", "xaa"]
        for vname in phase1_vars:
            if current_gap <= 0:
                break
            fixed = fix_var_list([vname], max_to_fix=current_gap, sort_by_abs_value=True)
            fixed_count += fixed
            current_gap = get_current_gap()
            if fixed > 0:
                logger.debug(f"Fixed {fixed} of {vname}, gap now {current_gap}")
        
        if current_gap <= 0:
            logger.info(f"MCP square after phase 1: {fixed_count} variables fixed")
            return fixed_count
        
        # Phase 2: make-route detail quantities/prices
        phase2_vars = ["x", "p_rai", "pp_rai", "xa", "pabs"]
        for vname in phase2_vars:
            if current_gap <= 0:
                break
            fixed = fix_var_list([vname], max_to_fix=current_gap, sort_by_abs_value=True)
            fixed_count += fixed
            current_gap = get_current_gap()
        
        if current_gap <= 0:
            logger.info(f"MCP square after phase 2: {fixed_count} variables fixed")
            return fixed_count
        
        # Phase 3: aggregate blocks and bilateral trade only as a last resort
        phase3_vars = ["xp", "px", "pe", "xw"]
        for vname in phase3_vars:
            if current_gap <= 0:
                break
            default = 0.0 if vname == "xw" else 1.0
            fixed = fix_var_list(
                [vname],
                default_val=default,
                max_to_fix=current_gap,
                sort_by_abs_value=True,
            )
            fixed_count += fixed
            current_gap = get_current_gap()
        
        if current_gap <= 0:
            logger.info(f"MCP square after phase 3: {fixed_count} variables fixed")
            return fixed_count
        
        # Log final state
        final_free = sum(1 for var in self.model.component_objects(Var, active=True)
                        for idx in var if not var[idx].fixed)
        final_gap = final_free - constraints
        
        logger.info(f"Aggressive fixing applied: {fixed_count} additional variables fixed")
        logger.info(f"MCP gap after aggressive fixing: {final_gap} (free={final_free}, cons={constraints})")
        
        return fixed_count
        
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
        
        # Apply conditional fixing based on SAM data (like GAMS)
        self.apply_conditional_fixing()
        
        # Get solver
        solver = SolverFactory(self.solver_name)
        
        if solver is None:
            raise RuntimeError(
                f"Solver backend '{self.solver_name}' not available "
                f"(requested '{self.requested_solver_name}')"
            )
        
        # Set defaults first, then override with user options
        if self.solver_name == "ipopt":
            solver.options.setdefault("mu_strategy", "adaptive")
            solver.options.setdefault("output_file", "ipopt.out")
        elif self.solver_name == "path":
            self._preflight_path_solver(solver)
            solver.options.setdefault("convergence_tolerance", 1e-6)
            solver.options.setdefault("major_iterations_limit", 500)
        
        # Apply solver options (overrides defaults)
        for key, value in self.solver_options.items():
            solver.options[key] = value
        
        if self.requested_solver_name in {"path-capi", "path_capi"}:
            logger.info(
                "Solving with requested solver '%s' via Pyomo PATH backend '%s'",
                self.requested_solver_name,
                self.solver_name,
            )
        else:
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

    def _preflight_path_solver(self, solver: Any) -> None:
        """Validate PATH executable linkage before attempting a solve.

        This avoids silent hangs when PATH is present but has broken dynamic
        library references (common after machine migrations or brew changes).
        """
        try:
            path_exe = solver.executable()
        except Exception:
            path_exe = None

        if not path_exe:
            raise RuntimeError("PATH executable was not resolved by Pyomo")

        try:
            otool = subprocess.run(
                ["otool", "-L", str(path_exe)],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            # If otool is unavailable, skip preflight on this platform.
            return

        missing_libs: List[str] = []
        macos_virtual_libs = {"/usr/lib/libSystem.B.dylib"}
        lines = (otool.stdout or "").splitlines()
        for line in lines[1:]:
            dep = line.strip().split(" ", 1)[0]
            if dep in macos_virtual_libs:
                continue
            if dep.startswith("/") and not Path(dep).exists():
                missing_libs.append(dep)

        if missing_libs:
            missing_joined = ", ".join(missing_libs)
            raise RuntimeError(
                "PATH runtime is broken: missing dynamic libraries "
                f"[{missing_joined}]. Reinstall PATH for this architecture "
                "or provide compatible runtime libs."
            )

        try:
            file_out = subprocess.run(
                ["file", str(path_exe)],
                capture_output=True,
                text=True,
                check=False,
            )
            host_arch = platform.machine().lower()
            exe_info = (file_out.stdout or "").lower()
            if host_arch == "arm64" and "x86_64" in exe_info and "arm64" not in exe_info:
                logger.warning(
                    "PATH executable is x86_64 on arm64 host. This can work via Rosetta, "
                    "but requires matching x86_64 runtime libraries."
                )
        except Exception:
            pass
    
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
        solver_message = results.solver.get("message", None)
        
        # Extract variable values
        variables = self._extract_variable_values()
        
        # Build message
        message = f"Solver {solver_status}, Termination: {term_cond}"
        if success:
            message = f"Converged in {iterations} iterations, Walras = {walras_val:.2e}"
        if solver_message:
            message = f"{message}; {solver_message}"
        
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
