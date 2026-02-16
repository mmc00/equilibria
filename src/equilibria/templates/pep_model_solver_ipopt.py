"""
PEP Model Solver with IPOPT

This module provides IPOPT-based nonlinear optimization for solving
the PEP CGE model. IPOPT (Interior Point OPTimizer) is a powerful
solver for large-scale nonlinear optimization problems.

Installation:
    pip install cyipopt

Or with conda:
    conda install -c conda-forge cyipopt

Usage:
    from equilibria.templates.pep_model_solver_ipopt import IPOPTSolver
    
    solver = IPOPTSolver(calibrated_state)
    solution = solver.solve()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional, Tuple

import numpy as np

from equilibria.templates.pep_model_equations import PEPModelEquations, PEPModelVariables, SolverResult

logger = logging.getLogger(__name__)

# Try to import IPOPT
try:
    import cyipopt
    IPOPT_AVAILABLE = True
    logger.info("IPOPT (cyipopt) is available")
except ImportError:
    IPOPT_AVAILABLE = False
    logger.warning("IPOPT (cyipopt) not available. Install with: pip install cyipopt")


class CGEProblem:
    """Defines the CGE model as an optimization problem for IPOPT.
    
    We formulate the CGE as a least-squares problem:
    minimize: 0.5 * sum(residuals^2)
    
    Where residuals are the equation errors from the CGE system.
    """
    
    def __init__(
        self,
        equations: PEPModelEquations,
        sets: dict[str, list[str]],
        n_variables: int,
        variable_info: dict[str, Any],
        residual_weights: dict[str, float] | None = None,
        hard_constraints: list[str] | None = None,
        reference_x: np.ndarray | None = None,
    ):
        """Initialize the CGE optimization problem.
        
        Args:
            equations: PEPModelEquations instance
            sets: Model sets
            n_variables: Number of decision variables
            variable_info: Dictionary mapping variable names to indices
        """
        self.equations = equations
        self.sets = sets
        self.n_variables = n_variables
        self.variable_info = variable_info
        self.n_evaluations = 0
        self._residual_scale: np.ndarray | None = None
        self._residual_names: list[str] | None = None
        self._constraint_scale: np.ndarray | None = None
        self.residual_weights = residual_weights or {}
        self.hard_constraints = hard_constraints or []
        self.reference_x = np.array(reference_x, dtype=float) if reference_x is not None else None
        self.reference_scale = (
            np.maximum(np.abs(self.reference_x), 1.0)
            if self.reference_x is not None
            else None
        )
        
    def objective(self, x: np.ndarray) -> float:
        """Compute objective function (0.5 * sum of squared residuals).
        
        Args:
            x: Decision variables
            
        Returns:
            Objective value
        """
        # CNS-style feasibility: all equations are hard constraints, and the
        # objective only regularizes toward the benchmark point.
        if self.reference_x is not None:
            dx = (x - self.reference_x) / self.reference_scale
            return 0.5 * float(np.dot(dx, dx))

        residuals = self._compute_residuals(x)
        return 0.5 * np.sum(residuals ** 2)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of objective using finite differences.
        
        For production use, analytical gradients should be implemented.
        
        Args:
            x: Decision variables
            
        Returns:
            Gradient vector
        """
        if self.reference_x is not None:
            return (x - self.reference_x) / (self.reference_scale**2)

        eps = 1e-8
        grad = np.zeros_like(x)
        f_x = self.objective(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = self.objective(x_plus)
            grad[i] = (f_plus - f_x) / eps

        return grad
    
    def _compute_residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute all equation residuals.
        
        Args:
            x: Decision variables
            
        Returns:
            Vector of residuals
        """
        # Convert array to variables
        vars = self._array_to_variables(x)
        
        # Calculate all residuals
        residual_dict = self.equations.calculate_all_residuals(vars)

        # Build a fixed residual ordering on first evaluation so vector length
        # remains constant under finite-difference perturbations.
        if self._residual_names is None:
            self._residual_names = [name for name in residual_dict if name not in self.hard_constraints]

        # Convert to array with optional block weights (prefix-based, e.g. EQ52)
        weighted_values = []
        for name in self._residual_names:
            value = residual_dict.get(name, 0.0)
            w = 1.0
            for prefix, weight in self.residual_weights.items():
                if name.startswith(prefix):
                    w = weight
                    break
            weighted_values.append(value * w)
        residuals = np.array(weighted_values, dtype=float)
        
        # Use fixed residual scaling from first evaluation to avoid the
        # non-smooth objective induced by per-iteration self-scaling.
        if self._residual_scale is None:
            self._residual_scale = np.maximum(np.abs(residuals), 1.0)
        residuals = residuals / self._residual_scale
        
        self.n_evaluations += 1
        
        if self.n_evaluations % 100 == 0:
            rms = np.sqrt(np.mean(residuals ** 2))
            logger.debug(f"Function evaluation {self.n_evaluations}: RMS residual = {rms:.2e}")
        
        return residuals

    def _compute_constraint_residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute hard-constraint residuals in declared order."""
        vars = self._array_to_variables(x)
        residual_dict = self.equations.calculate_all_residuals(vars)
        residuals = np.array(
            [residual_dict.get(eq, 0.0) for eq in self.hard_constraints],
            dtype=float,
        )
        # Fixed per-equation scaling improves conditioning for mixed-magnitude
        # constraints without changing the feasible set.
        if self._constraint_scale is None:
            self._constraint_scale = np.maximum(np.abs(residuals), 1.0)
        return residuals / self._constraint_scale

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """Equality constraints c(x)=0 for selected equation residuals."""
        if not self.hard_constraints:
            return np.array([], dtype=float)
        return self._compute_constraint_residuals(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Dense Jacobian (row-major flattened) for hard constraints via finite differences."""
        m = len(self.hard_constraints)
        n = len(x)
        if m == 0:
            return np.array([], dtype=float)

        eps = 1e-8
        base = self._compute_constraint_residuals(x)
        jac = np.zeros((m, n), dtype=float)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            c_plus = self._compute_constraint_residuals(x_plus)
            jac[:, i] = (c_plus - base) / eps
        return jac.ravel()

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Dense Jacobian structure indices."""
        m = len(self.hard_constraints)
        n = self.n_variables
        if m == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        rows = np.repeat(np.arange(m, dtype=int), n)
        cols = np.tile(np.arange(n, dtype=int), m)
        return rows, cols
    
    def _array_to_variables(self, x: np.ndarray) -> PEPModelVariables:
        """Convert optimization variables to PEPModelVariables.
        
        Args:
            x: Optimization variables array
            
        Returns:
            PEPModelVariables instance
        """
        vars = PEPModelVariables()
        info = self.variable_info
        
        # Unpack variables based on variable_info mapping
        idx = 0
        
        # Production variables
        for j in self.sets.get("J", []):
            if idx < len(x):
                vars.WC[j] = max(0.1, x[idx])  # Keep positive
                idx += 1
            if idx < len(x):
                vars.RC[j] = max(0.1, x[idx])
                idx += 1
            if idx < len(x):
                vars.PP[j] = max(0.1, x[idx])
                idx += 1
            if idx < len(x):
                vars.PVA[j] = max(0.1, x[idx])
                idx += 1
            if idx < len(x):
                vars.XST[j] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.VA[j] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.CI[j] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.LDC[j] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.KDC[j] = max(0.0, x[idx])
                idx += 1
                
            for l in self.sets.get("L", []):
                if idx < len(x):
                    vars.LD[(l, j)] = max(0.0, x[idx])
                    idx += 1
                if idx < len(x):
                    vars.WTI[(l, j)] = max(0.1, x[idx])
                    idx += 1
                    
            for k in self.sets.get("K", []):
                if idx < len(x):
                    vars.KD[(k, j)] = max(0.0, x[idx])
                    idx += 1
                if idx < len(x):
                    vars.RTI[(k, j)] = max(0.1, x[idx])
                    idx += 1
                if idx < len(x):
                    vars.R[(k, j)] = max(0.1, x[idx])
                    idx += 1
                    
            for i in self.sets.get("I", []):
                if idx < len(x):
                    vars.DI[(i, j)] = max(0.0, x[idx])
                    idx += 1
                if idx < len(x):
                    vars.XS[(j, i)] = max(0.0, x[idx])
                    idx += 1
                if idx < len(x):
                    vars.DS[(j, i)] = max(0.0, x[idx])
                    idx += 1
                if idx < len(x):
                    vars.EX[(j, i)] = max(0.0, x[idx])
                    idx += 1
                if idx < len(x):
                    vars.P[(j, i)] = max(0.1, x[idx])
                    idx += 1
        
        # Wages
        for l in self.sets.get("L", []):
            if idx < len(x):
                vars.W[l] = max(0.1, x[idx])
                idx += 1
        
        # Price and trade variables
        for i in self.sets.get("I", []):
            if idx < len(x):
                vars.PC[i] = max(0.1, x[idx])
                idx += 1
            if idx < len(x):
                vars.PD[i] = max(0.1, x[idx])
                idx += 1
            if idx < len(x):
                vars.PM[i] = max(0.1, x[idx])
                idx += 1
            if idx < len(x):
                vars.PE[i] = max(0.1, x[idx])
                idx += 1
            if idx < len(x):
                vars.PE_FOB[i] = max(0.1, x[idx])
                idx += 1
            if idx < len(x):
                vars.PL[i] = max(0.1, x[idx])
                idx += 1
            if idx < len(x):
                vars.PWM[i] = max(0.1, x[idx])
                idx += 1
            if idx < len(x):
                vars.IM[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.DD[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.Q[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.EXD[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.TIC[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.TIM[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.TIX[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.MRGN[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.DIT[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.INV[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.CG[i] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.VSTK[i] = max(0.0, x[idx])
                idx += 1
        
        # Income variables
        for h in self.sets.get("H", []):
            if idx < len(x):
                vars.YH[h] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.YHL[h] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.YHK[h] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.YHTR[h] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.YDH[h] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.CTH[h] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.SH[h] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.TDH[h] = max(0.0, x[idx])
                idx += 1
                
            for i in self.sets.get("I", []):
                if idx < len(x):
                    vars.C[(i, h)] = max(0.0, x[idx])
                    idx += 1
                if idx < len(x):
                    vars.CMIN[(i, h)] = max(0.0, x[idx])
                    idx += 1
                    
            for ag in self.sets.get("AG", []):
                if idx < len(x):
                    vars.TR[(h, ag)] = max(0.0, x[idx])
                    idx += 1
        
        for f in self.sets.get("F", []):
            if idx < len(x):
                vars.YF[f] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.YFK[f] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.YFTR[f] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.YDF[f] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.SF[f] = max(0.0, x[idx])
                idx += 1
            if idx < len(x):
                vars.TDF[f] = max(0.0, x[idx])
                idx += 1
                
            for ag in self.sets.get("AG", []):
                if idx < len(x):
                    vars.TR[(f, ag)] = max(0.0, x[idx])
                    idx += 1
        
        # Government
        if idx < len(x):
            vars.YG = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.YGK = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.TDHT = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.TDFT = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.TPRCTS = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.TPRODN = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.TIWT = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.TIKT = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.TIPT = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.TICT = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.TIMT = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.TIXT = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.YGTR = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.G = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.SG = max(0.0, x[idx])
            idx += 1
        
        # ROW
        if idx < len(x):
            vars.YROW = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.SROW = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.CAB = x[idx]  # Can be negative
            idx += 1
        
        # Investment
        if idx < len(x):
            vars.IT = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.GFCF = max(0.0, x[idx])
            idx += 1
        
        # GDP
        if idx < len(x):
            vars.GDP_BP = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.GDP_MP = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.GDP_IB = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.GDP_FD = max(0.0, x[idx])
            idx += 1
        
        # Price indices
        if idx < len(x):
            vars.PIXCON = max(0.1, x[idx])
            idx += 1
        if idx < len(x):
            vars.PIXGDP = max(0.1, x[idx])
            idx += 1
        if idx < len(x):
            vars.PIXGVT = max(0.1, x[idx])
            idx += 1
        if idx < len(x):
            vars.PIXINV = max(0.1, x[idx])
            idx += 1
        
        # Real variables
        for h in self.sets.get("H", []):
            if idx < len(x):
                vars.CTH_REAL[h] = max(0.0, x[idx])
                idx += 1
        if idx < len(x):
            vars.G_REAL = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.GDP_BP_REAL = max(0.0, x[idx])
            idx += 1
        if idx < len(x):
            vars.GDP_MP_REAL = max(0.0, x[idx])
            idx += 1
        
        # Exchange rate
        if idx < len(x):
            vars.e = max(0.1, x[idx])
            idx += 1
        
        return vars
    
    def _variables_to_array(self, vars: PEPModelVariables) -> np.ndarray:
        """Convert PEPModelVariables to array."""
        values = []
        
        # Production variables
        for j in self.sets.get("J", []):
            values.append(vars.WC.get(j, 1.0))
            values.append(vars.RC.get(j, 1.0))
            values.append(vars.PP.get(j, 1.0))
            values.append(vars.PVA.get(j, 1.0))
            values.append(vars.XST.get(j, 0))
            values.append(vars.VA.get(j, 0))
            values.append(vars.CI.get(j, 0))
            values.append(vars.LDC.get(j, 0))
            values.append(vars.KDC.get(j, 0))
            
            for l in self.sets.get("L", []):
                values.append(vars.LD.get((l, j), 0))
                values.append(vars.WTI.get((l, j), 1.0))
                
            for k in self.sets.get("K", []):
                values.append(vars.KD.get((k, j), 0))
                values.append(vars.RTI.get((k, j), 1.0))
                values.append(vars.R.get((k, j), 1.0))
                
            for i in self.sets.get("I", []):
                values.append(vars.DI.get((i, j), 0))
                values.append(vars.XS.get((j, i), 0))
                values.append(vars.DS.get((j, i), 0))
                values.append(vars.EX.get((j, i), 0))
                values.append(vars.P.get((j, i), 1.0))
        
        # Wages
        for l in self.sets.get("L", []):
            values.append(vars.W.get(l, 1.0))
        
        # Price and trade variables
        for i in self.sets.get("I", []):
            values.append(vars.PC.get(i, 1.0))
            values.append(vars.PD.get(i, 1.0))
            values.append(vars.PM.get(i, 1.0))
            values.append(vars.PE.get(i, 1.0))
            values.append(vars.PE_FOB.get(i, 1.0))
            values.append(vars.PL.get(i, 1.0))
            values.append(vars.PWM.get(i, 1.0))
            values.append(vars.IM.get(i, 0))
            values.append(vars.DD.get(i, 0))
            values.append(vars.Q.get(i, 0))
            values.append(vars.EXD.get(i, 0))
            values.append(vars.TIC.get(i, 0))
            values.append(vars.TIM.get(i, 0))
            values.append(vars.TIX.get(i, 0))
            values.append(vars.MRGN.get(i, 0))
            values.append(vars.DIT.get(i, 0))
            values.append(vars.INV.get(i, 0))
            values.append(vars.CG.get(i, 0))
            values.append(vars.VSTK.get(i, 0))
        
        # Income variables
        for h in self.sets.get("H", []):
            values.append(vars.YH.get(h, 0))
            values.append(vars.YHL.get(h, 0))
            values.append(vars.YHK.get(h, 0))
            values.append(vars.YHTR.get(h, 0))
            values.append(vars.YDH.get(h, 0))
            values.append(vars.CTH.get(h, 0))
            values.append(vars.SH.get(h, 0))
            values.append(vars.TDH.get(h, 0))
            
            for i in self.sets.get("I", []):
                values.append(vars.C.get((i, h), 0))
                values.append(vars.CMIN.get((i, h), 0))
                
            for ag in self.sets.get("AG", []):
                values.append(vars.TR.get((h, ag), 0))
        
        for f in self.sets.get("F", []):
            values.append(vars.YF.get(f, 0))
            values.append(vars.YFK.get(f, 0))
            values.append(vars.YFTR.get(f, 0))
            values.append(vars.YDF.get(f, 0))
            values.append(vars.SF.get(f, 0))
            values.append(vars.TDF.get(f, 0))
            
            for ag in self.sets.get("AG", []):
                values.append(vars.TR.get((f, ag), 0))
        
        # Government
        values.append(vars.YG)
        values.append(vars.YGK)
        values.append(vars.TDHT)
        values.append(vars.TDFT)
        values.append(vars.TPRCTS)
        values.append(vars.TPRODN)
        values.append(vars.TIWT)
        values.append(vars.TIKT)
        values.append(vars.TIPT)
        values.append(vars.TICT)
        values.append(vars.TIMT)
        values.append(vars.TIXT)
        values.append(vars.YGTR)
        values.append(vars.G)
        values.append(vars.SG)
        
        # ROW
        values.append(vars.YROW)
        values.append(vars.SROW)
        values.append(vars.CAB)
        
        # Investment
        values.append(vars.IT)
        values.append(vars.GFCF)
        
        # GDP
        values.append(vars.GDP_BP)
        values.append(vars.GDP_MP)
        values.append(vars.GDP_IB)
        values.append(vars.GDP_FD)
        
        # Price indices
        values.append(vars.PIXCON)
        values.append(vars.PIXGDP)
        values.append(vars.PIXGVT)
        values.append(vars.PIXINV)
        
        # Real variables
        for h in self.sets.get("H", []):
            values.append(vars.CTH_REAL.get(h, 0))
        values.append(vars.G_REAL)
        values.append(vars.GDP_BP_REAL)
        values.append(vars.GDP_MP_REAL)
        
        # Exchange rate
        values.append(vars.e)
        
        return np.array(values)


class IPOPTSolver:
    """IPOPT-based solver for PEP model."""
    
    def __init__(
        self,
        calibrated_state: Any,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        init_mode: Literal["strict_gams", "equation_consistent"] = "strict_gams",
    ):
        """Initialize the solver with calibrated model state.
        
        Args:
            calibrated_state: PEPModelState from calibration
            tolerance: Convergence tolerance for residuals
            max_iterations: Maximum number of iterations
        """
        self.state = calibrated_state
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.init_mode = init_mode
        
        # Extract sets and parameters from calibrated state
        self.sets = calibrated_state.sets
        self.params = self._extract_parameters(calibrated_state)
        
        # Initialize equations
        self.equations = PEPModelEquations(self.sets, self.params)
        
        logger.info(f"Initialized IPOPT Solver")
        logger.info(f"  Sets: {len(self.sets)} categories")
        logger.info(f"  Tolerance: {tolerance}")
        logger.info(f"  Max iterations: {max_iterations}")
        logger.info(f"  Init mode: {init_mode}")
    
    def _extract_parameters(self, state: Any) -> dict[str, Any]:
        """Extract all calibrated parameters from model state."""
        params = {}
        yh_base = state.income.get("YHO", {})
        ydh_base = state.income.get("YDHO", {})
        tro_base = state.income.get("TRO", {})
        ttdh0_base = state.income.get("ttdh0O", {})
        inferred_ttdh1 = {}
        inferred_ttdf1 = {}
        for h in state.sets.get("H", []):
            yh_h = yh_base.get(h, 0.0)
            tr_h_to_gvt = tro_base.get(("gvt", h), 0.0)
            tdh_h = max(yh_base.get(h, 0.0) - ydh_base.get(h, 0.0) - tr_h_to_gvt, 0.0)
            ttdh0_h = ttdh0_base.get(h, 0.0)
            inferred_ttdh1[h] = ((tdh_h - ttdh0_h) / yh_h) if abs(yh_h) > 1e-12 else 0.0
        yfo_base = state.income.get("YFO", {})
        ydfo_base = state.income.get("YDFO", {})
        yfko_base = state.income.get("YFKO", {})
        for f in state.sets.get("F", []):
            yfk_f = yfko_base.get(f, 0.0)
            tdf_f = max(yfo_base.get(f, 0.0) - ydfo_base.get(f, 0.0), 0.0)
            inferred_ttdf1[f] = (tdf_f / yfk_f) if abs(yfk_f) > 1e-12 else 0.0
        
        def _safe_sigma_plus_one(rho_val: float, fallback: float = 1.0) -> float:
            denom = 1 + rho_val
            if abs(denom) < 1e-12:
                return fallback
            return 1 / denom

        # Income parameters
        params.update({
            "lambda_RK": state.income.get("lambda_RK", {}),
            "lambda_WL": state.income.get("lambda_WL", {}),
            "lambda_TR_households": state.income.get("lambda_TR_households", {}),
            "lambda_TR_firms": state.income.get("lambda_TR_firms", {}),
            "sh1": state.income.get("sh1O", {}),
            "tr1": state.income.get("tr1O", {}),
        })
        
        # Production parameters
        params.update({
            "io": state.production.get("io", {}),
            "v": state.production.get("v", {}),
            "aij": state.production.get("aij", {}),
            "rho_VA": state.production.get("rho_VA", {}),
            "beta_VA": state.production.get("beta_VA", {}),
            "B_VA": state.production.get("B_VA", {}),
            "sigma_VA": {
                j: _safe_sigma_plus_one(state.production.get("rho_VA", {}).get(j, 0), 1.0)
                for j in state.production.get("rho_VA", {})
            },
            "rho_KD": state.production.get("rho_KD", {}),
            "beta_KD": state.production.get("beta_KD", {}),
            "B_KD": state.production.get("B_KD", {}),
            "sigma_KD": {j: 1/(1+params["rho_KD"].get(j, 0)) for j in params.get("rho_KD", {})},
            "rho_LD": state.production.get("rho_LD", {}),
            "beta_LD": state.production.get("beta_LD", {}),
            "B_LD": state.production.get("B_LD", {}),
            "sigma_LD": {j: 1/(1+params["rho_LD"].get(j, 0)) for j in params.get("rho_LD", {})},
            "ttiw": state.production.get("ttiwO", {}),
            "ttik": state.production.get("ttikO", {}),
            "ttip": state.production.get("ttipO", {}),
            "LS": state.production.get("LSO", {}),
            "KS": state.production.get("KSO", {}),
        })
        
        # Trade parameters
        params.update({
            "rho_XT": state.trade.get("rho_XT", {}),
            "beta_XT": state.trade.get("beta_XT", {}),
            "B_XT": state.trade.get("B_XT", {}),
            "sigma_XT": {j: 1/(params["rho_XT"].get(j, 1)-1) if params["rho_XT"].get(j, 1) != 1 else 2.0 for j in params.get("rho_XT", {})},
            "rho_X": state.trade.get("rho_X", {}),
            "beta_X": state.trade.get("beta_X", {}),
            "B_X": state.trade.get("B_X", {}),
            "sigma_X": {(j, i): 1/(params["rho_X"].get((j, i), 1)-1) if params["rho_X"].get((j, i), 1) != 1 else 2.0 for j, i in params.get("rho_X", {})},
            "rho_M": state.trade.get("rho_M", {}),
            "beta_M": state.trade.get("beta_M", {}),
            "B_M": state.trade.get("B_M", {}),
            "sigma_M": {i: 1/(1+params["rho_M"].get(i, -0.5)) for i in params.get("rho_M", {})},
            "sigma_XD": {i: 2.0 for i in state.sets.get("I", [])},
            "ttic": state.trade.get("tticO", {}),
            "ttim": state.trade.get("ttimO", {}),
            "ttix": state.trade.get("ttixO", {}),
            "tmrg": state.trade.get("tmrg", {}),
            "tmrg_X": state.trade.get("tmrg_X", {}),
            "EXDO": state.trade.get("EXDO", {}),
        })
        
        # LES parameters
        params.update({
            "gamma_LES": state.les_parameters.get("gamma_LES", {}),
            "sigma_Y": state.les_parameters.get("sigma_Y", {}),
            "frisch": state.les_parameters.get("frisch", {}),
        })
        
        # Additional parameters
        params.update({
            "eta": 1,
            "sh0": {},
            "tr0": {},
            "ttdh0": ttdh0_base,
            "ttdh1": inferred_ttdh1,
            "ttdf0": {},
            "ttdf1": inferred_ttdf1,
            "TRO": state.income.get("TRO", {}),
            "PWX": state.trade.get("PWXO", {}),
            "kmob": 1.0,
            "PT": state.production.get("PTO", {}),
        })

        inv_base = state.consumption.get("INVO", {})
        pc_base = state.trade.get("PCO", {})
        inv_nom = {i: inv_base.get(i, 0.0) * pc_base.get(i, 1.0) for i in state.sets.get("I", [])}
        inv_total = sum(inv_nom.values())
        params["gamma_INV"] = {
            i: (inv_nom.get(i, 0.0) / inv_total if abs(inv_total) > 1e-12 else 0.0)
            for i in state.sets.get("I", [])
        }

        cg_base = state.consumption.get("CGO", {})
        cg_nom = {i: cg_base.get(i, 0.0) * pc_base.get(i, 1.0) for i in state.sets.get("I", [])}
        cg_total = sum(cg_nom.values())
        params["gamma_GVT"] = {
            i: (cg_nom.get(i, 0.0) / cg_total if abs(cg_total) > 1e-12 else 0.0)
            for i in state.sets.get("I", [])
        }

        rho_kd = params.get("rho_KD", {})
        params["sigma_KD"] = {j: _safe_sigma_plus_one(rho_kd.get(j, 0), 1.0) for j in rho_kd}

        rho_ld = params.get("rho_LD", {})
        params["sigma_LD"] = {j: _safe_sigma_plus_one(rho_ld.get(j, 0), 1.0) for j in rho_ld}

        rho_xt = params.get("rho_XT", {})
        params["sigma_XT"] = {
            j: (1 / (rho_xt.get(j, 1) - 1) if rho_xt.get(j, 1) != 1 else 2.0)
            for j in rho_xt
        }

        rho_x = params.get("rho_X", {})
        params["sigma_X"] = {
            (j, i): (1 / (rho_x.get((j, i), 1) - 1) if rho_x.get((j, i), 1) != 1 else 2.0)
            for (j, i) in rho_x
        }

        rho_m = params.get("rho_M", {})
        params["sigma_M"] = {i: _safe_sigma_plus_one(rho_m.get(i, -0.5), 2.0) for i in rho_m}

        lambda_tr_combined: dict[tuple[str, str], float] = {}
        lambda_tr_combined.update(params.get("lambda_TR_households", {}))
        lambda_tr_combined.update(params.get("lambda_TR_firms", {}))
        params["lambda_TR"] = lambda_tr_combined

        for j in state.sets.get("J", []):
            params.setdefault("beta_VA", {}).setdefault(j, 1.0)
            params.setdefault("B_VA", {}).setdefault(j, 1.0)

        # Recompute intermediate-use coefficients with benchmark PCO and trade XSO.
        dio_raw = state.production.get("DIO", {})
        pco = state.trade.get("PCO", {})
        vao = state.production.get("VAO", {})
        xso = state.trade.get("XSO", {})
        I = state.sets.get("I", [])
        J = state.sets.get("J", [])

        dio_q: dict[tuple[str, str], float] = {}
        for i in I:
            p_i = pco.get(i, 1.0)
            for j in J:
                val = dio_raw.get((i, j), 0.0)
                dio_q[(i, j)] = (val / p_i) if abs(p_i) > 1e-12 else 0.0

        ci_q = {j: sum(dio_q.get((i, j), 0.0) for i in I) for j in J}
        xst_q = {j: sum(xso.get((j, i), 0.0) for i in I) for j in J}

        params["aij"] = {
            (i, j): (dio_q.get((i, j), 0.0) / ci_q[j] if abs(ci_q[j]) > 1e-12 else 0.0)
            for j in J
            for i in I
        }
        params["io"] = {j: (ci_q[j] / xst_q[j] if abs(xst_q[j]) > 1e-12 else 0.0) for j in J}
        params["v"] = {j: (vao.get(j, 0.0) / xst_q[j] if abs(xst_q[j]) > 1e-12 else 0.0) for j in J}

        lambda_tr_combined: dict[tuple[str, str], float] = {}
        lambda_tr_combined.update(params.get("lambda_TR_households", {}))
        lambda_tr_combined.update(params.get("lambda_TR_firms", {}))
        params["lambda_TR"] = lambda_tr_combined

        for j in state.sets.get("J", []):
            params.setdefault("beta_VA", {}).setdefault(j, 1.0)
            params.setdefault("B_VA", {}).setdefault(j, 1.0)
        
        return params
    
    def _create_initial_guess(self) -> PEPModelVariables:
        """Create initial guess for variables from calibrated values."""
        vars = PEPModelVariables()
        state = self.state
        
        # Initialize from calibrated state (using "O" suffix values as starting point)
        tro = self.params.get("TRO", {})
        for ag in self.sets.get("AG", []):
            for agj in self.sets.get("AG", []):
                vars.TR[(ag, agj)] = tro.get((ag, agj), 0)
        
        # Production variables
        for j in self.sets.get("J", []):
            vars.VA[j] = state.production.get("VAO", {}).get(j, 0)
            vars.CI[j] = state.production.get("CIO", {}).get(j, 0)
            vars.LDC[j] = state.production.get("LDCO", {}).get(j, 0)
            vars.KDC[j] = state.production.get("KDCO", {}).get(j, 0)
            vars.XST[j] = state.production.get("XSTO", {}).get(j, 0)
            vars.PVA[j] = state.production.get("PVAO", {}).get(j, 1.0)
            vars.PP[j] = state.production.get("PPO", {}).get(j, 1.0)
            vars.PCI[j] = state.production.get("PCIO", {}).get(j, 1.0)
            vars.WC[j] = state.production.get("WCO", {}).get(j, 1.0)
            vars.RC[j] = state.production.get("RCO", {}).get(j, 1.0)
            
            for l in self.sets.get("L", []):
                vars.LD[(l, j)] = state.production.get("LDO", {}).get((l, j), 0)
                vars.WTI[(l, j)] = state.production.get("WTIO", {}).get((l, j), 1.0)
            
            for k in self.sets.get("K", []):
                vars.KD[(k, j)] = state.production.get("KDO", {}).get((k, j), 0)
                vars.RTI[(k, j)] = state.production.get("RTIO", {}).get((k, j), 1.0)
                vars.R[(k, j)] = 1.0  # Base price
            
            for i in self.sets.get("I", []):
                vars.DI[(i, j)] = state.production.get("DIO", {}).get((i, j), 0)
                vars.XS[(j, i)] = state.trade.get("XSO", {}).get((j, i), 0)
                vars.DS[(j, i)] = state.trade.get("DSO", {}).get((j, i), 0)
                vars.EX[(j, i)] = state.trade.get("EXO", {}).get((j, i), 0)
                vars.P[(j, i)] = state.trade.get("PO", {}).get((j, i), 1.0)
        
        # Initialize wages
        for l in self.sets.get("L", []):
            vars.W[l] = 1.0  # Numeraire

        # Initialize tax-payment matrices from rate equations (TIWO/TIKO may be absent
        # in calibration output depending on phase configuration).
        for l in self.sets.get("L", []):
            for j in self.sets.get("J", []):
                key = (l, j)
                ttiw = self.params.get("ttiw", {}).get(key, 0.0)
                vars.TIW[key] = ttiw * vars.W.get(l, 1.0) * vars.LD.get(key, 0.0)

        for k in self.sets.get("K", []):
            for j in self.sets.get("J", []):
                key = (k, j)
                ttik = self.params.get("ttik", {}).get(key, 0.0)
                vars.TIK[key] = ttik * vars.R.get(key, 1.0) * vars.KD.get(key, 0.0)
        
        # Income variables
        for h in self.sets.get("H", []):
            vars.YH[h] = state.income.get("YHO", {}).get(h, 0)
            vars.YHL[h] = state.income.get("YHLO", {}).get(h, 0)
            vars.YHK[h] = state.income.get("YHKO", {}).get(h, 0)
            vars.YHTR[h] = state.income.get("YHTRO", {}).get(h, 0)
            vars.YDH[h] = state.income.get("YDHO", {}).get(h, 0)
            vars.CTH[h] = state.income.get("CTHO", {}).get(h, 0)
            vars.SH[h] = vars.YDH[h] - vars.CTH[h]
            vars.TDH[h] = vars.YH[h] - vars.YDH[h] - vars.TR.get(("gvt", h), 0.0)
            
            for ag in self.sets.get("AG", []):
                vars.TR[(h, ag)] = tro.get((h, ag), vars.TR.get((h, ag), 0))
        
        for f in self.sets.get("F", []):
            vars.YF[f] = state.income.get("YFO", {}).get(f, 0)
            vars.YFK[f] = state.income.get("YFKO", {}).get(f, 0)
            vars.YFTR[f] = state.income.get("YFTRO", {}).get(f, 0)
            vars.YDF[f] = state.income.get("YDFO", {}).get(f, 0)
            vars.SF[f] = vars.YDF[f]
            vars.TDF[f] = vars.YF[f] - vars.YDF[f]
            
            for ag in self.sets.get("AG", []):
                vars.TR[(f, ag)] = tro.get((f, ag), vars.TR.get((f, ag), 0))
        
        # Government variables
        vars.YG = state.income.get("YGO", 0)
        vars.YGK = state.income.get("YGKO", 0)
        vars.TDHT = state.income.get("TDHTO", 0)
        vars.TDFT = state.income.get("TDFTO", 0)
        vars.TPRCTS = state.income.get("TPRCTSO", 0)
        vars.TPRODN = state.income.get("TPRODNO", 0)
        vars.TIWT = state.income.get("TIWTO", 0)
        vars.TIKT = state.income.get("TIKTO", 0)
        vars.TIPT = state.income.get("TIPTO", 0)
        vars.TICT = state.income.get("TICTO", 0)
        vars.TIMT = state.income.get("TIMTO", 0)
        vars.TIXT = state.income.get("TIXTO", 0)
        vars.YGTR = state.income.get("YGTRO", 0)
        vars.G = state.consumption.get("GO", 0)
        vars.SG = state.income.get("SGO", 0)
        
        # Trade variables
        for i in self.sets.get("I", []):
            vars.IM[i] = state.trade.get("IMO", {}).get(i, 0)
            vars.DD[i] = state.trade.get("DDO", {}).get(i, 0)
            vars.Q[i] = state.trade.get("QO", {}).get(i, 0)
            vars.EXD[i] = state.trade.get("EXDO", {}).get(i, 0)
            vars.PC[i] = state.trade.get("PCO", {}).get(i, 1.0)
            vars.PD[i] = state.trade.get("PDO", {}).get(i, 1.0)
            vars.PM[i] = state.trade.get("PMO", {}).get(i, 1.0)
            vars.PE[i] = state.trade.get("PEO", {}).get(i, 1.0)
            vars.PE_FOB[i] = state.trade.get("PE_FOBO", {}).get(i, 1.0)
            vars.PWM[i] = state.trade.get("PWMO", {}).get(i, 1.0)
            vars.PL[i] = state.trade.get("PLO", {}).get(i, 1.0)
            vars.TIC[i] = state.trade.get("TICO", {}).get(i, 0)
            vars.TIM[i] = state.trade.get("TIMO", {}).get(i, 0)
            vars.TIX[i] = state.trade.get("TIXO", {}).get(i, 0)
            vars.MRGN[i] = 0
            vars.DIT[i] = state.production.get("DITO", {}).get(i, 0)
            vars.INV[i] = state.consumption.get("INVO", {}).get(i, 0)
            vars.CG[i] = state.consumption.get("CGO", {}).get(i, 0)
            vars.VSTK[i] = state.consumption.get("VSTKO", {}).get(i, 0)
        
        # Consumption variables
        for h in self.sets.get("H", []):
            for i in self.sets.get("I", []):
                vars.C[(i, h)] = state.consumption.get("CO", {}).get((i, h), 0)
                vars.CMIN[(i, h)] = state.les_parameters.get("CMINO", {}).get((i, h), 0)

        # Convert intermediate demand to quantities using benchmark PC(i).
        for i in self.sets.get("I", []):
            pc_i = vars.PC.get(i, 1.0)
            if abs(pc_i) < 1e-12:
                continue
            for j in self.sets.get("J", []):
                vars.DI[(i, j)] = vars.DI.get((i, j), 0.0) / pc_i

        # Recompute aggregate demand blocks from calibrated quantities.
        for j in self.sets.get("J", []):
            vars.CI[j] = sum(vars.DI.get((i, j), 0.0) for i in self.sets.get("I", []))
            ci_j = vars.CI.get(j, 0.0)
            if abs(ci_j) > 1e-12:
                vars.PCI[j] = (
                    sum(vars.PC.get(i, 1.0) * vars.DI.get((i, j), 0.0) for i in self.sets.get("I", []))
                    / ci_j
                )
        for i in self.sets.get("I", []):
            vars.DIT[i] = sum(vars.DI.get((i, j), 0.0) for j in self.sets.get("J", []))
        for j in self.sets.get("J", []):
            vars.XST[j] = sum(vars.XS.get((j, i), 0.0) for i in self.sets.get("I", []))
            xst_j = vars.XST.get(j, 0.0)
            if abs(xst_j) > 1e-12:
                vars.PP[j] = (
                    vars.PVA.get(j, 0.0) * vars.VA.get(j, 0.0) + vars.PCI.get(j, 0.0) * vars.CI.get(j, 0.0)
                ) / xst_j
            ttip = self.params.get("ttip", {}).get(j, 0.0)
            vars.TIP[j] = ttip * vars.PP.get(j, 0.0) * vars.XST.get(j, 0.0)
        vars.G = sum(vars.PC.get(i, 1.0) * vars.CG.get(i, 0.0) for i in self.sets.get("I", []))

        # Initialize trade margin demand from calibrated flows (EQ57 identity)
        for i in self.sets.get("I", []):
            mrgn_i = 0.0
            for ij in self.sets.get("I", []):
                tm = self.params.get("tmrg", {}).get((i, ij), 0.0)
                tm_x = self.params.get("tmrg_X", {}).get((i, ij), 0.0)
                mrgn_i += tm * vars.DD.get(ij, 0.0)
                mrgn_i += tm * vars.IM.get(ij, 0.0)
                mrgn_i += sum(tm_x * vars.EX.get((j, ij), 0.0) for j in self.sets.get("J", []))
            vars.MRGN[i] = mrgn_i

        # Recompute savings variables from accounting identities using calibrated TR.
        for h in self.sets.get("H", []):
            tr_out_h = sum(vars.TR.get((agng, h), 0.0) for agng in self.sets.get("AGNG", []))
            vars.SH[h] = vars.YDH.get(h, 0.0) - vars.CTH.get(h, 0.0) - tr_out_h

        for f in self.sets.get("F", []):
            tr_out_f = sum(vars.TR.get((ag, f), 0.0) for ag in self.sets.get("AG", []))
            vars.SF[f] = vars.YDF.get(f, 0.0) - tr_out_f

        tr_to_govt = sum(vars.TR.get((agng, "gvt"), 0.0) for agng in self.sets.get("AGNG", []))
        vars.SG = vars.YG - tr_to_govt - vars.G
        
        # ROW variables
        vars.YROW = state.income.get("YROWO", 0)
        vars.SROW = -state.income.get("CABO", 0)
        vars.CAB = state.income.get("CABO", 0)
        
        # Investment
        vars.IT = state.income.get("ITO", 0)
        stock_value = sum(vars.PC.get(i, 1.0) * vars.VSTK.get(i, 0.0) for i in self.sets.get("I", []))
        vars.GFCF = vars.IT - stock_value

        # Aggregate tax and transfer totals consistent with initialized detail.
        vars.TDHT = sum(vars.TDH.values())
        vars.TDFT = sum(vars.TDF.values())
        vars.TIWT = sum(vars.TIW.values())
        vars.TIKT = sum(vars.TIK.values())
        vars.TIPT = sum(vars.TIP.values())
        vars.TICT = sum(vars.TIC.values())
        vars.TIMT = sum(vars.TIM.values())
        vars.TIXT = sum(vars.TIX.values())
        vars.YGTR = sum(vars.TR.get(("gvt", agng), 0.0) for agng in self.sets.get("AGNG", []))
        vars.TPRODN = vars.TIWT + vars.TIKT + vars.TIPT
        vars.TPRCTS = vars.TICT + vars.TIMT + vars.TIXT
        vars.YG = vars.YGK + vars.TDHT + vars.TDFT + vars.TPRODN + vars.TPRCTS + vars.YGTR
        vars.SG = vars.YG - tr_to_govt - vars.G
        
        # GDP variables
        vars.GDP_BP = state.gdp.get("GDP_BPO", 0)
        vars.GDP_MP = state.gdp.get("GDP_MPO", 0)
        vars.GDP_IB = state.gdp.get("GDP_IBO", 0)
        gdp_fd = 0.0
        for i in self.sets.get("I", []):
            cons_i = sum(vars.C.get((i, h), 0.0) for h in self.sets.get("H", []))
            gdp_fd += vars.PC.get(i, 0.0) * (
                cons_i + vars.CG.get(i, 0.0) + vars.INV.get(i, 0.0) + vars.VSTK.get(i, 0.0)
            )
            gdp_fd += vars.PE_FOB.get(i, 0.0) * vars.EXD.get(i, 0.0)
            gdp_fd -= vars.PWM.get(i, 0.0) * vars.e * vars.IM.get(i, 0.0)
        vars.GDP_FD = gdp_fd
        
        # Price indices
        vars.PIXCON = state.real_variables.get("PIXCONO", 1.0)
        vars.PIXGDP = state.real_variables.get("PIXGDPO", 1.0)
        vars.PIXGVT = state.real_variables.get("PIXGVTO", 1.0)
        vars.PIXINV = state.real_variables.get("PIXINVO", 1.0)
        
        # Real variables
        for h in self.sets.get("H", []):
            vars.CTH_REAL[h] = state.real_variables.get("CTH_REALO", {}).get(h, 0)
        vars.G_REAL = state.real_variables.get("G_REALO", 0)
        vars.GDP_BP_REAL = state.real_variables.get("GDP_BP_REALO", 0)
        vars.GDP_MP_REAL = vars.GDP_MP / vars.PIXCON if abs(vars.PIXCON) > 1e-12 else 0.0
        
        # Exchange rate
        vars.e = state.trade.get("eO", 1.0)

        if self.init_mode == "equation_consistent":
            self._apply_equation_consistent_adjustments(vars)
        
        return vars

    def _apply_equation_consistent_adjustments(self, vars: PEPModelVariables) -> None:
        """Apply adjustments so initialized values satisfy benchmark equations."""
        for l in self.sets.get("L", []):
            for j in self.sets.get("J", []):
                ttiw = self.params.get("ttiw", {}).get((l, j), 0.0)
                vars.TIW[(l, j)] = ttiw * vars.W.get(l, 1.0) * vars.LD.get((l, j), 0.0)

        for k in self.sets.get("K", []):
            for j in self.sets.get("J", []):
                ttik = self.params.get("ttik", {}).get((k, j), 0.0)
                vars.TIK[(k, j)] = ttik * vars.R.get((k, j), 1.0) * vars.KD.get((k, j), 0.0)

        vars.TIWT = sum(vars.TIW.values())
        vars.TIKT = sum(vars.TIK.values())
        vars.TPRODN = vars.TIWT + vars.TIKT + vars.TIPT

        tixo_bench = self.state.trade.get("TIXO", {})
        for i in self.sets.get("I", []):
            vars.TIX[i] = tixo_bench.get(i, vars.TIX.get(i, 0.0))

        vars.TIXT = sum(vars.TIX.values())
        vars.TPRCTS = vars.TICT + vars.TIMT + vars.TIXT
        vars.YG = vars.YGK + vars.TDHT + vars.TDFT + vars.TPRODN + vars.TPRCTS + vars.YGTR

        tr_to_govt = sum(vars.TR.get((agng, "gvt"), 0) for agng in self.sets.get("AGNG", []))
        vars.SG = vars.YG - tr_to_govt - vars.G

        vars.CAB = self.state.income.get("CABO", vars.CAB)
        vars.SROW = -vars.CAB

        vars.IT = self.state.income.get("ITO", vars.IT)
        stock_value = sum(vars.PC.get(i, 1.0) * vars.VSTK.get(i, 0.0) for i in self.sets.get("I", []))
        vars.GFCF = vars.IT - stock_value

        gdp_ib = 0.0
        for l in self.sets.get("L", []):
            for j in self.sets.get("J", []):
                gdp_ib += vars.W.get(l, 1.0) * vars.LD.get((l, j), 0.0)
        for k in self.sets.get("K", []):
            for j in self.sets.get("J", []):
                gdp_ib += vars.R.get((k, j), 1.0) * vars.KD.get((k, j), 0.0)
        gdp_ib += vars.TPRODN + vars.TPRCTS
        vars.GDP_IB = gdp_ib
        vars.GDP_MP = vars.GDP_BP + vars.TPRCTS
        vars.GDP_MP_REAL = vars.GDP_MP / vars.PIXCON if abs(vars.PIXCON) > 1e-12 else 0.0

        gdp_fd = 0.0
        for i in self.sets.get("I", []):
            cons_i = sum(vars.C.get((i, h), 0.0) for h in self.sets.get("H", []))
            gdp_fd += vars.PC.get(i, 0.0) * (
                cons_i + vars.CG.get(i, 0.0) + vars.INV.get(i, 0.0) + vars.VSTK.get(i, 0.0)
            )
            gdp_fd += vars.PE_FOB.get(i, 0.0) * vars.EXD.get(i, 0.0)
            gdp_fd -= vars.PWM.get(i, 0.0) * vars.e * vars.IM.get(i, 0.0)
        vars.GDP_FD = gdp_fd
    
    def _variables_to_array(self, vars: PEPModelVariables) -> np.ndarray:
        """Convert variables to flat array for solver."""
        values = []
        
        # Production variables
        for j in self.sets.get("J", []):
            values.append(vars.WC.get(j, 1.0))
            values.append(vars.RC.get(j, 1.0))
            values.append(vars.PP.get(j, 1.0))
            values.append(vars.PVA.get(j, 1.0))
            values.append(vars.XST.get(j, 0))
            values.append(vars.VA.get(j, 0))
            values.append(vars.CI.get(j, 0))
            values.append(vars.LDC.get(j, 0))
            values.append(vars.KDC.get(j, 0))
            
            for l in self.sets.get("L", []):
                values.append(vars.LD.get((l, j), 0))
                values.append(vars.WTI.get((l, j), 1.0))
                
            for k in self.sets.get("K", []):
                values.append(vars.KD.get((k, j), 0))
                values.append(vars.RTI.get((k, j), 1.0))
                values.append(vars.R.get((k, j), 1.0))
                
            for i in self.sets.get("I", []):
                values.append(vars.DI.get((i, j), 0))
                values.append(vars.XS.get((j, i), 0))
                values.append(vars.DS.get((j, i), 0))
                values.append(vars.EX.get((j, i), 0))
                values.append(vars.P.get((j, i), 1.0))
        
        # Wages
        for l in self.sets.get("L", []):
            values.append(vars.W.get(l, 1.0))
        
        # Price and trade variables
        for i in self.sets.get("I", []):
            values.append(vars.PC.get(i, 1.0))
            values.append(vars.PD.get(i, 1.0))
            values.append(vars.PM.get(i, 1.0))
            values.append(vars.PE.get(i, 1.0))
            values.append(vars.PE_FOB.get(i, 1.0))
            values.append(vars.PL.get(i, 1.0))
            values.append(vars.PWM.get(i, 1.0))
            values.append(vars.IM.get(i, 0))
            values.append(vars.DD.get(i, 0))
            values.append(vars.Q.get(i, 0))
            values.append(vars.EXD.get(i, 0))
            values.append(vars.TIC.get(i, 0))
            values.append(vars.TIM.get(i, 0))
            values.append(vars.TIX.get(i, 0))
            values.append(vars.MRGN.get(i, 0))
            values.append(vars.DIT.get(i, 0))
            values.append(vars.INV.get(i, 0))
            values.append(vars.CG.get(i, 0))
            values.append(vars.VSTK.get(i, 0))
        
        # Income variables
        for h in self.sets.get("H", []):
            values.append(vars.YH.get(h, 0))
            values.append(vars.YHL.get(h, 0))
            values.append(vars.YHK.get(h, 0))
            values.append(vars.YHTR.get(h, 0))
            values.append(vars.YDH.get(h, 0))
            values.append(vars.CTH.get(h, 0))
            values.append(vars.SH.get(h, 0))
            values.append(vars.TDH.get(h, 0))
            
            for i in self.sets.get("I", []):
                values.append(vars.C.get((i, h), 0))
                values.append(vars.CMIN.get((i, h), 0))
                
            for ag in self.sets.get("AG", []):
                values.append(vars.TR.get((h, ag), 0))
        
        for f in self.sets.get("F", []):
            values.append(vars.YF.get(f, 0))
            values.append(vars.YFK.get(f, 0))
            values.append(vars.YFTR.get(f, 0))
            values.append(vars.YDF.get(f, 0))
            values.append(vars.SF.get(f, 0))
            values.append(vars.TDF.get(f, 0))
            
            for ag in self.sets.get("AG", []):
                values.append(vars.TR.get((f, ag), 0))
        
        # Government
        values.append(vars.YG)
        values.append(vars.YGK)
        values.append(vars.TDHT)
        values.append(vars.TDFT)
        values.append(vars.TPRCTS)
        values.append(vars.TPRODN)
        values.append(vars.TIWT)
        values.append(vars.TIKT)
        values.append(vars.TIPT)
        values.append(vars.TICT)
        values.append(vars.TIMT)
        values.append(vars.TIXT)
        values.append(vars.YGTR)
        values.append(vars.G)
        values.append(vars.SG)
        
        # ROW
        values.append(vars.YROW)
        values.append(vars.SROW)
        values.append(vars.CAB)
        
        # Investment
        values.append(vars.IT)
        values.append(vars.GFCF)
        
        # GDP
        values.append(vars.GDP_BP)
        values.append(vars.GDP_MP)
        values.append(vars.GDP_IB)
        values.append(vars.GDP_FD)
        
        # Price indices
        values.append(vars.PIXCON)
        values.append(vars.PIXGDP)
        values.append(vars.PIXGVT)
        values.append(vars.PIXINV)
        
        # Real variables
        for h in self.sets.get("H", []):
            values.append(vars.CTH_REAL.get(h, 0))
        values.append(vars.G_REAL)
        values.append(vars.GDP_BP_REAL)
        values.append(vars.GDP_MP_REAL)
        
        # Exchange rate
        values.append(vars.e)
        
        return np.array(values)

    def _build_variable_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build lower/upper bounds matching _variables_to_array ordering."""
        lb: list[float] = []
        ub: list[float] = []

        def add(lower: float, upper: float) -> None:
            lb.append(lower)
            ub.append(upper)

        POS = (1e-6, 1e10)   # strictly positive (prices, exchange rate)
        NONNEG = (0.0, 1e12) # non-negative quantities/flows
        FREE = (-1e12, 1e12) # potentially signed closure/tax terms

        # Production variables
        for _j in self.sets.get("J", []):
            add(*POS)     # WC
            add(*POS)     # RC
            add(*POS)     # PP
            add(*POS)     # PVA
            add(*NONNEG)  # XST
            add(*NONNEG)  # VA
            add(*NONNEG)  # CI
            add(*NONNEG)  # LDC
            add(*NONNEG)  # KDC

            for _l in self.sets.get("L", []):
                add(*NONNEG)  # LD
                add(*POS)     # WTI

            for _k in self.sets.get("K", []):
                add(*NONNEG)  # KD
                add(*POS)     # RTI
                add(*POS)     # R

            for _i in self.sets.get("I", []):
                add(*NONNEG)  # DI
                add(*NONNEG)  # XS
                add(*NONNEG)  # DS
                add(*NONNEG)  # EX
                add(*POS)     # P

        # Wages
        for _l in self.sets.get("L", []):
            add(*POS)  # W

        # Price and trade variables
        for _i in self.sets.get("I", []):
            add(*POS)     # PC
            add(*POS)     # PD
            add(*POS)     # PM
            add(*POS)     # PE
            add(*POS)     # PE_FOB
            add(*POS)     # PL
            add(*POS)     # PWM
            add(*NONNEG)  # IM
            add(*NONNEG)  # DD
            add(*NONNEG)  # Q
            add(*NONNEG)  # EXD
            add(*FREE)    # TIC
            add(*FREE)    # TIM
            add(*FREE)    # TIX
            add(*FREE)    # MRGN
            add(*FREE)    # DIT
            add(*FREE)    # INV
            add(*FREE)    # CG
            add(*FREE)    # VSTK

        # Income variables
        for _h in self.sets.get("H", []):
            add(*NONNEG)  # YH
            add(*NONNEG)  # YHL
            add(*NONNEG)  # YHK
            add(*FREE)    # YHTR
            add(*NONNEG)  # YDH
            add(*NONNEG)  # CTH
            add(*FREE)    # SH
            add(*FREE)    # TDH

            for _i in self.sets.get("I", []):
                add(*NONNEG)  # C
                add(*NONNEG)  # CMIN

            for _ag in self.sets.get("AG", []):
                add(*FREE)  # TR(h,ag)

        for _f in self.sets.get("F", []):
            add(*NONNEG)  # YF
            add(*NONNEG)  # YFK
            add(*FREE)    # YFTR
            add(*NONNEG)  # YDF
            add(*FREE)    # SF
            add(*FREE)    # TDF

            for _ag in self.sets.get("AG", []):
                add(*FREE)  # TR(f,ag)

        # Government
        add(*FREE)  # YG
        add(*FREE)  # YGK
        add(*FREE)  # TDHT
        add(*FREE)  # TDFT
        add(*FREE)  # TPRCTS
        add(*FREE)  # TPRODN
        add(*FREE)  # TIWT
        add(*FREE)  # TIKT
        add(*FREE)  # TIPT
        add(*FREE)  # TICT
        add(*FREE)  # TIMT
        add(*FREE)  # TIXT
        add(*FREE)  # YGTR
        add(*FREE)  # G
        add(*FREE)  # SG

        # ROW
        add(*FREE)  # YROW
        add(*FREE)  # SROW
        add(*FREE)  # CAB

        # Investment
        add(*FREE)  # IT
        add(*FREE)  # GFCF

        # GDP
        add(*FREE)  # GDP_BP
        add(*FREE)  # GDP_MP
        add(*FREE)  # GDP_IB
        add(*FREE)  # GDP_FD

        # Price indices
        add(*POS)  # PIXCON
        add(*POS)  # PIXGDP
        add(*POS)  # PIXGVT
        add(*POS)  # PIXINV

        # Real variables
        for _h in self.sets.get("H", []):
            add(*NONNEG)  # CTH_REAL
        add(*NONNEG)  # G_REAL
        add(*NONNEG)  # GDP_BP_REAL
        add(*NONNEG)  # GDP_MP_REAL

        # Exchange rate
        add(*POS)  # e

        return np.array(lb), np.array(ub)

    def _build_residual_weights(self) -> dict[str, float]:
        """Weights for equation blocks in IPOPT least-squares objective."""
        return {
            "EQ52": 3.0,  # LES demand block
            "EQ45": 2.0,  # CAB/ROW balance
        }

    def _build_hard_constraints(self) -> list[str]:
        """Equation residuals enforced as hard equalities c(x)=0."""
        vars0 = self._create_initial_guess()
        residuals0 = self.equations.calculate_all_residuals(vars0)
        return list(residuals0.keys())
    
    def solve_ipopt(self) -> SolverResult:
        """Solve model using IPOPT nonlinear optimization.
        
        Returns:
            SolverResult with solution
        """
        if not IPOPT_AVAILABLE:
            logger.error("IPOPT not available. Install with: pip install cyipopt")
            raise ImportError("cyipopt not installed")
        
        logger.info("=" * 70)
        logger.info("STARTING IPOPT SOLUTION")
        logger.info("=" * 70)
        
        # Create initial guess
        vars = self._create_initial_guess()
        x0 = self._variables_to_array(vars)
        n_vars = len(x0)
        
        logger.info(f"Number of variables: {n_vars}")

        # If benchmark initialization already satisfies the system, skip IPOPT.
        init_residuals = self.equations.calculate_all_residuals(vars)
        init_vals = np.array(list(init_residuals.values()), dtype=float)
        init_rms = float(np.sqrt(np.mean(init_vals ** 2))) if init_vals.size else 0.0
        if init_rms <= self.tolerance:
            logger.info(
                "Initial guess satisfies tolerance (RMS=%.3e <= %.3e); skipping IPOPT.",
                init_rms,
                self.tolerance,
            )
            return SolverResult(
                converged=True,
                iterations=0,
                final_residual=init_rms,
                variables=vars,
                residuals=init_residuals,
                message="Initial equation-consistent benchmark satisfies tolerance",
            )
        
        # Create problem
        hard_constraints = self._build_hard_constraints()
        problem = CGEProblem(
            equations=self.equations,
            sets=self.sets,
            n_variables=n_vars,
            variable_info={},
            residual_weights=self._build_residual_weights(),
            hard_constraints=hard_constraints,
            reference_x=x0,
        )
        
        # Set variable bounds by variable class (aligned with packing order)
        lb, ub = self._build_variable_bounds()
        if len(lb) != n_vars:
            raise ValueError(f"Bounds length {len(lb)} does not match variable count {n_vars}")
        
        # Create IPOPT problem
        class IPOPTProblem:
            def __init__(self, cge_problem):
                self.cge = cge_problem
                
            def objective(self, x):
                return self.cge.objective(x)
            
            def gradient(self, x):
                return self.cge.gradient(x)

            def constraints(self, x):
                return self.cge.constraints(x)

            def jacobian(self, x):
                return self.cge.jacobian(x)

            def jacobianstructure(self):
                return self.cge.jacobianstructure()
        
        ipopt_problem = IPOPTProblem(problem)
        
        # Configure IPOPT options using cyipopt API
        n_constraints = len(problem.hard_constraints)
        cl = np.zeros(n_constraints)
        cu = np.zeros(n_constraints)
        def _build_nlp(max_iter: int, warm_start: bool) -> cyipopt.Problem:
            nlp = cyipopt.Problem(
                n=n_vars,
                m=n_constraints,
                problem_obj=ipopt_problem,
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu,
            )
            nlp.add_option("tol", self.tolerance)
            nlp.add_option("acceptable_tol", max(self.tolerance * 10, 1e-6))
            nlp.add_option("acceptable_iter", 12)
            nlp.add_option("max_iter", max_iter)
            nlp.add_option("print_level", 5 if logger.isEnabledFor(logging.DEBUG) else 3)
            nlp.add_option("mu_strategy", "adaptive")
            nlp.add_option("mu_init", 1e-2 if not warm_start else 1e-4)
            nlp.add_option("hessian_approximation", "limited-memory")
            nlp.add_option("limited_memory_max_history", 25)
            nlp.add_option("nlp_scaling_method", "gradient-based")
            nlp.add_option("bound_frac", 1e-2)
            nlp.add_option("bound_push", 1e-2)
            nlp.add_option("watchdog_shortened_iter_trigger", 5)
            if warm_start:
                nlp.add_option("warm_start_init_point", "yes")
                nlp.add_option("warm_start_bound_push", 1e-8)
                nlp.add_option("warm_start_mult_bound_push", 1e-8)
                nlp.add_option("warm_start_slack_bound_push", 1e-8)
            return nlp

        def _info_iterations(info: dict[str, Any], fallback: int) -> int:
            for key in ("iter_count", "iter", "iterations"):
                val = info.get(key)
                if val is not None:
                    try:
                        return int(val)
                    except Exception:
                        pass
            return fallback

        logger.info("IPOPT options configured")
        logger.info(f"  Tolerance: {self.tolerance}")
        logger.info(f"  Max iterations: {self.max_iterations}")
        logger.info("Starting IPOPT optimization (pass 1)...")
        try:
            pass1_iter = max(80, min(self.max_iterations, 220))
            nlp = _build_nlp(max_iter=pass1_iter, warm_start=False)
            x1, info1 = nlp.solve(x0)
            msg1 = info1.get("status_msg", "Unknown")
            if isinstance(msg1, bytes):
                msg1 = msg1.decode("utf-8", errors="replace")
            logger.info("IPOPT pass 1 status=%s msg=%s", info1.get("status"), msg1)

            # Optional second pass with warm start if first pass did not converge.
            x_best, info_best = x1, info1
            if info1.get("status") != 0:
                logger.info("Starting IPOPT optimization (pass 2, warm start)...")
                nlp2 = _build_nlp(max_iter=self.max_iterations, warm_start=True)
                x2, info2 = nlp2.solve(x1)
                msg2 = info2.get("status_msg", "Unknown")
                if isinstance(msg2, bytes):
                    msg2 = msg2.decode("utf-8", errors="replace")
                logger.info("IPOPT pass 2 status=%s msg=%s", info2.get("status"), msg2)

                # Select best pass by convergence first, then lower residual RMS.
                v1 = problem._array_to_variables(x1)
                r1 = self.equations.calculate_all_residuals(v1)
                vals1 = np.array(list(r1.values()), dtype=float)
                rms1 = float(np.sqrt(np.mean(vals1**2))) if vals1.size else float("inf")

                v2 = problem._array_to_variables(x2)
                r2 = self.equations.calculate_all_residuals(v2)
                vals2 = np.array(list(r2.values()), dtype=float)
                rms2 = float(np.sqrt(np.mean(vals2**2))) if vals2.size else float("inf")

                if info2.get("status") == 0 and info1.get("status") != 0:
                    x_best, info_best = x2, info2
                elif info2.get("status") == info1.get("status") and rms2 < rms1:
                    x_best, info_best = x2, info2

            # Create result
            result = SolverResult()
            result.converged = info_best.get("status") == 0
            result.iterations = _info_iterations(info_best, self.max_iterations)
            result.variables = problem._array_to_variables(x_best)
            result.residuals = self.equations.calculate_all_residuals(result.variables)
            msg = info_best.get("status_msg", "Unknown")
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8", errors="replace")
            result.message = str(msg)
            vals = np.array(list(result.residuals.values()), dtype=float)
            result.final_residual = float(np.sqrt(np.mean(vals**2))) if vals.size else 0.0

            logger.info("IPOPT info keys: %s", sorted(info_best.keys()))
            logger.info(f"IPOPT finished: {result.message}")
            logger.info(f"  Status: {info_best.get('status')}")
            logger.info(f"  Iterations: {result.iterations}")
            logger.info(f"  Final objective: {info_best.get('obj_val', 0):.2e}")
            logger.info(f"  Function evaluations: {problem.n_evaluations}")

            return result

        except Exception as e:
            logger.error(f"IPOPT failed: {e}")
            result = SolverResult()
            result.message = f"IPOPT error: {str(e)}"
            return result
    
    def solve(self, method: str = "auto") -> SolverResult:
        """Solve using best available method.
        
        Args:
            method: "auto", "ipopt", or "simple_iteration"
            
        Returns:
            SolverResult
        """
        if method == "auto":
            if IPOPT_AVAILABLE:
                logger.info("Auto-selecting IPOPT solver")
                return self.solve_ipopt()
            else:
                logger.info("Auto-selecting simple iteration (IPOPT not available)")
                return super().solve(method="simple_iteration")
        elif method == "ipopt":
            return self.solve_ipopt()
        else:
            return super().solve(method="simple_iteration")
