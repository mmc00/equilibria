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
import copy
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple

import numpy as np

from equilibria.baseline.compatibility import (
    BaselineCompatibilityReport,
    evaluate_strict_gams_baseline_compatibility,
)
from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.blocks.equilibrium import PEPMacroClosureInit
from equilibria.blocks.production import PEPProductionAccountingInit
from equilibria.blocks.trade import (
    PEPCommodityBalanceInit,
    PEPTradeFlowInit,
    PEPTradeMarketClearingInit,
    PEPTradeTransformationInit,
)
from equilibria.core.sets import Set, SetManager
from equilibria.solver.guards import rebuild_tax_detail_from_rates
from equilibria.solver.transforms import pep_array_to_variables, pep_variables_to_array
from equilibria.templates.init_strategies import build_init_strategy, normalize_init_mode
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
        vars = pep_array_to_variables(x, self.sets)
        rebuild_tax_detail_from_rates(
            vars=vars,
            sets=self.sets,
            params=self.equations.params,
            include_tip=True,
        )
        return vars
    
    def _variables_to_array(self, vars: PEPModelVariables) -> np.ndarray:
        """Convert PEPModelVariables to array."""
        return pep_variables_to_array(vars, self.sets)


class IPOPTSolver:
    """IPOPT-based solver for PEP model."""
    
    def __init__(
        self,
        calibrated_state: Any,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        init_mode: Literal["gams", "excel"] | str = "excel",
        blockwise_commodity_alpha: float = 0.75,
        blockwise_trade_market_alpha: float = 0.5,
        blockwise_macro_alpha: float = 1.0,
        gams_results_gdx: Path | str | None = None,
        gams_results_slice: Literal["base", "sim1"] = "sim1",
        baseline_manifest: Path | str | None = None,
        require_baseline_manifest: bool = False,
        baseline_compatibility_rel_tol: float = 1e-4,
        enforce_strict_gams_baseline: bool = True,
        sam_file: Path | str | None = None,
        val_par_file: Path | str | None = None,
        gdxdump_bin: str = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
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
        requested_init_mode = str(init_mode).strip().lower()
        self.init_mode = normalize_init_mode(init_mode)
        self.blockwise_commodity_alpha = blockwise_commodity_alpha
        self.blockwise_trade_market_alpha = blockwise_trade_market_alpha
        self.blockwise_macro_alpha = blockwise_macro_alpha
        self.gams_results_gdx = Path(gams_results_gdx) if gams_results_gdx is not None else None
        self.gams_results_slice = gams_results_slice.lower()
        self.baseline_manifest = Path(baseline_manifest) if baseline_manifest is not None else None
        self.require_baseline_manifest = require_baseline_manifest
        self.baseline_compatibility_rel_tol = baseline_compatibility_rel_tol
        self.enforce_strict_gams_baseline = enforce_strict_gams_baseline
        self.sam_file = Path(sam_file) if sam_file is not None else None
        self.val_par_file = Path(val_par_file) if val_par_file is not None else None
        self.gdxdump_bin = gdxdump_bin
        self.strict_baseline_report: BaselineCompatibilityReport | None = None
        self._strict_baseline_checked = False
        
        # Extract sets and parameters from calibrated state
        self.sets = calibrated_state.sets
        self.params = self._extract_parameters(calibrated_state)
        
        # Initialize equations
        self.equations = PEPModelEquations(self.sets, self.params)
        
        logger.info(f"Initialized IPOPT Solver")
        logger.info(f"  Sets: {len(self.sets)} categories")
        logger.info(f"  Tolerance: {tolerance}")
        logger.info(f"  Max iterations: {max_iterations}")
        logger.info(f"  Init mode: {self.init_mode}")
        if requested_init_mode != self.init_mode:
            logger.info(
                "  Init mode alias normalized: %s -> %s",
                requested_init_mode,
                self.init_mode,
            )
        if self.gams_results_gdx is not None:
            logger.info(f"  GAMS levels: {self.gams_results_gdx}")
            logger.info(f"  GAMS slice: {self.gams_results_slice.upper()}")
        if self.baseline_manifest is not None:
            logger.info(f"  Baseline manifest: {self.baseline_manifest}")

    def _uses_excel_init(self) -> bool:
        """Return True when running the calibrated SAM/Excel initialization path."""
        return self.init_mode == "excel"

    def _build_block_set_manager(self) -> SetManager:
        """Build SetManager from calibrated set lists for block hooks."""
        sm = SetManager()
        for set_name, elems in self.sets.items():
            try:
                sm.add(Set(name=set_name, elements=tuple(str(e) for e in elems)))
            except Exception:
                continue
        return sm

    def _apply_trade_blockwise_flow(self, vars: PEPModelVariables) -> None:
        """Apply trade blockwise init/validation (EQ57-EQ64)."""
        block = PEPTradeFlowInit()
        sm = self._build_block_set_manager()

        params_block: dict[str, Any] = dict(self.params)
        params_block.update(
            {
                "QO0": self.state.trade.get("QO", {}),
                "DDO0": self.state.trade.get("DDO", {}),
                "IMO0": self.state.trade.get("IMO", {}),
                "EXDO0": self.state.trade.get("EXDO", {}),
                "PCO0": self.state.trade.get("PCO", {}),
                "PDO0": self.state.trade.get("PDO", {}),
                "PMO0": self.state.trade.get("PMO", {}),
                "MRGNO0": self.state.trade.get("MRGNO", {}),
                "DITO0": self.state.production.get("DITO", {}),
                "e": vars.e,
            }
        )

        vars_block: dict[str, Any] = {
            "Q": dict(vars.Q),
            "DD": dict(vars.DD),
            "IM": dict(vars.IM),
            "EXD": dict(vars.EXD),
            "PC": dict(vars.PC),
            "PD": dict(vars.PD),
            "PM": dict(vars.PM),
            "MRGN": dict(vars.MRGN),
            "DIT": dict(vars.DIT),
            "DI": dict(vars.DI),
            "XS": dict(vars.XS),
            "XST": dict(vars.XST),
            "DS": dict(vars.DS),
            "EX": dict(vars.EX),
            "P": dict(vars.P),
            "PT": dict(vars.PT),
            "PE": dict(vars.PE),
            "PL": dict(vars.PL),
            "PWM": dict(vars.PWM),
            "PE_FOB": dict(vars.PE_FOB),
        }

        block.initialize_levels(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
            mode="gams_blockwise",
        )

        vars.Q.update(vars_block.get("Q", {}))
        vars.DD.update(vars_block.get("DD", {}))
        vars.IM.update(vars_block.get("IM", {}))
        vars.EXD.update(vars_block.get("EXD", {}))
        vars.PC.update(vars_block.get("PC", {}))
        vars.PD.update(vars_block.get("PD", {}))
        vars.PM.update(vars_block.get("PM", {}))
        vars.MRGN.update(vars_block.get("MRGN", {}))
        vars.DIT.update(vars_block.get("DIT", {}))

        diagnostics = block.validate_initialization(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
        )
        if diagnostics:
            max_abs = max(abs(v) for v in diagnostics.values())
            top = sorted(diagnostics.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
            logger.info(
                "gams_blockwise trade residuals: max=%0.6e | %s",
                max_abs,
                ", ".join(f"{k}={v:.3e}" for k, v in top),
            )

    def _apply_trade_blockwise_transformation(self, vars: PEPModelVariables) -> None:
        """Apply trade transformation blockwise init/validation (EQ58-EQ59)."""
        block = PEPTradeTransformationInit()
        sm = self._build_block_set_manager()

        params_block: dict[str, Any] = dict(self.params)
        vars_block: dict[str, Any] = {
            "XS": dict(vars.XS),
            "XST": dict(vars.XST),
            "P": dict(vars.P),
            "PT": dict(vars.PT),
        }

        block.initialize_levels(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
            mode="gams_blockwise",
        )
        vars.XST.update(vars_block.get("XST", {}))

        diagnostics = block.validate_initialization(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
        )
        if diagnostics:
            max_abs = max(abs(v) for v in diagnostics.values())
            top = sorted(diagnostics.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
            logger.info(
                "gams_blockwise transformation residuals: max=%0.6e | %s",
                max_abs,
                ", ".join(f"{k}={v:.3e}" for k, v in top),
            )

    def _apply_production_blockwise_accounting(self, vars: PEPModelVariables) -> None:
        """Apply production accounting blockwise init/validation (EQ2/EQ65-EQ67)."""
        block = PEPProductionAccountingInit()
        sm = self._build_block_set_manager()

        params_block: dict[str, Any] = dict(self.params)
        vars_block: dict[str, Any] = {
            "XST": dict(vars.XST),
            "CI": dict(vars.CI),
            "PP": dict(vars.PP),
            "PT": dict(vars.PT),
            "PCI": dict(vars.PCI),
            "PVA": dict(vars.PVA),
            "VA": dict(vars.VA),
            "DI": dict(vars.DI),
            "DIT": dict(vars.DIT),
            "PC": dict(vars.PC),
        }

        block.initialize_levels(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
            mode="gams_blockwise",
        )

        vars.XST.update(vars_block.get("XST", {}))
        vars.CI.update(vars_block.get("CI", {}))
        vars.PP.update(vars_block.get("PP", {}))
        vars.PT.update(vars_block.get("PT", {}))
        vars.PCI.update(vars_block.get("PCI", {}))
        vars.DI.update(vars_block.get("DI", {}))
        vars.DIT.update(vars_block.get("DIT", {}))

        # Keep production-tax aggregates aligned with updated unit costs.
        for j in self.sets.get("J", []):
            ttip = self.params.get("ttip", {}).get(j, 0.0)
            vars.TIP[j] = ttip * vars.PP.get(j, 0.0) * vars.XST.get(j, 0.0)
        vars.TIPT = sum(vars.TIP.values())
        vars.TPRODN = vars.TIWT + vars.TIKT + vars.TIPT
        vars.YG = vars.YGK + vars.TDHT + vars.TDFT + vars.TPRODN + vars.TPRCTS + vars.YGTR
        tr_to_govt = sum(vars.TR.get((agng, "gvt"), 0.0) for agng in self.sets.get("AGNG", []))
        vars.SG = vars.YG - tr_to_govt - vars.G

        diagnostics = block.validate_initialization(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
        )
        if diagnostics:
            max_abs = max(abs(v) for v in diagnostics.values())
            top = sorted(diagnostics.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
            logger.info(
                "gams_blockwise production residuals: max=%0.6e | %s",
                max_abs,
                ", ".join(f"{k}={v:.3e}" for k, v in top),
            )

    def _apply_commodity_balance_blockwise(self, vars: PEPModelVariables) -> None:
        """Apply commodity balance blockwise init/validation (EQ57/EQ63/EQ79/EQ84)."""
        block = PEPCommodityBalanceInit()
        sm = self._build_block_set_manager()

        params_block: dict[str, Any] = dict(self.params)
        vars_block: dict[str, Any] = {
            "Q": dict(vars.Q),
            "DD": dict(vars.DD),
            "IM": dict(vars.IM),
            "EXD": dict(vars.EXD),
            "PC": dict(vars.PC),
            "PD": dict(vars.PD),
            "PM": dict(vars.PM),
            "MRGN": dict(vars.MRGN),
            "C": dict(vars.C),
            "CG": dict(vars.CG),
            "INV": dict(vars.INV),
            "VSTK": dict(vars.VSTK),
            "DIT": dict(vars.DIT),
        }

        block.initialize_levels(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
            mode="gams_blockwise",
        )

        alpha = max(0.0, min(1.0, float(self.blockwise_commodity_alpha)))
        for name, target in (
            ("Q", vars.Q),
            ("DD", vars.DD),
            ("IM", vars.IM),
            ("EXD", vars.EXD),
            ("MRGN", vars.MRGN),
        ):
            updated = vars_block.get(name, {})
            for k, v_new in updated.items():
                v_old = target.get(k, 0.0)
                target[k] = (1.0 - alpha) * v_old + alpha * float(v_new)

        diagnostics = block.validate_initialization(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
        )
        if diagnostics:
            max_abs = max(abs(v) for v in diagnostics.values())
            top = sorted(diagnostics.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
            logger.info(
                "gams_blockwise commodity residuals (alpha=%0.2f): max=%0.6e | %s",
                alpha,
                max_abs,
                ", ".join(f"{k}={v:.3e}" for k, v in top),
            )

    def _apply_trade_market_clearing_blockwise(self, vars: PEPModelVariables) -> None:
        """Apply trade market-clearing blockwise init/validation (EQ64/EQ88)."""
        block = PEPTradeMarketClearingInit()
        sm = self._build_block_set_manager()

        params_block: dict[str, Any] = dict(self.params)
        params_block["trade_market_alpha"] = float(self.blockwise_trade_market_alpha)
        vars_block: dict[str, Any] = {
            "DS": dict(vars.DS),
            "DD": dict(vars.DD),
            "IM": dict(vars.IM),
            "PD": dict(vars.PD),
            "PM": dict(vars.PM),
        }

        block.initialize_levels(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
            mode="gams_blockwise",
        )

        vars.DD.update(vars_block.get("DD", {}))
        vars.IM.update(vars_block.get("IM", {}))

        diagnostics = block.validate_initialization(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
        )
        if diagnostics:
            max_abs = max(abs(v) for v in diagnostics.values())
            top = sorted(diagnostics.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
            logger.info(
                "gams_blockwise market residuals (alpha=%0.2f): max=%0.6e | %s",
                float(self.blockwise_trade_market_alpha),
                max_abs,
                ", ".join(f"{k}={v:.3e}" for k, v in top),
            )

    def _apply_macro_closure_blockwise(self, vars: PEPModelVariables) -> None:
        """Apply macro closure blockwise init/validation (EQ44/EQ45/EQ46/EQ87/EQ93)."""
        block = PEPMacroClosureInit()
        sm = self._build_block_set_manager()

        params_block: dict[str, Any] = dict(self.params)
        params_block["macro_alpha"] = float(self.blockwise_macro_alpha)
        vars_block: dict[str, Any] = {
            "PWM": dict(vars.PWM),
            "IM": dict(vars.IM),
            "R": dict(vars.R),
            "KD": dict(vars.KD),
            "TR": dict(vars.TR),
            "PE_FOB": dict(vars.PE_FOB),
            "EXD": dict(vars.EXD),
            "SH": dict(vars.SH),
            "SF": dict(vars.SF),
            "PC": dict(vars.PC),
            "C": dict(vars.C),
            "CG": dict(vars.CG),
            "INV": dict(vars.INV),
            "VSTK": dict(vars.VSTK),
            "e": vars.e,
            "YROW": vars.YROW,
            "SROW": vars.SROW,
            "CAB": vars.CAB,
            "IT": vars.IT,
            "GFCF": vars.GFCF,
            "GDP_FD": vars.GDP_FD,
            "SG": vars.SG,
        }

        block.initialize_levels(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
            mode="gams_blockwise",
        )

        vars.YROW = float(vars_block.get("YROW", vars.YROW))
        vars.SROW = float(vars_block.get("SROW", vars.SROW))
        vars.CAB = float(vars_block.get("CAB", vars.CAB))
        vars.IT = float(vars_block.get("IT", vars.IT))
        vars.GFCF = float(vars_block.get("GFCF", vars.GFCF))
        vars.GDP_FD = float(vars_block.get("GDP_FD", vars.GDP_FD))

        diagnostics = block.validate_initialization(
            set_manager=sm,
            parameters=params_block,
            variables=vars_block,
        )
        if diagnostics:
            max_abs = max(abs(v) for v in diagnostics.values())
            top = sorted(diagnostics.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
            logger.info(
                "gams_blockwise macro residuals (alpha=%0.2f): max=%0.6e | %s",
                float(self.blockwise_macro_alpha),
                max_abs,
                ", ".join(f"{k}={v:.3e}" for k, v in top),
            )

    def _reconcile_tax_identities(self, vars: PEPModelVariables) -> None:
        """Recompute tax-payment variables from current prices/quantities."""
        for j in self.sets.get("J", []):
            ttip = self.params.get("ttip", {}).get(j, 0.0)
            vars.TIP[j] = ttip * vars.PP.get(j, 0.0) * vars.XST.get(j, 0.0)
        vars.TIPT = sum(vars.TIP.values())

        for i in self.sets.get("I", []):
            ttic = self.params.get("ttic", {}).get(i, 0.0)
            denom = 1.0 + ttic
            if abs(denom) < 1e-12:
                vars.TIC[i] = 0.0
            else:
                vars.TIC[i] = (ttic / denom) * (
                    vars.PD.get(i, 0.0) * vars.DD.get(i, 0.0)
                    + vars.PM.get(i, 0.0) * vars.IM.get(i, 0.0)
                )

            ttim = self.params.get("ttim", {}).get(i, 0.0)
            vars.TIM[i] = ttim * vars.e * vars.PWM.get(i, 1.0) * vars.IM.get(i, 0.0)

            ttix = self.params.get("ttix", {}).get(i, 0.0)
            margin_sum = sum(
                vars.PC.get(ij, 1.0) * self.params.get("tmrg_X", {}).get((ij, i), 0.0)
                for ij in self.sets.get("I", [])
            )
            vars.TIX[i] = ttix * (vars.PE.get(i, 0.0) + margin_sum) * vars.EXD.get(i, 0.0)

        vars.TICT = sum(vars.TIC.values())
        vars.TIMT = sum(vars.TIM.values())
        vars.TIXT = sum(vars.TIX.values())
        vars.TPRCTS = vars.TICT + vars.TIMT + vars.TIXT
        vars.TPRODN = vars.TIWT + vars.TIKT + vars.TIPT
        vars.YG = vars.YGK + vars.TDHT + vars.TDFT + vars.TPRODN + vars.TPRCTS + vars.YGTR
        tr_to_govt = sum(vars.TR.get((agng, "gvt"), 0.0) for agng in self.sets.get("AGNG", []))
        vars.SG = vars.YG - tr_to_govt - vars.G

    def _recompute_gdp_aggregates(self, vars: PEPModelVariables) -> None:
        """Recompute GDP aggregate identities from current levels."""
        vars.GDP_BP = sum(
            vars.PVA.get(j, 0.0) * vars.VA.get(j, 0.0) for j in self.sets.get("J", [])
        ) + vars.TIPT
        vars.GDP_MP = vars.GDP_BP + vars.TPRCTS

        gdp_ib = 0.0
        for l in self.sets.get("L", []):
            for j in self.sets.get("J", []):
                gdp_ib += vars.W.get(l, 1.0) * vars.LD.get((l, j), 0.0)
        for k in self.sets.get("K", []):
            for j in self.sets.get("J", []):
                gdp_ib += vars.R.get((k, j), 1.0) * vars.KD.get((k, j), 0.0)
        gdp_ib += vars.TPRODN + vars.TPRCTS
        vars.GDP_IB = gdp_ib

        gdp_fd = 0.0
        for i in self.sets.get("I", []):
            cons_i = sum(vars.C.get((i, h), 0.0) for h in self.sets.get("H", []))
            gdp_fd += vars.PC.get(i, 0.0) * (
                cons_i + vars.CG.get(i, 0.0) + vars.INV.get(i, 0.0) + vars.VSTK.get(i, 0.0)
            )
            gdp_fd += vars.PE_FOB.get(i, 0.0) * vars.EXD.get(i, 0.0)
            gdp_fd -= vars.PWM.get(i, 0.0) * vars.e * vars.IM.get(i, 0.0)
        vars.GDP_FD = gdp_fd

    def _reconcile_composite_prices(self, vars: PEPModelVariables) -> None:
        """Recompute PC(i) from EQ79 after DD/IM updates."""
        imo0 = self.params.get("IMO0", {})
        ddo0 = self.params.get("DDO0", {})
        for i in self.sets.get("I", []):
            q_i = float(vars.Q.get(i, 0.0))
            if abs(q_i) <= 1e-12:
                continue
            rhs = 0.0
            if abs(float(imo0.get(i, 0.0))) > 1e-12:
                rhs += float(vars.PM.get(i, 0.0)) * float(vars.IM.get(i, 0.0))
            if abs(float(ddo0.get(i, 0.0))) > 1e-12:
                rhs += float(vars.PD.get(i, 0.0)) * float(vars.DD.get(i, 0.0))
            vars.PC[i] = rhs / q_i

    @staticmethod
    def _coupled_trade_score(residuals: dict[str, float]) -> float:
        """Weighted score for coupled trade block (lower is better)."""
        keys = [
            k for k in residuals
            if k.startswith("EQ63_") or k.startswith("EQ64_") or k.startswith("EQ84_") or k.startswith("EQ88_")
        ]
        if not keys:
            return 0.0
        return float(sum(residuals[k] ** 2 for k in keys))

    @staticmethod
    def _residual_rms_max(residuals: dict[str, float]) -> tuple[float, float]:
        vals = np.array(list(residuals.values()), dtype=float)
        if vals.size == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(np.mean(vals ** 2)))
        mx = float(np.max(np.abs(vals)))
        return rms, mx

    @staticmethod
    def _focus_blockwise_residuals(residuals: dict[str, float]) -> dict[str, float]:
        """Subset of residuals that drive blockwise pre-solve stabilization."""
        selected: dict[str, float] = {}
        scalar_keys = {"EQ44", "EQ45", "EQ46", "EQ87", "EQ93"}
        prefixes = ("EQ57_", "EQ63_", "EQ64_", "EQ79_", "EQ84_", "EQ88_")
        for key, value in residuals.items():
            if key in scalar_keys or key.startswith(prefixes):
                selected[key] = value
        return selected

    def _apply_gams_blockwise_presolve(self, vars: PEPModelVariables) -> None:
        """
        Run iterative GAMS-style blockwise reconciliation before IPOPT.

        This mirrors the block sequence used in GAMS-oriented initialization and
        avoids jumping straight from raw calibrated levels to nonlinear solve.
        """
        max_passes = 8
        target_focus_max = 1e-3
        target_overall_max = 1e-3
        improve_ratio = 0.995  # require at least 0.5% score improvement
        stale_limit = 2

        prev_score = float("inf")
        stale = 0

        for step in range(1, max_passes + 1):
            self._apply_trade_blockwise_flow(vars)
            self._apply_trade_blockwise_transformation(vars)
            self._apply_production_blockwise_accounting(vars)
            self._apply_commodity_balance_blockwise(vars)
            self._apply_trade_market_clearing_blockwise(vars)
            self._attempt_coupled_trade_reconciliation(vars)
            self._reconcile_composite_prices(vars)
            self._reconcile_tax_identities(vars)
            self._apply_macro_closure_blockwise(vars)
            self._recompute_gdp_aggregates(vars)

            residuals = self.equations.calculate_all_residuals(vars)
            overall_rms, overall_max = self._residual_rms_max(residuals)
            focus = self._focus_blockwise_residuals(residuals)
            focus_rms, focus_max = self._residual_rms_max(focus)
            score = self._coupled_trade_score(residuals)

            logger.info(
                "gams_blockwise presolve pass %d/%d: focus_rms=%0.6e focus_max=%0.6e | overall_rms=%0.6e overall_max=%0.6e",
                step,
                max_passes,
                focus_rms,
                focus_max,
                overall_rms,
                overall_max,
            )

            if focus_max <= target_focus_max and overall_max <= target_overall_max:
                logger.info("gams_blockwise presolve reached practical residual target")
                break

            if score < prev_score * improve_ratio:
                prev_score = score
                stale = 0
            else:
                stale += 1
                if stale >= stale_limit:
                    logger.info(
                        "gams_blockwise presolve stopped after %d stale passes (no meaningful improvement)",
                        stale,
                    )
                    break

    def _attempt_coupled_trade_reconciliation(self, vars: PEPModelVariables) -> bool:
        """
        Try to reduce EQ63/EQ64/EQ84/EQ88 jointly; keep only if score improves.
        """
        baseline = copy.deepcopy(vars)
        res0 = self.equations.calculate_all_residuals(vars)
        score0 = self._coupled_trade_score(res0)
        rms0, max0 = self._residual_rms_max(res0)

        I = list(self.sets.get("I", []))
        J = list(self.sets.get("J", []))
        H = list(self.sets.get("H", []))
        ddo0 = self.params.get("DDO0", {})
        imo0 = self.params.get("IMO0", {})
        dso0 = self.params.get("DSO0", {})
        exdo0 = self.params.get("EXDO0", self.params.get("EXDO", {}))
        beta_m = self.params.get("beta_M", {})
        sigma_m = self.params.get("sigma_M", {})
        rho_m = self.params.get("rho_M", {})
        b_m = self.params.get("B_M", {})
        tmrg = self.params.get("tmrg", {})
        tmrg_x = self.params.get("tmrg_X", {})

        def _im_from64(i: str, dd: float) -> float:
            if abs(float(imo0.get(i, 0.0))) <= 1e-12 or abs(float(ddo0.get(i, 0.0))) <= 1e-12:
                return max(0.0, float(vars.IM.get(i, 0.0)))
            beta = float(beta_m.get(i, 0.0))
            sig = float(sigma_m.get(i, 2.0))
            pd = float(vars.PD.get(i, 0.0))
            pm = float(vars.PM.get(i, 0.0))
            if beta <= 0.0 or beta >= 1.0 or pd <= 0.0 or pm <= 0.0:
                return max(0.0, float(vars.IM.get(i, 0.0)))
            ratio = (beta / (1.0 - beta)) * (pd / pm)
            if ratio <= 0.0:
                return max(0.0, float(vars.IM.get(i, 0.0)))
            return max(0.0, float((ratio ** sig) * dd))

        def _q_from63(i: str, dd: float, im: float) -> float:
            rho = float(rho_m.get(i, -0.5))
            b = float(b_m.get(i, 1.0))
            beta = float(beta_m.get(i, 0.5))
            if abs(rho) <= 1e-12 or b <= 0.0:
                return max(0.0, float(vars.Q.get(i, 0.0)))
            term = 0.0
            if abs(float(imo0.get(i, 0.0))) > 1e-12 and im > 0.0:
                term += beta * (im ** (-rho))
            if abs(float(ddo0.get(i, 0.0))) > 1e-12 and dd > 0.0:
                term += (1.0 - beta) * (dd ** (-rho))
            if term <= 0.0:
                return 0.0
            try:
                q = b * (term ** (-1.0 / rho))
            except Exception:
                q = float(vars.Q.get(i, 0.0))
            return max(0.0, float(q))

        def _recompute_mrgn() -> None:
            for i in I:
                m = 0.0
                for ij in I:
                    t = float(tmrg.get((i, ij), 0.0))
                    if abs(float(ddo0.get(ij, 0.0))) > 1e-12:
                        m += t * float(vars.DD.get(ij, 0.0))
                    if abs(float(imo0.get(ij, 0.0))) > 1e-12:
                        m += t * float(vars.IM.get(ij, 0.0))
                    if abs(float(exdo0.get(ij, 0.0))) > 1e-12:
                        m += float(tmrg_x.get((i, ij), 0.0)) * float(vars.EXD.get(ij, 0.0))
                vars.MRGN[i] = m

        # Freeze DD from EQ88 supply as requested.
        dd_fixed: dict[str, float] = {}
        for i in I:
            if abs(float(ddo0.get(i, 0.0))) > 1e-12:
                dd_fixed[i] = sum(
                    float(vars.DS.get((j, i), 0.0))
                    for j in J
                    if abs(float(dso0.get((j, i), 0.0))) > 1e-12
                )
                vars.DD[i] = max(0.0, dd_fixed[i])

        # Weights for local coupled objective.
        w63 = 1.0
        w84 = 1.0
        w64 = 0.05

        for _ in range(3):
            _recompute_mrgn()
            for i in I:
                has_dd = abs(float(ddo0.get(i, 0.0))) > 1e-12
                has_im = abs(float(imo0.get(i, 0.0))) > 1e-12
                dd_i = max(0.0, float(vars.DD.get(i, 0.0))) if has_dd else 0.0

                q84 = max(
                    0.0,
                    sum(float(vars.C.get((i, h), 0.0)) for h in H)
                    + float(vars.CG.get(i, 0.0))
                    + float(vars.INV.get(i, 0.0))
                    + float(vars.VSTK.get(i, 0.0))
                    + float(vars.DIT.get(i, 0.0))
                    + float(vars.MRGN.get(i, 0.0)),
                )

                # If only domestic source, just balance Q between EQ63 and EQ84.
                if has_dd and not has_im:
                    q63 = _q_from63(i, dd_i, 0.0)
                    vars.IM[i] = 0.0
                    vars.Q[i] = (w63 * q63 + w84 * q84) / (w63 + w84)
                    continue

                # If only imports source, optimize IM around current value.
                if has_im and not has_dd:
                    im_cur = max(0.0, float(vars.IM.get(i, 0.0)))
                    lo = max(1e-8, 0.25 * max(1.0, im_cur))
                    hi = 1.75 * max(1.0, im_cur)
                    best_im = im_cur
                    best_obj = float("inf")
                    for k in range(41):
                        im_try = lo + (hi - lo) * (k / 40.0)
                        q63 = _q_from63(i, 0.0, im_try)
                        q_try = (w63 * q63 + w84 * q84) / (w63 + w84)
                        obj = w63 * (q_try - q63) ** 2 + w84 * (q_try - q84) ** 2
                        if obj < best_obj:
                            best_obj = obj
                            best_im = im_try
                    q63 = _q_from63(i, 0.0, best_im)
                    vars.IM[i] = best_im
                    vars.Q[i] = (w63 * q63 + w84 * q84) / (w63 + w84)
                    continue

                # No domestic/import market for this commodity.
                if not has_dd and not has_im:
                    vars.IM[i] = 0.0
                    vars.Q[i] = q84
                    continue

                # Main case: DD fixed from EQ88, optimize IM to balance EQ63+EQ84,
                # with soft anchor to EQ64 reference.
                im64_ref = _im_from64(i, dd_i)
                im_cur = max(0.0, float(vars.IM.get(i, im64_ref)))
                lo = max(1e-8, 0.25 * max(1.0, im64_ref, im_cur))
                hi = 1.75 * max(1.0, im64_ref, im_cur)
                best_im = im_cur
                best_obj = float("inf")
                for k in range(61):
                    im_try = lo + (hi - lo) * (k / 60.0)
                    q63 = _q_from63(i, dd_i, im_try)
                    q_try = (w63 * q63 + w84 * q84) / (w63 + w84)
                    eq64_pen = (im_try - im64_ref) / (abs(im64_ref) + 1.0)
                    obj = (
                        w63 * (q_try - q63) ** 2
                        + w84 * (q_try - q84) ** 2
                        + w64 * (eq64_pen ** 2)
                    )
                    if obj < best_obj:
                        best_obj = obj
                        best_im = im_try
                q63 = _q_from63(i, dd_i, best_im)
                vars.IM[i] = best_im
                vars.Q[i] = (w63 * q63 + w84 * q84) / (w63 + w84)

        # Keep EQ79 aligned after DD/IM/Q changes.
        self._reconcile_composite_prices(vars)
        _recompute_mrgn()

        res1 = self.equations.calculate_all_residuals(vars)
        score1 = self._coupled_trade_score(res1)
        rms1, max1 = self._residual_rms_max(res1)
        if score1 <= 0.99 * score0 and rms1 <= rms0 and max1 <= max0:
            logger.info(
                "coupled-trade AB accepted: score %.6e -> %.6e | rms %.6e -> %.6e | max %.6e -> %.6e",
                score0,
                score1,
                rms0,
                rms1,
                max0,
                max1,
            )
            return True

        # Revert if no meaningful improvement.
        vars.__dict__.clear()
        vars.__dict__.update(copy.deepcopy(baseline.__dict__))
        logger.info(
            "coupled-trade AB rejected: score %.6e -> %.6e | rms %.6e -> %.6e | max %.6e -> %.6e (reverted)",
            score0,
            score1,
            rms0,
            rms1,
            max0,
            max1,
        )
        return False

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
            "KDO0": state.production.get("KDO", {}),
            "LDO0": state.production.get("LDO", {}),
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
            "tr0": state.income.get("tr0O", {}),
            "ttdh0": ttdh0_base,
            "ttdh1": inferred_ttdh1,
            "ttdf0": {},
            "ttdf1": inferred_ttdf1,
            "TRO": state.income.get("TRO", {}),
            "PWX": state.trade.get("PWXO", {}),
            "CMIN0": state.les_parameters.get("CMINO", {}),
            "VSTK0": state.consumption.get("VSTKO", {}),
            "G0": state.consumption.get("GO", 0.0),
            "PWM0": state.trade.get("PWMO", {}),
            "CAB0": state.income.get("CABO", 0.0),
            "e0": 1.0,
            "PCO0": state.trade.get("PCO", {}),
            "PVAO0": state.production.get("PVAO", {}),
            "VAO0": state.production.get("VAO", {}),
            "TIPO0": {
                j: state.production.get("ttipO", {}).get(j, 0.0)
                * state.production.get("PPO", {}).get(j, 0.0)
                * state.production.get("XSTO", {}).get(j, 0.0)
                for j in state.sets.get("J", [])
            },
            "kmob": 1.0,
            "PT": state.production.get("PTO", {}),
            "EXDO0": state.trade.get("EXDO", {}),
            "IMO0": state.trade.get("IMO", {}),
            "DDO0": state.trade.get("DDO", {}),
            "DSO0": state.trade.get("DSO", {}),
            "EXO0": state.trade.get("EXO", {}),
            "XSO0": state.trade.get("XSO", {}),
            "XSTO0": state.production.get("XSTO", {}),
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

        # Recompute intermediate-use coefficients with trade-consistent XSO.
        # DIO in calibrated state is already in quantity units (deflated by PCO).
        dio_raw = state.production.get("DIO", {})
        vao = state.production.get("VAO", {})
        xso = state.trade.get("XSO", {})
        I = state.sets.get("I", [])
        J = state.sets.get("J", [])

        dio_q: dict[tuple[str, str], float] = dict(dio_raw)

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
            vars.PT[j] = state.production.get("PTO", {}).get(j, 1.0)
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
        for k in self.sets.get("K", []):
            vars.RK[k] = 1.0

        # Initialize detailed factor-tax payments from policy rates.
        rebuild_tax_detail_from_rates(
            vars=vars,
            sets=self.sets,
            params=self.params,
            include_tip=False,
        )
        
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

        # Keep calibrated *O values as benchmark levels.

        # Initialize trade margin demand from calibrated flows (EQ57 identity)
        for i in self.sets.get("I", []):
            mrgn_i = 0.0
            for ij in self.sets.get("I", []):
                tm = self.params.get("tmrg", {}).get((i, ij), 0.0)
                tm_x = self.params.get("tmrg_X", {}).get((i, ij), 0.0)
                mrgn_i += tm * vars.DD.get(ij, 0.0)
                mrgn_i += tm * vars.IM.get(ij, 0.0)
                mrgn_i += tm_x * vars.EXD.get(ij, 0.0)
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
        vars.GFCF_REAL = vars.GFCF / vars.PIXINV if abs(vars.PIXINV) > 1e-12 else 0.0
        vars.LEON = 0.0
        
        # Exchange rate
        vars.e = state.trade.get("eO", 1.0)

        # Exogenous policy/rate variables (GAMS variables fixed via .fx)
        vars.sh0 = dict(self.params.get("sh0", {}))
        vars.sh1 = dict(self.params.get("sh1", {}))
        vars.tr0 = dict(self.params.get("tr0", {}))
        vars.tr1 = dict(self.params.get("tr1", {}))
        vars.ttdh0 = dict(self.params.get("ttdh0", {}))
        vars.ttdh1 = dict(self.params.get("ttdh1", {}))
        vars.ttdf0 = dict(self.params.get("ttdf0", {}))
        vars.ttdf1 = dict(self.params.get("ttdf1", {}))
        vars.ttic = dict(self.params.get("ttic", {}))
        vars.ttim = dict(self.params.get("ttim", {}))
        vars.ttix = dict(self.params.get("ttix", {}))
        vars.ttip = dict(self.params.get("ttip", {}))
        vars.ttiw = dict(self.params.get("ttiw", {}))
        vars.ttik = dict(self.params.get("ttik", {}))

        build_init_strategy(self.init_mode).apply(self, vars)
        
        return vars

    def _sync_policy_params_from_vars(self, vars: PEPModelVariables) -> None:
        """Keep parameter maps aligned with loaded policy levels."""
        for name in (
            "sh0",
            "sh1",
            "tr0",
            "tr1",
            "ttdh0",
            "ttdh1",
            "ttdf0",
            "ttdf1",
            "ttic",
            "ttim",
            "ttix",
            "ttip",
            "ttiw",
            "ttik",
        ):
            v = getattr(vars, name, None)
            if isinstance(v, dict) and v:
                self.params[name] = dict(v)

    def _sync_lambda_tr_from_levels(self, vars: PEPModelVariables) -> None:
        """Align lambda_TR with loaded levels (avoids rounding drift)."""
        lam_h = dict(self.params.get("lambda_TR_households", {}))
        lam_f = dict(self.params.get("lambda_TR_firms", {}))
        for h in self.sets.get("H", []):
            ydh = vars.YDH.get(h, 0.0)
            if abs(ydh) <= 1e-12:
                continue
            for agng in self.sets.get("AGNG", []):
                lam_h[(agng, h)] = vars.TR.get((agng, h), 0.0) / ydh
        for f in self.sets.get("F", []):
            ydf = vars.YDF.get(f, 0.0)
            if abs(ydf) <= 1e-12:
                continue
            for ag in self.sets.get("AG", []):
                lam_f[(ag, f)] = vars.TR.get((ag, f), 0.0) / ydf
        if lam_h:
            self.params["lambda_TR_households"] = lam_h
        if lam_f:
            self.params["lambda_TR_firms"] = lam_f
        if lam_h or lam_f:
            lam = {}
            lam.update(lam_h)
            lam.update(lam_f)
            self.params["lambda_TR"] = lam

    def _resolve_gams_levels_path(self) -> Path | None:
        """Resolve GAMS Results.gdx used to initialize gams levels."""
        candidates: list[Path] = []
        if self.gams_results_gdx is not None:
            candidates.append(self.gams_results_gdx)
        candidates.extend(
            [
                Path("src/equilibria/templates/reference/pep2/scripts/Results.gdx"),
                Path("src/equilibria/templates/reference/pep/results.gdx"),
            ]
        )
        for c in candidates:
            if c.exists():
                return c
        return None

    def _ensure_strict_gams_baseline_compatibility(self) -> None:
        """Validate gams baseline compatibility once per solver instance."""
        if self._strict_baseline_checked:
            if self.enforce_strict_gams_baseline and self.strict_baseline_report is not None:
                if not self.strict_baseline_report.passed:
                    raise RuntimeError(self.strict_baseline_report.summary())
            return

        self._strict_baseline_checked = True
        if self.init_mode != "gams" or not self.enforce_strict_gams_baseline:
            return

        gdx_path = self._resolve_gams_levels_path()
        if gdx_path is None:
            raise RuntimeError("gams baseline check failed: no Results.gdx found")

        report = evaluate_strict_gams_baseline_compatibility(
            state=self.state,
            results_gdx=gdx_path,
            gams_slice=self.gams_results_slice,
            manifest_path=self.baseline_manifest,
            sam_file=self.sam_file,
            val_par_file=self.val_par_file,
            rel_tol=self.baseline_compatibility_rel_tol,
            gdxdump_bin=self.gdxdump_bin,
            require_manifest=self.require_baseline_manifest,
        )
        self.strict_baseline_report = report
        if report.passed:
            logger.info("gams baseline compatibility passed")
            return

        failures = [c for c in report.checks if not c.passed][:8]
        detail = "; ".join(f"{c.code}: {c.message}" for c in failures)
        raise RuntimeError(f"{report.summary()} | {detail}")

    def _overlay_with_gams_levels(self, vars: PEPModelVariables) -> None:
        """Overlay initial guess with BASE levels from GAMS Results.gdx."""
        gdx_path = self._resolve_gams_levels_path()
        if gdx_path is None:
            logger.warning("gams init requested but no Results.gdx found; using calibrated initial guess")
            return

        try:
            gdx = read_gdx(gdx_path)
            symbols = [s.get("name", "") for s in gdx.get("symbols", [])]
            n_updates = 0
            # Prefer gdxdump for Results.gdx overlays; the lightweight reader can
            # miss records on some val* symbols in this file.
            gdxdump_bin = shutil.which("gdxdump") or "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"
            gdxdump_available = Path(gdxdump_bin).exists()

            value_pattern = re.compile(
                r"((?:'[^']*'(?:\.'[^']*')*))\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)"
            )
            label_pattern = re.compile(r"'([^']*)'")

            for sym in symbols:
                if not sym.startswith("val"):
                    continue
                field = "e" if sym == "vale" else sym[3:]
                if not hasattr(vars, field):
                    continue
                target = getattr(vars, field)

                records: list[tuple[tuple[str, ...], float]] = []
                if gdxdump_available:
                    try:
                        out = subprocess.check_output(
                            [gdxdump_bin, str(gdx_path), f"symb={sym}"],
                            text=True,
                            stderr=subprocess.STDOUT,
                        )
                        for m in value_pattern.finditer(out):
                            labels = tuple(x.lower() for x in label_pattern.findall(m.group(1)))
                            value = float(m.group(2))
                            records.append((labels, value))
                    except Exception:
                        records = []

                # Fallback to in-process reader if gdxdump is unavailable or empty.
                if not records:
                    try:
                        values = read_parameter_values(gdx, sym)
                    except Exception:
                        continue
                    for raw_key, raw_val in values.items():
                        key_t = raw_key if isinstance(raw_key, tuple) else ((raw_key,) if raw_key != () else ())
                        labels = tuple(str(k).lower() for k in key_t)
                        records.append((labels, float(raw_val)))

                for labels, v in records:
                    idx = labels

                    # Keep BASE slice for scenario-indexed symbols.
                    if idx and idx[-1] in {"base", "sim1", "var"}:
                        if idx[-1] != self.gams_results_slice:
                            continue
                        idx = idx[:-1]

                    if isinstance(target, dict):
                        if len(idx) == 1:
                            target[idx[0]] = v
                            n_updates += 1
                        elif len(idx) >= 2:
                            target[(idx[0], idx[1])] = v
                            n_updates += 1
                    else:
                        if len(idx) == 0:
                            setattr(vars, field, v)
                            n_updates += 1

            logger.info("Applied %d gams level updates from %s", n_updates, gdx_path)
        except Exception as e:
            logger.warning("Failed loading gams levels from Results.gdx: %s", e)

    def _overlay_with_calibrated_levels(self, vars: PEPModelVariables) -> None:
        """
        Overlay initial guess with calibrated benchmark levels (*O maps).

        This mirrors GAMS initialization semantics:
          variable.l = variable0
        where `variable0` comes from the SAM/VAL_PAR calibration pipeline.
        """
        state = self.state
        production = state.production
        trade = state.trade
        income = state.income
        consumption = state.consumption
        les = state.les_parameters
        gdp = state.gdp
        real = state.real_variables

        tro = income.get("TRO", {})
        for ag in self.sets.get("AG", []):
            for agj in self.sets.get("AG", []):
                vars.TR[(ag, agj)] = float(tro.get((ag, agj), vars.TR.get((ag, agj), 0.0)))

        for j in self.sets.get("J", []):
            vars.VA[j] = float(production.get("VAO", {}).get(j, vars.VA.get(j, 0.0)))
            vars.CI[j] = float(production.get("CIO", {}).get(j, vars.CI.get(j, 0.0)))
            vars.LDC[j] = float(production.get("LDCO", {}).get(j, vars.LDC.get(j, 0.0)))
            vars.KDC[j] = float(production.get("KDCO", {}).get(j, vars.KDC.get(j, 0.0)))
            vars.XST[j] = float(production.get("XSTO", {}).get(j, vars.XST.get(j, 0.0)))
            vars.PVA[j] = float(production.get("PVAO", {}).get(j, vars.PVA.get(j, 1.0)))
            vars.PP[j] = float(production.get("PPO", {}).get(j, vars.PP.get(j, 1.0)))
            vars.PT[j] = float(production.get("PTO", {}).get(j, vars.PT.get(j, 1.0)))
            vars.PCI[j] = float(production.get("PCIO", {}).get(j, vars.PCI.get(j, 1.0)))
            vars.WC[j] = float(production.get("WCO", {}).get(j, vars.WC.get(j, 1.0)))
            vars.RC[j] = float(production.get("RCO", {}).get(j, vars.RC.get(j, 1.0)))

            for l in self.sets.get("L", []):
                key = (l, j)
                vars.LD[key] = float(production.get("LDO", {}).get(key, vars.LD.get(key, 0.0)))
                vars.WTI[key] = float(production.get("WTIO", {}).get(key, vars.WTI.get(key, 1.0)))

            for k in self.sets.get("K", []):
                key = (k, j)
                vars.KD[key] = float(production.get("KDO", {}).get(key, vars.KD.get(key, 0.0)))
                vars.RTI[key] = float(production.get("RTIO", {}).get(key, vars.RTI.get(key, 1.0)))

            for i in self.sets.get("I", []):
                key_ij = (i, j)
                key_ji = (j, i)
                vars.DI[key_ij] = float(production.get("DIO", {}).get(key_ij, vars.DI.get(key_ij, 0.0)))
                vars.XS[key_ji] = float(trade.get("XSO", {}).get(key_ji, vars.XS.get(key_ji, 0.0)))
                vars.DS[key_ji] = float(trade.get("DSO", {}).get(key_ji, vars.DS.get(key_ji, 0.0)))
                vars.EX[key_ji] = float(trade.get("EXO", {}).get(key_ji, vars.EX.get(key_ji, 0.0)))
                vars.P[key_ji] = float(trade.get("PO", {}).get(key_ji, vars.P.get(key_ji, 1.0)))

        # No explicit WO/RO/RKO in calibrated state; keep initialized numeraire defaults.
        for l in self.sets.get("L", []):
            vars.W[l] = float(vars.W.get(l, 1.0))
        for k in self.sets.get("K", []):
            vars.RK[k] = float(vars.RK.get(k, 1.0))
            for j in self.sets.get("J", []):
                key = (k, j)
                vars.R[key] = float(vars.R.get(key, 1.0))

        for i in self.sets.get("I", []):
            vars.IM[i] = float(trade.get("IMO", {}).get(i, vars.IM.get(i, 0.0)))
            vars.DD[i] = float(trade.get("DDO", {}).get(i, vars.DD.get(i, 0.0)))
            vars.Q[i] = float(trade.get("QO", {}).get(i, vars.Q.get(i, 0.0)))
            vars.EXD[i] = float(trade.get("EXDO", {}).get(i, vars.EXD.get(i, 0.0)))
            vars.PC[i] = float(trade.get("PCO", {}).get(i, vars.PC.get(i, 1.0)))
            vars.PD[i] = float(trade.get("PDO", {}).get(i, vars.PD.get(i, 1.0)))
            vars.PM[i] = float(trade.get("PMO", {}).get(i, vars.PM.get(i, 1.0)))
            vars.PE[i] = float(trade.get("PEO", {}).get(i, vars.PE.get(i, 1.0)))
            vars.PE_FOB[i] = float(trade.get("PE_FOBO", {}).get(i, vars.PE_FOB.get(i, 1.0)))
            vars.PWM[i] = float(trade.get("PWMO", {}).get(i, vars.PWM.get(i, 1.0)))
            vars.PL[i] = float(trade.get("PLO", {}).get(i, vars.PL.get(i, 1.0)))
            vars.TIC[i] = float(trade.get("TICO", {}).get(i, vars.TIC.get(i, 0.0)))
            vars.TIM[i] = float(trade.get("TIMO", {}).get(i, vars.TIM.get(i, 0.0)))
            vars.TIX[i] = float(trade.get("TIXO", {}).get(i, vars.TIX.get(i, 0.0)))
            vars.MRGN[i] = float(trade.get("MRGNO", {}).get(i, vars.MRGN.get(i, 0.0)))
            vars.DIT[i] = float(production.get("DITO", {}).get(i, vars.DIT.get(i, 0.0)))
            vars.INV[i] = float(consumption.get("INVO", {}).get(i, vars.INV.get(i, 0.0)))
            vars.CG[i] = float(consumption.get("CGO", {}).get(i, vars.CG.get(i, 0.0)))
            vars.VSTK[i] = float(consumption.get("VSTKO", {}).get(i, vars.VSTK.get(i, 0.0)))

        for h in self.sets.get("H", []):
            vars.YH[h] = float(income.get("YHO", {}).get(h, vars.YH.get(h, 0.0)))
            vars.YHL[h] = float(income.get("YHLO", {}).get(h, vars.YHL.get(h, 0.0)))
            vars.YHK[h] = float(income.get("YHKO", {}).get(h, vars.YHK.get(h, 0.0)))
            vars.YHTR[h] = float(income.get("YHTRO", {}).get(h, vars.YHTR.get(h, 0.0)))
            vars.YDH[h] = float(income.get("YDHO", {}).get(h, vars.YDH.get(h, 0.0)))
            vars.CTH[h] = float(income.get("CTHO", {}).get(h, vars.CTH.get(h, 0.0)))
            vars.SH[h] = float(
                income.get("SHO", {}).get(
                    h,
                    vars.sh0.get(h, 0.0) + vars.sh1.get(h, 0.0) * vars.YDH.get(h, 0.0),
                )
            )
            vars.TDH[h] = float(income.get("TDHO", {}).get(h, vars.YH.get(h, 0.0) - vars.YDH.get(h, 0.0) - vars.TR.get(("gvt", h), 0.0)))
            for i in self.sets.get("I", []):
                key = (i, h)
                vars.C[key] = float(consumption.get("CO", {}).get(key, vars.C.get(key, 0.0)))
                vars.CMIN[key] = float(les.get("CMINO", {}).get(key, vars.CMIN.get(key, 0.0)))

        for f in self.sets.get("F", []):
            vars.YF[f] = float(income.get("YFO", {}).get(f, vars.YF.get(f, 0.0)))
            vars.YFK[f] = float(income.get("YFKO", {}).get(f, vars.YFK.get(f, 0.0)))
            vars.YFTR[f] = float(income.get("YFTRO", {}).get(f, vars.YFTR.get(f, 0.0)))
            vars.YDF[f] = float(income.get("YDFO", {}).get(f, vars.YDF.get(f, 0.0)))
            vars.SF[f] = float(income.get("SFO", {}).get(f, vars.SF.get(f, vars.YDF.get(f, 0.0))))
            vars.TDF[f] = float(income.get("TDFO", {}).get(f, vars.YF.get(f, 0.0) - vars.YDF.get(f, 0.0)))

        vars.YG = float(income.get("YGO", vars.YG))
        vars.YGK = float(income.get("YGKO", vars.YGK))
        vars.TDHT = float(income.get("TDHTO", vars.TDHT))
        vars.TDFT = float(income.get("TDFTO", vars.TDFT))
        vars.TIWT = float(income.get("TIWTO", vars.TIWT))
        vars.TIKT = float(income.get("TIKTO", vars.TIKT))
        vars.TIPT = float(income.get("TIPTO", vars.TIPT))
        vars.TICT = float(income.get("TICTO", vars.TICT))
        vars.TIMT = float(income.get("TIMTO", vars.TIMT))
        vars.TIXT = float(income.get("TIXTO", vars.TIXT))
        vars.TPRODN = float(income.get("TPRODNO", vars.TPRODN))
        vars.TPRCTS = float(income.get("TPRCTSO", vars.TPRCTS))
        vars.YGTR = float(income.get("YGTRO", vars.YGTR))
        vars.G = float(consumption.get("GO", vars.G))
        vars.SG = float(income.get("SGO", vars.SG))

        vars.YROW = float(income.get("YROWO", vars.YROW))
        vars.CAB = float(income.get("CABO", vars.CAB))
        vars.SROW = float(income.get("SROWO", -vars.CAB))

        vars.IT = float(income.get("ITO", vars.IT))
        vars.GFCF = float(consumption.get("GFCFO", vars.GFCF))

        vars.GDP_BP = float(gdp.get("GDP_BPO", vars.GDP_BP))
        vars.GDP_MP = float(gdp.get("GDP_MPO", vars.GDP_MP))
        vars.GDP_IB = float(gdp.get("GDP_IBO", vars.GDP_IB))
        vars.GDP_FD = float(gdp.get("GDP_FDO", vars.GDP_FD))

        vars.PIXCON = float(real.get("PIXCONO", vars.PIXCON))
        vars.PIXGDP = float(real.get("PIXGDPO", vars.PIXGDP))
        vars.PIXGVT = float(real.get("PIXGVTO", vars.PIXGVT))
        vars.PIXINV = float(real.get("PIXINVO", vars.PIXINV))
        for h in self.sets.get("H", []):
            vars.CTH_REAL[h] = float(real.get("CTH_REALO", {}).get(h, vars.CTH_REAL.get(h, 0.0)))
        vars.G_REAL = float(real.get("G_REALO", vars.G_REAL))
        vars.GDP_BP_REAL = float(real.get("GDP_BP_REALO", vars.GDP_BP_REAL))
        vars.GDP_MP_REAL = float(real.get("GDP_MP_REALO", vars.GDP_MP_REAL))
        vars.GFCF_REAL = float(real.get("GFCF_REALO", vars.GFCF_REAL))

        vars.e = float(trade.get("eO", vars.e))
        vars.LEON = 0.0

        # Recover detailed factor/production tax payments where no explicit *O
        # detail is stored in state.
        for l in self.sets.get("L", []):
            for j in self.sets.get("J", []):
                ttiw = self.params.get("ttiw", {}).get((l, j), 0.0)
                vars.TIW[(l, j)] = ttiw * vars.W.get(l, 1.0) * vars.LD.get((l, j), 0.0)
        for k in self.sets.get("K", []):
            for j in self.sets.get("J", []):
                ttik = self.params.get("ttik", {}).get((k, j), 0.0)
                vars.TIK[(k, j)] = ttik * vars.R.get((k, j), 1.0) * vars.KD.get((k, j), 0.0)
        for j in self.sets.get("J", []):
            ttip = self.params.get("ttip", {}).get(j, 0.0)
            vars.TIP[j] = ttip * vars.PP.get(j, 0.0) * vars.XST.get(j, 0.0)

    def _apply_equation_consistent_adjustments(self, vars: PEPModelVariables) -> None:
        """Apply adjustments so initialized values satisfy benchmark equations."""
        # Align XST with trade transformation baseline (SUM_i XS(j,i)).
        # This removes the production-vs-trade benchmark mismatch present in
        # some calibrated states (notably ind/agr).
        for j in self.sets.get("J", []):
            vars.XST[j] = sum(vars.XS.get((j, i), 0.0) for i in self.sets.get("I", []))

        # Production identities: keep XST benchmark, enforce EQ1/EQ2/EQ9/EQ65/EQ67.
        for j in self.sets.get("J", []):
            xst_j = vars.XST.get(j, 0.0)
            io_j = self.params.get("io", {}).get(j, 0.0)
            v_j = self.params.get("v", {}).get(j, 0.0)
            vars.CI[j] = io_j * xst_j
            vars.VA[j] = v_j * xst_j

            for i in self.sets.get("I", []):
                aij_ij = self.params.get("aij", {}).get((i, j), 0.0)
                vars.DI[(i, j)] = aij_ij * vars.CI.get(j, 0.0)

            ci_j = vars.CI.get(j, 0.0)
            if abs(ci_j) > 1e-12:
                vars.PCI[j] = (
                    sum(vars.PC.get(i, 1.0) * vars.DI.get((i, j), 0.0) for i in self.sets.get("I", []))
                    / ci_j
                )
            if abs(xst_j) > 1e-12:
                vars.PP[j] = (
                    vars.PVA.get(j, 0.0) * vars.VA.get(j, 0.0)
                    + vars.PCI.get(j, 0.0) * vars.CI.get(j, 0.0)
                ) / xst_j
            vars.PT[j] = (1.0 + self.params.get("ttip", {}).get(j, 0.0)) * vars.PP.get(j, 0.0)

        for i in self.sets.get("I", []):
            vars.DIT[i] = sum(vars.DI.get((i, j), 0.0) for j in self.sets.get("J", []))

        # Demand-side absorption identity EQ84(i1).
        for i in self.sets.get("I1", []):
            cons_i = sum(vars.C.get((i, h), 0.0) for h in self.sets.get("H", []))
            vars.Q[i] = (
                cons_i
                + vars.CG.get(i, 0.0)
                + vars.INV.get(i, 0.0)
                + vars.VSTK.get(i, 0.0)
                + vars.DIT.get(i, 0.0)
                + vars.MRGN.get(i, 0.0)
            )

        rebuild_tax_detail_from_rates(
            vars=vars,
            sets=self.sets,
            params=self.params,
            include_tip=False,
        )

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
        vars.GFCF_REAL = vars.GFCF / vars.PIXINV if abs(vars.PIXINV) > 1e-12 else 0.0

        gdp_fd = 0.0
        for i in self.sets.get("I", []):
            cons_i = sum(vars.C.get((i, h), 0.0) for h in self.sets.get("H", []))
            gdp_fd += vars.PC.get(i, 0.0) * (
                cons_i + vars.CG.get(i, 0.0) + vars.INV.get(i, 0.0) + vars.VSTK.get(i, 0.0)
            )
            gdp_fd += vars.PE_FOB.get(i, 0.0) * vars.EXD.get(i, 0.0)
            gdp_fd -= vars.PWM.get(i, 0.0) * vars.e * vars.IM.get(i, 0.0)
        vars.GDP_FD = gdp_fd
        walras_i = "agr" if "agr" in self.sets.get("I", []) else (self.sets.get("I", [None])[0])
        if walras_i is not None:
            vars.LEON = (
                vars.Q.get(walras_i, 0.0)
                - sum(vars.C.get((walras_i, h), 0.0) for h in self.sets.get("H", []))
                - vars.CG.get(walras_i, 0.0)
                - vars.INV.get(walras_i, 0.0)
                - vars.VSTK.get(walras_i, 0.0)
                - vars.DIT.get(walras_i, 0.0)
                - vars.MRGN.get(walras_i, 0.0)
            )
    
    def _variables_to_array(self, vars: PEPModelVariables) -> np.ndarray:
        """Convert variables to flat array for solver."""
        return pep_variables_to_array(vars, self.sets)

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
            add(*POS)     # PT
            add(*POS)     # PVA
            add(*POS)     # PCI
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
        for _k in self.sets.get("K", []):
            add(*POS)  # RK

        # Price and trade variables
        for _i in self.sets.get("I", []):
            add(*POS)     # PC
            add(*POS)     # PD
            add(*POS)     # PM
            add(*POS)     # PE
            add(*POS)     # PE_FOB
            add(*POS)     # PL
            pwm_fix = float(self.state.trade.get("PWMO", {}).get(_i, 1.0))
            add(pwm_fix, pwm_fix)  # PWM fixed as in GAMS (.fx)
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
            vstk_fix = float(self.state.consumption.get("VSTKO", {}).get(_i, 0.0))
            add(vstk_fix, vstk_fix)  # VSTK fixed as in GAMS (.fx)

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
                cmin_fix = float(self.state.les_parameters.get("CMINO", {}).get((_i, _h), 0.0))
                add(cmin_fix, cmin_fix)  # CMIN fixed as in GAMS (.fx)


        for _f in self.sets.get("F", []):
            add(*NONNEG)  # YF
            add(*NONNEG)  # YFK
            add(*FREE)    # YFTR
            add(*NONNEG)  # YDF
            add(*FREE)    # SF
            add(*FREE)    # TDF

        # Full transfer matrix TR(ag,agj)
        tro_bench = self.params.get("TRO", {})
        for _ag in self.sets.get("AG", []):
            for _agj in self.sets.get("AG", []):
                # These two entries are not pinned by equations in GAMS and are
                # effectively dropped from CNS; fix them at benchmark levels.
                if (_ag, _agj) in {("gvt", "gvt"), ("row", "row")}:
                    tr_fix = float(tro_bench.get((_ag, _agj), 0.0))
                    add(tr_fix, tr_fix)
                else:
                    add(*FREE)    # TR(ag,agj)

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
        g_fix = float(self.state.consumption.get("GO", 0.0))
        add(g_fix, g_fix)  # G fixed as in GAMS (.fx)
        add(*FREE)  # SG

        # ROW
        add(*FREE)  # YROW
        add(*FREE)  # SROW
        cab_fix = float(self.state.income.get("CABO", 0.0))
        add(cab_fix, cab_fix)  # CAB fixed as in GAMS (.fx)

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
        add(*NONNEG)  # GFCF_REAL
        add(*FREE)    # LEON

        # Exchange rate
        e_fix = float(self.state.trade.get("eO", 1.0))
        add(e_fix, e_fix)  # e fixed as in GAMS (.fx)

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

        # For non-excel init modes, allow early return when initialization is
        # already feasible enough. In excel mode, we always execute solve.
        init_residuals = self.equations.calculate_all_residuals(vars)
        init_vals = np.array(list(init_residuals.values()), dtype=float)
        init_rms = float(np.sqrt(np.mean(init_vals ** 2))) if init_vals.size else 0.0
        init_max = float(np.max(np.abs(init_vals))) if init_vals.size else 0.0
        practical_tol = max(self.tolerance * 1e4, 1e-4)
        practical_max_tol = max(self.tolerance * 1e5, 1e-3)
        allow_init_shortcut = not self._uses_excel_init()
        if allow_init_shortcut and init_rms <= self.tolerance:
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
                message=f"Initial {self.init_mode} benchmark satisfies tolerance",
            )
        if allow_init_shortcut and init_rms <= practical_tol and init_max <= practical_max_tol:
            logger.info(
                "Initial guess accepted by practical tolerance (RMS=%.3e, Max=%.3e); skipping IPOPT.",
                init_rms,
                init_max,
            )
            return SolverResult(
                converged=True,
                iterations=0,
                final_residual=init_rms,
                variables=vars,
                residuals=init_residuals,
                message=f"Initial {self.init_mode} benchmark accepted by practical tolerance",
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

            v1 = problem._array_to_variables(x1)
            r1 = self.equations.calculate_all_residuals(v1)
            vals1 = np.array(list(r1.values()), dtype=float)
            rms1 = float(np.sqrt(np.mean(vals1**2))) if vals1.size else float("inf")
            # Track best candidate by convergence first, then RMS residual.
            if self._uses_excel_init():
                x_best, info_best, rms_best = x1, info1, rms1
            else:
                x_best, info_best, rms_best = x0, {
                    "status": -99,
                    "status_msg": f"Using initial {self.init_mode} levels (lower RMS than IPOPT passes)",
                    "iter_count": 0,
                    "obj_val": problem.objective(x0),
                }, init_rms
                if info1.get("status") == 0 or rms1 < rms_best:
                    x_best, info_best, rms_best = x1, info1, rms1

            # Optional second pass with warm start if first pass did not converge.
            if info1.get("status") != 0:
                logger.info("Starting IPOPT optimization (pass 2, warm start)...")
                nlp2 = _build_nlp(max_iter=self.max_iterations, warm_start=True)
                x2, info2 = nlp2.solve(x1)
                msg2 = info2.get("status_msg", "Unknown")
                if isinstance(msg2, bytes):
                    msg2 = msg2.decode("utf-8", errors="replace")
                logger.info("IPOPT pass 2 status=%s msg=%s", info2.get("status"), msg2)

                # Select candidate by convergence first, then lower residual RMS.
                v2 = problem._array_to_variables(x2)
                r2 = self.equations.calculate_all_residuals(v2)
                vals2 = np.array(list(r2.values()), dtype=float)
                rms2 = float(np.sqrt(np.mean(vals2**2))) if vals2.size else float("inf")

                if info2.get("status") == 0 and info_best.get("status") != 0:
                    x_best, info_best, rms_best = x2, info2, rms2
                elif info_best.get("status") != 0 and rms2 < rms_best:
                    x_best, info_best, rms_best = x2, info2, rms2
                elif info2.get("status") == 0 and info_best.get("status") == 0 and rms2 < rms_best:
                    x_best, info_best, rms_best = x2, info2, rms2

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
