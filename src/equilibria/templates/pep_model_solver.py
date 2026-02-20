"""
PEP Model Solver

This module provides a solver for the calibrated PEP model using
scipy.optimize.root or similar methods to solve the system of nonlinear equations.

Usage:
    from equilibria.templates.pep_model_solver import PEPModelSolver
    
    solver = PEPModelSolver(calibrated_state)
    solution = solver.solve()
    
    if solution.converged:
        print(f"Solution found: GDP = {solution.variables.GDP_BP}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from equilibria.solver.guards import rebuild_tax_detail_from_rates
from equilibria.solver.transforms import pep_array_to_variables, pep_variables_to_array
from equilibria.templates.init_strategies import normalize_init_mode
from equilibria.templates.pep_model_equations import PEPModelEquations, PEPModelVariables, SolverResult

logger = logging.getLogger(__name__)

# Import IPOPT solver if available
try:
    from equilibria.templates.pep_model_solver_ipopt import IPOPTSolver
    from equilibria.templates.pep_model_solver_ipopt import IPOPT_AVAILABLE
    logger.info(f"IPOPT import successful, IPOPT_AVAILABLE={IPOPT_AVAILABLE}")
except ImportError as e:
    IPOPT_AVAILABLE = False
    logger.warning(f"IPOPT import failed: {e}")


class PEPModelSolver:
    """Solver for the PEP CGE model."""
    
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
            init_mode: Initialization mode:
                - gams: initialize from GAMS levels
                - excel: initialize from SAM/Excel calibrated *O values
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
        
        # Extract sets and parameters from calibrated state
        self.sets = calibrated_state.sets
        self.params = self._extract_parameters(calibrated_state)
        
        # Initialize equations
        self.equations = PEPModelEquations(self.sets, self.params)
        
        logger.info(f"Initialized PEP Model Solver")
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
    
    def _extract_parameters(self, state: Any) -> dict[str, Any]:
        """Extract all calibrated parameters from model state."""
        params = {}
        
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
            "rho_KD": state.production.get("rho_KD", {}),
            "beta_KD": state.production.get("beta_KD", {}),
            "B_KD": state.production.get("B_KD", {}),
            "rho_LD": state.production.get("rho_LD", {}),
            "beta_LD": state.production.get("beta_LD", {}),
            "B_LD": state.production.get("B_LD", {}),
            "ttiw": state.production.get("ttiwO", {}),
            "ttik": state.production.get("ttikO", {}),
            "ttip": state.production.get("ttipO", {}),
            "LS": state.production.get("LSO", {}),
            "KS": state.production.get("KSO", {}),
            "KDO0": state.production.get("KDO", {}),
            "LDO0": state.production.get("LDO", {}),
            # Value added CES parameters
            "rho_VA": state.production.get("rho_VA", {}),
            "beta_VA": state.production.get("beta_VA", {}),
            "B_VA": state.production.get("B_VA", {}),
        })
        
        # Trade parameters
        params.update({
            "rho_XT": state.trade.get("rho_XT", {}),
            "beta_XT": state.trade.get("beta_XT", {}),
            "B_XT": state.trade.get("B_XT", {}),
            "rho_X": state.trade.get("rho_X", {}),
            "beta_X": state.trade.get("beta_X", {}),
            "B_X": state.trade.get("B_X", {}),
            "rho_M": state.trade.get("rho_M", {}),
            "beta_M": state.trade.get("beta_M", {}),
            "B_M": state.trade.get("B_M", {}),
            "sigma_XD": {i: 2.0 for i in state.sets.get("I", [])},  # Default
            "ttic": state.trade.get("tticO", {}),
            "ttim": state.trade.get("ttimO", {}),
            "ttix": state.trade.get("ttixO", {}),
            "tmrg": state.trade.get("tmrg", {}),
            "tmrg_X": state.trade.get("tmrg_X", {}),
            "EXDO": state.trade.get("EXDO", {}),
            "PWX": state.trade.get("PWXO", {}),
        })
        
        # LES parameters
        params.update({
            "gamma_LES": state.les_parameters.get("gamma_LES", {}),
            "sigma_Y": state.les_parameters.get("sigma_Y", {}),
            "frisch": state.les_parameters.get("frisch", {}),
        })
        
        # Additional parameters
        inferred_sh0 = {}
        sh1_base = state.income.get("sh1O", {})
        ydh_base = state.income.get("YDHO", {})
        cth_base = state.income.get("CTHO", {})
        lambda_tr_h_base = state.income.get("lambda_TR_households", {})
        for h in state.sets.get("H", []):
            tr_agng_h = sum(
                lambda_tr_h_base.get((agng, h), 0.0) * ydh_base.get(h, 0.0)
                for agng in state.sets.get("AGNG", [])
            )
            sh_base = ydh_base.get(h, 0) - cth_base.get(h, 0) - tr_agng_h
            inferred_sh0[h] = sh_base - sh1_base.get(h, 0) * ydh_base.get(h, 0)

        # In PEP benchmark data, transfer intercept is typically zero and TR is driven
        # by tr1*YH at benchmark prices.
        inferred_tr0 = {h: 0.0 for h in state.sets.get("H", [])}
        tr0_base = state.income.get("tr0O", {})
        tr1_base = state.income.get("tr1O", {})
        yh_base = state.income.get("YHO", {})

        inferred_ttdh1 = {}
        ttdh0_base = state.income.get("ttdh0O", {})
        tro_base = state.income.get("TRO", {})
        for h in state.sets.get("H", []):
            yh_h = yh_base.get(h, 0)
            tr_h_to_gvt = tro_base.get(("gvt", h), 0.0)
            tdh_h = max(yh_base.get(h, 0) - ydh_base.get(h, 0) - tr_h_to_gvt, 0)
            ttdh0_h = ttdh0_base.get(h, 0.0)
            inferred_ttdh1[h] = ((tdh_h - ttdh0_h) / yh_h) if abs(yh_h) > 1e-12 else 0.0

        inferred_ttdf1 = {}
        yfo_base = state.income.get("YFO", {})
        ydfo_base = state.income.get("YDFO", {})
        yfko_base = state.income.get("YFKO", {})
        for f in state.sets.get("F", []):
            yfk_f = yfko_base.get(f, 0)
            tdf_f = max(yfo_base.get(f, 0) - ydfo_base.get(f, 0), 0)
            inferred_ttdf1[f] = (tdf_f / yfk_f) if abs(yfk_f) > 1e-12 else 0.0

        params.update({
            "eta": 1,  # Matches GAMS scalar eta
            "sh0": (
                state.income.get("sh0O", {})
                if isinstance(state.income.get("sh0O", {}), dict) and state.income.get("sh0O", {})
                else inferred_sh0
            ),
            "tr0": (
                tr0_base
                if isinstance(tr0_base, dict) and tr0_base
                else inferred_tr0
            ),
            "ttdh0": ttdh0_base,
            "ttdh1": inferred_ttdh1,
            "ttdf0": {},
            "ttdf1": inferred_ttdf1,
            "TRO": state.income.get("TRO", {}),
        })

        # Derive sigma parameters after rho parameters have been loaded
        def _safe_sigma_plus_one(rho_val: float, fallback: float = 1.0) -> float:
            denom = 1 + rho_val
            if abs(denom) < 1e-12:
                return fallback
            return 1 / denom

        rho_kd = params.get("rho_KD", {})
        params["sigma_KD"] = {j: _safe_sigma_plus_one(rho_kd.get(j, 0), 1.0) for j in rho_kd}

        rho_ld = params.get("rho_LD", {})
        params["sigma_LD"] = {j: _safe_sigma_plus_one(rho_ld.get(j, 0), 1.0) for j in rho_ld}

        rho_va = params.get("rho_VA", {})
        params["sigma_VA"] = {
            j: (
                _safe_sigma_plus_one(rho_va.get(j, -0.5), 1.5)
                if rho_va.get(j, -0.5) != -0.5
                else 1.5
            )
            for j in rho_va
        }

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

        # Recompute LES budget shares from calibrated base values for consistency.
        c_base = state.consumption.get("CO", {})
        cmin_base = state.les_parameters.get("CMINO", {})
        cth_base = state.income.get("CTHO", {})
        pc_base = state.trade.get("PCO", {})
        gamma_base = {}
        for h in state.sets.get("H", []):
            cmin_val = sum(pc_base.get(i, 1.0) * cmin_base.get((i, h), 0) for i in state.sets.get("I", []))
            denom = cth_base.get(h, 0) - cmin_val
            if abs(denom) < 1e-12:
                continue
            for i in state.sets.get("I", []):
                numer = pc_base.get(i, 1.0) * (c_base.get((i, h), 0) - cmin_base.get((i, h), 0))
                gamma_base[(i, h)] = numer / denom
        if gamma_base:
            params["gamma_LES"] = gamma_base

        params["EXDO"] = state.trade.get("EXDO", {})
        params["EXDO0"] = state.trade.get("EXDO", {})
        params["IMO0"] = state.trade.get("IMO", {})
        params["DDO0"] = state.trade.get("DDO", {})
        params["DSO0"] = state.trade.get("DSO", {})
        params["EXO0"] = state.trade.get("EXO", {})
        params["XSO0"] = state.trade.get("XSO", {})
        params["XSTO0"] = state.production.get("XSTO", {})
        params["PWX"] = state.trade.get("PWXO", {})
        params["CMIN0"] = state.les_parameters.get("CMINO", {})
        params["VSTK0"] = state.consumption.get("VSTKO", {})
        params["G0"] = state.consumption.get("GO", 0.0)
        params["PWM0"] = state.trade.get("PWMO", {})
        params["CAB0"] = state.income.get("CABO", 0.0)
        params["e0"] = 1.0
        params["PCO0"] = state.trade.get("PCO", {})
        params["PVAO0"] = state.production.get("PVAO", {})
        params["VAO0"] = state.production.get("VAO", {})
        params["TIPO0"] = {
            j: state.production.get("ttipO", {}).get(j, 0.0)
            * state.production.get("PPO", {}).get(j, 0.0)
            * state.production.get("XSTO", {}).get(j, 0.0)
            for j in state.sets.get("J", [])
        }
        params["kmob"] = 1.0
        params["PT"] = state.production.get("PTO", {})

        inv_base = state.consumption.get("INVO", {})
        pc_base2 = state.trade.get("PCO", {})
        inv_nom = {i: inv_base.get(i, 0.0) * pc_base2.get(i, 1.0) for i in state.sets.get("I", [])}
        inv_total = sum(inv_nom.values())
        params["gamma_INV"] = {
            i: (inv_nom.get(i, 0.0) / inv_total if abs(inv_total) > 1e-12 else 0.0)
            for i in state.sets.get("I", [])
        }

        cg_base = state.consumption.get("CGO", {})
        cg_nom = {i: cg_base.get(i, 0.0) * pc_base2.get(i, 1.0) for i in state.sets.get("I", [])}
        cg_total = sum(cg_nom.values())
        params["gamma_GVT"] = {
            i: (cg_nom.get(i, 0.0) / cg_total if abs(cg_total) > 1e-12 else 0.0)
            for i in state.sets.get("I", [])
        }

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
        params["io"] = {
            j: (ci_q[j] / xst_q[j] if abs(xst_q[j]) > 1e-12 else 0.0)
            for j in J
        }
        params["v"] = {
            j: (vao.get(j, 0.0) / xst_q[j] if abs(xst_q[j]) > 1e-12 else 0.0)
            for j in J
        }
        
        return params
    
    def _create_initial_guess(self) -> PEPModelVariables:
        """Create initial guess for variables from calibrated values."""
        # Keep initialization logic centralized in IPOPTSolver to avoid drift
        # between solver frontends.
        try:
            ipopt_solver = IPOPTSolver(
                calibrated_state=self.state,
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
                init_mode=self.init_mode,
                blockwise_commodity_alpha=self.blockwise_commodity_alpha,
                blockwise_trade_market_alpha=self.blockwise_trade_market_alpha,
                blockwise_macro_alpha=self.blockwise_macro_alpha,
                gams_results_gdx=self.gams_results_gdx,
                gams_results_slice=self.gams_results_slice,
                baseline_manifest=self.baseline_manifest,
                require_baseline_manifest=self.require_baseline_manifest,
                baseline_compatibility_rel_tol=self.baseline_compatibility_rel_tol,
                enforce_strict_gams_baseline=self.enforce_strict_gams_baseline,
                sam_file=self.sam_file,
                val_par_file=self.val_par_file,
                gdxdump_bin=self.gdxdump_bin,
            )
            vars_ipopt = ipopt_solver._create_initial_guess()
            self.params = ipopt_solver.params
            self.equations = ipopt_solver.equations
            return vars_ipopt
        except Exception as e:
            if self.init_mode == "gams" and self.enforce_strict_gams_baseline:
                raise RuntimeError(f"gams initial-guess creation failed: {e}") from e
            logger.warning("Falling back to local initial-guess path: %s", e)

        vars = PEPModelVariables()
        state = self.state
        
        # Initialize from calibrated state (using "O" suffix values as starting point)
        
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
        tro = self.params.get("TRO", {})
        for ag in self.sets.get("AG", []):
            for agj in self.sets.get("AG", []):
                vars.TR[(ag, agj)] = tro.get((ag, agj), 0)
        for h in self.sets.get("H", []):
            vars.YH[h] = state.income.get("YHO", {}).get(h, 0)
            vars.YHL[h] = state.income.get("YHLO", {}).get(h, 0)
            vars.YHK[h] = state.income.get("YHKO", {}).get(h, 0)
            vars.YHTR[h] = state.income.get("YHTRO", {}).get(h, 0)
            vars.YDH[h] = state.income.get("YDHO", {}).get(h, 0)
            vars.CTH[h] = state.income.get("CTHO", {}).get(h, 0)
            # SH = YDH - CTH (savings = disposable income - consumption)
            vars.SH[h] = vars.YDH[h] - vars.CTH[h]
            vars.TDH[h] = vars.YH[h] - vars.YDH[h] - vars.TR.get(("gvt", h), 0.0)
            
            for ag in self.sets.get("AG", []):
                vars.TR[(h, ag)] = tro.get((h, ag), vars.TR.get((h, ag), 0))
        
        for f in self.sets.get("F", []):
            vars.YF[f] = state.income.get("YFO", {}).get(f, 0)
            vars.YFK[f] = state.income.get("YFKO", {}).get(f, 0)
            vars.YFTR[f] = state.income.get("YFTRO", {}).get(f, 0)
            vars.YDF[f] = state.income.get("YDFO", {}).get(f, 0)
            # Firm savings = disposable income (no consumption)
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
        vars.TPRODN = state.income.get("TPRODNO", 0)
        vars.YG = state.income.get("YGO", 0)
        vars.G = state.consumption.get("GO", 0)
        # SG will be recalculated after transfer initialization.
        vars.SG = 0

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

        # Initialize production taxes by sector for consistency with TIPT
        for j in self.sets.get("J", []):
            ttip = self.params.get("ttip", {}).get(j, 0)
            vars.TIP[j] = ttip * vars.PP.get(j, 0) * vars.XST.get(j, 0)
        
        # Consumption variables
        for h in self.sets.get("H", []):
            for i in self.sets.get("I", []):
                vars.C[(i, h)] = state.consumption.get("CO", {}).get((i, h), 0)
                vars.CMIN[(i, h)] = state.les_parameters.get("CMINO", {}).get((i, h), 0)

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
        
        # ROW variables (benchmark levels)
        vars.YROW = state.income.get("YROWO", 0)
        vars.SROW = -state.income.get("CABO", 0)
        vars.CAB = state.income.get("CABO", 0)
        
        # Keep investment aggregates at calibrated benchmark levels.
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

        # Recompute GDP identities with finalized aggregates in strict init too.
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

        gdp_fd = 0.0
        for i in self.sets.get("I", []):
            cons_i = sum(vars.C.get((i, h), 0.0) for h in self.sets.get("H", []))
            gdp_fd += vars.PC.get(i, 0.0) * (
                cons_i + vars.CG.get(i, 0.0) + vars.INV.get(i, 0.0) + vars.VSTK.get(i, 0.0)
            )
            gdp_fd += vars.PE_FOB.get(i, 0.0) * vars.EXD.get(i, 0.0)
            gdp_fd -= vars.PWM.get(i, 0.0) * vars.e * vars.IM.get(i, 0.0)
        vars.GDP_FD = gdp_fd
        vars.GDP_MP_REAL = vars.GDP_MP / vars.PIXCON if abs(vars.PIXCON) > 1e-12 else 0.0
        
        return vars

    def _apply_equation_consistent_adjustments(self, vars: PEPModelVariables) -> None:
        """Apply adjustments so initialized values satisfy benchmark equations."""
        # Factor tax payments from active ad-valorem wedges
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

        # Product-tax aggregates from component tax variables
        tixo_bench = self.state.trade.get("TIXO", {})
        for i in self.sets.get("I", []):
            vars.TIX[i] = tixo_bench.get(i, vars.TIX.get(i, 0.0))

        vars.TIXT = sum(vars.TIX.values())
        vars.TPRCTS = vars.TICT + vars.TIMT + vars.TIXT
        vars.YG = vars.YGK + vars.TDHT + vars.TDFT + vars.TPRODN + vars.TPRCTS + vars.YGTR

        # Keep SG and IT consistent with closure
        tr_to_govt = sum(vars.TR.get((agng, "gvt"), 0) for agng in self.sets.get("AGNG", []))
        vars.SG = vars.YG - tr_to_govt - vars.G

        # Match GAMS closure where CAB is fixed at benchmark.
        vars.CAB = self.state.income.get("CABO", vars.CAB)
        vars.SROW = -vars.CAB

        # Match government product-tax aggregate identity used in GAMS benchmark.
        vars.TPRCTS = vars.TICT + vars.TIMT + vars.TIXT

        # Keep benchmark investment level (do not force savings closure in init mode).
        vars.IT = self.state.income.get("ITO", vars.IT)
        stock_value = sum(vars.PC.get(i, 1.0) * vars.VSTK.get(i, 0.0) for i in self.sets.get("I", []))
        vars.GFCF = vars.IT - stock_value

        # Income-side GDP identity
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
        return pep_variables_to_array(vars, self.sets)
    
    def _array_to_variables(self, array: np.ndarray, template: PEPModelVariables) -> PEPModelVariables:
        """Convert array back to variables."""
        _ = template  # retained for backward-compatible signature
        vars = pep_array_to_variables(np.asarray(array, dtype=float), self.sets)
        rebuild_tax_detail_from_rates(
            vars=vars,
            sets=self.sets,
            params=self.params,
            include_tip=True,
        )
        return vars
    
    def solve(self, method: str = "auto") -> SolverResult:
        """Solve the PEP model.
        
        Args:
            method: Solution method ("auto", "ipopt", or "simple_iteration")
                   "auto" will try IPOPT first, then fall back to simple iteration
            
        Returns:
            SolverResult with solution status and variables
        """
        logger.info("=" * 70)
        logger.info("STARTING MODEL SOLUTION")
        logger.info("=" * 70)
        
        # Handle method selection
        if method == "auto":
            if IPOPT_AVAILABLE:
                logger.info("Auto-selecting IPOPT solver")
                method = "ipopt"
            else:
                logger.info("Auto-selecting simple iteration (IPOPT not available)")
                method = "simple_iteration"
        
        # Route to appropriate solver
        if method == "ipopt":
            if not IPOPT_AVAILABLE:
                logger.error("IPOPT requested but not available. Install with: pip install cyipopt")
                logger.error("Falling back to simple iteration")
                method = "simple_iteration"
            else:
                # Use IPOPT solver - import here to avoid issues if not available
                try:
                    from equilibria.templates.pep_model_solver_ipopt import IPOPTSolver
                    ipopt_solver = IPOPTSolver(
                        self.state,
                        self.tolerance,
                        self.max_iterations,
                        init_mode=self.init_mode,
                        blockwise_commodity_alpha=self.blockwise_commodity_alpha,
                        blockwise_trade_market_alpha=self.blockwise_trade_market_alpha,
                        blockwise_macro_alpha=self.blockwise_macro_alpha,
                        gams_results_gdx=self.gams_results_gdx,
                        gams_results_slice=self.gams_results_slice,
                        baseline_manifest=self.baseline_manifest,
                        require_baseline_manifest=self.require_baseline_manifest,
                        baseline_compatibility_rel_tol=self.baseline_compatibility_rel_tol,
                        enforce_strict_gams_baseline=self.enforce_strict_gams_baseline,
                        sam_file=self.sam_file,
                        val_par_file=self.val_par_file,
                        gdxdump_bin=self.gdxdump_bin,
                    )
                    return ipopt_solver.solve_ipopt()
                except Exception as e:
                    logger.error(f"IPOPT failed: {e}")
                    if self.init_mode == "gams" and self.enforce_strict_gams_baseline:
                        raise RuntimeError(f"gams solve failed due to baseline incompatibility: {e}") from e
                    logger.error("Falling back to simple iteration")
                    method = "simple_iteration"
        
        # Use simple iteration
        vars = self._create_initial_guess()
        return self._solve_simple_iteration(vars)
    
    def _solve_simple_iteration(self, vars: PEPModelVariables) -> SolverResult:
        """Solve using simple iteration (Gauss-Seidel style)."""
        result = SolverResult()
        result.variables = vars
        
        logger.info("Using simple iteration method")
        
        # Initialize variables for final result
        rms_residual = float('inf')
        residuals = {}
        
        for iteration in range(self.max_iterations):
            # Calculate all residuals
            residuals = self.equations.calculate_all_residuals(vars)
            
            # Calculate RMS residual
            residual_values = list(residuals.values())
            rms_residual = np.sqrt(np.mean([r**2 for r in residual_values]))
            max_residual = max(abs(r) for r in residual_values)
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: RMS residual = {rms_residual:.2e}, Max = {max_residual:.2e}")
            
            # Check convergence
            if rms_residual < self.tolerance:
                result.converged = True
                result.iterations = iteration
                result.final_residual = rms_residual
                result.residuals = residuals
                result.message = f"Converged after {iteration} iterations"
                logger.info(f"✓ Converged after {iteration} iterations")
                return result
            
            # Simple update: adjust prices to reduce residuals
            # This is a very simplified approach - real CGE solvers are more sophisticated
            damping = 0.1  # Damping factor to avoid oscillation
            
            for j in self.sets.get("J", []):
                # Adjust wage and rental rates
                eq4_resid = residuals.get(f"EQ4_{j}", 0)
                if abs(eq4_resid) > 1e-6:
                    vars.WC[j] *= (1 + damping * eq4_resid / max(abs(vars.WC[j]), 1))
                    vars.WC[j] = max(0.1, min(10.0, vars.WC[j]))  # Keep bounded
        
        # Did not converge
        result.iterations = self.max_iterations
        result.final_residual = rms_residual
        result.residuals = residuals
        result.message = f"Did not converge after {self.max_iterations} iterations"
        logger.warning(f"✗ Did not converge after {self.max_iterations} iterations")
        
        return result
    
    def validate_solution(self, result: SolverResult) -> dict[str, Any]:
        """Validate solution quality."""
        validation = {
            "converged": result.converged,
            "rms_residual": result.final_residual,
            "max_residual": max(abs(r) for r in result.residuals.values()) if result.residuals else float('inf'),
            "checks": {},
        }
        
        vars = result.variables
        
        # Check Walras' Law (savings = investment)
        total_savings = sum(vars.SH.values()) + sum(vars.SF.values()) + vars.SG + vars.SROW
        walras_error = abs(vars.IT - total_savings)
        validation["checks"]["walras_law"] = {
            "passed": walras_error < self.tolerance * 100,  # More lenient
            "error": walras_error,
            "investment": vars.IT,
            "savings": total_savings,
        }
        
        # Check GDP consistency
        gdp_diff = abs(vars.GDP_BP - vars.GDP_FD)
        validation["checks"]["gdp_consistency"] = {
            "passed": gdp_diff < self.tolerance * 1000,  # Lenient
            "error": gdp_diff,
            "gdp_bp": vars.GDP_BP,
            "gdp_fd": vars.GDP_FD,
        }
        
        # Check market clearing
        for i in self.sets.get("I", []):
            supply = sum(vars.DS.get((j, i), 0) for j in self.sets.get("J", []))
            demand = vars.DD.get(i, 0)
            if supply > 0:
                market_error = abs(supply - demand) / supply
                if market_error > 0.01:  # 1% tolerance
                    validation["checks"][f"market_clearing_{i}"] = {
                        "passed": False,
                        "error": market_error,
                        "supply": supply,
                        "demand": demand,
                    }
        
        return validation
