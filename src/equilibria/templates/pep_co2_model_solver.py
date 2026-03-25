"""Solver layer for the sector-CO2 PEP extension."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from equilibria.templates.pep_co2_data import (
    attach_co2_metrics,
    carbon_unit_tax,
    get_state_co2_block,
    set_state_co2_block,
    validate_and_fill_co2_block,
)
from equilibria.templates.pep_co2_model_equations import PEPCO2ModelEquations
from equilibria.templates.pep_contract import PEPContract
from equilibria.templates.pep_model_equations import PEPModelVariables, SolverResult
from equilibria.templates.pep_model_solver import DEBUG_SIMPLE_ITERATION_METHOD, PEPModelSolver
from equilibria.templates.pep_model_solver_ipopt import IPOPT_AVAILABLE, IPOPTSolver
from equilibria.templates.pep_runtime_config import PEPRuntimeConfig


class PEPCO2IPOPTSolver(IPOPTSolver):
    """IPOPT solver for the sector-CO2 PEP extension."""

    def __init__(
        self,
        calibrated_state: Any,
        tolerance: float | None = None,
        max_iterations: int | None = None,
        init_mode: Literal["gams", "excel"] | str = "excel",
        blockwise_commodity_alpha: float = 0.75,
        blockwise_trade_market_alpha: float = 0.5,
        blockwise_macro_alpha: float = 1.0,
        contract: str | Mapping[str, Any] | PEPContract | None = None,
        config: str | Mapping[str, Any] | PEPRuntimeConfig | None = None,
        gams_results_gdx: Path | str | None = None,
        gams_parameters_gdx: Path | str | None = None,
        gams_results_slice: Literal["base", "sim1"] = "sim1",
        baseline_manifest: Path | str | None = None,
        require_baseline_manifest: bool = False,
        baseline_compatibility_rel_tol: float = 1e-4,
        enforce_strict_gams_baseline: bool = True,
        sam_file: Path | str | None = None,
        val_par_file: Path | str | None = None,
        gdxdump_bin: str = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
        initial_vars: PEPModelVariables | None = None,
    ) -> None:
        super().__init__(
            calibrated_state=calibrated_state,
            tolerance=tolerance,
            max_iterations=max_iterations,
            init_mode=init_mode,
            blockwise_commodity_alpha=blockwise_commodity_alpha,
            blockwise_trade_market_alpha=blockwise_trade_market_alpha,
            blockwise_macro_alpha=blockwise_macro_alpha,
            contract=contract,
            config=config,
            gams_results_gdx=gams_results_gdx,
            gams_parameters_gdx=gams_parameters_gdx,
            gams_results_slice=gams_results_slice,
            baseline_manifest=baseline_manifest,
            require_baseline_manifest=require_baseline_manifest,
            baseline_compatibility_rel_tol=baseline_compatibility_rel_tol,
            enforce_strict_gams_baseline=enforce_strict_gams_baseline,
            sam_file=sam_file,
            val_par_file=val_par_file,
            gdxdump_bin=gdxdump_bin,
            initial_vars=initial_vars,
        )
        self.equations = PEPCO2ModelEquations(
            self.sets,
            self.params,
            activation_masks=self.contract.equations.activation_masks,
        )

    def _extract_parameters(self, state: Any) -> dict[str, Any]:
        params = super()._extract_parameters(state)
        block = validate_and_fill_co2_block(get_state_co2_block(state), state.sets.get("J", []))
        set_state_co2_block(state, block)
        params["co2_intensity"] = dict(block["co2_intensity"])
        params["tco2b"] = dict(block["tco2b"])
        params["tco2scal"] = float(block["tco2scal"])
        return params

    def _create_initial_guess(self) -> PEPModelVariables:
        vars_obj = super()._create_initial_guess()
        self._sync_co2_block(vars_obj, include_pt=True)
        return vars_obj

    def _reconcile_tax_identities(self, vars: PEPModelVariables) -> None:
        super()._reconcile_tax_identities(vars)
        self._sync_co2_block(vars, include_pt=False)

    def _apply_equation_consistent_adjustments(self, vars: PEPModelVariables) -> None:
        super()._apply_equation_consistent_adjustments(vars)
        self._sync_co2_block(vars, include_pt=True)

    def _sync_co2_block(self, vars: PEPModelVariables, *, include_pt: bool) -> None:
        if not self._has_active_carbon_tax():
            attach_co2_metrics(vars, self.params, self.sets.get("J", []))
            return

        for j in self.sets.get("J", []):
            ttip = self.params.get("ttip", {}).get(j, 0.0)
            carbon_tax = carbon_unit_tax(self.params, vars, j)
            if include_pt:
                vars.PT[j] = (1.0 + ttip) * vars.PP.get(j, 0.0) + carbon_tax
            vars.TIP[j] = (ttip * vars.PP.get(j, 0.0) + carbon_tax) * vars.XST.get(j, 0.0)

        vars.TIPT = sum(vars.TIP.values())
        vars.TPRODN = vars.TIWT + vars.TIKT + vars.TIPT
        vars.YG = vars.YGK + vars.TDHT + vars.TDFT + vars.TPRODN + vars.TPRCTS + vars.YGTR
        tr_to_govt = sum(vars.TR.get((agng, "gvt"), 0.0) for agng in self.sets.get("AGNG", []))
        vars.SG = vars.YG - tr_to_govt - vars.G
        self._recompute_gdp_aggregates(vars)
        vars.GDP_MP_REAL = vars.GDP_MP / vars.PIXCON if abs(vars.PIXCON) > 1e-12 else 0.0
        attach_co2_metrics(vars, self.params, self.sets.get("J", []))

    def _has_active_carbon_tax(self) -> bool:
        for j in self.sets.get("J", []):
            if abs(float(self.params.get("co2_intensity", {}).get(j, 0.0))) <= 1e-12:
                continue
            if abs(float(self.params.get("tco2b", {}).get(j, 0.0))) <= 1e-12:
                continue
            if abs(float(self.params.get("tco2scal", 1.0))) <= 1e-12:
                continue
            return True
        return False


class PEPCO2ModelSolver(PEPModelSolver):
    """PEPModelSolver variant with sector CO2 tax support."""

    def __init__(
        self,
        calibrated_state: Any,
        tolerance: float | None = None,
        max_iterations: int | None = None,
        init_mode: Literal["gams", "excel"] | str = "excel",
        blockwise_commodity_alpha: float = 0.75,
        blockwise_trade_market_alpha: float = 0.5,
        blockwise_macro_alpha: float = 1.0,
        contract: str | Mapping[str, Any] | PEPContract | None = None,
        config: str | Mapping[str, Any] | PEPRuntimeConfig | None = None,
        gams_results_gdx: Path | str | None = None,
        gams_parameters_gdx: Path | str | None = None,
        gams_results_slice: Literal["base", "sim1"] = "sim1",
        baseline_manifest: Path | str | None = None,
        require_baseline_manifest: bool = False,
        baseline_compatibility_rel_tol: float = 1e-4,
        enforce_strict_gams_baseline: bool = True,
        sam_file: Path | str | None = None,
        val_par_file: Path | str | None = None,
        gdxdump_bin: str = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
        initial_vars: PEPModelVariables | None = None,
    ) -> None:
        super().__init__(
            calibrated_state=calibrated_state,
            tolerance=tolerance,
            max_iterations=max_iterations,
            init_mode=init_mode,
            blockwise_commodity_alpha=blockwise_commodity_alpha,
            blockwise_trade_market_alpha=blockwise_trade_market_alpha,
            blockwise_macro_alpha=blockwise_macro_alpha,
            contract=contract,
            config=config,
            gams_results_gdx=gams_results_gdx,
            gams_parameters_gdx=gams_parameters_gdx,
            gams_results_slice=gams_results_slice,
            baseline_manifest=baseline_manifest,
            require_baseline_manifest=require_baseline_manifest,
            baseline_compatibility_rel_tol=baseline_compatibility_rel_tol,
            enforce_strict_gams_baseline=enforce_strict_gams_baseline,
            sam_file=sam_file,
            val_par_file=val_par_file,
            gdxdump_bin=gdxdump_bin,
            initial_vars=initial_vars,
        )
        self.equations = PEPCO2ModelEquations(
            self.sets,
            self.params,
            activation_masks=self.contract.equations.activation_masks,
        )

    def _extract_parameters(self, state: Any) -> dict[str, Any]:
        params = super()._extract_parameters(state)
        block = validate_and_fill_co2_block(get_state_co2_block(state), state.sets.get("J", []))
        set_state_co2_block(state, block)
        params["co2_intensity"] = dict(block["co2_intensity"])
        params["tco2b"] = dict(block["tco2b"])
        params["tco2scal"] = float(block["tco2scal"])
        return params

    def _build_ipopt_shadow_solver(self) -> PEPCO2IPOPTSolver:
        return PEPCO2IPOPTSolver(
            calibrated_state=self.state,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            init_mode=self.init_mode,
            blockwise_commodity_alpha=self.blockwise_commodity_alpha,
            blockwise_trade_market_alpha=self.blockwise_trade_market_alpha,
            blockwise_macro_alpha=self.blockwise_macro_alpha,
            contract=self.contract,
            config=self.runtime_config,
            gams_results_gdx=self.gams_results_gdx,
            gams_parameters_gdx=self.gams_parameters_gdx,
            gams_results_slice=self.gams_results_slice,
            baseline_manifest=self.baseline_manifest,
            require_baseline_manifest=self.require_baseline_manifest,
            baseline_compatibility_rel_tol=self.baseline_compatibility_rel_tol,
            enforce_strict_gams_baseline=self.enforce_strict_gams_baseline,
            sam_file=self.sam_file,
            val_par_file=self.val_par_file,
            gdxdump_bin=self.gdxdump_bin,
            initial_vars=self.initial_vars,
        )

    def _create_initial_guess(self) -> PEPModelVariables:
        if self.initial_vars is not None:
            vars_obj = copy.deepcopy(self.initial_vars)
            attach_co2_metrics(vars_obj, self.params, self.sets.get("J", []))
            return vars_obj

        ipopt_solver = self._build_ipopt_shadow_solver()
        vars_ipopt = ipopt_solver._create_initial_guess()
        self.params = ipopt_solver.params
        self.equations = ipopt_solver.equations
        self.last_closure_validation_report = ipopt_solver.last_closure_validation_report
        attach_co2_metrics(vars_ipopt, self.params, self.sets.get("J", []))
        return vars_ipopt

    def solve(self, method: str = "auto") -> SolverResult:
        if method == "simple_iteration":
            raise ValueError(
                "method='simple_iteration' is no longer part of the public API. "
                f"For internal debugging only, use method='{DEBUG_SIMPLE_ITERATION_METHOD}'."
            )

        if method == "auto":
            if not IPOPT_AVAILABLE:
                raise RuntimeError(
                    "method='auto' requires IPOPT (cyipopt). "
                    "Install with `uv sync --extra ipopt`."
                )
            method = "ipopt"

        if method == "ipopt":
            if not IPOPT_AVAILABLE:
                raise RuntimeError(
                    "method='ipopt' requested but cyipopt is not available. "
                    "Install with `uv sync --extra ipopt`."
                )
            ipopt_solver = self._build_ipopt_shadow_solver()
            result = ipopt_solver.solve_ipopt()
            self.params = ipopt_solver.params
            self.equations = ipopt_solver.equations
            self.last_closure_validation_report = ipopt_solver.last_closure_validation_report
            attach_co2_metrics(result.variables, self.params, self.sets.get("J", []))
            return result

        if method == DEBUG_SIMPLE_ITERATION_METHOD:
            vars_obj = self._create_initial_guess()
            result = self._solve_simple_iteration(vars_obj)
            attach_co2_metrics(result.variables, self.params, self.sets.get("J", []))
            return result

        raise ValueError(
            f"Unknown solve method '{method}'. Supported methods: auto, ipopt, "
            f"{DEBUG_SIMPLE_ITERATION_METHOD} (internal debug only)."
        )
