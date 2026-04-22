"""PEP-CRI solver layer — extends PEP solver with cross-border labor equations.

Hierarchy:
    IPOPTSolver          (standard PEP IPOPT solver)
        └── PEPCRIIPOPTSolver   (uses PEPCRIModelEquations in EQ44)

    PEPModelSolver       (standard multi-method solver)
        └── PEPCRIModelSolver   (delegates to PEPCRIIPOPTSolver)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from equilibria.templates.pep_contract import PEPContract
from equilibria.templates.pep_cri_model_equations import PEPCRIModelEquations
from equilibria.templates.pep_model_equations import PEPModelVariables
from equilibria.templates.pep_model_solver import PEPModelSolver
from equilibria.templates.pep_model_solver_ipopt import IPOPTSolver
from equilibria.templates.pep_runtime_config import PEPRuntimeConfig


class PEPCRIIPOPTSolver(IPOPTSolver):
    """IPOPT solver for CRI/ICIO SAMs — uses PEPCRIModelEquations (EQ44 with L→ROW)."""

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
        # Replace equations with CRI-extended version
        self.equations = PEPCRIModelEquations(
            self.sets,
            self.params,
            activation_masks=self.contract.equations.activation_masks,
        )


class PEPCRIModelSolver(PEPModelSolver):
    """Multi-method solver for CRI/ICIO SAMs — delegates to PEPCRIIPOPTSolver."""

    def _build_ipopt_shadow_solver(self) -> PEPCRIIPOPTSolver:
        return PEPCRIIPOPTSolver(
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
