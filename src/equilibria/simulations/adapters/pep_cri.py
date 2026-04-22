"""PEP-CRI adapter — extends PEP adapter with cross-border labor (L→ROW) for ICIO/IEEM SAMs.

Usage:
    from equilibria.simulations.adapters.pep_cri import PepCRIAdapter

    adapter = PepCRIAdapter(
        sam_file="sam_cri_pep_icio36.xlsx",
        val_par_file="VAL_PAR.xlsx",
        dynamic_sets=True,
    )
    state = adapter.fit_base_state()

The GAMS counterpart is PEP-1-1_v2_1_cri.gms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from equilibria.simulations.adapters.pep import PepAdapter
from equilibria.simulations.types import Scenario
from equilibria.templates.pep_calibration_unified import PEPModelState
from equilibria.templates.pep_cri_calibration import (
    PEPCRIModelCalibrator,
    PEPCRIModelCalibratorExcel,
    PEPCRIModelCalibratorDynamicSAM,
    PEPCRIModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_cri_solver import PEPCRIModelSolver
from equilibria.templates.pep_model_solver import PEPModelSolver


class PepCRIAdapter(PepAdapter):
    """PEP adapter for CRI/ICIO SAMs with cross-border labor compensation.

    Extends the standard PEP adapter by:
    - Using PEPCRIIncomeCalibrator (YROWO includes L→ROW flows)
    - Using PEPCRIModelEquations  (EQ44 includes L→ROW wages)
    - Using PEPCRIModelSolver / PEPCRIIPOPTSolver
    """

    def fit_base_state(self) -> PEPModelState:
        """Calibrate using CRI-specific calibrators (L→ROW included)."""
        sam_file_for_run = self._prepare_runtime_sam_file()
        self._runtime_sam_file = sam_file_for_run
        is_excel = sam_file_for_run.suffix.lower() in {".xlsx", ".xls"}
        self._run_runtime_sam_qa(sam_file_for_run, dynamic_sam=self.dynamic_sets)

        if self.dynamic_sets:
            if is_excel:
                calibrator = PEPCRIModelCalibratorExcelDynamicSAM(
                    sam_file=sam_file_for_run,
                    val_par_file=self.val_par_file,
                    accounts=self.accounts,
                )
            else:
                calibrator = PEPCRIModelCalibratorDynamicSAM(
                    sam_file=sam_file_for_run,
                    val_par_file=self.val_par_file,
                    accounts=self.accounts,
                )
        else:
            if is_excel:
                calibrator = PEPCRIModelCalibratorExcel(
                    sam_file=sam_file_for_run,
                    val_par_file=self.val_par_file,
                    dynamic_sets=False,
                )
            else:
                calibrator = PEPCRIModelCalibrator(
                    sam_file=sam_file_for_run,
                    val_par_file=self.val_par_file,
                    dynamic_sets=False,
                )

        state = calibrator.calibrate()
        self._sets = dict(state.sets)
        return state

    def solve_state(
        self,
        state: PEPModelState,
        *,
        initial_vars: Any | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
        scenario: Scenario | None = None,
    ) -> tuple[PEPModelSolver, Any, dict[str, Any]]:
        """Solve using PEPCRIModelSolver (EQ44 with L→ROW)."""
        effective_contract = self._resolve_contract_for_scenario(scenario)
        solver = PEPCRIModelSolver(
            calibrated_state=state,
            tolerance=self.solve_tolerance,
            max_iterations=self.max_iterations,
            init_mode=self.init_mode,
            contract=effective_contract,
            config=self.runtime_config,
            gams_results_gdx=reference_results_gdx,
            gams_results_slice=reference_slice.lower(),
            sam_file=self._runtime_sam_file,
            val_par_file=self.val_par_file,
            gdxdump_bin=self.gdxdump_bin,
            initial_vars=initial_vars,
        )
        solution = solver.solve(method=self.method)
        validation = solver.validate_solution(solution)
        return solver, solution, validation
