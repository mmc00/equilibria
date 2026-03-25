"""PEP adapter for generic scenario simulation API."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from equilibria.qa.reporting import format_report_summary
from equilibria.qa.sam_checks import run_sam_qa_from_file
from equilibria.simulations.adapters.base import BaseModelAdapter
from equilibria.simulations.pep_compare import compare_with_gams
from equilibria.simulations.pep_compare import key_indicators as pep_key_indicators
from equilibria.simulations.types import Scenario, Shock, ShockDefinition
from equilibria.templates.pep_calibration_unified import (
    PEPModelCalibrator,
    PEPModelState,
)
from equilibria.templates.pep_calibration_unified_dynamic import (
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_calibration_unified_excel import (
    PEPModelCalibratorExcel,
)
from equilibria.templates.pep_contract import PEPContract, build_pep_contract
from equilibria.templates.pep_model_solver import PEPModelSolver
from equilibria.templates.pep_cri_transform import (
    should_apply_cri_pep_fix,
    transform_sam_to_pep_compatible,
)
from equilibria.templates.pep_runtime_config import PEPRuntimeConfig, build_pep_runtime_config

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SAM_FILE = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
DEFAULT_VAL_PAR_FILE = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"
DEFAULT_GDXDUMP_BIN = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"
DEFAULT_CRI_ACCOUNTS = {
    "gvt": "gvt",
    "row": "row",
    "td": "td",
    "ti": "ti",
    "tm": "tm",
    "tx": "tx",
    "inv": "inv",
    "vstk": "vstk",
}

logger = logging.getLogger(__name__)


def _deep_merge_dict(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


class PepAdapter(BaseModelAdapter):
    """PEP-specific implementation of the simulation adapter contract."""

    def __init__(
        self,
        *,
        sam_file: Path | str = DEFAULT_SAM_FILE,
        val_par_file: Path | str | None = DEFAULT_VAL_PAR_FILE,
        method: str = "ipopt",
        init_mode: str = "excel",
        dynamic_sets: bool = True,
        solve_tolerance: float | None = None,
        max_iterations: int | None = None,
        contract: str | dict[str, Any] | PEPContract | None = None,
        config: str | dict[str, Any] | PEPRuntimeConfig | None = None,
        gdxdump_bin: str = DEFAULT_GDXDUMP_BIN,
        accounts: dict[str, str] | None = None,
        sam_qa_mode: str = "off",
        sam_qa_report: Path | str | None = None,
        sam_qa_balance_rel_tol: float = 1e-6,
        sam_qa_gdp_rel_tol: float = 0.08,
        sam_qa_max_samples: int = 8,
        sam_qa_strict_structural: bool = False,
        cri_fix_mode: str = "auto",
        cri_fix_output: Path | str | None = None,
        cri_fix_report: Path | str | None = None,
        cri_fix_target_mode: str = "geomean",
        cri_fix_margin_commodity: str = "ser",
    ) -> None:
        self.sam_file = Path(sam_file)
        self.val_par_file = Path(val_par_file) if val_par_file is not None else None
        self.method = str(method)
        self.init_mode = str(init_mode)
        self.dynamic_sets = bool(dynamic_sets)
        self.contract = build_pep_contract(contract)
        self.runtime_config = build_pep_runtime_config(config)
        self.solve_tolerance = (
            float(self.runtime_config.tolerance) if solve_tolerance is None else float(solve_tolerance)
        )
        self.max_iterations = (
            int(self.runtime_config.max_iterations) if max_iterations is None else int(max_iterations)
        )
        self.gdxdump_bin = str(gdxdump_bin)
        self.accounts = dict(accounts) if accounts is not None else dict(DEFAULT_CRI_ACCOUNTS)
        self.sam_qa_mode = str(sam_qa_mode).strip().lower()
        self.sam_qa_report = Path(sam_qa_report) if sam_qa_report is not None else None
        self.sam_qa_balance_rel_tol = float(sam_qa_balance_rel_tol)
        self.sam_qa_gdp_rel_tol = float(sam_qa_gdp_rel_tol)
        self.sam_qa_max_samples = int(sam_qa_max_samples)
        self.sam_qa_strict_structural = bool(sam_qa_strict_structural)
        self.cri_fix_mode = str(cri_fix_mode).strip().lower()
        self.cri_fix_output = Path(cri_fix_output) if cri_fix_output is not None else None
        self.cri_fix_report = Path(cri_fix_report) if cri_fix_report is not None else None
        self.cri_fix_target_mode = str(cri_fix_target_mode)
        self.cri_fix_margin_commodity = str(cri_fix_margin_commodity)
        self._sets: dict[str, list[str]] = {}
        self._runtime_sam_file = self.sam_file
        self.last_sam_qa_report: Any | None = None
        self.last_cri_fix_report: dict[str, Any] | None = None

    def capabilities(self) -> dict[str, Any]:
        capabilities = super().capabilities()
        capabilities.update(
            {
                "contract": self.contract.name,
                "runtime_config": self.runtime_config.name,
            }
        )
        return capabilities

    def _resolve_contract_for_scenario(self, scenario: Scenario | None) -> PEPContract:
        if scenario is None:
            return self.contract
        if scenario.closure is None:
            return self.contract

        contract_data = self.contract.model_dump(mode="python")
        merged = _deep_merge_dict(contract_data, {"closure": scenario.closure})
        return build_pep_contract(merged)

    def _prepare_runtime_sam_file(self) -> Path:
        sam_file_for_run = self.sam_file
        if should_apply_cri_pep_fix(sam_file_for_run, mode=self.cri_fix_mode):
            output_sam = (
                self.cri_fix_output
                if self.cri_fix_output is not None
                else Path("output") / f"{sam_file_for_run.stem}-pep-compatible{sam_file_for_run.suffix}"
            )
            report_json = (
                self.cri_fix_report
                if self.cri_fix_report is not None
                else Path("output") / f"{sam_file_for_run.stem}-pep-compatible-report.json"
            )
            self.last_cri_fix_report = transform_sam_to_pep_compatible(
                input_sam=sam_file_for_run,
                output_sam=output_sam,
                report_json=report_json,
                target_mode=self.cri_fix_target_mode,
                margin_commodity=self.cri_fix_margin_commodity,
            )
            sam_file_for_run = output_sam
        return sam_file_for_run

    def _run_runtime_sam_qa(self, sam_file_for_run: Path, *, dynamic_sam: bool) -> None:
        if self.sam_qa_mode == "off":
            self.last_sam_qa_report = None
            return

        report = run_sam_qa_from_file(
            sam_file=sam_file_for_run,
            dynamic_sam=dynamic_sam,
            accounts=self.accounts,
            balance_rel_tol=self.sam_qa_balance_rel_tol,
            gdp_rel_tol=self.sam_qa_gdp_rel_tol,
            max_samples=self.sam_qa_max_samples,
            strict_structural=self.sam_qa_strict_structural,
        )
        self.last_sam_qa_report = report
        if self.sam_qa_report is not None:
            report.save_json(self.sam_qa_report)

        summary = format_report_summary(report)
        if report.passed:
            logger.info(summary)
            return
        if self.sam_qa_mode == "hard_fail":
            raise RuntimeError(f"SAM QA failed: {summary}")
        logger.warning("%s; continuing because sam_qa_mode=%s", summary, self.sam_qa_mode)

    def fit_base_state(self) -> PEPModelState:
        sam_file_for_run = self._prepare_runtime_sam_file()
        self._runtime_sam_file = sam_file_for_run
        is_excel = sam_file_for_run.suffix.lower() in {".xlsx", ".xls"}
        self._run_runtime_sam_qa(sam_file_for_run, dynamic_sam=self.dynamic_sets)
        if self.dynamic_sets:
            if is_excel:
                calibrator = PEPModelCalibratorExcelDynamicSAM(
                    sam_file=sam_file_for_run,
                    val_par_file=self.val_par_file,
                    accounts=self.accounts,
                    i1_excluded_members=self.runtime_config.i1_excluded_members,
                )
            else:
                calibrator = PEPModelCalibratorDynamicSAM(
                    sam_file=sam_file_for_run,
                    val_par_file=self.val_par_file,
                    accounts=self.accounts,
                    i1_excluded_members=self.runtime_config.i1_excluded_members,
                )
        else:
            if is_excel:
                calibrator = PEPModelCalibratorExcel(
                    sam_file=sam_file_for_run,
                    val_par_file=self.val_par_file,
                    dynamic_sets=False,
                )
            else:
                calibrator = PEPModelCalibrator(
                    sam_file=sam_file_for_run,
                    val_par_file=self.val_par_file,
                    dynamic_sets=False,
                )
        state = calibrator.calibrate()
        self._sets = dict(state.sets)
        return state

    def available_shocks(self) -> list[ShockDefinition]:
        i_members = tuple(self._sets.get("I", [])) or None
        return [
            ShockDefinition(
                var="G",
                kind="scalar",
                domain=None,
                ops=("set", "scale", "add"),
                description="Government expenditure closure (GO/G).",
            ),
            ShockDefinition(
                var="PWM",
                kind="indexed",
                domain="I",
                members=i_members,
                ops=("set", "scale", "add"),
                description="World import prices PWMO(i).",
            ),
            ShockDefinition(
                var="PWX",
                kind="indexed",
                domain="I",
                members=i_members,
                ops=("set", "scale", "add"),
                description="World export prices PWXO(i).",
            ),
            ShockDefinition(
                var="ttix",
                kind="indexed",
                domain="I",
                members=i_members,
                ops=("set", "scale", "add"),
                description="Export tax rates ttixO(i).",
            ),
        ]

    def apply_shock(self, state: PEPModelState, shock: Shock) -> None:
        var = shock.var.strip().lower()
        op = shock.op.strip().lower()
        if op not in {"set", "scale", "add"}:
            raise ValueError(f"Unsupported shock op '{shock.op}'.")

        if var == "g":
            base_val = float(state.consumption.get("GO", 0.0))
            state.consumption["GO"] = self._apply_scalar_op(base_val, op, shock.values)
            return

        if var == "pwm":
            data = state.trade.get("PWMO", {})
            state.trade["PWMO"] = self._apply_indexed_op(
                data,
                op=op,
                values=shock.values,
                domain=set(state.sets.get("I", [])),
                var_name="PWM",
            )
            return

        if var == "pwx":
            data = state.trade.get("PWXO", {})
            state.trade["PWXO"] = self._apply_indexed_op(
                data,
                op=op,
                values=shock.values,
                domain=set(state.sets.get("I", [])),
                var_name="PWX",
            )
            return

        if var == "ttix":
            data = state.trade.get("ttixO", {})
            updated = self._apply_indexed_op(
                data,
                op=op,
                values=shock.values,
                domain=set(state.sets.get("I", [])),
                var_name="ttix",
            )
            state.trade["ttixO"] = updated
            self._sync_export_tax_aggregates(state)
            return

        raise ValueError(f"Variable '{shock.var}' is not shockable for model 'pep'.")

    def solve_state(
        self,
        state: PEPModelState,
        *,
        initial_vars: Any | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
        scenario: Scenario | None = None,
    ) -> tuple[PEPModelSolver, Any, dict[str, Any]]:
        effective_contract = self._resolve_contract_for_scenario(scenario)
        solver = PEPModelSolver(
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

    def compare_with_reference(
        self,
        *,
        solution_vars: Any,
        solution_params: dict[str, Any],
        reference_results_gdx: Path,
        reference_slice: str,
        abs_tol: float,
        rel_tol: float,
    ) -> dict[str, Any]:
        return compare_with_gams(
            solution_vars=solution_vars,
            solution_params=solution_params,
            gams_results_gdx=reference_results_gdx,
            gams_slice=reference_slice.lower(),
            gdxdump_bin=self.gdxdump_bin,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )

    def key_indicators(self, vars_obj: Any) -> dict[str, float]:
        return pep_key_indicators(vars_obj)

    @staticmethod
    def _apply_scalar_op(current: float, op: str, values: float | dict[str, float]) -> float:
        if isinstance(values, dict):
            if "*" not in values:
                raise ValueError("Scalar shock with dict values requires '*' entry.")
            value = float(values["*"])
        else:
            value = float(values)

        if op == "set":
            return value
        if op == "scale":
            return current * value
        if op == "add":
            return current + value
        raise ValueError(f"Unsupported op '{op}'.")

    @staticmethod
    def _apply_indexed_op(
        data: dict[str, float] | Any,
        *,
        op: str,
        values: float | dict[str, float],
        domain: set[str],
        var_name: str,
    ) -> dict[str, float]:
        src = dict(data) if isinstance(data, dict) else {}
        if not src and domain:
            src = dict.fromkeys(domain, 0.0)

        if isinstance(values, dict):
            unknown = {k for k in values if k != "*" and k not in domain}
            if unknown:
                bad = ", ".join(sorted(unknown))
                raise ValueError(f"{var_name}: unknown indices in shock values: {bad}")
            updates = dict(values)
        else:
            updates = {"*": float(values)}

        if "*" in updates:
            wildcard = float(updates["*"])
            for idx in list(src):
                src[idx] = PepAdapter._merge_value(src[idx], op, wildcard)

        for idx, value in updates.items():
            if idx == "*":
                continue
            if idx not in src:
                src[idx] = 0.0
            src[idx] = PepAdapter._merge_value(src[idx], op, float(value))

        return src

    @staticmethod
    def _merge_value(current: float, op: str, value: float) -> float:
        if op == "set":
            return value
        if op == "scale":
            return current * value
        if op == "add":
            return current + value
        raise ValueError(f"Unsupported op '{op}'.")

    @staticmethod
    def _sync_export_tax_aggregates(state: PEPModelState) -> None:
        ttix = state.trade.get("ttixO", {})
        exdo = state.trade.get("EXDO", {})
        pe = state.trade.get("PEO", {})
        tmrg_x = state.trade.get("tmrg_X", {})
        pco = state.trade.get("PCO", {})

        if not isinstance(ttix, dict) or not isinstance(exdo, dict):
            return

        tixo: dict[str, float] = {}
        for i, rate in ttix.items():
            ex_i = float(exdo.get(i, 0.0))
            if abs(ex_i) <= 1e-12:
                tixo[i] = 0.0
                continue
            margin = 0.0
            for ij, pc_val in pco.items():
                margin += float(pc_val) * float(tmrg_x.get((ij, i), 0.0))
            unit_tax_base = float(pe.get(i, 0.0)) + margin
            tixo[i] = float(rate) * unit_tax_base * ex_i

        state.trade["TIXO"] = tixo

        income = state.income
        if not isinstance(income, dict):
            return
        income["TIXTO"] = float(sum(tixo.values()))
        if {"TICTO", "TIMTO", "TIXTO"}.issubset(income):
            income["TPRCTSO"] = (
                float(income.get("TICTO", 0.0))
                + float(income.get("TIMTO", 0.0))
                + float(income.get("TIXTO", 0.0))
            )
        if {"YGKO", "TDHTO", "TDFTO", "TPRODNO", "TPRCTSO", "YGTRO"}.issubset(income):
            income["YGO"] = (
                float(income.get("YGKO", 0.0))
                + float(income.get("TDHTO", 0.0))
                + float(income.get("TDFTO", 0.0))
                + float(income.get("TPRODNO", 0.0))
                + float(income.get("TPRCTSO", 0.0))
                + float(income.get("YGTRO", 0.0))
            )
