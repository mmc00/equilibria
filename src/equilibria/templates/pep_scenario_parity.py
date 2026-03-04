"""PEP pep2 scenario runner with GAMS comparison for BASE and EXPORT_TAX."""

from __future__ import annotations

import copy
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from equilibria.babel.gdx.reader import read_gdx
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator, PEPModelState
from equilibria.templates.pep_calibration_unified_dynamic import PEPModelCalibratorDynamic
from equilibria.templates.pep_model_equations import PEPModelVariables
from equilibria.templates.pep_model_solver import IPOPT_AVAILABLE, PEPModelSolver


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAM_FILE = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
DEFAULT_VAL_PAR_FILE = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"
DEFAULT_RESULTS_GDX = REPO_ROOT / "src/equilibria/templates/reference/pep2/scripts/Results.gdx"
DEFAULT_GDXDUMP_BIN = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"

_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
_LAB_RE = re.compile(r"'([^']*)'")
_SCENARIOS = {"base", "sim1", "var"}


@dataclass
class ScenarioSolveReport:
    scenario: str
    export_tax_multiplier: float
    converged: bool
    iterations: int
    final_residual: float
    message: str
    key_indicators: dict[str, float]
    validation: dict[str, Any]


def get_solution_value(
    vars_obj: PEPModelVariables,
    symbol: str,
    idx: tuple[str, ...],
    params: dict[str, Any],
) -> float | None:
    """Map one `val*` symbol from Results.gdx to Python solution values."""
    if symbol == "valPWX" and len(idx) == 1:
        return params.get("PWX", {}).get(idx[0], 1.0)
    if symbol == "valPT" and len(idx) == 1:
        return vars_obj.PT.get(idx[0], params.get("PT", {}).get(idx[0], 1.0))
    if symbol == "valttdh1" and len(idx) == 1:
        return params.get("ttdh1", {}).get(idx[0])
    if symbol == "valttic" and len(idx) == 1:
        return params.get("ttic", {}).get(idx[0])
    if symbol == "valtr1" and len(idx) == 1:
        return params.get("tr1", {}).get(idx[0])
    if symbol == "valttim" and len(idx) == 1:
        return params.get("ttim", {}).get(idx[0])
    if symbol == "valttiw" and len(idx) == 2:
        return params.get("ttiw", {}).get((idx[0], idx[1]))
    if symbol == "valKS" and len(idx) == 1:
        return params.get("KS", {}).get(idx[0])
    if symbol == "valLS" and len(idx) == 1:
        return params.get("LS", {}).get(idx[0])
    if symbol == "valRK" and len(idx) == 1:
        return vars_obj.RK.get(idx[0], 1.0)
    if symbol == "valsh1" and len(idx) == 1:
        return params.get("sh1", {}).get(idx[0])
    if symbol == "valttip" and len(idx) == 1:
        return params.get("ttip", {}).get(idx[0])
    if symbol == "valttdf1" and len(idx) == 1:
        return params.get("ttdf1", {}).get(idx[0])
    if symbol == "valttik" and len(idx) == 2:
        return params.get("ttik", {}).get((idx[0], idx[1]))
    if symbol == "valttix" and len(idx) == 1:
        return params.get("ttix", {}).get(idx[0])
    if symbol == "valGFCF_REAL" and len(idx) == 0:
        pixinv = vars_obj.PIXINV if abs(vars_obj.PIXINV) > 1e-12 else 1.0
        return vars_obj.GFCF / pixinv

    field = "e" if symbol == "vale" else symbol[3:]
    if not hasattr(vars_obj, field):
        return None
    obj = getattr(vars_obj, field)

    if isinstance(obj, dict):
        if len(idx) == 0:
            return None
        if len(idx) == 1:
            return obj.get(idx[0])
        return obj.get(tuple(idx))

    if len(idx) != 0:
        return None
    try:
        return float(obj)
    except Exception:
        return None


class PEPScenarioParityRunner:
    """Run BASE and EXPORT_TAX scenarios and compare both with GAMS results."""

    def __init__(
        self,
        sam_file: Path | str = DEFAULT_SAM_FILE,
        val_par_file: Path | str | None = DEFAULT_VAL_PAR_FILE,
        gams_results_gdx: Path | str = DEFAULT_RESULTS_GDX,
        gdxdump_bin: str = DEFAULT_GDXDUMP_BIN,
        *,
        dynamic_sets: bool = True,
        init_mode: str = "excel",
        method: str = "ipopt",
        solve_tolerance: float = 1e-8,
        max_iterations: int = 300,
        export_tax_multiplier: float = 0.75,
        export_tax_homotopy: bool = True,
        export_tax_homotopy_steps: int = 5,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
    ) -> None:
        self.sam_file = Path(sam_file)
        self.val_par_file = Path(val_par_file) if val_par_file is not None else None
        self.gams_results_gdx = Path(gams_results_gdx)
        self.gdxdump_bin = gdxdump_bin
        self.dynamic_sets = bool(dynamic_sets)
        self.init_mode = init_mode
        self.method = method
        self.solve_tolerance = solve_tolerance
        self.max_iterations = max_iterations
        self.export_tax_multiplier = export_tax_multiplier
        self.export_tax_homotopy = bool(export_tax_homotopy)
        self.export_tax_homotopy_steps = int(export_tax_homotopy_steps)
        self.compare_abs_tol = compare_abs_tol
        self.compare_rel_tol = compare_rel_tol

    def run(self) -> dict[str, Any]:
        """Execute BASE and EXPORT_TAX scenarios and compare to GAMS."""
        base_state = self._calibrate_base_state()

        base_solver, base_solution, base_validation = self._solve_state(base_state)
        base_run = ScenarioSolveReport(
            scenario="base",
            export_tax_multiplier=1.0,
            converged=bool(base_solution.converged),
            iterations=int(base_solution.iterations),
            final_residual=float(base_solution.final_residual),
            message=str(base_solution.message),
            key_indicators=self._key_indicators(base_solution.variables),
            validation=base_validation,
        )

        export_state = self._clone_with_export_tax_shock(
            base_state,
            multiplier=self.export_tax_multiplier,
        )
        export_solver, export_solution, export_validation = self._solve_state(
            export_state,
            initial_vars=base_solution.variables,
        )

        # If IPOPT ended with non-zero status but residuals already satisfy target
        # tolerance, restart from current point once to convert status into a
        # clean converged run.
        if (
            not bool(export_solution.converged)
            and float(export_solution.final_residual) <= float(self.solve_tolerance)
            and self.method == "ipopt"
        ):
            export_solver, export_solution, export_validation = self._solve_state(
                export_state,
                initial_vars=export_solution.variables,
            )
            export_validation = dict(export_validation)
            export_validation["restart"] = {
                "enabled": True,
                "reason": "residual_below_tolerance_but_status_nonzero",
                "attempts": 1,
            }
        elif self._should_use_export_tax_homotopy() and not bool(export_solution.converged):
            (
                export_solver,
                export_solution,
                export_validation,
            ) = self._solve_export_tax_with_homotopy(
                base_state,
                initial_vars=export_solution.variables,
            )
        export_run = ScenarioSolveReport(
            scenario="export_tax",
            export_tax_multiplier=self.export_tax_multiplier,
            converged=bool(export_solution.converged),
            iterations=int(export_solution.iterations),
            final_residual=float(export_solution.final_residual),
            message=str(export_solution.message),
            key_indicators=self._key_indicators(export_solution.variables),
            validation=export_validation,
        )

        base_cmp = self._compare_solution_with_gams(
            solution_vars=base_solution.variables,
            solution_params=base_solver.params,
            gams_slice="base",
        )
        export_cmp = self._compare_solution_with_gams(
            solution_vars=export_solution.variables,
            solution_params=export_solver.params,
            gams_slice="sim1",
        )

        return {
            "config": {
                "sam_file": str(self.sam_file),
                "val_par_file": str(self.val_par_file) if self.val_par_file else None,
                "gams_results_gdx": str(self.gams_results_gdx),
                "gdxdump_bin": str(self._resolve_gdxdump_binary()),
                "dynamic_sets": self.dynamic_sets,
                "init_mode": self.init_mode,
                "method": self.method,
                "solve_tolerance": self.solve_tolerance,
                "max_iterations": self.max_iterations,
                "export_tax_multiplier": self.export_tax_multiplier,
                "export_tax_homotopy": self.export_tax_homotopy,
                "export_tax_homotopy_steps": self.export_tax_homotopy_steps,
                "compare_abs_tol": self.compare_abs_tol,
                "compare_rel_tol": self.compare_rel_tol,
            },
            "scenarios": {
                "base": {
                    "solve": base_run.__dict__,
                    "gams_comparison": base_cmp,
                },
                "export_tax": {
                    "solve": export_run.__dict__,
                    "gams_comparison": export_cmp,
                },
            },
        }

    def _calibrate_base_state(self) -> PEPModelState:
        if self.dynamic_sets:
            calibrator = PEPModelCalibratorDynamic(
                sam_file=self.sam_file,
                val_par_file=self.val_par_file,
            )
        else:
            calibrator = PEPModelCalibrator(
                sam_file=self.sam_file,
                val_par_file=self.val_par_file,
                dynamic_sets=False,
            )
        return calibrator.calibrate()

    def _solve_state(
        self,
        state: PEPModelState,
        *,
        initial_vars: PEPModelVariables | None = None,
        gams_slice: str = "base",
    ) -> tuple[PEPModelSolver, Any, dict[str, Any]]:
        if self.method == "ipopt" and not IPOPT_AVAILABLE:
            raise RuntimeError(
                "method='ipopt' requested but cyipopt is not available. "
                "Install with `uv sync --extra ipopt` or `uv pip install cyipopt`."
            )
        solver = PEPModelSolver(
            calibrated_state=state,
            tolerance=self.solve_tolerance,
            max_iterations=self.max_iterations,
            init_mode=self.init_mode,
            gams_results_gdx=self.gams_results_gdx,
            gams_results_slice=gams_slice,
            sam_file=self.sam_file,
            val_par_file=self.val_par_file,
            gdxdump_bin=str(self._resolve_gdxdump_binary()),
            initial_vars=initial_vars,
        )
        solution = solver.solve(method=self.method)
        validation = solver.validate_solution(solution)
        return solver, solution, validation

    def _should_use_export_tax_homotopy(self) -> bool:
        return (
            self.export_tax_homotopy
            and self.method == "ipopt"
            and self.init_mode == "excel"
            and self.export_tax_homotopy_steps > 0
            and abs(self.export_tax_multiplier - 1.0) > 1e-12
        )

    def _build_export_tax_homotopy_path(self) -> list[float]:
        n_steps = max(1, int(self.export_tax_homotopy_steps))
        return [
            1.0 + (self.export_tax_multiplier - 1.0) * (i / n_steps)
            for i in range(1, n_steps + 1)
        ]

    def _solve_export_tax_with_homotopy(
        self,
        base_state: PEPModelState,
        *,
        initial_vars: PEPModelVariables | None = None,
    ) -> tuple[PEPModelSolver, Any, dict[str, Any]]:
        step_reports: list[dict[str, Any]] = []
        prev_vars = initial_vars
        last_solver: PEPModelSolver | None = None
        last_solution: Any = None
        last_validation: dict[str, Any] | None = None

        for multiplier in self._build_export_tax_homotopy_path():
            step_state = self._clone_with_export_tax_shock(base_state, multiplier=multiplier)
            solver, solution, validation = self._solve_state(
                step_state,
                initial_vars=prev_vars,
                gams_slice="base",
            )
            step_reports.append(
                {
                    "multiplier": float(multiplier),
                    "converged": bool(solution.converged),
                    "iterations": int(solution.iterations),
                    "final_residual": float(solution.final_residual),
                    "message": str(solution.message),
                }
            )
            prev_vars = solution.variables
            last_solver = solver
            last_solution = solution
            last_validation = validation

        if last_solver is None or last_validation is None:
            raise RuntimeError("Homotopy path produced no solve steps.")

        last_validation = dict(last_validation)
        last_validation["homotopy"] = {
            "enabled": True,
            "steps": step_reports,
        }
        return last_solver, last_solution, last_validation

    @staticmethod
    def _key_indicators(vars_obj: PEPModelVariables) -> dict[str, float]:
        total_exports = float(sum(vars_obj.EXD.values()))
        total_imports = float(sum(vars_obj.IM.values()))
        return {
            "GDP_BP": float(vars_obj.GDP_BP),
            "GDP_MP": float(vars_obj.GDP_MP),
            "GDP_IB": float(vars_obj.GDP_IB),
            "GDP_FD": float(vars_obj.GDP_FD),
            "IT": float(vars_obj.IT),
            "CAB": float(vars_obj.CAB),
            "TIXT": float(vars_obj.TIXT),
            "TPRODN": float(vars_obj.TPRODN),
            "TPRCTS": float(vars_obj.TPRCTS),
            "total_exports": total_exports,
            "total_imports": total_imports,
            "trade_balance": total_exports - total_imports,
            "PIXCON": float(vars_obj.PIXCON),
            "e": float(vars_obj.e),
        }

    @staticmethod
    def _clone_with_export_tax_shock(state: PEPModelState, *, multiplier: float) -> PEPModelState:
        """Clone state and apply `ttix := ttix * multiplier` export-tax shock."""
        shocked = copy.deepcopy(state)
        ttix = shocked.trade.get("ttixO", {})
        shocked.trade["ttixO"] = {i: float(v) * float(multiplier) for i, v in ttix.items()}

        tixo = shocked.trade.get("TIXO", {})
        if isinstance(tixo, dict):
            shocked.trade["TIXO"] = {i: float(v) * float(multiplier) for i, v in tixo.items()}

        if isinstance(shocked.income, dict):
            if "TIXTO" in shocked.income and isinstance(shocked.trade.get("TIXO"), dict):
                shocked.income["TIXTO"] = float(sum(shocked.trade.get("TIXO", {}).values()))
            if {"TICTO", "TIMTO", "TIXTO"}.issubset(set(shocked.income.keys())):
                shocked.income["TPRCTSO"] = (
                    float(shocked.income.get("TICTO", 0.0))
                    + float(shocked.income.get("TIMTO", 0.0))
                    + float(shocked.income.get("TIXTO", 0.0))
                )
            required = {"YGKO", "TDHTO", "TDFTO", "TPRODNO", "TPRCTSO", "YGTRO"}
            if required.issubset(set(shocked.income.keys())):
                shocked.income["YGO"] = (
                    float(shocked.income.get("YGKO", 0.0))
                    + float(shocked.income.get("TDHTO", 0.0))
                    + float(shocked.income.get("TDFTO", 0.0))
                    + float(shocked.income.get("TPRODNO", 0.0))
                    + float(shocked.income.get("TPRCTSO", 0.0))
                    + float(shocked.income.get("YGTRO", 0.0))
                )

        return shocked

    def _resolve_gdxdump_binary(self) -> Path:
        """Resolve gdxdump path with robust fallbacks."""
        raw = str(self.gdxdump_bin).strip()
        if raw:
            candidate = Path(raw)
            if candidate.exists():
                return candidate
            resolved = shutil.which(raw)
            if resolved:
                return Path(resolved)

        fallback = Path(DEFAULT_GDXDUMP_BIN)
        if fallback.exists():
            return fallback

        resolved = shutil.which("gdxdump")
        if resolved:
            return Path(resolved)

        raise FileNotFoundError(
            "gdxdump binary not found. Set --gdxdump-bin to your GAMS gdxdump path."
        )

    @staticmethod
    def _gdxdump_records(gdxdump_bin: Path, gdx_file: Path, symbol: str) -> list[tuple[tuple[str, ...], float]]:
        out = subprocess.check_output(
            [str(gdxdump_bin), str(gdx_file), f"symb={symbol}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        rows: list[tuple[tuple[str, ...], float]] = []
        for raw in out.splitlines():
            line = raw.strip()
            if not line or line.startswith(("/", "Parameter ", "Set ", "*")):
                continue
            nums = _NUM_RE.findall(line)
            if not nums:
                continue
            labels = tuple(x.lower() for x in _LAB_RE.findall(line))
            value = float(nums[-1])
            rows.append((labels, value))
        return rows

    def _iter_slice_records(
        self,
        gdxdump_bin: Path,
        symbol: str,
        gams_slice: str,
    ) -> list[tuple[tuple[str, ...], float]]:
        wanted = gams_slice.lower()
        out: list[tuple[tuple[str, ...], float]] = []
        for labels, value in self._gdxdump_records(gdxdump_bin, self.gams_results_gdx, symbol):
            if labels and labels[-1] in _SCENARIOS:
                if labels[-1] != wanted:
                    continue
                out.append((labels[:-1], value))
                continue
            if wanted == "base":
                out.append((labels, value))
        return out

    def _compare_solution_with_gams(
        self,
        *,
        solution_vars: PEPModelVariables,
        solution_params: dict[str, Any],
        gams_slice: str,
    ) -> dict[str, Any]:
        gdxdump_bin = self._resolve_gdxdump_binary()
        symbols = [s["name"] for s in read_gdx(self.gams_results_gdx).get("symbols", []) if s["name"].startswith("val")]

        compared = 0
        missing = 0
        mismatches: list[dict[str, Any]] = []

        for symbol in symbols:
            for idx, gams_val in self._iter_slice_records(gdxdump_bin, symbol, gams_slice):
                py_val = get_solution_value(solution_vars, symbol, idx, solution_params)
                if py_val is None:
                    missing += 1
                    continue
                compared += 1
                abs_diff = abs(float(py_val) - float(gams_val))
                rel_diff = abs_diff / max(abs(float(gams_val)), abs(float(py_val)), 1.0)
                if abs_diff > self.compare_abs_tol and rel_diff > self.compare_rel_tol:
                    mismatches.append(
                        {
                            "symbol": symbol,
                            "key": list(idx),
                            "gams": float(gams_val),
                            "python": float(py_val),
                            "abs_diff": abs_diff,
                            "rel_diff": rel_diff,
                        }
                    )

        mismatches.sort(key=lambda x: x["abs_diff"], reverse=True)
        max_abs = max((m["abs_diff"] for m in mismatches), default=0.0)
        max_rel = max((m["rel_diff"] for m in mismatches), default=0.0)
        rms = (
            math.sqrt(sum(m["abs_diff"] ** 2 for m in mismatches) / len(mismatches))
            if mismatches
            else 0.0
        )

        return {
            "gams_slice": gams_slice.lower(),
            "compared": compared,
            "missing": missing,
            "mismatches": len(mismatches),
            "passed": len(mismatches) == 0,
            "compare_abs_tol": self.compare_abs_tol,
            "compare_rel_tol": self.compare_rel_tol,
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "rms_abs_diff": rms,
            "top_mismatches": mismatches[:30],
        }
