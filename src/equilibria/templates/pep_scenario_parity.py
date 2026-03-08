"""PEP pep2 scenario runners with GAMS comparison."""

from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from equilibria.simulations import pep_compare as _pep_compare
from equilibria.templates.pep_calibration_unified import (
    PEPModelCalibrator,
    PEPModelState,
)
from equilibria.templates.pep_calibration_unified_dynamic import (
    PEPModelCalibratorDynamic,
)
from equilibria.templates.pep_model_equations import PEPModelVariables
from equilibria.templates.pep_model_solver import IPOPT_AVAILABLE, PEPModelSolver

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAM_FILE = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
DEFAULT_VAL_PAR_FILE = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"
DEFAULT_RESULTS_GDX = REPO_ROOT / "src/equilibria/templates/reference/pep2/scripts/Results.gdx"
DEFAULT_GDXDUMP_BIN = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"


def get_solution_value(
    vars_obj: PEPModelVariables,
    symbol: str,
    idx: tuple[str, ...],
    params: dict[str, Any],
) -> float | None:
    """Compatibility wrapper re-exporting the new compare helper."""
    return _pep_compare.get_solution_value(vars_obj, symbol, idx, params)


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
    import_price_commodity: str | None = None
    import_price_multiplier: float | None = None
    government_spending_multiplier: float | None = None

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
                "equation_mode": "gams_strict",
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
        return _pep_compare.key_indicators(vars_obj)

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

    @staticmethod
    def _clone_with_import_price_shock(
        state: PEPModelState,
        *,
        commodity: str,
        multiplier: float,
    ) -> PEPModelState:
        """Clone state and apply `PWMO(commodity) := PWMO(commodity) * multiplier`."""
        shocked = copy.deepcopy(state)
        pwmo = shocked.trade.get("PWMO", {})
        if not isinstance(pwmo, dict):
            pwmo = {}
        shocked_pwmo = dict(pwmo)
        base_val = float(shocked_pwmo.get(commodity, 1.0))
        shocked_pwmo[commodity] = base_val * float(multiplier)
        shocked.trade["PWMO"] = shocked_pwmo
        return shocked

    @staticmethod
    def _clone_with_import_price_all_shock(
        state: PEPModelState,
        *,
        multiplier: float,
    ) -> PEPModelState:
        """Clone state and apply `PWMO(i) := PWMO(i) * multiplier` for all i."""
        shocked = copy.deepcopy(state)
        pwmo = shocked.trade.get("PWMO", {})
        if not isinstance(pwmo, dict):
            pwmo = {}
        shocked_pwmo = {i: float(v) * float(multiplier) for i, v in pwmo.items()}
        shocked.trade["PWMO"] = shocked_pwmo
        return shocked

    @staticmethod
    def _clone_with_government_spending_shock(
        state: PEPModelState,
        *,
        multiplier: float,
    ) -> PEPModelState:
        """Clone state and apply `GO := GO * multiplier`."""
        shocked = copy.deepcopy(state)
        go = float(shocked.consumption.get("GO", 0.0))
        shocked.consumption["GO"] = go * float(multiplier)
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

    def _compare_solution_with_gams(
        self,
        *,
        solution_vars: PEPModelVariables,
        solution_params: dict[str, Any],
        gams_slice: str,
    ) -> dict[str, Any]:
        return _pep_compare.compare_with_gams(
            solution_vars=solution_vars,
            solution_params=solution_params,
            gams_results_gdx=self.gams_results_gdx,
            gams_slice=gams_slice.lower(),
            abs_tol=self.compare_abs_tol,
            rel_tol=self.compare_rel_tol,
            gdxdump_bin=str(self._resolve_gdxdump_binary()),
        )


class PEPExportTaxParityRunner(PEPScenarioParityRunner):
    """Explicit API runner for BASE + EXPORT_TAX parity."""

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
        super().__init__(
            sam_file=sam_file,
            val_par_file=val_par_file,
            gams_results_gdx=gams_results_gdx,
            gdxdump_bin=gdxdump_bin,
            dynamic_sets=dynamic_sets,
            init_mode=init_mode,
            method=method,
            solve_tolerance=solve_tolerance,
            max_iterations=max_iterations,
            export_tax_multiplier=export_tax_multiplier,
            export_tax_homotopy=export_tax_homotopy,
            export_tax_homotopy_steps=export_tax_homotopy_steps,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
        )

    def run(self) -> dict[str, Any]:
        return super().run()


class PEPImportPriceParityRunner(PEPScenarioParityRunner):
    """API runner for BASE + IMPORT_PRICE parity."""

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
        import_price_commodity: str = "agr",
        import_price_multiplier: float = 1.25,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            sam_file=sam_file,
            val_par_file=val_par_file,
            gams_results_gdx=gams_results_gdx,
            gdxdump_bin=gdxdump_bin,
            dynamic_sets=dynamic_sets,
            init_mode=init_mode,
            method=method,
            solve_tolerance=solve_tolerance,
            max_iterations=max_iterations,
            export_tax_multiplier=1.0,
            export_tax_homotopy=False,
            export_tax_homotopy_steps=0,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
        )
        self.import_price_commodity = str(import_price_commodity).strip().lower()
        self.import_price_multiplier = float(import_price_multiplier)

    def run(self) -> dict[str, Any]:
        """Execute BASE and IMPORT_PRICE scenarios and compare to GAMS."""
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

        import_state = self._clone_with_import_price_shock(
            base_state,
            commodity=self.import_price_commodity,
            multiplier=self.import_price_multiplier,
        )
        import_solver, import_solution, import_validation = self._solve_state(
            import_state,
            initial_vars=base_solution.variables,
            gams_slice="sim1",
        )

        if (
            not bool(import_solution.converged)
            and float(import_solution.final_residual) <= float(self.solve_tolerance)
            and self.method == "ipopt"
        ):
            import_solver, import_solution, import_validation = self._solve_state(
                import_state,
                initial_vars=import_solution.variables,
                gams_slice="sim1",
            )
            import_validation = dict(import_validation)
            import_validation["restart"] = {
                "enabled": True,
                "reason": "residual_below_tolerance_but_status_nonzero",
                "attempts": 1,
            }

        scenario_name = f"import_price_{self.import_price_commodity}"
        import_run = ScenarioSolveReport(
            scenario=scenario_name,
            export_tax_multiplier=1.0,
            import_price_commodity=self.import_price_commodity,
            import_price_multiplier=self.import_price_multiplier,
            converged=bool(import_solution.converged),
            iterations=int(import_solution.iterations),
            final_residual=float(import_solution.final_residual),
            message=str(import_solution.message),
            key_indicators=self._key_indicators(import_solution.variables),
            validation=import_validation,
        )

        base_cmp = self._compare_solution_with_gams(
            solution_vars=base_solution.variables,
            solution_params=base_solver.params,
            gams_slice="base",
        )
        import_cmp = self._compare_solution_with_gams(
            solution_vars=import_solution.variables,
            solution_params=import_solver.params,
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
                "equation_mode": "gams_strict",
                "solve_tolerance": self.solve_tolerance,
                "max_iterations": self.max_iterations,
                "import_price_commodity": self.import_price_commodity,
                "import_price_multiplier": self.import_price_multiplier,
                "compare_abs_tol": self.compare_abs_tol,
                "compare_rel_tol": self.compare_rel_tol,
            },
            "scenarios": {
                "base": {
                    "solve": base_run.__dict__,
                    "gams_comparison": base_cmp,
                },
                scenario_name: {
                    "solve": import_run.__dict__,
                    "gams_comparison": import_cmp,
                },
            },
        }


class PEPGovernmentSpendingParityRunner(PEPScenarioParityRunner):
    """API runner for BASE + GOVERNMENT_SPENDING parity."""

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
        government_spending_multiplier: float = 1.2,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            sam_file=sam_file,
            val_par_file=val_par_file,
            gams_results_gdx=gams_results_gdx,
            gdxdump_bin=gdxdump_bin,
            dynamic_sets=dynamic_sets,
            init_mode=init_mode,
            method=method,
            solve_tolerance=solve_tolerance,
            max_iterations=max_iterations,
            export_tax_multiplier=1.0,
            export_tax_homotopy=False,
            export_tax_homotopy_steps=0,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
        )
        self.government_spending_multiplier = float(government_spending_multiplier)

    def run(self) -> dict[str, Any]:
        """Execute BASE and GOVERNMENT_SPENDING scenarios and compare to GAMS."""
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

        gov_state = self._clone_with_government_spending_shock(
            base_state,
            multiplier=self.government_spending_multiplier,
        )
        gov_solver, gov_solution, gov_validation = self._solve_state(
            gov_state,
            initial_vars=base_solution.variables,
            gams_slice="sim1",
        )

        if (
            not bool(gov_solution.converged)
            and float(gov_solution.final_residual) <= float(self.solve_tolerance)
            and self.method == "ipopt"
        ):
            gov_solver, gov_solution, gov_validation = self._solve_state(
                gov_state,
                initial_vars=gov_solution.variables,
                gams_slice="sim1",
            )
            gov_validation = dict(gov_validation)
            gov_validation["restart"] = {
                "enabled": True,
                "reason": "residual_below_tolerance_but_status_nonzero",
                "attempts": 1,
            }

        scenario_name = "government_spending"
        gov_run = ScenarioSolveReport(
            scenario=scenario_name,
            export_tax_multiplier=1.0,
            government_spending_multiplier=self.government_spending_multiplier,
            converged=bool(gov_solution.converged),
            iterations=int(gov_solution.iterations),
            final_residual=float(gov_solution.final_residual),
            message=str(gov_solution.message),
            key_indicators=self._key_indicators(gov_solution.variables),
            validation=gov_validation,
        )

        base_cmp = self._compare_solution_with_gams(
            solution_vars=base_solution.variables,
            solution_params=base_solver.params,
            gams_slice="base",
        )
        gov_cmp = self._compare_solution_with_gams(
            solution_vars=gov_solution.variables,
            solution_params=gov_solver.params,
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
                "equation_mode": "gams_strict",
                "solve_tolerance": self.solve_tolerance,
                "max_iterations": self.max_iterations,
                "government_spending_multiplier": self.government_spending_multiplier,
                "compare_abs_tol": self.compare_abs_tol,
                "compare_rel_tol": self.compare_rel_tol,
            },
            "scenarios": {
                "base": {
                    "solve": base_run.__dict__,
                    "gams_comparison": base_cmp,
                },
                scenario_name: {
                    "solve": gov_run.__dict__,
                    "gams_comparison": gov_cmp,
                },
            },
        }


class PEPImportShockParityRunner(PEPScenarioParityRunner):
    """API runner for BASE + IMPORT_SHOCK parity (all import prices)."""

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
        import_price_multiplier: float = 1.25,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            sam_file=sam_file,
            val_par_file=val_par_file,
            gams_results_gdx=gams_results_gdx,
            gdxdump_bin=gdxdump_bin,
            dynamic_sets=dynamic_sets,
            init_mode=init_mode,
            method=method,
            solve_tolerance=solve_tolerance,
            max_iterations=max_iterations,
            export_tax_multiplier=1.0,
            export_tax_homotopy=False,
            export_tax_homotopy_steps=0,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
        )
        self.import_price_multiplier = float(import_price_multiplier)

    def run(self) -> dict[str, Any]:
        """Execute BASE and IMPORT_SHOCK scenarios and compare to GAMS."""
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

        import_state = self._clone_with_import_price_all_shock(
            base_state,
            multiplier=self.import_price_multiplier,
        )
        import_solver, import_solution, import_validation = self._solve_state(
            import_state,
            initial_vars=base_solution.variables,
            gams_slice="sim1",
        )

        if (
            not bool(import_solution.converged)
            and float(import_solution.final_residual) <= float(self.solve_tolerance)
            and self.method == "ipopt"
        ):
            import_solver, import_solution, import_validation = self._solve_state(
                import_state,
                initial_vars=import_solution.variables,
                gams_slice="sim1",
            )
            import_validation = dict(import_validation)
            import_validation["restart"] = {
                "enabled": True,
                "reason": "residual_below_tolerance_but_status_nonzero",
                "attempts": 1,
            }

        scenario_name = "import_shock"
        import_run = ScenarioSolveReport(
            scenario=scenario_name,
            export_tax_multiplier=1.0,
            import_price_multiplier=self.import_price_multiplier,
            converged=bool(import_solution.converged),
            iterations=int(import_solution.iterations),
            final_residual=float(import_solution.final_residual),
            message=str(import_solution.message),
            key_indicators=self._key_indicators(import_solution.variables),
            validation=import_validation,
        )

        base_cmp = self._compare_solution_with_gams(
            solution_vars=base_solution.variables,
            solution_params=base_solver.params,
            gams_slice="base",
        )
        import_cmp = self._compare_solution_with_gams(
            solution_vars=import_solution.variables,
            solution_params=import_solver.params,
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
                "equation_mode": "gams_strict",
                "solve_tolerance": self.solve_tolerance,
                "max_iterations": self.max_iterations,
                "import_price_multiplier": self.import_price_multiplier,
                "compare_abs_tol": self.compare_abs_tol,
                "compare_rel_tol": self.compare_rel_tol,
            },
            "scenarios": {
                "base": {
                    "solve": base_run.__dict__,
                    "gams_comparison": base_cmp,
                },
                scenario_name: {
                    "solve": import_run.__dict__,
                    "gams_comparison": import_cmp,
                },
            },
        }
