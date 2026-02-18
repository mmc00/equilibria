"""Example 11: PEP GDP comparison using original SAM vs CRI SAM.

This example runs the Python PEP model on two datasets:
1) Original PEP2 SAM (Excel)
2) CRI SAM (Excel, dynamic SAM accounts)

For each run it reports GDP indicators and a demand-side decomposition.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from equilibria.templates.pep_calibration_unified_dynamic import PEPModelCalibratorExcelDynamicSAM
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_model_solver import PEPModelSolver


ROOT = Path(__file__).resolve().parents[1]
PEP2_ROOT = ROOT / "src/equilibria/templates/reference/pep2"
PEP2_DATA = PEP2_ROOT / "data"
PEP2_SCRIPTS = PEP2_ROOT / "scripts"
GDXDUMP_BIN = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump")


def _sum_private_consumption(vars_obj: Any, sets: dict[str, list[str]]) -> float:
    total = 0.0
    for i in sets.get("I", []):
        c_i = sum(vars_obj.C.get((i, h), 0.0) for h in sets.get("H", []))
        total += vars_obj.PC.get(i, 0.0) * c_i
    return total


def _sum_government_consumption(vars_obj: Any, sets: dict[str, list[str]]) -> float:
    return sum(vars_obj.PC.get(i, 0.0) * vars_obj.CG.get(i, 0.0) for i in sets.get("I", []))


def _sum_inventory(vars_obj: Any, sets: dict[str, list[str]]) -> float:
    return sum(vars_obj.PC.get(i, 0.0) * vars_obj.VSTK.get(i, 0.0) for i in sets.get("I", []))


def _sum_exports(vars_obj: Any, sets: dict[str, list[str]], params: dict[str, Any]) -> float:
    exdo0 = params.get("EXDO0", params.get("EXDO", {}))
    return sum(
        vars_obj.PE_FOB.get(i, 0.0) * vars_obj.EXD.get(i, 0.0)
        for i in sets.get("I", [])
        if abs(exdo0.get(i, 0.0)) > 1e-12
    )


def _sum_imports(vars_obj: Any, sets: dict[str, list[str]], params: dict[str, Any]) -> float:
    imo0 = params.get("IMO0", {})
    return sum(
        vars_obj.PWM.get(i, 0.0) * vars_obj.e * vars_obj.IM.get(i, 0.0)
        for i in sets.get("I", [])
        if abs(imo0.get(i, 0.0)) > 1e-12
    )


def _extract_report(name: str, solver: PEPModelSolver, vars_obj: Any, converged: bool, iterations: int, message: str) -> dict[str, Any]:
    priv = _sum_private_consumption(vars_obj, solver.sets)
    gvt = _sum_government_consumption(vars_obj, solver.sets)
    inv = vars_obj.GFCF
    vstk = _sum_inventory(vars_obj, solver.sets)
    exp = _sum_exports(vars_obj, solver.sets, solver.params)
    imp = _sum_imports(vars_obj, solver.sets, solver.params)
    gdp_fd_calc = priv + gvt + inv + vstk + exp - imp

    return {
        "dataset": name,
        "solver": {
            "converged": converged,
            "iterations": iterations,
            "message": message,
            "init_mode": solver.init_mode,
            "gams_slice": solver.gams_results_slice,
        },
        "gdp_levels": {
            "GDP_BP": vars_obj.GDP_BP,
            "GDP_MP": vars_obj.GDP_MP,
            "GDP_IB": vars_obj.GDP_IB,
            "GDP_FD": vars_obj.GDP_FD,
            "GDP_BP_REAL": vars_obj.GDP_BP_REAL,
            "GDP_MP_REAL": vars_obj.GDP_MP_REAL,
        },
        "demand_side_components": {
            "private_consumption": priv,
            "government_consumption": gvt,
            "gross_fixed_capital_formation": inv,
            "inventories": vstk,
            "exports_fob": exp,
            "imports_cif": imp,
            "gdp_fd_rebuilt": gdp_fd_calc,
            "gdp_fd_gap": vars_obj.GDP_FD - gdp_fd_calc,
        },
    }


def _read_gdx_scalar(gdx_file: Path, symbol: str, label: str) -> float | None:
    if not gdx_file.exists() or not GDXDUMP_BIN.exists():
        return None
    try:
        out = subprocess.check_output(
            [str(GDXDUMP_BIN), str(gdx_file), f"symb={symbol}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return None

    pat = re.compile(r"'([^']+)'\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
    values = {m.group(1).lower(): float(m.group(2)) for m in pat.finditer(out)}
    return values.get(label.lower())


def _pick_init_mode(state: Any, gdx_file: Path, desired_slice: str) -> tuple[str, str]:
    """Choose strict_gams only when GDX slice exists and matches SAM scale."""
    gdx_gdp_bp = _read_gdx_scalar(gdx_file, "valGDP_BP", desired_slice)
    sam_gdp_bp = float(state.gdp.get("GDP_BPO", 0.0))
    if gdx_gdp_bp is None:
        return "equation_consistent", "GDX slice missing; fallback to equation_consistent"
    if abs(sam_gdp_bp) <= 1e-12:
        return "strict_gams", "Using strict_gams (SAM GDP baseline is zero)"

    rel_gap = abs(gdx_gdp_bp - sam_gdp_bp) / abs(sam_gdp_bp)
    if rel_gap > 0.2:
        return (
            "equation_consistent",
            f"GDX slice scale mismatch (gdx={gdx_gdp_bp:.6g}, sam={sam_gdp_bp:.6g}); fallback to equation_consistent",
        )
    return "strict_gams", "Using strict_gams (GDX slice matches SAM scale)"


def _run_original() -> dict[str, Any]:
    calibrator = PEPModelCalibratorExcel(
        sam_file=PEP2_DATA / "SAM-V2_0.xls",
        val_par_file=PEP2_DATA / "VAL_PAR.xlsx",
    )
    state = calibrator.calibrate()
    init_mode, reason = _pick_init_mode(state, PEP2_SCRIPTS / "Results.gdx", "base")
    solver = PEPModelSolver(
        calibrated_state=state,
        tolerance=1e-8,
        max_iterations=200,
        init_mode=init_mode,
        gams_results_gdx=PEP2_SCRIPTS / "Results.gdx",
        gams_results_slice="base",
    )
    if init_mode == "strict_gams":
        result = solver.solve(method="ipopt")
        vars_obj = result.variables
        converged = result.converged
        iterations = result.iterations
        message = result.message
    else:
        vars_obj = solver._create_initial_guess()
        converged = False
        iterations = 0
        message = "Reported from equation_consistent initialization (no IPOPT solve)"
    rep = _extract_report("pep2_original_sam", solver, vars_obj, converged, iterations, message)
    rep["solver"]["init_reason"] = reason
    return rep


def _run_cri() -> dict[str, Any]:
    calibrator = PEPModelCalibratorExcelDynamicSAM(
        sam_file=PEP2_DATA / "SAM-CRI-gams-fixed.xlsx",
        val_par_file=PEP2_DATA / "VAL_PAR-CRI-gams.xlsx",
    )
    state = calibrator.calibrate()
    init_mode, reason = _pick_init_mode(state, PEP2_SCRIPTS / "Results.gdx", "sim1")
    solver = PEPModelSolver(
        calibrated_state=state,
        tolerance=1e-8,
        max_iterations=200,
        init_mode=init_mode,
        gams_results_gdx=PEP2_SCRIPTS / "Results.gdx",
        gams_results_slice="sim1",
    )
    if init_mode == "strict_gams":
        result = solver.solve(method="ipopt")
        vars_obj = result.variables
        converged = result.converged
        iterations = result.iterations
        message = result.message
    else:
        vars_obj = solver._create_initial_guess()
        converged = False
        iterations = 0
        message = "Reported from equation_consistent initialization (no IPOPT solve)"
    rep = _extract_report("pep2_cri_sam", solver, vars_obj, converged, iterations, message)
    rep["solver"]["init_reason"] = reason
    return rep


def _print_block(rep: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"Dataset: {rep['dataset']}")
    print(f"Converged: {rep['solver']['converged']} | Iterations: {rep['solver']['iterations']}")
    print(f"Init mode: {rep['solver']['init_mode']} ({rep['solver'].get('init_reason', 'n/a')})")
    print(f"Message: {rep['solver']['message']}")
    print("-" * 78)
    print("GDP Levels")
    for k, v in rep["gdp_levels"].items():
        print(f"  {k:14s}: {v:,.6f}")
    print("-" * 78)
    print("Demand-side decomposition")
    for k, v in rep["demand_side_components"].items():
        print(f"  {k:30s}: {v:,.6f}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PEP GDP comparison: original SAM vs CRI SAM.")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    reports = [_run_original(), _run_cri()]
    for rep in reports:
        _print_block(rep)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump({"reports": reports}, f, indent=2)
        print(f"Saved JSON report to: {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
