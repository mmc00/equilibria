#!/usr/bin/env python3
"""Run PEP dynamic-SAM solver for CRI dataset (GDX or Excel)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.templates.pep_calibration_unified_dynamic import (  # noqa: E402
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_model_solver import PEPModelSolver  # noqa: E402


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve CRI dynamic-SAM PEP model in Python")
    parser.add_argument(
        "--sam-file",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/data/SAM-CRI-gams-fixed.xlsx"),
    )
    parser.add_argument(
        "--val-par-file",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/data/VAL_PAR-CRI-gams.xlsx"),
    )
    parser.add_argument("--acc-gvt", type=str, default="gvt")
    parser.add_argument("--acc-row", type=str, default="row")
    parser.add_argument("--acc-td", type=str, default="td")
    parser.add_argument("--acc-ti", type=str, default="ti")
    parser.add_argument("--acc-tm", type=str, default="tm")
    parser.add_argument("--acc-tx", type=str, default="tx")
    parser.add_argument("--acc-inv", type=str, default="inv")
    parser.add_argument("--acc-vstk", type=str, default="vstk")
    parser.add_argument("--method", choices=["auto", "ipopt", "simple_iteration"], default="auto")
    parser.add_argument("--init-mode", choices=["strict_gams", "equation_consistent"], default="strict_gams")
    parser.add_argument(
        "--gams-results-gdx",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/scripts/Results.gdx"),
        help="GAMS Results.gdx used for strict_gams initial levels",
    )
    parser.add_argument(
        "--gams-results-slice",
        choices=["base", "sim1"],
        default="sim1",
        help="Scenario slice in Results.gdx to use for strict_gams levels",
    )
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--save-solution", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.sam_file.exists():
        print(f"SAM file not found: {args.sam_file}")
        return 1

    accounts = {
        "gvt": args.acc_gvt,
        "row": args.acc_row,
        "td": args.acc_td,
        "ti": args.acc_ti,
        "tm": args.acc_tm,
        "tx": args.acc_tx,
        "inv": args.acc_inv,
        "vstk": args.acc_vstk,
    }

    if args.sam_file.suffix.lower() == ".xlsx":
        calibrator = PEPModelCalibratorExcelDynamicSAM(
            sam_file=args.sam_file,
            val_par_file=args.val_par_file,
            accounts=accounts,
        )
    else:
        calibrator = PEPModelCalibratorDynamicSAM(
            sam_file=args.sam_file,
            val_par_file=args.val_par_file,
            accounts=accounts,
        )

    state = calibrator.calibrate()
    calibrator.print_report()

    solver = PEPModelSolver(
        calibrated_state=state,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        init_mode=args.init_mode,
        gams_results_gdx=args.gams_results_gdx,
        gams_results_slice=args.gams_results_slice,
    )
    solution = solver.solve(method=args.method)

    print(solution.summary())
    validation = solver.validate_solution(solution)
    print(f"RMS Residual: {validation['rms_residual']:.2e}")
    print(f"Max Residual: {validation['max_residual']:.2e}")

    if args.save_solution:
        import json

        args.save_solution.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_solution, "w") as f:
            json.dump(
                {
                    "converged": solution.converged,
                    "iterations": solution.iterations,
                    "final_residual": solution.final_residual,
                    "message": solution.message,
                    "validation": validation,
                },
                f,
                indent=2,
            )
        print(f"Saved solution to: {args.save_solution}")

    return 0 if solution.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
