#!/usr/bin/env python3
"""
Run PEP model solver using SAM loaded directly from Excel.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.templates.pep_calibration_unified import PEPModelState
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_model_solver import PEPModelSolver


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve calibrated PEP model (Excel SAM)")
    parser.add_argument(
        "--sam-file",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/data/SAM-V2_0.xls"),
        help="Path to SAM Excel file (.xls/.xlsx)",
    )
    parser.add_argument(
        "--val-par-file",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"),
        help="Path to VAL_PAR file (.xlsx/.gdx)",
    )
    parser.add_argument("--calibrated-state", type=Path, default=None)
    parser.add_argument("--save-solution", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "ipopt", "simple_iteration"],
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="strict_gams",
        choices=["strict_gams", "equation_consistent"],
    )
    parser.add_argument(
        "--dynamic-sets",
        action="store_true",
        help="Derive model sets dynamically from SAM instead of using template defaults",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    print("=" * 70)
    print("PEP-1-1_v2_1 MODEL SOLVER (EXCEL SAM)")
    print("=" * 70)
    print()

    if args.calibrated_state and args.calibrated_state.exists():
        import json

        with open(args.calibrated_state, "r") as f:
            state_data = json.load(f)
        state = PEPModelState(**state_data)
        print(f"✓ Loaded calibrated state: {args.calibrated_state}")
    else:
        if not args.sam_file.exists():
            print(f"Error: SAM file not found: {args.sam_file}")
            return 1
        print(f"Running calibration from Excel SAM: {args.sam_file}")
        calibrator = PEPModelCalibratorExcel(
            sam_file=args.sam_file,
            val_par_file=args.val_par_file if args.val_par_file else None,
            dynamic_sets=args.dynamic_sets,
        )
        state = calibrator.calibrate()
        calibrator.print_report()

    print("\n" + "=" * 70)
    print("SOLVING MODEL")
    print("=" * 70)

    solver = PEPModelSolver(
        calibrated_state=state,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        init_mode=args.init_mode,
    )
    solution = solver.solve(method=args.method)

    print()
    print(solution.summary())
    validation = solver.validate_solution(solution)
    print(f"\nRMS Residual: {validation['rms_residual']:.2e}")
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
        print(f"✓ Solution saved to: {args.save_solution}")

    return 0 if solution.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
