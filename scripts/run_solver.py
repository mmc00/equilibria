#!/usr/bin/env python3
"""
Run PEP model solver.

This script solves the calibrated PEP model using the equation system.

Usage:
    python run_solver.py
    python run_solver.py --calibrated-state output/calibrated_state.json
    python run_solver.py --save-solution output/solution.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_model_solver import PEPModelSolver


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Solve calibrated PEP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate and solve in one step
  python run_solver.py
  
  # Use pre-calibrated state
  python run_solver.py --calibrated-state output/state.json
  
  # Save solution
  python run_solver.py --save-solution output/solution.json
  
  # Custom tolerance
  python run_solver.py --tolerance 1e-8
        """,
    )
    
    parser.add_argument(
        "--sam-file",
        type=Path,
        default=Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.gdx"),
        help="Path to SAM GDX file",
    )
    parser.add_argument(
        "--calibrated-state",
        type=Path,
        default=None,
        help="Path to pre-calibrated state JSON file (optional)",
    )
    parser.add_argument(
        "--save-solution",
        type=Path,
        default=None,
        help="Save solution to JSON file",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Convergence tolerance",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum iterations",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "ipopt", "simple_iteration"],
        help="Solver method (auto: try IPOPT first, fall back to simple iteration)",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="strict_gams",
        choices=["strict_gams", "equation_consistent"],
        help="Initialization mode for solver state",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("=" * 70)
    print("PEP-1-1_v2_1 MODEL SOLVER")
    print("=" * 70)
    print()
    
    # Get calibrated state
    if args.calibrated_state and args.calibrated_state.exists():
        print(f"Loading calibrated state from: {args.calibrated_state}")
        import json
        with open(args.calibrated_state, 'r') as f:
            state_data = json.load(f)
        
        # Recreate state object (simplified)
        from equilibria.templates.pep_calibration_unified import PEPModelState
        state = PEPModelState(**state_data)
        print("✓ Loaded calibrated state")
    else:
        print("Running calibration first...")
        print()
        
        if not args.sam_file.exists():
            print(f"Error: SAM file not found: {args.sam_file}")
            return 1
        
        calibrator = PEPModelCalibrator(sam_file=args.sam_file)
        state = calibrator.calibrate()
        calibrator.print_report()
        print()
    
    # Create solver and solve
    print("=" * 70)
    print("SOLVING MODEL")
    print("=" * 70)
    print()
    
    solver = PEPModelSolver(
        calibrated_state=state,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        init_mode=args.init_mode,
    )
    
    solution = solver.solve(method=args.method)
    
    # Print results
    print()
    print(solution.summary())
    print()
    
    # Validate
    validation = solver.validate_solution(solution)
    
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    print(f"\nRMS Residual: {validation['rms_residual']:.2e}")
    print(f"Max Residual: {validation['max_residual']:.2e}")
    
    print("\nChecks:")
    for check_name, check_result in validation['checks'].items():
        status = "✓" if check_result.get('passed', False) else "✗"
        print(f"  {status} {check_name}")
        if 'error' in check_result:
            print(f"      Error: {check_result['error']:.2e}")
    
    # Save solution if requested
    if args.save_solution:
        args.save_solution.parent.mkdir(parents=True, exist_ok=True)
        
        solution_data = {
            "converged": solution.converged,
            "iterations": solution.iterations,
            "final_residual": solution.final_residual,
            "message": solution.message,
            "validation": validation,
            "key_variables": {
                "GDP_BP": solution.variables.GDP_BP,
                "GDP_MP": solution.variables.GDP_MP,
                "GDP_IB": solution.variables.GDP_IB,
                "GDP_FD": solution.variables.GDP_FD,
                "total_consumption": sum(solution.variables.CTH.values()),
                "total_investment": solution.variables.IT,
                "trade_balance": sum(solution.variables.EXD.values()) - sum(solution.variables.IM.values()),
                "PIXCON": solution.variables.PIXCON,
                "exchange_rate": solution.variables.e,
            },
        }
        
        import json
        with open(args.save_solution, 'w') as f:
            json.dump(solution_data, f, indent=2)
        
        print(f"\n✓ Solution saved to: {args.save_solution}")
    
    print()
    print("=" * 70)
    if solution.converged:
        print("✓ MODEL SOLVED SUCCESSFULLY")
    else:
        print("✗ MODEL SOLUTION INCOMPLETE")
    print("=" * 70)
    
    return 0 if solution.converged else 1


if __name__ == "__main__":
    sys.exit(main())
