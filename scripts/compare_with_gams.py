#!/usr/bin/env python3
"""
Compare solver solution with GAMS baseline.

This script runs the solver and compares results with GAMS baseline values.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_model_solver import PEPModelSolver

def main():
    print("=" * 70)
    print("PEP MODEL SOLUTION vs GAMS BASELINE COMPARISON")
    print("=" * 70)
    print()
    
    # Calibrate model
    print("Step 1: Calibrating model...")
    sam_file = Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.gdx")
    calibrator = PEPModelCalibrator(sam_file=sam_file)
    state = calibrator.calibrate()
    
    # GAMS Baseline values (from calibration - these are the correct equilibrium values)
    gams_baseline = {
        "GDP_BP": state.gdp.get("GDP_BPO", 46707.0),
        "GDP_MP": state.gdp.get("GDP_MPO", 48628.0),
        "GDP_IB": state.gdp.get("GDP_IBO", 93269.0),  # Note: This has issues
        "GDP_FD": state.gdp.get("GDP_FDO", 50536.48),
        "total_consumption": sum(state.consumption.get("CO", {}).values()),
        "total_investment": sum(state.consumption.get("INVO", {}).values()),
        "total_exports": sum(state.trade.get("EXDO", {}).values()),
        "total_imports": sum(state.trade.get("IMO", {}).values()),
        "trade_balance": sum(state.trade.get("EXDO", {}).values()) - sum(state.trade.get("IMO", {}).values()),
        "government_spending": state.consumption.get("GO", 0),
    }
    
    print(f"✓ Model calibrated")
    print()
    
    # Solve model
    print("Step 2: Solving model with IPOPT...")
    solver = PEPModelSolver(
        calibrated_state=state,
        tolerance=1e-8,
        max_iterations=200
    )
    
    solution = solver.solve(method="ipopt")
    
    print(f"✓ Solver finished: {solution.message}")
    print()
    
    # Extract solution values
    vars = solution.variables
    solver_results = {
        "GDP_BP": vars.GDP_BP,
        "GDP_MP": vars.GDP_MP,
        "GDP_IB": vars.GDP_IB,
        "GDP_FD": vars.GDP_FD,
        "total_consumption": sum(vars.CTH.values()),
        "total_investment": vars.IT,
        "total_exports": sum(vars.EXD.values()),
        "total_imports": sum(vars.IM.values()),
        "trade_balance": sum(vars.EXD.values()) - sum(vars.IM.values()),
        "government_spending": vars.G,
    }
    
    # Compare results
    print("=" * 70)
    print("COMPARISON: Solver vs GAMS Baseline")
    print("=" * 70)
    print()
    print(f"{'Variable':<25} {'GAMS Baseline':>18} {'Solver Result':>18} {'Difference':>15} {'Error %':>10}")
    print("-" * 90)
    
    all_close = True
    for key in gams_baseline:
        gams_val = gams_baseline[key]
        solver_val = solver_results[key]
        
        diff = solver_val - gams_val
        if abs(gams_val) > 1:
            pct_error = (diff / gams_val) * 100
        else:
            pct_error = 0
        
        status = "✓" if abs(pct_error) < 1 else "✗"
        if abs(pct_error) >= 1:
            all_close = False
            
        print(f"{status} {key:<23} {gams_val:>18.2f} {solver_val:>18.2f} {diff:>15.2f} {pct_error:>9.2f}%")
    
    print("-" * 90)
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Solver converged: {solution.converged}")
    print(f"Iterations: {solution.iterations}")
    print(f"Final residual: {solution.final_residual:.2e}")
    print()
    
    if all_close:
        print("✓ ALL CHECKS PASSED - Solver matches GAMS baseline within 1%")
    else:
        print("✗ SOME CHECKS FAILED - Significant differences from GAMS baseline")
        print()
        print("Possible causes:")
        print("  - Solver needs more iterations")
        print("  - Variable initialization issues")
        print("  - Scaling problems in optimization")
        print("  - Equation implementation differences")
    
    print()
    print("=" * 70)
    
    # Save detailed comparison
    comparison = {
        "gams_baseline": gams_baseline,
        "solver_results": solver_results,
        "converged": solution.converged,
        "iterations": solution.iterations,
        "residual": solution.final_residual,
        "all_close": all_close,
    }
    
    output_file = Path("output/gams_comparison.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Detailed comparison saved to: {output_file}")

if __name__ == "__main__":
    main()
