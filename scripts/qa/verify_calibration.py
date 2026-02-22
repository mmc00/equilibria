#!/usr/bin/env python3
"""
Verify that calibrated values satisfy equilibrium conditions.

For a CGE model, the calibrated values should be the equilibrium solution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_model_solver import PEPModelSolver

def main():
    print("=" * 70)
    print("VERIFYING CALIBRATED VALUES ARE EQUILIBRIUM")
    print("=" * 70)
    print()
    
    # Calibrate model
    print("Step 1: Calibrating model...")
    sam_file = Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.gdx")
    calibrator = PEPModelCalibrator(sam_file=sam_file)
    state = calibrator.calibrate()
    print(f"✓ Model calibrated")
    print()
    
    # Create solver just to check residuals at calibrated values
    print("Step 2: Computing residuals at calibrated values...")
    solver = PEPModelSolver(calibrated_state=state)
    
    # Create initial guess (which are the calibrated values)
    vars = solver._create_initial_guess()
    
    # Calculate all residuals
    residuals = solver.equations.calculate_all_residuals(vars)
    
    # Statistics
    residual_values = list(residuals.values())
    rms = np.sqrt(np.mean([r**2 for r in residual_values]))
    max_resid = max(abs(r) for r in residual_values)
    
    print(f"Number of equations: {len(residuals)}")
    print(f"RMS residual: {rms:.2e}")
    print(f"Max residual: {max_resid:.2e}")
    print()
    
    # Show worst residuals
    print("Top 10 worst residuals:")
    sorted_resids = sorted(residuals.items(), key=lambda x: abs(x[1]), reverse=True)
    for eq_name, resid in sorted_resids[:10]:
        print(f"  {eq_name:<30}: {resid:>15.2e}")
    
    print()
    
    # Check if calibrated values satisfy equilibrium
    if rms < 1e-3:
        print("✓ CALIBRATED VALUES SATISFY EQUILIBRIUM (RMS < 1e-3)")
        print("  The solver should find that no optimization is needed.")
    elif rms < 1:
        print("⚠ CALIBRATED VALUES CLOSE TO EQUILIBRIUM (RMS < 1)")
        print("  Minor solver adjustments should achieve convergence.")
    else:
        print("✗ CALIBRATED VALUES DO NOT SATISFY EQUILIBRIUM")
        print("  There may be issues with:")
        print("    - Equation implementation")
        print("    - Parameter extraction")
        print("    - Variable initialization")
    
    print()
    print("=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
For a CGE model, the calibration process ensures that the base year data
satisfies all model equations. The calibrated values ARE the equilibrium
solution. The solver's job is to:

1. Verify that the calibrated values satisfy all equations
2. Find new equilibrium when parameters change (shocks)

If the residuals at calibrated values are large, there may be bugs in:
- Equation implementation (pep_model_equations.py)
- Parameter extraction from calibrated state
- Variable initialization

The solver should NOT need to run if residuals are already near zero.
""")

if __name__ == "__main__":
    main()
