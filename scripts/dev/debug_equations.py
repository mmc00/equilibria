#!/usr/bin/env python3
"""
Debug script to find equation discrepancies.

This script analyzes the top equations with large residuals and compares
the calculated values vs expected values from GAMS formulas.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_model_solver import PEPModelSolver

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM_FILE = REPO_ROOT / "src" / "equilibria" / "templates" / "reference" / "pep2" / "data" / "SAM-V2_0.gdx"

def debug_equation(solver, vars, eq_name, expected_calc, actual_value):
    """Debug a single equation."""
    residual = actual_value - expected_calc
    print(f"\n{eq_name}:")
    print(f"  Expected: {expected_calc:,.2f}")
    print(f"  Actual:   {actual_value:,.2f}")
    print(f"  Residual: {residual:,.2f}")
    return residual

def main():
    print("=" * 70)
    print("EQUATION DEBUGGING - Finding discrepancies")
    print("=" * 70)
    print()
    
    # Calibrate
    print("Step 1: Calibrating model...")
    sam_file = DEFAULT_SAM_FILE
    calibrator = PEPModelCalibrator(sam_file=sam_file)
    state = calibrator.calibrate()
    
    solver = PEPModelSolver(calibrated_state=state)
    vars = solver._create_initial_guess()
    
    print(f"✓ Model calibrated")
    print()
    
    # Get all residuals
    print("Step 2: Computing residuals...")
    residuals = solver.equations.calculate_all_residuals(vars)
    
    # Sort by absolute value
    sorted_resids = sorted(residuals.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Total equations: {len(residuals)}")
    print(f"Top 10 worst residuals:")
    for eq_name, resid in sorted_resids[:10]:
        print(f"  {eq_name:<30}: {resid:>15,.2e}")
    print()
    
    # Debug top equations
    print("=" * 70)
    print("DETAILED DEBUG OF TOP EQUATIONS")
    print("=" * 70)
    
    sets = solver.sets
    params = solver.params
    
    # EQ63 - CES between imports and local (largest residual)
    print("\n" + "=" * 70)
    print("EQ63 - CES between imports and local production")
    print("=" * 70)
    print("GAMS: Q(i) = B_M(i)*{[beta_M(i)*IM(i)**(-rho_M(i))]$IMO(i) + [(1-beta_M(i))*DD(i)**(-rho_M(i))]$DDO(i)}**(-1/rho_M(i))")
    
    for i in sets.get("I", [])[:2]:  # First 2 commodities
        print(f"\nCommodity: {i}")
        
        rho_m = params.get("rho_M", {}).get(i, -0.5)
        B_m = params.get("B_M", {}).get(i, 1)
        beta_m = params.get("beta_M", {}).get(i, 0.5)
        
        im = vars.IM.get(i, 0)
        dd = vars.DD.get(i, 0)
        q_actual = vars.Q.get(i, 0)
        
        print(f"  Parameters:")
        print(f"    rho_M: {rho_m:.4f}")
        print(f"    B_M: {B_m:.4f}")
        print(f"    beta_M: {beta_m:.4f}")
        
        print(f"  Variables:")
        print(f"    IM: {im:,.2f}")
        print(f"    DD: {dd:,.2f}")
        print(f"    Q (actual): {q_actual:,.2f}")
        
        # Calculate expected Q
        if im > 0 or dd > 0:
            try:
                term1 = beta_m * (im ** (-rho_m)) if im > 0 else 0
                term2 = (1 - beta_m) * (dd ** (-rho_m)) if dd > 0 else 0
                expected_q = B_m * (term1 + term2) ** (-1 / rho_m)
                
                print(f"  Calculation:")
                print(f"    term1 (beta_M * IM^(-rho_M)): {term1:,.2f}")
                print(f"    term2 ((1-beta_M) * DD^(-rho_M)): {term2:,.2f}")
                print(f"    sum: {term1 + term2:,.2f}")
                print(f"    sum^(-1/rho_M): {(term1 + term2) ** (-1/rho_m):,.2f}")
                print(f"    B_M * ...: {expected_q:,.2f}")
                
                residual = q_actual - expected_q
                print(f"  Residual: {residual:,.2f}")
                
                if abs(residual) > 1e6:
                    print(f"  ⚠ LARGE RESIDUAL - Check formula or parameters")
            except Exception as e:
                print(f"  Error in calculation: {e}")
    
    # EQ43 - Government savings
    print("\n" + "=" * 70)
    print("EQ43 - Government savings")
    print("=" * 70)
    print("GAMS: SG = YG - SUM[agng,TR(agng,'gvt')] - G")
    
    yg = vars.YG
    g = vars.G
    sg_actual = vars.SG
    
    # Sum transfers TO government
    tr_to_govt = sum(vars.TR.get((agng, "gvt"), 0) for agng in sets.get("AGNG", []))
    
    print(f"\nVariables:")
    print(f"  YG (government income): {yg:,.2f}")
    print(f"  G (government spending): {g:,.2f}")
    print(f"  TR to govt: {tr_to_govt:,.2f}")
    print(f"  SG (actual): {sg_actual:,.2f}")
    
    expected_sg = yg - tr_to_govt - g
    residual_sg = sg_actual - expected_sg
    
    print(f"\nCalculation:")
    print(f"  Expected SG: {yg:,.2f} - {tr_to_govt:,.2f} - {g:,.2f} = {expected_sg:,.2f}")
    print(f"  Residual: {residual_sg:,.2f}")
    
    if abs(residual_sg) > 1000:
        print(f"\n  ⚠ ISSUE: Government spending G is {g:,.2f} but should be:")
        print(f"     - GO from calibration: {state.consumption.get('GO', 0):,.2f}")
        print(f"     - CGO values: {state.consumption.get('CGO', {})}")
    
    # EQ27-28 - Tax totals
    print("\n" + "=" * 70)
    print("EQ27 - Total indirect taxes on wages")
    print("=" * 70)
    print("GAMS: TIWT = SUM[(l,j)$LDO(l,j),TIW(l,j)]")
    
    tiwt_actual = vars.TIWT
    tiwt_calc = sum(vars.TIW.get((l, j), 0) for l in sets.get("L", []) for j in sets.get("J", []))
    
    print(f"  TIWT (actual): {tiwt_actual:,.2f}")
    print(f"  SUM TIW: {tiwt_calc:,.2f}")
    print(f"  Residual: {tiwt_actual - tiwt_calc:,.2f}")
    
    # Show some TIW values
    print(f"\n  Individual TIW values:")
    for l in sets.get("L", []):
        for j in sets.get("J", []):
            val = vars.TIW.get((l, j), 0)
            if val != 0:
                print(f"    TIW({l},{j}): {val:,.2f}")
    
    print("\n" + "=" * 70)
    print("EQ28 - Total indirect taxes on capital")
    print("=" * 70)
    print("GAMS: TIKT = SUM[(k,j)$KDO(k,j),TIK(k,j)]")
    
    tikt_actual = vars.TIKT
    tikt_calc = sum(vars.TIK.get((k, j), 0) for k in sets.get("K", []) for j in sets.get("J", []))
    
    print(f"  TIKT (actual): {tikt_actual:,.2f}")
    print(f"  SUM TIK: {tikt_calc:,.2f}")
    print(f"  Residual: {tikt_actual - tikt_calc:,.2f}")
    
    # Check GDP_FD
    print("\n" + "=" * 70)
    print("EQ93 - GDP from final demand")
    print("=" * 70)
    print("GAMS: GDP_FD = SUM[i,PC(i)*(SUM[h,C(i,h)]+CG(i)+INV(i)+VSTK(i))] + SUM[i,PE_FOB(i)*EXD(i)] - SUM[i,PWM(i)*e*IM(i)]")
    
    gdp_fd_actual = vars.GDP_FD
    
    # Calculate
    domestic_demand = 0
    for i in sets.get("I", []):
        pc_i = vars.PC.get(i, 1.0)
        cons = sum(vars.C.get((i, h), 0) for h in sets.get("H", []))
        cg_i = vars.CG.get(i, 0)
        inv_i = vars.INV.get(i, 0)
        vst_i = vars.VSTK.get(i, 0)
        domestic_demand += pc_i * (cons + cg_i + inv_i + vst_i)
    
    exports = sum(vars.PE_FOB.get(i, 0) * vars.EXD.get(i, 0) for i in sets.get("I", []))
    imports = sum(vars.PWM.get(i, 0) * vars.e * vars.IM.get(i, 0) for i in sets.get("I", []))
    
    gdp_fd_calc = domestic_demand + exports - imports
    
    print(f"  Domestic demand: {domestic_demand:,.2f}")
    print(f"  Exports: {exports:,.2f}")
    print(f"  Imports: {imports:,.2f}")
    print(f"  GDP_FD (calculated): {gdp_fd_calc:,.2f}")
    print(f"  GDP_FD (actual): {gdp_fd_actual:,.2f}")
    print(f"  Residual: {gdp_fd_actual - gdp_fd_calc:,.2f}")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
1. EQ63 (CES imports): Large residuals likely due to:
   - IM and DD values being 0 or very small
   - Parameter values (rho_M, beta_M, B_M) not matching calibration
   
2. EQ43 (Govt savings): G (government spending) is 0 but should be:
   - Calculated from CGO (government consumption)
   - Bug in calibration or variable initialization
   
3. EQ27-28 (Tax totals): TIWT and TIKT values don't match sum of TIW/TIK
   - TIW and TIK might not be initialized correctly
   
4. EQ93 (GDP_FD): Large residual suggests:
   - Component values (C, CG, INV, VSTK) not matching
   - Prices (PC, PE_FOB, PWM) issues
   
NEXT STEPS:
- Check calibration of government consumption (CGO, GO)
- Verify TIW and TIK are being calculated correctly
- Check parameter extraction for trade (rho_M, beta_M, B_M)
- Ensure all variables are initialized from calibrated values
""")

if __name__ == "__main__":
    main()
