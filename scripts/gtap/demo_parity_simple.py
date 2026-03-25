#!/usr/bin/env python3
"""
Demo: Parity Comparison Between Python and GAMS

This script demonstrates comparing Python GTAP results with GAMS results.
It creates simulated data for both models and runs the parity comparison.
"""

import sys
from pathlib import Path

# Add equilibria to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from equilibria.templates.gtap.gtap_parity_pipeline import (
    GTAPParityComparison,
    GTAPGAMSReference,
    GTAPSets,
    GTAPVariableSnapshot,
    compare_gtap_gams_parity,
)


def create_demo_sets():
    """Create minimal GTAP sets for demo."""
    sets = GTAPSets()
    sets.r = ["USA", "EUR"]
    sets.i = ["agr", "mfg"]
    sets.a = ["agr", "mfg"]
    sets.f = ["lab", "cap"]
    sets.mf = ["lab"]
    sets.sf = ["cap"]
    sets.aggregation_name = "demo"
    return sets


def create_python_results(sets):
    """Create simulated Python results (benchmark = 1.0)."""
    print("Creating Python model results...")
    
    # Initialize all dictionaries
    xp, x, px, pp = {}, {}, {}, {}
    xs, xds = {}, {}
    ps, pd, pa, pmt, pet = {}, {}, {}, {}, {}
    pmcif, pe, pefob = {}, {}, {}
    xe, xw, xmt, xet = {}, {}, {}, {}
    xft, pft, xf, pf = {}, {}, {}, {}
    xc, xg, xi = {}, {}, {}
    regy, yc, yg, yi, pabs = {}, {}, {}, {}, {}
    
    # Production - all at benchmark (1.0)
    for r in sets.r:
        for a in sets.a:
            xp[(r, a)] = 1.0
            px[(r, a)] = 1.0
            pp[(r, a)] = 1.0
            for i in sets.i:
                x[(r, a, i)] = 1.0
    
    # Supply
    for r in sets.r:
        for i in sets.i:
            xs[(r, i)] = 1.0
            xds[(r, i)] = 1.0
    
    # Prices - Supply
    for r in sets.r:
        for i in sets.i:
            ps[(r, i)] = 1.0
            pd[(r, i)] = 1.0
            pa[(r, i)] = 1.0
            # Import prices
            pmt[(r, i)] = 1.0
            for rp in sets.r:
                if r != rp:
                    pmcif[(r, i, rp)] = 1.0
            # Export prices
            pet[(r, i)] = 1.0
            for rp in sets.r:
                if r != rp:
                    pe[(r, i, rp)] = 1.0
                    pefob[(r, i, rp)] = 1.0
    
    # Trade flows
    for r in sets.r:
        for i in sets.i:
            xmt[(r, i)] = 1.0
            xet[(r, i)] = 1.0
            for rp in sets.r:
                if r != rp:
                    xe[(r, i, rp)] = 1.0
                    xw[(r, i, rp)] = 1.0
    
    # Factors
    for r in sets.r:
        for f in sets.f:
            xft[(r, f)] = 1.0
            pft[(r, f)] = 1.0
            for a in sets.a:
                xf[(r, f, a)] = 1.0
                pf[(r, f, a)] = 1.0
    
    # Demand
    for r in sets.r:
        for i in sets.i:
            xc[(r, i)] = 1.0
            xg[(r, i)] = 1.0
            xi[(r, i)] = 1.0
    
    # Income
    for r in sets.r:
        regy[r] = 200.0
        yc[r] = 100.0
        yg[r] = 50.0
        yi[r] = 50.0
        pabs[r] = 1.0
    
    # Create snapshot with all values
    py_snap = GTAPVariableSnapshot(
        # Production
        xp=xp, x=x, px=px, pp=pp,
        # Supply
        xs=xs, xds=xds,
        # Prices
        ps=ps, pd=pd, pa=pa, 
        pmt=pmt, pmcif=pmcif,
        pet=pet, pe=pe, pefob=pefob,
        # Trade
        xe=xe, xet=xet, xw=xw, xmt=xmt,
        # Factors
        xft=xft, pft=pft, xf=xf, pf=pf,
        # Demand
        xc=xc, xg=xg, xi=xi,
        # Income
        regy=regy, yc=yc, yg=yg, yi=yi, pabs=pabs,
        # Indices
        pnum=1.0,
        walras=1e-12,
    )
    
    # Count total variables
    total_vars = (len(xp) + len(x) + len(xs) + len(xds) + 
                  len(ps) + len(pmt) + len(pmcif) + len(pet) + len(pe) + len(pefob) +
                  len(xe) + len(xet) + len(xw) + len(xmt) +
                  len(xf) + len(xft) + len(pf) + len(pft) +
                  len(xc) + len(xg) + len(xi) +
                  len(regy) + len(yc) + len(yg) + len(yi) + len(pabs) + 2)  # +2 for pnum, walras
    
    print("  ✓ Python snapshot created")
    print(f"  ✓ Total variables: ~{total_vars}")
    
    return py_snap


def create_gams_results(sets, match_perfectly=True):
    """Create simulated GAMS results."""
    print("\nCreating GAMS model results...")
    
    if match_perfectly:
        noise_factor = 0.0
        print("  Mode: Perfect match with Python")
    else:
        noise_factor = 0.0001
        print("  Mode: Small differences (0.01% noise)")
    
    import random
    random.seed(42)
    
    def add_noise(val):
        if noise_factor == 0:
            return val
        return val * (1 + random.uniform(-noise_factor, noise_factor))
    
    # Initialize all dictionaries
    xp, x, px, pp = {}, {}, {}, {}
    xs, xds = {}, {}
    ps, pd, pa, pmt, pet = {}, {}, {}, {}, {}
    pmcif, pe, pefob = {}, {}, {}
    xe, xw, xmt, xet = {}, {}, {}, {}
    xft, pft, xf, pf = {}, {}, {}, {}
    xc, xg, xi = {}, {}, {}
    regy, yc, yg, yi, pabs = {}, {}, {}, {}, {}
    
    # Production
    for r in sets.r:
        for a in sets.a:
            xp[(r, a)] = add_noise(1.0)
            px[(r, a)] = add_noise(1.0)
            pp[(r, a)] = add_noise(1.0)
            for i in sets.i:
                x[(r, a, i)] = add_noise(1.0)
    
    # Supply
    for r in sets.r:
        for i in sets.i:
            xs[(r, i)] = add_noise(1.0)
            xds[(r, i)] = add_noise(1.0)
    
    # Prices - Supply
    for r in sets.r:
        for i in sets.i:
            ps[(r, i)] = add_noise(1.0)
            pd[(r, i)] = add_noise(1.0)
            pa[(r, i)] = add_noise(1.0)
            # Import prices
            pmt[(r, i)] = add_noise(1.0)
            for rp in sets.r:
                if r != rp:
                    pmcif[(r, i, rp)] = add_noise(1.0)
            # Export prices
            pet[(r, i)] = add_noise(1.0)
            for rp in sets.r:
                if r != rp:
                    pe[(r, i, rp)] = add_noise(1.0)
                    pefob[(r, i, rp)] = add_noise(1.0)
    
    # Trade flows
    for r in sets.r:
        for i in sets.i:
            xmt[(r, i)] = add_noise(1.0)
            xet[(r, i)] = add_noise(1.0)
            for rp in sets.r:
                if r != rp:
                    xe[(r, i, rp)] = add_noise(1.0)
                    xw[(r, i, rp)] = add_noise(1.0)
    
    # Factors
    for r in sets.r:
        for f in sets.f:
            xft[(r, f)] = add_noise(1.0)
            pft[(r, f)] = add_noise(1.0)
            for a in sets.a:
                xf[(r, f, a)] = add_noise(1.0)
                pf[(r, f, a)] = add_noise(1.0)
    
    # Demand
    for r in sets.r:
        for i in sets.i:
            xc[(r, i)] = add_noise(1.0)
            xg[(r, i)] = add_noise(1.0)
            xi[(r, i)] = add_noise(1.0)
    
    # Income
    for r in sets.r:
        regy[r] = add_noise(200.0)
        yc[r] = add_noise(100.0)
        yg[r] = add_noise(50.0)
        yi[r] = add_noise(50.0)
        pabs[r] = add_noise(1.0)
    
    # Create snapshot
    gams_snap = GTAPVariableSnapshot(
        # Production
        xp=xp, x=x, px=px, pp=pp,
        # Supply
        xs=xs, xds=xds,
        # Prices
        ps=ps, pd=pd, pa=pa,
        pmt=pmt, pmcif=pmcif,
        pet=pet, pe=pe, pefob=pefob,
        # Trade
        xe=xe, xet=xet, xw=xw, xmt=xmt,
        # Factors
        xft=xft, pft=pft, xf=xf, pf=pf,
        # Demand
        xc=xc, xg=xg, xi=xi,
        # Income
        regy=regy, yc=yc, yg=yg, yi=yi, pabs=pabs,
        # Indices
        pnum=1.0,
        walras=add_noise(1e-12),
    )
    
    print("  ✓ GAMS snapshot created")
    
    return gams_snap


def run_comparison(python_snapshot, gams_snapshot, sets, tolerance):
    """Run parity comparison."""
    # Create GAMS reference
    gams_ref = GTAPGAMSReference(
        gdx_path=Path("simulated.gdx"),
        sets=sets,
        snapshot=gams_snapshot,
        modelstat=1.0,  # Optimal
        solvestat=1.0,  # Normal completion
        solve_time=1.5,
    )
    
    # Compare
    comparison = compare_gtap_gams_parity(
        python_snapshot,
        gams_ref,
        tolerance=tolerance,
    )
    
    return comparison


def print_results(comparison, tolerance):
    """Print comparison results."""
    print("\n" + "=" * 70)
    print(f"PARITY COMPARISON RESULTS (tolerance={tolerance})")
    print("=" * 70)
    
    status = "✓ PASSED" if comparison.passed else "✗ FAILED"
    print(f"\nStatus: {status}")
    print(f"Variables compared: {comparison.n_variables_compared}")
    print(f"Mismatches found: {comparison.n_mismatches}")
    print(f"Max absolute difference: {comparison.max_abs_diff:.6e}")
    print(f"Max relative difference: {comparison.max_rel_diff:.6e}")
    
    if comparison.mismatches:
        print(f"\nTop 10 Mismatches:")
        print("-" * 70)
        print(f"{'#':>3} {'Variable':>20} {'Python':>12} {'GAMS':>12} {'Diff':>12}")
        print("-" * 70)
        for i, m in enumerate(comparison.mismatches[:10], 1):
            key_str = str(m['key'])
            var_name = f"{m['group']}{key_str}"
            print(f"{i:>3} {var_name:>20} {m['python']:>12.6f} "
                  f"{m['gams']:>12.6f} {m['abs_diff']:>12.6e}")
    else:
        print("\n✓ No mismatches found!")
    
    print("=" * 70)


def main():
    """Main demonstration."""
    print("\n" + "=" * 70)
    print("GTAP PARITY COMPARISON DEMONSTRATION")
    print("Comparing Python and GAMS Model Solutions")
    print("=" * 70)
    
    # Create sets
    print("\n1. Setting up GTAP model structure...")
    sets = create_demo_sets()
    print(f"   Regions: {sets.r}")
    print(f"   Commodities: {sets.i}")
    print(f"   Factors: {sets.f}")
    
    # Scenario 1: Perfect match
    print("\n" + "=" * 70)
    print("SCENARIO 1: Perfect Match (Both models identical)")
    print("=" * 70)
    
    py_snap = create_python_results(sets)
    gams_snap = create_gams_results(sets, match_perfectly=True)
    
    comparison = run_comparison(py_snap, gams_snap, sets, tolerance=1e-6)
    print_results(comparison, tolerance=1e-6)
    
    # Scenario 2: Small differences
    print("\n" + "=" * 70)
    print("SCENARIO 2: Small Differences (0.01% noise in GAMS)")
    print("=" * 70)
    
    gams_snap_diff = create_gams_results(sets, match_perfectly=False)
    
    comparison_diff = run_comparison(py_snap, gams_snap_diff, sets, tolerance=1e-6)
    print_results(comparison_diff, tolerance=1e-6)
    
    # Scenario 3: Sensitivity analysis
    print("\n" + "=" * 70)
    print("SCENARIO 3: Sensitivity Analysis (Different Tolerances)")
    print("=" * 70)
    
    tolerances = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
    print(f"\n{'Tolerance':>12} {'Status':>10} {'Mismatches':>12} {'Max Diff':>12}")
    print("-" * 70)
    
    for tol in tolerances:
        comp = run_comparison(py_snap, gams_snap_diff, sets, tolerance=tol)
        status = "✓ PASS" if comp.passed else "✗ FAIL"
        print(f"{tol:>12.0e} {status:>10} {comp.n_mismatches:>12} {comp.max_abs_diff:>12.2e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
This demonstration shows how to compare GTAP model solutions:

1. Python Model: Solves the CGE model using Pyomo/IPOPT
2. GAMS Model: Solves using CGEBox (or loads existing results)
3. Parity Check: Compares all variables and reports differences

In production:
  - Python results come from GTAPSolver.solve()
  - GAMS results come from CGEBox output GDX files
  - Tolerance depends on required precision (typically 1e-6)
  - Mismatches indicate differences in:
    * Equation implementations
    * Solver tolerances
    * Initialization
    * Data handling

To run with real data:
  python scripts/gtap/run_gtap_parity.py \\
    --gdx-file data/asa7x5.gdx \\
    --gams-results results/gams_baseline.gdx
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
