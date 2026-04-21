#!/usr/bin/env python3
"""
Demo: Run Both Python GTAP and Simulated GAMS - Compare Solutions

This script demonstrates the full parity workflow by:
1. Creating a minimal GTAP model and solving it in Python
2. Creating a simulated GAMS baseline (with small differences)
3. Running the parity comparison
4. Displaying detailed results

This is a demonstration - in production, you would use real GDX files.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple

# Add equilibria to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from equilibria.templates.gtap import (
    GTAPSets,
    GTAPParameters,
    GTAPModelEquations,
    GTAPSolver,
    build_gtap_contract,
)
from equilibria.templates.gtap.gtap_parity_pipeline import (
    GTAPParityRunner,
    GTAPGAMSReference,
    GTAPVariableSnapshot,
    compare_gtap_gams_parity,
)


@dataclass
class MinimalGTAPData:
    """Minimal GTAP data for demonstration."""
    sets: GTAPSets = field(default_factory=GTAPSets)
    params: GTAPParameters = field(default_factory=GTAPParameters)
    
    def __post_init__(self):
        """Initialize minimal data."""
        # Create minimal sets
        self.sets.r = ["USA", "EUR"]
        self.sets.i = ["agr", "mfg"]
        self.sets.a = ["agr", "mfg"]
        self.sets.f = ["lab", "cap"]
        self.sets.mf = ["lab"]
        self.sets.sf = ["cap"]
        self.sets.m = ["sea"]
        self.sets.aggregation_name = "demo"
        
        # Create minimal parameters
        self._create_elasticities()
        self._create_benchmark()
        self._create_taxes()
        self._create_shares()
    
    def _create_elasticities(self):
        """Create minimal elasticities."""
        for r in self.sets.r:
            for a in self.sets.a:
                self.params.elasticities.esubva[(r, a)] = 1.0
            for i in self.sets.i:
                self.params.elasticities.esubd[(r, i)] = 2.0
                self.params.elasticities.esubm[(r, i)] = 4.0
                self.params.elasticities.omegax[(r, i)] = 2.0
                self.params.elasticities.omegaw[(r, i)] = 4.0
        
        for f in self.sets.f:
            self.params.elasticities.etrae[f] = float('inf') if f == "lab" else 0.0
    
    def _create_benchmark(self):
        """Create minimal benchmark values."""
        # Production
        for r in self.sets.r:
            for a in self.sets.a:
                self.params.benchmark.vom[(r, a)] = 100.0
        
        # Factors
        for r in self.sets.r:
            for f in self.sets.f:
                for a in self.sets.a:
                    self.params.benchmark.vfm[(r, f, a)] = 50.0
        
        # Trade
        for r in self.sets.r:
            for i in self.sets.i:
                for rp in self.sets.r:
                    if r != rp:
                        self.params.benchmark.vxmd[(r, i, rp)] = 20.0
                        self.params.benchmark.viws[(r, i, rp)] = 20.0
    
    def _create_taxes(self):
        """Create minimal taxes."""
        for r in self.sets.r:
            for a in self.sets.a:
                self.params.taxes.rto[(r, a)] = 0.0
    
    def _create_shares(self):
        """Create minimal shares."""
        for r in self.sets.r:
            for a in self.sets.a:
                for i in self.sets.i:
                    self.params.shares.p_gx[(r, a, i)] = 0.5


def run_python_model(data: MinimalGTAPData) -> Tuple[GTAPModelEquations, object]:
    """Run the Python GTAP model.
    
    Returns:
        Tuple of (equations, model)
    """
    print("=" * 70)
    print("RUNNING PYTHON GTAP MODEL")
    print("=" * 70)
    
    # Build model
    contract = build_gtap_contract("gtap_cgebox_v1")
    equations = GTAPModelEquations(data.sets, data.params, contract.closure)
    model = equations.build_model()
    
    print(f"✓ Model built: {len(list(model.component_objects()))} components")
    
    # Try to solve
    try:
        solver = GTAPSolver(model, contract.closure, solver_name="ipopt")
        result = solver.solve(tee=False)
        
        print(f"\nPython Results:")
        print(f"  Status: {result.status.value}")
        print(f"  Success: {'✓' if result.success else '✗'}")
        if result.success:
            print(f"  Iterations: {result.iterations}")
            print(f"  Solve time: {result.solve_time:.2f}s")
            print(f"  Walras check: {result.walras_value:.2e}")
        else:
            print(f"  Message: {result.message}")
            
    except ImportError as e:
        print(f"\n⚠ Solver not available: {e}")
        print("  (Using initialized values for demo)")
    
    return equations, model


def create_gams_baseline(data: MinimalGTAPData, add_noise: bool = True) -> GTAPGAMSReference:
    """Create simulated GAMS baseline.
    
    Args:
        data: GTAP data
        add_noise: If True, add small random differences to simulate GAMS results
        
    Returns:
        GTAPGAMSReference with simulated GAMS results
    """
    print("\n" + "=" * 70)
    print("LOADING GAMS BASELINE")
    print("=" * 70)
    
    import random
    random.seed(42)  # For reproducibility
    
    def add_small_noise(value: float, magnitude: float = 0.001) -> float:
        """Add small noise to simulate GAMS vs Python differences."""
        if not add_noise:
            return value
        noise = random.uniform(-magnitude, magnitude)
        return value * (1 + noise)
    
    # Create GAMS snapshot (simulated)
    gams_snap = GTAPVariableSnapshot()
    
    # Production (simulated with small differences)
    for r in data.sets.r:
        for a in data.sets.a:
            gams_snap.xp[(r, a)] = add_small_noise(1.0, 0.0001)
            gams_snap.px[(r, a)] = add_small_noise(1.0, 0.0001)
            gams_snap.pp[(r, a)] = add_small_noise(1.0, 0.0001)
    
    # Prices
    for r in data.sets.r:
        for i in data.sets.i:
            gams_snap.ps[(r, i)] = add_small_noise(1.0, 0.0001)
            gams_snap.pd[(r, i)] = add_small_noise(1.0, 0.0001)
            gams_snap.pa[(r, i)] = add_small_noise(1.0, 0.0001)
    
    # Factors
    for r in data.sets.r:
        for f in data.sets.f:
            gams_snap.xft[(r, f)] = add_small_noise(1.0, 0.0001)
            gams_snap.pft[(r, f)] = add_small_noise(1.0, 0.0001)
            for a in data.sets.a:
                gams_snap.xf[(r, f, a)] = add_small_noise(1.0, 0.0001)
                gams_snap.pf[(r, f, a)] = add_small_noise(1.0, 0.0001)
    
    # Demand
    for r in data.sets.r:
        for i in data.sets.i:
            gams_snap.xc[(r, i)] = add_small_noise(1.0, 0.0001)
            gams_snap.xg[(r, i)] = add_small_noise(1.0, 0.0001)
            gams_snap.xi[(r, i)] = add_small_noise(1.0, 0.0001)
    
    # Income
    for r in data.sets.r:
        gams_snap.regy[r] = add_small_noise(200.0, 0.0001)
        gams_snap.yc[r] = add_small_noise(100.0, 0.0001)
        gams_snap.yg[r] = add_small_noise(50.0, 0.0001)
        gams_snap.yi[r] = add_small_noise(50.0, 0.0001)
    
    # Indices
    gams_snap.pnum = 1.0
    gams_snap.walras = add_small_noise(0.0, 0.000001)
    
    print("✓ GAMS baseline loaded (simulated)")
    print(f"  Variables: ~100+")
    print(f"  Add noise: {add_noise}")
    
    return GTAPGAMSReference(
        gdx_path=Path("simulated.gdx"),
        sets=data.sets,
        snapshot=gams_snap,
        modelstat=1.0,
        solvestat=1.0,
        solve_time=1.5,
    )


def run_parity_comparison(python_model, gams_reference: GTAPGAMSReference) -> None:
    """Run parity comparison and display results."""
    print("\n" + "=" * 70)
    print("RUNNING PARITY COMPARISON")
    print("=" * 70)
    
    # Compare
    comparison = compare_gtap_gams_parity(
        python_model,
        gams_reference,
        tolerance=1e-3,  # Loose tolerance for demo
    )
    
    # Display results
    print(f"\nComparison Results:")
    print(f"  Status: {'✓ PASSED' if comparison.passed else '✗ FAILED'}")
    print(f"  Variables compared: {comparison.n_variables_compared}")
    print(f"  Mismatches: {comparison.n_mismatches}")
    print(f"  Max absolute diff: {comparison.max_abs_diff:.2e}")
    print(f"  Max relative diff: {comparison.max_rel_diff:.2e}")
    
    if comparison.mismatches:
        print(f"\n  Top 10 Mismatches:")
        for i, m in enumerate(comparison.mismatches[:10], 1):
            key_str = str(m['key'])
            print(f"    {i:2d}. {m['group']:8s}{key_str:25s} "
                  f"Py={m['python']:10.6f} GAMS={m['gams']:10.6f} "
                  f"Diff={m['abs_diff']:10.6e}")
    else:
        print(f"\n  ✓ No mismatches found!")
    
    # Summary
    print("\n" + "=" * 70)
    if comparison.passed:
        print("✓ PARITY CHECK PASSED")
        print("Both Python and GAMS models reached the same solution")
        print("within the specified tolerance.")
    else:
        print("✗ PARITY CHECK FAILED")
        print(f"Found {comparison.n_mismatches} mismatches between Python and GAMS.")
        print("This could indicate:")
        print("  - Different equation implementations")
        print("  - Different solver tolerances")
        print("  - Different initialization")
        print("  - Data loading differences")
    print("=" * 70)


def main():
    """Main demonstration."""
    print("\n" + "=" * 70)
    print("GTAP PARITY DEMONSTRATION")
    print("Python GTAP vs GAMS CGEBox Comparison")
    print("=" * 70)
    print("\nThis script demonstrates running both Python and GAMS GTAP models")
    print("and comparing their solutions for parity.")
    print("=" * 70)
    
    # Step 1: Create minimal data
    print("\n1. Creating minimal GTAP data...")
    data = MinimalGTAPData()
    print(f"   ✓ Regions: {data.sets.r}")
    print(f"   ✓ Commodities: {data.sets.i}")
    print(f"   ✓ Factors: {data.sets.f}")
    
    # Step 2: Run Python model
    equations, model = run_python_model(data)
    
    # Step 3: Load/Create GAMS baseline
    gams_reference = create_gams_baseline(data, add_noise=True)
    
    # Step 4: Run parity comparison
    run_parity_comparison(model, gams_reference)
    
    # Additional analysis with different tolerances
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS (Different Tolerances)")
    print("=" * 70)
    
    tolerances = [1e-2, 1e-3, 1e-4, 1e-6]
    print(f"\n{'Tolerance':>12} {'Status':>10} {'Mismatches':>12} {'Max Diff':>12}")
    print("-" * 70)
    
    for tol in tolerances:
        comp = compare_gtap_gams_parity(model, gams_reference, tolerance=tol)
        status = "✓ PASS" if comp.passed else "✗ FAIL"
        print(f"{tol:>12.0e} {status:>10} {comp.n_mismatches:>12} {comp.max_abs_diff:>12.2e}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nIn production, you would use:")
    print("  1. Real GTAP GDX data file")
    print("  2. Real GAMS results GDX from CGEBox")
    print("  3. Run: python scripts/gtap/run_gtap_parity.py \\")
    print("           --gdx-file data/asa7x5.gdx \\")
    print("           --gams-results results/gams_baseline.gdx")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
