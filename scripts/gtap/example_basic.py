#!/usr/bin/env python3
"""
Example: Basic GTAP Model Usage

This example demonstrates how to:
1. Load GTAP data from GDX
2. Build the CGE model
3. Solve the baseline
4. Apply a simple shock

Requirements:
- GTAP GDX file (e.g., asa7x5.gdx)
- Pyomo installed
- IPOPT solver installed (optional, for solving)
"""

from pathlib import Path
import sys

# Add equilibria to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from equilibria.templates.gtap import (
    GTAPSets,
    GTAPParameters,
    GTAPModelEquations,
    GTAPSolver,
    build_gtap_contract,
)


def main():
    # Path to GTAP GDX file
    gdx_file = Path("data/asa7x5.gdx")
    
    if not gdx_file.exists():
        print(f"GDX file not found: {gdx_file}")
        print("Please provide a valid GTAP GDX file")
        return
    
    print("=" * 60)
    print("GTAP CGE Model Example")
    print("=" * 60)
    
    # Step 1: Load Sets
    print("\n1. Loading GTAP sets...")
    sets = GTAPSets()
    sets.load_from_gdx(gdx_file)
    
    info = sets.get_info()
    print(f"   Regions:      {info['n_regions']} - {sets.r}")
    print(f"   Commodities:  {info['n_commodities']} - {sets.i}")
    print(f"   Factors:      {info['n_factors']} - {sets.f}")
    print(f"   Mobile:       {info['n_mobile_factors']} - {sets.mf}")
    print(f"   Specific:     {info['n_specific_factors']} - {sets.sf}")
    
    if info['valid']:
        print("   ✓ Sets are valid")
    else:
        print(f"   ✗ Validation errors: {info['errors']}")
        return
    
    # Step 2: Load Parameters
    print("\n2. Loading parameters...")
    params = GTAPParameters()
    params.load_from_gdx(gdx_file)
    
    info = params.get_info()
    print(f"   Elasticities: {info['n_elasticities']}")
    print(f"   Benchmark:    {info['n_benchmark_values']}")
    print(f"   Tax rates:    {info['n_tax_rates']}")
    print(f"   Share params: {info['n_share_params']}")
    
    if info['valid']:
        print("   ✓ Parameters are valid")
    else:
        print(f"   ✗ Validation errors: {info['errors']}")
    
    # Step 3: Build Model
    print("\n3. Building GTAP model...")
    contract = build_gtap_contract("gtap_standard")
    print(f"   Closure: {contract.closure.name}")
    print(f"   Numeraire: {contract.closure.numeraire}")
    
    equations = GTAPModelEquations(sets, params, contract.closure)
    model = equations.build_model()
    
    # Get model statistics
    from pyomo.environ import Var, Constraint
    n_vars = sum(1 for _ in model.component_objects(Var, active=True))
    n_constr = sum(1 for _ in model.component_objects(Constraint, active=True))
    
    print(f"   Variables:    {n_vars}")
    print(f"   Constraints:  {n_constr}")
    print("   ✓ Model built successfully")
    
    # Step 4: Solve (if IPOPT available)
    print("\n4. Solving model...")
    try:
        solver = GTAPSolver(model, contract.closure, solver_name="ipopt")
        result = solver.solve(tee=False)
        
        print(f"   Status:       {result.status.value}")
        print(f"   Success:      {'✓ Yes' if result.success else '✗ No'}")
        print(f"   Iterations:   {result.iterations}")
        print(f"   Solve time:   {result.solve_time:.2f}s")
        print(f"   Walras check: {result.walras_value:.2e}")
        
        if result.success:
            print("\n   ✓ Baseline solved successfully!")
            
            # Step 5: Apply shock
            print("\n5. Applying shock...")
            print("   Shock: 10% increase in USA agricultural productivity")
            
            shock = {
                "variable": "axp",
                "index": (sets.r[0], sets.a[0]),  # First region, first activity
                "value": 1.10
            }
            solver.apply_shock(shock)
            
            print("   Re-solving with shock...")
            result2 = solver.solve(tee=False)
            
            print(f"   Status:       {result2.status.value}")
            print(f"   Walras check: {result2.walras_value:.2e}")
            
            if result2.success:
                print("\n   ✓ Shock simulation completed!")
        else:
            print("\n   ✗ Solve failed")
            print(f"   Message: {result.message}")
            
    except ImportError as e:
        print(f"   ⚠ Solver not available: {e}")
        print("   Install IPOPT: pip install cyipopt")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
