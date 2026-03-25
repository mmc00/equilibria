#!/usr/bin/env python3
"""
Test Complete GTAP Python Model

This script tests the fully functional Python GTAP model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from equilibria.templates.gtap import (
    GTAPSets,
    GTAPParameters,
    GTAPModelEquations,
    GTAPSolver,
    build_gtap_contract,
)


def create_test_data():
    """Create test data for full GTAP model."""
    print("Creating test data...")
    
    sets = GTAPSets()
    sets.r = ["USA", "EUR", "CHN"]
    sets.i = ["agr", "mfg", "ser"]
    sets.a = ["agr", "mfg", "ser"]
    sets.f = ["lab", "cap"]
    sets.mf = ["lab"]
    sets.sf = ["cap"]
    sets.aggregation_name = "test3x3"
    
    params = GTAPParameters()
    params.sets = sets
    
    # Add some elasticities
    for r in sets.r:
        for a in sets.a:
            params.elasticities.esubva[(r, a)] = 1.0
        for i in sets.i:
            params.elasticities.omegax[(r, i)] = 2.0
    
    # Add benchmark values
    for r in sets.r:
        for a in sets.a:
            params.benchmark.vom[(r, a)] = 100.0
        for f in sets.f:
            for a in sets.a:
                params.benchmark.vfm[(r, f, a)] = 50.0
    
    return sets, params


def test_model_build():
    """Test that model builds successfully."""
    print("\n" + "=" * 70)
    print("TEST 1: Building GTAP Model")
    print("=" * 70)
    
    try:
        sets, params = create_test_data()
        contract = build_gtap_contract("gtap_cgebox_v1")
        
        equations = GTAPModelEquations(sets, params, contract.closure)
        model = equations.build_model()
        
        # Count components
        from pyomo.environ import Var, Constraint, Objective
        n_vars = sum(1 for _ in model.component_objects(Var))
        n_constr = sum(1 for _ in model.component_objects(Constraint))
        n_obj = sum(1 for _ in model.component_objects(Objective))
        
        print(f"✓ Model built successfully")
        print(f"  Variables: {n_vars}")
        print(f"  Constraints: {n_constr}")
        print(f"  Objectives: {n_obj}")
        
        return model
        
    except Exception as e:
        print(f"✗ Error building model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_solve(model):
    """Test that model solves."""
    print("\n" + "=" * 70)
    print("TEST 2: Solving GTAP Model")
    print("=" * 70)
    
    try:
        from pyomo.environ import SolverFactory, value
        
        print("Attempting to solve with IPOPT...")
        
        # Try to use IPOPT
        solver = SolverFactory('ipopt')
        if solver is None:
            print("⚠ IPOPT not available, trying other solvers...")
            # Try other solvers
            for solver_name in ['gurobi', 'cplex', 'cbc', 'glpk']:
                solver = SolverFactory(solver_name)
                if solver is not None:
                    print(f"  Using {solver_name}")
                    break
        
        if solver is None:
            print("✗ No solver available")
            print("  Install IPOPT: conda install -c conda-forge ipopt")
            return False
        
        # Solve
        print("Solving...")
        result = solver.solve(model, tee=False)
        
        # Check status
        from pyomo.opt import TerminationCondition
        if result.solver.termination_condition in [
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible
        ]:
            print("✓ Model solved successfully")
            print(f"  Status: {result.solver.termination_condition}")
            
            # Show some results
            print("\nSample results:")
            for r in model.r:
                xp_val = value(model.xp[r, 'agr'])
                print(f"  xp[{r},agr] = {xp_val:.4f}")
            
            return True
        else:
            print(f"✗ Solve failed: {result.solver.termination_condition}")
            return False
            
    except Exception as e:
        print(f"✗ Error solving: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_square_system(model):
    """Test if system is approximately square."""
    print("\n" + "=" * 70)
    print("TEST 3: Checking Square System")
    print("=" * 70)
    
    from pyomo.environ import Var, Constraint
    
    n_vars = sum(1 for _ in model.component_objects(Var))
    n_constr = sum(1 for _ in model.component_objects(Constraint))
    
    print(f"Variables: {n_vars}")
    print(f"Constraints: {n_constr}")
    
    if n_vars == n_constr:
        print("✓ Perfect square system!")
    elif abs(n_vars - n_constr) < 5:
        print(f"⚠ Approximately square (diff={abs(n_vars - n_constr)})")
    else:
        print(f"✗ Not square (diff={abs(n_vars - n_constr)})")
    
    return n_vars, n_constr


def main():
    """Run all tests."""
    print("=" * 70)
    print("COMPLETE GTAP MODEL TEST")
    print("=" * 70)
    
    # Test 1: Build model
    model = test_model_build()
    if model is None:
        print("\n✗ FAILED: Could not build model")
        return 1
    
    # Test 2: Check square system
    n_vars, n_constr = test_square_system(model)
    
    # Test 3: Solve model
    solved = test_model_solve(model)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model building: {'✓ PASS' if model else '✗ FAIL'}")
    print(f"Square system: {n_vars} vars, {n_constr} constr")
    print(f"Model solving: {'✓ PASS' if solved else '✗ FAIL'}")
    
    if model and solved:
        print("\n🎉 SUCCESS: Complete GTAP Python model is functional!")
        return 0
    else:
        print("\n⚠ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
