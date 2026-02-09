"""
Solve PEP model using Pyomo backend.

This script solves the PEP CGE model using the Pyomo backend with IPOPT solver.
"""

import sys
from pathlib import Path

sys.path.insert(0, '/Users/marmol/proyectos/equilibria/src')

from equilibria.templates import PEP1R
from equilibria.babel.gdx.reader import read_gdx
from equilibria.backends.pyomo_backend import PyomoBackend


def solve_pep_model():
    """Create and solve PEP model."""
    print("=" * 70)
    print("Solving PEP Model with Pyomo/IPOPT")
    print("=" * 70)
    
    # Load data
    data_path = Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original")
    sam_file = data_path / "SAM-V2_0.gdx"
    
    print(f"\nLoading data from: {data_path}")
    sam_data = read_gdx(sam_file)
    print(f"✓ Loaded SAM: {len(sam_data['symbols'])} symbols")
    
    # Create model
    print("\nCreating PEP model...")
    template = PEP1R()
    template.extract_sets_from_gdx_data(sam_data)
    
    print(f"Sets:")
    print(f"  Sectors: {template.sectors}")
    print(f"  Labor types: {template.labor_types}")
    print(f"  Capital types: {template.capital_types}")
    print(f"  Households: {template.households}")
    
    model = template.create_model(calibrate=False)
    
    stats = model.statistics
    print(f"\nModel Statistics:")
    print(f"  Variables: {stats.variables}")
    print(f"  Equations: {stats.equations}")
    print(f"  Degrees of Freedom: {stats.degrees_of_freedom}")
    
    # Build and solve with Pyomo
    print("\n" + "-" * 70)
    print("Building Pyomo model...")
    print("-" * 70)
    
    try:
        backend = PyomoBackend(solver='ipopt')
        backend.build(model)
        print("✓ Pyomo model built successfully")
        
        # Print model info
        print(f"\nPyomo model: {backend.pyomo_model.name}")
        print(f"  Sets: {len(list(backend.pyomo_model.component_objects()))}")
        
        # Solve
        print("\n" + "-" * 70)
        print("Solving with IPOPT...")
        print("-" * 70)
        
        solution = backend.solve(options={
            'tol': 1e-6,
            'max_iter': 300,
            'print_level': 5,
        })
        
        print(f"\n✓ Solution found!")
        print(f"  Status: {solution.status}")
        print(f"  Objective: {solution.objective_value}")
        print(f"  Solve time: {solution.solve_time:.2f} seconds")
        
        # Export results
        output_path = Path("/Users/marmol/proyectos/equilibria/results/python_pep_solved.gdx")
        print(f"\nExporting results to: {output_path}")
        
        # TODO: Export solution to GDX
        # For now, just save solution object
        import pickle
        with open(output_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(solution, f)
        print(f"✓ Solution saved to: {output_path.with_suffix('.pkl')}")
        
        return solution
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install required packages:")
        print("  pip install pyomo ipopt")
        return None
    except Exception as e:
        print(f"\n❌ Error during solve: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution."""
    solution = solve_pep_model()
    
    if solution:
        print("\n" + "=" * 70)
        print("SUCCESS: Model solved!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("FAILED: Could not solve model")
        print("=" * 70)


if __name__ == "__main__":
    main()
