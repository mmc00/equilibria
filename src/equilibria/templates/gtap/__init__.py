"""GTAP CGE Template (CGEBox version)

This package provides a complete GTAP CGE model implementation
following the CGEBox specification.

Modules:
    gtap_sets: GTAP set definitions
    gtap_parameters: GTAP parameters (elasticities, taxes, shares)
    gtap_contract: Contract and closure configurations
    gtap_model_equations: All CGEBox model equations
    gtap_solver: Solver interface (IPOPT, PATH)

Example:
    >>> from equilibria.templates.gtap import GTAPSets, GTAPParameters, GTAPSolver
    >>> from equilibria.templates.gtap import build_gtap_contract
    >>> 
    >>> # Load data
    >>> sets = GTAPSets()
    >>> sets.load_from_gdx("asa7x5.gdx")
    >>> 
    >>> # Load parameters
    >>> params = GTAPParameters()
    >>> params.load_from_gdx("asa7x5.gdx")
    >>> 
    >>> # Build and solve model
    >>> from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    >>> eq_builder = GTAPModelEquations(sets, params)
    >>> model = eq_builder.build_model()
    >>> 
    >>> # Solve
    >>> solver = GTAPSolver(model)
    >>> result = solver.solve()
    >>> print(f"Status: {result.status}")
"""

from equilibria.templates.gtap.gtap_contract import (
    GTAPBoundsConfig,
    GTAPClosureConfig,
    GTAPContract,
    GTAPEquationConfig,
    build_gtap_closure_config,
    build_gtap_contract,
    default_gtap_contract,
)
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_parameters import (
    GTAPBenchmarkValues,
    GTAPElasticities,
    GTAPParameters,
    GTAPShareParameters,
    GTAPTaxRates,
)
from equilibria.templates.gtap.gtap_sets import GTAPSets
from equilibria.templates.gtap.gtap_solver import GTAPSolver, SolverResult, SolverStatus
from equilibria.templates.gtap.gtap_parity_pipeline import (
    GTAPParityComparison,
    GTAPParityRunner,
    GTAPGAMSReference,
    GTAPVariableSnapshot,
    compare_gtap_gams_parity,
    load_gtap_gams_reference,
    run_gtap_parity_test,
)

__all__ = [
    # Sets
    "GTAPSets",
    
    # Parameters
    "GTAPParameters",
    "GTAPElasticities",
    "GTAPBenchmarkValues",
    "GTAPTaxRates",
    "GTAPShareParameters",
    
    # Contract
    "GTAPContract",
    "GTAPClosureConfig",
    "GTAPEquationConfig",
    "GTAPBoundsConfig",
    "build_gtap_closure_config",
    "build_gtap_contract",
    "default_gtap_contract",
    
    # Model
    "GTAPModelEquations",
    
    # Solver
    "GTAPSolver",
    "SolverResult",
    "SolverStatus",
    
    # Parity
    "GTAPParityComparison",
    "GTAPParityRunner",
    "GTAPGAMSReference",
    "GTAPVariableSnapshot",
    "compare_gtap_gams_parity",
    "load_gtap_gams_reference",
    "run_gtap_parity_test",
]
