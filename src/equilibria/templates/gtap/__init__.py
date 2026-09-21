"""GTAP CGE Template (Standard GTAP 7)

This package provides a complete GTAP CGE model implementation
following the GTAP Standard 7 specification with 9x10 base.

Modules:
    gtap_sets: GTAP set definitions
    gtap_parameters: GTAP parameters (elasticities, taxes, shares)
    gtap_contract: Contract and closure configurations
    gtap_model_equations: All GTAP Standard 7 model equations
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
    >>>     >>> eq_builder = GTAPModelEquations(sets, params)
    >>> model = eq_builder.build_model()
    >>>
    >>> # Solve
    >>> solver = GTAPSolver(model)
    >>> result = solver.solve()
    >>> print(f"Status: {result.status}")
"""

from equilibria.templates.gtap.altertax import (
    ALTERTAX_ELASTICITY_DEFAULTS,
    AltertaxElasticityOverrides,
    AltertaxRebalanceResult,
    apply_altertax_elasticities,
    rebalance_to_altertax_dataset,
)
from equilibria.templates.gtap.calibration_compare import (
    CalibrationDiff,
    collect_python_benchmark_levels,
    collect_python_calibration_maps,
    compare_calibration,
)
from equilibria.templates.gtap.gtap_contract import (
    GTAPBoundsConfig,
    GTAPClosureConfig,
    GTAPContract,
    GTAPEquationConfig,
    build_gtap_closure_config,
    build_gtap_contract,
    default_gtap_contract,
)
from equilibria.templates.gtap.gtap_parameters import (
    GAMSCalibrationDump,
    GTAPBenchmarkValues,
    GTAPCalibratedShares,
    GTAPElasticities,
    GTAPParameters,
    GTAPShareParameters,
    GTAPTaxRates,
)
from equilibria.templates.gtap.gtap_parity_pipeline import (
    GTAPGAMSReference,
    GTAPParityComparison,
    GTAPParityRunner,
    GTAPVariableSnapshot,
    compare_gtap_gams_parity,
    load_gtap_gams_reference,
    run_gtap_parity_test,
)
from equilibria.templates.gtap.gtap_sets import GTAPSets
from equilibria.templates.gtap.gtap_solver import GTAPSolver, SolverResult, SolverStatus
from equilibria.templates.gtap.shocks import (
    ShockMode,
    apply_shock,
    apply_tariff_shock,
    list_shock_targets,
)

__all__ = [
    # Sets
    "GTAPSets",
    # Parameters
    "GTAPParameters",
    "GAMSCalibrationDump",
    "GTAPElasticities",
    "GTAPBenchmarkValues",
    "GTAPCalibratedShares",
    "GTAPTaxRates",
    "GTAPShareParameters",
    "CalibrationDiff",
    "compare_calibration",
    "collect_python_benchmark_levels",
    "collect_python_calibration_maps",
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
    # Shocks
    "apply_shock",
    "apply_tariff_shock",
    "list_shock_targets",
    "ShockMode",
    # Parity
    "GTAPParityComparison",
    "GTAPParityRunner",
    "GTAPGAMSReference",
    "GTAPVariableSnapshot",
    "compare_gtap_gams_parity",
    "load_gtap_gams_reference",
    "run_gtap_parity_test",
    # Altertax
    "ALTERTAX_ELASTICITY_DEFAULTS",
    "AltertaxElasticityOverrides",
    "AltertaxRebalanceResult",
    "apply_altertax_elasticities",
    "rebalance_to_altertax_dataset",
]


def __getattr__(name: str):
    """Carga perezosa del monolito.

    ``gtap_model_equations`` es REFERENCIA MANUAL (ver su docstring): ningun gate
    salvo ``nl`` lo mide, y el camino vivo es el compuesto por bloques. Importarlo
    aqui de forma eager hacia que CUALQUIER ``import equilibria.templates.gtap``
    --incluidos los tests que solo ejercitan bloques-- construyera ese modulo.

    ``from equilibria.templates.gtap import GTAPModelEquations`` sigue funcionando;
    solo que ahora paga el import quien de verdad lo usa.
    """
    if name == "GTAPModelEquations":
        from equilibria.templates.gtap.gtap_model_equations import (
            GTAPModelEquations as _GME,
        )

        return _GME
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
