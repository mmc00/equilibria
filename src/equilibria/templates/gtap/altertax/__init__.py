"""GTAP Altertax — Malcolm (1998) CD-elasticity dataset rebalance.

Re-balance a GTAP dataset under a tax/parameter change using the
Cobb-Douglas (CD) share-stability property: with all elasticities set
to 1, equilibrium quantities don't move under tax shocks, so the new
SAM is internally consistent at the new tax structure.

Reference: cgebox/gtap/gams/postModel/altertax.gms (W. Britz, 2016).

Usage:
    from equilibria.templates.gtap import build_gtap_contract
    from equilibria.templates.gtap.altertax import (
        apply_altertax_elasticities,
        rebalance_to_altertax_dataset,
    )

    contract = build_gtap_contract("altertax")
    altertax_params = apply_altertax_elasticities(base_params)
    # ... build & solve model with shocked taxes ...
    rebalance_to_altertax_dataset(
        base_params, shock_params, shock_model,
        output_path="reports/altertax/9x10_alttax.har",
    )
"""

from equilibria.templates.gtap.altertax.altertax_snapshot import (
    build_altertax_warm_start_snapshot,
)
from equilibria.templates.gtap.altertax.outer_loop import (
    apply_recalibration,
    recalibrate_from_solution,
)
from equilibria.templates.gtap.altertax.parameter_overrides import (
    ALTERTAX_ELASTICITY_DEFAULTS,
    AltertaxElasticityOverrides,
    apply_altertax_elasticities,
)
from equilibria.templates.gtap.altertax.postmodel import (
    AltertaxRebalanceResult,
    rebalance_to_altertax_dataset,
)

__all__ = [
    "ALTERTAX_ELASTICITY_DEFAULTS",
    "AltertaxElasticityOverrides",
    "apply_altertax_elasticities",
    "AltertaxRebalanceResult",
    "apply_recalibration",
    "build_altertax_warm_start_snapshot",
    "recalibrate_from_solution",
    "rebalance_to_altertax_dataset",
]
