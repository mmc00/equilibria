"""Altertax elasticity overrides.

Mirror of ``cgebox/gtap/scen/Parameters/parameter_altertax.gms``:

| Param           | Original GTAP | Altertax |
|-----------------|---------------|----------|
| ``esubva``      | calibrated    | **1**    |
| ``esubinv``     | calibrated    | **1**    |
| ``esubd``       | calibrated    | **0.95** |
| ``esubm``       | calibrated    | **0.95** |
| ``etrae``       | calibrated    | **1**    |
| ``omegaf``      | calibrated    | **1**    |

Equilibria does not (yet) carry a separate ``esubinv`` container —
investment substitution sits under the same Leontief/CES bucket as
the production aggregator. We override the closest equivalents
(``sigmav``/``sigmap``) instead.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

from equilibria.templates.gtap.gtap_parameters import (
    GTAPElasticities,
    GTAPParameters,
)


@dataclass(frozen=True)
class AltertaxElasticityOverrides:
    """Altertax preset values. Tweak only if you know what you're doing."""

    esubva: float = 1.0
    esubd: float = 0.95
    esubm: float = 0.95
    etrae: float = 1.0
    omegaf: float = 1.0
    # Nested CES for production (downstream of esubva)
    sigmav: float = 1.0
    sigmap: float = 1.0
    sigmand: float = 1.0
    # Make/aggregation
    omegas: float = 1.0
    sigmas: float = 1.0
    # Trade CET (export side)
    omegax: float = 1.0
    omegaw: float = 1.0


ALTERTAX_ELASTICITY_DEFAULTS = AltertaxElasticityOverrides()


def _override_dict(target: Dict, value: float) -> int:
    """Replace every value in ``target`` with ``value``. Returns count."""
    n = 0
    for key in list(target.keys()):
        target[key] = value
        n += 1
    return n


def apply_altertax_elasticities(
    params: GTAPParameters,
    overrides: AltertaxElasticityOverrides | None = None,
    *,
    in_place: bool = False,
) -> GTAPParameters:
    """Return a copy of ``params`` with altertax elasticity overrides.

    The defaults match cgebox ``parameter_altertax.gms``: CD value-added
    (esubva=1), Armington 0.95, mobile factors with CD CET (etrae=1).

    Args:
        params: source GTAPParameters (calibrated baseline).
        overrides: replacement values (defaults to ALTERTAX_ELASTICITY_DEFAULTS).
        in_place: mutate ``params`` directly. Default False (deep copy).

    Returns:
        GTAPParameters with overridden elasticities. Calibrated shares,
        SAM benchmarks, and tax rates are untouched — only elasticities.
    """
    o = overrides or ALTERTAX_ELASTICITY_DEFAULTS
    target = params if in_place else copy.deepcopy(params)
    e: GTAPElasticities = target.elasticities

    _override_dict(e.esubva, o.esubva)
    _override_dict(e.esubd, o.esubd)
    _override_dict(e.esubm, o.esubm)
    _override_dict(e.omegax, o.omegax)
    _override_dict(e.omegaw, o.omegaw)
    _override_dict(e.omegas, o.omegas)
    _override_dict(e.sigmas, o.sigmas)
    _override_dict(e.omegaf, o.omegaf)
    _override_dict(e.sigmav, o.sigmav)
    _override_dict(e.sigmap, o.sigmap)
    _override_dict(e.sigmand, o.sigmand)
    for key in list(e.etrae.keys()):
        e.etrae[key] = o.etrae

    return target
