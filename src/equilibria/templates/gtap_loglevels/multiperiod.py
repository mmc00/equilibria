"""Multi-period driver for the log-levels GTAP model.

Subclasses GTAPBlockMultiPeriodModel and swaps ONLY the single-period source to the
log-wrapped block composer. All downstream machinery — var reflection, Fisher rows,
seeding, closure_stack, MCP pairing — is inherited unchanged (it operates on
variable/equation NAMES, which the log-wrap preserves; only each equation's body
changes from lhs==rhs to log(lhs)==log(rhs)). Run it on the core driver:
``Exercise(GTAP(params, closure, builder="loglevels"), gtap_static(...)).run()``.
"""

from __future__ import annotations

from typing import Any

from equilibria.templates.gtap.gtap_block_model import (
    GTAPBlockMultiPeriodModel,
)
from equilibria.templates.gtap.residual_region import resolve_residual_region

from .composer import build_loglevels_model


class GTAPLogLevelsMultiPeriodModel(GTAPBlockMultiPeriodModel):
    """GTAPBlockMultiPeriodModel with the single-period source swapped to the
    log-wrapped levels blocks."""

    def _block_sp(self):
        return build_loglevels_model(
            self.params, self.sets, self.closure, self.residual_region
        )


def build_loglevels_model_mp(
    params: Any,
    sets: Any,
    closure: Any,
    residual_region: str,
    base_calibrated: bool = False,
    ref_gdx: Any = None,
):
    """Build the full multi-period log-levels model (unseeded). Returns (pyomo_model, mp).

    Mirror of build_block_model but with the log-levels MP driver. ``GTAP(...,
    builder="loglevels")`` calls it (with ``seed_gdx`` to seed every period).
    """
    residual_region = resolve_residual_region(params, sets, closure, residual_region)
    from equilibria.blocks.gtap.capflex_risk import stamp_capflex_risk

    stamp_capflex_risk(params, closure)

    mp = GTAPLogLevelsMultiPeriodModel(
        sets, params, closure, residual_region=residual_region
    )
    m = mp.build_sets()
    mp.build_vars(m)
    from equilibria.templates.gtap.gtap_model_multiperiod import PERIODS

    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = residual_region
    m._base_calibrated = base_calibrated
    m._settled_seed = None
    return m, mp
