"""Multi-period driver for the log-levels GTAP model.

Subclasses GTAPBlockMultiPeriodModel and swaps ONLY the single-period source to the
log-wrapped block composer. All downstream machinery — var reflection, Fisher rows,
seeding, closure_stack, MCP pairing, and the solve_multiperiod driver — is inherited
unchanged (it operates on variable/equation NAMES, which the log-wrap preserves; only
each equation's body changes from lhs==rhs to log(lhs)==log(rhs)).
"""

from __future__ import annotations

from typing import Any

from equilibria.templates.gtap.gtap_block_model import (
    GTAPBlockMultiPeriodModel,
    _calibrate_capflex_risk,
)

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

    Mirror of build_block_model but with the log-levels MP driver. Seed with
    mp.seed_all_periods and solve with the monolith's solve_multiperiod (inherited).
    """
    if (
        str(getattr(closure, "savf_flag", "capFix")) == "capFlex"
        and getattr(params, "_capflex_risk", None) is None
    ):
        params._capflex_risk = _calibrate_capflex_risk(
            params, sets, closure, residual_region, ref_gdx=ref_gdx
        )

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
