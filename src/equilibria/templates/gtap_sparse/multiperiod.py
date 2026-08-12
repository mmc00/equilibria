"""Multi-period driver for the sparse-trade GTAP model.

Builds the standard block multi-period model, then fixes padding trade routes
(base flow ~0) to their ~0 value and deactivates their constraints. All downstream
machinery (Fisher rows, seeding, solve_multiperiod) is inherited unchanged.
"""

from __future__ import annotations

from typing import Any

from equilibria.templates.gtap.gtap_block_model import (
    GTAPBlockMultiPeriodModel,
    _calibrate_capflex_risk,
)

from .composer import fix_padding_routes


def build_sparse_model_mp(
    params: Any,
    sets: Any,
    closure: Any,
    residual_region: str,
    base_calibrated: bool = False,
    ref_gdx: Any = None,
):
    """Build the multi-period sparse-trade model (unseeded, padding fixed).
    Returns (pyomo_model, mp, fix_stats)."""
    if (
        str(getattr(closure, "savf_flag", "capFix")) == "capFlex"
        and getattr(params, "_capflex_risk", None) is None
    ):
        params._capflex_risk = _calibrate_capflex_risk(
            params, sets, closure, residual_region, ref_gdx=ref_gdx
        )

    mp = GTAPBlockMultiPeriodModel(
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

    # SPARSE layer: fix padding-route vars to ~0 + deactivate their constraints.
    fix_stats = fix_padding_routes(m, params, sets)

    if base_calibrated:
        from equilibria.blocks.gtap.factor import FactorBlock

        _fb = FactorBlock(sets=sets, params=params)
        m._settled_seed = _fb.calibrate_base(
            params, sets, closure, residual_region, ref_gdx=None
        )
    return m, mp, fix_stats
