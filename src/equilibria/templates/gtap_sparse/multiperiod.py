"""Multi-period driver for the sparse-trade GTAP model.

Builds the standard block multi-period model, then fixes padding trade routes
(base flow ~0) to their ~0 value and deactivates their constraints. All downstream
machinery (Fisher rows, seeding) is inherited unchanged. Run it on the core driver:
``Exercise(GTAP(params, closure, builder="sparse"), gtap_static(...)).run()``.
"""

from __future__ import annotations

from typing import Any

from equilibria.templates.gtap.gtap_block_model import (
    GTAPBlockMultiPeriodModel,
)
from equilibria.templates.gtap.residual_region import resolve_residual_region

from .composer import fix_padding_routes, relax_dissaving_domain


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
    residual_region = resolve_residual_region(params, sets, closure, residual_region)
    from equilibria.blocks.gtap.capflex_risk import stamp_capflex_risk

    stamp_capflex_risk(params, closure)

    mp = GTAPBlockMultiPeriodModel(
        sets, params, closure, residual_region=residual_region
    )
    m = mp.build_sets()
    sp = mp.build_vars(m)
    mp.build_equations_all_periods(m, sp=sp)
    del sp  # que no viva mas que el build
    mp.build_equations_fisher(m)
    m._residual_region = residual_region
    m._base_calibrated = base_calibrated
    m._settled_seed = None

    # SPARSE layer: fix padding-route vars to ~0 + deactivate their constraints.
    fix_stats = fix_padding_routes(m, params, sets)

    # DISSAVING fix: relax us domain to Reals for regions with negative benchmark
    # savings (EGY in 20x41) — their us is genuinely negative (verified vs GEMPACK:
    # SAVE[EGY]=-12635, u[EGY]=-0.28), and NonNegativeReals forces an overflow.
    diss_stats = relax_dissaving_domain(m, params, sets)
    fix_stats["dissaving"] = diss_stats

    if base_calibrated:
        from equilibria.blocks.gtap.factor import FactorBlock

        _fb = FactorBlock(sets=sets, params=params)
        m._settled_seed = _fb.calibrate_base(
            params, sets, closure, residual_region, ref_gdx=None
        )
    return m, mp, fix_stats
