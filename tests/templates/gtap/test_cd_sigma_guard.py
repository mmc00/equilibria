"""Production nests never run at sigma = 1 (#192).

cal.gms turns sigmap, sigmand and sigmav = 1 into 1.01, so pxeq, pndeq and pvaeq
always pin px, pnd and pva. These tests hold the three consequences: no loaded
dataset keeps a 1, a model built with a 1 refuses instead of writing 1 = 1, and
altertax no longer holds the nest at its seed.
"""

from __future__ import annotations

import pytest

from equilibria.sources import dataset_dir, find_reference

DATASETS = ["gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7", "gtap7_15x10"]
NESTS = ("sigmap", "sigmand", "sigmav")


def _params(dataset: str):
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir(dataset)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("altertax", [False, True], ids=["pure", "altertax"])
def test_no_production_nest_is_left_at_one(dataset, altertax):
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities

    p = _params(dataset)
    if altertax:
        p = apply_altertax_elasticities(p, in_place=False)
    for nest in NESTS:
        values = getattr(p.elasticities, nest)
        assert values, nest
        at_one = [k for k, v in values.items() if v == 1.0]
        assert not at_one, (nest, at_one[:3])


@pytest.mark.parametrize("nest", NESTS)
@pytest.mark.parametrize("builder", ["blocks", "monolith"])
def test_a_model_built_at_sigma_one_refuses(nest, builder):
    """Without the guard the price equation would read 1 = 1 and stop pinning
    its price, the #192 bug; the build must say so instead."""
    from equilibria.backends.pyomo_backend import BridgeTranslationError
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    p = _params("gtap7_3x3")
    r, a = next(iter(getattr(p.elasticities, nest)))
    getattr(p.elasticities, nest)[(r, a)] = 1.0
    closure = GTAPClosureConfig(name="base", closure_type="MCP")
    # blocks wrap the equation's ValueError, naming the equation
    with pytest.raises((ValueError, BridgeTranslationError), match="#192"):
        if builder == "blocks":
            from equilibria.templates.gtap.gtap_block_model import build_block_model

            build_block_model(p, p.sets, closure, residual_region="ROW")
        else:
            from equilibria.templates.gtap.gtap_model_equations import (
                GTAPModelEquations,
            )

            GTAPModelEquations(p.sets, p, closure).build_model()


@pytest.mark.integration
@pytest.mark.needs_path
def test_altertax_by_default_leaves_the_production_nest_free(monkeypatch):
    """pva, pnd and px are solved, not held at their seed, in check and shock.

    Read right after each period is prepared: once the shock is solved the driver
    freezes the check, so the end state says nothing about how it was solved.
    """
    from equilibria.templates.gtap import period_prep
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    held: dict[str, list] = {}
    prepare = period_prep.PeriodPreparer.prepare

    def recording_prepare(self, m, period, params):
        closure = prepare(self, m, period, params)
        for vn, eqn in (("pva", "eq_pvaeq"), ("pnd", "eq_pndeq"), ("px", "eq_pxeq")):
            var, eq = getattr(m, vn), getattr(m, eqn)
            held.setdefault(period, []).extend(
                [(vn, i) for i in var if i[-1] == period and var[i].fixed]
                + [(eqn, i) for i in eq if i[-1] == period and not eq[i].active]
            )
        return closure

    monkeypatch.setattr(period_prep.PeriodPreparer, "prepare", recording_prepare)

    folder = find_reference(
        "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=1, fbep="header"
    )
    assert folder is not None
    gdx = folder / "out.gdx"
    p = _params("gtap7_3x3")
    p.set_va_subsidy_basis("gempack")
    pa = apply_altertax_elasticities(p, in_place=False)
    closure = GTAPClosureConfig(
        name="altertax",
        closure_type="MCP",
        capital_mobility="mobile",
        fix_endowments=False,
        fix_taxes=True,
        fix_technology=True,
        if_sub=True,
        numeraire="pnum",
    )
    rr = list(p.sets.r)[-1]
    m, mp = build_block_model(pa, pa.sets, closure, residual_region=rr)
    mp.seed_all_periods(m, gdx)
    res = solve_multiperiod(
        m, p, closure, ref_gdx=gdx, skip_base_solve=True, mode="altertax"
    )
    assert all(res[k]["code"] == 1 for k in ("check", "shock")), res
    assert set(held) >= {"check", "shock"}, held.keys()
    for period in ("check", "shock"):
        assert not held[period], (period, held[period][:3])
