"""``trade_balance="ratio"`` inside the shock's Newton system: one bordered solve.

The world income ``W = sum(regy)`` is one more unknown with its own row ``W - g.x = 0``
and the held ``savf`` of every non-residual region follows it, ``savf_check * W /
W_check``: ``J dx + v dW = -F``, ``dW = g.dx`` (``v = dF/dW`` once). The solution is the
fixed point the outer iterations used to reach: a plain solve with the savf held at the
solution's targets gives the same point and the same ``W``."""

from __future__ import annotations

import numpy as np
import pytest

from tests.family._prep import ROOT, _newton_mumps

pytest.importorskip("pyomo")
REF = "gempack_altertax-rungtap_tms-rate10_s8-16-32"


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        mp.setenv(k, v)
    root = tmp_path_factory.mktemp("cache")
    mp.setenv("EQUILIBRIA_FAMILY_CACHE", str(root))
    yield root
    mp.undo()


def _solve(ds, cache):
    """The RunGTAP ALTERTAX run of ``ds`` with ``trade_balance="ratio"`` on the core,
    flown period by period as ``solve_altertax`` flies it: ``(results, solves, mp
    after the shock, the shock session of a plain run, shock)``. The session is the
    run's (template, records, arrays after the check) with the nominal recipe: its
    shock holds savf where it is put, no border."""
    from equilibria.core.driver import fly_period
    from equilibria.family.altertax.closure import (
        ALTAX_RESIDUAL_REGION,
        altertax_closure,
        residual_region_from_exp,
    )
    from equilibria.family.altertax.run import ALTAX_NEWTON_TOL, load_altertax_params
    from equilibria.family.altertax.shocks import load_tms_shock
    from equilibria.family.shock_session import ShockSession
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP

    d = ROOT / "data/gtap7" / ds
    ref = d / REF
    cmf = ref / "run.cmf"
    rres = residual_region_from_exp(cmf) if cmf.exists() else ALTAX_RESIDUAL_REGION
    shock = load_tms_shock(ref / "tms_rate10.shk")
    template = GTAP(
        load_altertax_params(d, ref / "parameters.prm"),
        altertax_closure(),
        residual_region=rres,
        base_calibrated=True,
        backend="family",
        cache_root=cache,
        newton_tol=ALTAX_NEWTON_TOL,
    )
    recipe = gtap_static(
        "gtap", trade_balance="ratio", base_calibrated=True, solve_check=True
    )
    model = template.build(recipe.names)
    run = model.run
    run.nominal_savf = tuple(r for r in run.regions if r != run.residual_region)
    results, after_check = {}, None
    with template.run_scope():
        for period in recipe.names:
            if period == "shock":
                after_check = model.mp.copy()
            results[period] = fly_period(
                model, recipe, period, template=template, shock=shock
            )
    assert after_check is not None
    plain = ShockSession(
        template,
        gtap_static(
            "gtap", trade_balance="nominal", base_calibrated=True, solve_check=True
        ),
        run,
        after_check,
        ratio=False,
    )
    return results, [dict(r) for r in run.solves], model.mp, plain, shock


@pytest.mark.parametrize("ds", ["gtap7_3x3", "nus333"])
def test_ratio_shock_is_one_bordered_solve(ds, cache):
    import inspect

    from equilibria.templates.gtap.recipe import gtap_static

    results, solves, _, _, _ = _solve(ds, cache)
    shock = [r for r in solves if r["solve"].startswith("shock")]
    assert [r["solve"] for r in shock] == ["shock"]
    assert shock[0]["code"] == 1
    assert results["shock"]["code"] == "optimal"
    # the shock's solver is Newton-TR bordered with the world income
    solver = gtap_static(trade_balance="ratio").plans["shock"].solver
    assert solver is not None
    assert inspect.getclosurevars(solver).nonlocals["border"] == "ratio"


@pytest.mark.parametrize("ds", ["gtap7_3x3", "nus333"])
def test_bordered_solution_is_the_ratio_fixed_point(ds, cache):
    """A plain shock solve with savf held at the bordered solution's targets (what the
    last outer iteration solved) lands on the same point, and its own W is the
    bordered W: the fixed point of the outer iterations."""
    from equilibria.family import period_newton as pn
    from equilibria.family.period_newton import world_income

    _, _, mp, session, shock = _solve(ds, cache)
    w_chk, w = world_income(mp, "check"), world_income(mp, "shock")
    sv = mp.vars["savf"]
    targets = {
        r: float(sv.value[sv.pos((r, "check"))]) * w / w_chk
        for r in session.run.nominal_savf
    }
    ss, sq, mp_sq = session.shock_system(shock, savf_targets=targets)
    # the run's tolerance and MUMPS state, as the core's solver sets them
    tol = pn._NEWTON_TOL.set(session.run.newton_tol)
    keep = pn._MUMPS_KEEP.set(session.run.mumps)
    try:
        res, _, code = pn._newton(ss, sq)
    finally:
        pn._NEWTON_TOL.reset(tol)
        pn._MUMPS_KEEP.reset(keep)
    assert code == 1
    plain = session.finish_shock(mp_sq, sq, res.x)
    assert world_income(plain, "shock") == pytest.approx(w, rel=1e-12)
    for name, v in mp.vars.items():
        sel = np.flatnonzero(v.period == "shock")
        a = np.asarray(v.value, dtype=float)[sel]
        b = np.asarray(plain.vars[name].value, dtype=float)[sel]
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, err_msg=name)
