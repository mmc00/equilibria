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
    from equilibria.family.altertax.closure import (
        ALTAX_RESIDUAL_REGION,
        altertax_closure,
        residual_region_from_exp,
    )
    from equilibria.family.altertax.run import ALTAX_NEWTON_TOL, load_altertax_params
    from equilibria.family.altertax.shocks import load_tms_shock
    from equilibria.family.driver import solve_gtap_family

    d = ROOT / "data/gtap7" / ds
    ref = d / REF
    cmf = ref / "run.cmf"
    rres = residual_region_from_exp(cmf) if cmf.exists() else ALTAX_RESIDUAL_REGION
    shock = load_tms_shock(ref / "tms_rate10.shk")
    res = solve_gtap_family(
        load_altertax_params(d, ref / "parameters.prm"),
        altertax_closure(),
        rres,
        shock=shock,
        cache_root=cache,
        nominal_trade_balance="ratio",
        keep_session=True,
        newton_tol=ALTAX_NEWTON_TOL,
    )
    return res, shock


@pytest.mark.parametrize("ds", ["gtap7_3x3", "nus333"])
def test_ratio_shock_is_one_bordered_solve(ds, cache):
    res, _ = _solve(ds, cache)
    shock = [r for r in res.solves if r["solve"].startswith("shock")]
    assert [r["solve"] for r in shock] == ["shock"]
    assert shock[0]["code"] == 1 and shock[0]["ratio"] == "bordered"
    assert res.results["shock"]["code"] == 1


@pytest.mark.parametrize("ds", ["gtap7_3x3", "nus333"])
def test_bordered_solution_is_the_ratio_fixed_point(ds, cache):
    """A plain shock solve with savf held at the bordered solution's targets (what the
    last outer iteration solved) lands on the same point, and its own W is the
    bordered W: the fixed point of the outer iterations."""
    from equilibria.family.driver import _solve_period, _world_income

    res, shock = _solve(ds, cache)
    mp = res.mp
    run = res.session.run
    w_chk, w = _world_income(mp, "check"), _world_income(mp, "shock")
    sv = mp.vars["savf"]
    run.savf_targets = {
        r: float(sv.value[sv.pos((r, "check"))]) * w / w_chk for r in run.nominal_savf
    }
    try:
        plain, stats = _solve_period(
            run,
            res.session.after_check,
            "shock",
            base_calibrated=True,
            solve_check=True,
            shock=shock,
        )
    finally:
        run.savf_targets = None
    assert stats["code"] == 1
    assert _world_income(plain, "shock") == pytest.approx(w, rel=1e-12)
    for name, v in mp.vars.items():
        sel = np.flatnonzero(v.period == "shock")
        a = np.asarray(v.value, dtype=float)[sel]
        b = np.asarray(plain.vars[name].value, dtype=float)[sel]
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, err_msg=name)
