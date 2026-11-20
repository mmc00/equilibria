"""The array driver end to end on 3x3 with ifSUB=0 (the nine report rows are live).

Same checks as ``test_family_e2e.py`` against the ifSUB=0 Pyomo run: no Pyomo model
is built by the family route (spy on ``ConcreteModel.__init__``, ``PyomoNLP.__init__``
and ``identify_variables``), every Var cell of the three periods agrees with the saved
Pyomo run (``final.pkl``, same ``linsolve``) within 1e-9 (``|a - b| <= 1e-9 *
max(1, |b|)``, NaN equals NaN), identical fixed flags and bounds, code 1. Plus: the
record cache key differs from ifSUB=1, ``MPArrays.build`` has the nine report rows
active per period, and the ``imptx`` accept/reject set equals Pyomo's
``_can_take_shock``. Iteration counts are reported, never asserted.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

from tests.family._prep import _newton_mumps, capture_dir
from tests.family.test_mp_build import diff_to_photo

pytest.importorskip("pyomo")
DATASET = "gtap7_3x3"
IFSUB = 0
REPORT_ROWS = (
    "eq_pp_rai",
    "eq_xwmg",
    "eq_xmgm",
    "eq_pwmg",
    "eq_pefobeq",
    "eq_pmcifeq",
    "eq_pmeq",
    "eq_pfaeq",
    "eq_pfyeq",
)


def _closure0():
    from tests.family.test_squared_system import _route_closure

    return _route_closure().model_copy(update={"if_sub": False})


@pytest.fixture(scope="module")
def linsolve():
    return "mumps" if importlib.util.find_spec("mumps") else "direct"


@pytest.fixture(scope="module")
def reference(linsolve):
    """The Pyomo ifSUB=0 run, captured before the spies are armed."""
    env = {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield capture_dir(DATASET, linsolve, IFSUB)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_spy_sees_the_pyomo_route():
    """The spy would have caught a build: a ConcreteModel built under the same wrap
    is counted (re-armed here, since the fixture has already undone its patches)."""
    from pyomo.environ import ConcreteModel

    calls = {"n": 0}
    orig = ConcreteModel.__init__

    def wrapped(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ConcreteModel, "__init__", wrapped)
        ConcreteModel()
    assert calls["n"] == 1
    assert ConcreteModel.__init__ is orig


def test_cache_key_differs_from_ifsub1():
    from equilibria.family.disk_cache import cache_key
    from tests.family.test_prep_recipe import _params
    from tests.family.test_squared_system import _route_closure

    p = _params(DATASET)
    rr = list(p.sets.r)[-1]
    k1 = cache_key(p, _route_closure(), rr)
    k0 = cache_key(p, _closure0(), rr)
    assert k0 is not None and k1 is not None and k0 != k1


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from equilibria.family import cross_period
    from equilibria.family.disk_cache import load_or_build
    from equilibria.family.mp_arrays import MPArrays
    from tests.family.test_prep_recipe import _params

    p = _params(DATASET)
    closure = _closure0()
    rr = list(p.sets.r)[-1]
    root = tmp_path_factory.mktemp("records0")
    backend, state = load_or_build(p, closure, rr, root)
    cross = cross_period.load_or_record(
        p, closure, rr, root, sets=backend.sets, var_domains=backend.var_domains
    )
    mp = MPArrays.build(state, backend, cross, params=p)
    return p, closure, rr, backend, state, mp


def test_mparrays_rows_equal_pyomos_with_the_report_rows(built, reference):
    from equilibria.family.prep_capture import load

    *_, mp = built
    for fam in REPORT_ROWS:
        assert fam in mp.rows, fam
        row = mp.rows[fam]
        assert row.active.all()
        assert {k[-1] for k in row.keys} == {"base", "check", "shock"}, fam
    photo = load(reference / "001_A_base.pkl")
    assert diff_to_photo(mp, photo) == []


def test_imptx_accept_reject_equals_pyomos(built):
    from equilibria.family.shock import can_take_shock
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.shocks import _can_take_shock

    p, closure, rr, backend, _state, mp = built
    m = build_block_model(p, p.sets, closure, rr, base_calibrated=False)[0]
    cells = [k[:-1] for k in mp.vars["imptx"].keys if k[-1] == "shock"]
    cells.append(("NOWHERE", "NOGOOD", "NOWHERE"))
    got = {c: can_take_shock(mp, backend, "imptx", c) for c in cells}
    want = {c: _can_take_shock(m, "imptx", c) for c in cells}
    assert got == want
    assert any(want.values())


# The core route (``Exercise(GTAP(backend="family"), gtap_static)``) against the
# Pyomo photo.
@pytest.fixture(scope="module")
def core_run(reference, linsolve, tmp_path_factory):
    """``_solve_shock`` on the family backend, spied."""
    from tests.family.test_family_e2e import _spies
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    mpatch = pytest.MonkeyPatch()
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        mpatch.setenv(k, v)
    mpatch.setenv("EQUILIBRIA_FAMILY_CACHE", str(tmp_path_factory.mktemp("cache")))
    try:
        with _spies() as calls:
            model, code = _solve_shock(
                DATASET,
                IFSUB,
                "capFix",
                solve_nlp=True,
                backend="family",
            )
    finally:
        mpatch.undo()
    return model, code, calls


def test_core_route_builds_no_pyomo_model(core_run):
    _, _, calls = core_run
    assert calls == {"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}


def test_core_route_every_cell_equals_final_pkl(core_run, reference):
    from equilibria.family.prep_capture import load

    model, code, _ = core_run
    assert code == 1
    final = load(reference / "final.pkl")
    assert diff_to_photo(model.mp, final, 1e-9, lb_rtol=1e-9) == []
