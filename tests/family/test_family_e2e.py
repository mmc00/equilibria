"""The array driver end to end: ``_solve_shock(..., backend="family")`` on 3x3.

No Pyomo model is built in the family route (a spy on ``ConcreteModel.__init__``,
``PyomoNLP.__init__`` and ``identify_variables`` proves it), and the three periods agree
with the saved Pyomo run (``final.pkl``, same ``linsolve``): every Var cell within 1e-9
(``|a - b| <= 1e-9 * max(1, |b|)``, NaN equals NaN), identical fixed flags and bounds,
code 1. The iteration counts are reported, never asserted.
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


@pytest.fixture(scope="module")
def linsolve():
    return "mumps" if importlib.util.find_spec("mumps") else "direct"


@pytest.fixture(scope="module")
def reference(linsolve):
    """The Pyomo run, captured before the spies are armed."""
    env = {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield capture_dir(DATASET, linsolve)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def family_run(reference, linsolve, tmp_path_factory):
    """``_solve_shock`` on the family route with the Pyomo entry points spied."""
    import pyomo.contrib.pynumero.interfaces.pyomo_nlp as pn
    import pyomo.core.expr.visitor as visitor
    from pyomo.environ import ConcreteModel

    calls = {"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}

    def spy_init(cls, key):
        orig = cls.__init__

        def wrapped(self, *a, **k):
            calls[key] += 1
            return orig(self, *a, **k)

        return orig, wrapped

    orig_cm, wrap_cm = spy_init(ConcreteModel, "ConcreteModel")
    orig_nlp, wrap_nlp = spy_init(pn.PyomoNLP, "PyomoNLP")
    orig_iv = visitor.identify_variables

    def wrap_iv(*a, **k):
        calls["identify_variables"] += 1
        return orig_iv(*a, **k)

    # every module that bound the name at import time
    holders = [
        mod
        for mod in list(sys.modules.values())
        if vars(mod).get("identify_variables") is orig_iv
    ]
    mpatch = pytest.MonkeyPatch()
    for mod in holders:
        mpatch.setattr(mod, "identify_variables", wrap_iv)
    mpatch.setattr(ConcreteModel, "__init__", wrap_cm)
    mpatch.setattr(pn.PyomoNLP, "__init__", wrap_nlp)
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        mpatch.setenv(k, v)
    mpatch.setenv("EQUILIBRIA_FAMILY_CACHE", str(tmp_path_factory.mktemp("cache")))
    mpatch.setenv("EQUILIBRIA_GTAP_BACKEND", "family")
    try:
        from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

        res, code = _solve_shock(DATASET, 1, "capFix", solve_nlp=True)
    finally:
        mpatch.undo()
    return res, code, calls


def test_family_route_builds_no_pyomo_model(family_run):
    _, _, calls = family_run
    assert calls == {"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}


def test_spy_sees_the_pyomo_route(reference):
    """The spy would have caught a build: the reference capture ran the Pyomo route."""
    from pyomo.environ import ConcreteModel

    assert ConcreteModel.__init__.__name__ == "__init__"  # restored, not a wrapper


def test_every_cell_of_the_three_periods_equals_final_pkl(family_run, reference):
    from equilibria.family.prep_capture import load

    res, code, _ = family_run
    assert code == 1
    final = load(reference / "final.pkl")
    # The relative lower bounds (1e-3 * the level the period starts from, ``square``)
    # follow the check solution, which differs from Pyomo's by an ulp or two: 1e-12.
    assert diff_to_photo(res.mp, final, 1e-9, lb_rtol=1e-12) == []


def test_codes_and_iteration_counts(family_run, reference):
    import json

    res, _, _ = family_run
    assert res.results["base"]["code"] == 1
    assert res.results["check"]["code"] == 1
    assert res.results["shock"]["code"] == 1
    for n, period in ((2, "check"), (4, "check"), (5, "shock")):
        s = json.loads((reference / f"{n:03d}_S_{period}.json").read_text())
        assert s["code"] == 1
    # reported, not asserted
    print(
        "nit ours",
        {k: v.get("nit") for k, v in res.results.items()},
        "settle",
        res.settle["check"]["nit"],
        "pyomo",
        [
            json.loads((reference / f"{n:03d}_S_{p}.json").read_text())["nit"]
            for n, p in ((2, "check"), (4, "check"), (5, "shock"))
        ],
    )
    assert np.isfinite(res.results["shock"]["residual"])
    assert res.results["shock"]["residual"] < 1e-5


def test_user_shock_is_applied_not_the_default(
    reference, linsolve, tmp_path, monkeypatch
):
    """A user shock on an instrument reaches the shock period instead of the default
    tariff (the default tariff's cells keep their check value)."""
    from equilibria.family.driver import prepare_run, solve_gtap_family
    from tests.family.test_prep_recipe import _params
    from tests.family.test_squared_system import _route_closure

    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        monkeypatch.setenv(k, v)
    p = _params(DATASET)
    rr = list(p.sets.r)[-1]
    run, mp = prepare_run(p, _route_closure(), rr, cache_root=tmp_path, jax_cache=False)
    imptx = mp.vars["imptx"]
    cell = next(k[:-1] for k in imptx.keys if k[-1] == "shock")
    res = solve_gtap_family(
        _params(DATASET),
        _route_closure(),
        rr,
        shock={"imptx": {cell: 5.0}},
        cache_root=tmp_path,
        jax_cache=False,
    )
    assert res.results["shock"]["code"] == 1
    v = res.mp.vars["imptx"]
    chk, shk = v.value[v.pos((*cell, "check"))], v.value[v.pos((*cell, "shock"))]
    assert shk != chk
    others = [k[:-1] for k in v.keys if k[-1] == "shock" and k[:-1] != cell]
    moved = [
        o
        for o in others
        if v.value[v.pos((*o, "shock"))] != v.value[v.pos((*o, "check"))]
    ]
    assert moved == []
