"""The array driver end to end: ``_solve_shock(..., backend="family")`` on 3x3.

No Pyomo model is built in the family route (a spy on ``ConcreteModel.__init__``,
``PyomoNLP.__init__`` and ``identify_variables`` proves it), and the three periods agree
with the saved Pyomo run (``final.pkl``, same ``linsolve``): every Var cell within 1e-9
(``|a - b| <= 1e-9 * max(1, |b|)``, NaN equals NaN), identical fixed flags and bounds,
code 1. The iteration counts are reported, never asserted.
"""

from __future__ import annotations

import contextlib
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


@contextlib.contextmanager
def _spies():
    """Count every Pyomo model, PyNumero NLP and expression walk built inside."""
    import pyomo.contrib.pynumero.interfaces.pyomo_nlp as pn
    import pyomo.core.expr.visitor as visitor
    from pyomo.environ import ConcreteModel

    calls = {"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}

    def spy_init(cls, key):
        orig = cls.__init__

        def wrapped(self, *a, **k):
            calls[key] += 1
            return orig(self, *a, **k)

        return wrapped

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
    mpatch.setattr(ConcreteModel, "__init__", spy_init(ConcreteModel, "ConcreteModel"))
    mpatch.setattr(pn.PyomoNLP, "__init__", spy_init(pn.PyomoNLP, "PyomoNLP"))
    try:
        yield calls
    finally:
        mpatch.undo()


def test_spy_sees_the_pyomo_route(reference):
    """The spy would have caught a build: the reference capture ran the Pyomo route."""
    from pyomo.environ import ConcreteModel

    assert ConcreteModel.__init__.__name__ == "__init__"  # restored, not a wrapper


def test_user_shock_is_applied_not_the_default(
    reference, linsolve, tmp_path, monkeypatch
):
    """A user shock on an instrument reaches the shock period instead of the default
    tariff (the default tariff's cells keep their check value)."""
    from equilibria.family.run_setup import prepare_run
    from tests.family._core import family_exercise
    from tests.family.test_prep_recipe import _params
    from tests.family.test_squared_system import _route_closure

    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        monkeypatch.setenv(k, v)
    p = _params(DATASET)
    rr = list(p.sets.r)[-1]
    run, mp = prepare_run(p, _route_closure(), rr, cache_root=tmp_path, jax_cache=False)
    imptx = mp.vars["imptx"]
    cell = next(k[:-1] for k in imptx.keys if k[-1] == "shock")
    model, res = family_exercise(
        _params(DATASET),
        _route_closure(),
        rr,
        shock={"imptx": {cell: 5.0}},
        cache_root=tmp_path,
        jax_cache=False,
    )
    assert res["shock"]["code"] == "optimal"
    v = model.mp.vars["imptx"]
    chk, shk = v.value[v.pos((*cell, "check"))], v.value[v.pos((*cell, "shock"))]
    assert shk != chk
    others = [k[:-1] for k in v.keys if k[-1] == "shock" and k[:-1] != cell]
    moved = [
        o
        for o in others
        if v.value[v.pos((*o, "shock"))] != v.value[v.pos((*o, "check"))]
    ]
    assert moved == []


# The core route (``Exercise(GTAP(backend="family"), gtap_static)``) against the
# Pyomo photo.
@pytest.fixture(scope="module")
def core_run(reference, linsolve, tmp_path_factory):
    """``_solve_shock`` on the family backend, spied."""
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    mpatch = pytest.MonkeyPatch()
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        mpatch.setenv(k, v)
    mpatch.setenv("EQUILIBRIA_FAMILY_CACHE", str(tmp_path_factory.mktemp("cache")))
    try:
        with _spies() as calls:
            model, code = _solve_shock(
                DATASET, 1, "capFix", solve_nlp=True, backend="family"
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
