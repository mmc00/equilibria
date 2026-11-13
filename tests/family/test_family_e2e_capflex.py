"""The array driver end to end on 3x3 with savf_flag=capFlex (ifSUB=1).

Same checks as ``test_family_e2e_ifsub0.py`` against the Pyomo capFlex run captured in
the session (``capture_dir(..., savf="capFlex")``): no Pyomo model is built by the
family route (spy on ``ConcreteModel.__init__``, ``PyomoNLP.__init__`` and
``identify_variables``) and ``_calibrate_capflex_risk`` (Pyomo's twin solve) is never
called; every Var cell of the three periods agrees with ``final.pkl`` (same
``linsolve``) within 1e-9, identical fixed flags and bounds, code 1. Plus:
``MPArrays.build`` has ``eq_savf`` in every region and no ``eq_rorg`` per period,
equal to Pyomo's row names. Iteration counts are reported, never asserted.
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
IFSUB = 1
SAVF = "capFlex"


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
        yield capture_dir(DATASET, linsolve, IFSUB, SAVF)
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

    def no_twin_solve(*a, **k):
        raise AssertionError("_calibrate_capflex_risk (Pyomo twin solve) was called")

    import equilibria.templates.gtap.gtap_block_model as bm

    mpatch.setattr(bm, "_calibrate_capflex_risk", no_twin_solve)
    mpatch.setenv("EQUILIBRIA_FAMILY_CACHE", str(tmp_path_factory.mktemp("cache")))
    mpatch.setenv("EQUILIBRIA_GTAP_BACKEND", "family")
    try:
        from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

        res, code = _solve_shock(DATASET, IFSUB, SAVF, solve_nlp=True)
    finally:
        mpatch.undo()
    return res, code, calls


@pytest.mark.needs_gdxdump  # reads rore/rorg from the reference GDX
def test_family_route_builds_no_pyomo_model(family_run):
    _, _, calls = family_run
    assert calls == {"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}


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


@pytest.mark.needs_gdxdump  # reads rore/rorg from the reference GDX
def test_every_cell_of_the_three_periods_equals_final_pkl(family_run, reference):
    from equilibria.family.prep_capture import load

    res, code, _ = family_run
    assert code == 1
    final = load(reference / "final.pkl")
    # The relative lower bounds (1e-3 * the level the period starts from, ``square``)
    # follow the check solution, which differs from Pyomo's by ~1e-10: the same 1e-9 as the values.
    assert diff_to_photo(res.mp, final, 1e-9, lb_rtol=1e-9) == []


@pytest.mark.needs_gdxdump  # reads rore/rorg from the reference GDX
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


def _closure_flex():
    from tests.family._oracle import comp_stat_closure

    return comp_stat_closure(True, SAVF)


@pytest.mark.needs_gdxdump  # reads rore/rorg from the reference GDX
def test_cache_key_differs_from_capfix(stamped):
    from equilibria.family.disk_cache import cache_key
    from tests.family._oracle import comp_stat_closure
    from tests.family.test_prep_recipe import _params

    p = _params(DATASET)
    rr = list(p.sets.r)[-1]
    k_fix = cache_key(p, comp_stat_closure(True), rr)
    k_flex = cache_key(stamped, _closure_flex(), rr)
    assert k_fix is not None and k_flex is not None and k_fix != k_flex


@pytest.fixture(scope="module")
def stamped():
    from tests.family._oracle import _stamped_copy
    from tests.family.test_prep_recipe import _params

    return _stamped_copy(_params(DATASET), _closure_flex(), DATASET, bool(IFSUB))


@pytest.fixture(scope="module")
def built(stamped, tmp_path_factory):
    from equilibria.family import cross_period
    from equilibria.family.disk_cache import load_or_build
    from equilibria.family.mp_arrays import MPArrays

    p, closure = stamped, _closure_flex()
    rr = list(p.sets.r)[-1]
    root = tmp_path_factory.mktemp("records_flex")
    backend, state = load_or_build(p, closure, rr, root)
    cross = cross_period.load_or_record(
        p, closure, rr, root, sets=backend.sets, var_domains=backend.var_domains
    )
    mp = MPArrays.build(state, backend, cross, params=p)
    return p, closure, rr, backend, state, mp


@pytest.mark.needs_gdxdump  # reads rore/rorg from the reference GDX
def test_mparrays_rows_equal_pyomos_with_eq_savf_everywhere(built, reference):
    from equilibria.family.prep_capture import load

    p, *_, mp = built
    assert "eq_rorg" not in mp.rows
    row = mp.rows["eq_savf"]
    assert row.active.all()
    for period in ("base", "check", "shock"):
        regions = {k[0] for k in row.keys if k[-1] == period}
        assert regions == {str(r) for r in p.sets.r}  # the residual region included
    photo = load(reference / "001_A_base.pkl")
    assert diff_to_photo(mp, photo) == []


@pytest.mark.needs_gdxdump  # reads rore/rorg from the reference GDX
def test_phase_timing_covers_the_run(family_run):
    res, _, _ = family_run
    ph = res.phases
    for name in (
        "compose_state",
        "mp_build",
        "settle_check.newton_J",
        "shock.newton_J",
    ):
        assert ph[name] >= 0.0, name
    assert sum(ph.values()) <= res.timing["total_s"] + 1e-6


@pytest.mark.needs_gdxdump  # reads rore/rorg from the reference GDX
def test_max_relative_difference_per_period_reported(family_run, reference):
    from equilibria.family.prep_capture import load

    res, _, _ = family_run
    final = load(reference / "final.pkl")
    worst = dict.fromkeys(("base", "check", "shock"), 0.0)
    for name, c in final.vars.items():
        v = res.mp.vars[name]
        order = np.array([v.pos(k) for k in c.keys])
        gv = v.value[order]
        ok = np.isfinite(gv) & np.isfinite(c.value)
        rel = np.where(ok, np.abs(gv - c.value) / np.maximum(1.0, np.abs(c.value)), 0.0)
        for i, k in enumerate(c.keys):
            worst[k[-1]] = max(worst[k[-1]], float(rel[i]))
    print("max rel diff per period", worst)
    assert max(worst.values()) <= 1e-9
