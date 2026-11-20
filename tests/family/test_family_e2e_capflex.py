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


def _closure_flex():
    from tests.family._oracle import comp_stat_closure

    return comp_stat_closure(True, SAVF)


@pytest.mark.needs_gdxdump  # reads rore/rorg from the reference GDX
def test_cache_key_differs_from_capfix(stamped):
    from equilibria.family.disk_cache import cache_key
    from tests.family._oracle import comp_stat_closure
    from tests.family.test_sp_arrays import _params

    p = _params(DATASET)
    rr = list(p.sets.r)[-1]
    k_fix = cache_key(p, comp_stat_closure(True), rr)
    k_flex = cache_key(stamped, _closure_flex(), rr)
    assert k_fix is not None and k_flex is not None and k_fix != k_flex


@pytest.fixture(scope="module")
def stamped():
    from tests.family._oracle import _stamped_copy
    from tests.family.test_sp_arrays import _params

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
    import equilibria.templates.gtap.gtap_block_model as bm

    def no_twin_solve(*a, **k):
        raise AssertionError("_calibrate_capflex_risk (Pyomo twin solve) was called")

    mpatch.setattr(bm, "_calibrate_capflex_risk", no_twin_solve)
    try:
        with _spies() as calls:
            model, code = _solve_shock(
                DATASET, IFSUB, SAVF, solve_nlp=True, backend="family"
            )
    finally:
        mpatch.undo()
    return model, code, calls


@pytest.mark.needs_gdxdump  # reads rore/rorg from the reference GDX
def test_core_route_builds_no_pyomo_model(core_run):
    _, _, calls = core_run
    assert calls == {"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}


@pytest.mark.needs_gdxdump  # reads rore/rorg from the reference GDX
def test_core_route_every_cell_equals_final_pkl(core_run, reference):
    from equilibria.family.prep_capture import load

    model, code, _ = core_run
    assert code == 1
    final = load(reference / "final.pkl")
    assert diff_to_photo(model.mp, final, 1e-9, lb_rtol=1e-9) == []
