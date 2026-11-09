"""ALTERTAX on nus333 through the array driver: base, check and shock, no Pyomo.

The check period must reproduce the base (every ``EQ_VARS`` cell within
``CHECK_TOL * max(1, |base|)``, see below); the family route builds no Pyomo model (spies as in
``test_family_e2e_capflex.py``); a shock cell outside the model raises before any solve.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from equilibria._local_refs import nus333_dir
from equilibria.family.altertax.levels import family_levels
from equilibria.family.altertax.run import ALTAX_SHOCK, solve_altertax
from tests.family._prep import ROOT, _newton_mumps

pytest.importorskip("pyomo")

# The brief asked for 1e-9. Measured: the benchmark carries ~1e-8 of float32 noise in its
# shares and the nests at sigma = 1.01 amplify it by 1 / |1 - sigma| = 100, so check
# departs from base by up to 1.2e-6 (xiagg ROW); the amplification falls with |1 - sigma|
# (7e-8 at 1.1, 4e-8 at 2.0, the floor default.prm shows at 6e-8). 5e-6 bounds the
# measured 1.2e-6 and stays orders below any real closure or numeraire mismatch
# (without the esubd guard the same cells departed by 3e-2).
CHECK_TOL = 5e-6


def _eq_vars() -> tuple[str, ...]:
    path = ROOT / "scripts" / "gtap" / "compare_gempack_levels.py"
    spec = importlib.util.spec_from_file_location("compare_gempack_levels", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EQ_VARS


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        mp.setenv(k, v)
    mp.setenv("EQUILIBRIA_FAMILY_CACHE", str(tmp_path_factory.mktemp("cache")))
    yield tmp_path_factory
    mp.undo()


@pytest.fixture(scope="module")
def spied_run(env):
    """``solve_altertax`` on nus333 with the Pyomo entry points spied."""
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
        run = solve_altertax(nus333_dir(), cache_root=env.mktemp("altax_cache"))
    finally:
        mpatch.undo()
    return run, calls


def test_check_reproduces_base(spied_run):
    run, _ = spied_run
    assert run.codes == {"check": 1, "shock": 1}
    lv = family_levels(run.result.mp, _eq_vars())
    bad = []
    for name, cells in lv.items():
        for key, chk in cells.items():
            if key[-1] != "check":
                continue
            base = cells.get((*key[:-1], "base"))
            if base is None or (np.isnan(base) and np.isnan(chk)):
                continue
            if abs(chk - base) > CHECK_TOL * max(1.0, abs(base)):
                bad.append((name, key, base, chk))
    assert not bad, bad[:10]


def test_no_pyomo(spied_run):
    _, calls = spied_run
    assert calls == {"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}


def test_shock_is_the_book_cell():
    assert ALTAX_SHOCK == {"prdtx_rai": {("USA", "AGR", "AGR"): 0.0012}}


def test_bad_shock_cell_raises(env):
    with pytest.raises((KeyError, ValueError)):
        solve_altertax(
            nus333_dir(), shock={"prdtx_rai": {("NOPE", "AGR", "USA"): 0.0012}}
        )
