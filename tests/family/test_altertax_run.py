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
from equilibria.family.altertax.tolerances import CHECK_TOL
from tests.family._prep import ROOT, _newton_mumps

pytest.importorskip("pyomo")


def _eq_vars() -> tuple[str, ...]:
    from equilibria.family.altertax.compare import load_compare_script

    return load_compare_script().EQ_VARS


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
    """``solve_altertax`` on nus333 with the Pyomo entry points spied (its own solve: the
    spies must be installed before it runs, which the shared ``altax_run`` cannot do)."""
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


def test_bad_shock_cell_raises(monkeypatch):
    """Raises on the cell itself, before ``solve_gtap_family`` is reached."""
    import equilibria.family.altertax.run as run_mod

    def no_solve(*a, **k):
        raise AssertionError("solve_gtap_family was reached")

    monkeypatch.setattr(run_mod, "solve_gtap_family", no_solve)
    with pytest.raises(ValueError, match="'NOPE' is not in the r set"):
        solve_altertax(
            nus333_dir(), shock={"prdtx_rai": {("NOPE", "AGR", "USA"): 0.0012}}
        )


def test_altertax_params_have_esubi_zero():
    """GTAPv7 E_qia (gtapv7.tab ~1717): qia(c,r) = qinv(r), Leontief; ESBI is not read."""
    from equilibria.family.altertax.run import load_altertax_params

    p = load_altertax_params(nus333_dir())
    assert p.elasticities.esubi
    assert all(v == 0.0 for v in p.elasticities.esubi.values())
