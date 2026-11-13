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


def test_trade_balance_nominal(altax_run):
    """GEMPACK's ``del_tbal("USA") = 0`` (altax.exp): the nominal balance of the
    non-residual region is held, so savf[USA] is the same in check and shock, to the bit,
    while pigbl (which a real ``savf = pigbl * savf_bar`` would carry into savf) moves."""
    v = altax_run.result.mp.vars
    savf, pigbl = v["savf"], v["pigbl"]

    def at(var, key, period):
        return float(var.value[var.keys.index((*key, period))])

    assert at(savf, ("USA",), "shock") == at(savf, ("USA",), "check")
    assert at(pigbl, (), "shock") != at(pigbl, (), "check")


def test_default_path_keeps_real_savf(env):
    """Default off: the shock's eq_savf[USA] stays active (savf = pigbl * savf_bar, real),
    and with ``nominal_trade_balance`` it is off; the option is refused outside capFix."""
    import inspect

    from equilibria.family.altertax.closure import (
        ALTAX_RESIDUAL_REGION,
        altertax_closure,
    )
    from equilibria.family.altertax.run import load_altertax_params
    from equilibria.family.driver import solve_gtap_family

    param = inspect.signature(solve_gtap_family).parameters["nominal_trade_balance"]
    assert param.default is False
    res = solve_gtap_family(
        load_altertax_params(nus333_dir()),
        altertax_closure(),
        ALTAX_RESIDUAL_REGION,
        shock=ALTAX_SHOCK,
        cache_root=env.mktemp("default_cache"),
    )
    eq = res.mp.rows["eq_savf"]
    assert bool(eq.active[eq.keys.index(("USA", "shock"))])


def test_nominal_balance_drops_the_row(altax_run):
    eq = altax_run.result.mp.rows["eq_savf"]
    assert not bool(eq.active[eq.keys.index(("USA", "shock"))])


def test_nominal_balance_needs_capfix():
    from equilibria.family.altertax.closure import altertax_closure
    from equilibria.family.driver import solve_gtap_family

    closure = altertax_closure().model_copy(update={"savf_flag": "capFlex"})
    with pytest.raises(NotImplementedError, match="nominal_trade_balance"):
        solve_gtap_family(object(), closure, "ROW", nominal_trade_balance=True)


def test_altertax_solves_converge_to_1e10_whatever_the_env(env, monkeypatch):
    """The ALTERTAX solves carry their own tolerance (scaled residual 1e-10, absolute
    1e-12): the env default of the measure script (RELTOL 1e-6) stopped the shock ~6e-7
    from the root and left small cancelling indices (pfactor[USA] of 3x4) off by 3e-6 pp.
    Other routes keep reading the env: ``FamilyRun.newton_tol`` defaults to ``None``."""
    import equilibria.solver.newton_tr as nt
    from equilibria.family.driver import FamilyRun

    assert FamilyRun.__dataclass_fields__["newton_tol"].default is None
    monkeypatch.setenv("EQUILIBRIA_GTAP_TR_RELTOL", "1e-6")
    monkeypatch.setenv("EQUILIBRIA_GTAP_TR_FTOL", "1e-7")
    seen = []
    orig = nt.newton_tr

    def spy(*a, **k):
        seen.append((k.get("reltol"), k.get("ftol")))
        return orig(*a, **k)

    monkeypatch.setattr(nt, "newton_tr", spy)
    run = solve_altertax(nus333_dir(), cache_root=env.mktemp("altax_tol"))
    assert run.codes == {"check": 1, "shock": 1}
    assert seen and set(seen) == {(1e-10, 1e-12)}
