"""The structure of a solve is built once per pattern and reused (speed cut C).

A run solves the check twice (the settle, then the calibrated check) and the shock
again in the decomposition path, each time on the same squared pattern. The pairing
(adjacency and matching), the compiled family systems, the fscale rows and the MUMPS
symbolic analysis depend on the pattern only, so the second build of a pattern takes
them from the first and updates the values. Gate: the reused system equals a fresh
build (start point, F and J within 1e-12, fscale exactly) and so does the solution.
"""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pytest

from tests.family._data import load_params
from tests.family._prep import _newton_mumps
from tests.family.test_prep_recipe import _params
from tests.family.test_squared_system import _route_closure

DS = "gtap7_3x3"
TOL = 1e-12


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def _fresh(run):
    """The same run context with no structure kept from earlier solves."""
    return dataclasses.replace(run, cell_index=None, adjacency={}, mumps={})


@pytest.fixture(scope="module")
def settled(monkeypatch_module, tmp_path_factory):
    """After the settle (which built the check's structure) and the calibrated base:
    the run, the arrays the check starts from, the check's recipe output and the
    arrays the run started from."""
    from equilibria.family.driver import (
        _period_recipe,
        _solve_period,
        calibrate_base_arrays,
        prepare_run,
    )

    load_params(DS)
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        monkeypatch_module.setenv(k, v)
    p = _params(DS)
    run, mp = prepare_run(
        p,
        _route_closure(),
        list(p.sets.r)[-1],
        cache_root=tmp_path_factory.mktemp("records"),
        jax_cache=False,
    )
    settled, _ = calibrate_base_arrays(mp, run)
    cur = dataclasses.replace(mp, settled_seed=settled, base_calibrated=True)
    cur, _ = _solve_period(run, cur, "base", base_calibrated=True, solve_check=True)
    mp1 = _period_recipe(run, cur, "check", base_calibrated=True, solve_check=True)
    return run, cur, mp1, mp


def _fscale_by_names(ss) -> np.ndarray:
    """The per-row scale as ``SquaredSystem.fscale`` computed it before the cut: a
    Python walk of every record's leaves, rows matched by name."""
    from equilibria.family.fscale import record_fscale
    from equilibria.family.squared_system import is_cross_period

    pos = {r: k for k, r in enumerate(ss._row_names)}
    out = np.ones(len(ss.x0))
    for records, values, free, _k, row_name, single in ss._fscale_ctx:
        recs = {f: r for f, r in records.items() if not (single and is_cross_period(f))}
        names = record_fscale(
            recs, values, free, keep=lambda n, rn=row_name: rn(n) in pos
        )
        for n, v in names.items():
            out[pos[row_name(n)]] = v
    return out


def _same_system(a, b) -> None:
    ss, sq = a
    ss_f, sq_f = b
    assert list(sq.row_names) == list(sq_f.row_names)
    assert list(sq.col_names) == list(sq_f.col_names)
    np.testing.assert_allclose(ss.x0, ss_f.x0, rtol=0, atol=TOL)
    rng = np.random.default_rng(0)
    x1 = ss_f.x0 * (1.0 + 1e-3 * rng.standard_normal(len(ss_f.x0)))
    for x in (ss_f.x0, x1):
        np.testing.assert_allclose(ss.F(x), ss_f.F(x), rtol=0, atol=TOL)
        ja, jb = ss.J(x).tocsr(), ss_f.J(x).tocsr()
        assert ja.shape == jb.shape
        assert abs(ja - jb).max() <= TOL
    assert np.array_equal(ss.fscale(), ss_f.fscale())
    assert np.array_equal(ss_f.fscale(), _fscale_by_names(ss_f))


def test_check_reuses_the_settle_structure(settled, monkeypatch):
    """The check is squared on the settle's pattern: no new matching and no new
    compile of the family systems, and the system equals a fresh build."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family import prep_square
    from equilibria.family.driver import _period_system

    run, _cur, mp1, _ = settled

    def refused(*_a, **_k):
        raise AssertionError("rebuilt a structure the settle already built")

    with monkeypatch.context() as m:
        m.setattr(prep_square, "match_rows", refused)
        m.setattr(FamilyBackend, "system", refused)
        sq, _, ss = _period_system(run, mp1, "check")
    sq_f, _, ss_f = _period_system(_fresh(run), mp1, "check")
    _same_system((ss, sq), (ss_f, sq_f))


def test_a_changed_zero_pattern_compiles_again(settled, monkeypatch):
    """A fixed cell that becomes 0 folds terms away: that pattern is compiled anew
    (not taken from the cache), and equals a fresh build."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.driver import _period_system

    run, _cur, mp1, _ = settled
    mp2 = mp1.copy()
    name = next(
        n
        for n in sorted(mp2.exogenous)
        if n in run.backend.var_domains
        and n in mp2.vars
        and (
            (mp2.vars[n].period == "check")
            & mp2.vars[n].fixed
            & (mp2.vars[n].value != 0.0)
        ).any()
    )
    v = mp2.vars[name]
    i = int(np.flatnonzero((v.period == "check") & v.fixed & (v.value != 0.0))[0])
    v.value[i] = 0.0
    compiled = []
    orig = FamilyBackend.system

    def counted(self, layout):
        compiled.append(self)
        return orig(self, layout)

    with monkeypatch.context() as m:
        m.setattr(FamilyBackend, "system", counted)
        sq, _, ss = _period_system(run, mp2, "check")
    assert run.backend in compiled
    sq_f, _, ss_f = _period_system(_fresh(run), mp2, "check")
    _same_system((ss, sq), (ss_f, sq_f))


def test_check_solution_equals_a_fresh_build(settled):
    """The check solved on the reused structure ends where a run with nothing kept
    ends: every level within 1e-12 (relative above 1)."""
    from equilibria.family.driver import _solve_period

    run, cur, _, _ = settled
    a, ra = _solve_period(run, cur, "check", base_calibrated=True, solve_check=True)
    b, rb = _solve_period(
        _fresh(run), cur, "check", base_calibrated=True, solve_check=True
    )
    assert ra["code"] == rb["code"] == 1
    for name, va in a.vars.items():
        vb = b.vars[name].value
        ok = np.isnan(va.value) == np.isnan(vb)
        assert ok.all(), name
        d = np.abs(np.nan_to_num(va.value) - np.nan_to_num(vb))
        assert (d <= TOL * np.maximum(1.0, np.abs(np.nan_to_num(vb)))).all(), name


@pytest.mark.skipif(
    importlib.util.find_spec("mumps") is None, reason="MUMPS reuse needs pymumps"
)
def test_check_reuses_the_settle_mumps_analysis(settled, monkeypatch):
    """Under MUMPS the check factors on the settle's symbolic analysis (same pattern):
    no new analysis, and the same solution as a run with nothing kept (1e-12)."""
    from pyomo.contrib.pynumero.linalg.mumps_interface import (
        MumpsCentralizedAssembledLinearSolver as Mumps,
    )

    from equilibria.family.driver import _solve_period, calibrate_base_arrays

    monkeypatch.setenv("EQUILIBRIA_GTAP_TR_LINSOLVE", "mumps")
    run0, cur, _, mp0 = settled
    run = _fresh(run0)
    calibrate_base_arrays(mp0, run)  # the settle under MUMPS: keeps its analysis
    count = {"n": 0}
    orig = Mumps.do_symbolic_factorization

    def counted(self, *a, **k):
        count["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(Mumps, "do_symbolic_factorization", counted)
    a, ra = _solve_period(run, cur, "check", base_calibrated=True, solve_check=True)
    assert count["n"] == 0
    b, rb = _solve_period(
        _fresh(run), cur, "check", base_calibrated=True, solve_check=True
    )
    assert count["n"] == 1
    assert ra["code"] == rb["code"] == 1
    for name, va in a.vars.items():
        vb = np.nan_to_num(b.vars[name].value)
        d = np.abs(np.nan_to_num(va.value) - vb)
        assert (d <= TOL * np.maximum(1.0, np.abs(vb))).all(), name
