"""newton_tr(): Newton-TR as a recipe solver for the family GTAP model (3x3)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from tests.family._data import load_params
from tests.family._prep import _newton_mumps
from tests.family.test_prep_recipe import _params
from tests.family.test_squared_system import _route_closure

DS = "gtap7_3x3"


@pytest.fixture
def env(monkeypatch):
    load_params(DS)
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        monkeypatch.setenv(k, v)
    return monkeypatch


def _check_ready(tmp_path):
    """The run and the arrays the old route hands its check solve (base not solved,
    check recipe applied)."""
    from equilibria.family import driver as fd

    p = _params(DS)
    run, mp = fd.prepare_run(
        p, _route_closure(), list(p.sets.r)[-1], cache_root=tmp_path, jax_cache=False
    )
    mp, _ = fd._solve_period(run, mp, "base", base_calibrated=False, solve_check=True)
    mp1 = fd._period_recipe(run, mp, "check", base_calibrated=False, solve_check=True)
    return run, mp1


def _model(run, mp):
    from equilibria.family.gtap_periods import GTAPFamilyPeriodModel

    m = GTAPFamilyPeriodModel(run, mp.copy())
    m.stage(run.closure, run.params, gtap_mode=True, period="check")
    return m


def _same_arrays(a, b) -> None:
    for name, v in a.vars.items():
        w = b.vars[name]
        for f in ("value", "fixed", "lb", "ub"):
            x, y = getattr(v, f), getattr(w, f)
            assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), (name, f)
    for name, r in a.rows.items():
        assert np.array_equal(r.active, b.rows[name].active), name


def test_solves_the_check_like_solve_period(env, tmp_path) -> None:
    from equilibria.family import driver as fd
    from equilibria.family.post_solve import writeback
    from equilibria.solver.newton_gtap import newton_tr

    run, mp1 = _check_ready(tmp_path)
    sq, mp_sq, ss = fd._period_system(run, mp1, "check")
    res, resid, code = fd._newton(ss, sq)
    writeback(mp_sq, sq, res.x)
    m = _model(run, mp1)
    assert newton_tr()(m) == "optimal" and code == 1
    _same_arrays(m.mp, mp_sq)
    assert m.residual("check") == resid
    assert run.solves[-1]["code"] == 1


def test_failed_newton_reports_code_2(env, tmp_path) -> None:
    from equilibria.family import driver as fd
    from equilibria.solver.newton_gtap import newton_tr

    run, mp1 = _check_ready(tmp_path)
    real = fd._newton

    def failing(ss, sq, border=None):
        res, resid, _ = real(ss, sq, border)
        return res, resid, 2

    env.setattr(fd, "_newton", failing)
    m = _model(run, mp1)
    assert newton_tr()(m) == "code 2"
    assert m.last_solve["termination_code"] == 2


def test_default_solver_follows_the_backend(monkeypatch) -> None:
    from equilibria.solver import newton_gtap, path_gtap
    from equilibria.templates.gtap.recipe import gtap_static

    called: list[Any] = []
    monkeypatch.setattr(
        newton_gtap, "newton_tr", lambda **kw: lambda m: called.append(m) or "optimal"
    )

    def no_path() -> Any:
        raise AssertionError("PATH loaded for a family model")

    monkeypatch.setattr(path_gtap, "_load_path_capi", no_path)
    fake = cast("Any", SimpleNamespace(backend="family"))
    solver = gtap_static().plan(gtap_static().periods[1]).solver
    assert solver is not None
    assert solver(fake) == "optimal" and called == [fake]


def test_path_refuses_family_model() -> None:
    from equilibria.solver.path_gtap import path

    with pytest.raises(TypeError, match="family"):
        path()(SimpleNamespace(backend="family", staged=None))
