"""newton_tr(): Newton-TR as a recipe solver for the family GTAP model (3x3)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from tests.family._data import load_params
from tests.family._prep import _newton_mumps
from tests.family.test_sp_arrays import _params
from tests.family.test_squared_system import _route_closure

DS = "gtap7_3x3"


@pytest.fixture
def env(monkeypatch):
    load_params(DS)
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        monkeypatch.setenv(k, v)
    return monkeypatch


def _check_ready(tmp_path):
    """The run and the arrays the core hands the check solve: the base flown, then
    the check's steps without its solve."""
    import dataclasses

    from equilibria.core.driver import fly_period
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP

    t = GTAP(
        _params(DS),
        _route_closure(),
        backend="family",
        cache_root=tmp_path,
        jax_cache=False,
    )
    recipe = gtap_static(base_calibrated=False, solve_check=True)
    model = t.build(recipe.names)
    fly_period(model, recipe, "base", template=t)
    check = next(p for p in recipe.periods if p.name == "check")
    plans = dict(recipe.plans)
    plans[check.kind] = dataclasses.replace(plans[check.kind], solver=None, after=())
    fly_period(model, dataclasses.replace(recipe, plans=plans), "check", template=t)
    return t._family_run, model.mp


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


def test_solves_the_check_like_the_family_newton(env, tmp_path) -> None:
    from equilibria.family import period_newton as fd
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
    from equilibria.family import period_newton as fd
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
