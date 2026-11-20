"""GTAP(backend="family"): the template builds the family model (gtap7_3x3)."""

from __future__ import annotations

import dataclasses
import gc
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
def env(monkeypatch, tmp_path):
    load_params(DS)
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        monkeypatch.setenv(k, v)
    return monkeypatch, tmp_path


def _gtap(tmp_path, **kw):
    from equilibria.templates.gtap.template import GTAP

    p = _params(DS)
    return GTAP(
        p,
        _route_closure(),
        backend="family",
        cache_root=tmp_path / "records",
        jax_cache=False,
        **kw,
    )


def _same_arrays(a, b) -> None:
    assert a.vars.keys() == b.vars.keys() and a.rows.keys() == b.rows.keys()
    for name, v in a.vars.items():
        w = b.vars[name]
        for f in ("value", "fixed", "lb", "ub"):
            x, y = getattr(v, f), getattr(w, f)
            assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), (name, f)
    for name, r in a.rows.items():
        assert np.array_equal(r.active, b.rows[name].active), name


def test_family_build_matches_prepare_run(env) -> None:
    from equilibria.family.driver import calibrate_base_cached, prepare_run

    _, tmp_path = env
    model = _gtap(tmp_path, base_calibrated=True).build(("base", "check", "shock"))
    p = _params(DS)
    run, mp = prepare_run(
        p,
        _route_closure(),
        list(p.sets.r)[-1],
        cache_root=tmp_path / "records",
        jax_cache=False,
    )
    settled, _ = calibrate_base_cached(mp, run)
    want = dataclasses.replace(mp, settled_seed=settled, base_calibrated=True)
    _same_arrays(model.mp, want)
    assert model.settled_seed == settled and model.base_calibrated
    assert model.backend == "family"


_UNSUPPORTED = [
    ({"mode": "altertax"}, "altertax"),
    ({"solve_base": True}, "solve_base"),
    ({"seed_from_prior": True}, "seed_from_prior"),
    ({"mute_welfare": False}, "mute_welfare"),
    ({"base_calibrated": True, "solve_check": False}, "solve_check"),
    ({"shock_steps": (0.5, 1.0)}, "shock_steps"),
    ({"shock_method": "johansen"}, "johansen"),
]


@pytest.mark.parametrize(("options", "named"), _UNSUPPORTED, ids=lambda x: str(x))
def test_family_refuses_before_any_solve(env, options, named) -> None:
    from equilibria.core.exercise import Exercise
    from equilibria.templates.gtap.recipe import gtap_static

    _, tmp_path = env
    calls: list[Any] = []

    def spy(model: Any) -> str:
        calls.append(model)
        return "optimal"

    template = _gtap(tmp_path)
    with pytest.raises(NotImplementedError, match=named):
        Exercise(template, gtap_static(**options, solver=spy)).run()
    assert calls == []


def test_family_refuses_overwrite_hook(env) -> None:
    from equilibria.blocks.gtap.overwrite import overwrite

    monkeypatch, tmp_path = env
    monkeypatch.setattr(type(overwrite), "active", lambda self: True)
    with pytest.raises(NotImplementedError, match="@overwrite"):
        _gtap(tmp_path).build(("base", "check", "shock"))


def test_trade_balance_refused_on_pyomo() -> None:
    from equilibria.templates.gtap.recipe import gtap_static

    recipe = gtap_static(trade_balance="nominal")
    first = recipe.plan(recipe.periods[0]).before[0]
    ctx = SimpleNamespace(
        model=SimpleNamespace(backend="pyomo"),
        template=SimpleNamespace(closure=SimpleNamespace(savf_flag="capFix")),
    )
    with pytest.raises(NotImplementedError, match="trade_balance"):
        first(cast("Any", ctx))


def test_trade_balance_needs_capfix() -> None:
    from equilibria.templates.gtap.recipe import gtap_static

    recipe = gtap_static(trade_balance="ratio")
    first = recipe.plan(recipe.periods[0]).before[0]
    ctx = SimpleNamespace(
        model=SimpleNamespace(backend="family"),
        template=SimpleNamespace(closure=SimpleNamespace(savf_flag="capFlex")),
    )
    with pytest.raises(NotImplementedError, match="capFix"):
        first(cast("Any", ctx))


def test_bad_trade_balance() -> None:
    from equilibria.templates.gtap.recipe import gtap_static

    with pytest.raises(ValueError, match="trade_balance"):
        gtap_static(trade_balance="real")


def test_two_backends_in_one_process(env) -> None:
    from equilibria.core.exercise import Exercise
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP

    _, tmp_path = env
    gc.unfreeze()
    fam = _gtap(tmp_path, base_calibrated=True)
    ex = Exercise(fam, gtap_static(base_calibrated=True, solve_check=True))
    res = ex.run()
    assert res["shock"]["code"] == "optimal"
    assert gc.get_freeze_count() == 0
    assert cast("Any", ex.model).run.mumps == {}
    p = _params(DS)
    py = Exercise(GTAP(p, _route_closure()), gtap_static()).run()
    assert py["shock"]["code"] == "optimal"
    assert gc.get_freeze_count() == 0


def test_settle_through_core_equals_calibrate_base_arrays(env) -> None:
    """The settled seed comes from the core recipe (settle_only), never from the
    old array recipe, and equals what calibrate_base_arrays gives, key by key."""
    from equilibria.family import driver as fd

    monkeypatch, tmp_path = env
    p = _params(DS)
    run, mp = fd.prepare_run(
        p,
        _route_closure(),
        list(p.sets.r)[-1],
        cache_root=tmp_path / "records",
        jax_cache=False,
    )
    want, _ = fd.calibrate_base_arrays(mp, run)

    def old_route(*a: Any, **k: Any) -> Any:
        raise AssertionError("the build settled through the old array recipe")

    monkeypatch.setattr(fd, "calibrate_base_arrays", old_route)
    model = _gtap(tmp_path, base_calibrated=True).build(("base", "check", "shock"))
    assert model.settled_seed == want


def test_settle_cache_hit_skips_the_solves(env) -> None:
    from equilibria.family import period_newton

    monkeypatch, tmp_path = env
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path / "seeds"))
    calls: list[int] = []
    real = period_newton._newton

    def counted(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(period_newton, "_newton", counted)
    first = _gtap(tmp_path, base_calibrated=True).build(("base", "check", "shock"))
    assert calls, "the counter must see the settle solves of a cold build"
    calls.clear()
    second = _gtap(tmp_path, base_calibrated=True).build(("base", "check", "shock"))
    assert calls == []
    assert second.settled_seed == first.settled_seed
