"""Opt-in disk cache of the family settle (``EQUILIBRIA_FAMILY_SETTLE_CACHE=1``).

The settle is a pure function of the data, the closure, the residual region, the ref
GDX, the build + solve code, the Newton tolerances, the ``EQUILIBRIA_*`` environment
and the nominal-savf rule. A key over all of them lets a repeated run skip it; off by
default, a converged settle only, and any change in those inputs misses.
"""

from __future__ import annotations

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
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path / "seeds"))
    return monkeypatch, tmp_path


def _template(tmp_path, **kw):
    from equilibria.templates.gtap.template import GTAP

    return GTAP(
        _params(DS),
        _route_closure(),
        backend="family",
        base_calibrated=True,
        cache_root=tmp_path / "records",
        jax_cache=False,
        **kw,
    )


def _build(tmp_path, **kw):
    """``(template, settled point)`` of a family build with the base calibrated."""
    t = _template(tmp_path, **kw)
    model = t.build(("base", "check", "shock"))
    return t, model.settled_seed


def _seed_files(tmp_path):
    d = tmp_path / "seeds"
    return sorted(d.glob("family-seed-*.json")) if d.exists() else []


def _counting(monkeypatch):
    """Counts the settles: the core flies ``gtap_static(settle_only=True)`` for each."""
    from equilibria.templates.gtap import recipe

    calls = []
    real = recipe.gtap_static

    def counted(*a, **k):
        if k.get("settle_only"):
            calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(recipe, "gtap_static", counted)
    return calls


def test_off_by_default_settles_and_writes_nothing(env):
    monkeypatch, tmp_path = env
    monkeypatch.delenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", raising=False)
    calls = _counting(monkeypatch)
    for _ in range(2):
        t, _ = _build(tmp_path)
        assert t._family_run.settle_key is None
    assert len(calls) == 2 and _seed_files(tmp_path) == []


def test_second_run_loads_the_same_settled_point(env):
    monkeypatch, tmp_path = env
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    calls = _counting(monkeypatch)
    _, settled = _build(tmp_path)
    assert len(calls) == 1
    assert len(_seed_files(tmp_path)) == 1
    _, settled2 = _build(tmp_path)
    assert len(calls) == 1  # no second settle
    assert settled2 == settled  # exact: JSON keeps every float bit


@pytest.mark.parametrize(
    "change",
    ["newton_tol", "linsolve_env", "nominal_savf", "family_code"],
)
def test_any_input_change_misses(env, change):
    from equilibria.family import run_setup

    monkeypatch, tmp_path = env
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    t, _ = _build(tmp_path)
    key = run_setup._settle_key(t._family_run)
    if change == "nominal_savf":
        p = _params(DS)
        run2, _ = run_setup.prepare_run(
            p,
            _route_closure(),
            list(p.sets.r)[-1],
            cache_root=tmp_path / "records",
            jax_cache=False,
        )
        run2.nominal_savf = tuple(run2.regions[:1])
        assert run_setup._settle_key(run2) != key
        return
    calls = _counting(monkeypatch)
    kw = {}
    if change == "newton_tol":
        kw["newton_tol"] = (1e-11, 1e-11)
    elif change == "linsolve_env":
        monkeypatch.setenv("EQUILIBRIA_GTAP_TR_LINSOLVE", "scipy")
        p = _params(DS)
        run2, _ = run_setup.prepare_run(
            p,
            _route_closure(),
            list(p.sets.r)[-1],
            cache_root=tmp_path / "records",
            jax_cache=False,
        )
        # "scipy" is keyed only; it need not solve here
        assert run_setup._settle_key(run2) != key
        return
    elif change == "family_code":
        monkeypatch.setattr(run_setup, "_family_code_digest", lambda: "other-code")
    t2, _ = _build(tmp_path, **kw)
    assert run_setup._settle_key(t2._family_run) != key
    assert len(calls) == 1


def test_a_failed_settle_is_not_saved(env):
    from equilibria.core import driver

    monkeypatch, tmp_path = env
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    real = driver.fly

    def failing(*a, **k):
        res = real(*a, **k)
        res["check"] = {**res["check"], "code": "code 2"}
        return res

    monkeypatch.setattr(driver, "fly", failing)
    _build(tmp_path)
    assert _seed_files(tmp_path) == []


def test_run_from_the_cached_settle_matches_a_fresh_run(env):
    from tests.family._core import family_exercise

    monkeypatch, tmp_path = env
    p = _params(DS)
    rr = list(p.sets.r)[-1]

    def solve():
        return family_exercise(
            _params(DS),
            _route_closure(),
            rr,
            cache_root=tmp_path / "records",
            jax_cache=False,
        )

    fresh, fresh_res = solve()
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    solve()  # writes
    calls = _counting(monkeypatch)
    hit, hit_res = solve()
    assert calls == []
    assert {k: r["code"] for k, r in hit_res.items()} == {
        k: r["code"] for k, r in fresh_res.items()
    }
    for name, v in fresh.mp.vars.items():
        w = hit.mp.vars[name]
        np.testing.assert_allclose(w.value, v.value, rtol=1e-9, atol=1e-9, err_msg=name)


def test_the_seed_cache_switch_turns_it_off(env):
    monkeypatch, tmp_path = env
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    calls = _counting(monkeypatch)
    for _ in range(2):
        _build(tmp_path)
    assert len(calls) == 2 and _seed_files(tmp_path) == []
