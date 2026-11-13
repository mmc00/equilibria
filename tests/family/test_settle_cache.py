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
from tests.family.test_prep_recipe import _params
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


def _run(tmp_path, **kw):
    from equilibria.family.driver import prepare_run

    p = _params(DS)
    return prepare_run(
        p,
        _route_closure(),
        list(p.sets.r)[-1],
        cache_root=tmp_path / "records",
        jax_cache=False,
        **kw,
    )


def _seed_files(tmp_path):
    d = tmp_path / "seeds"
    return sorted(d.glob("family-seed-*.json")) if d.exists() else []


def _counting(monkeypatch):
    from equilibria.family import driver

    calls = []
    real = driver.calibrate_base_arrays

    def counted(mp, run):
        calls.append(1)
        return real(mp, run)

    monkeypatch.setattr(driver, "calibrate_base_arrays", counted)
    return calls


def test_off_by_default_settles_and_writes_nothing(env):
    from equilibria.family.driver import calibrate_base_cached

    monkeypatch, tmp_path = env
    monkeypatch.delenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", raising=False)
    calls = _counting(monkeypatch)
    for _ in range(2):
        run, mp = _run(tmp_path)
        assert run.settle_key is None
        calibrate_base_cached(mp, run)
    assert len(calls) == 2 and _seed_files(tmp_path) == []


def test_second_run_loads_the_same_settled_point(env):
    from equilibria.family.driver import calibrate_base_cached

    monkeypatch, tmp_path = env
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    calls = _counting(monkeypatch)
    run, mp = _run(tmp_path)
    settled, settle = calibrate_base_cached(mp, run)
    assert len(calls) == 1 and settle["check"]["code"] == 1
    assert len(_seed_files(tmp_path)) == 1
    run2, mp2 = _run(tmp_path)
    settled2, settle2 = calibrate_base_cached(mp2, run2)
    assert len(calls) == 1  # no second settle
    assert settled2 == settled  # exact: JSON keeps every float bit
    assert all(s["cached"] and s["code"] == 1 for s in settle2.values())


@pytest.mark.parametrize(
    "change",
    ["newton_tol", "linsolve_env", "nominal_savf", "family_code"],
)
def test_any_input_change_misses(env, change):
    from equilibria.family import driver
    from equilibria.family.driver import calibrate_base_cached

    monkeypatch, tmp_path = env
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    run, mp = _run(tmp_path)
    calibrate_base_cached(mp, run)
    calls = _counting(monkeypatch)
    kw = {}
    if change == "newton_tol":
        kw["newton_tol"] = (1e-11, 1e-11)
    elif change == "linsolve_env":
        monkeypatch.setenv("EQUILIBRIA_GTAP_TR_LINSOLVE", "scipy")
    elif change == "family_code":
        monkeypatch.setattr(driver, "_family_code_digest", lambda: "other-code")
    run2, mp2 = _run(tmp_path, **kw)
    if change == "nominal_savf":
        run2.nominal_savf = tuple(run2.regions[:1])
    assert driver._settle_key(run2) != driver._settle_key(run)
    if change != "linsolve_env":  # "scipy" is keyed only; it need not solve here
        calibrate_base_cached(mp2, run2)
        assert len(calls) == 1


def test_a_failed_settle_is_not_saved(env):
    from equilibria.family import driver
    from equilibria.family.driver import calibrate_base_cached

    monkeypatch, tmp_path = env
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    monkeypatch.setattr(
        driver,
        "calibrate_base_arrays",
        lambda mp, run: (
            {"x": {("a",): 1.0}},
            {"base": {"code": 1}, "check": {"code": 0}},
        ),
    )
    run, mp = _run(tmp_path)
    calibrate_base_cached(mp, run)
    assert _seed_files(tmp_path) == []


def test_run_from_the_cached_settle_matches_a_fresh_run(env):
    from equilibria.family.driver import solve_gtap_family

    monkeypatch, tmp_path = env
    p = _params(DS)
    rr = list(p.sets.r)[-1]

    def solve():
        return solve_gtap_family(
            _params(DS),
            _route_closure(),
            rr,
            cache_root=tmp_path / "records",
            jax_cache=False,
        )

    fresh = solve()
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    solve()  # writes
    calls = _counting(monkeypatch)
    hit = solve()
    assert calls == []
    assert {k: r["code"] for k, r in hit.results.items()} == {
        k: r["code"] for k, r in fresh.results.items()
    }
    for name, v in fresh.mp.vars.items():
        w = hit.mp.vars[name]
        np.testing.assert_allclose(w.value, v.value, rtol=1e-9, atol=1e-9, err_msg=name)


def test_the_seed_cache_switch_turns_it_off(env):
    from equilibria.family.driver import calibrate_base_cached

    monkeypatch, tmp_path = env
    monkeypatch.setenv("EQUILIBRIA_FAMILY_SETTLE_CACHE", "1")
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    calls = _counting(monkeypatch)
    for _ in range(2):
        run, mp = _run(tmp_path)
        calibrate_base_cached(mp, run)
    assert len(calls) == 2 and _seed_files(tmp_path) == []
