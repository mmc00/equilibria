"""A multi-instrument user shock gives the same result on the Pyomo and the family routes.

The shock (``imptx`` on two cells, ``aft``, ``pnum``, ``lambdava``) is applied with
``shocks.apply_shock`` to the Pyomo model before ``solve_multiperiod``, and passed as
``shock=`` to ``solve_gtap_family``. Every Var cell of the three periods agrees within
1e-9 (value, fixed flag and bounds as in ``diff_to_photo``), codes 1 on both.
"""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyomo")

from tests.family._prep import _newton_mumps  # noqa: E402


def _multishock(m) -> dict:
    """The shock of the test, from the cells of ``m`` that can take it."""
    from equilibria.templates.gtap.shocks import _can_take_shock

    def cells(name, positive=False):
        var = getattr(m, name, None)
        if var is None:
            return []
        return [
            tuple(k[:-1])
            for k in var
            if k[-1] == "shock"
            and (not positive or (var[k].value or 0) > 0)
            and _can_take_shock(m, name, tuple(k[:-1]))
        ]

    imptx, aft, lamb = cells("imptx"), cells("aft", True), cells("lambdava")
    shock: dict = {"imptx": {imptx[1]: 8.6637, imptx[-1]: 4.0}, "aft": {aft[0]: 10.0}}
    if _can_take_shock(m, "pnum", ()):
        shock["pnum"] = {(): 5.0}
    if lamb:
        shock["lambdava"] = {lamb[0]: 3.0}
    return shock


def _multishock_ifsub0(m) -> dict:
    """Under ifSUB=0: ``fcttx`` and ``kappaf`` cells (rejected under ifSUB=1) plus an
    ``imptx`` and an ``aft`` cell."""
    from equilibria.templates.gtap.shocks import _can_take_shock

    def cells(name):
        var = getattr(m, name)
        return [
            tuple(k[:-1])
            for k in var
            if k[-1] == "shock"
            and abs(var[k].value or 0) > 0
            and _can_take_shock(m, name, tuple(k[:-1]))
        ]

    fcttx, kappaf, imptx, aft = (cells(n) for n in ("fcttx", "kappaf", "imptx", "aft"))
    assert fcttx and kappaf, "ifSUB=0 must accept fcttx and kappaf cells"
    return {
        "fcttx": {fcttx[0]: 4.0, fcttx[-1]: 2.0},
        "kappaf": {kappaf[0]: 3.0},
        "imptx": {imptx[1]: 8.6637},
        "aft": {aft[0]: 10.0},
    }


def _check(dataset: str, linsolve: str, monkeypatch, tmp_path, ifsub: int = 1) -> None:
    from equilibria.family import prep_capture
    from equilibria.family.driver import solve_gtap_family
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.shocks import apply_shock
    from tests.family._data import load_params
    from tests.family.test_mp_build import diff_to_photo
    from tests.family.test_squared_system import _route_closure
    from tests.templates.gtap.test_gtap7_gempack_parity import find_reference

    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        monkeypatch.setenv(k, v)

    def params():
        return copy.deepcopy(load_params(dataset))

    p = params()
    closure = _route_closure()
    if ifsub == 0:
        closure = closure.model_copy(update={"if_sub": False})
    rr = list(p.sets.r)[-1]
    ref = find_reference(
        dataset, "gams", form="mcp", mode="gtap", ifsub=ifsub, fbep="header"
    ) or find_reference(
        dataset, "gams", form="nlp", mode="gtap", ifsub=ifsub, fbep="header"
    )
    gdx = None if ref is None else ref / "out.gdx"

    m, _ = build_block_model(p, p.sets, closure, rr, base_calibrated=True, ref_gdx=gdx)
    shock = _multishock(m) if ifsub == 1 else _multishock_ifsub0(m)
    assert len(shock) >= 3  # several instruments, or the test says nothing
    apply_shock(m, shock)

    photo_dir = tmp_path / "photo"
    monkeypatch.setenv(prep_capture.ENV, str(photo_dir))
    prep_capture.reset()
    res = solve_multiperiod(
        m,
        p,
        closure,
        ref_gdx=gdx,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        mode="gtap",
        solve_check=True,
    )
    monkeypatch.delenv(prep_capture.ENV)
    assert {k: v.get("code") for k, v in res.items()} == {
        "base": 1,
        "check": 1,
        "shock": 1,
    }
    final = prep_capture.load(Path(photo_dir) / "final.pkl")

    fres = solve_gtap_family(
        params(),
        closure,
        rr,
        shock=shock,
        cache_root=tmp_path / "cache",
        jax_cache=False,
    )
    assert {k: v["code"] for k, v in fres.results.items()} == {
        "base": 1,
        "check": 1,
        "shock": 1,
    }
    assert diff_to_photo(fres.mp, final, 1e-9, lb_rtol=1e-9) == []


def test_multishock_matches_pyomo_3x3(monkeypatch, tmp_path):
    _check("gtap7_3x3", "direct", monkeypatch, tmp_path)


def test_multishock_matches_pyomo_3x3_ifsub0(monkeypatch, tmp_path):
    """ifSUB=0: ``fcttx`` and ``kappaf`` cells, rejected under ifSUB=1, in the shock."""
    _check("gtap7_3x3", "direct", monkeypatch, tmp_path, ifsub=0)


@pytest.mark.slow
def test_multishock_matches_pyomo_10x7(monkeypatch, tmp_path):
    if importlib.util.find_spec("mumps") is None:
        pytest.skip("pymumps not available")
    _check("gtap7_10x7", "mumps", monkeypatch, tmp_path)


def _equal_bitwise(a, b) -> None:
    """Every Var cell of two ``MPArrays`` equal bit for bit (NaN equals NaN)."""
    assert set(a.vars) == set(b.vars)
    for name, va in a.vars.items():
        vb = b.vars[name]
        assert list(va.keys) == list(vb.keys), name
        assert np.array_equal(va.value, vb.value, equal_nan=True), name
        assert np.array_equal(va.fixed, vb.fixed), name


@pytest.fixture
def aot_state():
    from equilibria.family import aot_cache

    aot_cache.set_dir(None)
    aot_cache.reset()
    yield aot_cache
    aot_cache.set_dir(None)
    aot_cache.reset()


def test_multishock_aot_cache_bitwise_3x3(monkeypatch, tmp_path, aot_state):
    """Two different shocks with the AOT cache on give the arrays of the same runs with
    it off, bit for bit, and the second shock is served by the cache."""
    from equilibria.family.driver import solve_gtap_family
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from tests.family._data import load_params
    from tests.family.test_squared_system import _route_closure

    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        monkeypatch.setenv(k, v)
    cache = aot_state

    def params():
        return copy.deepcopy(load_params("gtap7_3x3"))

    p = params()
    rr = list(p.sets.r)[-1]
    m, _ = build_block_model(p, p.sets, _route_closure(), rr, base_calibrated=True)
    shock_a = _multishock(m)
    assert len(shock_a) >= 3
    shock_b = {
        inst: {c: pct * 0.5 for c, pct in cells.items()}
        for inst, cells in shock_a.items()
    }

    def run(shock):
        res = solve_gtap_family(
            params(),
            _route_closure(),
            rr,
            shock=shock,
            cache_root=tmp_path / "cache",
            jax_cache=False,
        )
        assert {k: v["code"] for k, v in res.results.items()} == {
            "base": 1,
            "check": 1,
            "shock": 1,
        }
        return res.mp

    off = {"a": run(shock_a), "b": run(shock_b)}

    cache.set_dir(tmp_path / "aot")
    cache.reset()
    on_a = run(shock_a)
    assert cache.stats["miss"] > 0  # the first run compiled and stored
    miss_a = cache.stats["miss"]
    on_b = run(shock_b)
    assert cache.stats["miss"] == miss_a  # the second shock compiled nothing new
    assert list((tmp_path / "aot").glob("*.aot"))
    _equal_bitwise(on_a, off["a"])
    _equal_bitwise(on_b, off["b"])
    # the two shocks are different runs, or the comparison says nothing
    with pytest.raises(AssertionError):
        _equal_bitwise(on_a, on_b)


@pytest.mark.slow
def test_aot_cache_bitwise_20x41(monkeypatch, tmp_path, aot_state):
    """The default shock on gtap7_20x41, family route with the AOT cache on and off:
    every Var cell equal bit for bit per period.

    Two full 20x41 solves, so it is skipped unless ``EQUILIBRIA_RUN_20X41=1``. Needs a
    Python with pymumps (see scripts/gtap/perf_budget_20x41.py) and the 20x41 data; the Newton environment is the one of
    ``scripts/gtap/perf_budget_20x41.py`` (set here). Run it with::

        EQUILIBRIA_RUN_20X41=1 PYTHONPATH=<worktree>/src:<worktree> python \\
            -m pytest tests/family/test_family_multishock.py -m slow -k 20x41
    """
    if os.environ.get("EQUILIBRIA_RUN_20X41") != "1":
        pytest.skip("set EQUILIBRIA_RUN_20X41=1 to run the 20x41 solves")
    if importlib.util.find_spec("mumps") is None:
        pytest.skip("pymumps not available")
    from equilibria.sources import has_dataset

    if not has_dataset("gtap7_20x41"):
        pytest.skip("gtap7_20x41 not available")
    from equilibria.family.driver import solve_gtap_family
    from tests.family._data import load_params
    from tests.family.test_squared_system import _route_closure

    for k, v in _newton_mumps().items():
        monkeypatch.setenv(k, v)
    cache = aot_state

    def run():
        p = copy.deepcopy(load_params("gtap7_20x41"))
        res = solve_gtap_family(
            p,
            _route_closure(),
            list(p.sets.r)[-1],
            cache_root=tmp_path / "cache",
            jax_cache=False,
        )
        assert {k: v["code"] for k, v in res.results.items()} == {
            "base": 1,
            "check": 1,
            "shock": 1,
        }
        return res.mp

    off = run()
    cache.set_dir(tmp_path / "aot")
    cache.reset()
    on = run()
    assert cache.stats["miss"] > 0
    _equal_bitwise(on, off)
