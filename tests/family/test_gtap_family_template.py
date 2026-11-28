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


def test_family_build_matches_the_old_settle(env) -> None:
    """The family build (``prepare_run`` + the core's settle) leaves the arrays and
    the settled point the old family driver's settle left, recorded on main before
    PR 4 deleted it (``tests/_old_route_golden.py``)."""
    from tests._old_route_golden import case, mp_mismatches

    _, tmp_path = env
    model = _gtap(tmp_path, base_calibrated=True).build(("base", "check", "shock"))
    want = case("template_build")
    assert mp_mismatches(model.mp, want["mp"]) == []
    assert _settled_mismatches(model.settled_seed, want["settled"]) == []
    assert model.base_calibrated
    assert model.backend == "family"


def _settled_mismatches(got: dict, want: dict) -> list[str]:
    from tests._golden import mismatches
    from tests._old_route_golden import ABS, REL

    as_repr = {
        repr(v): {repr(k): x for k, x in cells.items()} for v, cells in got.items()
    }
    return mismatches(as_repr, want, rel=REL, abs_tol=ABS)


_UNSUPPORTED = [
    ({"mode": "altertax"}, "altertax"),
    ({"solve_base": True}, "solve_base"),
    ({"seed_from_prior": True}, "seed_from_prior"),
    ({"mute_welfare": False}, "mute_welfare"),
    ({"base_calibrated": True, "solve_check": False}, "solve_check"),
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


def _register_wage_hooks() -> None:
    """TBL65B's closure on gtap7_3x3: unskilled labour unemployment in the USA, with
    the real wage fixed (``tests/templates/gtap/_unemployment.py`` on nus333)."""
    from pyomo.environ import value

    from equilibria.blocks.gtap import (
        ClosureBlock,
        ShockBlock,
        overwrite,
        ppriv_tornqvist,
    )

    @overwrite(ShockBlock, period="shock")
    def unemployment(b):
        b.endogenous("aft", ("USA", "UnSkLab"))

    @overwrite(ClosureBlock, period="shock")
    def real_wage(b):
        b.equation(
            "eq_wreal",
            ("USA", "UnSkLab"),
            lambda m, r, f: m.pft[r, f] == value(m.pft[r, f]) * ppriv_tornqvist(m, r),
        )


def test_family_runs_an_overwrite_hook(env) -> None:
    """The family template builds under @overwrite hooks: the freed cell is not
    exogenous at the shock, and still is at the check."""
    from equilibria.blocks.gtap.overwrite import overwrite

    _, tmp_path = env
    _register_wage_hooks()
    try:
        model = _gtap(tmp_path).build(("base", "check", "shock"))
    finally:
        overwrite.clear()
    assert ("USA", "UnSkLab") not in model.exogenous_cells("aft", "shock")
    assert ("USA", "UnSkLab") in model.exogenous_cells("aft", "check")


_CACHE_RUN = """
import sys
from pathlib import Path
from tests.family.test_gtap_family_template import _register_wage_hooks
from tests.family.test_sp_arrays import _params
from tests.family.test_squared_system import _route_closure
from equilibria.core.exercise import Exercise
from equilibria.templates.gtap.recipe import gtap_static
from equilibria.templates.gtap.template import GTAP

if sys.argv[2] == "hooks":
    _register_wage_hooks()
gtap = GTAP(_params("gtap7_3x3"), _route_closure(), backend="family",
            cache_root=Path(sys.argv[1]), jax_cache=True)
out = Exercise(gtap, gtap_static()).run()
assert out["shock"]["code"] == "optimal", out
"""


def _cache_files(tmp_path, hooks: str) -> list:
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cache = tmp_path / hooks
    script = tmp_path / f"run_{hooks}.py"
    script.write_text(_CACHE_RUN, encoding="utf-8")
    env = {
        **os.environ,
        **_newton_mumps(),
        "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct",
        "PYTHONPATH": os.pathsep.join([str(root), str(root / "src")]),
    }
    r = subprocess.run(
        [sys.executable, str(script), str(cache), hooks],
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
        cwd=root,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return [f for f in cache.rglob("*") if f.is_file()] if cache.exists() else []


def test_no_family_cache_under_hooks(env, tmp_path) -> None:
    """Under @overwrite hooks a family run writes nothing to its cache root (records,
    pairings, compiled JAX programs); without them the same run does."""
    load_params(DS)
    assert _cache_files(tmp_path, "plain")
    assert _cache_files(tmp_path, "hooks") == []


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
