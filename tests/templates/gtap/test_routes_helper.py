"""The route helper (``_routes.solve``) runs one GTAP case through the old driver
or through the core (``Exercise``) and returns the same thing for both.

Every parity gate is parametrized over ``ROUTES`` with it, so these tests guard
what would make that comparison lie: a failed core solve read as a success, a
route parameter that never reaches the solve, inputs changed by the first run, a
hook or an environment variable leaking into the next one.
"""

from __future__ import annotations

import math
import os
import pickle
from typing import Any

import pytest
from tests.templates.gtap import _period_snapshot as snap
from tests.templates.gtap._routes import ROUTES, core_codes, solve

from equilibria.sources import dataset_dir

PATH_LIB = dataset_dir("gtap7_3x3").parents[2] / ".cache" / "path_capi"

needs_3x3 = pytest.mark.skipif(
    not (snap.DATASET.exists() and PATH_LIB.exists()),
    reason="gtap7_3x3 dataset or the PATH library not present",
)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    monkeypatch.delenv("EQUILIBRIA_GTAP_MODEL_CACHE", raising=False)
    monkeypatch.delenv("EQUILIBRIA_GTAP_SHOCK_CONTINUATION", raising=False)
    monkeypatch.delenv("EQUILIBRIA_GTAP_BACKEND", raising=False)
    monkeypatch.delenv("PATH_CAPI_OPTIONS", raising=False)


def _closure() -> Any:
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )


def _case(route: str, params: Any = None, **kw: Any):
    p = params if params is not None else snap._load_params()
    return solve(route, p, _closure(), list(p.sets.r)[-1], skip_base_solve=True, **kw)


def _levels(m: Any) -> dict[tuple[str, Any], float]:
    from pyomo.environ import Var

    out = {}
    for v in m.component_objects(Var):
        for k in v:
            x = v[k].value
            out[v.name, k] = math.nan if x is None else float(x)
    return out


def _diff(a: dict, b: dict) -> list:
    assert a.keys() == b.keys()
    return [
        (k, a[k], b[k])
        for k in a
        if not (a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])))
    ]


def test_routes() -> None:
    assert ROUTES == ("old", "core")


def test_core_codes_maps_every_status() -> None:
    res = {
        "base": {"code": "not solved"},
        "check": {"code": "optimal"},
        "shock": {"code": "code 2"},
    }
    assert core_codes(res, skip_base_solve=True) == {"base": 1, "check": 1, "shock": 2}
    assert core_codes(res, skip_base_solve=False) == {"check": 1, "shock": 2}
    with pytest.raises(ValueError, match="weird"):
        core_codes({"check": {"code": "weird"}}, skip_base_solve=True)


def test_unknown_route_raises() -> None:
    with pytest.raises(ValueError, match="route"):
        solve("new", None, None, "ROW")


@needs_3x3
def test_both_routes_give_the_same_cells_3x3() -> None:
    old, core = _case("old"), _case("core")
    assert (old.route, core.route) == ("old", "core")
    assert old.codes == core.codes == {"base": 1, "check": 1, "shock": 1}
    bad = _diff(_levels(old.model), _levels(core.model))
    assert not bad, f"{len(bad)} cells differ, first: {bad[:10]}"


@needs_3x3
def test_core_route_never_calls_the_old_driver(monkeypatch) -> None:
    from equilibria.family import driver as fam
    from equilibria.templates.gtap import gtap_multiperiod_driver as drv

    def boom(*a, **k):
        raise AssertionError("the old driver was called")

    monkeypatch.setattr(drv, "solve_multiperiod", boom)
    monkeypatch.setattr(fam, "solve_gtap_family", boom)
    assert _case("core").codes == {"base": 1, "check": 1, "shock": 1}


@needs_3x3
def test_old_route_never_flies_the_recipe(monkeypatch) -> None:
    from equilibria.core import driver, exercise

    def boom(*a, **k):
        raise AssertionError("the core driver was called")

    monkeypatch.setattr(driver, "fly", boom)
    monkeypatch.setattr(exercise, "fly", boom)
    assert _case("old").codes == {"base": 1, "check": 1, "shock": 1}


@needs_3x3
@pytest.mark.parametrize("route", ROUTES)
def test_params_are_not_mutated(route: str) -> None:
    p = snap._load_params()
    before = pickle.dumps(p)
    _case(route, p)
    assert pickle.dumps(p) == before


@needs_3x3
@pytest.mark.parametrize("route", ROUTES)
def test_backend_env_is_respected_and_restored(route: str, monkeypatch) -> None:
    """The helper decides the backend: a leftover ``EQUILIBRIA_GTAP_BACKEND`` does
    not move a pyomo case to family, and the helper leaves the variable as it
    found it."""
    monkeypatch.setenv("EQUILIBRIA_GTAP_BACKEND", "family")
    s = _case(route)
    assert s.backend == "pyomo"
    assert hasattr(s.model, "component_objects")  # a Pyomo model, not MPArrays
    assert os.environ["EQUILIBRIA_GTAP_BACKEND"] == "family"


@needs_3x3
@pytest.mark.parametrize("route", ROUTES)
def test_overwrite_does_not_leak(route: str) -> None:
    """A hook registered for one case and cleared after it does not reach the
    next case (#214): the unhooked run equals a run that never saw a hook."""
    from equilibria.blocks.gtap import ShockBlock, overwrite

    clean = _levels(_case(route).model)
    overwrite.clear()

    @overwrite(ShockBlock, period="shock")
    def unemployment(b):  # TBL65B's hook on 3x3 labels
        b.endogenous("aft", ("USA", "UnSkLab"))

    try:
        hooked = _levels(_case(route).model)
    finally:
        overwrite.clear()
    assert _diff(clean, hooked)  # the hook did change the run
    assert not _diff(clean, _levels(_case(route).model))


@pytest.mark.parametrize(
    "name,cell,factor,pct",
    [
        ("lambdava", ("USA", "SER"), 1.10, 10.0),
        ("aft", ("USA", "CAPITAL"), 1.10, 10.0),
    ],
)
@pytest.mark.integration
def test_factor_shock_is_a_percent_shock(name, cell, factor, pct) -> None:
    """``fix_instrument_shock(factor=f)`` and the ``Exercise`` shock
    ``{name: {cell: 100·(f-1)}}`` fix the same level, to the bit."""
    level = _fixed_level(name, cell, factor=factor)
    assert level == _fixed_level(name, cell, pct=pct)


@pytest.mark.integration
def test_power_value_shock_is_a_percent_shock() -> None:
    """TBL65B's ``prdtx_rai`` value ``(1+t0)·(1-0.109414)-1`` is the GEMPACK
    power shock -10.9414 %."""
    from pyomo.environ import value

    idx = ("USA", "MFG", "MFG")
    m = _nus333_model()
    t0 = float(value(m.prdtx_rai[(*idx, "check")]))
    target = (1 + t0) * (1 - 0.109414) - 1
    assert _fixed_level("prdtx_rai", idx, value=target) == _fixed_level(
        "prdtx_rai", idx, pct=-10.9414
    )


def _nus333_model() -> Any:
    from tests.templates.gtap._nus333 import closure, nus333_params

    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = nus333_params()
    m, _ = build_block_model(
        p, p.sets, closure(), "ROW", base_calibrated=False, ref_gdx=None
    )
    return m


def _fixed_level(name: str, cell: tuple, *, pct=None, **kw) -> float:
    from equilibria.templates.gtap.instruments import fix_instrument_shock
    from equilibria.templates.gtap.shocks import apply_shock

    m = _nus333_model()
    if pct is None:
        fix_instrument_shock(m, name, cell, **kw)
    else:
        apply_shock(m, {name: {cell: pct}})
    return float(getattr(m, name)[(*cell, "shock")].value)
