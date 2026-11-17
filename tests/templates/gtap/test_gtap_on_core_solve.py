"""The core route solves GTAP to the same point as ``solve_multiperiod``.

Real PATH solves on gtap7_3x3 (gtap and altertax, ifSUB 0 and 1, the shock in
steps and Johansen) and on nus333 TBL65B (an ``@overwrite`` closure): each case
runs through the old driver and through ``Exercise(GTAP, gtap_static)`` in the
same process, and every Var cell must be equal (``==``; the same operations in
the same order give the same floats), with the same convergence per period.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
from tests.templates.gtap import _period_snapshot as snap

from equilibria.sources import dataset_dir

PATH_LIB = dataset_dir("gtap7_3x3").parents[2] / ".cache" / "path_capi"


def _closure(if_sub: bool) -> Any:
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=if_sub,
        numeraire="pnum",
    )


def _levels(m: Any) -> dict[tuple[str, Any], float]:
    from pyomo.environ import Var

    out = {}
    for v in m.component_objects(Var):
        for k in v:
            x = v[k].value
            out[v.name, k] = math.nan if x is None else float(x)
    return out


def _same(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def _compare(old_m: Any, old_res: dict, new_m: Any, new_res: dict) -> None:
    for period, r in new_res.items():
        old = old_res.get(period)
        if old is None or old.get("skipped_solve"):
            assert r["code"] == "not solved", (period, r)
            continue
        assert (old["code"] == 1) == (r["code"] == "optimal"), (period, old, r)
    a, b = _levels(old_m), _levels(new_m)
    assert a.keys() == b.keys()
    bad = [(k, a[k], b[k]) for k in a if not _same(a[k], b[k])]
    assert not bad, f"{len(bad)} cells differ, first: {bad[:20]}"


def _run_both(
    p: Any,
    closure: Any,
    rr: str,
    *,
    mode: str = "gtap",
    shock: dict | None = None,
    driver_kw: dict | None = None,
    recipe_kw: dict | None = None,
) -> None:
    from equilibria.core.exercise import Exercise
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.shocks import apply_shock
    from equilibria.templates.gtap.template import GTAP

    m, _ = build_block_model(p, p.sets, closure, rr)
    if shock:
        apply_shock(m, shock)
    old_res = solve_multiperiod(
        m, p, closure, mode=mode, skip_base_solve=True, **(driver_kw or {})
    )
    ex = Exercise(
        template=GTAP(p, closure, residual_region=rr),
        recipe=gtap_static(mode, **(recipe_kw or {})),
        shock=shock,
    )
    new_res = ex.run()
    assert ex.model is not None
    _compare(m, old_res, cast("Any", ex.model).pyomo, new_res)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    monkeypatch.delenv("EQUILIBRIA_GTAP_MODEL_CACHE", raising=False)
    monkeypatch.delenv("EQUILIBRIA_GTAP_SHOCK_CONTINUATION", raising=False)
    monkeypatch.delenv("PATH_CAPI_OPTIONS", raising=False)


needs_3x3 = pytest.mark.skipif(
    not (snap.DATASET.exists() and PATH_LIB.exists()),
    reason="gtap7_3x3 dataset or the PATH library not present",
)


@needs_3x3
@pytest.mark.parametrize("mode", ["gtap", "altertax"])
@pytest.mark.parametrize("if_sub", [False, True])
def test_static_3x3(mode: str, if_sub: bool) -> None:
    p = snap._load_params()
    _run_both(p, _closure(if_sub), list(p.sets.r)[-1], mode=mode)


@needs_3x3
def test_shock_in_steps_3x3() -> None:
    p = snap._load_params()
    _run_both(
        p,
        _closure(False),
        list(p.sets.r)[-1],
        driver_kw={"shock_steps": (0.5, 1.0)},
        recipe_kw={"shock_steps": (0.5, 1.0)},
    )


@needs_3x3
def test_johansen_3x3() -> None:
    p = snap._load_params()
    _run_both(
        p,
        _closure(False),
        list(p.sets.r)[-1],
        driver_kw={"shock_method": "johansen"},
        recipe_kw={"shock_method": "johansen"},
    )


@pytest.mark.integration
def test_overwrite_unemployment_nus333() -> None:
    from tests.templates.gtap._nus333 import closure, nus333_params
    from tests.templates.gtap._unemployment import register_unemployment_hooks

    from equilibria.blocks.gtap import overwrite

    if not dataset_dir("nus333").exists() or not PATH_LIB.exists():
        pytest.skip("nus333 dataset or the PATH library not present")
    p = nus333_params()
    register_unemployment_hooks()
    try:
        _run_both(
            p,
            closure(),
            "ROW",
            shock={"prdtx_rai": {("USA", "MFG", "MFG"): -10.9414}},
            driver_kw={"solve_check": True},
            recipe_kw={"solve_check": True},
        )
    finally:
        overwrite.clear()
