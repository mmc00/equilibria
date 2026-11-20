"""The core route solves GTAP to the same point as the old multiperiod driver.

Real PATH solves on gtap7_3x3 (gtap and altertax, ifSUB 0 and 1, the shock in
steps and Johansen) and on nus333 TBL65B (an ``@overwrite`` closure): each case
runs through ``Exercise(GTAP, gtap_static)`` and every Var cell must equal what
the old driver gave (``tests/_old_route_golden.py``, recorded on main before PR 4
of gtap-on-core deleted it, case ``on_core_<name>``), with the same convergence
per period.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from tests import _old_route_golden as golden
from tests.templates.gtap import _period_snapshot as snap
from tests.templates.gtap._routes import solve

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


def _run_both(
    name: str, p: Any, closure: Any, rr: str, *, mode: str = "gtap", **kw: Any
) -> None:
    """The core run against the old driver's, recorded as case ``on_core_<name>``."""
    old = golden.case(f"on_core_{name}")
    core = solve("core", p, closure, rr, mode=mode, skip_base_solve=True, **kw)
    assert core.codes == old["codes"]
    bad = golden.level_mismatches(golden.pyomo_levels(core.model), old["levels"])
    assert not bad, f"{len(bad)} cells differ, first: {bad[:20]}"


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
    _run_both(
        f"static_{mode}_ifsub{int(if_sub)}",
        p,
        _closure(if_sub),
        list(p.sets.r)[-1],
        mode=mode,
    )


@needs_3x3
def test_shock_in_steps_3x3() -> None:
    p = snap._load_params()
    _run_both(
        "steps",
        p,
        _closure(False),
        list(p.sets.r)[-1],
        shock_steps=(0.5, 1.0),
    )


@needs_3x3
def test_johansen_3x3() -> None:
    p = snap._load_params()
    _run_both(
        "johansen",
        p,
        _closure(False),
        list(p.sets.r)[-1],
        shock_method="johansen",
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
            "overwrite_nus333",
            p,
            closure(),
            "ROW",
            shock={"prdtx_rai": {("USA", "MFG", "MFG"): -10.9414}},
            solve_check=True,
        )
    finally:
        overwrite.clear()
