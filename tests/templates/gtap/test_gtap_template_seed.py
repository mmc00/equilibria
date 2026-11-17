"""``GTAP(..., seed_gdx=gdx)``: the built model starts at a GAMS solution.

The GAMS parity gates seed every period from the reference GDX between the build
and the solve (``seed_all_periods``). On the core route the build happens inside
``Exercise.run``, so the template does it: same cells as building with
``build_block_model`` and seeding by hand.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from tests.templates.gtap import _period_snapshot as snap
from tests.templates.gtap.test_gtap7_gempack_parity import _reference_gdx

pytestmark = pytest.mark.needs_gdxdump

DS = "gtap7_3x3"
PERIODS = ("base", "check", "shock")


def _closure() -> Any:
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        numeraire="pnum",
    )


def _gdx() -> Any:
    gdx = _reference_gdx(DS, 1)
    if gdx is None or not gdx.exists():
        pytest.skip("gtap7_3x3 GAMS reference GDX not present")
    return gdx


def _levels(m: Any) -> dict:
    from pyomo.environ import Var

    out = {}
    for v in m.component_objects(Var):
        for k in v:
            x = v[k].value
            out[v.name, k] = math.nan if x is None else float(x)
    return out


def _same(a: dict, b: dict) -> list:
    assert a.keys() == b.keys()
    return [
        (k, a[k], b[k])
        for k in a
        if not (a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])))
    ]


def test_seed_gdx_seeds_every_period_as_by_hand() -> None:
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.template import GTAP

    gdx, p = _gdx(), snap._load_params()
    rr = list(p.sets.r)[-1]
    m, mp = build_block_model(p, p.sets, _closure(), rr)
    unseeded = _levels(m)
    mp.seed_all_periods(m, gdx)
    by_hand = _levels(m)
    assert _same(unseeded, by_hand)  # the seed moves cells: the check can fail

    built = GTAP(p, _closure(), residual_region=rr, seed_gdx=gdx).build(PERIODS)
    assert not _same(by_hand, _levels(built.pyomo))


def test_without_seed_gdx_the_build_is_unseeded() -> None:
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.template import GTAP

    p = snap._load_params()
    rr = list(p.sets.r)[-1]
    m, _ = build_block_model(p, p.sets, _closure(), rr)
    built = GTAP(p, _closure(), residual_region=rr).build(PERIODS)
    assert not _same(_levels(m), _levels(built.pyomo))


def test_family_refuses_seed_gdx() -> None:
    from equilibria.templates.gtap.template import GTAP

    p = snap._load_params()
    with pytest.raises(NotImplementedError, match="seed_gdx"):
        GTAP(p, _closure(), backend="family", seed_gdx=_gdx()).build(PERIODS)
