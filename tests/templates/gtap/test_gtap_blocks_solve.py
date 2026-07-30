"""North-star gate (F3 Task 5): gtap7_3x3 solves via the symbolic block framework
and matches GAMS on NLP AND MCP, multi-period.

The 7 GTAP block units (``equilibria.blocks.gtap``) are composed into a solvable
multi-period model by ``gtap_block_model`` (compose -> strip _con -> apply the
monolith's benchmark scaling -> reflect into base/check/shock + Fisher), seeded from
the GAMS reference GDX, and solved through the existing PATH/IPOPT driver. Each
period must converge (code==1) and match the GAMS levels cell-by-cell at >= the
matrix floor.

This is the framework's FIRST solve+parity test — no prior test solves a symbolic
block model and asserts feasibility + GAMS parity. The measure logic (SKIP/RF report
sets, ALIAS, prefix-strip, 1% band) is reused verbatim from ``measure_nlp_vs_nlp`` so
the block gate is identical to the monolith's parity gate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "scripts" / "gtap"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
_PATH_CAPI = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI.exists() and str(_PATH_CAPI) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI))

pytestmark = pytest.mark.integration

DATASET = "gtap7_3x3"
RESIDUAL_REGION = "ROW"
FLOOR = 0.99
GDX = ROOT / "tests" / "fixtures" / "gtap7" / DATASET / "out_gtap_shock_ifsub0.gdx"
PERIODS = ("base", "check", "shock")


def _load_params():
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    d = ROOT / "datasets" / DATASET
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def _closure():
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


def _measure(m, period):
    """Cell-by-cell match% of the block model vs the GAMS GDX for one period.

    Reuses the exact SKIP/RF/ALIAS report sets and 1%-band tolerance from
    measure_nlp_vs_nlp so this gate is identical to the monolith's parity gate.
    """
    from _diff_core import gams_levels, list_populated_vars, split_t
    from measure_nlp_vs_nlp import ALIAS, RF, SKIP
    from pyomo.environ import value

    def _strip(x):
        return (
            x[2:]
            if (isinstance(x, str) and len(x) > 2 and x[1] == "_" and x[0] in "acfr")
            else x
        )

    gdx = str(GDX)
    tot = 0
    match = 0
    for vn in list_populated_vars(gdx):
        if vn.lower() in SKIP or vn.lower() in RF:
            continue
        try:
            g = gams_levels(gdx, vn)
        except Exception:
            continue
        pv = getattr(m, ALIAS.get(vn, vn), None) or getattr(m, vn.lower(), None)
        if pv is None:
            continue
        for fk, gval in g.items():
            try:
                body, t = split_t(fk)
            except Exception:
                continue
            if t != period:
                continue
            st = tuple(_strip(x) for x in body)
            idx = (
                (period,)
                if not st
                else ((st[0], period) if len(st) == 1 else (*st, period))
            )
            val = None
            for cand in [idx, (*body, period) if body else (period,)]:
                try:
                    val = float(value(pv[cand]))
                    break
                except Exception:
                    pass
            if val is None:
                continue
            tot += 1
            d_abs = abs(val - gval)
            rel = (
                d_abs / abs(gval)
                if abs(gval) > 1e-12
                else (0.0 if d_abs < 1e-6 else 9e9)
            )
            if d_abs <= 1e-6 or rel <= 1e-2:
                match += 1
    return tot, (match / tot if tot else 0.0)


def _build_seed_solve(nlp: bool):
    from equilibria.templates.gtap.gtap_block_model import (
        build_block_model,
        solve_block_model,
    )

    if nlp:
        os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = "1"
    else:
        os.environ.pop("EQUILIBRIA_GTAP_SOLVE_NLP", None)

    p = _load_params()
    gc = _closure()
    m, mp = build_block_model(p, p.sets, gc, RESIDUAL_REGION)
    mp.seed_all_periods(m, GDX)
    res = solve_block_model(m, p, gc, ref_gdx=GDX, mode="gtap")
    return m, res


@pytest.mark.skipif(not GDX.exists(), reason="gtap7_3x3 reference GDX missing")
def test_3x3_nlp_via_blocks_matches_gams():
    """gtap7_3x3 solves via the composed 7-block framework (NLP/IPOPT) and matches
    GAMS >= 99% on every period."""
    try:
        m, res = _build_seed_solve(nlp=True)
    finally:
        os.environ.pop("EQUILIBRIA_GTAP_SOLVE_NLP", None)
    for per in PERIODS:
        assert res[per]["code"] == 1, f"NLP {per} did not converge: {res[per]}"
    for per in PERIODS:
        tot, pct = _measure(m, per)
        assert tot > 0, f"NLP {per}: no comparable cells"
        assert pct >= FLOOR, f"NLP {per} match {pct:.4f} < {FLOOR}"


@pytest.mark.skipif(not GDX.exists(), reason="gtap7_3x3 reference GDX missing")
def test_3x3_mcp_via_blocks_matches_gams():
    """gtap7_3x3 solves via the composed 7-block framework (MCP/PATH) and matches
    GAMS >= 99% on every period."""
    pytest.importorskip("path_capi_python")
    m, res = _build_seed_solve(nlp=False)
    for per in PERIODS:
        assert res[per]["code"] == 1, f"MCP {per} did not converge: {res[per]}"
    for per in PERIODS:
        tot, pct = _measure(m, per)
        assert tot > 0, f"MCP {per}: no comparable cells"
        assert pct >= FLOOR, f"MCP {per} match {pct:.4f} < {FLOOR}"
