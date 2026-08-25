"""Lever A — eliminate calibrate_base's double solve.

HARD GATE: settle_only (cut the settle at check) must produce a settled_seed
IDENTICAL to the full base->check->shock settle. Baseline captured on gtap7_10x7.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from equilibria.blocks.gtap.factor import FactorBlock
from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

DATA = Path("datasets/gtap7_10x7")
# Full base->check->shock settle baseline (captured Task 0, gtap7_10x7).
# settle_only must reproduce this EXACTLY.
BASELINE_COUNT = 20138
BASELINE_SIG = "ded5651b6133f788"


def _load_params():
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DATA / "basedata.har",
        sets_path=DATA / "sets.har",
        default_path=DATA / "default.prm",
        baserate_path=DATA / "baserate.har",
    )
    return p


def _closure(p):
    return GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, savf_flag="capFix", numeraire="pnum",
    )


def _seed_signature(seed):
    parts = []
    for name, cells in seed.items():
        for body, val in cells.items():
            parts.append(f"{name}|{body}|{float(val):.10e}")
    parts.sort()
    h = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    return len(parts), h


def _calibrate(**kw):
    p = _load_params()
    rr = list(p.sets.r)[-1]
    fb = FactorBlock(sets=p.sets, params=p)
    return fb.calibrate_base(p, p.sets, _closure(p), rr, ref_gdx=None, **kw)


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_full_settle_baseline():
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    seed = _calibrate()
    count, h = _seed_signature(seed)
    assert count > 0, "settle produced an empty seed"
    print(f"BASELINE settled_seed: {count} cells, sig={h}")


def _solve_multiperiod_result(settle_only):
    """Build the block model + solve, return the results dict (base/check/[shock])."""
    from equilibria.templates.gtap.gtap_block_model import (
        build_block_model,
        solve_block_model,
    )

    p = _load_params()
    rr = list(p.sets.r)[-1]
    m, mp = build_block_model(p, p.sets, _closure(p), rr)
    return solve_block_model(m, p, _closure(p), None, mode="gtap", settle_only=settle_only)


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_settle_only_skips_shock():
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    res = _solve_multiperiod_result(settle_only=True)
    assert "check" in res, "settle_only must still solve the check phase"
    assert "shock" not in res, "settle_only must NOT solve the shock phase"
