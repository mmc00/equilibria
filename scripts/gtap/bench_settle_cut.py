#!/usr/bin/env python
"""A/B wall-clock of calibrate_base's settle: settle_only (cut at check) vs the
full base->check->shock settle. Measures the lever-A win.

Usage: bench_settle_cut.py [dataset]   (default gtap7_15x10)
Env (free solver): EQUILIBRIA_GTAP_SOLVER=scipy_newton_tr TR_LINSOLVE=mumps SOLVE_NLP=1
Always runs with EQUILIBRIA_SEED_CACHE_DISABLE=1 (measure the live settle, not a cache hit).

Throwaway benchmark, committed for reproducibility of the perf claim.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"

from equilibria.blocks.gtap.factor import FactorBlock  # noqa: E402
from equilibria.templates.gtap import GTAPParameters  # noqa: E402
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig  # noqa: E402
import equilibria.templates.gtap.gtap_block_model as _bm  # noqa: E402


def _load(dataset):
    d = ROOT / "datasets" / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def _closure(p):
    return GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, savf_flag="capFix", numeraire="pnum",
    )


def _time_settle(p, full):
    rr = list(p.sets.r)[-1]
    fb = FactorBlock(sets=p.sets, params=p)
    orig = _bm.solve_block_model
    if full:
        def _full(m, params, closure, ref_gdx, *, mode="gtap", settle_only=False):
            return orig(m, params, closure, ref_gdx, mode=mode, settle_only=False)
        _bm.solve_block_model = _full
    try:
        t0 = time.perf_counter()
        fb.calibrate_base(p, p.sets, _closure(p), rr, ref_gdx=None)
        return time.perf_counter() - t0
    finally:
        _bm.solve_block_model = orig


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "gtap7_15x10"
    p = _load(dataset)
    full = _time_settle(p, full=True)
    cut = _time_settle(p, full=False)
    saved = (1.0 - cut / full) * 100.0 if full else 0.0
    print(f"dataset={dataset}  (calibrate_base settle)")
    print(f"  full base->check->shock : {full:7.1f} s")
    print(f"  settle_only (cut@check) : {cut:7.1f} s")
    print(f"  saved                   : {saved:6.1f}%  ({full / max(cut, 1e-9):.2f}x)")


if __name__ == "__main__":
    main()
