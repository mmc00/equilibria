#!/usr/bin/env python
"""A/B wall-clock of the two multi-period equations-build paths on a dataset.

  build_equations_intra x3 (per-period, del/recreate)  vs
  build_equations_all_periods (build once)

Usage: bench_build_once.py [dataset]   (default gtap7_15x10)

Throwaway benchmark, committed for reproducibility of the perf claim.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from equilibria.templates.gtap import GTAPParameters  # noqa: E402
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig  # noqa: E402
from equilibria.templates.gtap.gtap_model_multiperiod import (  # noqa: E402
    PERIODS,
    GTAPMultiPeriodModel,
)


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


def _mp(p):
    rr = list(p.sets.r)[-1]
    gc = GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, numeraire="pnum",
    )
    return GTAPMultiPeriodModel(p.sets, p, gc, residual_region=rr)


def _time_build(p, use_all):
    mp = _mp(p)
    m = mp.build_sets()
    mp.build_vars(m)
    t0 = time.perf_counter()
    if use_all:
        mp.build_equations_all_periods(m)
    else:
        for per in PERIODS:
            mp.build_equations_intra(m, per)
    return time.perf_counter() - t0


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "gtap7_15x10"
    p = _load(dataset)
    old = _time_build(p, use_all=False)
    new = _time_build(p, use_all=True)
    ratio = (1.0 - new / old) * 100.0 if old else 0.0
    print(f"dataset={dataset}")
    print(f"  build_equations_intra x3 : {old:7.2f} s")
    print(f"  build_equations_all      : {new:7.2f} s")
    print(f"  saved                    : {ratio:6.1f}%  ({old / max(new, 1e-9):.2f}x)")


if __name__ == "__main__":
    main()
