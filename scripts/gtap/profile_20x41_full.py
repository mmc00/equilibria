#!/usr/bin/env python
"""Instrument the FULL 20x41 end-to-end (build + calibrate + solve) to find where
the ~1h47m wall actually goes — refuting/confirming the R6 phase split with a
FINE breakdown of the "41 min solve" phase.

Three tools, because none is trustworthy alone:
  1. Manual phase timers (time.perf_counter) — the ground truth report_timing misses.
  2. pyomo report_timing — per-component Pyomo construction time.
  3. cProfile over the WHOLE run — per-function tottime/cumtime; dumped to .prof.

Usage:
  profile_20x41_full.py [dataset]           default gtap7_20x41
Env (set BEFORE run for the free solver):
  EQUILIBRIA_GTAP_SOLVER=scipy_newton_tr TR_LINSOLVE=mumps SOLVE_NLP=1
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gtap"))

from pyomo.common.timing import report_timing  # noqa: E402

_PHASES: list[tuple[str, float]] = []


class _Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"[phase] START {self.name}", flush=True)
        return self

    def __exit__(self, *a):
        dt = time.perf_counter() - self.t0
        _PHASES.append((self.name, dt))
        print(f"[phase] END   {self.name}: {dt:.1f} s", flush=True)


def run(dataset):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_block_model import solve_block_model
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp

    d = ROOT / "datasets" / dataset
    gdx = ROOT / "tests/fixtures/gtap7" / dataset / "out_gtap_shock_ifsub0.gdx"

    with _Phase("load_params"):
        p = GTAPParameters()
        p.load_from_har(
            basedata_path=d / "basedata.har",
            sets_path=d / "sets.har",
            default_path=d / "default.prm",
            baserate_path=d / "baserate.har",
        )
    rr = list(p.sets.r)[-1]
    gc = GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, numeraire="pnum",
    )

    # report_timing captures Pyomo component construction during build+calibrate.
    rt_buf = io.StringIO()
    report_timing(rt_buf)
    with _Phase("build_sparse_model_mp(base_calibrated=True)  [build + calibrate settle]"):
        m, mp, _ = build_sparse_model_mp(
            p, p.sets, gc, rr, base_calibrated=True, ref_gdx=str(gdx) if gdx.exists() else None
        )
    report_timing(False)

    with _Phase("solve_block_model  [squareness patches + base->check->shock PATH]"):
        res = solve_block_model(m, p, gc, ref_gdx=str(gdx) if gdx.exists() else None, mode="gtap")

    print("\n===== report_timing (Pyomo component construction) =====", flush=True)
    print(rt_buf.getvalue(), flush=True)

    print("\n===== solve result codes =====", flush=True)
    for per, d_ in (res or {}).items():
        if isinstance(d_, dict):
            print(f"  {per}: code={d_.get('code')} resid={d_.get('residual')}", flush=True)
    return res


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "gtap7_20x41"
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    try:
        run(dataset)
    finally:
        prof.disable()
        total = time.perf_counter() - t0
        out = ROOT / f"profile_{dataset}.prof"
        prof.dump_stats(str(out))

        print(f"\n===== PHASE BREAKDOWN (total {total:.1f} s) =====", flush=True)
        for name, dt in _PHASES:
            print(f"  {dt:8.1f} s  ({100*dt/total:4.1f}%)  {name}", flush=True)

        print("\n===== cProfile TOP 25 by cumulative =====", flush=True)
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(25)
        print(s.getvalue(), flush=True)

        print("\n===== cProfile TOP 25 by tottime (own time = the volume) =====", flush=True)
        s2 = io.StringIO()
        pstats.Stats(prof, stream=s2).sort_stats("tottime").print_stats(25)
        print(s2.getvalue(), flush=True)
        print(f"[saved cProfile dump] {out}", flush=True)


if __name__ == "__main__":
    main()
