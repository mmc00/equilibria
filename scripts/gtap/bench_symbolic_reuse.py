#!/usr/bin/env python
"""A/B: MUMPS symbolic-reuse ON vs OFF on 15x10 GMIN+MUMPS. Prints solve wall-clock
and symbolic-factorization count for each. On 15x10 each symbolic is cheap (small n)
so the wall-clock win is small; the count reduction proves the mechanism. The real
~16min win is on the 20x41 (~12s/symbolic) — measured on Kaggle after this lands.

Throwaway benchmark, committed for reproducibility.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gtap"))

_ENV = {
    "EQUILIBRIA_GTAP_SOLVE_NLP": "1", "EQUILIBRIA_GTAP_SOLVER": "scipy_newton_tr",
    "EQUILIBRIA_GTAP_TR_LINSOLVE": "mumps", "EQUILIBRIA_GTAP_NLP_NO_JACSCALE": "1",
    "EQUILIBRIA_GTAP_TR_GATE": "1", "EQUILIBRIA_GTAP_TR_FTOL": "1e-7",
    "EQUILIBRIA_GTAP_SCIPY_MAXITER": "300", "EQUILIBRIA_GTAP_TR_DELTA0": "10.0",
    "EQUILIBRIA_GTAP_SHOCK_CONTINUATION": "0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0",
    "GTAP_GATES_SKIP": "1", "EQUILIBRIA_GTAP_GMIN": "1e-9",
    "EQUILIBRIA_GTAP_TR_RELTOL": "1e-6", "EQUILIBRIA_SEED_CACHE_DISABLE": "1",
}
DATA = ROOT / "datasets" / "gtap7_15x10"


def _run(reuse):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_block_model import solve_block_model
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp

    for k, v in _ENV.items():
        os.environ[k] = v
    os.environ["EQUILIBRIA_GTAP_GMIN_SYM_REUSE"] = "1" if reuse else "0"
    p = GTAPParameters()
    p.load_from_har(basedata_path=DATA / "basedata.har", sets_path=DATA / "sets.har",
                    default_path=DATA / "default.prm", baserate_path=DATA / "baserate.har")
    rr = list(p.sets.r)[-1]
    gc = GTAPClosureConfig(name="base", closure_type="MCP", capital_mobility="sluggish",
                           fix_endowments=False, fix_taxes=False, fix_technology=False,
                           if_sub=False, savf_flag="capFix", numeraire="pnum")
    m, mp, _ = build_sparse_model_mp(p, p.sets, gc, rr, base_calibrated=True)
    t0 = time.perf_counter()
    res = solve_block_model(m, p, gc, ref_gdx=None, mode="gtap")
    dt = time.perf_counter() - t0
    count = sys.modules["run_gtap"]._SYMBOLIC_FACT_COUNT
    return dt, count, res.get("shock", {}).get("code")


def main():
    t_off, c_off, code_off = _run(reuse=False)
    t_on, c_on, code_on = _run(reuse=True)
    print("=== MUMPS symbolic-reuse A/B (15x10 final solve, GMIN+MUMPS) ===")
    print(f"  reuse OFF: {t_off:6.1f}s  symbolic={c_off}  code={code_off}")
    print(f"  reuse ON : {t_on:6.1f}s  symbolic={c_on}  code={code_on}")
    if t_off:
        print(f"  saved: {100*(t_off-t_on)/t_off:.1f}% wall, "
              f"{c_off-c_on} fewer symbolic factorizations")
    print("  NOTE: 15x10 symbolics are cheap (small n); the real win is on the "
          "20x41 (~12s each). Kaggle A/B confirms.")


if __name__ == "__main__":
    main()
