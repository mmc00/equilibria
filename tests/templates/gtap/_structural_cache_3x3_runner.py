"""Subprocess runner for test_structural_cache_parity.py.

Runs a single gtap7_3x3 base->check->shock NLP solve and prints a JSON line with the
per-phase result + every free variable's solved value (by name). A separate process per
run is required because EQUILIBRIA_GTAP_STRUCT_CACHE is read at call time and
run_gtap._STRUCT_MATCH_CACHE is a module-level singleton — isolating each run avoids any
cross-run leakage and keeps the OFF/ON comparison honest.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

os.environ.setdefault("EQUILIBRIA_GTAP_SOLVE_NLP", "1")
os.environ.setdefault("EQUILIBRIA_GTAP_SOLVER", "scipy_newton_tr")
os.environ.setdefault("GTAP_GATES_SKIP", "1")
os.environ.setdefault("EQUILIBRIA_SEED_CACHE_DISABLE", "1")

from pyomo.environ import Var

from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_block_model import solve_block_model
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp


def main() -> None:
    d = ROOT / "datasets" / "gtap7_3x3"
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]
    ac = GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, savf_flag="capFix", numeraire="pnum",
    )

    t0 = time.perf_counter()
    m, mp, _ = build_sparse_model_mp(p, p.sets, ac, rr, base_calibrated=False, ref_gdx=None)
    res = solve_block_model(m, p, ac, ref_gdx=None, mode="gtap")
    wall = time.perf_counter() - t0

    cells = {}
    for v in m.component_data_objects(Var, active=True, descend_into=True):
        val = v.value
        if val is not None:
            cells[v.name] = float(val)

    out = {
        "cache_on": os.environ.get("EQUILIBRIA_GTAP_STRUCT_CACHE") == "1",
        "wall_s": round(wall, 2),
        "result": {k: str(v) for k, v in (res or {}).items()},
        "n_cells": len(cells),
        "cells": cells,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
