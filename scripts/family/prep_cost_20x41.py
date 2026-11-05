"""Cost of the period preparation on arrays, from a two-photo capture.

    <python> scripts/family/prep_cost_20x41.py [--dataset gtap7_20x41]
        [--capture DIR] [--out FILE]

For each solve call of the capture (settle check, check, shock): rebuild the state
from photo A, run the recipe and the squaring on arrays, and check the result
against photo B. ``from_photo`` replaces today's Pyomo model and is reported apart;
``sp_reference`` (the single-period reference, once per run) and the per-call
``run_recipe`` + ``square`` are the prep cost.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path


def _closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    # The driver's base_closure for the test route (ifSUB=1).
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


def _params(dataset: str):
    from equilibria.sources import dataset_dir
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir(dataset)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    p = copy.deepcopy(p)
    p.set_va_subsidy_basis("gempack")  # as build_block_model leaves the run's params
    return p


def measure(dataset: str, capture: Path) -> dict:
    from equilibria.family.mp_arrays import MPArrays, compare, squared_from_photo
    from equilibria.family.prep_capture import pairs
    from equilibria.family.prep_recipe import run_recipe, sp_reference_arrays
    from equilibria.family.prep_square import square

    p = _params(dataset)
    closure = _closure()
    t = time.perf_counter()
    ref = sp_reference_arrays(p, closure, list(p.sets.r)[-1])
    sp_reference_s = time.perf_counter() - t

    calls = []
    for a, b in pairs(capture):
        inc = {r: list(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
        hard = {(e, v) for e, v, h in b.forced if h}
        t0 = time.perf_counter()
        mp = MPArrays.from_photo(a)
        t1 = time.perf_counter()
        mp = run_recipe(
            mp,
            a.period,
            params=p,
            closure=closure,
            sp_ref=ref,
            mode="gtap",
            mute_welfare=True,
            seed_from_prior=False,
        )
        t2 = time.perf_counter()
        sq = square(mp, a.period, params=p, closure=closure, incidence=inc)
        t3 = time.perf_counter()
        diffs = compare(sq, squared_from_photo(b), inc, hard)
        calls.append(
            {
                "period": a.period,
                "rows": len(b.con_names),
                "from_photo_s": t1 - t0,
                "run_recipe_s": t2 - t1,
                "square_s": t3 - t2,
                "parity": not diffs,
                "differences": diffs[:10],
            }
        )
    prep = sp_reference_s + sum(c["run_recipe_s"] + c["square_s"] for c in calls)
    return {
        "dataset": dataset,
        "sp_reference_s": sp_reference_s,
        "calls": calls,
        "prep_total_s": prep,
        "parity": all(c["parity"] for c in calls),
    }


if __name__ == "__main__":
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_20x41")
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    cap = a.capture or Path.home() / ".cache/equilibria/family/prep_capture" / a.dataset
    text = json.dumps(measure(a.dataset, cap), indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
