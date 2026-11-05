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
import contextlib
import copy
import functools
import gc
import json
import time
from pathlib import Path

# square()'s sub-steps, timed by wrapping prep_square's module functions.
SQUARE_STEPS = (
    "initialize_benchmark",
    "apply_closure",
    "conditional_fixing",
    "check_gap",
    "fix_sluggish_pft",
    "deactivate_fixed_quantity_eqs",
    "check_overdetermined",
    "relative_floors",
    "names",
    "edges",
    "gams_pairs",
    "match",
    "cells",
)


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


@contextlib.contextmanager
def _timed_square_steps(steps: dict[str, float]):
    """Accumulate each sub-step's time into ``steps``; nested calls count once."""
    import equilibria.family.prep_square as ps

    depth = [0]
    saved = {}

    def wrap(name, fn):
        @functools.wraps(fn)
        def timed(*args, **kwargs):
            depth[0] += 1
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                depth[0] -= 1
                if depth[0] == 0:
                    steps[name] = steps.get(name, 0.0) + time.perf_counter() - t

        return timed

    for name in SQUARE_STEPS:
        attr = name if hasattr(ps, name) else f"_{name}"
        saved[attr] = getattr(ps, attr)
        setattr(ps, attr, wrap(name, saved[attr]))
    try:
        yield
    finally:
        for attr, fn in saved.items():
            setattr(ps, attr, fn)


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
        steps = {name: 0.0 for name in SQUARE_STEPS}
        t2 = time.perf_counter()
        with _timed_square_steps(steps):
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
                "square_steps": steps,
                "parity": not diffs,
                "differences": diffs[:10],
            }
        )
    per_call = sum(c["run_recipe_s"] + c["square_s"] for c in calls)
    return {
        "dataset": dataset,
        # compose_block_state: the budget already counts it as compose_state.
        "sp_reference_s": sp_reference_s,
        "calls": calls,
        # The prep cost on top of the budget: recipe + square, all calls.
        "prep_per_call_total_s": per_call,
        "prep_total_s": sp_reference_s + per_call,
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
