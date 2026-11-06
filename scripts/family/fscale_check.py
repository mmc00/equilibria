"""Family ``fscale`` against Pyomo's, call by call, on a reference capture.

    <python> scripts/family/fscale_check.py [--dataset gtap7_20x41]
        [--reference DIR] [--cache-root DIR] [--out FILE]

For each solve call: recipe on photo A, family incidence, ``square``,
``SquaredSystem.build`` (cross rows from their families), then
``SquaredSystem.fscale()`` against ``pyomo_fscale_from_photo`` (photo B's incidence,
fixed flags and levels). Reports, per call, how many rows differ, the largest
relative difference and the families involved. Bitwise equality is the target; a
difference is reported, not a failure (spec answer Q4).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_fscale(ss, sq, b, a2) -> dict:
    """Rows where the family fscale differs from Pyomo's recomputed from the photos."""
    from equilibria.family.fscale import pyomo_fscale_from_photo

    got = ss.fscale()
    want_by_row = pyomo_fscale_from_photo(b, a2)
    want = np.array([want_by_row[r] for r in sq.row_names])
    bad = np.flatnonzero(got.view(np.int64) != want.view(np.int64))
    rel = np.abs(got[bad] - want[bad]) / np.maximum(1.0, np.abs(want[bad]))
    fams = Counter(sq.row_names[i].partition("[")[0] for i in bad)
    worst = np.argsort(-rel)[:5]
    return {
        "rows": len(want),
        "differ": int(len(bad)),
        "max_rel": float(rel.max()) if len(bad) else 0.0,
        "families": dict(fams.most_common(10)),
        "worst": [
            (sq.row_names[bad[i]], float(got[bad[i]]), float(want[bad[i]]))
            for i in worst
        ],
    }


def measure(dataset: str, reference: Path, cache_root: Path | None = None) -> dict:
    from equilibria.family import cross_period
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_capture import load_a2, pairs
    from equilibria.family.prep_recipe import run_recipe, sp_reference_arrays
    from equilibria.family.prep_square import ConditionalMasks, square
    from equilibria.family.squared_system import SquaredSystem, period_incidence

    prep, budget = _sibling("prep_cost_20x41"), _sibling("budget_20x41")
    cache_root = cache_root or Path.home() / ".cache" / "equilibria" / "family"
    t_start = time.perf_counter()
    p = prep._params(dataset)
    closure = prep._closure()
    rr = list(p.sets.r)[-1]
    ref = sp_reference_arrays(p, closure, rr)
    backend, state = prep._load_backend(p, cache_root)
    cross = cross_period.load_or_record(
        p,
        budget._closure(),
        rr,
        cache_root,
        sets=backend.sets,
        var_domains=backend.var_domains,
    )
    calls, masks = [], None
    for a, b in pairs(reference):
        mp = run_recipe(
            MPArrays.from_photo(a),
            a.period,
            params=p,
            closure=closure,
            sp_ref=ref,
            mode="gtap",
            backend=backend,
            mute_welfare=True,
            seed_from_prior=False,
            base_calibrated=True,
            solve_check=True,
            cross=cross,
        )
        masks = masks or ConditionalMasks.from_params(p, mp)
        inc = period_incidence(backend, state, a.period, cross)
        sq, mp_sq = square(
            mp,
            a.period,
            params=p,
            closure=closure,
            incidence=inc,
            exogenous=ref.exogenous,
            masks=masks,
            return_mp=True,
        )
        ss = SquaredSystem.build(backend, state, mp_sq, a.period, sq, cross=cross)
        row = {"n": a.n, "period": a.period}
        row.update(compare_fscale(ss, sq, b, load_a2(reference, a)))
        calls.append(row)
        print(json.dumps(row), flush=True)
    return {
        "dataset": dataset,
        "calls": calls,
        "wall_s": time.perf_counter() - t_start,
    }


if __name__ == "__main__":
    gc.disable()
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--dataset", default="gtap7_20x41")
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--cache-root", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    ref = (
        a.reference
        or Path.home() / ".cache/equilibria/family/reference" / a.dataset / "tol_today"
    )
    text = json.dumps(measure(a.dataset, ref, a.cache_root), indent=1)
    if a.out:
        a.out.write_text(text)
