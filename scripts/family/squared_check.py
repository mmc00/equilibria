"""The squared system of every solve call with the cross-period rows from their families.

    <python> scripts/family/squared_check.py [--dataset gtap7_20x41]
        [--reference DIR] [--cache-root DIR] [--out FILE] [--no-system]

For each call of a Pyomo reference (photos ``<n>_{A,B}_<period>.pkl``): recipe on
photo A (``mq_*`` refreshed), incidence from the J patterns of the single-period AND
the cross-period records, ``square``, and ``compare`` with photo B. Then
``SquaredSystem.build`` with the cross recording (no proxy rows allowed) and one
F / J evaluation. The pairing is the family's own (Hopcroft-Karp), so rows paired
differently from Pyomo are reported (``pairs_not_pyomo``), not a failure.

Peak resident memory (``ru_maxrss``) and wall time come with the result; run it once
on 20x41 with the Pedro python and the Newton/MUMPS env.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _peak_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def measure(
    dataset: str, reference: Path, cache_root: Path | None = None, system: bool = True
) -> dict:
    from equilibria.family import cross_period
    from equilibria.family.disk_cache import cache_key
    from equilibria.family.mp_arrays import MPArrays, compare, squared_from_photo
    from equilibria.family.prep_capture import pairs
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
    t = time.perf_counter()
    bclosure = budget._closure()
    if cache_key(p, bclosure, rr) is None:
        raise RuntimeError("no cache key (uncoverable input)")
    cross = cross_period.load_or_record(
        p,
        bclosure,
        rr,
        cache_root,
        sets=backend.sets,
        var_domains=backend.var_domains,
    )
    cross_load_s = time.perf_counter() - t

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
        t = time.perf_counter()
        inc = period_incidence(backend, state, a.period, cross)
        incidence_s = time.perf_counter() - t
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
        hard = {(e, v) for e, v, h in b.forced if h}
        diffs = compare(
            sq, squared_from_photo(b), {r: set(c) for r, c in inc.items()}, hard
        )
        pb = dict(zip(b.con_names, b.var_names, strict=True))
        row = {
            "n": a.n,
            "period": a.period,
            "rows": len(sq.row_names),
            "incidence_s": incidence_s,
            "compare_differences": diffs[:10],
            "pairs_not_pyomo": sum(
                pb[r] != c for r, c in zip(sq.row_names, sq.col_names, strict=True)
            ),
        }
        if system:
            t = time.perf_counter()
            ss = SquaredSystem.build(backend, state, mp_sq, a.period, sq, cross=cross)
            row["build_s"] = time.perf_counter() - t
            t = time.perf_counter()
            F, J = ss.F(ss.x0), ss.J(ss.x0)
            row["FJ_first_s"] = time.perf_counter() - t
            row["proxy_rows"] = len(ss.proxy_rows)
            row["nnz"] = int(J.nnz)
            row["F_inf_at_x0"] = float(abs(F).max())
        row["peak_mb"] = _peak_mb()
        calls.append(row)
        print(json.dumps(row), flush=True)
    return {
        "dataset": dataset,
        "cross_load_s": cross_load_s,
        "calls": calls,
        "compare_ok": all(not c["compare_differences"] for c in calls),
        "wall_s": time.perf_counter() - t_start,
        "peak_mb": _peak_mb(),
    }


if __name__ == "__main__":
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_20x41")
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--cache-root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-system", action="store_true")
    a = ap.parse_args()
    ref = (
        a.reference
        or Path.home() / ".cache/equilibria/family/reference" / a.dataset / "tol_today"
    )
    text = json.dumps(measure(a.dataset, ref, a.cache_root, not a.no_system), indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
