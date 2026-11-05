"""Cost of today's Newton step on the family squared system, per solve call.

    <python with pymumps> scripts/family/newton_cost_20x41.py
        [--dataset gtap7_20x41] [--capture DIR] [--cache-root DIR] [--out FILE]

For each solve call of a two-photo capture (settle check, check, shock): rebuild the
state from photo A, run the recipe and the squaring on arrays (rows squared from the
J pattern, cross-period rows from photo B), build ``SquaredSystem`` on the family
backend and time what one Newton iteration costs: one F+J, then MUMPS symbolic
(job 1), numeric (job 2) and solve (job 3) with today's ICNTL. The projection
multiplies by the iteration/factorization counts of today's run.

Rows with no family function (``proxy_rows``) are a timing placeholder (F = x - x0,
unit J entry); the count is reported per call.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# Newton iterations and numeric factorizations of the three solve calls (settle
# check, check, shock) on 20x41, plans/paper/mediciones/fases_20x41_d44692d.log.
NIT = (3, 3, 22)
NFACT = (2, 2, 21)
# MUMPS settings of today's driver (path_capi.py, Newton-TR): no MUMPS permutation
# (the pairing is already on the diagonal), AMD ordering, iterative scaling, null
# pivot detection with the driver's CNTL(3); ICNTL(14) working-space increase.
ICNTL = {14: 100, 6: 0, 7: 0, 8: 8, 24: 1}
CNTL3 = -1e-8


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mumps_step(J, F):
    """Symbolic, numeric, solve of ``J dx = -F``; times, MUMPS codes and residual.

    A failing job is recorded (``mumps_ok`` False) with the codes, not worked around.
    """
    import mumps
    import numpy as np

    rhs0 = -np.asarray(F, dtype=float)
    out: dict = {
        "symbolic_s": float("nan"),
        "factor_s": float("nan"),
        "solve_s": float("nan"),
        "mumps_ok": False,
        "infog1": None,
        "infog28": None,
        "residual": None,
    }
    ctx = mumps.DMumpsContext(sym=0, par=1)
    try:
        ctx.set_silent()
        for k, v in ICNTL.items():
            ctx.set_icntl(k, v)
        ctx.set_cntl(3, CNTL3)
        ctx.set_centralized_sparse(J.tocoo())
        for job, key in ((1, "symbolic_s"), (2, "factor_s"), (3, "solve_s")):
            if job == 3:
                rhs = rhs0.copy()
                ctx.set_rhs(rhs)
            t = time.perf_counter()
            try:
                ctx.run(job=job)
            except Exception as exc:  # pymumps raises on INFOG(1) < 0
                out[key] = time.perf_counter() - t
                out["infog1"] = int(ctx.get_infog(1))
                out["error"] = f"job {job}: {exc!r}"
                return out
            out[key] = time.perf_counter() - t
            out["infog1"] = int(ctx.get_infog(1))
            if job == 2:
                out["infog28"] = int(ctx.get_infog(28))
        out["residual"] = float(np.abs(J @ rhs - rhs0).max())
        out["mumps_ok"] = True
    finally:
        ctx.destroy()
    return out


def _same_pattern(J, prev) -> bool:
    if prev is None or J.shape != prev.shape or J.nnz != prev.nnz:
        return False
    import numpy as np

    return bool(
        np.array_equal(J.indptr, prev.indptr) and np.array_equal(J.indices, prev.indices)
    )


def projected(calls: list[dict]) -> float | None:
    """Σ_k [symbolic_k + NIT_k (FJ_k + solve_k) + NFACT_k factor_k]; symbolic_k = 0
    when the call has the pattern of the previous one. ``None`` if MUMPS failed."""
    if len(calls) != len(NIT) or not all(c["mumps_ok"] for c in calls):
        return None
    return float(
        sum(
            (0.0 if c["same_pattern_as_prev"] else c["symbolic_s"])
            + NIT[k] * (c["FJ_s"] + c["solve_s"])
            + NFACT[k] * c["factor_s"]
            for k, c in enumerate(calls)
        )
    )


def measure(dataset: str, capture: Path, cache_root: Path | None = None) -> dict:
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.disk_cache import cache_key, enable_jax_cache
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_capture import pairs
    from equilibria.family.prep_recipe import run_recipe, sp_reference_arrays
    from equilibria.family.prep_square import square
    from equilibria.family.squared_system import SquaredSystem, period_incidence
    from equilibria.templates.gtap import gtap_block_model as gbm

    prep, budget = _sibling("prep_cost_20x41"), _sibling("budget_20x41")
    cache_root = cache_root or Path.home() / ".cache" / "equilibria" / "family"
    enable_jax_cache(cache_root)

    p = prep._params(dataset)
    closure = prep._closure()
    rr = list(p.sets.r)[-1]
    ref = sp_reference_arrays(p, closure, rr)

    # The single-period state and the backend (recorded once, then loaded from disk).
    bclosure = budget._closure()
    model, state = gbm.compose_block_state(p, p.sets, bclosure, rr)
    key = cache_key(p, bclosure, rr)
    if key is None:
        raise RuntimeError("no cache key (uncoverable input)")
    path = cache_root / key / "records.pkl"
    if not path.exists():
        b0 = FamilyBackend()
        b0.build(model, skip=state.ifsub_skipped)
        b0.save(path)
    t = time.perf_counter()
    backend = FamilyBackend.load(path)
    backend_load_s = time.perf_counter() - t

    inc_cache: dict[str, tuple[dict, float]] = {}
    calls, prev_J = [], None
    for a, b in pairs(capture):
        mp = MPArrays.from_photo(a)
        mp = run_recipe(
            mp,
            a.period,
            params=p,
            closure=closure,
            sp_ref=ref,
            mode="gtap",
            mute_welfare=True,
            seed_from_prior=False,
            base_calibrated=True,
            solve_check=True,
        )
        if a.period not in inc_cache:
            t = time.perf_counter()
            pat = period_incidence(backend, state, a.period)
            inc_cache[a.period] = (pat, time.perf_counter() - t)
        pat, incidence_s = inc_cache[a.period]
        # B's incidence only for the rows the pattern does not give (cross-period).
        merged = {r: list(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
        merged.update(pat)
        sq = square(
            mp,
            a.period,
            params=p,
            closure=closure,
            incidence=merged,
            exogenous=ref.exogenous,
        )

        t = time.perf_counter()
        ss = SquaredSystem.build(backend, state, mp, a.period, sq)
        build_s = time.perf_counter() - t
        t = time.perf_counter()
        ss.F(ss.x0)
        ss.J(ss.x0)
        fj_first_s = time.perf_counter() - t  # includes loading the compiled programs
        t = time.perf_counter()
        F = ss.F(ss.x0)
        J = ss.J(ss.x0)
        fj_s = time.perf_counter() - t
        J.sort_indices()

        m = _mumps_step(J, F)
        calls.append(
            {
                "period": a.period,
                "n": len(sq.row_names),
                "nnz": int(J.nnz),
                "proxy_rows": len(ss.proxy_rows),
                "incidence_s": incidence_s,
                "build_s": build_s,
                "FJ_first_s": fj_first_s,
                "FJ_s": fj_s,
                "same_pattern_as_prev": _same_pattern(J, prev_J),
                **m,
            }
        )
        prev_J = J
    return {
        "dataset": dataset,
        "NIT": list(NIT),
        "NFACT": list(NFACT),
        "ICNTL": {str(k): v for k, v in ICNTL.items()},
        "CNTL3": CNTL3,
        "backend_load_s": backend_load_s,
        "calls": calls,
        "newton_projected_s": projected(calls),
    }


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()


if __name__ == "__main__":
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_20x41")
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--cache-root", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    cap = a.capture or Path.home() / ".cache/equilibria/family/prep_capture" / a.dataset
    result = {"sha": _sha(), **measure(a.dataset, cap, a.cache_root)}
    text = json.dumps(result, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
