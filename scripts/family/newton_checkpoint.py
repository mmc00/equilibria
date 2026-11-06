"""Family Newton-TR against Pyomo's, call by call, on a reference capture (Task 6).

    <python> scripts/family/newton_checkpoint.py <dataset>
        [--capture DIR] [--out FILE] [--linsolve mumps|direct] [--cache-root DIR]

For each solve call of the reference (settle check, check, shock): photo A ->
``run_recipe`` -> ``square`` -> ``SquaredSystem.build`` (real cross rows, no proxies)
-> ``newton_tr`` with the family pairing (the system's column order, i.e. the identity
column permutation), the family ``fscale``, ``ICNTL(6)=7`` under ``mumps`` and the TR
settings read from the environment exactly as the Pyomo path reads them (export
``NEWTON_MUMPS``). The result is compared with the call's photo P (Pyomo's solution)
and stats S (its ``nit``, ``nfact``, trace).

The gate is the solved cells: every squared column within ``1e-9`` of P in
``|a - b| / max(1, |b|)`` and ``code == 1`` (success and ``||F||_inf < 1e-5``, as
``_ok_sr``). ``nit``/``nfact``/trace are recorded, never a gate. Timings run with the
garbage collector off: recipe / square / build, F+J, MUMPS analysis / factorization /
solve and the whole Newton, plus the peak RSS of the process. JSON to ``--out``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOL = 1e-9  # solved cells vs Pyomo's P, |a - b| / max(1, |b|)
ICNTL6_FAMILY = 7  # MUMPS ICNTL(6) on the family squared system (Pyomo keeps 0)


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_to_p(sq, x: np.ndarray, p) -> dict:
    """Max ``|a - b| / max(1, |b|)`` of every squared column against photo P."""
    from equilibria.family.mp_arrays import _name

    pv = {}
    for comp, vc in p.vars.items():
        for k, v in zip(vc.keys, vc.value, strict=True):
            pv[_name(comp, k)] = float(v)
    want = np.full(len(x), np.nan)
    missing = []
    for i, c in enumerate(sq.col_names):
        if c in pv:
            want[i] = pv[c]
        else:
            missing.append(c)
    have = np.array([c in pv for c in sq.col_names])
    both_nan = np.isnan(x) & np.isnan(want) & have
    rel = np.abs(x - want) / np.maximum(1.0, np.abs(want))
    rel[~have] = np.inf
    rel[both_nan] = 0.0
    rel = np.where(np.isnan(rel), np.inf, rel)
    worst = np.argsort(-rel)[:20]
    return {
        "n_cells": len(x),
        "n_missing_in_P": len(missing),
        "max_rel_diff": float(rel.max()) if len(rel) else 0.0,
        "n_over_tol": int((rel > TOL).sum()),
        "worst20": [
            (sq.col_names[i], float(x[i]), float(want[i]), float(rel[i])) for i in worst
        ],
    }


@contextlib.contextmanager
def _timed_mumps(acc: dict):
    """Time analysis / factorization / solve of the Pyomo MUMPS wrapper and keep INFOG(1)."""
    from pyomo.contrib.pynumero.linalg.mumps_interface import (
        MumpsCentralizedAssembledLinearSolver as M,
    )

    names = {
        "do_symbolic_factorization": "analysis_s",
        "do_numeric_factorization": "factor_s",
        "do_back_solve": "solve_s",
    }
    saved = {k: getattr(M, k) for k in names}

    def wrap(meth, key):
        def timed(self, *a, **k):
            t = time.perf_counter()
            try:
                return saved[meth](self, *a, **k)
            finally:
                acc[key] = acc.get(key, 0.0) + time.perf_counter() - t
                acc.setdefault("infog1", []).append(_infog1(self))

        return timed

    for meth, key in names.items():
        setattr(M, meth, wrap(meth, key))
    try:
        yield
    finally:
        for meth, fn in saved.items():
            setattr(M, meth, fn)


def _infog1(solver) -> int | None:
    try:
        return int(solver.get_infog(1))
    except Exception:
        return None


def solve_squared(ss, sq, linsolve: str) -> dict:
    """Newton-TR on the squared system; returns the result and its timings."""
    from equilibria.solver.newton_tr import newton_tr

    n = len(ss.x0)
    lb = np.array([sq.cells[c][2] for c in sq.col_names], dtype=float)
    ub = np.array([sq.cells[c][3] for c in sq.col_names], dtype=float)
    lb[~np.isfinite(lb)] = -1e20  # as the Pyomo path
    ub[~np.isfinite(ub)] = 1e20
    # as the Pyomo path: the relative test needs RELTOL > 0, then fscale applies
    reltol = float(os.environ.get("EQUILIBRIA_GTAP_TR_RELTOL", "0") or 0.0)
    t = time.perf_counter()
    fscale = ss.fscale() if reltol > 0.0 else None
    t_fscale = time.perf_counter() - t

    acc = {"fj_s": 0.0, "f_s": 0.0, "j_s": 0.0, "n_f": 0, "n_j": 0}

    def F(z):
        t0 = time.perf_counter()
        try:
            return ss.F(np.clip(z, lb, ub))
        finally:
            acc["f_s"] += time.perf_counter() - t0
            acc["n_f"] += 1

    def J(z):
        t0 = time.perf_counter()
        try:
            return ss.J(np.clip(z, lb, ub))
        finally:
            acc["j_s"] += time.perf_counter() - t0
            acc["n_j"] += 1

    mumps_acc: dict = {}
    ctx = _timed_mumps(mumps_acc) if linsolve == "mumps" else contextlib.nullcontext()
    t0 = time.perf_counter()
    with ctx:
        res = newton_tr(
            F,
            J,
            np.clip(ss.x0, lb, ub),
            lb,
            ub,
            colperm=np.arange(n),  # row k is paired with column k
            fscale=fscale,
            reltol=reltol,
            icntl6=ICNTL6_FAMILY if linsolve == "mumps" else None,
            row_names=sq.row_names,
            col_names=sq.col_names,
            linsolve=linsolve,
            maxit=int(os.environ.get("EQUILIBRIA_GTAP_SCIPY_MAXITER", "300")),
        )
    newton_s = time.perf_counter() - t0
    resid = float(np.linalg.norm(ss.F(np.clip(res.x, lb, ub)), np.inf))
    acc["fj_s"] = acc["f_s"] + acc["j_s"]
    return {
        "res": res,
        "residual": resid,
        "code": int(bool(res.success) and resid < 1e-5),
        "timing": {
            "fscale_s": t_fscale,
            "newton_s": newton_s,
            **acc,
            **{k: v for k, v in mumps_acc.items() if k != "infog1"},
        },
        "infog1": mumps_acc.get("infog1", []),
    }


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def measure(
    dataset: str,
    capture: Path,
    linsolve: str = "mumps",
    cache_root: Path | None = None,
) -> dict:
    from equilibria.family import cross_period
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_capture import load, pairs
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
    setup_s = time.perf_counter() - t_start

    calls, masks = [], None
    for a, _b in pairs(capture):
        photo_p = load(capture / f"{a.n:03d}_P_{a.period}.pkl")
        stats = json.loads((capture / f"{a.n:03d}_S_{a.period}.json").read_text())
        gc.disable()  # timed region
        try:
            t = time.perf_counter()
            mp = run_recipe(
                MPArrays.from_photo(a),
                a.period,
                params=p,
                closure=closure,
                sp_ref=ref,
                mode="gtap",
                mute_welfare=True,
                seed_from_prior=False,
                base_calibrated=True,
                solve_check=True,
                cross=cross,
            )
            t_recipe = time.perf_counter() - t
            masks = masks or ConditionalMasks.from_params(p, mp)
            t = time.perf_counter()
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
            t_square = time.perf_counter() - t
            t = time.perf_counter()
            ss = SquaredSystem.build(backend, state, mp_sq, a.period, sq, cross=cross)
            t_build = time.perf_counter() - t
            out = solve_squared(ss, sq, linsolve)
        finally:
            gc.enable()
        res = out["res"]
        cmp = compare_to_p(sq, res.x, photo_p)
        row = {
            "n": a.n,
            "period": a.period,
            "rows": len(sq.row_names),
            "linsolve": linsolve,
            "code": out["code"],
            "success": bool(res.success),
            "residual": out["residual"],
            "gate_ok": bool(out["code"] == 1 and cmp["max_rel_diff"] <= TOL),
            "ours": {
                "nit": res.nit,
                "nfact": res.nfact,
                "trace": res.trace,
                "null_pivots": res.null_pivots,
                "cfg": res.cfg,
                "infog1": out["infog1"],
            },
            "pyomo": {
                "code": stats["code"],
                "nit": stats["nit"],
                "nfact": stats["nfact"],
                "trace_f_inf": stats["trace_f_inf"],
                "null_pivots": stats["null_pivots"],
                "linsolve": stats["linsolve"],
                "gmin": stats["gmin"],
                "icntl": stats["icntl"],
            },
            **cmp,
            "timing": {
                "recipe_s": t_recipe,
                "square_s": t_square,
                "build_s": t_build,
                **out["timing"],
            },
            "peak_rss_mb": _peak_rss_mb(),
        }
        calls.append(row)
        print(
            f"call {a.n:03d} {a.period:<6} code={row['code']} nit ours={res.nit} "
            f"pyomo={stats['nit']} max_rel={cmp['max_rel_diff']:.2e} "
            f"over_tol={cmp['n_over_tol']} gate={'OK' if row['gate_ok'] else 'FAIL'} "
            f"newton={out['timing']['newton_s']:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return {
        "dataset": dataset,
        "linsolve": linsolve,
        "setup_s": setup_s,
        "tr_env": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith("EQUILIBRIA_")
        },
        "calls": calls,
        "gate_ok": all(c["gate_ok"] for c in calls),
        "peak_rss_mb": _peak_rss_mb(),
        "wall_s": time.perf_counter() - t_start,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("dataset")
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--linsolve", choices=("mumps", "direct"), default="mumps")
    ap.add_argument("--cache-root", type=Path)
    a = ap.parse_args()
    cap = (
        a.capture
        or Path.home() / ".cache/equilibria/family/reference" / a.dataset / "tol_today"
    )
    result = measure(a.dataset, cap, a.linsolve, a.cache_root)
    text = json.dumps(result, indent=1)
    if a.out:
        a.out.write_text(text)
    else:
        print(text)
    return 0 if result["gate_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
