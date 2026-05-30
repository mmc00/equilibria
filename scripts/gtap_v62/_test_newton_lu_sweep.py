"""Sweep Newton + scipy sparse LU across gtap6_3x3, 5x5, 10x7, 15x10."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pyomo.environ import Objective, minimize as pyo_min, value, Var
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore


DATASETS = [
    {"name": "gtap6_3x3",   "shock": ("Food",      "USA", "EU_28"), "gempack_viws": 62.359},
    {"name": "gtap6_5x5",   "shock": ("Food",      "USA", "EU_28"), "gempack_viws": 64.553},
    {"name": "gtap6_10x7",  "shock": ("FoodProc",  "USA", "EU_28"), "gempack_viws": 64.391},
    {"name": "gtap6_15x10", "shock": ("OtherFood", "USA", "EU_28"), "gempack_viws": 66.359},
]


def _project(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)


def _max_alpha_to_bounds(x, dx, lb, ub, tau=0.995):
    alpha = 1.0
    pos = dx > 0
    neg = dx < 0
    if np.any(pos):
        finite = np.isfinite(ub) & pos
        if np.any(finite):
            ratio = (ub[finite] - x[finite]) / dx[finite]
            ratio = ratio[ratio > 0]
            if ratio.size:
                alpha = min(alpha, tau * float(ratio.min()))
    if np.any(neg):
        finite = np.isfinite(lb) & neg
        if np.any(finite):
            ratio = (lb[finite] - x[finite]) / dx[finite]
            ratio = ratio[ratio > 0]
            if ratio.size:
                alpha = min(alpha, tau * float(ratio.min()))
    return max(alpha, 1.0e-16)


def newton_solve(nlp, *, tol=1.0e-6, max_iter=60, ls_min_alpha=1.0e-10,
                 ls_shrink=0.5, ls_armijo=1.0e-4, verbose=True):
    n = nlp.n_primals()
    m_eq = nlp.n_eq_constraints()
    m_ineq = nlp.n_ineq_constraints()
    if m_ineq != 0:
        raise RuntimeError(f"Newton expects only equality constraints, got {m_ineq} ineq.")
    if n != m_eq:
        raise RuntimeError(f"System not square: n={n}, m_eq={m_eq}")

    lb = nlp.primals_lb().copy()
    ub = nlp.primals_ub().copy()
    x = nlp.get_primals().copy()
    x = _project(x, lb, ub)

    nlp.set_primals(x)
    c = nlp.evaluate_eq_constraints()
    inf0 = float(np.max(np.abs(c))) if c.size else 0.0

    if verbose:
        print(f"  [Newton] n={n}  m_eq={m_eq}  nnz_J={nlp.nnz_jacobian_eq()}", flush=True)
        print(f"  [Newton] k=00  ||c||_inf={inf0:.3e}", flush=True)

    if inf0 < tol:
        return {"converged": True, "iters": 0, "inf": inf0, "x": x, "time": 0.0}

    t0 = time.perf_counter()
    prev_inf = inf0

    for k in range(1, max_iter + 1):
        t_iter = time.perf_counter()
        J = nlp.evaluate_jacobian_eq()
        J_csc = J.tocsc() if sp.issparse(J) else sp.csc_matrix(J)

        try:
            dx = spla.spsolve(J_csc, -c)
        except RuntimeError as e:
            if verbose:
                print(f"  [Newton] k={k:02d} LU SOLVE FAILED: {e}", flush=True)
            try:
                lu = spla.splu(J_csc)
                dx = lu.solve(-c)
            except Exception as e2:
                return {"converged": False, "iters": k, "inf": prev_inf, "x": x,
                        "time": time.perf_counter() - t0,
                        "error": f"LU failed: {e2}"}

        if not np.all(np.isfinite(dx)):
            return {"converged": False, "iters": k, "inf": prev_inf, "x": x,
                    "time": time.perf_counter() - t0,
                    "error": "dx has non-finite entries (singular J?)"}

        alpha_max = _max_alpha_to_bounds(x, dx, lb, ub, tau=0.995)
        alpha = min(1.0, alpha_max)

        c_norm2_prev = 0.5 * float(c @ c)
        grad_dot_dx = -2.0 * c_norm2_prev

        accepted = False
        x_new = x
        c_new = c
        inf_new = prev_inf
        while alpha > ls_min_alpha:
            x_trial = _project(x + alpha * dx, lb, ub)
            nlp.set_primals(x_trial)
            try:
                c_trial = nlp.evaluate_eq_constraints()
            except Exception:
                alpha *= ls_shrink
                continue
            if not np.all(np.isfinite(c_trial)):
                alpha *= ls_shrink
                continue
            inf_trial = float(np.max(np.abs(c_trial)))
            c_norm2_trial = 0.5 * float(c_trial @ c_trial)
            if (c_norm2_trial <= c_norm2_prev + ls_armijo * alpha * grad_dot_dx) \
               or (inf_trial < prev_inf):
                accepted = True
                x_new, c_new, inf_new = x_trial, c_trial, inf_trial
                break
            alpha *= ls_shrink

        if not accepted:
            if verbose:
                print(f"  [Newton] k={k:02d} LINE SEARCH FAILED  alpha_min={alpha:.2e}", flush=True)
            return {"converged": False, "iters": k, "inf": prev_inf, "x": x,
                    "time": time.perf_counter() - t0,
                    "error": "line search failed"}

        x, c = x_new, c_new
        iter_dt = time.perf_counter() - t_iter
        total_dt = time.perf_counter() - t0

        if verbose:
            print(f"  [Newton] k={k:02d}  ||c||_inf={inf_new:.3e}  "
                  f"alpha={alpha:.3e}  ||dx||_inf={float(np.max(np.abs(dx))):.3e}  "
                  f"dt={iter_dt:5.2f}s  T={total_dt:6.2f}s", flush=True)

        if inf_new < tol:
            nlp.set_primals(x)
            return {"converged": True, "iters": k, "inf": inf_new, "x": x,
                    "time": total_dt}
        prev_inf = inf_new

    nlp.set_primals(x)
    return {"converged": False, "iters": max_iter, "inf": prev_inf, "x": x,
            "time": time.perf_counter() - t0}


def _attach_trivial_objective(model):
    if hasattr(model, "obj"):
        try:
            model.del_component("obj")
        except Exception:
            pass
    model.obj = Objective(expr=0.0, sense=pyo_min)


def _walras(model):
    try:
        return float(value(model.walras))
    except Exception:
        return float("nan")


def run_dataset(spec):
    DATA = Path(f"datasets/{spec['name']}")
    SHOCK_KEY = spec['shock']
    print(f"\n{'='*70}\n{spec['name']} | shock={SHOCK_KEY} | GEMPACK ref +{spec['gempack_viws']:.3f}%\n{'='*70}", flush=True)

    if not DATA.exists():
        print(f"SKIP: {DATA} not found", flush=True)
        return None

    t_load = time.perf_counter()
    sets = GTAPv62Sets()
    sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
    params = GTAPv62Parameters()
    params.load_from_har(basedata_path=DATA / "basedata.har",
                         default_prm_path=DATA / "default.prm", sets=sets)
    model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()
    pipe = apply_v62_pipeline(model, mode="nlp", bake_tolerance=1.0e-9,
                              params=params, drop_dead_rows_threshold=0.0)
    closure = pipe['closure']
    n_vars = closure['free_vars']
    print(f"load+build+closure: {time.perf_counter()-t_load:.1f}s | free={n_vars} baked={pipe['prebalance']['n_baked']}", flush=True)

    _attach_trivial_objective(model)

    print("\n--- BASELINE ---", flush=True)
    nlp = PyomoNLP(model)
    res = newton_solve(nlp, tol=1.0e-6, max_iter=30)
    nlp.set_primals(res["x"])
    nlp.load_state_into_pyomo()
    w_base = _walras(model)
    viws_0 = float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))
    print(f"BASELINE: conv={res['converged']} iters={res['iters']} inf={res['inf']:.3e} "
          f"walras={w_base:.4e} VIWS_0={viws_0:.4e} time={res['time']:.2f}s", flush=True)

    if not res['converged']:
        print(f"  ERROR: {res.get('error', '(no error string)')}", flush=True)
        return {"name": spec['name'], "n_vars": n_vars, "ok": False, "stage": "baseline",
                "iters_b": res['iters'], "iters_s": 0, "walras_b": w_base, "walras_s": float('nan'),
                "viws_pct": float('nan'), "gempack": spec['gempack_viws'],
                "gap_pp": float('nan'), "rel_pct": float('nan'),
                "time_total": res['time']}

    print("\n--- SHOCKED ---", flush=True)
    old_tms = float(value(model.tms[SHOCK_KEY]))
    new_tms = (1.0 + old_tms) * 0.9 - 1.0
    model.tms[SHOCK_KEY] = new_tms
    print(f"  tms[{SHOCK_KEY}]: {old_tms:.6f} -> {new_tms:.6f}", flush=True)

    nlp_s = PyomoNLP(model)
    res_s = newton_solve(nlp_s, tol=1.0e-6, max_iter=60)
    nlp_s.set_primals(res_s["x"])
    nlp_s.load_state_into_pyomo()
    w_shock = _walras(model)
    viws_final = float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))
    viws_pct = 100.0 * (viws_final - viws_0) / viws_0
    gap_pp = viws_pct - spec['gempack_viws']
    rel_pct = abs(gap_pp) / spec['gempack_viws'] * 100.0
    print(f"SHOCKED:  conv={res_s['converged']} iters={res_s['iters']} inf={res_s['inf']:.3e} "
          f"walras={w_shock:.4e} time={res_s['time']:.2f}s", flush=True)
    if not res_s['converged']:
        print(f"  ERROR: {res_s.get('error', '(no error string)')}", flush=True)
    print(f"VIWS = {viws_pct:+.4f}%  vs GEMPACK +{spec['gempack_viws']:.3f}%  "
          f"gap={gap_pp:+.3f}pp ({rel_pct:.2f}%)", flush=True)

    return {
        "name": spec['name'], "n_vars": n_vars, "ok": res_s['converged'],
        "iters_b": res['iters'], "iters_s": res_s['iters'],
        "walras_b": w_base, "walras_s": w_shock,
        "viws_pct": viws_pct, "gempack": spec['gempack_viws'],
        "gap_pp": gap_pp, "rel_pct": rel_pct,
        "time_total": res['time'] + res_s['time'],
    }


def main():
    results = []
    for spec in DATASETS:
        try:
            r = run_dataset(spec)
            if r is not None:
                results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nEXCEPTION on {spec['name']}: {e}", flush=True)
            results.append({"name": spec['name'], "n_vars": -1, "ok": False,
                            "iters_b": -1, "iters_s": -1,
                            "walras_b": float('nan'), "walras_s": float('nan'),
                            "viws_pct": float('nan'), "gempack": spec['gempack_viws'],
                            "gap_pp": float('nan'), "rel_pct": float('nan'),
                            "time_total": 0.0})

    print(f"\n\n{'='*90}\nSUMMARY Newton+LU sweep\n{'='*90}", flush=True)
    print(f"{'dataset':<14s} {'n_vars':>8s} {'it_b':>5s} {'it_s':>5s} {'walras_s':>11s} "
          f"{'VIWS%':>10s} {'gap_pp':>8s} {'rel%':>7s} {'time(s)':>9s}", flush=True)
    for r in results:
        print(f"{r['name']:<14s} {r['n_vars']:>8d} {r['iters_b']:>5d} {r['iters_s']:>5d} "
              f"{r['walras_s']:>11.2e} {r['viws_pct']:>+10.4f} {r['gap_pp']:>+8.3f} "
              f"{r['rel_pct']:>6.2f}% {r['time_total']:>9.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
