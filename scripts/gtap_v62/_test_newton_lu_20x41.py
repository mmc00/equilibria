"""Newton + scipy sparse LU on gtap6_20x41 (~286K vars).

Standalone — never solved before by any solver. Tolerance laxa: <5% gap.
Tipos de timing por iter: jac eval, spsolve, line search.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# UMFPACK with manual context (need to bump ALLOC_INIT to avoid OOM on
# 286K x 286K with high fill-in). SuperLU's dgstrf is single-threaded and
# took >20 min for the first factorization in earlier runs.
import scikits.umfpack as umf

# Build a reusable UMFPACK context with bigger workspace allocation.
_UMF_CTX = umf.UmfpackContext('di')
# Bump initial workspace from 70% to 5x of estimate — avoids OOM error
# when high fill-in pushes UMFPACK past its conservative default.
_UMF_CTX.control[umf.UMFPACK_ALLOC_INIT] = 5.0
# Strategy=2 forces UNSYMMETRIC (auto-detect can mis-pick symmetric on
# near-symmetric patterns and produce bad fill).
_UMF_CTX.control[umf.UMFPACK_STRATEGY] = 2.0
# Aggressive absorption helps reduce fill on near-block-triangular patterns.
_UMF_CTX.control[umf.UMFPACK_AGGRESSIVE] = 1.0

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pyomo.environ import Objective, minimize as pyo_min, value
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore


DATA = Path("datasets/gtap6_20x41")
SHOCK_KEY = ("FoodProd", "USA", "EU_28")
GEMPACK_VIWS_PCT = 51.432
TOL_VIWS_GAP_REL = 5.0  # <5% relative gap


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


def newton_solve_verbose(nlp, *, tol=1.0e-6, max_iter=30,
                         ls_min_alpha=1.0e-10, ls_shrink=0.5, ls_armijo=1.0e-4):
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
    t_c0 = time.perf_counter()
    c = nlp.evaluate_eq_constraints()
    dt_c0 = time.perf_counter() - t_c0
    inf0 = float(np.max(np.abs(c))) if c.size else 0.0

    print(f"  [Newton] n={n}  m_eq={m_eq}  nnz_J={nlp.nnz_jacobian_eq()}", flush=True)
    print(f"  [Newton] initial c eval dt={dt_c0:.2f}s  ||c||_inf={inf0:.3e}", flush=True)

    if inf0 < tol:
        return {"converged": True, "iters": 0, "inf": inf0, "x": x, "time": 0.0}

    t0 = time.perf_counter()
    prev_inf = inf0

    for k in range(1, max_iter + 1):
        t_iter = time.perf_counter()

        t_jac = time.perf_counter()
        J = nlp.evaluate_jacobian_eq()
        J_csc = J.tocsc() if sp.issparse(J) else sp.csc_matrix(J)
        J_csc.sort_indices()
        dt_jac = time.perf_counter() - t_jac

        t_solve = time.perf_counter()
        try:
            # symbolic + numeric + solve via context (with tuned Control)
            _UMF_CTX.symbolic(J_csc)
            t_num = time.perf_counter()
            _UMF_CTX.numeric(J_csc)
            dt_num = time.perf_counter() - t_num
            dx = _UMF_CTX.solve(umf.UMFPACK_A, J_csc, -c, autoTranspose=True)
            _UMF_CTX.free()
            print(f"  [Newton] k={k:02d}  umfpack numeric dt={dt_num:.2f}s", flush=True)
        except Exception as e:
            print(f"  [Newton] k={k:02d} umfpack failed: {e}", flush=True)
            return {"converged": False, "iters": k, "inf": prev_inf, "x": x,
                    "time": time.perf_counter() - t0,
                    "error": f"UMFPACK failed: {e}"}
        dt_solve = time.perf_counter() - t_solve

        if not np.all(np.isfinite(dx)):
            return {"converged": False, "iters": k, "inf": prev_inf, "x": x,
                    "time": time.perf_counter() - t0,
                    "error": "dx has non-finite entries (singular J?)"}

        t_ls = time.perf_counter()
        alpha_max = _max_alpha_to_bounds(x, dx, lb, ub, tau=0.995)
        alpha = min(1.0, alpha_max)

        c_norm2_prev = 0.5 * float(c @ c)
        grad_dot_dx = -2.0 * c_norm2_prev

        accepted = False
        x_new = x
        c_new = c
        inf_new = prev_inf
        ls_tries = 0
        while alpha > ls_min_alpha:
            ls_tries += 1
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
        dt_ls = time.perf_counter() - t_ls

        if not accepted:
            print(f"  [Newton] k={k:02d} LINE SEARCH FAILED  alpha_min={alpha:.2e}  tries={ls_tries}", flush=True)
            return {"converged": False, "iters": k, "inf": prev_inf, "x": x,
                    "time": time.perf_counter() - t0,
                    "error": "line search failed"}

        x, c = x_new, c_new
        iter_dt = time.perf_counter() - t_iter
        total_dt = time.perf_counter() - t0

        print(f"  [Newton] k={k:02d}  ||c||_inf={inf_new:.3e}  alpha={alpha:.3e}  "
              f"||dx||_inf={float(np.max(np.abs(dx))):.3e}  "
              f"jac={dt_jac:5.2f}s  solve={dt_solve:6.2f}s  ls={dt_ls:5.2f}s(t{ls_tries})  "
              f"iter={iter_dt:6.2f}s  T={total_dt:7.2f}s", flush=True)

        if inf_new < tol:
            nlp.set_primals(x)
            return {"converged": True, "iters": k, "inf": inf_new, "x": x, "time": total_dt}
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


def main():
    print(f"=== gtap6_20x41 Newton + UMFPACK direct ===", flush=True)
    print(f"shock target: tms{SHOCK_KEY}  GEMPACK ref +{GEMPACK_VIWS_PCT}%", flush=True)

    t_total = time.perf_counter()

    t_load = time.perf_counter()
    sets = GTAPv62Sets()
    sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
    print(f"Sectors |i|={len(sets.i)} regions |r|={len(sets.r)} factors |f|={len(sets.f)}", flush=True)

    shock_key = SHOCK_KEY
    if shock_key[0] not in sets.i:
        alts = [c for c in sets.i if 'food' in c.lower() or 'proc' in c.lower()]
        print(f"WARNING: '{shock_key[0]}' not in sets.i. Alternatives: {alts}", flush=True)
        if alts:
            shock_key = (alts[0], shock_key[1], shock_key[2])
            print(f"Using {shock_key} instead", flush=True)
        else:
            print("FATAL: no food-related commodity found", flush=True)
            return 1

    params = GTAPv62Parameters()
    params.load_from_har(basedata_path=DATA / "basedata.har",
                         default_prm_path=DATA / "default.prm", sets=sets)
    print(f"params loaded ({time.perf_counter()-t_load:.1f}s)", flush=True)

    t_build = time.perf_counter()
    model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()
    print(f"model built ({time.perf_counter()-t_build:.1f}s)", flush=True)

    t_pipe = time.perf_counter()
    pipe = apply_v62_pipeline(model, mode="nlp", bake_tolerance=1.0e-9,
                              params=params, conditional_fixing=True,
                              drop_dead_rows_threshold=0.0)
    closure = pipe['closure']
    print(f"pipeline ({time.perf_counter()-t_pipe:.1f}s) | free={closure['free_vars']} "
          f"cons={closure['active_cons']} mismatch={closure['mismatch']} "
          f"baked={pipe['prebalance']['n_baked']}", flush=True)

    _attach_trivial_objective(model)

    # ---------- BASELINE ----------
    print("\n--- BASELINE ---", flush=True)
    t_nlp = time.perf_counter()
    nlp = PyomoNLP(model)
    print(f"PyomoNLP construction dt={time.perf_counter()-t_nlp:.1f}s", flush=True)

    res = newton_solve_verbose(nlp, tol=1.0e-6, max_iter=30)
    nlp.set_primals(res["x"])
    nlp.load_state_into_pyomo()
    try:
        w_base = float(value(model.walras))
    except Exception:
        w_base = float('nan')
    viws_0 = float(value(model.qxs[shock_key]) * value(model.pmcif[shock_key]))
    print(f"\nBASELINE: converged={res['converged']} iters={res['iters']} "
          f"inf={res['inf']:.3e} walras={w_base:.4e} VIWS_0={viws_0:.4e} time={res['time']:.2f}s",
          flush=True)
    if not res['converged']:
        print(f"  ERROR: {res.get('error', '(no error string)')}", flush=True)
        print(f"  total wall-time so far: {time.perf_counter()-t_total:.1f}s", flush=True)
        return 1

    # ---------- SHOCKED ----------
    print("\n--- SHOCKED ---", flush=True)
    old_tms = float(value(model.tms[shock_key]))
    new_tms = (1.0 + old_tms) * 0.9 - 1.0
    model.tms[shock_key] = new_tms
    print(f"  tms[{shock_key}]: {old_tms:.6f} -> {new_tms:.6f}", flush=True)

    t_nlp2 = time.perf_counter()
    nlp_s = PyomoNLP(model)
    print(f"PyomoNLP reconstruction dt={time.perf_counter()-t_nlp2:.1f}s", flush=True)

    res_s = newton_solve_verbose(nlp_s, tol=1.0e-6, max_iter=60)
    nlp_s.set_primals(res_s["x"])
    nlp_s.load_state_into_pyomo()
    try:
        w_shock = float(value(model.walras))
    except Exception:
        w_shock = float('nan')
    viws_f = float(value(model.qxs[shock_key]) * value(model.pmcif[shock_key]))
    viws_pct = 100.0 * (viws_f - viws_0) / viws_0
    gap_pp = viws_pct - GEMPACK_VIWS_PCT
    rel_pct = abs(gap_pp) / GEMPACK_VIWS_PCT * 100.0

    print(f"\nSHOCKED: converged={res_s['converged']} iters={res_s['iters']} "
          f"inf={res_s['inf']:.3e} walras={w_shock:.4e} time={res_s['time']:.2f}s", flush=True)
    if not res_s['converged']:
        print(f"  ERROR: {res_s.get('error', '(no error string)')}", flush=True)

    total_dt = time.perf_counter() - t_total
    print(f"\n=== RESULT ===", flush=True)
    print(f"VIWS baseline: {viws_0:.4e}", flush=True)
    print(f"VIWS shocked:  {viws_f:.4e}", flush=True)
    print(f"VIWS change:   {viws_pct:+.4f}%   GEMPACK ref +{GEMPACK_VIWS_PCT}%   "
          f"gap={gap_pp:+.4f}pp ({rel_pct:.2f}%)", flush=True)
    print(f"Total wall-time: {total_dt:.1f}s ({total_dt/60:.2f} min)", flush=True)

    ok_base = res['converged'] and abs(w_base) < 1.0e-2
    ok_shock = res_s['converged'] and rel_pct < TOL_VIWS_GAP_REL
    print(f"\nverdict: baseline_ok={ok_base}  shocked_ok={ok_shock}", flush=True)
    return 0 if (ok_base and ok_shock) else 1


if __name__ == "__main__":
    sys.exit(main())
