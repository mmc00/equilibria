"""Newton + scipy sparse LU (SuperLU) solver for GTAP v6.2 gtap6_3x3.

Square root-finding on c(x) = 0 using analytic Jacobian from PyomoNLP.
At each iteration: solve J(x_k) dx = -c(x_k), then line search with bounds.

Validation:
  - baseline: should converge in 1 iter (already baked at residual < 1e-6).
  - shocked tms(Food,USA,EU_28) via tm_pct: new = (1+old)*0.9 - 1.
    GEMPACK reference VIWS change: +62.3585%. Tolerance: gap < 2%.
"""
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


DATA = Path("datasets/gtap6_3x3")
SHOCK_KEY = ("Food", "USA", "EU_28")
GEMPACK_VIWS_PCT = 62.3585
TOL_VIWS_GAP_PCT = 2.0


def _build_pyomo_model():
    sets = GTAPv62Sets()
    sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
    params = GTAPv62Parameters()
    params.load_from_har(basedata_path=DATA / "basedata.har",
                         default_prm_path=DATA / "default.prm", sets=sets)
    model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()
    pipe = apply_v62_pipeline(model, mode="nlp", bake_tolerance=1.0e-9,
                              params=params, drop_dead_rows_threshold=0.0)
    return model, params, pipe


def _attach_trivial_objective(model):
    if hasattr(model, "obj"):
        try:
            model.del_component("obj")
        except Exception:
            pass
    model.obj = Objective(expr=0.0, sense=pyo_min)


def _project(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


def _max_alpha_to_bounds(x: np.ndarray, dx: np.ndarray,
                         lb: np.ndarray, ub: np.ndarray,
                         tau: float = 0.995) -> float:
    """Largest alpha in (0, 1] such that x + alpha*dx stays strictly inside bounds.

    Uses the fraction-to-the-boundary rule (tau < 1).
    Returns 1.0 if no bound is binding along the direction.
    """
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


def newton_solve(nlp: PyomoNLP, *,
                 tol: float = 1.0e-6,
                 max_iter: int = 50,
                 ls_min_alpha: float = 1.0e-10,
                 ls_shrink: float = 0.5,
                 ls_armijo: float = 1.0e-4,
                 verbose: bool = True):
    n = nlp.n_primals()
    m_eq = nlp.n_eq_constraints()
    m_ineq = nlp.n_ineq_constraints()
    if m_ineq != 0:
        raise RuntimeError(f"Newton solver expects only equality constraints, got {m_ineq} ineq.")
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
        if sp.issparse(J):
            J_csc = J.tocsc()
        else:
            J_csc = sp.csc_matrix(J)

        try:
            dx = spla.spsolve(J_csc, -c)
        except RuntimeError as e:
            if verbose:
                print(f"  [Newton] k={k:02d} LU SOLVE FAILED: {e}", flush=True)
            # Try LU with partial pivoting via splu explicitly to get error detail
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

        # Cap step by fraction-to-the-boundary
        alpha_max = _max_alpha_to_bounds(x, dx, lb, ub, tau=0.995)
        alpha = min(1.0, alpha_max)

        # Armijo-like backtracking on ||c||_inf (or 0.5||c||^2)
        c_norm2_prev = 0.5 * float(c @ c)
        # Descent in 0.5||c||^2 along Newton direction is exact: grad . dx = -2 * c_norm2_prev
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
            # Accept if Armijo on 0.5||c||^2 OR strict reduction on inf-norm
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


def _walras(model) -> float:
    try:
        return float(value(model.walras))
    except Exception:
        return float("nan")


def _viws(model) -> float:
    return float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))


def main():
    print("=== gtap6_3x3 Newton + scipy sparse LU (SuperLU) ===", flush=True)

    model, params, pipe = _build_pyomo_model()
    closure = pipe['closure']
    print(f"free={closure['free_vars']} cons={closure['active_cons']} "
          f"mismatch={closure['mismatch']} baked={pipe['prebalance']['n_baked']}",
          flush=True)

    _attach_trivial_objective(model)

    # ---------- BASELINE ----------
    print("\n--- BASELINE ---", flush=True)
    nlp = PyomoNLP(model)
    res = newton_solve(nlp, tol=1.0e-6, max_iter=30)
    nlp.set_primals(res["x"])
    nlp.load_state_into_pyomo()
    w_base = _walras(model)
    viws_0 = _viws(model)
    print(f"\nBASELINE: converged={res['converged']}  iters={res['iters']}  "
          f"inf={res['inf']:.3e}  walras={w_base:.4e}  "
          f"VIWS_0={viws_0:.4e}  time={res['time']:.2f}s", flush=True)
    if not res['converged']:
        print(f"  ERROR: {res.get('error', '(no error string)')}", flush=True)
        return 1

    # ---------- SHOCKED ----------
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
    viws_final = _viws(model)
    viws_pct = 100.0 * (viws_final - viws_0) / viws_0
    gap = viws_pct - GEMPACK_VIWS_PCT
    print(f"\nSHOCKED: converged={res_s['converged']}  iters={res_s['iters']}  "
          f"inf={res_s['inf']:.3e}  walras={w_shock:.4e}  time={res_s['time']:.2f}s",
          flush=True)
    if not res_s['converged']:
        print(f"  ERROR: {res_s.get('error', '(no error string)')}", flush=True)

    print("\n=== RESULT ===", flush=True)
    print(f"VIWS change: {viws_pct:+.4f}%   GEMPACK ref +{GEMPACK_VIWS_PCT}%   "
          f"gap={gap:+.4f}pp", flush=True)

    ok_base = res['converged'] and abs(w_base) < 1.0e-3
    ok_shock = res_s['converged'] and abs(gap) < TOL_VIWS_GAP_PCT
    print(f"\nverdict: baseline_ok={ok_base}  shocked_ok={ok_shock}", flush=True)
    return 0 if (ok_base and ok_shock) else 1


if __name__ == "__main__":
    sys.exit(main())
