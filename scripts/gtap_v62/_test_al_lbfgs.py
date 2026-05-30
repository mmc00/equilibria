"""Augmented Lagrangian + L-BFGS-B solver for GTAP v6.2.

100% Python (no Julia, no GAMS, no IPOPT). Outer AL loop manages multipliers
and penalty; inner L-BFGS-B uses analytic gradients from PyomoNLP.

Tested on gtap6_3x3: baseline + shocked (tms Food/USA/EU_28 -10%).
Reference VIWS change: +62.3585% (GEMPACK).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pyomo.environ import (
    Objective, minimize as pyo_min, value, Var, ConstraintList,
)
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
    # Trivial constant objective: AL handles the feasibility entirely.
    model.obj = Objective(expr=0.0, sense=pyo_min)


class ALSolver:
    """Augmented Lagrangian outer loop with L-BFGS-B inner.

    L_rho(x, lam) = f(x) + lam^T c(x) + (rho/2) ||c(x)||^2
    grad_x L_rho  = grad_f + J(x)^T (lam + rho c(x))
    """

    def __init__(self, nlp: PyomoNLP, *, rho0: float = 10.0,
                 rho_max: float = 1.0e12, eta: float = 0.75,
                 inner_maxiter: int = 1000, inner_gtol: float = 1.0e-8,
                 inner_ftol: float = 1.0e-15):
        self.nlp = nlp
        self.rho = rho0
        self.rho_max = rho_max
        self.eta = eta
        self.inner_maxiter = inner_maxiter
        self.inner_gtol = inner_gtol
        self.inner_ftol = inner_ftol

        self.n = nlp.n_primals()
        self.m_eq = nlp.n_eq_constraints()
        self.m_ineq = nlp.n_ineq_constraints()
        if self.m_ineq != 0:
            raise RuntimeError(f"AL solver expects only equality constraints, got {self.m_ineq} ineq.")

        self.lb = nlp.primals_lb().copy()
        self.ub = nlp.primals_ub().copy()
        # scipy L-BFGS-B treats inf bounds via None
        self.bounds = [
            (None if not np.isfinite(self.lb[i]) else float(self.lb[i]),
             None if not np.isfinite(self.ub[i]) else float(self.ub[i]))
            for i in range(self.n)
        ]

        self.x = nlp.get_primals().copy()
        # Clamp inside bounds to avoid bound violations at start.
        for i in range(self.n):
            lo, hi = self.bounds[i]
            if lo is not None and self.x[i] < lo:
                self.x[i] = lo
            if hi is not None and self.x[i] > hi:
                self.x[i] = hi
        self.lam = np.zeros(self.m_eq)

        # Constraint scaling: row-equilibrate by max-abs Jacobian element at x0.
        nlp.set_primals(self.x)
        try:
            J0 = nlp.evaluate_jacobian_eq()
            if sp.issparse(J0):
                Jcsr = J0.tocsr()
                row_max = np.zeros(self.m_eq)
                for i in range(self.m_eq):
                    row = Jcsr.getrow(i)
                    if row.nnz > 0:
                        row_max[i] = float(np.max(np.abs(row.data)))
            else:
                row_max = np.max(np.abs(J0), axis=1)
            self.cscale = np.where(row_max > 1.0e-12, 1.0 / row_max, 1.0)
        except Exception:
            self.cscale = np.ones(self.m_eq)

    def _set_x(self, x: np.ndarray):
        self.nlp.set_primals(x)

    def _c(self) -> np.ndarray:
        return self.nlp.evaluate_eq_constraints() * self.cscale

    def _c_raw(self) -> np.ndarray:
        return self.nlp.evaluate_eq_constraints()

    def _grad_f(self) -> np.ndarray:
        return self.nlp.evaluate_grad_objective()

    def _jac_eq(self):
        J = self.nlp.evaluate_jacobian_eq()
        # Scale rows
        if sp.issparse(J):
            D = sp.diags(self.cscale)
            return (D @ J).tocsr()
        return (self.cscale[:, None] * J)

    def _aug_fg(self, x: np.ndarray):
        # Avoid NaN/Inf propagation: clamp x then re-eval.
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            return 1.0e20, np.zeros_like(x)
        self._set_x(x)
        try:
            c = self._c()
            gf = self._grad_f()
            J = self._jac_eq()
        except Exception:
            return 1.0e20, np.zeros_like(x)
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(gf))):
            return 1.0e20, np.zeros_like(x)
        f = self.nlp.evaluate_objective()
        # AL value
        L = float(f) + float(self.lam @ c) + 0.5 * self.rho * float(c @ c)
        # AL gradient (J is CSR)
        mult = self.lam + self.rho * c
        if sp.issparse(J):
            g = gf + J.T.dot(mult)
        else:
            g = gf + J.T @ mult
        if not np.all(np.isfinite(g)):
            return 1.0e20, np.zeros_like(x)
        return float(L), np.asarray(g, dtype=float)

    def inner_solve(self):
        res = minimize(
            self._aug_fg, self.x, jac=True, method="L-BFGS-B",
            bounds=self.bounds,
            options={
                "maxiter": self.inner_maxiter,
                "gtol": self.inner_gtol,
                "ftol": 0.0,  # disable f-reduction stop; only stop on gradient
                "maxcor": 100,
                "maxfun": 20 * self.inner_maxiter,
            },
        )
        self.x = np.asarray(res.x, dtype=float)
        return res

    def diagnose_worst(self, k: int = 5):
        c = self._c()
        if c.size == 0:
            return []
        names = self.nlp.equality_constraint_names()
        order = np.argsort(-np.abs(c))[:k]
        return [(names[i], float(c[i])) for i in order]

    def diagnose_gradient(self):
        """Verify AL gradient magnitude + bound activity at worst constraints."""
        c = self._c()
        gf = self._grad_f()
        J = self._jac_eq()
        mult = self.lam + self.rho * c
        g = gf + (J.T.dot(mult) if sp.issparse(J) else J.T @ mult)
        # Active set diag
        var_names = self.nlp.primals_names()
        n_at_lb = sum(1 for i in range(self.n)
                      if self.bounds[i][0] is not None and self.x[i] <= self.bounds[i][0] + 1.0e-10)
        n_at_ub = sum(1 for i in range(self.n)
                      if self.bounds[i][1] is not None and self.x[i] >= self.bounds[i][1] - 1.0e-10)
        # Largest gradient components on free vars (not at bound)
        free_mask = np.ones(self.n, dtype=bool)
        for i in range(self.n):
            if self.bounds[i][0] is not None and self.x[i] <= self.bounds[i][0] + 1.0e-10:
                free_mask[i] = False
            if self.bounds[i][1] is not None and self.x[i] >= self.bounds[i][1] - 1.0e-10:
                free_mask[i] = False
        g_free = np.where(free_mask, g, 0.0)
        gnorm = float(np.max(np.abs(g_free)))
        return {"gnorm_free": gnorm, "n_lb": n_at_lb, "n_ub": n_at_ub,
                "var_names": var_names, "g": g, "free_mask": free_mask}

    def solve(self, *, outer_maxiter: int = 80,
              feas_tol: float = 1.0e-6, verbose: bool = True,
              diag_every: int = 0):
        self._set_x(self.x)
        c0 = self._c()
        prev_inf = float(np.max(np.abs(c0))) if c0.size else 0.0
        history = []
        t0 = time.perf_counter()
        if verbose:
            print(f"  [AL] start: rho={self.rho:.2e}  ||c||_inf={prev_inf:.3e}", flush=True)

        stagnation = 0
        for k in range(1, outer_maxiter + 1):
            t_iter = time.perf_counter()
            res = self.inner_solve()
            self._set_x(self.x)
            c = self._c()
            inf = float(np.max(np.abs(c))) if c.size else 0.0
            iter_dt = time.perf_counter() - t_iter
            total_dt = time.perf_counter() - t0

            # Multiplier update (safeguarded)
            self.lam = self.lam + self.rho * c
            # Cap multipliers to avoid blow-up
            np.clip(self.lam, -1.0e12, 1.0e12, out=self.lam)

            improvement = (prev_inf - inf) / max(prev_inf, 1.0e-30)
            if inf > self.eta * prev_inf:
                self.rho = min(self.rho * 2.0, self.rho_max)
            prev_inf = inf

            # Stagnation tracking
            if improvement < 1.0e-4:
                stagnation += 1
            else:
                stagnation = 0

            history.append({"k": k, "rho": self.rho, "inf": inf,
                            "inner_iter": int(res.nit), "dt": iter_dt})
            if verbose:
                status = res.message if isinstance(res.message, str) else str(res.message)
                status = status[:60]
                print(f"  [AL] k={k:02d}  rho={self.rho:.2e}  "
                      f"||c||_inf={inf:.3e}  inner={int(res.nit):3d}  "
                      f"dt={iter_dt:5.1f}s  T={total_dt:6.1f}s  "
                      f"({status})", flush=True)
                if diag_every > 0 and k % diag_every == 0:
                    for nm, vv in self.diagnose_worst(5):
                        print(f"    worst: {nm[:80]} = {vv:+.3e}", flush=True)

            if inf < feas_tol:
                if verbose:
                    print(f"  [AL] converged: ||c||_inf={inf:.3e} < {feas_tol:.0e}", flush=True)
                return {"converged": True, "iters": k, "inf": inf,
                        "time": total_dt, "history": history}

            if stagnation >= 10:
                if verbose:
                    print(f"  [AL] STAGNATION: feasibility flat for 10 iters at "
                          f"||c||_inf={inf:.3e}", flush=True)
                    for nm, vv in self.diagnose_worst(10):
                        print(f"    worst: {nm[:80]} = {vv:+.3e}", flush=True)
                    diag = self.diagnose_gradient()
                    print(f"    AL gradient on free vars: max |g|={diag['gnorm_free']:.3e}  "
                          f"vars_at_lb={diag['n_lb']}  vars_at_ub={diag['n_ub']}", flush=True)
                    # Show variables with largest gradient components on free vars
                    g = diag['g']; mask = diag['free_mask']
                    g_masked = np.where(mask, np.abs(g), 0.0)
                    top_idx = np.argsort(-g_masked)[:8]
                    for i in top_idx:
                        if g_masked[i] > 0:
                            print(f"    grad: {diag['var_names'][i][:60]} = {g[i]:+.3e}  x={self.x[i]:.3e}",
                                  flush=True)
                return {"converged": False, "iters": k, "inf": inf,
                        "time": total_dt, "history": history,
                        "stagnated": True}

        return {"converged": False, "iters": outer_maxiter, "inf": prev_inf,
                "time": time.perf_counter() - t0, "history": history}


def _walras(model) -> float:
    try:
        return float(value(model.walras))
    except Exception:
        return float("nan")


def _viws(model) -> float:
    return float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))


def main():
    print("=== gtap6_3x3 Augmented Lagrangian + L-BFGS-B ===", flush=True)

    model, params, pipe = _build_pyomo_model()
    closure = pipe['closure']
    print(f"free={closure['free_vars']} cons={closure['active_cons']} "
          f"mismatch={closure['mismatch']} baked={pipe['prebalance']['n_baked']}",
          flush=True)

    _attach_trivial_objective(model)

    # ---- BASELINE ----
    print("\n--- BASELINE ---", flush=True)
    t0 = time.perf_counter()
    nlp = PyomoNLP(model)
    print(f"NLP: n={nlp.n_primals()}  m_eq={nlp.n_eq_constraints()}  "
          f"m_ineq={nlp.n_ineq_constraints()}  nnz_J_eq={nlp.nnz_jacobian_eq()}",
          flush=True)

    al = ALSolver(nlp, rho0=10.0, inner_maxiter=300, inner_gtol=1.0e-7)
    res = al.solve(outer_maxiter=80, feas_tol=1.0e-6)
    nlp.set_primals(al.x)
    nlp.load_state_into_pyomo()
    t_base = time.perf_counter() - t0
    w_base = _walras(model)
    viws_0 = _viws(model)
    print(f"\nBASELINE: converged={res['converged']}  inf={res['inf']:.3e}  "
          f"walras={w_base:.4e}  VIWS_0={viws_0:.4e}  time={t_base:.1f}s",
          flush=True)

    if not res['converged'] or abs(w_base) > 1.0e-3:
        print(f"BASELINE FAILED: walras={w_base:.3e} (need < 1e-6 ideally)", flush=True)
        # Continue to shocked anyway to see behavior, but flag.

    # ---- SHOCKED (continuation in N steps) ----
    print("\n--- SHOCKED (continuation) ---", flush=True)
    old_tms = float(value(model.tms[SHOCK_KEY]))
    target_tms = (1.0 + old_tms) * 0.9 - 1.0
    n_cont = 10
    print(f"  tms[{SHOCK_KEY}]: {old_tms:.6f} -> {target_tms:.6f} in {n_cont} steps",
          flush=True)

    t0 = time.perf_counter()
    res_s = None
    for step in range(1, n_cont + 1):
        frac = step / n_cont
        new_tms = old_tms + frac * (target_tms - old_tms)
        model.tms[SHOCK_KEY] = new_tms
        print(f"\n  [cont {step}/{n_cont}] tms = {new_tms:.6f}", flush=True)
        nlp_s = PyomoNLP(model)
        al_s = ALSolver(nlp_s, rho0=10.0, inner_maxiter=500, inner_gtol=1.0e-8)
        res_s = al_s.solve(outer_maxiter=40, feas_tol=1.0e-6, diag_every=20)
        nlp_s.set_primals(al_s.x)
        nlp_s.load_state_into_pyomo()
        if not res_s['converged']:
            print(f"  [cont {step}] FAILED to converge; stopping continuation", flush=True)
            break
    t_shock = time.perf_counter() - t0
    w_shock = _walras(model)
    viws_final = _viws(model)
    viws_pct = 100.0 * (viws_final - viws_0) / viws_0
    gap = viws_pct - GEMPACK_VIWS_PCT
    print(f"\nSHOCKED: converged={res_s['converged']}  inf={res_s['inf']:.3e}  "
          f"walras={w_shock:.4e}  time={t_shock:.1f}s", flush=True)

    print("\n=== RESULT ===", flush=True)
    print(f"VIWS change: {viws_pct:+.4f}%   GEMPACK ref +{GEMPACK_VIWS_PCT}%   "
          f"gap={gap:+.4f}pp", flush=True)

    ok_base = res['converged'] and abs(w_base) < 1.0e-3
    ok_shock = res_s['converged'] and abs(gap) < TOL_VIWS_GAP_PCT
    print(f"\nverdict: baseline_ok={ok_base}  shocked_ok={ok_shock}", flush=True)
    return 0 if (ok_base and ok_shock) else 1


if __name__ == "__main__":
    sys.exit(main())
