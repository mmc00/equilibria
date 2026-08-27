"""cuDSS (NVIDIA GPU multifrontal direct solver) backend for the TR linear solve.

Proven on the real 20x41 GTAP Jacobian (Kaggle kernel ``cudss-config`` v11, 2026-08-26):
the squared J·P system (n=395310, nnz=1.64M, unsymmetric, wide dynamic range 1e-31..1e2)
solves at rel_res=5.25e-16 — BETTER than MUMPS's 1.37e-13 — in 3.82s vs MUMPS's 50.6s
(13.3x). The decisive lever is ``matching_algorithm=AUTO`` (cuDSS's max-weight matching,
the analogue of MUMPS ICNTL(6): it moves large entries onto the diagonal so the near-zero
GMIN diagonals don't wreck the factorization). Iterative refinement (``ir_num_steps=2``)
then drives the residual to machine precision. Explicit ``pivot_type`` is NOT_SUPPORTED on
older GPUs (e.g. P100) and is not needed — matching + IR suffices.

This is an OPT-IN backend (``EQUILIBRIA_GTAP_TR_LINSOLVE=cudss``); MUMPS stays the default.
``cudss_available()`` is the guard the caller uses to decide fallback; it never raises.
"""
from __future__ import annotations

import os

# cuDSS matching / IR knobs are overridable via env for tuning, but default to the proven
# v11 config. matching_algorithm=AUTO (6) + ir_num_steps=2 is the sweet spot.
_MATCHING_DEFAULT = os.environ.get("EQUILIBRIA_GTAP_CUDSS_MATCHING", "AUTO")
_IR_STEPS_DEFAULT = int(os.environ.get("EQUILIBRIA_GTAP_CUDSS_IR_STEPS", "2"))


def cudss_available() -> bool:
    """True iff cupy + nvmath (cuDSS) import AND a CUDA device is present.

    This is the fallback guard: it MUST return a bool and never raise, so the caller can
    do ``if cudss_available(): ... else: <fallback>`` safely on any machine.
    """
    try:
        import cupy as cp  # noqa: F401
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def cudss_solve(Jm_csr, rhs):
    """Solve ``Jm_csr @ x = rhs`` on the GPU with the proven cuDSS config.

    Parameters
    ----------
    Jm_csr : scipy.sparse.csr_matrix (float64, square, zero-free diagonal)
        The J·P operator run_gtap already builds for MUMPS (colperm-paired, GMIN applied).
    rhs : 1-D numpy.ndarray (float64)
        The right-hand side (run_gtap passes ``-_F_tr``).

    Returns
    -------
    (x, info) : (numpy.ndarray or None, dict)
        ``x`` is the solution (numpy, host) or None on failure. ``info`` has keys
        ``ok`` (bool), ``rel_res`` (float, when computed) and ``err`` (str, on failure).
        On ANY failure the function returns ``(None, {"ok": False, "err": ...})`` rather
        than raising — the caller falls back to MUMPS / the gradient step.
    """
    import numpy as np

    info: dict = {"ok": False}
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        from nvmath.bindings import cudss as cb
        from nvmath.sparse.advanced import DirectSolver

        A = Jm_csr.tocsr()
        # cuDSS wants int32 indices + float64 values.
        A = A.astype(np.float64)
        A_g = csp.csr_matrix(
            (
                cp.asarray(A.data, dtype=cp.float64),
                cp.asarray(A.indices, dtype=cp.int32),
                cp.asarray(A.indptr, dtype=cp.int32),
            ),
            shape=A.shape,
        )
        b_g = cp.asarray(np.asarray(rhs, dtype=np.float64))

        matching = getattr(cb.MatchingAlg, _MATCHING_DEFAULT, cb.MatchingAlg.AUTO)

        s = DirectSolver(A_g, b_g)
        try:
            # THE lever: max-weight matching (MUMPS ICNTL(6) analogue) — without it the
            # near-zero GMIN diagonals give a garbage factorization (rel_res ~ 9).
            s.plan_config.matching_algorithm = matching
            # iterative refinement -> machine-precision residual on the ill-conditioned system
            if _IR_STEPS_DEFAULT > 0:
                s.solution_config.ir_num_steps = _IR_STEPS_DEFAULT
            s.plan()
            s.factorize()
            x_g = s.solve()
            cp.cuda.Device().synchronize()
            x = cp.asnumpy(x_g).reshape(-1)
        finally:
            try:
                s.free()
            except Exception:
                pass

        if not np.all(np.isfinite(x)):
            info["err"] = "non-finite solution"
            return None, info
        rel_res = float(
            np.linalg.norm(A @ x - np.asarray(rhs, dtype=np.float64))
            / max(1.0, float(np.linalg.norm(rhs)))
        )
        info.update(ok=True, rel_res=rel_res)
        return x, info
    except Exception as e:  # noqa: BLE001 — any GPU/binding failure -> clean fallback
        info["err"] = f"{type(e).__name__}: {e}"
        return None, info


def _pattern_sig(Jm_csr):
    """A cheap, exact signature of the sparsity PATTERN (not the values) of a CSR matrix.

    Two matrices with the same nnz + identical indices/indptr have the same signature, so
    the cuDSS plan (symbolic analysis, pattern-only) can be reused. Any pattern change flips
    the signature and forces a re-plan. Mirrors lever B2's MUMPS symbolic-reuse guard.
    """
    import hashlib

    A = Jm_csr.tocsr()
    h = hashlib.sha256()
    h.update(A.indices.tobytes())
    h.update(A.indptr.tobytes())
    return (A.shape, int(A.nnz), h.hexdigest())


class CudssReusableSolver:
    """Stateful cuDSS solver that REUSES the plan (symbolic analysis) across Newton steps.

    The per-factorization win (13x vs MUMPS on the 20x41) only translates end-to-end if the
    expensive plan() (analysis/reordering, pattern-only) is done ONCE and reused — otherwise
    every Newton step re-analyzes, exactly the waste lever B2 fixed for MUMPS. This class holds
    a live DirectSolver + the GPU CSR buffers; on a same-pattern call it updates the values in
    place and re-factorizes (skips plan()); on a pattern change it rebuilds and re-plans.

    ``n_plans`` counts how many times plan() actually ran — the reuse assertion in the tests.
    Any GPU/binding failure returns ``(None, {"ok": False, ...})`` and resets state, so the
    caller falls back cleanly (never raises).
    """

    def __init__(self):
        self.n_plans = 0
        self._sig = None
        self._solver = None
        self._A_g = None  # cupy CSR (values updated in place across reuse)
        self._b_g = None
        self._cp = None

    def _reset(self):
        if self._solver is not None:
            try:
                self._solver.free()
            except Exception:
                pass
        self._solver = None
        self._A_g = None
        self._b_g = None
        self._sig = None

    def solve(self, Jm_csr, rhs):
        import numpy as np

        info: dict = {"ok": False}
        try:
            import cupy as cp
            import cupyx.scipy.sparse as csp
            from nvmath.bindings import cudss as cb
            from nvmath.sparse.advanced import DirectSolver

            self._cp = cp
            A = Jm_csr.tocsr().astype(np.float64)
            b = np.asarray(rhs, dtype=np.float64)
            sig = _pattern_sig(A)
            matching = getattr(
                cb.MatchingAlg, _MATCHING_DEFAULT, cb.MatchingAlg.AUTO
            )

            if self._solver is not None and sig == self._sig:
                # SAME pattern → reuse plan: update the VALUES in the shared GPU buffer IN
                # PLACE and re-factorize only. CRITICAL: do NOT pass a= to reset_operands —
                # that hands cuDSS a "new" LHS and invalidates the plan ("Factorization cannot
                # be performed before plan() has been called"). nvmath's own guidance is
                # "update the values in place and refactorize"; passing only b= keeps the plan
                # (verified on the real matrix: plans=1 across a simulated Newton loop,
                # rel_res 1e-13). Each plan() costs ~3.6s at n=395k — this is what we save.
                self._A_g.data[:] = cp.asarray(A.data, dtype=cp.float64)
                self._b_g[:] = cp.asarray(b)
                self._solver.reset_operands(b=self._b_g)
                self._solver.factorize()
                x_g = self._solver.solve()
            else:
                # NEW/changed pattern → (re)build the solver and plan once.
                self._reset()
                self._A_g = csp.csr_matrix(
                    (
                        cp.asarray(A.data, dtype=cp.float64),
                        cp.asarray(A.indices, dtype=cp.int32),
                        cp.asarray(A.indptr, dtype=cp.int32),
                    ),
                    shape=A.shape,
                )
                self._b_g = cp.asarray(b)
                self._solver = DirectSolver(self._A_g, self._b_g)
                self._solver.plan_config.matching_algorithm = matching
                if _IR_STEPS_DEFAULT > 0:
                    self._solver.solution_config.ir_num_steps = _IR_STEPS_DEFAULT
                self._solver.plan()
                self.n_plans += 1
                self._sig = sig
                self._solver.factorize()
                x_g = self._solver.solve()

            cp.cuda.Device().synchronize()
            x = cp.asnumpy(x_g).reshape(-1)
            if not np.all(np.isfinite(x)):
                info["err"] = "non-finite solution"
                self._reset()
                return None, info
            rel_res = float(
                np.linalg.norm(A @ x - b) / max(1.0, float(np.linalg.norm(b)))
            )
            info.update(ok=True, rel_res=rel_res)
            return x, info
        except Exception as e:  # noqa: BLE001
            info["err"] = f"{type(e).__name__}: {e}"
            self._reset()
            return None, info
