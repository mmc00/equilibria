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
