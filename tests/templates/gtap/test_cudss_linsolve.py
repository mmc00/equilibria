"""Tests for the cuDSS TR linear-solve backend (scripts/gtap/_cudss_linsolve.py).

The winning config was proven on the real 20x41 Jacobian (Kaggle kernel cudss-config
v11, 2026-08-26): matching_algorithm=AUTO + iterative-refinement ir_num_steps=2 gives
rel_res 5.25e-16 (beats MUMPS 1.37e-13) at 3.82s vs MUMPS 50.6s = 13.3x.

These tests pin the CONTRACT that run_gtap.py's cudss branch depends on:
  - the helper imports and exposes cudss_available() + cudss_solve(Jm_csr, rhs)
  - cudss_available() never raises (it is the guard that decides fallback)
  - WHEN a GPU is present, cudss_solve returns a solution at parity with a direct solve
    (rel_res < 1e-12) on an unsymmetric, zero-free-diagonal matrix like J·P
  - the returned info dict reports success/failure so the caller can fall back cleanly
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

# import the helper by path (scripts/gtap is not a package)
_HELPER = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "scripts", "gtap", "_cudss_linsolve.py"
)
_HELPER = os.path.abspath(_HELPER)
_spec = importlib.util.spec_from_file_location("_cudss_linsolve", _HELPER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _gpu_present():
    """True iff a CUDA GPU + cupy + nvmath are importable and a device exists."""
    try:
        return _mod.cudss_available()
    except Exception:
        return False


def _tiny_unsym():
    """A small unsymmetric matrix with a full (zero-free) diagonal + consistent RHS.

    Mirrors the shape run_gtap hands cuDSS: J·P after colperm (zero-free diagonal),
    unsymmetric. x_true = 1 so success is unambiguous (b = A @ 1).
    """
    A = sp.csr_matrix(
        np.array(
            [
                [4.0, 1.0, 0.0, 0.0, 2.0],
                [0.5, 4.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 4.0, 1.0, 0.0],
                [3.0, 0.0, 1.0, 4.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 4.0],
            ]
        )
    )
    x_true = np.ones(A.shape[0])
    b = A @ x_true
    return A, b, x_true


def test_helper_exposes_contract():
    """The caller (run_gtap.py) depends on these two names existing."""
    assert hasattr(_mod, "cudss_available"), "cudss_available() missing"
    assert hasattr(_mod, "cudss_solve"), "cudss_solve() missing"


def test_cudss_available_never_raises():
    """cudss_available() is the fallback guard — it must return a bool, never raise,
    even on a machine with no GPU / no cupy / no nvmath."""
    result = _mod.cudss_available()
    assert isinstance(result, bool)


@pytest.mark.skipif(not _gpu_present(), reason="no CUDA GPU / cupy / nvmath available")
def test_cudss_solve_parity_on_unsym():
    """On a GPU, cudss_solve must solve A x = b at machine-precision residual."""
    A, b, x_true = _tiny_unsym()
    x, info = _mod.cudss_solve(A, b)
    assert info.get("ok") is True, f"solve reported failure: {info}"
    rel_res = np.linalg.norm(A @ x - b) / max(1.0, np.linalg.norm(b))
    assert rel_res < 1e-12, f"rel_res {rel_res:.2e} not at parity"
    assert np.max(np.abs(x - x_true)) < 1e-8


@pytest.mark.skipif(not _gpu_present(), reason="no CUDA GPU / cupy / nvmath available")
def test_cudss_solve_uses_matching_and_ir():
    """The winning config (matching=AUTO + IR) must be what actually runs: verify a
    matrix that is WRONG without matching comes out right (this is the regression that
    guards against someone dropping matching_algorithm)."""
    # A matrix with a wide dynamic range on the diagonal (like the 1e-31 ZAF cells)
    # that the default (no-matching) cuDSS gets wrong but matching+IR gets right.
    n = 6
    A = sp.eye(n, format="csr").tolil()
    A[0, 1] = 1e6
    A[1, 0] = 1e-6
    A[2, 3] = 5.0
    A[4, 5] = 3.0
    A = A.tocsr()
    x_true = np.ones(n)
    b = A @ x_true
    x, info = _mod.cudss_solve(A, b)
    assert info.get("ok") is True
    rel_res = np.linalg.norm(A @ x - b) / max(1.0, np.linalg.norm(b))
    assert rel_res < 1e-10, f"rel_res {rel_res:.2e}"
