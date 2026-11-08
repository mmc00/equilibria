"""Sparse Newton with a dogleg trust region, shared by the Pyomo and the family paths.

Extracted verbatim from the inline loop of ``path_capi`` (``scipy_newton_tr``): the
exact Jacobian is re-evaluated every step, the Newton direction comes from one linear
backend (``mumps`` | ``direct`` | ``cudss`` | ``gmres`` | ``spilu``) whose factor is
reused for ``EQUILIBRIA_GTAP_TR_LU_REFRESH`` steps (Shamanskii), and a step is accepted
only if it lowers ``0.5 * ||F||^2``. The caller supplies callables, so nothing here
knows about Pyomo: ``F(x)`` and ``J(x)`` evaluate the residual and the sparse Jacobian
at ``x`` (the system state follows ``x``). The environment options
(``EQUILIBRIA_GTAP_TR_*``, ``EQUILIBRIA_GTAP_MUMPS_*``, ``EQUILIBRIA_GTAP_GMIN``) keep
their meaning; ``linsolve`` and ``icntl6``, when given, win over their variables.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np


@dataclass
class NewtonResult:
    x: np.ndarray
    success: bool
    nit: int
    nfact: int
    # (||F||_inf, ||F/fscale||_inf or None) at the top of each iteration
    trace: list = field(default_factory=list)
    null_pivots: list = field(default_factory=list)  # MUMPS INFOG(28) per factor
    cfg: dict = field(default_factory=dict)  # GMIN / ICNTL actually used
    linsolve: str = "direct"
    gmin_factors: int = 0  # numeric factorizations that ran with GMIN on the diagonal


def _nm(names: Sequence | None, i: int) -> str:
    return str(names[i]) if names is not None else f"#{i}"


def newton_tr(
    F: Callable[[np.ndarray], np.ndarray],
    J: Callable[[np.ndarray], Any],
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    colperm: np.ndarray | None = None,
    paired_colperm: Callable[[], tuple[np.ndarray | None, int]] | None = None,
    fscale: np.ndarray | None = None,
    reltol: float | None = None,
    icntl6: int | None = None,
    row_names: Sequence | None = None,
    col_names: Sequence | None = None,
    col_is_fixed: Callable[[int], bool] | None = None,
    linsolve: str | None = None,
    maxit: int = 300,
    f0: np.ndarray | None = None,
    on_symbolic_factorization: Callable[[], None] | None = None,
    gmin: float | None = None,
    gmin_policy: str = "always",
    mumps_values_at_analysis: bool = False,
) -> NewtonResult:
    """Solve ``F(x) = 0`` from ``x0`` inside ``[lb, ub]``.

    ``colperm``: column permutation giving the Jacobian a zero-free diagonal. When
    ``None``, ``paired_colperm() -> (perm | None, matched)`` is tried at the first
    factorization (the Pyomo path's GAMS pairing), then a numeric bipartite matching.
    ``fscale``/``reltol``: converge on ``||F/fscale||_inf < reltol`` (``reltol=None``
    reads ``EQUILIBRIA_GTAP_TR_RELTOL``; ignored without ``fscale``). ``f0``: ``F`` at
    the clipped ``x0`` when the caller already has it. ``row_names``/``col_names``/
    ``col_is_fixed`` only feed diagnostics and the structural gate.
    ``gmin``: diagonal regularization of J·P (``None`` reads ``EQUILIBRIA_GTAP_GMIN``).
    ``gmin_policy="always"`` (default) adds it to every factorization;
    ``"on_null_pivot"`` factors J·P as is and, only when MUMPS reports null pivots
    (INFOG(28) > 0), refactors that same matrix with ``gmin`` (counted in
    ``NewtonResult.gmin_factors``).
    ``mumps_values_at_analysis``: hand MUMPS the matrix values at the analysis (``job=1``)
    so a value-based ICNTL(6) (e.g. 7) can use them; the default analyses the pattern
    alone, as Pyomo's MUMPS wrapper does.
    """
    if gmin_policy not in ("always", "on_null_pivot"):
        raise ValueError(f"unknown gmin_policy {gmin_policy!r}")
    _on_null_pivot = gmin_policy == "on_null_pivot"
    _gmin_cfg = (
        float(gmin)
        if gmin is not None
        else float(os.environ.get("EQUILIBRIA_GTAP_GMIN", "0") or 0.0)
    )
    _lu_gmin_pattern = False  # the live MUMPS analysis was done on the GMIN pattern
    _tr_gmin_factors = 0
    import time as _time

    n = int(np.asarray(x0).size)
    _t0_sr = _time.perf_counter()
    _fscale_tr = fscale

    def _paired_colperm():
        return paired_colperm() if paired_colperm is not None else (None, 0)

    from scipy.sparse import eye as _eye_tr  # noqa: F401
    from scipy.sparse.linalg import spilu as _spilu_tr

    _maxit_tr = int(maxit)
    _ftol_tr = float(os.environ.get("EQUILIBRIA_GTAP_TR_FTOL", "1e-7"))
    _delta_tr = float(os.environ.get("EQUILIBRIA_GTAP_TR_DELTA0", "1.0"))
    _dmax_tr = float(os.environ.get("EQUILIBRIA_GTAP_TR_DELTAMAX", "1e3"))
    _drop_tr = float(os.environ.get("EQUILIBRIA_GTAP_TR_DROPTOL", "1e-5"))
    _ff_tr = float(os.environ.get("EQUILIBRIA_GTAP_TR_FILLFACTOR", "40"))
    # spilu factorization is the per-iteration cost (MEASURED 6.6s on 15x10,
    # much more on 20x41). Reuse it across REFRESH iterations (Shamanskii/chord
    # Newton): the exact J is re-evaluated every step (cheap, 3ms) so the dogleg
    # and ρ-gating stay accurate, but the spilu LU (only an approximate J⁻¹ for
    # the Newton direction) is refactored only every REFRESH steps OR whenever a
    # step is REJECTED (ρ≤0 → the stale LU is misleading the direction). 0/1 =
    # refactor every iteration (old behaviour).
    _refresh_tr = int(os.environ.get("EQUILIBRIA_GTAP_TR_LU_REFRESH", "5"))
    # inner linear solve for the Newton step:
    #  "direct" (DEFAULT) — zero-free-diagonal column permutation + COMPLETE splu
    #     (exact). ~560x faster than naive spilu on the zero-diagonal GTAP Jacobian
    #     (MEASURED 15x10: 439s→0.78s). EXACT but scipy's LU needs ~35GB at 393k →
    #     OOM on a 32GB kernel; use "mumps" for 20x41.
    #  "mumps" — MUMPS sparse-direct (Fortran multifrontal, own pivoting, EXACT,
    #     RAM-bounded out-of-Python). The CGE-scale route (IPOPT/GEMPACK use it
    #     internally). Symbolic factor once + Shamanskii numeric refresh. USE FOR 20x41.
    #  "spilu" — incomplete LU, NO permutation (legacy; catastrophic fill here).
    #  "gmres" — matrix-free (stalls without a real preconditioner on this system).
    from scipy.sparse.linalg import splu as _splu_full_tr

    _linsolve_tr = (
        linsolve or os.environ.get("EQUILIBRIA_GTAP_TR_LINSOLVE", "direct")
    ).lower()
    _colperm_tr = colperm  # structural column permutation (computed once, reused)
    _gmres_tol_tr = float(os.environ.get("EQUILIBRIA_GTAP_TR_GMRES_TOL", "1e-3"))
    _gmres_restart_tr = int(os.environ.get("EQUILIBRIA_GTAP_TR_GMRES_RESTART", "50"))
    _gmres_maxit_tr = int(os.environ.get("EQUILIBRIA_GTAP_TR_GMRES_MAXIT", "200"))
    _x_tr = np.clip(np.array(x0, dtype=float), lb, ub)
    _F_tr = f0 if f0 is not None else F(_x_tr)
    _phi_tr = 0.5 * float(_F_tr @ _F_tr)  # merit = ½‖F‖²
    _reltol_tr = (
        float(os.environ.get("EQUILIBRIA_GTAP_TR_RELTOL", "0") or 0.0)
        if reltol is None
        else float(reltol)
    )
    # STRUCTURAL PRE-SOLVE GATE (on by default; EQUILIBRIA_GTAP_TR_GATE=0 to skip).
    # A zero Jacobian row (empty equation) or zero column (variable in no equation)
    # makes the squared system STRUCTURALLY SINGULAR — no direct/Krylov solver can
    # touch it and a naive splu burns ~30min of fill before giving up (MEASURED:
    # 20x41 has 486 empty rows from zero-cell tuples the disaggregation exposes).
    # The matching costs ~0.01s even at 393k, so verify up front and ABORT with the
    # offending equation NAMES instead of hanging. Same spirit as the other hard
    # gates: let the system tell you, don't discover it by hand three weeks later.
    if os.environ.get("EQUILIBRIA_GTAP_TR_GATE", "1") == "1":
        _Jg0 = J(_x_tr).tocsr()
        _rpr_g = np.diff(_Jg0.indptr)
        _emptyr_g = np.where(_rpr_g == 0)[0]
        _Jg0c = _Jg0.tocsc()
        _emptyc_all_g = np.where(np.diff(_Jg0c.indptr) == 0)[0]
        # An empty COLUMN whose variable is FIXED is square-neutral, not a
        # singularity: a fixed var carries no DOF, so it can be absent from every
        # equation harmlessly (this is exactly the paired var=0 + eq-Skip pattern).
        # PyomoNLP keeps fixed variables as Jacobian columns, so filter them out;
        # only a FREE variable in no equation is a real structural problem.

        _emptyc_g = np.array(
            [
                _j
                for _j in _emptyc_all_g
                if not (col_is_fixed is not None and col_is_fixed(_j))
            ],
            dtype=int,
        )
        if len(_emptyr_g) or len(_emptyc_g):
            _bad_eqs = [_nm(row_names, _i) for _i in _emptyr_g[:15]]
            _bad_vars = [_nm(col_names, _i) for _i in _emptyc_g[:15]]
            _msg_g = (
                f"STRUCTURAL SINGULARITY: {len(_emptyr_g)} empty Jacobian rows "
                f"(zero-derivative equations) and {len(_emptyc_g)} empty columns "
                f"(variables in no equation). The squared system is not solvable. "
                f"First offending equations: {_bad_eqs}. "
                f"First offending variables: {_bad_vars}. "
                f"Fix the squaring (filter zero-cell tuples like GAMS $-conditions, "
                f"or fix the associated variable) — set EQUILIBRIA_GTAP_TR_GATE=0 "
                f"to bypass this check."
            )
            print(f"[nlp-square] {_msg_g}", file=sys.stderr, flush=True)
            raise RuntimeError(_msg_g)

    _conv_tr = False
    _k_tr = 0
    _lu_tr: Any = None
    _lu_age_tr = 0  # iterations since last refactor
    _cudss_solver_tr = None  # persists cuDSS plan across Newton steps (reuse)
    _gmin_pattern_sig = None  # (nnz, hash) of the last gmin-applied pattern
    # Bookkeeping for the opt-in reference capture (prep_capture.capture_solve);
    # plain appends/counters, they never feed back into the iteration.
    _tr_trace: list = []  # (‖F‖∞, ‖F/fscale‖∞ or None) at each loop top
    _tr_nfact = 0  # numeric factorizations
    _tr_nullpiv: list = []  # MUMPS INFOG(28) after each numeric factor
    _tr_cfg: dict = {}  # GMIN / ICNTL actually used
    for _k_tr in range(_maxit_tr):
        _rinf_tr = float(np.linalg.norm(_F_tr, np.inf))
        _tr_trace.append(
            (
                _rinf_tr,
                None
                if _fscale_tr is None
                else float(np.linalg.norm(_F_tr / _fscale_tr, np.inf)),
            )
        )
        # absolute-tol convergence (default) OR relative-tol (TR_RELTOL): the
        # per-equation scaled residual ‖F/fscale‖_inf, so huge-flow cells with a
        # small RELATIVE residual don't block convergence (see fscale note above).
        if _fscale_tr is not None:
            _rrel_tr = float(np.linalg.norm(_F_tr / _fscale_tr, np.inf))
            if _rrel_tr < _reltol_tr:
                _conv_tr = True
                if _k_tr % 10 == 0 or _rrel_tr < _reltol_tr:
                    print(
                        f"[nlp-square] CONVERGED (relative): "
                        f"‖F/scale‖_inf={_rrel_tr:.2e} < {_reltol_tr:.1e} "
                        f"(‖F‖_inf abs={_rinf_tr:.2e})",
                        file=sys.stderr,
                        flush=True,
                    )
                break
        if _rinf_tr < _ftol_tr:
            _conv_tr = True
            break
        # WORST-RESIDUAL NAMER (env EQUILIBRIA_GTAP_NAME_WORST_RESID=1): which
        # equations hold the largest |F|? Tells whether the stuck residual lives
        # in the GMIN-touched degenerate rows (→ ε is the lever) or in normal rows
        # (→ it's the step/TR). Report the top-8 by |F| every 10 iters.
        if (
            os.environ.get("EQUILIBRIA_GTAP_NAME_WORST_RESID") == "1"
            and _k_tr % 10 == 0
        ):
            try:
                _cons_wr = row_names if row_names is not None else ()

                _absF = np.abs(_F_tr)
                _topk = np.argsort(_absF)[::-1][:8]
                _wr = "; ".join(
                    f"{str(_cons_wr[int(_i)])}={_absF[int(_i)]:.2e}" for _i in _topk
                )
                print(
                    f"[nlp-square] WORST-RESID(abs) it={_k_tr}: {_wr}",
                    file=sys.stderr,
                    flush=True,
                )
                # RELATIVE worst — the cell that actually blocks rel-tol
                # convergence (|F|/fscale). This is the one that matters once
                # abs residual is dominated by huge-flow-but-solved cells.
                if _fscale_tr is not None:
                    _relF = _absF / _fscale_tr
                    _topr = np.argsort(_relF)[::-1][:8]
                    _wrr = "; ".join(
                        f"{str(_cons_wr[int(_i)])}={_relF[int(_i)]:.2e}"
                        f"(abs={_absF[int(_i)]:.1e})"
                        for _i in _topr
                    )
                    print(
                        f"[nlp-square] WORST-RESID(rel) it={_k_tr}: {_wrr}",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as _wre:
                print(
                    f"[nlp-square] worst-resid namer failed: {_wre}",
                    file=sys.stderr,
                    flush=True,
                )
        _J_tr = J(_x_tr).tocsc()
        _g_tr = _J_tr.T @ _F_tr  # gradient of merit = JᵀF
        # NEWTON STEP: solve J·pN = -F. Two backends:
        #  spilu (default): factor J once, reuse (Shamanskii). FAST per-solve but
        #    the factorization DOESN'T SCALE — MEASURED ~30min/iter on 20x41 (393k).
        #  gmres (EQUILIBRIA_GTAP_TR_LINSOLVE=gmres): matrix-free — solve J·pN=-F by
        #    GMRES using ONLY exact sparse J·v products (J @ v, ~ms), NEVER factoring
        #    the 393k system. Jacobi (diagonal) preconditioner is cheap and helps.
        #    This is the JFNK inner solve BUT wrapped in the robust trust region and
        #    with the EXACT J (no finite-diff "zero vector"). The lever that makes
        #    20x41 tractable: kills the 30min factorization entirely.
        if _linsolve_tr == "gmres":
            from scipy.sparse.linalg import LinearOperator as _LO_real
            from scipy.sparse.linalg import (
                gmres as _gmres_tr,
            )

            _LO_tr = cast(
                Any, _LO_real
            )  # scipy's __new__ signature confuses the checker

            _diag_tr = _J_tr.diagonal()
            _diag_tr = np.where(np.abs(_diag_tr) > 1e-12, _diag_tr, 1.0)
            _Mdiag_tr = _LO_tr(
                (n, n),
                matvec=lambda v: v / _diag_tr,
                dtype=_x_tr.dtype,
            )
            _Jop_tr = _LO_tr(
                (n, n),
                matvec=lambda v: _J_tr @ v,
                dtype=_x_tr.dtype,
            )
            _pN_tr, _info_tr = _gmres_tr(
                _Jop_tr,
                -_F_tr,
                M=_Mdiag_tr,
                rtol=_gmres_tol_tr,
                atol=0.0,
                restart=_gmres_restart_tr,
                maxiter=_gmres_maxit_tr,
            )
            if not np.all(np.isfinite(_pN_tr)):
                _pN_tr = -_g_tr
        elif _linsolve_tr == "mumps":
            # MUMPS SPARSE DIRECT (the CGE-scale route: the solver IPOPT/GEMPACK
            # use internally). Multifrontal Fortran; unlike scipy's COMPLETE splu
            # (~35GB → OOM on the 32GB kernel at 393k) it keeps the factor
            # out-of-Python and RAM-bounded. EXACT (no drop-tol).
            #
            # CRITICAL: the squared GTAP Jacobian has a ~100% ZERO diagonal
            # (structural: eq-i vs var-i by name). Handed the raw matrix, MUMPS's
            # analysis declares it SINGULAR IN STRUCTURE (INFOG(1)=-6) even though
            # the transversal is full. Fix (same lesson as `direct`): column-
            # permute to a zero-free diagonal via maximum_bipartite_matching FIRST,
            # solve (J P) y = -F, then un-permute p[colperm]=y. Plus ICNTL(6)=5
            # (MUMPS's own scaling + zero-free-diagonal column permutation) as a
            # belt-and-suspenders. The permutation is STRUCTURAL (pattern-only,
            # unchanged across Newton steps) so it is computed ONCE and reused;
            # symbolic factor once + Shamanskii numeric refresh.
            from pyomo.contrib.pynumero.linalg.mumps_interface import (
                MumpsCentralizedAssembledLinearSolver as _Mumps_tr,
            )
            from scipy.sparse.csgraph import (
                maximum_bipartite_matching as _mbm_m,
            )

            class _MumpsValuesAtAnalysis(_Mumps_tr):
                """Pyomo's wrapper with the values set before the analysis."""

                def do_symbolic_factorization(self, matrix, raise_on_error=True):
                    coo = matrix.tocoo()
                    ctx = cast(Any, self._mumps)
                    set_rows_cols = ctx.set_centralized_assembled_rows_cols

                    def rows_cols_then_values(irn, jcn):
                        set_rows_cols(irn, jcn)
                        ctx.set_centralized_assembled_values(coo.data)

                    ctx.set_centralized_assembled_rows_cols = rows_cols_then_values
                    try:
                        return super().do_symbolic_factorization(coo, raise_on_error)
                    finally:
                        del ctx.set_centralized_assembled_rows_cols

            _pf_m = os.environ.get("EQUILIBRIA_GTAP_PROGRESS_FILE")

            def _plog_m(_m):
                if _pf_m:
                    try:
                        with open(_pf_m, "a") as _fh:
                            _fh.write(f"  [mumps] {_m}\n")
                    except Exception:
                        pass
                print(
                    f"[nlp-square] [mumps] {_m}",
                    file=sys.stderr,
                    flush=True,
                )

            # structural column permutation → zero-free diagonal (compute once)
            if _colperm_tr is None:
                _t_m = __import__("time").perf_counter()
                # PREFER the GAMS pairing (structural_matching), which resolves
                # the MCP forced pairs the numeric matching cannot (see
                # _paired_colperm above). Fall back to numeric matching, then
                # identity, only if the pairing is unavailable/incomplete.
                _pp_m, _pm_matched = _paired_colperm()
                if _pp_m is not None and _pm_matched >= n:
                    _colperm_tr = _pp_m
                    _plog_m(
                        "paired colperm (GAMS structural_matching) "
                        f"{_pm_matched}/{n} in "
                        f"{__import__('time').perf_counter() - _t_m:.1f}s"
                    )
                else:
                    _colperm_tr = _mbm_m(_J_tr.tocsr(), perm_type="column")
                    if np.any(_colperm_tr < 0):
                        if _pp_m is not None:
                            _colperm_tr = _pp_m
                            _plog_m(
                                "numeric matching NOT full — using PAIRED "
                                f"colperm ({_pm_matched}/{n} matched)"
                            )
                        else:
                            _plog_m(
                                "matching NOT full — using identity (rank-deficient)"
                            )
                            _colperm_tr = np.arange(n)
                    else:
                        _plog_m(
                            "col matching done in "
                            f"{__import__('time').perf_counter() - _t_m:.1f}s "
                            f"(n={n})"
                        )
            _need_lu = (
                _lu_tr is None
                or _refresh_tr <= 1
                or _lu_age_tr >= _refresh_tr
                or _lu_age_tr < 0
            )
            _Jm_tr = _J_tr[:, _colperm_tr].tocsr()
            # GMIN REGULARIZATION (SPICE's floating-node fix). The degenerate CET
            # cells (omegax=inf zero-share) leave xds/xet/... indeterminate → a
            # null pivot on those rows of J·P, exactly like a floating node in a
            # circuit. Circuit simulators add a tiny conductance to ground (GMIN)
            # to every node: (J·P + ε·I). ε is small enough not to move the root
            # (like GMIN→0) but removes the singularity so the direct factorization
            # completes and the Newton step is well-defined on those DOF. This is
            # the STANDARD floating-node remedy, applied globally — no per-family
            # patching. Env EQUILIBRIA_GTAP_GMIN (default 0 = off).
            _Jm_base = _Jm_tr
            for _attempt in (0, 1):
                _Jm_tr = _Jm_base
                if _attempt == 0:
                    _gmin = 0.0 if _on_null_pivot else _gmin_cfg
                    if _on_null_pivot and _lu_gmin_pattern and _need_lu:
                        _lu_tr = None  # last analysis was on the GMIN pattern
                        _lu_gmin_pattern = False
                else:
                    _gmin = _gmin_cfg  # retry: refactor this same matrix with GMIN
                    _lu_tr = None  # the pattern gains the diagonal: re-analyze
                    _need_lu = True
                    _lu_gmin_pattern = True
                _factored_now = False
                if _gmin > 0.0:
                    # Add ε to EVERY diagonal, materialized as an explicit stored entry
                    # (setdiag on lil forces the slot to exist even where J·P had a 0).
                    # This keeps the nonzero PATTERN fixed across Newton steps — MUMPS
                    # analyzes the symbol once and reuses it, and aborts "nonzeros
                    # changed between symbolic and numeric" if the degenerate diagonals
                    # flip in/out (measured n9). Also FORCE re-symbolization each factor
                    # when GMIN is on (below) so the analyzed pattern always matches.
                    _Jm_lil = _Jm_tr.tolil()
                    _Jm_lil.setdiag(_Jm_lil.diagonal() + _gmin)
                    _Jm_tr = _Jm_lil.tocsr()
                    # Symbolic reuse (lever B2): the symbolic factorization depends only
                    # on the PATTERN (nnz + indices/indptr), not the values. Under GMIN
                    # the ~30 degenerate diagonals CAN change the stored-nnz between
                    # steps, so we cannot reuse blindly — but the spike showed the
                    # pattern changes RARELY (3 distinct patterns in 28 steps), so
                    # re-analyzing every step is mostly wasted. Detect a pattern change
                    # by signature and force re-analysis ONLY on a change; otherwise let
                    # the existing _need_lu/_lu_age_tr reuse machinery run. Reusing only
                    # on a byte-identical pattern makes the MUMPS "nonzeros changed"
                    # abort impossible. EQUILIBRIA_GTAP_GMIN_SYM_REUSE=0 restores the
                    # old unconditional re-analyze.
                    if os.environ.get("EQUILIBRIA_GTAP_GMIN_SYM_REUSE", "1") != "0":
                        import hashlib as _hl_sr

                        _sig = (
                            _Jm_tr.nnz,
                            _hl_sr.sha256(
                                _Jm_tr.indices.tobytes() + _Jm_tr.indptr.tobytes()
                            ).hexdigest(),
                        )
                        if _sig != _gmin_pattern_sig:
                            _lu_tr = None  # pattern changed → re-analyze
                            _gmin_pattern_sig = _sig
                        # else: pattern unchanged → keep _lu_tr, reuse the symbolic
                    else:
                        _lu_tr = None  # opt-out: old unconditional re-analyze
                # GMIN on keeps the cadence of one factorization per step; the policy
                # only moves the regularization, not when J·P is refactored.
                if _need_lu or _gmin_cfg > 0.0:
                    try:
                        _t_m = __import__("time").perf_counter()
                        if _lu_tr is None:
                            # sym=0 (unsymmetric). ICNTL(14): working-space %-increase
                            # (avoid fill-underestimate failures at 393k).
                            # ICNTL(6): column permutation / scaling strategy. We ALREADY
                            # give MUMPS a zero-free-diagonal matrix (_Jm_tr = J·P with P
                            # from the GAMS pairing). ICNTL(6)=5 makes MUMPS compute ITS
                            # OWN max-transversal permutation ON TOP, which discards our
                            # (correct) P and re-introduced the structural singularity →
                            # error -6 even after the paired colperm made the diagonal
                            # full (measured: kernel n2, "paired colperm 391839/391839"
                            # then "MUMPS error -6"). With our own P applied, default to
                            # ICNTL(6)=0 (no MUMPS permutation). Overridable to A/B other
                            # strategies (0/7) via EQUILIBRIA_GTAP_MUMPS_ICNTL6.
                            _icntl6 = (
                                int(icntl6)
                                if icntl6 is not None
                                else int(
                                    os.environ.get("EQUILIBRIA_GTAP_MUMPS_ICNTL6", "0")
                                )
                            )
                            # ICNTL(7): fill-reducing ordering. Default PORD SEGFAULTED
                            # in libpord (findIndMultisecs) on the 393k GTAP pattern
                            # (measured n3). METIS (5) is robust at this scale — switch
                            # away from PORD. ICNTL(8): row/col SCALING. =8 is MUMPS's
                            # iterative row/col equilibration (Ruiz-like) — the numeric
                            # fix for error -10 (numerically singular pivot 391545 = the
                            # ill-conditioned eq_pfeq[Land] row, coef ~680 vs xft~2e-4).
                            # ICNTL(24)=1: detect+handle null pivots instead of aborting
                            # -10; CNTL(3) sets the null-pivot threshold (small relative).
                            # Default AMD (0): universally available (unlike METIS/5,
                            # which needs MUMPS compiled --with-metis — not guaranteed
                            # on Kaggle's libmumps) and NOT PORD, so it avoids the
                            # libpord segfault. Override to 5 (METIS) if the build has it.
                            _icntl7 = int(
                                os.environ.get("EQUILIBRIA_GTAP_MUMPS_ICNTL7", "0")
                            )  # 0=AMD (safe, avoids PORD segfault)
                            _icntl8 = int(
                                os.environ.get("EQUILIBRIA_GTAP_MUMPS_ICNTL8", "8")
                            )  # 8=iterative row/col scaling (MUMPS Ruiz)
                            _icntl24 = int(
                                os.environ.get("EQUILIBRIA_GTAP_MUMPS_ICNTL24", "1")
                            )  # 1=null-pivot detection
                            _tr_cfg.update(
                                icntl={
                                    "6": _icntl6,
                                    "7": _icntl7,
                                    "8": _icntl8,
                                    "14": 100,
                                    "24": _icntl24,
                                }
                            )
                            _lu_tr = (
                                _MumpsValuesAtAnalysis(sym=0)
                                if mumps_values_at_analysis
                                else _Mumps_tr(sym=0)
                            )
                            _lu_tr.set_icntl(14, 100)
                            _lu_tr.set_icntl(6, _icntl6)
                            _lu_tr.set_icntl(7, _icntl7)
                            _lu_tr.set_icntl(8, _icntl8)
                            _lu_tr.set_icntl(24, _icntl24)
                            # CNTL(3): null-pivot threshold (used when ICNTL(24)=1).
                            # Small relative value so only genuinely ~0 pivots are
                            # treated as null (not the merely small-but-valid ones).
                            if _icntl24:
                                try:
                                    _lu_tr.set_cntl(
                                        3,
                                        float(
                                            os.environ.get(
                                                "EQUILIBRIA_GTAP_MUMPS_CNTL3",
                                                "-1e-8",
                                            )
                                        ),
                                    )
                                except Exception:
                                    pass
                            _plog_m(
                                f"ICNTL 6={_icntl6}(perm) 7={_icntl7}(order) "
                                f"8={_icntl8}(scale) 24={_icntl24}(nullpiv)"
                            )
                            if on_symbolic_factorization is not None:
                                on_symbolic_factorization()
                            _lu_tr.do_symbolic_factorization(_Jm_tr)
                            _plog_m(
                                "symbolic factorization done in "
                                f"{__import__('time').perf_counter() - _t_m:.1f}s "
                                f"(n={n})"
                            )
                            _t_m = __import__("time").perf_counter()
                        _lu_tr.do_numeric_factorization(_Jm_tr)
                        _tr_nfact += 1
                        _factored_now = True
                        if _gmin > 0.0:
                            _tr_gmin_factors += 1
                        _tr_cfg["gmin"] = _gmin_cfg
                        _tr_cfg["gmin_policy"] = gmin_policy
                        # INFOG(28) = number of null pivots detected (ICNTL(24)=1). A
                        # handful is fine (the TR rejects a bad step); many means the
                        # scaling didn't rescue the ill-conditioned rows.
                        try:
                            _nnull = _lu_tr.get_infog(28)
                            _tr_nullpiv.append(int(_nnull or 0))
                            if _nnull:
                                _plog_m(f"null pivots detected: {_nnull}")
                                # NAME the null-pivot rows (env EQUILIBRIA_GTAP_MUMPS_
                                # NAME_NULLPIV=1). MUMPS fills id.pivnul_list with the
                                # 1-based indices of the null pivots. In the permuted
                                # matrix _Jm_tr, position i corresponds to constraints[i]
                                # (rows are NOT permuted; only columns via _colperm_tr).
                                # So pivnul index → constraint name → equation FAMILY.
                                if (
                                    os.environ.get("EQUILIBRIA_GTAP_MUMPS_NAME_NULLPIV")
                                    == "1"
                                ):
                                    try:
                                        # ROBUST namer (no MUMPS ctypes internals — the
                                        # pivnul_list pointer isn't portably iterable
                                        # across PyMUMPS builds). A null pivot ≈ a row
                                        # whose diagonal entry in J·P is ~0 relative to
                                        # the row's magnitude. _Jm_tr = J[:, colperm], so
                                        # its diagonal IS the paired-pivot value for each
                                        # row. Rank-deficient rows = |diag| / rowmax tiny.
                                        _JmC = _Jm_tr.tocsr()
                                        _diagv = _JmC.diagonal()
                                        _absJ = abs(_JmC)
                                        _rowmax = np.asarray(
                                            _absJ.max(axis=1).todense()
                                        ).flatten()
                                        _rowmax[_rowmax <= 0] = 1.0
                                        _ratio = np.abs(_diagv) / _rowmax
                                        _bad = np.where(_ratio < 1e-10)[0]
                                        _cons_nlp2 = (
                                            row_names if row_names is not None else ()
                                        )
                                        _vars_nlp2 = (
                                            col_names if col_names is not None else ()
                                        )
                                        from collections import Counter as _Ctr

                                        _famc = _Ctr()
                                        _examples = []
                                        for _ri in _bad.tolist():
                                            if 0 <= _ri < len(_cons_nlp2):
                                                _cn = str(_cons_nlp2[_ri])
                                                _fam = _cn.split("[")[0]
                                                _famc[_fam] += 1
                                                _cj = (
                                                    int(_colperm_tr[_ri])
                                                    if _ri < len(_colperm_tr)
                                                    else -1
                                                )
                                                _vn = (
                                                    str(_vars_nlp2[_cj])
                                                    if 0 <= _cj < len(_vars_nlp2)
                                                    else "?"
                                                )
                                                if len(_examples) < 20:
                                                    _examples.append(f"{_cn}⊥{_vn}")
                                        _plog_m(
                                            f"NULLPIV rows(|diag|/rowmax<1e-10)="
                                            f"{len(_bad)} families: "
                                            + ", ".join(
                                                f"{_f}={_c}"
                                                for _f, _c in _famc.most_common()
                                            )
                                        )
                                        _plog_m(
                                            "NULLPIV examples: " + "; ".join(_examples)
                                        )
                                    except Exception as _ne:
                                        _plog_m(f"nullpiv naming failed: {_ne}")
                        except Exception:
                            pass
                        _plog_m(
                            "numeric factorization done in "
                            f"{__import__('time').perf_counter() - _t_m:.1f}s"
                        )
                        _lu_age_tr = 0
                    except Exception as _e_m:
                        # Log INFOG(1)/INFOG(2) so the failure is diagnosable from the
                        # log instead of guessed: -6/-10 = structurally/numerically
                        # singular (INFOG(2)=offending pivot), -9/-8 = workspace too
                        # small (bump ICNTL(14)), etc.
                        _infog = ""
                        try:
                            _lu_any = cast(Any, _lu_tr)
                            _infog = (
                                f" INFOG(1)={_lu_any.get_infog(1)} "
                                f"INFOG(2)={_lu_any.get_infog(2)}"
                            )
                        except Exception:
                            pass
                        _plog_m(f"MUMPS FAILED: {_e_m}{_infog}")
                        _lu_tr = None
                else:
                    _lu_age_tr += 1
                if (
                    _attempt == 0
                    and _on_null_pivot
                    and _gmin_cfg > 0.0
                    and _factored_now
                    and _tr_nullpiv
                    and _tr_nullpiv[-1] > 0
                ):
                    continue
                break
            try:
                if _lu_tr is not None:
                    _y_m, _ = _lu_tr.do_back_solve(-_F_tr)  # (J P) y = -F
                    if _y_m is None or not np.all(np.isfinite(_y_m)):
                        _pN_tr = -_g_tr
                    else:
                        _pN_tr = np.empty(n)
                        _pN_tr[_colperm_tr] = _y_m  # un-permute: p = P y
                else:
                    _pN_tr = -_g_tr
            except Exception:
                _pN_tr = -_g_tr
        elif _linsolve_tr == "cudss":
            # cuDSS GPU DIRECT (NVIDIA multifrontal). Same operator as the mumps
            # route — the colperm-paired, GMIN-applied J·P (zero-free diagonal) —
            # solved on the GPU. Proven on the real 20x41 (kernel cudss-config v11):
            # rel_res 5.25e-16 (BEATS MUMPS 1.37e-13) at 3.82s vs 50.6s = 13.3x. The
            # lever is matching_algorithm=AUTO (MUMPS ICNTL(6) analogue) + iterative
            # refinement (ir_num_steps=2), both applied inside cudss_solve. OPT-IN;
            # any GPU/binding failure falls back cleanly to the gradient step (the TR
            # then rejects/retries), so a missing GPU never crashes the solve.
            import importlib.util as _ilu_c

            from scipy.sparse.csgraph import (
                maximum_bipartite_matching as _mbm_c,
            )

            _pf_c = os.environ.get("EQUILIBRIA_GTAP_PROGRESS_FILE")

            def _plog_c(_m):
                if _pf_c:
                    try:
                        with open(_pf_c, "a") as _fh:
                            _fh.write(f"  [cudss] {_m}\n")
                    except Exception:
                        pass
                print(
                    f"[nlp-square] [cudss] {_m}",
                    file=sys.stderr,
                    flush=True,
                )

            # structural column permutation → zero-free diagonal (compute once),
            # identical policy to the mumps branch (GAMS pairing, then numeric).
            if _colperm_tr is None:
                _pp_c, _pc_matched = _paired_colperm()
                if _pp_c is not None and _pc_matched >= n:
                    _colperm_tr = _pp_c
                    _plog_c(
                        f"paired colperm (GAMS structural_matching) {_pc_matched}/{n}"
                    )
                else:
                    _colperm_tr = _mbm_c(_J_tr.tocsr(), perm_type="column")
                    if np.any(_colperm_tr < 0):
                        if _pp_c is not None:
                            _colperm_tr = _pp_c
                        else:
                            _colperm_tr = np.arange(n)

            _Jm_tr = _J_tr[:, _colperm_tr].tocsr()
            _gmin_c = float(os.environ.get("EQUILIBRIA_GTAP_GMIN", "0") or 0.0)
            if _gmin_c > 0.0:
                _Jm_lil_c = _Jm_tr.tolil()
                _Jm_lil_c.setdiag(_Jm_lil_c.diagonal() + _gmin_c)
                _Jm_tr = _Jm_lil_c.tocsr()

            # load the helper by path (scripts/gtap is not a package) and use the
            # PERSISTENT reusable solver: it caches the cuDSS plan (symbolic analysis)
            # across Newton steps and only re-factorizes when the values change (or
            # re-plans if the pattern changes) — the symbolic-reuse win (lever B2's
            # analogue) that turns the per-factor 13x into an end-to-end gain.
            try:
                _cud_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "_cudss_linsolve.py",
                )
                _cud_spec = _ilu_c.spec_from_file_location("_cudss_linsolve", _cud_path)
                assert _cud_spec is not None and _cud_spec.loader is not None
                _cud = _ilu_c.module_from_spec(_cud_spec)
                _cud_spec.loader.exec_module(_cud)
                if _cudss_solver_tr is None:
                    _cudss_solver_tr = _cud.CudssReusableSolver()
                _x_c, _info_c = _cudss_solver_tr.solve(_Jm_tr, -_F_tr)
            except Exception as _e_c:
                _x_c, _info_c = None, {"ok": False, "err": str(_e_c)}

            if _x_c is not None and _info_c.get("ok"):
                _nplans_c = getattr(_cudss_solver_tr, "n_plans", "?")
                _plog_c(
                    f"solve ok rel_res={_info_c.get('rel_res'):.2e} plans={_nplans_c}"
                )
                _pN_tr = np.empty(n)
                _pN_tr[_colperm_tr] = _x_c  # un-permute: p = P y
            else:
                _plog_c(
                    f"solve FAILED ({_info_c.get('err')}), "
                    "falling back to gradient step"
                )
                _pN_tr = -_g_tr
        elif _linsolve_tr == "direct":
            # DIRECT SOLVE with zero-free-diagonal permutation (GEMPACK/MA48/KLU
            # approach). The squared GTAP Jacobian has a ~100% ZERO diagonal
            # (structural: eq-i vs var-i by name), so a naive splu produces
            # catastrophic fill — MEASURED 15x10: raw splu factor=439s fill=1530x
            # (this is the ~30min/iter that killed 20x41). Permuting COLUMNS by a
            # maximum bipartite matching gives a zero-free diagonal → COMPLETE
            # (EXACT) splu on the permuted matrix: MEASURED 15x10 factor=0.78s
            # fill=37x — ~560x faster, and EXACT (no drop-tol approximation). The
            # column permutation is STRUCTURAL (depends only on the sparsity
            # pattern, unchanged across Newton steps) so it is computed ONCE and
            # reused; only the numeric refactor happens each step (Shamanskii).
            from scipy.sparse.csgraph import (
                maximum_bipartite_matching as _mbm_tr,
            )

            _pf_d = os.environ.get("EQUILIBRIA_GTAP_PROGRESS_FILE")

            def _plog_d(_m):
                if _pf_d:
                    try:
                        with open(_pf_d, "a") as _fh:
                            _fh.write(f"  [direct] {_m}\n")
                    except Exception:
                        pass
                print(
                    f"[nlp-square] [direct] {_m}",
                    file=sys.stderr,
                    flush=True,
                )

            if _colperm_tr is None:
                _t_d = __import__("time").perf_counter()
                # PREFER the GAMS pairing (resolves MCP forced pairs the numeric
                # matching misses); fall back to numeric matching, then identity.
                _pp_d, _pd_matched = _paired_colperm()
                if _pp_d is not None and _pd_matched >= n:
                    _colperm_tr = _pp_d
                    _plog_d(
                        "paired colperm (GAMS structural_matching) "
                        f"{_pd_matched}/{n} in "
                        f"{__import__('time').perf_counter() - _t_d:.1f}s"
                    )
                else:
                    _colperm_tr = _mbm_tr(_J_tr.tocsr(), perm_type="column")
                    _plog_d(
                        f"matching done in {__import__('time').perf_counter() - _t_d:.1f}s "
                        f"(n={n})"
                    )
                    if np.any(_colperm_tr < 0):
                        if _pp_d is not None:
                            _colperm_tr = _pp_d  # paired fallback
                            _plog_d(
                                "numeric matching NOT full — using PAIRED "
                                f"colperm ({_pd_matched}/{n} matched)"
                            )
                        else:
                            _colperm_tr = np.arange(n)  # rank-deficient fallback
            _need_lu = (
                _lu_tr is None
                or _refresh_tr <= 1
                or _lu_age_tr >= _refresh_tr
                or _lu_age_tr < 0
            )
            if _need_lu:
                try:
                    _t_d = __import__("time").perf_counter()
                    _Jperm_tr = _J_tr[:, _colperm_tr].tocsc()
                    _lu_tr = _splu_full_tr(_Jperm_tr, permc_spec="COLAMD")
                    _tr_nfact += 1
                    _plog_d(
                        f"splu factor done in "
                        f"{__import__('time').perf_counter() - _t_d:.1f}s "
                        f"nnz(LU)={_lu_tr.L.nnz + _lu_tr.U.nnz}"
                    )
                    _lu_age_tr = 0
                except Exception as _e_d:
                    _plog_d(f"splu FAILED: {_e_d}")
                    _lu_tr = None
            else:
                _lu_age_tr += 1
            try:
                if _lu_tr is not None:
                    _y_tr = _lu_tr.solve(-_F_tr)  # solves (J P) y = -F
                    _pN_tr = np.empty(n)
                    _pN_tr[_colperm_tr] = _y_tr  # un-permute: p = P y
                else:
                    _pN_tr = -_g_tr
            except Exception:
                _pN_tr = -_g_tr
        else:
            # (re)factor spilu only when stale (first step / every N / after reject)
            _need_lu = (
                _lu_tr is None
                or _refresh_tr <= 1
                or _lu_age_tr >= _refresh_tr
                or _lu_age_tr < 0
            )
            if _need_lu:
                try:
                    _lu_tr = _spilu_tr(
                        _J_tr + 1e-10 * _eye_tr(n, format="csc"),
                        drop_tol=_drop_tr,
                        fill_factor=_ff_tr,
                    )
                    _lu_age_tr = 0
                except Exception:
                    _lu_tr = None
            else:
                _lu_age_tr += 1
            try:
                _pN_tr = _lu_tr.solve(-_F_tr) if _lu_tr is not None else -_g_tr
            except Exception:
                _pN_tr = -_g_tr  # fall back to gradient if LU solve fails
        _Jg_tr = _J_tr @ _g_tr
        _gg_tr = float(_g_tr @ _g_tr)
        _Jg2_tr = float(_Jg_tr @ _Jg_tr)
        _tau_tr = (_gg_tr / _Jg2_tr) if _Jg2_tr > 0 else 0.0
        _pC_tr = -_tau_tr * _g_tr  # Cauchy (steepest-descent) point
        # dogleg: pick step within trust radius Δ blending Cauchy→Newton
        _nN_tr = float(np.linalg.norm(_pN_tr))
        _nC_tr = float(np.linalg.norm(_pC_tr))
        if _nN_tr <= _delta_tr:
            _p_tr = _pN_tr
        elif _nC_tr >= _delta_tr:
            _p_tr = (_delta_tr / _nC_tr) * _pC_tr if _nC_tr > 0 else _pC_tr
        else:
            # solve ‖pC + β(pN-pC)‖ = Δ for β∈[0,1]
            _d_tr = _pN_tr - _pC_tr
            _a_tr = float(_d_tr @ _d_tr)
            _b_tr = 2.0 * float(_pC_tr @ _d_tr)
            _c_tr = _nC_tr * _nC_tr - _delta_tr * _delta_tr
            _disc_tr = max(_b_tr * _b_tr - 4 * _a_tr * _c_tr, 0.0)
            _beta_tr = (-_b_tr + _disc_tr**0.5) / (2 * _a_tr) if _a_tr > 0 else 0.0
            _p_tr = _pC_tr + _beta_tr * _d_tr
        # predicted reduction from linear model: ½‖F‖² - ½‖F + Jp‖²
        _Jp_tr = _J_tr @ _p_tr
        _pred_tr = _phi_tr - 0.5 * float((_F_tr + _Jp_tr) @ (_F_tr + _Jp_tr))
        _x_new_tr = np.clip(_x_tr + _p_tr, lb, ub)
        try:
            _Fnew_tr = F(_x_new_tr)
            if not np.all(np.isfinite(_Fnew_tr)):
                raise ValueError("non-finite")
            _phinew_tr = 0.5 * float(_Fnew_tr @ _Fnew_tr)
        except Exception:
            _phinew_tr = np.inf
            _Fnew_tr = _F_tr
        _ared_tr = _phi_tr - _phinew_tr  # actual reduction
        _rho_tr = (_ared_tr / _pred_tr) if _pred_tr > 1e-300 else -1.0
        if _rho_tr > 0.0:  # accept
            _x_tr = _x_new_tr
            _F_tr = _Fnew_tr
            _phi_tr = _phinew_tr
        else:  # reject: the reused LU is misleading the direction → force refactor
            _lu_age_tr = -1
        if _rho_tr > 0.75:
            _delta_tr = min(2.0 * _delta_tr, _dmax_tr)  # expand
        elif _rho_tr < 0.25:
            _delta_tr = max(0.25 * _delta_tr, 1e-10)  # shrink
        if _k_tr % 5 == 0:
            _relstr = ""
            if _fscale_tr is not None:
                _relstr = f" ||F/scale||={float(np.linalg.norm(_F_tr / _fscale_tr, np.inf)):.3e}"
            print(
                f"[nlp-square] NEWTON-TR it={_k_tr} ||F||_inf={_rinf_tr:.3e}"
                f"{_relstr} ρ={_rho_tr:.2f} Δ={_delta_tr:.2e} LUage={_lu_age_tr}",
                file=sys.stderr,
                flush=True,
            )
            # live progress file for remote (Kaggle) runs — see driver hook
            _pf_tr = os.environ.get("EQUILIBRIA_GTAP_PROGRESS_FILE")
            if _pf_tr:
                try:
                    with open(_pf_tr, "a") as _pfh_tr:
                        _pfh_tr.write(
                            f"  NEWTON-TR it={_k_tr} ||F||_inf={_rinf_tr:.3e} "
                            f"ρ={_rho_tr:.2f} Δ={_delta_tr:.2e} "
                            f"wall={__import__('time').perf_counter() - _t0_sr:.0f}s\n"
                        )
                except Exception:
                    pass
    return NewtonResult(
        x=_x_tr,
        success=_conv_tr,
        nit=_k_tr + 1,
        nfact=_tr_nfact,
        trace=_tr_trace,
        null_pivots=_tr_nullpiv,
        cfg=_tr_cfg,
        linsolve=_linsolve_tr,
        gmin_factors=_tr_gmin_factors,
    )
