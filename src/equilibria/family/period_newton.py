"""One period's squared system and its Newton-TR solve on the family arrays.

``_period_system`` squares the staged arrays and builds the JAX system;
``_newton`` solves it with Newton-TR + MUMPS (``_NEWTON_TOL`` and ``_MUMPS_KEEP``
set around the call); ``RatioBorder`` borders it with world income for
``trade_balance="ratio"``. Moved here from the old family driver (PR 3d of
gtap-on-core): the core solver (``solver/newton_gtap``) uses them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import numpy as np

from equilibria.family.mp_arrays import MPArrays, Squared
from equilibria.family.phases import phase

if TYPE_CHECKING:  # run_setup imports ICNTL6_FAMILY from here
    from equilibria.family.run_setup import FamilyRun
ICNTL6_FAMILY = 7  # MUMPS ICNTL(6) on the family squared system (Pyomo keeps 0)
CODE_RESIDUAL = 1e-5  # code 1 needs ||F||_inf below this, as ``_ok_sr``


# (reltol, ftol) of the solve in progress; set by ``_solve_period_inner`` from the run
_NEWTON_TOL: ContextVar[tuple[float, float] | None] = ContextVar(
    "_NEWTON_TOL", default=None
)


# the run's ``mumps`` dict for the solve in progress; set with ``_NEWTON_TOL``
_MUMPS_KEEP: ContextVar[dict | None] = ContextVar("_MUMPS_KEEP", default=None)


def _newton(ss: Any, sq: Squared, border: Any = None) -> tuple[Any, float, int]:
    """Newton-TR on the squared system: ``(NewtonResult, ||F||_inf, code)``.

    ``border`` (a :class:`RatioBorder`): the system bordered with the world-income
    unknown and its row (``trade_balance="ratio"``); ``res.x`` then ends with ``W``.

    The settings are read from the environment as the Pyomo path reads them (the
    ``NEWTON_MUMPS`` set); the family pairing is the system's own column order, and
    MUMPS runs with ICNTL(6)=7. ``code`` is 1 iff Newton succeeded and the residual of
    the squared system is below ``CODE_RESIDUAL`` (``_ok_sr``), else 2 as the Pyomo path.
    ``_NEWTON_TOL`` (the run's ``newton_tol``, set around the call) replaces the
    environment's two stopping tolerances. ``_MUMPS_KEEP`` (the run's ``mumps``, set
    the same way): a MUMPS analysis kept from the previous solve is reused when the
    pattern is the same."""
    from equilibria.solver.newton_tr import newton_tr

    linsolve = os.environ.get("EQUILIBRIA_GTAP_TR_LINSOLVE", "direct").lower()
    lb = np.array([sq.cells[c][2] for c in sq.col_names], dtype=float)
    ub = np.array([sq.cells[c][3] for c in sq.col_names], dtype=float)
    F, x0 = ss.F, np.asarray(ss.x0, dtype=float)
    J = lambda z: ss.J(z)  # noqa: E731 - looked up at the first Jacobian
    row_names, col_names = list(sq.row_names), list(sq.col_names)
    if border is not None:
        lb, ub = np.append(lb, -np.inf), np.append(ub, np.inf)
        F, J, x0 = border.F, border.J, np.append(x0, border.w0)
        row_names.append(border.ROW)
        col_names.append(border.COL)
    lb[~np.isfinite(lb)] = -1e20  # as the Pyomo path
    ub[~np.isfinite(ub)] = 1e20
    n = len(x0)
    # the relative test needs RELTOL > 0, then fscale applies
    tol = _NEWTON_TOL.get()
    if tol is None:
        reltol, ftol = (
            float(os.environ.get("EQUILIBRIA_GTAP_TR_RELTOL", "0") or 0.0),
            None,
        )
    else:
        reltol, ftol = tol
    with phase("fscale"):
        fscale = ss.fscale() if reltol > 0.0 else None
    if fscale is not None and border is not None:
        fscale = np.append(fscale, border.fscale_row)

    def timed_F(z):
        with phase("newton_F"):
            return F(np.clip(z, lb, ub))

    def timed_J(z):
        with phase("newton_J"):
            return J(np.clip(z, lb, ub))

    with phase("newton_other"):  # analysis, factor, solve and the TR logic
        res = newton_tr(
            timed_F,
            timed_J,
            np.clip(x0, lb, ub),
            lb,
            ub,
            colperm=np.arange(n),  # row k is paired with column k
            fscale=fscale,
            reltol=reltol,
            ftol=ftol,
            icntl6=ICNTL6_FAMILY if linsolve == "mumps" else None,
            row_names=row_names,
            col_names=col_names,
            linsolve=linsolve,
            maxit=int(os.environ.get("EQUILIBRIA_GTAP_SCIPY_MAXITER", "300")),
            # GMIN lands on other entries under the family pairing than under Pyomo's
            # and moves weakly determined cells by ~1e-7: factor without it and
            # regularize only when MUMPS reports null pivots.
            gmin=float(os.environ.get("EQUILIBRIA_GTAP_GMIN", "0") or 0.0),
            gmin_policy="on_null_pivot",
            # analysed on the pattern alone (Pyomo's wrapper), ICNTL(6)=7 cannot use
            # the values and met a spurious null pivot on the 20x41 capFlex settle
            mumps_values_at_analysis=True,
            mumps_keep=_MUMPS_KEEP.get(),
        )
    with phase("residual_check"):
        resid = float(np.linalg.norm(F(np.clip(res.x, lb, ub)), np.inf))
    return res, resid, 1 if (bool(res.success) and resid < CODE_RESIDUAL) else 2


def _period_system(run: FamilyRun, mp1: MPArrays, period: str) -> tuple[Any, Any, Any]:
    """The square and system-build steps of ``_solve_period_inner`` on the recipe's
    arrays ``mp1``: ``(sq, mp_sq, ss)``."""
    from equilibria.family.cell_index import CellIndex
    from equilibria.family.prep_square import square
    from equilibria.family.squared_system import ArrayIncidence, SquaredSystem

    with phase("square"):
        if run.cell_index is None or not run.cell_index.fits(mp1):
            run.cell_index = CellIndex(mp1)
        if period not in run.adjacency:
            run.adjacency[period] = ArrayIncidence(
                run.backend, run.state, period, run.cross, store=run.records_dir
            )
        sq, mp_sq = square(
            mp1,
            period,
            params=run.params,
            closure=run.closure,
            incidence=run.adjacency[period],
            index=run.cell_index,
            exogenous=run.ref.exogenous,
            masks=run.masks,
            return_mp=True,
        )
    with phase("system_build"):
        ss = SquaredSystem.build(
            run.backend, run.state, mp_sq, period, sq, cross=run.cross
        )
    return sq, mp_sq, ss


def world_income(mp: MPArrays, period: str) -> float:
    v = mp.vars["regy"]
    return float(sum(v.value[i] for i in np.flatnonzero(v.period == period)))


class RatioBorder:
    """GEMPACK ``del_tbalry(r) = 0`` inside the shock's Newton system. ``savf[r]`` of
    every non-residual region is held at ``savf_check[r] * W / W_check`` (``W`` = world
    income = ``sum(regy)``), the level form of ``INCOMEWLD * del_tbalry = TB * y - TBAL *
    ywld`` with ``del_tbalry = 0`` (gtapv7.tab, E_del_tbalry). ``W`` is one more unknown
    (the last), with its own row ``W - g.x = 0`` (``g`` the ``regy`` shock columns);
    ``F`` writes the savf targets of ``W`` into the system's fixed slots
    (``SquaredSystem.set_fixed``) before evaluating it. The Jacobian is bordered: ``[[J,
    v], [-g, 1]]``, ``v = dF/dW`` through the targets (finite difference, once, at the
    start). The solution is the fixed point the outer iterations of a plain solve at
    ``T(W)`` reach (tested), in one Newton solve."""

    ROW = "eq_ratio_world_income"
    COL = "ratio_world_income"

    def __init__(self, ss: Any, sq: Squared, chk: Mapping[str, float], w_chk: float):
        self.ss, self.chk, self.w_chk = ss, dict(chk), float(w_chk)
        g = np.fromiter(
            (c.startswith("regy[") and c.endswith(",shock]") for c in sq.col_names),
            dtype=bool,
            count=len(sq.col_names),
        )
        self.g_ix = np.flatnonzero(g)
        self.n = len(sq.col_names)
        x0 = np.asarray(ss.x0, dtype=float)
        self.w0 = float(x0[self.g_ix].sum())
        self.fscale_row = max(1.0, abs(self.w0))
        self._w: float | None = None
        f0 = self._F_at(x0, self.w0)
        d = 1e-6 * self.w_chk
        v = (self._F_at(x0, self.w0 + d) - f0) / d
        self._F_at(x0, self.w0)  # the slots back at the start
        self.v_rows = np.flatnonzero(v)
        self.v_vals = v[self.v_rows]

    def targets(self, w: float) -> dict[str, float]:
        """``{region: savf}`` at world income ``w``."""
        return {r: x * w / self.w_chk for r, x in self.chk.items()}

    def _set(self, w: float) -> None:
        if w != self._w:
            self.ss.set_fixed(
                "savf", {(r, "shock"): x for r, x in self.targets(w).items()}
            )
            self._w = w

    def _F_at(self, x: np.ndarray, w: float) -> np.ndarray:
        self._set(w)
        return self.ss.F(x)

    def F(self, z: np.ndarray) -> np.ndarray:
        x, w = z[:-1], float(z[-1])
        return np.append(self._F_at(x, w), w - x[self.g_ix].sum())

    def J(self, z: np.ndarray) -> Any:
        import scipy.sparse as sp

        self._set(float(z[-1]))
        r, c, d = self.ss.J_entries(z[:-1])
        n, ng, nv = self.n, len(self.g_ix), len(self.v_rows)
        rows = np.concatenate([r, self.v_rows, np.full(ng + 1, n)])
        cols = np.concatenate([c, np.full(nv, n), self.g_ix, [n]])
        data = np.concatenate([d, self.v_vals, -np.ones(ng), [1.0]])
        return sp.csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))
