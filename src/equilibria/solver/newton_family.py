"""A recipe solver for ``GenericFamilyPeriodModel``: Newton on one period's rows.

The period is the one whose rows are active (``freeze_other_periods`` leaves one).
Its free cells are the columns and its active rows the equations; the system must
be square. No closure, matching table or environment tuning: those are GTAP's
(``newton_gtap``). The step is ``newton_tr`` with a direct sparse solve and the
numeric column matching it does when given no permutation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _active_period(model: Any) -> str:
    periods = {t for r in model.mp.rows.values() for t in np.unique(r.period[r.active])}
    if len(periods) != 1:
        raise ValueError(f"expected rows active in one period, got {sorted(periods)}")
    return periods.pop()


def newton_family(*, tol: float = 1e-10, maxit: int = 50) -> Callable[[Any], str]:
    """Solve the active period to ``||F||_inf < tol``; ``"optimal"`` or ``"failed"``."""

    def solve(model: Any) -> str:
        from equilibria.solver.newton_tr import newton_tr

        period = _active_period(model)
        ps = model.period_system(period)
        rows = np.flatnonzero(ps.active)
        n_cols = ps.layout.n_cols
        if rows.size != n_cols:
            raise ValueError(
                f"not square: {rows.size} rows, {n_cols} free cells in {period}"
            )
        lb, ub = np.full(n_cols, -np.inf), np.full(n_cols, np.inf)
        for name, (slots, at) in ps.cols.items():
            lb[slots] = model.mp.vars[name].lb[at]
            ub[slots] = model.mp.vars[name].ub[at]

        def F(x: np.ndarray) -> np.ndarray:
            return ps.system.F(x)[rows]

        def J(x: np.ndarray) -> Any:
            return ps.system.jac(x).tocsr()[rows]

        res = newton_tr(F, J, ps.x0(), lb, ub, ftol=tol, linsolve="direct", maxit=maxit)
        model.write_back(ps, res.x)
        residual = float(np.max(np.abs(F(res.x)))) if rows.size else 0.0
        code = "optimal" if res.success and residual < tol else "failed"
        model.last_solve = {"code": code, "residual": residual, "iterations": res.nit}
        return code

    return solve
