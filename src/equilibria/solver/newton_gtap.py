"""Newton-TR as a recipe solver for GTAP on the family backend (ADR-0011).

It does what the family driver does between a period's recipe and its post-solve
(the old family driver's ``_solve_period_inner``): square the arrays for the staged period,
build the JAX system, solve it with Newton-TR + MUMPS, write the solution back.
The squared copy (with the closure fixings, deactivated rows and floors it carries)
becomes the model's arrays, as Pyomo keeps what PATH's squaring left on its model.

The squaring reads the run's closure (``FamilyRun.closure``, the one the old family
route squares with); the cell index, the adjacency and the MUMPS analysis kept for
the next solve live on the run, so the solver holds no state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def newton_tr(*, border: str | None = None) -> Callable[[Any], str]:
    """A recipe solver for ``GTAPFamilyPeriodModel``.

    Returns ``"optimal"`` when Newton converged with the squared residual under
    ``CODE_RESIDUAL`` (code 1), else ``"code <n>"``. ``border="ratio"`` (shock
    only): GEMPACK ``del_tbalry = 0``, the system bordered with world income
    (``RatioBorder``), savf of every non-residual region at its check ratio."""
    if border not in (None, "ratio"):
        raise ValueError(f"border must be None or 'ratio', got {border!r}")

    def solve(model: Any) -> str:
        from equilibria.family import period_newton as fd
        from equilibria.family.post_solve import writeback
        from equilibria.family.run_setup import solve_record

        if getattr(model, "backend", None) != "family":
            raise TypeError(
                f"newton_tr solves the family model, got backend "
                f"{getattr(model, 'backend', None)!r}"
            )
        staged = model.staged
        if staged is None:
            raise RuntimeError(
                "no closure staged: a GTAP step (stage_closure) must run before "
                "the solve"
            )
        run, period = model.run, staged.period
        sq, mp_sq, ss = fd._period_system(run, model.mp, period)
        ratio = None
        if border == "ratio":
            if period != "shock":
                raise ValueError("border='ratio' is for the shock period only")
            held = [r for r in run.regions if r != run.residual_region]
            sv = model.mp.vars["savf"]
            chk = {r: float(sv.value[sv.pos((r, "check"))]) for r in held}
            ratio = fd.RatioBorder(ss, sq, chk, fd.world_income(model.mp, "check"))
            if len(ratio.g_ix) != len(run.regions):
                raise ValueError(
                    f"trade_balance='ratio': {len(ratio.g_ix)} regy shock columns "
                    f"for {len(run.regions)} regions"
                )
        token = fd._NEWTON_TOL.set(run.newton_tol)
        keep = fd._MUMPS_KEEP.set(run.mumps)
        try:
            res, resid, code = fd._newton(ss, sq, ratio)
        finally:
            fd._NEWTON_TOL.reset(token)
            fd._MUMPS_KEEP.reset(keep)
        x = np.asarray(res.x, dtype=float)
        if ratio is not None:
            w, x = float(x[-1]), x[:-1]
            sv1 = mp_sq.vars["savf"]
            for r, value in ratio.targets(w).items():
                sv1.value[sv1.pos((r, "shock"))] = value
        writeback(mp_sq, sq, x)
        model.mp = mp_sq
        model.record_solve(
            {"termination_code": code, "residual": resid, "nit": res.nit}
        )
        run.solves.append(solve_record(period, res, resid, code))
        return "optimal" if code == 1 else f"code {code}"

    return solve
