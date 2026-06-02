"""Multi-period sequential solve loop for GTAP recursive-dynamic MCP.

Mirrors GAMS solveloop.gms structure: one fresh ConcreteModel per period,
with lagged-state Vars fixed from the previously-solved period before each
solve. Single-period t_set=('base',) gives bit-identical results to the
existing _run_path_capi_nonlinear_full single solve.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from pyomo.environ import ConcreteModel

from equilibria.templates.gtap.gtap_iterloop import fix_lagged_state
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

logger = logging.getLogger(__name__)


@dataclass
class PeriodResult:
    """Holds the solved model + metadata for a single period in t_set."""
    period: str
    model: ConcreteModel
    residual: float
    solver_metadata: dict = field(default_factory=dict)


def _default_solver():
    """Lazy import of the production single-period solver.

    Resolved on demand so that importing this module does not require the
    ``scripts/gtap`` directory to be on sys.path. Callers that prefer to
    avoid the sys.path mutation should pass ``solver_fn`` explicitly.
    """
    import sys
    scripts_dir = Path(__file__).resolve().parents[4] / "scripts" / "gtap"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_gtap import _run_path_capi_nonlinear_full  # type: ignore
    return _run_path_capi_nonlinear_full


def solve_sequential(
    params: Any,
    closure: Any,
    t_set: tuple[str, ...] = ("base",),
    *,
    solver_fn: Optional[Callable] = None,
    per_period_shock_fn: Optional[Callable[[str, ConcreteModel, Any], None]] = None,
    equation_scaling: bool = True,
    path_capi_convergence_tol: float = 1e-8,
) -> dict[str, PeriodResult]:
    """Solve a sequence of periods using the multi-instance + lagged-fix design.

    For each period in ``t_set``:

    * Build a fresh :class:`GTAPModelEquations` ConcreteModel.
    * For periods after the base, copy & fix lagged-state Vars from the
      previously-solved model via :func:`fix_lagged_state`.
    * If ``per_period_shock_fn`` is given, call it with
      ``(period_name, new_model, params)`` BEFORE solving (so the caller
      can mutate model params/Vars for shocks).
    * Solve via ``solver_fn`` (defaults to
      ``run_gtap._run_path_capi_nonlinear_full``).

    Parameters
    ----------
    params
        GTAPParameters instance with ``.sets`` and loaded data.
    closure
        Closure config passed to GTAPModelEquations and the solver.
    t_set
        Ordered tuple of period names. First element MUST be ``"base"``
        (validated by :class:`GTAPModelEquations`).
    solver_fn
        Callable
        ``(model, params, closure_config, equation_scaling, path_capi_convergence_tol)``
        returning a dict with at least ``'residual'``. Defaults to the
        production PATH C API solver via lazy import.
    per_period_shock_fn
        Optional ``(period_name, new_model, params) -> None`` callback,
        invoked AFTER ``fix_lagged_state`` and BEFORE solve.
    equation_scaling
        Forwarded to ``solver_fn``. Default ``True`` matches the rest of
        the codebase (baseline residual ~1e-9 vs ~1e-6 without scaling).
    path_capi_convergence_tol
        Forwarded to ``solver_fn``.

    Returns
    -------
    dict[str, PeriodResult]
        Mapping period name → :class:`PeriodResult`. All intermediate
        models are preserved so the caller can inspect any period.
    """
    if not t_set:
        raise ValueError("t_set must be non-empty")

    if solver_fn is None:
        solver_fn = _default_solver()

    results: dict[str, PeriodResult] = {}
    prev_result: Optional[PeriodResult] = None

    for period in t_set:
        logger.info("Solving period %s", period)

        builder = GTAPModelEquations(
            params.sets, params, closure, t_set=t_set,
        )
        model = builder.build_model()

        if prev_result is not None:
            n_fixed = fix_lagged_state(model, prev_result.model)
            logger.info("Fixed %d lagged-state values for period %s",
                        n_fixed, period)

        if per_period_shock_fn is not None:
            per_period_shock_fn(period, model, params)

        t0 = time.time()
        solver_out = solver_fn(
            model, params,
            closure_config=closure,
            equation_scaling=equation_scaling,
            path_capi_convergence_tol=path_capi_convergence_tol,
        )
        elapsed = time.time() - t0

        residual = float(solver_out.get("residual", float("nan")))
        logger.info("Solved %s in %.2fs, residual=%.3e", period, elapsed, residual)

        pr = PeriodResult(
            period=period,
            model=model,
            residual=residual,
            solver_metadata=dict(solver_out),
        )
        results[period] = pr
        prev_result = pr

    return results
