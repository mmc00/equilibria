"""PATH as a recipe solver for GTAP (ADR-0011).

The solver solves the GTAP period model as the steps left it, with the closure
and parameters a step staged (``GTAPPeriodModel.stage``): the driver never reads
the closure. It calls the same PATH entry point the old multiperiod driver
called, with the same arguments.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from equilibria.solver.gc_tuning import _gc_family_scope, _gc_freeze_in_scope

# The options the altertax reference bundles were solved with (the path.opt
# inlined in the regenerated NEOS bundles). Pure gtap bundles carry no path.opt:
# NEOS solved them with PATH's defaults, so gtap passes none.
ALTERTAX_PATH_OPTIONS = (
    "convergence_tolerance 1e-10\n"
    "major_iteration_limit 5000\n"
    "minor_iteration_limit 100000\n"
    "cumulative_iteration_limit 500000\n"
    "crash_method none\n"
    "crash_perturb yes\n"
    "nms_searchtype line\n"
    "gradient_step_limit 1e6\n"
    "proximal_perturbation 0\n"
)


def _load_path_capi() -> Any:
    from equilibria.solver import path_capi

    return path_capi


def _options(options: str | None, gtap_mode: bool) -> str | None:
    """The argument wins, then the caller's ``PATH_CAPI_OPTIONS``, then the
    altertax default; gtap mode has none."""
    if options is not None:
        return options
    env = os.environ.get("PATH_CAPI_OPTIONS")
    if env:
        return env
    return None if gtap_mode else ALTERTAX_PATH_OPTIONS


def path(
    *, options: str | None = None, equation_scaling: bool = True
) -> Callable[[Any], str]:
    """A recipe solver for ``GTAPPyomoPeriodModel``.

    Returns ``"optimal"`` when PATH converged (code 1), else ``"code <n>"``. The
    whole result dict goes to ``model.record_solve``. The GC is tuned for the
    call unless a run scope already holds it (``solver/gc_tuning.py``); no
    environment variable outlives it.
    """

    def solve(model: Any) -> str:
        staged = model.staged
        if staged is None:
            raise RuntimeError(
                "no closure staged: a GTAP step (stage_closure) must run before "
                "the solve"
            )
        # Reentrant: inside GTAP.run_scope (one freeze for the whole run) this
        # changes nothing; a solve called on its own still gets the GC setup.
        with _gc_family_scope():
            # Inside the run scope: freeze what the steps built since (the
            # per-period index caches, the reference model), so the full passes
            # from here on skip it. Measured on 20x41: the gen-2 pass after the
            # base walked them, +3.5 s. Outside a scope this does nothing.
            _gc_freeze_in_scope()
            r = _load_path_capi()._run_path_capi_nonlinear_full(
                model.pyomo,
                staged.params,
                enforce_post_checks=False,
                strict_path_capi=False,
                closure_config=staged.closure,
                equation_scaling=equation_scaling,
                gtap_mode=staged.gtap_mode,
                solver_output=bool(os.environ.get("EQUILIBRIA_DEBUG_PATH_VERBOSE")),
                solution_hint=None,
                path_options=_options(options, staged.gtap_mode),
            )
        model.record_solve(r)
        code = int(r.get("termination_code") or 0)
        return "optimal" if code == 1 else f"code {code}"

    return solve
