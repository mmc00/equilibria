"""PATH C-API solver helper for GTAP v6.2.

Wraps ``path_capi_python`` for the v6.2 MCP after closure squaring.
The v6.2 model is simpler than v7 so we don't need the GTAPSolver
helper from ``scripts/gtap``; the auto-square in ``_make_square.py``
already produces ``free_vars == active_cons``.

Environment:
    PATH_LICENSE_STRING  PATH licensing string (required for n>300)
    PATH_CAPI_LIBPATH    Path to PATH shared library (e.g. path52.dll)
    PATH_CAPI_LIBLUSOL   Path to LUSOL shared library (e.g. lusol.dll)
    PATH_CAPI_SRC        Path to path-capi-python source (optional)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pyomo.environ import Constraint, Var, value


PATH_CAPI_SRC_DEFAULT = Path("c:/Documentos/proyectos/path-capi-python/src")


def _ensure_path_capi_on_syspath() -> None:
    src = Path(os.environ.get("PATH_CAPI_SRC") or PATH_CAPI_SRC_DEFAULT)
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass
class PathCapiSolveResult:
    termination_code: int
    residual: float
    major_iterations: int
    minor_iterations: int
    function_evaluations: int
    jacobian_evaluations: int
    n_vars: int
    n_cons: int


def solve_v62_with_path_capi(
    model: Any,
    *,
    output: bool = False,
    license_string: str | None = None,
    path_lib: str | None = None,
    lusol_lib: str | None = None,
) -> PathCapiSolveResult:
    """Solve the (already squared) v6.2 Pyomo model with PATH C-API.

    The caller must have applied closure and ``apply_v62_closure_and_square``
    so that ``#free_vars == #active_constraint_cells``.

    Returns a result dict with termination code and stats. Writes the
    solution back into the Pyomo Var objects.
    """
    _ensure_path_capi_on_syspath()

    if license_string:
        os.environ["PATH_LICENSE_STRING"] = license_string
    if path_lib:
        os.environ["PATH_CAPI_LIBPATH"] = path_lib
    if lusol_lib:
        os.environ["PATH_CAPI_LIBLUSOL"] = lusol_lib

    from path_capi_python import PATHLoader, PyomoMCPAdapter  # type: ignore
    from path_capi_python.mcp import solve_nonlinear_mcp  # type: ignore

    runtime = PATHLoader.from_environment().load()

    free_vars: List[Any] = []
    for v in model.component_objects(Var, active=True):
        for idx in v:
            if not v[idx].fixed:
                free_vars.append(v[idx])

    active_cons: List[Any] = []
    for c in model.component_objects(Constraint, active=True):
        for idx in c:
            if c[idx].active:
                active_cons.append(c[idx])

    n_vars = len(free_vars)
    n_cons = len(active_cons)
    if n_vars != n_cons:
        raise RuntimeError(
            f"Model not square: {n_vars} free vars vs {n_cons} active cons"
        )

    # Build nonlinear callbacks (residual + Jacobian) using
    # reverse_numeric mode — faster than symbolic for this model size.
    adapter = PyomoMCPAdapter()
    data = adapter.build_nonlinear_from_equality_constraints(
        model,
        constraints=active_cons,
        variables=free_vars,
        jacobian_eval_mode="reverse_numeric",
    )

    result = solve_nonlinear_mcp(
        runtime,
        n=len(data.variable_names),
        lb=data.lb,
        ub=data.ub,
        x0=data.x0,
        callback_f=data.callback_f,
        callback_jac=data.callback_jac,
        jacobian_structure=data.jacobian_structure,
        output=output,
    )

    # Write the solution back into the Pyomo Var objects.
    for var, val in zip(free_vars, result.x):
        var.set_value(float(val), skip_validation=True)

    return PathCapiSolveResult(
        termination_code=result.termination_code,
        residual=result.residual,
        major_iterations=result.major_iterations,
        minor_iterations=result.minor_iterations,
        function_evaluations=result.function_evaluations,
        jacobian_evaluations=result.jacobian_evaluations,
        n_vars=n_vars,
        n_cons=n_cons,
    )
