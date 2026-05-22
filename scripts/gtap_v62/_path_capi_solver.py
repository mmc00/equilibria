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
    variable_scaling: bool = True,
    equation_scaling: bool = True,
) -> PathCapiSolveResult:
    """Solve the (already squared) v6.2 Pyomo model with PATH C-API.

    The caller must have applied closure and ``apply_v62_closure_and_square``
    so that ``#free_vars == #active_constraint_cells``.

    Variable scaling (Phase 3.12)
    -----------------------------
    The v6.2 v.f. variables span ~12 orders of magnitude (prices ~ 1,
    factor demands ~ 1e6). PATH's LUSOL factorization becomes
    ill-conditioned and terminates at code=2 (NoProgress) on the
    shocked solve. We apply a diagonal pre-scaling y = x / scale_x
    (where scale_x = max(|x_0|, 1.0)) so PATH sees all variables at
    O(1). The Jacobian columns are scaled correspondingly.

    Equation scaling
    ----------------
    Constraint residuals also span many magnitudes (eq_market ~ 1e4
    at SAM, eq_pms ~ 1). We scale each residual by 1/max(|body_0|, 1)
    so PATH sees residuals at O(1).

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

    # ---------------- Phase 3.12 diagonal scaling -----------------------
    n = len(data.variable_names)

    # var_scale[i]: divides variable i so PATH sees y_i = x_i / var_scale[i].
    if variable_scaling:
        var_scale = [max(abs(v), 1.0) for v in data.x0]
    else:
        var_scale = [1.0] * n

    # eq_scale[i]: divides residual i. Use the baseline body magnitude
    # if available; otherwise fall back to 1.
    if equation_scaling:
        f0 = list(data.callback_f(list(data.x0)))
        # Add a residual reference floor of 1 so equations that are
        # already exactly satisfied at baseline aren't divided by 0.
        eq_scale = [max(abs(fi), 1.0) for fi in f0]
    else:
        eq_scale = [1.0] * n

    # Wrap the callbacks to apply the diagonal scaling. Variable scale
    # acts on the COLUMN side (so col j of Jacobian gets * var_scale[j]),
    # equation scale acts on the ROW side (so each F_i and Jacobian
    # row i gets / eq_scale[i]).
    orig_f = data.callback_f
    orig_jac = data.callback_jac
    structure = data.jacobian_structure

    def f_scaled(y: List[float]) -> List[float]:
        x = [yi * var_scale[i] for i, yi in enumerate(y)]
        f = orig_f(x)
        return [fi / eq_scale[i] for i, fi in enumerate(f)]

    def jac_scaled(y: List[float]) -> List[float]:
        x = [yi * var_scale[i] for i, yi in enumerate(y)]
        jvals = list(orig_jac(x))
        # Jacobian is stored column-major (CSC-like). PATH/JacobianStructure
        # uses 1-BASED indices for col_starts and row_indices, but
        # the values list (jvals) is 0-indexed in Python. Convert.
        col_starts = structure.col_starts  # 1-based
        col_lengths = structure.col_lengths
        row_indices = structure.row_indices  # 1-based
        out: List[float] = [0.0] * len(jvals)
        for j in range(n):
            start_1based = col_starts[j]
            length = col_lengths[j]
            scale_col = var_scale[j]
            for kk_1based in range(start_1based, start_1based + length):
                kk_0based = kk_1based - 1
                row_0based = row_indices[kk_0based] - 1
                out[kk_0based] = (
                    jvals[kk_0based] * scale_col / eq_scale[row_0based]
                )
        return out

    y0 = [v / var_scale[i] for i, v in enumerate(data.x0)]

    def _scale_bound(b: float, s: float) -> float:
        # PATH uses 1e20 as +/-infinity sentinel; keep it as-is.
        if abs(b) >= 1e19:
            return b
        return b / s

    lb_scaled = [_scale_bound(b, var_scale[i]) for i, b in enumerate(data.lb)]
    ub_scaled = [_scale_bound(b, var_scale[i]) for i, b in enumerate(data.ub)]

    result = solve_nonlinear_mcp(
        runtime,
        n=n,
        lb=lb_scaled,
        ub=ub_scaled,
        x0=y0,
        callback_f=f_scaled,
        callback_jac=jac_scaled,
        jacobian_structure=structure,
        output=output,
    )

    # Un-scale the solution before writing it back.
    x_solution = [yi * var_scale[i] for i, yi in enumerate(result.x)]
    for var, val in zip(free_vars, x_solution):
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
