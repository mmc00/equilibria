"""Solve the PEP Pyomo model and read variable values out.

NLP form: IPOPT on the raw model with ``nlp_scaling_method=none`` (faithful to GAMS,
which solves the raw model; the GTAP saga proved Pyomo's default gradient pre-scaling
steers IPOPT to a wrong basin — see project_gtap7_altertax_ref_and_nlp_scaling).

MCP form: PATH via path-capi-python (walras⊥LEON free-row, e fixed numeraire), mirroring
the GTAP MCP path. Falls back to a clear error if PATH is unavailable.

The benchmark BASE reproduces the SAM, so seeding at *O levels and solving should return
residual≈0 (the cyipopt solver early-exits there) — that is the parity anchor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyomo.environ import SolverFactory, Var, Constraint, value


def _ensure_path_lib() -> None:
    """Point PATH_CAPI_LIBPATH at the PATH C-API dylib (like GTAP's run_gtap), so the
    MCP solve is self-contained. Searches the known locations; no-op if already set."""
    import os
    from pathlib import Path
    if os.environ.get("PATH_CAPI_LIBPATH"):
        return
    for cand in (
        "/Users/marmol/proyectos2/equilibria/.cache/path_capi/libpath50.silicon.dylib",
        "/Users/marmol/proyectos/path-capi-python/notes/tmp/path_capi_artifacts/"
        "aarch64-apple-darwin/libpath.dylib",
        "/Library/Frameworks/GAMS.framework/Versions/53/Resources/libpath52.dylib",
    ):
        if Path(cand).exists():
            os.environ["PATH_CAPI_LIBPATH"] = cand
            return


@dataclass
class PEPSolveResult:
    code: int                      # 1 = optimal/feasible, 2 = not converged
    max_residual: float
    values: dict[str, Any] = field(default_factory=dict)   # {var_name: {idx: value}}
    message: str = ""


def _collect_values(m) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for v in m.component_objects(Var, active=True):
        d = {}
        for idx in v:
            try:
                val = value(v[idx], exception=False)
            except Exception:
                val = None
            if val is not None:
                d[idx] = float(val)
        out[v.name] = d
    return out


def _max_residual(m) -> float:
    worst = 0.0
    for c in m.component_objects(Constraint, active=True):
        for idx in c:
            con = c[idx]
            try:
                body = value(con.body, exception=False)
                lo = value(con.lower, exception=False) if con.lower is not None else None
                up = value(con.upper, exception=False) if con.upper is not None else None
            except Exception:
                continue
            if body is None:
                continue
            target = lo if lo is not None else up
            if target is not None:
                worst = max(worst, abs(body - target))
    return worst


def solve_pep(m, tol: float = 1e-7, max_iter: int = 3000) -> PEPSolveResult:
    """Solve the built PEP model. NLP forms go through IPOPT (raw, no pre-scaling)."""
    form = m._pep.get("form", "nlp")
    if form == "mcp":
        return _solve_mcp(m, tol)
    # Early-exit: if the seed already satisfies the system (the benchmark BASE point IS
    # the calibration answer), do NOT re-solve — return it. Mirrors the original PEP
    # cyipopt solver (skips IPOPT when the guess is feasible, pep_model_solver_ipopt:2601)
    # and GAMS CNS (which confirms the loaded calibration, not re-optimizes). Otherwise a
    # constant-0 objective gives IPOPT no reason to stay and it drifts to another feasible
    # point (the homogeneous CGE has a manifold of them).
    seed_resid = _max_residual(m)
    if seed_resid <= max(tol * 100, 1e-4):
        return PEPSolveResult(code=1, max_residual=seed_resid,
                              values=_collect_values(m), message="feasible-at-seed (no re-solve)")
    opt = SolverFactory("ipopt")
    opt.options["nlp_scaling_method"] = "none"   # faithful to GAMS raw solve
    opt.options["tol"] = tol
    opt.options["max_iter"] = max_iter
    opt.options["print_level"] = 0
    res = opt.solve(m, tee=False, load_solutions=True)
    tc = str(res.solver.termination_condition)
    resid = _max_residual(m)
    ok = tc in ("optimal", "locallyOptimal", "feasible") or resid <= max(tol * 100, 1e-4)
    return PEPSolveResult(code=1 if ok else 2, max_residual=resid,
                          values=_collect_values(m), message=tc)


def _solve_mcp(m, tol: float) -> PEPSolveResult:
    """MCP via PATH — the square PEP system (equality constraints, e fixed numeraire,
    WALRAS⊥LEON free-row) solved through path-capi's PATHCAPIBridgeSolver, which pairs
    equations↔variables and calls PATH. Requires path-capi-python; clear error if absent."""
    import importlib.util
    if importlib.util.find_spec("path_capi_python") is None:
        return PEPSolveResult(code=2, max_residual=float("nan"),
                              message="path_capi_python unavailable for MCP solve")
    _ensure_path_lib()
    import path_capi_python  # noqa: F401  (registers the 'path_capi_bridge' SolverFactory)
    # Early-exit at a feasible seed, same as NLP (BASE is the calibration point).
    seed_resid = _max_residual(m)
    if seed_resid <= max(tol * 100, 1e-4):
        return PEPSolveResult(code=1, max_residual=seed_resid,
                              values=_collect_values(m),
                              message="feasible-at-seed (no re-solve, MCP)")
    opt = SolverFactory("path_capi_bridge")
    if not opt.available(exception_flag=False):
        return PEPSolveResult(code=2, max_residual=float("nan"),
                              message="path_capi_bridge solver unavailable")
    # Pass the exact free-variable list so the bridge doesn't infer it (its inference
    # excludes now-fixed structural-zero cells that still appear in market/income eqs,
    # giving a spurious expr≠var mismatch). The model is square (358 free vars = 358 eqs).
    from pyomo.environ import Var
    free_vars = [v for v in m.component_data_objects(Var, active=True) if not v.fixed]
    try:
        res = opt.solve(m, load_solutions=True, variables=free_vars)
        tc = str(res.solver.termination_condition)
    except Exception as e:  # noqa: BLE001
        return PEPSolveResult(code=2, max_residual=float("nan"),
                              message=f"PATH solve error: {e}")
    resid = _max_residual(m)
    ok = tc in ("optimal", "feasible") or resid <= max(tol * 100, 1e-4)
    return PEPSolveResult(code=1 if ok else 2, max_residual=resid,
                          values=_collect_values(m), message=f"PATH {tc}")
