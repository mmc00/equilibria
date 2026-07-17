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
    """MCP via PATH (walras⊥LEON). Requires path-capi-python; error clearly if absent."""
    import importlib.util
    if importlib.util.find_spec("path_capi_python") is None:
        return PEPSolveResult(code=2, max_residual=float("nan"),
                              message="path_capi_python unavailable for MCP solve")
    # The square PEP system with e fixed + walras⊥LEON is solved as a complementarity
    # problem; reuse the nonlinear-full PATH driver the GTAP template exposes.
    from path_capi_python import solve_nonlinear_full  # type: ignore
    code = solve_nonlinear_full(m, tol=tol)  # returns PATH status code (1=solved)
    resid = _max_residual(m)
    return PEPSolveResult(code=1 if code == 1 else 2, max_residual=resid,
                          values=_collect_values(m), message=f"PATH code={code}")
