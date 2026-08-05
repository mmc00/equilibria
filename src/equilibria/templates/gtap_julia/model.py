"""Assemble and solve the gtap_julia model.

Builds all six equation groups on a seeded Pyomo model, applies the standard
closure (fix exogenous vars + numeraire), and solves as an NLP with IPOPT. The
seed is Julia's solved point, so the base solve should reproduce it (and a shock
re-solves from there). This is the log-value-balance GTAPv7 model that converges
capFlex where the levels block model cannot.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from .closure import apply_closure
from .equations import build_group
from .solution import seed_model

_GROUPS = ("production", "factors", "trade", "final_demand", "income", "capital")


def build_model(sol: dict[str, Any], rordelta: int = 1) -> pyo.ConcreteModel:
    """Seed a model, build all equation groups + closure. Returns the model."""
    m = pyo.ConcreteModel()
    seed_model(m, sol)
    for g in _GROUPS:
        if g == "capital":
            build_group(m, sol, g, rordelta=rordelta)
        else:
            build_group(m, sol, g)
    apply_closure(m, sol)
    _fix_orphan_vars(m)
    # objective: pure feasibility (CNS) — IPOPT needs one; constant is fine
    m._obj = pyo.Objective(expr=0.0)
    return m


def _fix_orphan_vars(m) -> None:
    """Fix any free Var not appearing in a constraint (non-existent cells the δ
    masks exclude: unused factor slots, empty margin routes). Julia deletes these;
    here we pin them at their seed so the system is square. A NaN seed → fix at 1."""
    from pyomo.core.expr.visitor import identify_variables

    used = set()
    for c in m.component_data_objects(pyo.Constraint):
        for v in identify_variables(c.body):
            used.add(id(v))
    for v in m.component_data_objects(pyo.Var):
        if not v.fixed and id(v) not in used:
            val = v.value
            v.fix(1.0 if val is None or val != val else val)


def solve(sol: dict[str, Any], rordelta: int = 1, tee: bool = False) -> dict[str, Any]:
    """Build + solve the base model with IPOPT. Returns {status, model}."""
    m = build_model(sol, rordelta=rordelta)
    opt = pyo.SolverFactory("ipopt")
    opt.options["tol"] = 1e-8
    opt.options["constr_viol_tol"] = 1e-8
    res = opt.solve(m, tee=tee, load_solutions=True)
    tc = str(res.solver.termination_condition)
    return {
        "status": tc,
        "ok": tc in ("optimal", "locallyOptimal"),
        "model": m,
    }
