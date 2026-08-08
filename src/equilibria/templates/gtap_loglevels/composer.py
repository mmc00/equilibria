"""GTAP log-levels composer.

Composes the 7 log-wrapped levels blocks (equilibria.blocks.gtap_loglevels) through
the neutral PyomoBackend and solves as one NLP (IPOPT, feasibility objective). This is
OUR levels model in log form — the savf capital account, Fisher price index, and
gy-indexed tax streams of blocks/gtap, with equations log(lhs)==log(rhs) where valid.

Mirrors gtap_block_model.build_block_single_period's assembly but with the log-wrapped
block order, WITHOUT the monolith's levels-specific benchmark scaling (the log form is
already well-scaled). Closure (exogenous fixings + numeraire) comes from the levels
ClosureBlock's own equations + the standard exogenous set.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from equilibria.backends.pyomo_backend import PyomoBackend
from equilibria.blocks.gtap_loglevels import GTAP_LOGLEVELS_BLOCK_ORDER
from equilibria.core.sets import Set as ESet
from equilibria.model import Model
from equilibria.templates.gtap.gtap_block_model import (
    _mk_unit,
    _set_elems,
    _strip_con_suffix,
)


def build_loglevels_model(
    params: Any,
    sets: Any,
    closure: Any = None,
    residual_region: str | None = None,
):
    """Compose the 7 log-wrapped levels blocks → Pyomo ConcreteModel (+ feasibility obj)."""
    if_sub = bool(getattr(closure, "if_sub", False))
    savf_flag = str(getattr(closure, "savf_flag", "capFix"))
    setmap = _set_elems(sets)
    model = Model(name="gtap_loglevels_sp")
    for name, elems in setmap.items():
        model.add_set(ESet(name=name, elements=tuple(elems)))
    for cls in GTAP_LOGLEVELS_BLOCK_ORDER:
        model.add_block(
            _mk_unit(
                cls,
                sets,
                params,
                residual_region or "ROW",
                if_sub=if_sub,
                savf_flag=savf_flag,
            )
        )
    backend = PyomoBackend()
    backend.build(model)
    pm = backend.pyomo_model
    assert pm is not None  # PyomoBackend.build populates pyomo_model
    _strip_con_suffix(pm)
    pm._residual_region = residual_region
    pm.add_component("_obj", pyo.Objective(expr=0.0))
    return pm


def _ipopt():
    opt = pyo.SolverFactory("ipopt")
    opt.options["tol"] = 1e-8
    opt.options["constr_viol_tol"] = 1e-8
    opt.options["bound_push"] = 1e-15
    return opt


def solve(
    params: Any,
    sets: Any,
    closure: Any = None,
    residual_region: str | None = None,
    tee: bool = False,
) -> dict[str, Any]:
    """Build + solve the base log-levels model with IPOPT."""
    pm = build_loglevels_model(params, sets, closure, residual_region)
    res = _ipopt().solve(pm, tee=tee, load_solutions=True)
    tc = str(res.solver.termination_condition)
    return {"status": tc, "ok": tc in ("optimal", "locallyOptimal"), "model": pm}
