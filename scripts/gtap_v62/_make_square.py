"""Auto-square helper for GTAP v6.2 model (Phase 3 closure).

PATH MCP solvers require ``#free_vars == #active_constraints``. This
helper applies the canonical v6.2 closure (numeraire, endowments,
foreign savings) and then adds three identity equation families plus
auto-fixes any remaining dangling Pyomo variable cells (cells that
don't appear in any active constraint).

Usage::

    from scripts.gtap_v62._make_square import apply_v62_closure_and_square

    closure_info = apply_v62_closure_and_square(model)
    assert closure_info["mismatch"] == 0
"""

from __future__ import annotations

from typing import Any, Dict, List

from pyomo.environ import Constraint, Var, value


def _vars_in_constraint(c) -> set:
    """Return the set of (var.name, idx) tuples that ``c.body`` references."""
    from pyomo.core.expr.visitor import identify_variables
    out: set = set()
    for v in identify_variables(c.body, include_fixed=False):
        out.add((v.parent_component().name, v.index()))
    return out


def apply_v62_closure_and_square(model: Any) -> Dict[str, Any]:
    """Fix the v6.2 canonical exogenous vars + auto-fix dangling vars.

    Returns a dict with closure stats and the final var/cons counts.
    """
    info: Dict[str, Any] = {
        "fixed_explicit": [],
        "fixed_dangling": [],
        "added_identity_eqs": [],
    }

    # ----- canonical v6.2 closure --------------------------------------
    # Numeraire
    model.pgdpwld.fix(1.0)
    info["fixed_explicit"].append(("pgdpwld", 1))

    # Endowments are exogenous (static closure)
    n_qoes = 0
    for f in model.f:
        for r in model.r:
            model.qoes[f, r].fix(model.qoes[f, r].value)
            n_qoes += 1
    info["fixed_explicit"].append(("qoes", n_qoes))

    # Net foreign savings exogenous (current-account fixed)
    n_savf = 0
    for r in model.r:
        model.savf[r].fix(model.savf[r].value)
        n_savf += 1
    info["fixed_explicit"].append(("savf", n_savf))

    # Capital stocks exogenous in static
    for r in model.r:
        model.kb[r].fix(model.kb[r].value)
        model.ke[r].fix(model.ke[r].value)
    info["fixed_explicit"].append(("kb", len(list(model.r))))
    info["fixed_explicit"].append(("ke", len(list(model.r))))

    # Global rate of return fixed at 1 (no investment dynamics)
    model.rorg.fix(1.0)
    info["fixed_explicit"].append(("rorg", 1))

    # GDP deflator pinned at 1 (numeraire-equivalent)
    for r in model.r:
        model.pgdpmp[r].fix(1.0)
    info["fixed_explicit"].append(("pgdpmp", len(list(model.r))))

    # ----- identity equations -----------------------------------------
    # eq_qds: qds(i,r) = sum_j qfd(i,j,r) + qpd(i,r) + qgd(i,r)
    if not hasattr(model, "eq_qds"):
        def eq_qds_rule(m, i, r):
            return m.qds[i, r] == (
                sum(m.qfd[i, j, r] for j in m.j)
                + m.qpd[i, r] + m.qgd[i, r]
            )
        model.eq_qds = Constraint(model.i, model.r, rule=eq_qds_rule)
        info["added_identity_eqs"].append(("eq_qds", 9))

    # eq_psave: psave(r) = pcgds("CGDS", r)
    if not hasattr(model, "eq_psave"):
        def eq_psave_rule(m, r):
            return m.psave[r] == m.pcgds["CGDS", r]
        model.eq_psave = Constraint(model.r, rule=eq_psave_rule)
        info["added_identity_eqs"].append(("eq_psave", 3))

    # eq_gdpmp: nominal GDP = regional income y
    if not hasattr(model, "eq_gdpmp"):
        def eq_gdpmp_rule(m, r):
            return m.gdpmp[r] == m.y[r]
        model.eq_gdpmp = Constraint(model.r, rule=eq_gdpmp_rule)
        info["added_identity_eqs"].append(("eq_gdpmp", 3))

    # eq_rgdpmp: real GDP = gdpmp / pgdpmp
    if not hasattr(model, "eq_rgdpmp"):
        def eq_rgdpmp_rule(m, r):
            return m.rgdpmp[r] == m.gdpmp[r] / m.pgdpmp[r]
        model.eq_rgdpmp = Constraint(model.r, rule=eq_rgdpmp_rule)
        info["added_identity_eqs"].append(("eq_rgdpmp", 3))

    # ----- auto-fix dangling variables --------------------------------
    used_vars: set = set()
    for c in model.component_objects(Constraint, active=True):
        for idx in c:
            used_vars |= _vars_in_constraint(c[idx])

    dangling_by_family: Dict[str, int] = {}
    for v in model.component_objects(Var, active=True):
        for idx in v:
            if v[idx].fixed:
                continue
            key = (v.name, idx)
            if key not in used_vars:
                v[idx].fix(v[idx].value or 0.0)
                dangling_by_family[v.name] = dangling_by_family.get(v.name, 0) + 1

    info["fixed_dangling"] = sorted(dangling_by_family.items(), key=lambda x: -x[1])

    # ----- iterative squaring -----------------------------------------
    # After fixing dangling vars, some equations may have become
    # "trivially satisfied" (all their free vars got fixed). Walk over
    # the system once more, deactivating constraints whose body has no
    # free variables left.
    from pyomo.core.expr.visitor import identify_variables
    deactivated_by_family: Dict[str, int] = {}
    for c in model.component_objects(Constraint, active=True):
        for idx in list(c):
            free_in_body = [
                v for v in identify_variables(c[idx].body, include_fixed=False)
            ]
            if not free_in_body:
                c[idx].deactivate()
                deactivated_by_family[c.name] = deactivated_by_family.get(c.name, 0) + 1
    info["deactivated_trivial_cons"] = sorted(
        deactivated_by_family.items(), key=lambda x: -x[1]
    )

    # ----- final counts -----------------------------------------------
    free = sum(1 for v in model.component_objects(Var, active=True)
               for idx in v if not v[idx].fixed)
    cons = sum(1 for c in model.component_objects(Constraint, active=True)
               for _ in c)
    info["free_vars"] = free
    info["active_cons"] = cons
    info["mismatch"] = free - cons

    return info
