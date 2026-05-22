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

from pyomo.environ import Constraint, ConstraintList, Var, value


def _vars_in_constraint(c) -> set:
    """Return the set of (var.name, idx) tuples that ``c.body`` references."""
    from pyomo.core.expr.visitor import identify_variables
    out: set = set()
    for v in identify_variables(c.body, include_fixed=False):
        out.add((v.parent_component().name, v.index()))
    return out


def bake_baseline_residuals_as_slacks(
    model: Any,
    *,
    tolerance: float = 1.0e-3,
) -> Dict[str, Any]:
    """Pre-balance the SAM by baking each non-zero baseline residual into
    its equation as a constant slack.

    Rationale
    ---------
    GEMPACK never asks whether ``F(x_0) == 0`` because it solves the
    linearized (Johansen) form: derivatives at the SAM are all that
    matter, and any baseline imbalance is implicitly dragged along as
    a constant. A levels-MCP solver (PATH) instead enforces
    ``F(x) == 0`` strictly and reports ``code=2`` (stationary point of
    the merit function) when the SAM has structural imperfections it
    cannot resolve (e.g. intra-region VTWR, ~1% market-clearing gap).

    For BOOK3X3 the SAM imperfections are:
      * ``eq_qtm``  ~6.6e4  — VTWR[m,i,s,s] diagonal (~65,838 total)
      * ``eq_market`` ~2.3e4 — export margin pushed onto uses side
        through ``qxs_0 = vxwd`` calibration (Phase 2d)
      * ``eq_qo``   ~6e-2   — implicit output-tax wedge (held by
        the calibrated ``to`` param; already balances)

    Strategy
    --------
    For each ACTIVE constraint cell, evaluate ``body - rhs`` at the
    benchmark. If |residual| > ``tolerance``, deactivate the cell and
    replace it with a new constraint of the form ``body == rhs +
    residual_0`` (i.e. add ``residual_0`` as a constant on the RHS).
    The deltas (dF/dx · dx) are unchanged; only the constant offset
    moves. Dynamics propagate identically.

    Returns
    -------
    A dict ``{eq_name: [(idx, residual_0), ...]}`` of baked cells, plus
    a top-level ``"n_baked"`` count and ``"max_abs_baked"`` magnitude.
    """
    info: Dict[str, Any] = {
        "n_baked": 0,
        "max_abs_baked": 0.0,
        "by_eq": {},
    }

    # Holder for the replacement constraints. ConstraintList lets us add
    # one cell at a time without per-equation rule scaffolding.
    if not hasattr(model, "sam_baked_residuals"):
        model.sam_baked_residuals = ConstraintList()

    to_bake: List[Tuple[Any, Any, float, Any, Any, Any]] = []
    for c in model.component_objects(Constraint, active=True):
        for idx in list(c):
            cobj = c[idx]
            if not cobj.active:
                continue
            body_val = value(cobj.body)
            if cobj.upper is not None and cobj.lower is not None \
                    and cobj.upper is cobj.lower:
                rhs = value(cobj.upper)
                residual = body_val - rhs
            elif cobj.upper is not None:
                rhs = value(cobj.upper)
                residual = body_val - rhs
            elif cobj.lower is not None:
                rhs = value(cobj.lower)
                residual = body_val - rhs
            else:
                rhs = 0.0
                residual = body_val

            if abs(residual) <= tolerance:
                continue
            to_bake.append((c, idx, residual, cobj.body, cobj.lower, cobj.upper))

    for c, idx, residual, body, lower, upper in to_bake:
        # Deactivate the original cell.
        c[idx].deactivate()

        # Replacement: body - residual_0 == rhs  (equivalent to body == rhs + residual_0)
        # For an equality constraint (lower == upper == rhs), build a new one.
        if lower is not None and upper is not None:
            rhs = value(upper)
            model.sam_baked_residuals.add(body - residual == rhs)
        elif upper is not None:
            rhs = value(upper)
            model.sam_baked_residuals.add(body - residual <= rhs)
        elif lower is not None:
            rhs = value(lower)
            model.sam_baked_residuals.add(body - residual >= rhs)
        else:
            model.sam_baked_residuals.add(body - residual == 0)

        info["n_baked"] += 1
        info["max_abs_baked"] = max(info["max_abs_baked"], abs(residual))
        eq_name = c.name
        info["by_eq"].setdefault(eq_name, []).append((str(idx), residual))

    return info


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

    # eq_qim: composite import quantity = sum of bilateral imports
    # qim(i,r) is used as input in eq_qxs but has no defining equation;
    # the CES aggregator identity pim*qim = sum_s pms*qxs gives the
    # economic content.
    if not hasattr(model, "eq_qim"):
        # Phase 3.16: sum over all sources (incl. s == r) per GEMPACK.
        def eq_qim_rule(m, i, r):
            return m.pim[i, r] * m.qim[i, r] == sum(
                m.pms[i, s, r] * m.qxs[i, s, r]
                for s in m.s
            )
        model.eq_qim = Constraint(model.i, model.r, rule=eq_qim_rule)
        info["added_identity_eqs"].append(("eq_qim", 9))

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

    # ----- bipartite matching for MCP square --------------------------
    # An MCP solver (PATH) needs one equation per variable. Use
    # Hopcroft-Karp bipartite matching to find the largest matching.
    # Variables that don't get matched are "structurally redundant"
    # for this constraint system and must be fixed at their current
    # value.
    try:
        import networkx as nx
    except ImportError:
        info["matched"] = None
        info["fixed_unmatched"] = []
    else:
        G = nx.Graph()
        for c in model.component_objects(Constraint, active=True):
            for idx in c:
                eq_node = ("EQ", c.name, idx)
                G.add_node(eq_node, bipartite=0)
                for v in identify_variables(c[idx].body, include_fixed=False):
                    var_node = ("VAR", v.parent_component().name, v.index())
                    G.add_node(var_node, bipartite=1)
                    G.add_edge(eq_node, var_node)

        eq_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}
        var_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 1}
        match = nx.bipartite.maximum_matching(G, top_nodes=eq_nodes)
        matched_vars = {n for n in match if n in var_nodes}
        unmatched_vars = var_nodes - matched_vars

        unmatched_by_family: Dict[str, int] = {}
        for _, var_name, idx in unmatched_vars:
            var = getattr(model, var_name)
            v = var[idx]
            if not v.fixed:
                v.fix(v.value or 0.0)
                unmatched_by_family[var_name] = unmatched_by_family.get(var_name, 0) + 1

        info["matched"] = len(matched_vars)
        info["fixed_unmatched"] = sorted(
            unmatched_by_family.items(), key=lambda x: -x[1]
        )

        # Deactivate equations whose body has no free variables anymore.
        # This includes both trivially-satisfied identities and equations
        # that became redundant after the unmatched-fix.
        deactivated2: Dict[str, int] = {}
        for c in model.component_objects(Constraint, active=True):
            for idx in list(c):
                free_in_body = list(
                    identify_variables(c[idx].body, include_fixed=False)
                )
                if not free_in_body:
                    c[idx].deactivate()
                    deactivated2[c.name] = deactivated2.get(c.name, 0) + 1
        info["deactivated_after_match"] = sorted(
            deactivated2.items(), key=lambda x: -x[1]
        )

        # Re-do matching to find equations that are now over-constraining
        # (more eqs than vars). Deactivate the unmatched eqs.
        G2 = nx.Graph()
        for c in model.component_objects(Constraint, active=True):
            for idx in c:
                eq_node = ("EQ", c.name, idx)
                G2.add_node(eq_node, bipartite=0)
                for v in identify_variables(c[idx].body, include_fixed=False):
                    var_node = ("VAR", v.parent_component().name, v.index())
                    G2.add_node(var_node, bipartite=1)
                    G2.add_edge(eq_node, var_node)
        eq_nodes2 = {n for n, d in G2.nodes(data=True) if d.get("bipartite") == 0}
        var_nodes2 = {n for n, d in G2.nodes(data=True) if d.get("bipartite") == 1}
        match2 = nx.bipartite.maximum_matching(G2, top_nodes=eq_nodes2)
        matched_eqs = {n for n in match2 if n in eq_nodes2}
        unmatched_eqs = eq_nodes2 - matched_eqs

        # Deactivate redundant eqs
        red_by_family: Dict[str, int] = {}
        for _, eq_name, idx in unmatched_eqs:
            eq = getattr(model, eq_name)
            if eq[idx].active:
                eq[idx].deactivate()
                red_by_family[eq_name] = red_by_family.get(eq_name, 0) + 1
        info["deactivated_redundant_eqs"] = sorted(
            red_by_family.items(), key=lambda x: -x[1]
        )

    # ----- final counts (respecting cell-level deactivation) -----------
    free = sum(1 for v in model.component_objects(Var, active=True)
               for idx in v if not v[idx].fixed)
    # Pyomo's ``component_objects(Constraint, active=True)`` returns
    # parent families, not cells; iterate cells and check ``active``
    # at the cell level.
    cons = 0
    for c in model.component_objects(Constraint, active=True):
        for idx in c:
            if c[idx].active:
                cons += 1
    info["free_vars"] = free
    info["active_cons"] = cons
    info["mismatch"] = free - cons

    return info
