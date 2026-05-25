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


def apply_v62_conditional_fixing(
    model: Any,
    params: Any,
    *,
    tol: float = 1.0e-9,
) -> Dict[str, Any]:
    """Phase 3.28: data-driven fixing of variables whose benchmark flow is 0.

    Ports the ``apply_conditional_fixing`` logic from
    ``templates/gtap/gtap_solver.py:341``. Looking at the benchmark V*M /
    V*A values, fix variables associated with inactive routes:

    - Bilateral trade vars (qxs, pms, pmcif, pe, pwmg) when both VXMD
      and VXWD are 0 for that route.
    - Household / government / firm absorption vars (qpd, qpm, qgd,
      qgm, qfd, qfm) when their respective V*M flow is 0.
    - Factor demand qfe when VFM is 0.

    Removing structurally-irrelevant vars from the active MCP BEFORE
    bipartite matching gives PATH a smaller, denser Jacobian — analogue
    of v7's data-driven fixing that helps PATH converge.

    Returns dict with per-family fix counts.
    """
    b = params.benchmark
    fixed_by_family: Dict[str, int] = {}

    def _fix_idx(var_name: str, idx, fallback: float = 0.0):
        if not hasattr(model, var_name):
            return False
        var = getattr(model, var_name)
        if idx not in var:
            return False
        cell = var[idx]
        if cell.fixed:
            return False
        val = cell.value if cell.value is not None else fallback
        # Some vars have lower bound > 0; relax to 0 before fixing at 0.
        if val == 0.0 and cell.lb is not None and float(cell.lb) > 0.0:
            cell.setlb(0.0)
        cell.fix(float(val))
        fixed_by_family[var_name] = fixed_by_family.get(var_name, 0) + 1
        return True

    # 1. Bilateral trade — no flow on a route ⇒ fix all related vars.
    for i in model.i:
        for s in model.s:
            for d in model.rp:
                vxmd = float(b.vxmd.get((i, s, d), 0.0))
                vxwd = float(b.vxwd.get((i, s, d), 0.0))
                if vxmd <= tol and vxwd <= tol:
                    for vname in ("qxs", "pms", "pmcif", "pe", "pwmg"):
                        _fix_idx(vname, (i, s, d))

    # 2. Household absorption — no V*M ⇒ no demand for that good.
    for i in model.i:
        for r in model.r:
            if float(b.vdpm.get((i, r), 0.0)) <= tol:
                _fix_idx("qpd", (i, r))
            if float(b.vipm.get((i, r), 0.0)) <= tol:
                _fix_idx("qpm", (i, r))
            if float(b.vdgm.get((i, r), 0.0)) <= tol:
                _fix_idx("qgd", (i, r))
            if float(b.vigm.get((i, r), 0.0)) <= tol:
                _fix_idx("qgm", (i, r))

    # 3. Firm intermediate absorption.
    for i in model.i:
        for j in model.j:
            for r in model.r:
                if float(b.vdfm.get((i, j, r), 0.0)) <= tol:
                    _fix_idx("qfd", (i, j, r))
                if float(b.vifm.get((i, j, r), 0.0)) <= tol:
                    _fix_idx("qfm", (i, j, r))

    # 4. Factor demand — no VFM ⇒ factor not used by sector.
    # (Phase 3.18 already does this under "qfe_no_factor_use"; we
    # repeat here for completeness in case it's called standalone.)
    for f in model.f:
        for j in model.j:
            for r in model.r:
                if float(b.vfm.get((f, j, r), 0.0)) <= tol:
                    _fix_idx("qfe", (f, j, r))

    return {
        "fixed_by_family": fixed_by_family,
        "n_fixed_total": sum(fixed_by_family.values()),
    }


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

    # Track baked cells for later rebake at shocked state (Phase 3.32).
    if not hasattr(model, "_baked_cell_metadata"):
        model._baked_cell_metadata = []

    for c, idx, residual, body, lower, upper in to_bake:
        # Deactivate the original cell.
        c[idx].deactivate()

        # Replacement: body - residual_0 == rhs  (equivalent to body == rhs + residual_0)
        # ConstraintList.add returns the new Constraint cell (Pyomo); we
        # store it so rebake can locate it via the live reference.
        if lower is not None and upper is not None:
            rhs = value(upper)
            replacement = model.sam_baked_residuals.add(body - residual == rhs)
        elif upper is not None:
            rhs = value(upper)
            replacement = model.sam_baked_residuals.add(body - residual <= rhs)
        elif lower is not None:
            rhs = value(lower)
            replacement = model.sam_baked_residuals.add(body - residual >= rhs)
        else:
            replacement = model.sam_baked_residuals.add(body - residual == 0)

        model._baked_cell_metadata.append({
            "orig_con": c,
            "orig_idx": idx,
            "body": body,
            "lower": lower,
            "upper": upper,
            "replacement": replacement,  # live Constraint cell
        })

        info["n_baked"] += 1
        info["max_abs_baked"] = max(info["max_abs_baked"], abs(residual))
        eq_name = c.name
        info["by_eq"].setdefault(eq_name, []).append((str(idx), residual))

    return info


def rebake_residuals_at_current_state(
    model: Any,
    *,
    tolerance: float = 1.0e-3,
) -> Dict[str, Any]:
    """Phase 3.32: re-evaluate baked residuals at the CURRENT model state.

    The original prebalance bake captures residuals at the BASELINE x_0
    as constants. After a shock, the body expression evaluates to a new
    value but the baked residual_0 is stale → F jumps from 0 to ~676.

    Uses ``model._baked_cell_metadata`` (populated by
    ``bake_baseline_residuals_as_slacks``) to update IN PLACE, without
    re-activating originally-baked constraints (which preserves the
    closure square-ness).

    Returns dict with n_rebaked, max_abs, etc.
    """
    info = {"n_rebaked": 0, "max_abs": 0.0, "by_eq": {}, "n_skipped": 0}
    if not hasattr(model, "_baked_cell_metadata"):
        return info

    for meta in model._baked_cell_metadata:
        body = meta["body"]
        lower = meta["lower"]
        upper = meta["upper"]
        old_replacement = meta["replacement"]

        # Evaluate body at CURRENT state
        try:
            body_val = value(body)
        except Exception:
            info["n_skipped"] += 1
            continue

        if upper is not None:
            rhs = value(upper)
        elif lower is not None:
            rhs = value(lower)
        else:
            rhs = 0.0
        new_residual = body_val - rhs

        # Deactivate old replacement (regardless of whether new is trivial)
        if old_replacement.active:
            old_replacement.deactivate()

        if abs(new_residual) <= tolerance:
            info["n_skipped"] += 1
            continue

        # Add new replacement with updated residual
        if lower is not None and upper is not None:
            new_replacement = model.sam_baked_residuals.add(body - new_residual == rhs)
        elif upper is not None:
            new_replacement = model.sam_baked_residuals.add(body - new_residual <= rhs)
        elif lower is not None:
            new_replacement = model.sam_baked_residuals.add(body - new_residual >= rhs)
        else:
            new_replacement = model.sam_baked_residuals.add(body - new_residual == 0)

        meta["replacement"] = new_replacement
        info["n_rebaked"] += 1
        info["max_abs"] = max(info["max_abs"], abs(new_residual))
        eq_name = meta["orig_con"].name
        info["by_eq"].setdefault(eq_name, 0)
        info["by_eq"][eq_name] += 1

    return info


def apply_v62_pipeline(
    model: Any,
    *,
    mode: str = None,
    bake_tolerance: float = 1e-3,
    params: Any = None,
    conditional_fixing: bool = True,
    drop_dead_rows_threshold: float = 0.0,
) -> Dict[str, Any]:
    """End-to-end closure + (optional) prebalance for v6.2.

    Dispatches based on ``mode``:

    - ``"nlp"`` (IPOPT/CONOPT): apply canonical closure → square the
      system via bipartite matching → bake baseline residuals as
      constant slacks so ``F(x_0) = 0`` exactly. IPOPT's walras
      objective then absorbs any residual imbalance.

    - ``"mcp"`` (PATH): apply canonical closure → square via bipartite
      matching → **skip** the prebalance bake. PATH expects an MCP
      formulation where one equation per variable holds at the
      equilibrium; the baseline solve must converge from a near-
      feasible starting point. The model is built without ``walras`` in
      MCP mode (Walras' law makes one market clearing eq redundant
      automatically), so the bipartite matcher produces a square
      system.

    If ``mode`` is None it is read from ``model._mode`` (set by
    ``GTAPv62ModelEquations`` at build time), defaulting to "nlp".

    Returns a merged stats dict with closure and prebalance info.
    """
    if mode is None:
        mode = getattr(model, "_mode", "nlp")
    if mode not in ("nlp", "mcp"):
        raise ValueError(f"mode must be 'nlp' or 'mcp', got {mode!r}")

    # Phase 3.28: data-driven conditional fixing BEFORE closure
    # squaring. Fixes vars where the benchmark V*M flow is 0, mirroring
    # v7's apply_conditional_fixing. This removes structurally-
    # irrelevant vars from the active MCP, giving PATH a tighter
    # Jacobian.
    if conditional_fixing and params is not None:
        cond_info = apply_v62_conditional_fixing(model, params)
    else:
        cond_info = {"fixed_by_family": {}, "n_fixed_total": 0}

    closure_info = apply_v62_closure_and_square(model)
    closure_info["conditional_fixing"] = cond_info

    # Phase 3.30: optionally drop Jacobian rows with norm below threshold.
    # When threshold > 0, runs after bipartite matching to remove
    # structurally-redundant equations (e.g., BOOK3X3's
    # eq_qxs[svces, r, r] which has row_norm ≈ 2e-9 due to amgm[m,i,r,r]=0).
    # Each dropped row pair-deactivates one free variable to keep the
    # system square.
    drop_info: Dict[str, Any] = {"n_dropped": 0, "dropped": []}
    if drop_dead_rows_threshold > 0:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).parent))
        from diagnose_health import drop_dead_rows  # type: ignore
        drop_info = drop_dead_rows(
            model, params=params, threshold=drop_dead_rows_threshold,
        )
    closure_info["dead_rows_dropped"] = drop_info

    # Prebalance: bake non-zero baseline residuals as constant slacks.
    # In Phase 3.26 we keep this for BOTH modes by default, because the
    # SAM imperfections (eq_cgds_balance ~1M, eq_qtm intra-region VTWR
    # ~65K, eq_market ~25K) prevent F(x_0) = 0 from holding otherwise.
    # The bake is a structural offset that preserves derivatives in
    # both NLP and MCP formulations. Pass ``bake_tolerance=0`` to skip.
    if bake_tolerance > 0:
        prebal_info = bake_baseline_residuals_as_slacks(
            model, tolerance=bake_tolerance
        )
    else:
        prebal_info = {
            "n_baked": 0,
            "max_abs_baked": 0.0,
            "by_eq": {},
            "skipped_reason": "Prebalance skipped (bake_tolerance=0)",
        }

    return {
        "mode": mode,
        "closure": closure_info,
        "prebalance": prebal_info,
        "free_vars": closure_info["free_vars"],
        "active_cons": closure_info["active_cons"],
        "mismatch": closure_info["mismatch"],
    }


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

    # Phase 3.18: fix variables that are trivially zero/unit by SAM structure.
    # This removes them from the bipartite graph so the matching pairs the
    # remaining (genuinely endogenous) variables with their defining equations.
    n_trivial_pwmg = 0
    n_trivial_pva = 0
    n_trivial_qfe = 0

    # pwmg[i, s, d] is the per-unit transport margin. It's economically 0 when
    # (a) the commodity is itself a margin (svces) — services aren't shipped,
    # or (b) intra-region (s == d, no transport required), or
    # (c) the bilateral pwmg_0 calibration came out at 0 (no VTWR for this route).
    for i in model.i:
        for s in model.s:
            for d in model.rp:
                pwmg_var = model.pwmg[i, s, d]
                if pwmg_var.fixed:
                    continue
                pwmg0 = pwmg_var.value or 0.0
                # 1e-6 is the model-wide lower-bound floor; anything at
                # the floor is economically zero.
                if abs(pwmg0) <= 1e-5:
                    pwmg_var.fix(pwmg0)
                    n_trivial_pwmg += 1
    info["fixed_explicit"].append(("pwmg_trivial", n_trivial_pwmg))

    # pva[j, r]: VA price by sector. For CGDS (capital goods) production uses
    # NO factor inputs (only intermediate goods), so pva[CGDS, r] doesn't have
    # an economic role. Fix at 1.0 so eq_qo can still reference it without
    # leaving it dangling.
    for j in model.j:
        for r in model.r:
            sva = float(value(model.share_va[j, r])) if hasattr(model, "share_va") else 1.0
            if sva <= 1e-12 and not model.pva[j, r].fixed:
                model.pva[j, r].fix(1.0)
                n_trivial_pva += 1
    info["fixed_explicit"].append(("pva_no_VA", n_trivial_pva))

    # qfe[f, j, r]: factor demand by sector. 0 when VFM[f, j, r] = 0
    # (factor f not used by sector j in region r). For BOOK3X3:
    # - CGDS uses no factors (capital goods produced from intermediates).
    # - svces uses no Land (services don't use land in this aggregation).
    for f in model.f:
        for j in model.j:
            for r in model.r:
                qfe_var = model.qfe[f, j, r]
                if qfe_var.fixed:
                    continue
                qfe0 = qfe_var.value or 0.0
                if abs(qfe0) <= 1e-5:
                    qfe_var.fix(qfe0)
                    n_trivial_qfe += 1
    info["fixed_explicit"].append(("qfe_no_factor_use", n_trivial_qfe))

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
            return m.psave[r] == m.pcgds[next(iter(m.cgds)), r]
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

    # eq_qim: Phase 3.23 — MKTCLIMP (gtap.tab line 2413-2419).
    #
    # The CES cost identity pim*qim = sum_s pms*qxs is TRIVIALLY satisfied
    # by the bottom-Armington CES dual + eq_qxs, so it leaves qim
    # effectively unconstrained. GEMPACK enforces a separate market-
    # clearing equation linking qim to the SUM OF AGENT DEMANDS for the
    # imported composite (households, government, firms):
    #
    #   qim(i,r) = sum_j SHRIFM(i,j,r) * qfm(i,j,r)
    #            + SHRIPM(i,r)         * qpm(i,r)
    #            + SHRIGM(i,r)         * qgm(i,r)              (linearized)
    #
    # In LEVELS we use the budget-share-weighted ratio identity that
    # linearizes to this MKTCLIMP form at the benchmark:
    #
    #   qim = (1/pim_0) * [sum_j qfm + qpm + qgm]
    #
    # The 1/pim_0 factor reconciles the basic-price units of qim
    # (= sum_s VXWD) with the agent-price units of qpm/qgm/qfm
    # (= VIPM/VIGM/VIFM). At benchmark sum_agent = pim_0 * qim_0, so
    # the identity gives qim = qim_0 ✓. Differentiating gives the
    # share-weighted percent-change form of MKTCLIMP.
    if not hasattr(model, "eq_qim"):
        def eq_qim_rule(m, i, r):
            pim_0_val = float(value(m.pim_0[i, r])) if hasattr(m, "pim_0") else 1.0
            if pim_0_val <= 1e-12:
                return Constraint.Skip
            return m.qim[i, r] == (1.0 / pim_0_val) * (
                sum(m.qfm[i, j, r] for j in m.j)
                + m.qpm[i, r] + m.qgm[i, r]
            )
        model.eq_qim = Constraint(model.i, model.r, rule=eq_qim_rule)
        info["added_identity_eqs"].append(("eq_qim", 9))

    # Phase 3.18: eq_cgds_balance defines qo[CGDS, r] via regional
    # savings-investment identity. The constant residual at benchmark
    # (VDEP + DPGOV — components not modelled explicitly in v6.2 static)
    # is absorbed by bake_baseline_residuals_as_slacks, leaving the
    # derivatives intact so shock propagation is correct.
    if not hasattr(model, "eq_cgds_balance"):
        cgds_label = next(iter(model.cgds))

        def eq_cgds_balance_rule(m, r):
            return (
                m.pcgds[cgds_label, r] * m.qo[cgds_label, r]
                == m.y[r] - m.yp[r] - m.yg[r] + m.savf[r]
            )
        model.eq_cgds_balance = Constraint(model.r, rule=eq_cgds_balance_rule)
        info["added_identity_eqs"].append(("eq_cgds_balance", 3))

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
