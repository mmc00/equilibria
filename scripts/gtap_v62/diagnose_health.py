"""Phase 3.29 — Health diagnostic for the v6.2 model.

Two-stage diagnosis to flag conditions that cause PATH to fail with
``term_code=2`` (near-singular Jacobian / merit-function trap):

  Stage 1 — DATA-LEVEL (cheap, run after loading SAM):
    * Zero-flow trade routes vs. total routes
    * Very small (near-zero) flows that conditional_fixing won't catch
    * Intra-region (diagonal) trade volume vs. inter-region
    * Margin commodity diagonal share

  Stage 2 — JACOBIAN-LEVEL (medium-cost, after build + closure):
    * Row 2-norm: identifies equations with very weak coupling
    * Column 2-norm: identifies variables not effectively constrained
    * Diagonal magnitude: identifies trivially-coupled (eq, var) pairs
    * Reports the top-K weakest rows/columns mapped back to
      (equation_name, index) / (var_name, index) — these are the
      precise (commodity, region) cells responsible for the
      near-singularity.

Both stages emit findings dict with severity tags. Stage 2 requires
the Pyomo model to be built and closure applied; you call it before
the PATH solve, so the helpful diagnostics print BEFORE PATH would
hit term_code=2.

Usage::

    from scripts.gtap_v62.diagnose_health import (
        check_dataset_for_path,
        check_jacobian_singularity,
    )

    data_health = check_dataset_for_path(sets, params)
    if data_health["severity"] == "error":
        print("PATH likely to fail; consider IPOPT or smaller aggregation")

    # ... build model, apply closure, then:
    jac_health = check_jacobian_singularity(model, params)
    for issue in jac_health["weakest_rows"][:5]:
        print(issue)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Stage 1 — Data-level diagnostic
# ----------------------------------------------------------------------

def check_dataset_for_path(sets: Any, params: Any) -> Dict[str, Any]:
    """Inspect SAM data for patterns that cause PATH convergence problems.

    Returns a dict with:
      - 'severity': 'ok' | 'warn' | 'error'
      - 'stats': summary numbers
      - 'findings': list of (severity, key, description) tuples
      - 'recommendation': short text

    Designed to be CHEAP (no model build) so it can run right after
    ``GTAPv62Parameters.load_from_har``.
    """
    b = params.benchmark
    n_i = len(sets.i)
    n_r = len(sets.r)
    n_routes = n_i * n_r * n_r
    findings: List[Tuple[str, str, str]] = []

    # 1. Trade-flow density
    vxmd = b.vxmd
    n_zero_routes = sum(
        1 for i in sets.i for s in sets.r for d in sets.r
        if abs(vxmd.get((i, s, d), 0.0)) <= 1e-9
    )
    pct_zero = 100.0 * n_zero_routes / max(n_routes, 1)
    n_nonzero = n_routes - n_zero_routes

    # 2. Near-zero routes (flow between 1e-9 and 1e-3 of mean)
    nonzero_vals = [
        abs(vxmd.get((i, s, d), 0.0))
        for i in sets.i for s in sets.r for d in sets.r
        if abs(vxmd.get((i, s, d), 0.0)) > 1e-9
    ]
    mean_flow = float(np.mean(nonzero_vals)) if nonzero_vals else 0.0
    near_zero_threshold = mean_flow * 1e-3
    n_near_zero = sum(1 for v in nonzero_vals if v < near_zero_threshold) if mean_flow > 0 else 0
    pct_near_zero = 100.0 * n_near_zero / max(n_nonzero, 1)

    # 3. Intra-region diagonal trade
    diag_flow = sum(
        vxmd.get((i, r, r), 0.0)
        for i in sets.i for r in sets.r
    )
    total_flow = sum(nonzero_vals)
    pct_diag = 100.0 * diag_flow / max(total_flow, 1e-9)

    # 4. Margin diagonal
    diag_margin = 0.0
    for m_lbl in sets.marg:
        for i in sets.i:
            for r in sets.r:
                diag_margin += b.vtwr.get((m_lbl, i, r, r), 0.0)
    total_margin = sum(
        b.vtwr.get((m_lbl, i, s, d), 0.0)
        for m_lbl in sets.marg for i in sets.i for s in sets.r for d in sets.r
    )
    pct_diag_margin = 100.0 * diag_margin / max(total_margin, 1e-9)

    # 5. Region trade-imbalance check (each region: trade in vs out)
    region_imbalance = {}
    for r in sets.r:
        exports = sum(vxmd.get((i, r, d), 0.0) for i in sets.i for d in sets.r if d != r)
        imports = sum(vxmd.get((i, s, r), 0.0) for i in sets.i for s in sets.r if s != r)
        if max(exports, imports) > 0:
            ratio = min(exports, imports) / max(exports, imports)
            region_imbalance[r] = ratio
        else:
            region_imbalance[r] = float("nan")
            findings.append(("error", f"isolated_region:{r}",
                f"Region '{r}' has no trade activity (exports=0, imports=0). "
                f"Will create dead rows in Jacobian."))

    # Severity rules
    if pct_zero > 50.0:
        findings.append(("warn", "zero_routes_dominant",
            f"{pct_zero:.1f}% of trade routes are zero. Conditional fixing will "
            f"remove {n_zero_routes * 5} vars but the model may still have "
            f"near-singular Jacobian from boundary effects."))

    if pct_near_zero > 20.0:
        findings.append(("warn", "many_near_zero_flows",
            f"{n_near_zero}/{n_nonzero} ({pct_near_zero:.1f}%) of NON-zero "
            f"flows are less than 0.1% of mean. These produce near-zero "
            f"Jacobian rows that cause PATH term_code=2."))

    if pct_diag > 30.0:
        findings.append(("warn", "high_intra_region",
            f"{pct_diag:.1f}% of trade is intra-region (diagonal). This is "
            f"typical for aggregations grouping countries (e.g., EU_28). "
            f"GTAPinGAMS handles via amgm[m,i,r,r]=0; ensure prebalance bake "
            f"of eq_qtm is active."))

    if n_r >= 10:
        findings.append(("warn", "large_region_count",
            f"{n_r} regions × {n_r} destinations = {n_r*n_r} bilateral cells "
            f"per commodity. PATH per-iter cost scales as O(n^2.5) — expect "
            f">10 min per Newton iter at this size."))

    # Compute severity
    if any(f[0] == "error" for f in findings):
        severity = "error"
    elif any(f[0] == "warn" for f in findings):
        severity = "warn"
    else:
        severity = "ok"

    stats = {
        "n_commodities": n_i, "n_regions": n_r,
        "n_trade_routes": n_routes,
        "n_zero_routes": n_zero_routes,
        "pct_zero_routes": pct_zero,
        "n_near_zero_routes": n_near_zero,
        "pct_near_zero_routes": pct_near_zero,
        "pct_intra_region_trade": pct_diag,
        "pct_intra_region_margin": pct_diag_margin,
        "region_trade_balance_ratios": region_imbalance,
        "mean_nonzero_flow": mean_flow,
    }

    rec_lines = []
    if severity == "ok":
        rec_lines.append("Dataset health looks good for PATH.")
    elif severity == "warn":
        rec_lines.append("PATH may struggle on shocked solve due to near-singular Jacobian.")
        rec_lines.append("Recommended: use IPOPT (--solver ipopt) for shocks ≥ 5%.")
        rec_lines.append("If PATH required: implement Gragg-style multi-step homotopy.")
    else:  # error
        rec_lines.append("Dataset has structural issues that will break BOTH solvers.")
        rec_lines.append("Fix the underlying SAM imperfections before attempting any solve.")

    return {
        "stage": "data",
        "severity": severity,
        "stats": stats,
        "findings": findings,
        "recommendation": "\n".join(rec_lines),
    }


# ----------------------------------------------------------------------
# Stage 2 — Jacobian-level diagnostic
# ----------------------------------------------------------------------

def _build_sparse_jacobian(model: Any):
    """Build sparse CSC Jacobian via PATH adapter. Returns (J, free_var_data,
    active_cons_data)."""
    from pyomo.environ import Constraint, Var
    from scipy.sparse import csc_matrix

    sys.path.insert(0, str(Path(__file__).parent))
    from _path_capi_solver import _ensure_path_capi_on_syspath  # type: ignore
    _ensure_path_capi_on_syspath()
    from path_capi_python import PyomoMCPAdapter  # type: ignore

    free_var_data = [
        v[idx] for v in model.component_objects(Var, active=True)
        for idx in v if not v[idx].fixed
    ]
    active_cons_data = [
        c[idx] for c in model.component_objects(Constraint, active=True)
        for idx in c if c[idx].active
    ]

    adapter = PyomoMCPAdapter()
    data = adapter.build_nonlinear_from_equality_constraints(
        model, constraints=active_cons_data, variables=free_var_data,
        jacobian_eval_mode="reverse_numeric",
    )

    n = len(data.x0)
    jvals = list(data.callback_jac(list(data.x0)))
    js = data.jacobian_structure
    row_indices = np.asarray(list(js.row_indices), dtype=np.int64) - 1
    col_starts = np.asarray(list(js.col_starts), dtype=np.int64) - 1
    col_lengths = np.asarray(list(js.col_lengths), dtype=np.int64)
    jvals = np.asarray(jvals, dtype=np.float64)

    # Build CSC
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(col_lengths)
    ri = np.empty(int(indptr[-1]), dtype=np.int64)
    dv = np.empty(int(indptr[-1]), dtype=np.float64)
    cursor = 0
    for j in range(n):
        start = int(col_starts[j])
        ln = int(col_lengths[j])
        ri[cursor:cursor + ln] = row_indices[start:start + ln]
        dv[cursor:cursor + ln] = jvals[start:start + ln]
        cursor += ln
    J = csc_matrix((dv, ri, indptr), shape=(n, n))

    f0 = list(data.callback_f(list(data.x0)))
    return J, free_var_data, active_cons_data, np.array(f0)


def check_jacobian_singularity(
    model: Any,
    params: Any = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Diagnose Jacobian-level near-singularity.

    Computes column and row 2-norms; the k smallest norms point to the
    variables/equations responsible for the near-null space. Each cell
    is reported with its Pyomo (var_name, idx) / (eq_name, idx) so the
    user can identify the specific (commodity, region) cell.

    Returns dict:
      - 'sigma_max', 'sigma_min', 'condition_number' (estimated via norms)
      - 'weakest_rows': list of (eq_name, idx, row_2norm)
      - 'weakest_cols': list of (var_name, idx, col_2norm)
      - 'f_norm': ||F(x_0)||
      - 'severity': 'ok' | 'warn' | 'error'
      - 'findings', 'recommendation'
    """
    J, free_vars, active_cons, f0 = _build_sparse_jacobian(model)
    n = J.shape[0]

    # Row 2-norms (axis=1)
    row_norms = np.sqrt(np.asarray((J.multiply(J)).sum(axis=1)).flatten())
    col_norms = np.sqrt(np.asarray((J.multiply(J)).sum(axis=0)).flatten())

    # Estimated bounds for κ(J)
    sigma_max_est = float(row_norms.max())  # actually a weak upper bound
    sigma_min_est = float(row_norms.min())  # weak lower bound
    cond_est = sigma_max_est / max(sigma_min_est, 1e-300)

    f_norm = float(np.linalg.norm(f0))

    # k smallest row norms → weakest equations
    row_order = np.argsort(row_norms)
    weakest_rows = []
    for i in row_order[:top_k]:
        if i < len(active_cons):
            con = active_cons[i]
            parent = con.parent_component()
            weakest_rows.append({
                "eq_name": parent.name,
                "idx": str(con.index()),
                "row_2norm": float(row_norms[i]),
            })

    # k smallest col norms → weakest variables
    col_order = np.argsort(col_norms)
    weakest_cols = []
    for j in col_order[:top_k]:
        if j < len(free_vars):
            v = free_vars[j]
            parent = v.parent_component()
            weakest_cols.append({
                "var_name": parent.name,
                "idx": str(v.index()),
                "col_2norm": float(col_norms[j]),
                "value": float(v.value) if v.value is not None else None,
            })

    findings: List[Tuple[str, str, str]] = []

    if sigma_min_est < 1e-8:
        findings.append(("error", "near_singular",
            f"Smallest row norm = {sigma_min_est:.2e} (≈0). Jacobian is "
            f"effectively singular. PATH WILL FAIL with term_code=2 on "
            f"shocked solve."))
    elif sigma_min_est < 1e-4:
        findings.append(("warn", "weak_coupling",
            f"Smallest row norm = {sigma_min_est:.2e}. Marginal Jacobian "
            f"conditioning. PATH may converge baseline but struggle on shock."))

    if cond_est > 1e10:
        findings.append(("error", "ill_conditioned",
            f"κ(J) ≈ {cond_est:.2e} (way too high). Newton steps will be "
            f"unreliable. Switch to IPOPT or fix the source equations."))

    # Group weakest rows by equation family
    weak_eq_families: Dict[str, int] = {}
    for r in weakest_rows:
        weak_eq_families[r["eq_name"]] = weak_eq_families.get(r["eq_name"], 0) + 1

    if weak_eq_families:
        top_family = max(weak_eq_families, key=weak_eq_families.get)
        findings.append(("info", "weak_eq_family",
            f"Top problem equation family: {top_family} "
            f"({weak_eq_families[top_family]}/{top_k} weakest rows). "
            f"Consider auditing this equation's calibration."))

    if any(f[0] == "error" for f in findings):
        severity = "error"
    elif any(f[0] == "warn" for f in findings):
        severity = "warn"
    else:
        severity = "ok"

    rec_lines = []
    if severity == "error":
        rec_lines.append(
            f"Near-singular Jacobian (σ_min ≈ {sigma_min_est:.2e}). "
            f"Switch to IPOPT or rebuild closure to drop the dependent equations."
        )
        if weakest_rows:
            top = weakest_rows[0]
            rec_lines.append(
                f"  → most degenerate equation: {top['eq_name']}{top['idx']} "
                f"(row norm = {top['row_2norm']:.2e})"
            )
        if weakest_cols:
            top = weakest_cols[0]
            rec_lines.append(
                f"  → most degenerate variable: {top['var_name']}{top['idx']} "
                f"(col norm = {top['col_2norm']:.2e}, value = {top['value']})"
            )
    elif severity == "warn":
        rec_lines.append("Marginal conditioning — PATH baseline OK, shocked may fail.")
    else:
        rec_lines.append("Jacobian conditioning looks healthy for PATH.")

    return {
        "stage": "jacobian",
        "severity": severity,
        "n_vars": n,
        "n_nonzeros": int(J.nnz),
        "density": J.nnz / (n * n),
        "sigma_max_est": sigma_max_est,
        "sigma_min_est": sigma_min_est,
        "condition_number_est": cond_est,
        "f_norm": f_norm,
        "weakest_rows": weakest_rows,
        "weakest_cols": weakest_cols,
        "weak_eq_families": weak_eq_families,
        "findings": findings,
        "recommendation": "\n".join(rec_lines),
    }


# ----------------------------------------------------------------------
# Phase 3.30 — Auto-drop dead rows
# ----------------------------------------------------------------------

def drop_dead_rows(
    model: Any,
    params: Any = None,
    threshold: float = 1e-6,
) -> Dict[str, Any]:
    """Deactivate Jacobian rows with norm below threshold.

    Identifies structurally-redundant equations — those whose row in
    J(x_0) has 2-norm below ``threshold``, meaning the equation
    contributes essentially no usable Newton direction. For each such
    equation:

    1. Deactivate the constraint (removed from active set)
    2. Fix one of its free variables at its current value
       (so the system stays square)

    The mismatch (#free_vars - #active_cons) should remain unchanged
    because each drop removes one of each.

    Use this AFTER bipartite matching but BEFORE prebalance baking —
    no point in baking residuals on equations we're about to drop.

    Args:
        model: closed Pyomo model (bipartite-matched, mismatch=0)
        params: GTAPv62Parameters (used by helpers but optional here)
        threshold: row 2-norm cutoff. Default 1e-6 (six orders below 1).

    Returns:
        dict with:
          - n_dropped: number of (eq, var) pairs dropped
          - threshold: threshold used
          - dropped: list of {eq_name, idx, row_norm, var_fixed}
    """
    from pyomo.core.expr.visitor import identify_variables

    J, free_vars, active_cons, _ = _build_sparse_jacobian(model)
    row_norms = np.sqrt(np.asarray((J.multiply(J)).sum(axis=1)).flatten())

    # Sort dead rows by norm ascending so deterministic
    dead_idx = sorted(
        [i for i, n in enumerate(row_norms) if n < threshold and i < len(active_cons)],
        key=lambda i: row_norms[i],
    )

    dropped: List[Dict[str, Any]] = []
    for i in dead_idx:
        con = active_cons[i]
        if not con.active:
            continue

        # Find a free variable that appears in this constraint.
        var_to_fix = None
        for v in identify_variables(con.body, include_fixed=False):
            if not v.fixed:
                var_to_fix = v
                break

        # Deactivate the equation; fix one of its vars if any.
        con.deactivate()
        if var_to_fix is not None:
            val = var_to_fix.value if var_to_fix.value is not None else 0.0
            # If var has a positive lower bound but we want to fix at 0,
            # relax the bound first.
            if val == 0.0 and var_to_fix.lb is not None and float(var_to_fix.lb) > 0.0:
                var_to_fix.setlb(0.0)
            var_to_fix.fix(float(val))

        parent = con.parent_component()
        dropped.append({
            "eq_name": parent.name,
            "idx": str(con.index()),
            "row_norm": float(row_norms[i]),
            "var_fixed": (
                f"{var_to_fix.parent_component().name}{var_to_fix.index()}"
                if var_to_fix is not None else None
            ),
        })

    return {
        "n_dropped": len(dropped),
        "threshold": threshold,
        "dropped": dropped,
    }


# ----------------------------------------------------------------------
# Pretty printer
# ----------------------------------------------------------------------

def print_health_report(health: Dict[str, Any], indent: str = "  ") -> None:
    """Human-readable health report."""
    severity = health.get("severity", "ok")
    icon = {"ok": "✓", "warn": "⚠", "error": "✗"}.get(severity, "?")
    stage = health.get("stage", "unknown")
    print(f"\n{icon} v6.2 {stage}-level health: {severity.upper()}")

    if stage == "data":
        s = health["stats"]
        print(f"{indent}Commodities: {s['n_commodities']}  Regions: {s['n_regions']}")
        print(f"{indent}Trade routes: {s['n_trade_routes']} total, "
              f"{s['n_zero_routes']} zero ({s['pct_zero_routes']:.1f}%)")
        print(f"{indent}Near-zero non-zero routes: "
              f"{s['n_near_zero_routes']}/{s['n_trade_routes']-s['n_zero_routes']} "
              f"({s['pct_near_zero_routes']:.1f}%)")
        print(f"{indent}Intra-region trade share: {s['pct_intra_region_trade']:.1f}%")
        print(f"{indent}Intra-region margin share: {s['pct_intra_region_margin']:.1f}%")
    elif stage == "jacobian":
        print(f"{indent}n_vars: {health['n_vars']}  nnz: {health['n_nonzeros']}  "
              f"density: {health['density']:.4%}")
        print(f"{indent}σ_max ≈ {health['sigma_max_est']:.2e}  "
              f"σ_min ≈ {health['sigma_min_est']:.2e}  "
              f"κ ≈ {health['condition_number_est']:.2e}")
        print(f"{indent}‖F(x₀)‖ = {health['f_norm']:.2e}")
        if health["weakest_rows"]:
            print(f"{indent}Weakest equations (smallest row norm):")
            for r in health["weakest_rows"][:5]:
                print(f"{indent}  • {r['eq_name']}[{r['idx']}]  "
                      f"row_norm={r['row_2norm']:.2e}")
        if health["weakest_cols"]:
            print(f"{indent}Weakest variables (smallest col norm):")
            for v in health["weakest_cols"][:5]:
                val_str = f", val={v['value']:.2e}" if v['value'] is not None else ""
                print(f"{indent}  • {v['var_name']}[{v['idx']}]  "
                      f"col_norm={v['col_2norm']:.2e}{val_str}")

    if health["findings"]:
        print(f"{indent}Findings:")
        for sev, key, msg in health["findings"]:
            sev_icon = {"info": "ℹ", "warn": "⚠", "error": "✗"}.get(sev, "?")
            print(f"{indent}  {sev_icon} [{key}] {msg}")

    print(f"{indent}Recommendation:")
    for line in health.get("recommendation", "").splitlines():
        print(f"{indent}  {line}")
