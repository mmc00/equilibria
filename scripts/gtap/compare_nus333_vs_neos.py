"""NUS333 baseline + 10% import-tariff power shock — compare vs NEOS GAMS.

Mirrors comp_nus333.gms shock (imptx_new = (1+imptx)*1.10 - 1, gated by xwFlag).
Reports gdpmp, regy, u deltas alongside NEOS reference (job 18744693).

Run:
    uv run --with harpy3 python scripts/gtap/compare_nus333_vs_neos.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/marmol/proyectos/path-capi-python/src")
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from pyomo.environ import Var, Constraint, value
from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_solver import GTAPSolver
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from path_capi_python import PATHLoader, PyomoMCPAdapter, solve_nonlinear_mcp


NUS333 = Path("/Users/marmol/Downloads/10284")
PATH_LIB = ROOT / ".cache/path_capi/libpath50.silicon.dylib"

# NEOS reference (job 18744693, comp_nus333.gms after tariff power shock).
NEOS_REF = {
    "gdpmp": {"USA": (14.0617801139, 14.7063426837), "ROW": (41.7695611026, 42.4668724204)},
    "regy":  {"USA": (12.8018441072, 13.3823164142), "ROW": (37.0999386636, 37.6635926896)},
    "u":     {"USA": (1.0,            1.0016493028),  "ROW": (1.0,           0.9918010965)},
}


def _structural_matching(constraints, free_vars, forced_pairs=None):
    """Hopcroft-Karp maximum bipartite matching: eq row → var col.

    Adapter pairs F[i] with var[i] positionally. Without a structural matching,
    alphabetical sort can pair an equation with an unrelated spectator var that
    sits at its lower bound, allowing PATH to terminate "feasible" with a large
    F[i] residual. Returns var permutation so var[i] is structurally tied to eq i.

    forced_pairs: optional list of (eq_name, var_name) tuples to pin upfront
    (mirrors GAMS `model gtap / eq.var, ... /` declared matching).
    """
    from collections import deque
    from pyomo.core.expr.visitor import identify_variables

    n = len(constraints)
    var_id_to_col = {id(v): j for j, v in enumerate(free_vars)}
    adjacency: list[list[int]] = []
    for con in constraints:
        cols = []
        seen = set()
        for var_data in identify_variables(con.body, include_fixed=False):
            if var_data.fixed:
                continue
            col = var_id_to_col.get(id(var_data))
            if col is None or col in seen:
                continue
            seen.add(col)
            cols.append(col)
        adjacency.append(cols)

    # Diagnostic: identify spectator vars (free vars not referenced in any active eq).
    referenced = set()
    for cols in adjacency:
        referenced.update(cols)
    spectators = [free_vars[j].name for j in range(n) if j not in referenced]
    if spectators:
        print(f"[matching] {len(spectators)} spectator vars (no active eq mentions them):")
        for name in spectators[:30]:
            print(f"    {name}")
        if len(spectators) > 30:
            print(f"    ... +{len(spectators)-30} more")

    pair_left = [-1] * n   # eq row → var col
    pair_right = [-1] * n  # var col → eq row
    distance = [0] * n
    INF = 10 ** 9

    # Apply forced pairings BEFORE Hopcroft-Karp (mirrors GAMS declared matching).
    eq_name_to_row = {c.name: i for i, c in enumerate(constraints)}
    var_name_to_col = {v.name: j for j, v in enumerate(free_vars)}
    if forced_pairs:
        for eq_name, var_name in forced_pairs:
            r = eq_name_to_row.get(eq_name)
            c = var_name_to_col.get(var_name)
            if r is None or c is None:
                print(f"[matching] WARN: forced pair {eq_name}↔{var_name} not found")
                continue
            if c not in adjacency[r]:
                print(f"[matching] WARN: forced pair {eq_name}↔{var_name} not in adjacency")
                continue
            pair_left[r] = c
            pair_right[c] = r
            print(f"[matching] forced: {eq_name} ↔ {var_name}")

    def bfs():
        q = deque()
        found = False
        for u in range(n):
            if pair_left[u] == -1:
                distance[u] = 0
                q.append(u)
            else:
                distance[u] = INF
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                m = pair_right[v]
                if m == -1:
                    found = True
                elif distance[m] == INF:
                    distance[m] = distance[u] + 1
                    q.append(m)
        return found

    def dfs(u):
        for v in adjacency[u]:
            m = pair_right[v]
            if m == -1 or (distance[m] == distance[u] + 1 and dfs(m)):
                pair_left[u] = v
                pair_right[v] = u
                return True
        distance[u] = INF
        return False

    while bfs():
        for u in range(n):
            if pair_left[u] == -1:
                dfs(u)

    # Patch any unmatched equations with leftover vars (preserves DOF).
    leftover = [j for j, m in enumerate(pair_right) if m == -1]
    li = 0
    for u in range(n):
        if pair_left[u] == -1:
            pair_left[u] = leftover[li]
            li += 1

    perm = pair_left  # var col to put at position i (== row i)
    return [free_vars[c] for c in perm]


def _solve(model, params, *, label: str):
    from _closure_patches import apply_squareness_patches

    solver_helper = GTAPSolver(model, solver_name="path", params=params)
    solver_helper.apply_closure()
    solver_helper.apply_aggressive_fixing_for_mcp()

    apply_squareness_patches(model, params, label=label)

    constraints = sorted(
        model.component_data_objects(Constraint, active=True), key=lambda c: c.name
    )
    free_vars = sorted(
        (v for v in model.component_data_objects(Var, active=True) if not v.fixed),
        key=lambda v: v.name,
    )
    if len(free_vars) != len(constraints):
        print(f"[{label}] WARNING: DOF mismatch: {len(free_vars)} vars, {len(constraints)} eqs (continuing)")
    assert len(free_vars) == len(constraints), \
        f"degrees-of-freedom mismatch: {len(free_vars)} vars, {len(constraints)} eqs"

    # Reorder free_vars so position i is structurally matched to constraints[i].
    # GAMS-declared MCP pairings (model.gms:1419): force these to mirror GAMS.
    from _closure_patches import structural_matching as _sm
    free_vars = _sm(
        constraints, free_vars,
        forced_pairs=[("eq_pwfact", "pwfact")],
        label=label,
    )

    loader = PATHLoader(path_lib=PATH_LIB)
    runtime = loader.load()
    adapter = PyomoMCPAdapter()
    data = adapter.build_nonlinear_from_equality_constraints(
        model,
        constraints=constraints,
        variables=free_vars,
        jacobian_eval_mode="reverse_numeric",
    )

    # Equation scaling (row + col), as required by CLAUDE.md.
    jac_vals = data.callback_jac(list(data.x0))
    row_indices = data.jacobian_structure.row_indices
    col_for_nnz = []
    for j, (start, length) in enumerate(
        zip(data.jacobian_structure.col_starts, data.jacobian_structure.col_lengths)
    ):
        col_for_nnz.extend([j] * length)

    n_eq = len(data.variable_names)
    row_max = [0.0] * n_eq
    col_max = [0.0] * n_eq
    for k, row_idx in enumerate(row_indices):
        v = abs(jac_vals[k])
        if v > row_max[row_idx - 1]:
            row_max[row_idx - 1] = v
        if v > col_max[col_for_nnz[k]]:
            col_max[col_for_nnz[k]] = v

    _CAP = 1e6
    sr = [min(1.0 / max(v, 1e-12), _CAP) for v in row_max]
    sc = [min(1.0 / max(v, 1e-12), _CAP) for v in col_max]
    x0_s = [sc[j] * v for j, v in enumerate(data.x0)]
    # Force pure-equality MCP: relax all bounds so PATH cannot satisfy the
    # complementarity by parking a variable at its bound while leaving the
    # paired equation unsatisfied. Variable-equation pairing is positional
    # (sorted by name), so a bound-active solution can mask real violations.
    lb_s = [sc[j] * v if v > -1e19 else v for j, v in enumerate(data.lb)]
    ub_s = [sc[j] * v if v < 1e19 else v for j, v in enumerate(data.ub)]

    _f0, _j0 = data.callback_f, data.callback_jac
    f_scaled = lambda x: [sr[i] * fi for i, fi in enumerate(_f0([x[j] / sc[j] for j in range(len(x))]))]
    jac_scaled = lambda x: [
        jv * sr[row_indices[k] - 1] / sc[col_for_nnz[k]]
        for k, jv in enumerate(_j0([x[j] / sc[j] for j in range(len(x))]))
    ]

    print(f"[{label}] Solving ({n_eq} eqs)...")
    result = solve_nonlinear_mcp(
        runtime,
        n=n_eq,
        lb=lb_s, ub=ub_s, x0=x0_s,
        callback_f=f_scaled, callback_jac=jac_scaled,
        jacobian_structure=data.jacobian_structure,
        output=False,
    )

    # Unscale solution back into model.
    x_orig = [result.x[j] / sc[j] for j in range(n_eq)]
    for j, var_name in enumerate(data.variable_names):
        free_vars[j].set_value(x_orig[j])

    print(f"[{label}] code={result.termination_code}  residual={result.residual:.3e}")

    # Find largest equation residuals at the solution.
    f_vals = data.callback_f(list(x_orig))
    abs_idx = sorted(range(n_eq), key=lambda i: -abs(f_vals[i]))
    # Inspect the largest residual equation in detail.
    if abs(f_vals[abs_idx[0]]) > 1e-3:
        from pyomo.environ import value as _v
        bigi = abs_idx[0]
        big_con = constraints[bigi]
        print(f"[{label}] LARGEST residual on {big_con.name}:")
        from pyomo.core.expr.visitor import identify_variables
        for var_data in identify_variables(big_con.body, include_fixed=True):
            try:
                print(f"    {var_data.name} = {_v(var_data):.6g} (fixed={var_data.fixed})")
            except Exception:
                pass

    print(f"[{label}] top-25 residuals:")
    for i in abs_idx[:25]:
        cname = constraints[i].name if i < len(constraints) else f"eq{i}"
        # Cross-check: directly evaluate constraint body via Pyomo.
        try:
            from pyomo.environ import value as _v2
            direct = _v2(constraints[i].body) - _v2(constraints[i].lower) if constraints[i].equality else None
        except Exception:
            direct = None
        # Get paired variable info (PATH MCP: pairs eqs to vars)
        var_i = free_vars[i] if i < len(free_vars) else None
        var_info = ""
        if var_i is not None:
            try:
                from pyomo.environ import value as _v3
                var_info = f"  paired_var={var_i.name}={_v3(var_i):.4g}  bounds=[{var_i.lb},{var_i.ub}]  sc={sc[i]:.2e}"
            except Exception:
                pass
        print(f"   {cname}: callback={f_vals[i]:+.4e}  sr={sr[i]:.2e}{var_info}")
    return result


def _dump_facty_decomp(model, label):
    """Show pf*xf/xscale per (r,f,a) and depreciation term for facty."""
    print(f"\n--- FACTY [{label}] ---")
    for r in model.r:
        rs = str(r)
        total_fac = 0.0
        per_f = {}
        for f in model.f:
            sub = 0.0
            for a in model.a:
                pf_v = float(value(model.pf[r, f, a]))
                xf_v = float(value(model.xf[r, f, a]))
                xs_v = float(value(model.xscale[r, a]))
                contrib = pf_v * xf_v / xs_v
                sub += contrib
            per_f[str(f)] = sub
            total_fac += sub
        depr = float(value(model.fdepr[r])) * float(value(model.pi[r])) * float(value(model.kstock[r]))
        facty_v = float(value(model.facty[r]))
        print(f"  {rs}: facty={facty_v:+.4f}  sum(pf*xf/xs)={total_fac:+.4f}  depr={depr:+.4f}  byF={per_f}")


def _dump_tax_streams(model, label):
    """Per-stream tax revenue + import volume to localize tariff-revenue bug."""
    print(f"\n--- TAX-STREAMS [{label}] ---")
    streams = ["pt", "ft", "fc", "pc", "gc", "ic", "et", "mt", "dt"]
    for r in model.r:
        rs = str(r)
        per = {}
        for gy in streams:
            try:
                per[gy] = float(value(model.ytax[r, gy]))
            except Exception:
                per[gy] = float("nan")
        # Import volume + average tariff
        m_total = 0.0
        mt_revenue = 0.0
        for rp in model.rp:
            if str(rp) == rs:
                continue
            for i in model.i:
                try:
                    pmcif_v = float(value(model.pmcif[rp, i, r]))
                    xw_v = float(value(model.xw[rp, i, r]))
                    m_total += pmcif_v * xw_v
                except Exception:
                    pass
        items = "  ".join(f"{k}={v:+.4f}" for k, v in per.items())
        print(f"  {rs}: ytaxTot={sum(per.values()):+.4f}  M(pmcif*xw)={m_total:+.4f}  byStream={items}")


def _dump_gdpmp_decomp(model, label):
    """Decompose gdpmp = absorption(C+G+I) + (X − M) per region."""
    print(f"\n--- GDPMP-DECOMP [{label}] ---")
    for r in model.r:
        rs = str(r)
        absorp = 0.0
        absorp_by_agent = {"hhd": 0.0, "gov": 0.0, "inv": 0.0}
        for i in model.i:
            for aa in ("hhd", "gov", "inv"):
                try:
                    pa_v = float(value(model.pa[r, i, aa]))
                    xaa_v = float(value(model.xaa[r, i, aa]))
                    absorp_by_agent[aa] += pa_v * xaa_v
                    absorp += pa_v * xaa_v
                except Exception:
                    pass
        exports = 0.0
        imports = 0.0
        for i in model.i:
            for rp in model.rp:
                if str(rp) == rs:
                    continue
                try:
                    pefob_v = float(value(model.pefob[r, i, rp]))
                    xw_exp = float(value(model.xw[r, i, rp]))
                    exports += pefob_v * xw_exp
                except Exception:
                    pass
                try:
                    pmcif_v = float(value(model.pmcif[rp, i, r]))
                    xw_imp = float(value(model.xw[rp, i, r]))
                    imports += pmcif_v * xw_imp
                except Exception:
                    pass
        gdp_var = float(value(model.gdpmp[r]))
        nx = exports - imports
        print(f"  {rs}: gdpmp={gdp_var:+.4f}  absorption={absorp:+.4f} "
              f"(hhd={absorp_by_agent['hhd']:.3f} gov={absorp_by_agent['gov']:.3f} "
              f"inv={absorp_by_agent['inv']:.3f})  X={exports:+.4f}  M={imports:+.4f}  NX={nx:+.4f}")


def _dump_price_chain(model, label):
    """Print pmcif → pm → pmt → pmp → pa price chain to localize tariff propagation break."""
    print(f"\n--- PRICE-CHAIN [{label}] ---")
    for r in model.r:
        rs = str(r)
        for i in model.i:
            isi = str(i)
            try:
                pd_v = float(value(model.pd[r, i]))
            except Exception:
                pd_v = float("nan")
            try:
                pmt_v = float(value(model.pmt[r, i]))
            except Exception:
                pmt_v = float("nan")
            pm_pmcif = []
            for rp in model.rp:
                if str(rp) == rs:
                    continue
                try:
                    pmcif_v = float(value(model.pmcif[rp, i, r]))
                    pm_v = float(value(model.pm[rp, i, r]))
                    pm_pmcif.append(f"{rp}->{rs}:pmcif={pmcif_v:.4f},pm={pm_v:.4f}")
                except Exception:
                    pass
            pa_per_aa = []
            for aa in ("hhd", "gov", "inv"):
                try:
                    pa_v = float(value(model.pa[r, i, aa]))
                    pdp_v = float(value(model.pdp[r, i, aa]))
                    pmp_v = float(value(model.pmp[r, i, aa]))
                    pa_per_aa.append(f"{aa}:pa={pa_v:.4f},pdp={pdp_v:.4f},pmp={pmp_v:.4f}")
                except Exception:
                    pass
            for a in model.a:
                try:
                    pa_v = float(value(model.pa[r, i, a]))
                    pa_per_aa.append(f"{a}:pa={pa_v:.4f}")
                except Exception:
                    pass
            print(f"  {rs}/{isi}: pd={pd_v:.4f}  pmt={pmt_v:.4f}  | {' '.join(pm_pmcif)}")
            print(f"      agents: {' '.join(pa_per_aa)}")


def _dump_diagnostics(model, label):
    """Print savf/chif/regy/yi/xiagg/kstock per region for diagnosis."""
    print(f"\n--- DIAG [{label}] ---")
    for r in model.r:
        rs = str(r)
        vals = {}
        for v in ("savf", "chif", "regy", "yi", "xiagg", "kstock", "rsav", "facty",
                 "ytax_ind", "pi", "depr", "pigbl", "pnum", "pfact", "pwfact", "pgdpmp"):
            try:
                comp = getattr(model, v, None)
                if comp is None:
                    continue
                if v in ("pigbl", "pnum", "pwfact"):
                    vals[v] = float(value(comp))
                else:
                    vals[v] = float(value(comp[r]))
            except Exception:
                pass
        items = "  ".join(f"{k}={v_:+.4g}" for k, v_ in vals.items())
        print(f"  {rs}: {items}")


def _extract_key(model, params):
    """Compute gdpmp, regy, u per region — same convention as NEOS GAMS comp.gms."""
    out = {"gdpmp": {}, "regy": {}, "u": {}}
    GTAP_HH, GTAP_GOV, GTAP_INV = "hhd", "gov", "inv"

    def _ratio(num, den):
        return float(num) / float(den) if abs(den) > 1e-14 else 1.0

    for r in model.r:
        rs = str(r)
        regy_val = float(value(model.regy[r])) if hasattr(model, "regy") else 0.0
        yc_val = float(value(model.yc[r])) if hasattr(model, "yc") else 0.0
        yg_val = float(value(model.yg[r])) if hasattr(model, "yg") else 0.0

        # Prefer the model's own gdpmp variable when available (matches GAMS).
        if hasattr(model, "gdpmp"):
            try:
                out["gdpmp"][rs] = float(value(model.gdpmp[r]))
                out["regy"][rs] = regy_val
                try:
                    u_val = float(value(model.u[r])) if hasattr(model, "u") else (yc_val + yg_val)
                except (KeyError, ValueError):
                    u_val = yc_val + yg_val
                out["u"][rs] = u_val
                continue
            except (KeyError, ValueError):
                pass

        # gdpmp: absorption (xaa*pa) + net exports
        gdp = 0.0
        if hasattr(model, "xaa") and hasattr(model, "pa"):
            for i in model.i:
                for aa in (GTAP_HH, GTAP_GOV, GTAP_INV):
                    try:
                        gdp += float(value(model.pa[r, i, aa])) * float(value(model.xaa[r, i, aa]))
                    except (KeyError, ValueError):
                        pass
        if hasattr(model, "xw") and hasattr(model, "pe"):
            for i in model.i:
                for rp in model.rp:
                    if str(rp) == rs:
                        continue
                    vxsb_e = params.benchmark.vxsb.get((rs, str(i), str(rp)), 0.0)
                    vfob_e = params.benchmark.vfob.get((rs, str(i), str(rp)), 0.0)
                    pefob_ratio = _ratio(vfob_e, vxsb_e) if vxsb_e > 0 else 1.0
                    pe_e = float(value(model.pe[r, i, rp])) if (rs, str(i), str(rp)) in [(rs, str(i), str(rp))] else 1.0
                    try:
                        pe_e = float(value(model.pe[r, i, rp]))
                    except (KeyError, ValueError):
                        pe_e = 1.0
                    try:
                        xw_out = float(value(model.xw[r, i, rp]))
                    except (KeyError, ValueError):
                        xw_out = 0.0
                    gdp += pe_e * pefob_ratio * xw_out

                    vxsb_i = params.benchmark.vxsb.get((str(rp), str(i), rs), 0.0)
                    vcif_i = params.benchmark.vcif.get((str(rp), str(i), rs), 0.0)
                    pmcif_ratio = _ratio(vcif_i, vxsb_i) if vxsb_i > 0 else 1.0
                    try:
                        pe_i = float(value(model.pe[rp, i, r]))
                    except (KeyError, ValueError):
                        pe_i = 1.0
                    try:
                        xw_in = float(value(model.xw[rp, i, r]))
                    except (KeyError, ValueError):
                        xw_in = 0.0
                    gdp -= pe_i * pmcif_ratio * xw_in

        out["gdpmp"][rs] = gdp
        out["regy"][rs] = regy_val
        # GAMS u(r) — utility index. Approximate as yc/yc_base when available.
        try:
            u_val = float(value(model.u[r])) if hasattr(model, "u") else (yc_val + yg_val)
        except (KeyError, ValueError):
            u_val = yc_val + yg_val
        out["u"][rs] = u_val

    return out


def _apply_tariff_shock(params, factor: float = 1.10):
    """imptx_new = (1+imptx)*factor - 1 for ALL xwFlag pairs.

    GAMS comp_nus333.gms:148 applies the shock to imptx(r,i,rp) for every
    pair with xwFlag(r,i,rp), INCLUDING the diagonal (intra-region trade
    e.g. ROW->ROW which represents aggregated within-ROW trade).
    """
    imptx = params.taxes.imptx
    n = 0
    for key in list(imptx.keys()):
        cur = float(imptx[key])
        imptx[key] = (1.0 + cur) * factor - 1.0
        # Mirror to rtms if it shadows imptx.
        rtms = getattr(params.taxes, "rtms", None)
        if isinstance(rtms, dict) and key in rtms:
            rtms[key] = imptx[key]
        n += 1
    print(f"Applied 10% tariff power shock to {n} imptx entries")


def _copy_var_levels(src_model, dst_model) -> int:
    """Mirror GAMS iterloop.gms: copy .l of each Var from src to dst.

    Skips fixed vars (their value is preserved by .fix() on dst). Returns
    number of (var, index) pairs copied.
    """
    from pyomo.environ import Var
    src_vars = {v.local_name: v for v in src_model.component_objects(Var, active=True)}
    n = 0
    for dv in dst_model.component_objects(Var, active=True):
        sv = src_vars.get(dv.local_name)
        if sv is None:
            continue
        for idx in dv:
            try:
                src_val = value(sv[idx])
            except (KeyError, ValueError):
                continue
            if src_val is None:
                continue
            try:
                dvi = dv[idx]
            except KeyError:
                continue
            if dvi.fixed:
                continue
            dvi.set_value(float(src_val))
            n += 1
    return n


def main():
    params = GTAPParameters()
    params.load_from_har(
        basedata_path=NUS333 / "basedata.har",
        sets_path=NUS333 / "sets.har",
        default_path=NUS333 / "default.prm",
        baserate_path=NUS333 / "baserate.har",
    )
    print(f"Sets: r={params.sets.r}, i={params.sets.i}, f={params.sets.f}")

    # NUS333: residual region must be ROW (matches comp_nus333.gms `set rres /ROW/`).
    # comp_nus333.gms uses ifSUB=0 → use explicit price equations, not macros.
    closure = GTAPClosureConfig(if_sub=False)

    # ---- BASE (t=base) ----
    builder_b = GTAPModelEquations(params.sets, params, residual_region="ROW", closure=closure)
    model_b = builder_b.build_model()
    _solve(model_b, params, label="base")
    _dump_diagnostics(model_b, "base")
    _dump_gdpmp_decomp(model_b, "base")
    _dump_facty_decomp(model_b, "base")
    _dump_tax_streams(model_b, "base")
    base = _extract_key(model_b, params)

    # ---- CHECK (t=check) — same params, warm-start from base ----
    # GAMS iterloop.gms copies .l from previous tsim into current tsim before
    # the solve. The check iteration recalibrates and stabilizes the state
    # before the shock; it should yield ≈ base.
    builder_c = GTAPModelEquations(params.sets, params, residual_region="ROW", closure=closure)
    model_c = builder_c.build_model()
    n_copied = _copy_var_levels(model_b, model_c)
    print(f"[check] warm-start: copied {n_copied} (var,index) levels from base")
    _solve(model_c, params, label="check")
    check = _extract_key(model_c, params)

    # ---- SHOCK (t=shock) — apply shock, warm-start from check ----
    _apply_tariff_shock(params, factor=1.10)
    builder_s = GTAPModelEquations(
        params.sets, params, residual_region="ROW", closure=closure,
        t0_snapshot=model_b,
    )
    model_s = builder_s.build_model()
    n_copied = _copy_var_levels(model_c, model_s)
    print(f"[shock] warm-start: copied {n_copied} (var,index) levels from check")

    # Branch-flip experiment: optionally pin pe[USA, MFG, ROW] to its GAMS
    # shock value to break the price-level symmetry.
    import os
    pin_pe = os.environ.get("NUS333_PIN_PE_USA_MFG", "")
    if pin_pe:
        val = float(pin_pe)
        if hasattr(model_s, "pe"):
            try:
                model_s.pe["USA", "MFG", "ROW"].fix(val)
                print(f"[shock] pinned pe[USA,MFG,ROW] = {val} (GAMS anchor)")
            except KeyError:
                pass
    # Anti-deflation: lower-bound pgdpmp[USA] ≥ 1.0 to exclude the deflation
    # branch (PATH otherwise returns there from any initialization tested).
    pgdpmp_lo = os.environ.get("NUS333_PGDPMP_USA_LO", "")
    if pgdpmp_lo and hasattr(model_s, "pgdpmp"):
        v = float(pgdpmp_lo)
        try:
            model_s.pgdpmp["USA"].setlb(v)
            print(f"[shock] set pgdpmp[USA].lb = {v}")
        except KeyError:
            pass
    bump = float(os.environ.get("NUS333_PRICE_BUMP", "0.0"))
    if bump != 0.0:
        from pyomo.environ import Var
        price_prefixes = ("pa", "pd", "pm", "pmt", "pmcif", "pe", "pefob", "ps",
                          "pf", "pft", "pfa", "pfy", "pi", "pdp", "pmp",
                          "pgdpmp", "pabs", "pfact", "pmuv", "psave", "pigbl",
                          "pva", "pc", "pg")
        n_bumped = 0
        for v in model_s.component_objects(Var, active=True):
            name = v.local_name
            if name in ("pwfact", "pnum"):
                continue
            if not any(name == p or name.startswith(p + "[") or name == p
                       for p in price_prefixes):
                # exact-name match path
                if name not in price_prefixes:
                    continue
            for idx in v:
                vi = v[idx]
                if vi.fixed:
                    continue
                cur = value(vi)
                if cur is None or cur <= 0:
                    continue
                vi.set_value(cur * (1.0 + bump))
                n_bumped += 1
        print(f"[shock] price-bump +{bump*100:.1f}%: nudged {n_bumped} price vars")

    _solve(model_s, params, label="shock")
    _dump_diagnostics(model_s, "shock")
    _dump_price_chain(model_s, "shock")
    _dump_gdpmp_decomp(model_s, "shock")
    _dump_facty_decomp(model_s, "shock")
    _dump_tax_streams(model_s, "shock")
    shock = _extract_key(model_s, params)

    # ---- CHECK SANITY: check should ≈ base ----
    print("\n--- CHECK vs BASE (sanity: should be ≈ identical) ---")
    for var in ("gdpmp", "regy", "u"):
        for r in params.sets.r:
            b = base[var].get(r, float("nan"))
            c = check[var].get(r, float("nan"))
            d = (c / b - 1.0) * 100 if b else float("nan")
            print(f"  {var:<6} {r:<6} base={b:>10.4f}  check={c:>10.4f}  Δ%={d:+.4f}")

    # ---- REPORT ----
    print("\n" + "=" * 78)
    print("NUS333 — Python equilibria vs NEOS GAMS (job 18744693)")
    print("=" * 78)
    header = f"{'var':<8}{'r':<6}{'py_base':>12}{'py_shock':>12}{'py_Δ%':>9} | "\
             f"{'gams_base':>12}{'gams_shock':>12}{'gams_Δ%':>9}{'  diff_Δ%':>10}"
    print(header)
    print("-" * len(header))
    for var in ("gdpmp", "regy", "u"):
        for r in params.sets.r:
            py_b = base[var].get(r, float("nan"))
            py_s = shock[var].get(r, float("nan"))
            py_d = (py_s / py_b - 1.0) * 100 if py_b else float("nan")
            ref_b, ref_s = NEOS_REF[var][r]
            ref_d = (ref_s / ref_b - 1.0) * 100 if ref_b else float("nan")
            diff = py_d - ref_d
            print(
                f"{var:<8}{r:<6}{py_b:>12.4f}{py_s:>12.4f}{py_d:>+8.3f}% | "
                f"{ref_b:>12.4f}{ref_s:>12.4f}{ref_d:>+8.3f}%{diff:>+9.3f}%"
            )
    print("=" * 78)


if __name__ == "__main__":
    main()
