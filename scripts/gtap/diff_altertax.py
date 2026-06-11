"""Cell-by-cell diff: Python altertax vs GAMS NEOS altertax out.gdx.

Mirrors GAMS comp_altertax.gms three-period structure:
  base   → Python baseline (standard GTAP, no shock)
  check  → Python altertax re-solve (CD elasticities, all factors mobile, no imptx shock)
  shock  → Python altertax shock (+10% imptx, warm-started from check)

Compares shock-period levels cell-by-cell against the NEOS altertax reference GDX
(Job #19466592, +10% tariff shock).

Usage:
    uv run python scripts/gtap/diff_altertax.py
    uv run python scripts/gtap/diff_altertax.py --gdx output/9x10_altertax_neos_bundle/out.gdx
    uv run python scripts/gtap/diff_altertax.py --show-worst --tol-rel 1e-3
"""
from __future__ import annotations
import argparse, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import (
    list_populated_vars, gams_levels, find_py_var, compare_phase,
    diff_phase_rows, write_csv, git_short_sha, build_derived,
)

GDX_9X10 = ROOT / "src/equilibria/templates/reference/gtap/data/basedata-9x10.gdx"
DEFAULT_NEOS_GDX = ROOT / "output/9x10_altertax_neos_bundle/out.gdx"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdx", type=Path, default=DEFAULT_NEOS_GDX,
                    help="GAMS altertax reference GDX (default: NEOS job #19466592)")
    ap.add_argument("--tol-rel", type=float, default=1e-3)
    ap.add_argument("--tol-abs", type=float, default=1e-6)
    ap.add_argument("--show-worst", action="store_true",
                    help="Print the worst diverging cell for each variable")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    from equilibria.templates.gtap import (
        GTAPParameters, GTAPModelEquations,
    )
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot

    import importlib.util as _u
    spec = _u.spec_from_file_location(
        "run_gtap", str(ROOT / "scripts" / "gtap" / "run_gtap.py")
    )
    run_gtap = _u.module_from_spec(spec)
    sys.modules["run_gtap"] = run_gtap
    spec.loader.exec_module(run_gtap)

    # Standard closure: if_sub=False (match GAMS ifSUB=0 reference)
    contract = run_gtap._build_gtap_contract_with_calibration("gtap_standard7_9x10")
    base_closure = contract.closure.model_copy(update={"if_sub": False})

    # =========================================================================
    # [1/3] Python GTAP 9x10 baseline (mirrors GAMS 'base' period calibration)
    # =========================================================================
    print("=== [1/3] Python GTAP 9x10 baseline ===")
    p_b = GTAPParameters()
    p_b.load_from_gdx(GDX_9X10)
    eq_b = GTAPModelEquations(p_b.sets, p_b, base_closure, residual_region="NAmerica")
    m_b = eq_b.build_model()
    t0 = time.perf_counter()
    r_b = run_gtap._run_path_capi_nonlinear_full(
        m_b, p_b, enforce_post_checks=False, strict_path_capi=False,
        closure_config=base_closure, equation_scaling=True,
    )
    sec_b = time.perf_counter() - t0
    res_b = float(r_b.get("residual") or 0.0)
    print(f"  baseline residual={res_b:.3e}  code={r_b.get('termination_code')}  t={sec_b:.2f}s")

    # =========================================================================
    # [2/3] Python altertax check period
    #   - Apply CD elasticities + all factors mobile (altertax overrides)
    #   - NO imptx shock (same as GAMS 'check' period)
    #   - Warm-start from baseline
    # =========================================================================
    print("\n=== [2/3] Python altertax check period (CD elast, mobile factors, no shock) ===")
    p_alt = apply_altertax_elasticities(p_b, in_place=False)

    # Altertax closure: all factors mobile, taxes fixed, standard numeraire
    alt_closure = GTAPClosureConfig(
        name="altertax",
        closure_type="MCP",
        capital_mobility="mobile",
        fix_endowments=False,
        fix_taxes=True,
        fix_technology=True,
        if_sub=False,
        numeraire=base_closure.numeraire,
        rmuv=base_closure.rmuv,
        imuv=base_closure.imuv,
    )

    # Check period: altertax params, no imptx shock, warm-start from baseline
    warm_b = GTAPVariableSnapshot.from_python_model(m_b)
    eq_chk = GTAPModelEquations(
        p_alt.sets, p_alt, alt_closure,
        is_counterfactual=True, residual_region="NAmerica", t0_snapshot=m_b,
    )
    m_chk = eq_chk.build_model()
    t0 = time.perf_counter()
    r_chk = run_gtap._run_path_capi_nonlinear_full(
        m_chk, p_alt,
        enforce_post_checks=False, strict_path_capi=False,
        closure_config=alt_closure, equation_scaling=True,
        solution_hint=warm_b,
    )
    sec_chk = time.perf_counter() - t0
    res_chk = float(r_chk.get("residual") or 0.0)
    print(f"  check residual={res_chk:.3e}  code={r_chk.get('termination_code')}  t={sec_chk:.2f}s")

    # =========================================================================
    # [3/3] Python altertax shock period
    #   - Apply +10% imptx shock on top of altertax params
    #   - Warm-start from check period solution
    # =========================================================================
    print("\n=== [3/3] Python altertax shock (+10% imptx, warm-started from check) ===")
    import copy
    p_alt_shock = copy.deepcopy(p_alt)
    for key in list(p_alt_shock.taxes.imptx.keys()):
        old = float(p_alt_shock.taxes.imptx[key] or 0.0)
        p_alt_shock.taxes.imptx[key] = old * 1.10

    warm_chk = GTAPVariableSnapshot.from_python_model(m_chk)
    eq_alt = GTAPModelEquations(
        p_alt_shock.sets, p_alt_shock, alt_closure,
        is_counterfactual=True, residual_region="NAmerica", t0_snapshot=m_b,
    )
    m_alt = eq_alt.build_model()
    t0 = time.perf_counter()
    r_alt = run_gtap._run_path_capi_nonlinear_full(
        m_alt, p_alt_shock,
        enforce_post_checks=False, strict_path_capi=False,
        closure_config=alt_closure, equation_scaling=True,
        solution_hint=warm_chk,
    )
    sec_alt = time.perf_counter() - t0
    res_alt = float(r_alt.get("residual") or 0.0)
    print(f"  shock residual={res_alt:.3e}  code={r_alt.get('termination_code')}  t={sec_alt:.2f}s")

    gdx_path = args.gdx
    var_names = list_populated_vars(gdx_path)
    print(f"\nPopulated GAMS Vars in {gdx_path.name}: {len(var_names)}")

    # NEOS out.gdx uses period "shock" for the +10% tariff altertax result
    phase = "shock"
    print(f"\n{'='*120}")
    print(f"PHASE: altertax → comparing Python m_alt vs GAMS {gdx_path.name} period='{phase}'")
    print(f"  tol_rel={args.tol_rel}  tol_abs={args.tol_abs}")
    print(f"{'='*120}")
    print(f"{'gams_var':<14s} {'py_var':<14s} {'cells':>7s} {'match':>7s} {'diverge':>8s} {'missing':>8s} {'max_abs':>10s} {'max_rel':>10s}  status")
    print("-" * 120)

    git_sha = git_short_sha(ROOT)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows, agg = diff_phase_rows(
        dataset="9x10_altertax", phase=phase, var_names=var_names,
        gdx_path=gdx_path, model_py=m_alt,
        tol_rel=args.tol_rel, tol_abs=args.tol_abs,
        residual=res_alt, git_sha=git_sha, generated_at=generated_at,
        derived=build_derived(m_alt),
        solve_seconds=sec_alt,
    )

    diverge_details = []
    for r in rows:
        if r["var"] == "__SUMMARY__":
            continue
        cells = int(r["cells"])
        match = int(r["match"])
        diverge = int(r["diverge"])
        missing = int(r["missing"])
        mx_abs = r["max_abs_err"] or "—"
        mx_rel = r["max_rel_err"] or "—"
        if not r["py_var"]:
            status = "no-py"
            py = "<n/a>"
        elif diverge == 0 and missing == 0:
            status = "ok"
            py = r["py_var"]
        else:
            status = "diff" if diverge else "miss"
            py = r["py_var"]
        print(f"{r['var']:<14s} {py:<14s} {cells:>7d} {match:>7d} "
              f"{diverge:>8d} {missing:>8d} {mx_abs:>10s} {mx_rel:>10s}  {status}")

        if args.show_worst and (diverge > 0 or missing > 0) and r["py_var"]:
            gams_all = gams_levels(gdx_path, r["var"])
            py_var, _ = find_py_var(m_alt, r["var"], derived=build_derived(m_alt))
            if py_var is not None:
                s = compare_phase(py_var, gams_all, phase,
                                  tol_rel=args.tol_rel, tol_abs=args.tol_abs)
                if s["worst"]:
                    diverge_details.append((r["var"], r["py_var"], s))

    print("-" * 120)
    print(f"  Vars total:           {agg['vars_total']}")
    print(f"  Vars all-match:       {agg['vars_match_all']}")
    print(f"  Vars partial/diverge: {agg['vars_partial']}")
    print(f"  Vars not in Python:   {agg['vars_no_py']}")
    print(f"  Cells total:          {agg['cells_total']}")
    print(f"  Cells match:          {agg['cells_match']}")
    print(f"  Cells diverge:        {agg['cells_diverge']}")
    print(f"  Cells missing/no-py:  {agg['cells_missing']}")
    coverage = (agg["cells_match"] / agg["cells_total"] * 100.0) if agg["cells_total"] else 0.0
    print(f"  Match rate:           {coverage:.2f}%")

    if args.show_worst and diverge_details:
        print(f"\n  Worst diverging cell per variable (altertax):")
        for name, py_name, stats in diverge_details[:30]:
            w = stats["worst"]
            if w is None:
                continue
            key, p_val, g_val, d, rel = w
            rel_str = f"{rel*100:.3f}%" if rel != float("inf") else "inf"
            print(f"    {name:<12s} {str(key):<60s}  py={p_val:+.6e}  gams={g_val:+.6e}  Δ={d:+.3e}  rel={rel_str}")

    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
