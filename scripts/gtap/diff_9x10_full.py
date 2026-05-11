"""Full variable-by-variable diff: Python GTAP 9x10 vs GAMS COMP.gdx.

Iterates every Var symbol in COMP.gdx, looks for a matching Pyomo Var on
the Python model, aligns indices (dropping the GAMS time axis t∈{base,
check, shock}), and reports counts of matching/diverging cells per
variable. Symbols that don't exist on either side, or that are constant
zero, are reported as 'skipped'.

Pass --csv PATH to emit a long-form CSV consumed by the docs benchmark
page (one row per (dataset, phase, var) plus a __SUMMARY__ row).
"""
from __future__ import annotations
import argparse, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import (
    list_populated_vars, gams_levels, find_py_var, compare_phase,
    diff_phase_rows, write_csv, git_short_sha, split_t, build_derived,
)

GDX = ROOT / "src/equilibria/templates/reference/gtap/data/basedata-9x10.gdx"
GAMS_COMP = ROOT / "src/equilibria/templates/reference/gtap/output/COMP.gdx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["base", "shock", "both"], default="both")
    ap.add_argument("--tol-rel", type=float, default=1e-3,
                    help="Relative tolerance for match (default 0.1%%)")
    ap.add_argument("--tol-abs", type=float, default=1e-6,
                    help="Absolute tolerance fallback (default 1e-6)")
    ap.add_argument("--show-worst", action="store_true",
                    help="Print the single worst cell for each diverging symbol")
    ap.add_argument("--csv", type=Path, default=None,
                    help="If set, write benchmark rows to this CSV path")
    args = ap.parse_args()

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from run_gtap import _run_path_capi_nonlinear_full, _build_gtap_contract_with_calibration

    contract = _build_gtap_contract_with_calibration("gtap_standard7_9x10")

    # Match validate_gams_parity.py: GAMS NEOS reference uses ifSUB=0.
    # Default Python closure has if_sub=True (substitutes/fixes pm, pmcif, pefob,
    # pfa, pfy, xwmg, xmgm, pwmg, pp_rai); override to if_sub=False so these
    # bilateral price/margin variables stay free and can adjust to the shock.
    new_closure = contract.closure.model_copy(update={"if_sub": False})
    contract = contract.model_copy(update={"closure": new_closure})
    print(f"  closure: if_sub={contract.closure.if_sub}  numeraire={contract.closure.numeraire}")

    print("=== Python baseline 9x10 ===")
    p_b = GTAPParameters()
    p_b.load_from_gdx(GDX)
    eq_b = GTAPModelEquations(p_b.sets, p_b, contract.closure)
    m_b = eq_b.build_model()
    r_b = _run_path_capi_nonlinear_full(
        m_b, p_b, enforce_post_checks=False, strict_path_capi=False,
        closure_config=contract.closure, equation_scaling=True,
    )
    res_b = float(r_b.get("residual") or 0.0)
    print(f"  baseline residual={res_b:.3e}  code={r_b.get('termination_code')}")

    m_s = None
    res_s = 0.0
    if args.phase != "base":
        print("\n=== Python shock 9x10 (10% imptx, rate scaling) ===")
        p_s = GTAPParameters()
        p_s.load_from_gdx(GDX)
        for k in list(p_s.taxes.imptx.keys()):
            p_s.taxes.imptx[k] = float(p_s.taxes.imptx[k]) * 1.10
        eq_s = GTAPModelEquations(
            p_s.sets, p_s, contract.closure,
            is_counterfactual=True, t0_snapshot=m_b,
        )
        m_s = eq_s.build_model()
        from pyomo.environ import Var
        from pyomo.core import value
        for comp in m_b.component_objects(Var, active=True):
            dst = getattr(m_s, comp.name, None)
            if dst is None:
                continue
            for idx in comp:
                try:
                    v = float(value(comp[idx]))
                    if dst[idx].lb is not None and v < float(dst[idx].lb): v = float(dst[idx].lb)
                    if dst[idx].ub is not None and v > float(dst[idx].ub): v = float(dst[idx].ub)
                    dst[idx].set_value(v)
                except Exception:
                    pass
        r_s = _run_path_capi_nonlinear_full(
            m_s, p_s, enforce_post_checks=False, strict_path_capi=False,
            closure_config=contract.closure, equation_scaling=True,
        )
        res_s = float(r_s.get("residual") or 0.0)
        print(f"  shock residual={res_s:.3e}  code={r_s.get('termination_code')}")

    var_names = list_populated_vars(GAMS_COMP)
    print(f"\nPopulated GAMS Vars in COMP.gdx: {len(var_names)}")

    phases = [("base", m_b, res_b)]
    if args.phase != "base":
        phases.append(("shock", m_s, res_s))

    csv_rows: list[dict] = []
    git_sha = git_short_sha(ROOT)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for phase, m_py, residual in phases:
        print(f"\n{'='*120}")
        print(f"PHASE: {phase}    (tol_rel={args.tol_rel}  tol_abs={args.tol_abs})")
        print(f"{'='*120}")
        print(f"{'gams_var':<14s} {'py_var':<14s} {'cells':>7s} {'match':>7s} {'diverge':>8s} {'missing':>8s} {'max_abs':>10s} {'max_rel':>10s}  status")
        print("-" * 120)

        rows, agg = diff_phase_rows(
            dataset="9x10", phase=phase, var_names=var_names,
            gdx_path=GAMS_COMP, model_py=m_py,
            tol_rel=args.tol_rel, tol_abs=args.tol_abs,
            residual=residual, git_sha=git_sha, generated_at=generated_at,
            derived=build_derived(m_py),
        )
        csv_rows.extend(rows)

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

            # Recompute worst-cell on demand for show-worst.
            if args.show_worst and (diverge > 0 or missing > 0) and r["py_var"]:
                gams_all = gams_levels(GAMS_COMP, r["var"])
                py_var, _ = find_py_var(m_py, r["var"], derived=build_derived(m_py))
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
            print(f"\n  Worst diverging cell per variable (phase={phase}):")
            for name, py_name, stats in diverge_details[:30]:
                w = stats["worst"]
                if w is None:
                    continue
                key, p_val, g_val, d, rel = w
                rel_str = f"{rel*100:.3f}%" if rel != float("inf") else "inf"
                print(f"    {name:<12s} {str(key):<60s}  py={p_val:+.6e}  gams={g_val:+.6e}  Δ={d:+.3e}  rel={rel_str}")

    if args.csv:
        write_csv(args.csv, csv_rows)
        print(f"\nWrote {len(csv_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
