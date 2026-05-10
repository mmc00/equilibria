"""Full variable-by-variable diff: Python GTAP NUS333 vs GAMS NEOS out.gdx.

Mirrors diff_9x10_full.py but for the NUS333 (3 sectors × 3 regions × 3
factors) dataset. Reuses the solver pipeline from compare_nus333_vs_neos
(HAR loader, residual_region="ROW", power-form 10% imptx shock, structural
matching with eq_pwfact pin) and walks every populated Var symbol in the
NEOS reference GDX (output/nus333_neos/out.gdx) to count match/diverge
cells per phase (base, shock).

Pass --csv PATH to emit benchmark rows in the same schema as the 9x10 diff.
"""
from __future__ import annotations
import argparse, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/marmol/proyectos/path-capi-python/src")
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import (
    list_populated_vars, gams_levels, find_py_var, compare_phase,
    diff_phase_rows, write_csv, git_short_sha, build_derived,
)

NUS333_HAR = Path("/Users/marmol/Downloads/10284")
GAMS_OUT = ROOT / "output/nus333_neos/out.gdx"


def _nus333_key_remap(body: tuple) -> tuple:
    """GAMS NUS333 prefixes commodities with 'c_' and activities with 'a_';
    Python sets drop the prefix. Strip leading 'c_'/'a_' from each component."""
    out = []
    for k in body:
        if isinstance(k, str):
            if k.startswith("c_"):
                out.append(k[2:])
                continue
            if k.startswith("a_"):
                out.append(k[2:])
                continue
        out.append(k)
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["base", "shock", "both"], default="both")
    ap.add_argument("--tol-rel", type=float, default=1e-3)
    ap.add_argument("--tol-abs", type=float, default=1e-6)
    ap.add_argument("--show-worst", action="store_true")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    # Reuse the proven NUS333 build/solve pipeline.
    from compare_nus333_vs_neos import _solve, _apply_tariff_shock, _copy_var_levels
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    params = GTAPParameters()
    params.load_from_har(
        basedata_path=NUS333_HAR / "basedata.har",
        sets_path=NUS333_HAR / "sets.har",
        default_path=NUS333_HAR / "default.prm",
        baserate_path=NUS333_HAR / "baserate.har",
    )
    closure = GTAPClosureConfig(if_sub=False)

    print("=== Python baseline NUS333 ===")
    builder_b = GTAPModelEquations(params.sets, params, residual_region="ROW", closure=closure)
    m_b = builder_b.build_model()
    r_b = _solve(m_b, params, label="base")
    res_b = float(getattr(r_b, "residual", 0.0) or 0.0)

    m_s = None
    res_s = 0.0
    if args.phase != "base":
        print("\n=== Python shock NUS333 (10% imptx, power scaling) ===")
        _apply_tariff_shock(params, factor=1.10)
        builder_s = GTAPModelEquations(
            params.sets, params, residual_region="ROW", closure=closure,
            t0_snapshot=m_b,
        )
        m_s = builder_s.build_model()
        _copy_var_levels(m_b, m_s)
        r_s = _solve(m_s, params, label="shock")
        res_s = float(getattr(r_s, "residual", 0.0) or 0.0)

    var_names = list_populated_vars(GAMS_OUT)
    print(f"\nPopulated GAMS Vars in {GAMS_OUT.name}: {len(var_names)}")

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
            dataset="nus333", phase=phase, var_names=var_names,
            gdx_path=GAMS_OUT, model_py=m_py,
            tol_rel=args.tol_rel, tol_abs=args.tol_abs,
            residual=residual, git_sha=git_sha, generated_at=generated_at,
            derived=build_derived(m_py), key_remap=_nus333_key_remap,
        )
        csv_rows.extend(rows)

        diverge_details = []
        for r in rows:
            if r["var"] == "__SUMMARY__":
                continue
            cells = int(r["cells"]); match = int(r["match"])
            diverge = int(r["diverge"]); missing = int(r["missing"])
            mx_abs = r["max_abs_err"] or "—"; mx_rel = r["max_rel_err"] or "—"
            if not r["py_var"]:
                status, py = "no-py", "<n/a>"
            elif diverge == 0 and missing == 0:
                status, py = "ok", r["py_var"]
            else:
                status = "diff" if diverge else "miss"
                py = r["py_var"]
            print(f"{r['var']:<14s} {py:<14s} {cells:>7d} {match:>7d} "
                  f"{diverge:>8d} {missing:>8d} {mx_abs:>10s} {mx_rel:>10s}  {status}")

            if args.show_worst and (diverge > 0 or missing > 0) and r["py_var"]:
                gams_all = gams_levels(GAMS_OUT, r["var"])
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
