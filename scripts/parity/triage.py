#!/usr/bin/env python
"""Generic equilibria parity-debug CLI: locate → isolate → trace → check-warmstart → check-solution.

Usage:
  triage.py --list
  triage.py --template T --dataset D --scenario S --gdx-ref PATH [--top N]
  triage.py --template T --dataset D --scenario S --gdx-ref PATH --inspect VAR
  triage.py --template T --dataset D --scenario S --gdx-ref PATH --trace VAR --cell K1,K2,...
  triage.py --template T --dataset D --scenario S --gdx-ref PATH --check-warmstart
  triage.py --template T --dataset D --scenario S --gdx-ref PATH --check-solution

--check-warmstart reports (without running the solver):
  1. Equations with large residuals at the warm-start point — reveals which
     equilibrium basin the solver starts from and whether the warm-start is
     consistent with the GAMS solution.
  2. Variables where the warm-started value differs from the GAMS reference —
     pinpoints key normalization failures (e.g. 'a_Food' prefix not stripped).

--check-solution reports (after running the solver):
  Equations with large residuals at the post-solve point.  If PATH reports
  convergence but an active equation still has residual > tol, it means the
  variable paired with that equation has a zero Jacobian column — PATH never
  moved it (spurious MCP: variable stuck at lower bound).  This catches bugs
  invisible to all other tools (e.g. uh stuck at lb=0.001 because eq_uh has
  zero ∂/∂uh in Python but not in GAMS).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def cmd_list() -> int:
    from parity._adapter_protocol import AdapterRegistry
    for tpl in AdapterRegistry.list_templates():
        cls = AdapterRegistry.get(tpl)
        combos = cls().enumerate_combinations()
        print(f"{tpl}:")
        for ds, sc in combos:
            print(f"  {ds}/{sc}")
    return 0


def cmd_run(args) -> int:
    from parity._adapter_protocol import AdapterRegistry
    from parity._triage_steps import step_locate, step_isolate, step_trace, step_check_warmstart, step_check_solution

    try:
        adapter_cls = AdapterRegistry.get(args.template)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    adapter = adapter_cls()

    if (args.dataset, args.scenario) not in adapter.enumerate_combinations():
        print(
            f"error: ({args.dataset!r}, {args.scenario!r}) not in "
            f"{adapter.enumerate_combinations()}",
            file=sys.stderr,
        )
        return 2

    if args.check_warmstart:
        print(f"[1/2] Building (unsolved) model for {args.template}/{args.dataset}/{args.scenario}...", flush=True)
        model = adapter.build_warmstarted_model(args.dataset, args.scenario)
        print(f"[2/2] Loading GAMS ref from {args.gdx_ref}...", flush=True)
        _load_ref = getattr(adapter, "load_gams_reference_for_scenario", None)
        if _load_ref is not None:
            ref = _load_ref(Path(args.gdx_ref), args.dataset, args.scenario)
        else:
            ref = adapter.load_gams_reference(Path(args.gdx_ref))
        result = step_check_warmstart(adapter, model, ref, top_n=args.top,
                                      tol_rel=args.tol_rel, tol_abs=args.tol_abs)
        print(f"\n=== Equation residuals at warm-start point (top {args.top}) ===")
        if not result["equation_residuals"]:
            print("  All active equations satisfied within tol — warm-start is consistent.")
        for r in result["equation_residuals"]:
            print(f"  {r['name']:<35s} idx={str(r['index']):<30s} resid={r['residual']:.3e}  body={r['body']:.4e}  target={r['target']:.4e}")
        print(f"\n=== Variable warm-start gaps vs GAMS ref (top {args.top}) ===")
        if not result["var_gaps"]:
            print("  All variables match GAMS ref — warm-start complete.")
        for r in result["var_gaps"]:
            if r["not_set"]:
                print(f"  {r['gams_var']:<14s} → {r['py_var']:<14s}  {r['_not_set_count']} cells NOT FOUND in Python model (key mismatch?)")
            else:
                print(f"  {r['gams_var']:<14s} → {r['py_var']:<14s}  key={r['key']}  py={r['py_val']:+.4e}  gams={r['gams_val']:+.4e}  rel={r['rel_err']:.2%}")
        cov = result.get("seeding_coverage", [])
        if cov:
            not_seeded = [r for r in cov if r["matched"] == 0 and r["diverged"] > 0]
            partial = [r for r in cov if 0 < r["frac"] < 0.95 and r["diverged"] > 0]
            print(f"\n=== Warm-start seeding coverage ({len(cov)} vars with GAMS ref) ===")
            if not_seeded:
                print(f"  ⚠️  {len(not_seeded)} vars with 0% match (likely GDX name mismatch — not seeded):")
                for r in not_seeded[:args.top]:
                    print(f"    {r['py_var']:<20s}  matched=0/{r['matched']+r['diverged']}  diverged={r['diverged']}")
            else:
                print("  ✅ All vars had at least some cells matching GAMS ref.")
            if partial:
                print(f"  ℹ️  {len(partial)} vars partially matching GAMS ref:")
                for r in partial[:args.top]:
                    print(f"    {r['py_var']:<20s}  matched={r['matched']}/{r['matched']+r['diverged']} ({r['frac']:.0%})")
        return 0

    print(f"[1/3] Building+solving {args.template}/{args.dataset}/{args.scenario}...", flush=True)
    model = adapter.build_solved_model(args.dataset, args.scenario)

    print(f"[2/3] Loading GAMS ref from {args.gdx_ref}...", flush=True)
    _load_ref = getattr(adapter, "load_gams_reference_for_scenario", None)
    if _load_ref is not None:
        ref = _load_ref(Path(args.gdx_ref), args.dataset, args.scenario)
    else:
        ref = adapter.load_gams_reference(Path(args.gdx_ref))

    if args.check_solution:
        print(f"[3/3] Checking equation residuals at post-solve point (top {args.top})...")
        result = step_check_solution(model, top_n=args.top, tol_abs=args.tol_abs)
        print(f"\n=== Active equations with residual > {args.tol_abs:.0e} at PATH solution ===")
        print("  (non-zero residual after convergence = variable has zero Jacobian column)")
        print(f"  {'equation':<40s} {'index':<30s} {'residual':>12s}  {'body':>12s}")
        print("  " + "-" * 100)
        if not result:
            print("  All active equations satisfied within tol — no decoupled variables detected.")
        for r in result:
            print(f"  {r['name']:<40s} {str(r['index']):<30s} {r['residual']:>12.3e}  {r['body']:>12.4e}")
        return 0

    if args.trace and args.cell:
        cell = tuple(args.cell.split(","))
        print(f"[3/3] Tracing constraint residuals (top {args.top}) for cell {cell}...")
        residuals = step_trace(model, top_n=args.top)
        for r in residuals:
            print(f"  {r['name']:<30s} idx={r['index']}  resid={r['residual']:.3e}  "
                  f"body={r['body']:.4e}  target={r['target']:.4e}")
        return 0

    if args.inspect:
        print(f"[2.5/3] Worst diverging cell of {args.inspect}...")
        worst = step_isolate(adapter, model, ref, args.inspect, tol_rel=args.tol_rel, tol_abs=args.tol_abs)
        if worst is None:
            print(f"  All cells of {args.inspect} match within tol.")
        else:
            print(f"  key={worst['key']}  py={worst['py_val']:+.6e}  "
                  f"gams={worst['gams_val']:+.6e}  rel={worst['rel_err']:.3%}")
        return 0

    print(f"[3/3] Top {args.top} diverging variables...")
    rows = step_locate(adapter, model, ref, top_n=args.top, tol_rel=args.tol_rel, tol_abs=args.tol_abs)
    if not rows:
        print("  No diverging variables — parity complete.")
        return 0
    print(f"  {'gams_var':<14s} {'py_var':<14s} {'cells':>7s} {'diverge':>8s} {'max_rel':>10s}  worst_key")
    for r in rows:
        print(f"  {r['gams_var']:<14s} {r['py_var']:<14s} "
              f"{r['cells']:>7d} {r['diverge']:>8d} {r['max_rel_err']:>9.2%}  {r['worst_key']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="List all known templates and combinations.")
    ap.add_argument("--template", help="Template name (gtap, pep, simple_open).")
    ap.add_argument("--dataset", help="Dataset identifier (e.g. 9x10, nus333, pep_base, default).")
    ap.add_argument("--scenario", help="Scenario identifier (e.g. baseline, altertax, shock_tm10).")
    ap.add_argument("--gdx-ref", help="Path to GAMS reference GDX.")
    ap.add_argument("--top", type=int, default=10, help="Top-N rows for locate/trace.")
    ap.add_argument("--inspect", help="Show worst cell of this variable.")
    ap.add_argument("--trace", help="Show constraint residuals (use with --cell).")
    ap.add_argument("--cell", help="Comma-separated key for --trace (e.g. EU_28,Land,a_agricultur).")
    ap.add_argument("--tol-rel", type=float, default=1e-3)
    ap.add_argument("--tol-abs", type=float, default=1e-6)
    ap.add_argument("--check-warmstart", action="store_true",
                    help="Check equation residuals and variable gaps at the warm-start point "
                         "(before solver runs). Requires adapter.build_warmstarted_model().")
    ap.add_argument("--check-solution", action="store_true",
                    help="After solving, report active equations whose residual is still > tol. "
                         "Non-zero residual post-convergence means the variable paired with that "
                         "equation has a zero Jacobian column — PATH never moved it (spurious MCP). "
                         "Catches decoupled-variable bugs invisible to .nl/closure/value diffs.")
    args = ap.parse_args()

    if args.list:
        return cmd_list()

    missing = [k for k in ("template", "dataset", "scenario", "gdx_ref") if getattr(args, k) is None]
    if missing:
        ap.error(f"missing required arguments: {missing}")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
