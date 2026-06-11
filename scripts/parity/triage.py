#!/usr/bin/env python
"""Generic equilibria parity-debug CLI: locate → isolate → trace.

Usage:
  triage.py --list
  triage.py --template T --dataset D --scenario S --gdx-ref PATH [--top N]
  triage.py --template T --dataset D --scenario S --gdx-ref PATH --inspect VAR
  triage.py --template T --dataset D --scenario S --gdx-ref PATH --trace VAR --cell K1,K2,...
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
    from parity._triage_steps import step_locate, step_isolate, step_trace

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

    print(f"[1/3] Building+solving {args.template}/{args.dataset}/{args.scenario}...", flush=True)
    model = adapter.build_solved_model(args.dataset, args.scenario)

    print(f"[2/3] Loading GAMS ref from {args.gdx_ref}...", flush=True)
    ref = adapter.load_gams_reference(Path(args.gdx_ref))

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
    args = ap.parse_args()

    if args.list:
        return cmd_list()

    missing = [k for k in ("template", "dataset", "scenario", "gdx_ref") if getattr(args, k) is None]
    if missing:
        ap.error(f"missing required arguments: {missing}")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
