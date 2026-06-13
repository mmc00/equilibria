#!/usr/bin/env python
"""Cached, flag-driven parity probe.

See docs/superpowers/specs/2026-06-12-probe-cached-parity-tool-design.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "parity"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _adapter_protocol import AdapterRegistry  # noqa: E402
from _probe_cache import (  # noqa: E402
    compute_cache_key, load_solution, store_solution, clear_cache,
)
from _probe_queries import (  # noqa: E402
    extract_solution, inject_solution, query_show, query_residuals,
    seed_gams_point,
)

# Closure name per scenario (matches GTAPParityAdapter altertax closures).
_CLOSURE_NAME = {
    "altertax_check": "altertax",
    "altertax_shock": "altertax",
    "baseline": "base",
    "shock_tm10": "base",
}


def _build_and_value(adapter, dataset, scenario, cache_dir, no_cache):
    """Build the model (always) and supply solved values from cache or solve."""
    closure = _CLOSURE_NAME.get(scenario, scenario)
    key = compute_cache_key(dataset, scenario, closure)
    cached = None if no_cache else load_solution(key, cache_dir=cache_dir)
    if cached is not None:
        model = adapter.build_warmstarted_model(dataset, scenario)
        n = inject_solution(model, cached)
        print(f"[cache hit] key={key}  injected {n} cells")
        return model, key
    t0 = time.time()
    model = adapter.build_solved_model(dataset, scenario)
    sol = extract_solution(model)
    store_solution(key, sol, cache_dir=cache_dir)
    print(f"[solved + cached {time.time()-t0:.0f}s] key={key}")
    return model, key


def _print_show(rows):
    for r in rows:
        v = r["value"]
        vs = f"{v:.6f}" if isinstance(v, float) else str(v)
        print(f"  {r['var']:<10s} {str(r['idx']):<32s} {vs}")


def _print_residuals(rows):
    for r in rows:
        print(f"  {r['eq']:<28s} idx={str(r['idx']):<28s} resid={r['resid']:.4e}")


def _query_to_dict(model, args):
    """Run the requested read-only query and return {label: value}."""
    out = {}
    if args.show:
        for r in query_show(model, args.show.split(","), region=args.region,
                            index_filter=args.index):
            out[f"{r['var']}{r['idx']}"] = r["value"]
    if args.residuals:
        for r in query_residuals(model, top_n=args.top, family=args.family):
            out[f"{r['eq']}{r['idx']}"] = r["resid"]
    return out


def _run_compare_ref(adapter, args) -> int:
    """A/B the query: HEAD vs args.compare_ref, side-by-side."""
    model, _ = _build_and_value(adapter, args.dataset, args.scenario,
                                args.cache_dir, args.no_cache)
    head_vals = _query_to_dict(model, args)

    tmp = Path(tempfile.mkdtemp(prefix="probe_compare_"))
    wt = tmp / "wt"
    ref_vals = {}
    try:
        subprocess.run(["git", "worktree", "add", "--detach", str(wt),
                        args.compare_ref], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        sub_cmd = [
            "uv", "run", "python", "scripts/parity/probe.py",
            "--template", args.template, "--dataset", args.dataset,
            "--scenario", args.scenario, "--emit-json",
            "--cache-dir", str(tmp / "refcache"),
        ]
        if args.show:
            sub_cmd += ["--show", args.show]
            if args.region:
                sub_cmd += ["--region", args.region]
        if args.residuals:
            sub_cmd += ["--residuals", "--top", str(args.top)]
        sub = subprocess.run(sub_cmd, cwd=wt, capture_output=True, text=True,
                             timeout=600)
        for line in sub.stdout.splitlines():
            if line.startswith("__JSON__"):
                ref_vals = json.loads(line[len("__JSON__"):])
                break
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                       cwd=ROOT, capture_output=True, text=True)

    keys = sorted(set(head_vals) | set(ref_vals))
    print(f"{'key':<40s} {'HEAD':>14s} {args.compare_ref[:10]:>14s} {'Δ':>14s}")
    for k in keys:
        h = head_vals.get(k)
        rv = ref_vals.get(k)
        delta = ((h - rv) if (isinstance(h, (int, float))
                              and isinstance(rv, (int, float))) else None)
        hs = f"{h:.6f}" if isinstance(h, float) else str(h)
        rs = f"{rv:.6f}" if isinstance(rv, float) else str(rv)
        ds = f"{delta:.6f}" if isinstance(delta, float) else "—"
        print(f"  {k:<38s} {hs:>14s} {rs:>14s} {ds:>14s}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Cached parity probe")
    ap.add_argument("--template", default="gtap")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--show", help="comma-separated var names")
    ap.add_argument("--region")
    ap.add_argument("--index")
    ap.add_argument("--residuals", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--family")
    ap.add_argument("--seed-gams", dest="seed_gams", help="period: base|check|shock")
    ap.add_argument("--seed-threshold", type=float, default=0.95)
    ap.add_argument("--gdx-ref")
    ap.add_argument("--compare-ref", dest="compare_ref")
    ap.add_argument("--emit-json", action="store_true",
                    help="emit query result as __JSON__-prefixed line (for --compare-ref subprocess)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.clear_cache:
        n = clear_cache(cache_dir=args.cache_dir)
        print(f"cleared {n} cache files")
        return 0

    try:
        adapter = AdapterRegistry.get(args.template)()
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if (args.dataset, args.scenario) not in adapter.enumerate_combinations():
        print(f"error: ({args.dataset!r}, {args.scenario!r}) not available; "
              f"have {adapter.enumerate_combinations()}", file=sys.stderr)
        return 2

    if args.compare_ref:
        return _run_compare_ref(adapter, args)

    model, key = _build_and_value(adapter, args.dataset, args.scenario,
                                  args.cache_dir, args.no_cache)

    if args.seed_gams:
        if not args.gdx_ref:
            print("error: --seed-gams requires --gdx-ref", file=sys.stderr)
            return 2
        res = seed_gams_point(model, Path(args.gdx_ref), args.seed_gams,
                              threshold=args.seed_threshold)
        print(f"[seed-gams {args.seed_gams}] set {res['cells_set']}/"
              f"{res['total_cells']} cells ({res['coverage']:.0%})")
        if res["below_threshold"]:
            print(f"  WARNING: coverage {res['coverage']:.0%} < "
                  f"{args.seed_threshold:.0%} — results may be unreliable")

    if args.emit_json:
        print("__JSON__" + json.dumps(_query_to_dict(model, args)))
        return 0

    if args.show:
        rows = query_show(model, args.show.split(","), region=args.region,
                          index_filter=args.index)
        print(f"=== show {args.show} ===")
        _print_show(rows)
    if args.residuals:
        rows = query_residuals(model, top_n=args.top, family=args.family)
        print(f"=== residuals (top {args.top}) ===")
        _print_residuals(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
