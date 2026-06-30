#!/usr/bin/env python
"""Cached, flag-driven parity probe — Tool #1 of the cascade (seed-at-GAMS scan).

See docs/superpowers/specs/2026-06-12-probe-cached-parity-tool-design.md.

PHASE 1 (JSON-only output): this tool now prints EXACTLY one pure-JSON object to
stdout in every mode and every path. There is no text mode and no `--json` flag —
JSON is the output. All build/solve/log chatter is redirected to stderr by
`_parity_json.run_tool`/`stdout_to_stderr`, so stdout is always parseable.

The canonical cascade mode is `--seed-gams P --residuals` (seed Python at the GAMS
point and scan equation residuals); that mode fills the common schema
(status/headline/violations). Every OTHER mode (--params, --params-compare-builds,
--show, --residuals without seed, --compare-ref, --clear-cache) still emits JSON,
but with its own shape under meta.mode (it is not forced into `violations`).

Input flags are UNCHANGED from the pre-JSON version.
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
from _probe_params import diff_params_vs_gams, diff_param_builds  # noqa: E402
from _parity_json import (  # noqa: E402
    make_violation, build_payload, emit, run_tool, normalize_period,
)

# Closure name per scenario (matches GTAPParityAdapter altertax closures).
_CLOSURE_NAME = {
    "altertax_check": "altertax",
    "altertax_shock": "altertax",
    "baseline": "base",
    "shock_tm10": "base",
}

# GAMS period per scenario, for --params vs the GDX.
_PARAMS_PERIOD = {
    "altertax_check": "check",
    "altertax_shock": "shock",
    "baseline": "base",
    "shock_tm10": "shock",
}


def _build_and_value(adapter, dataset, scenario, cache_dir, no_cache):
    """Build the model (always) and supply solved values from cache or solve."""
    closure = _CLOSURE_NAME.get(scenario, scenario)
    key = compute_cache_key(dataset, scenario, closure)
    cached = None if no_cache else load_solution(key, cache_dir=cache_dir)
    if cached is not None:
        model = adapter.build_warmstarted_model(dataset, scenario)
        n = inject_solution(model, cached)
        print(f"[cache hit] key={key}  injected {n} cells")  # -> stderr under capture
        return model, key
    t0 = time.time()
    model = adapter.build_solved_model(dataset, scenario)
    sol = extract_solution(model)
    store_solution(key, sol, cache_dir=cache_dir)
    print(f"[solved + cached {time.time()-t0:.0f}s] key={key}")  # -> stderr
    return model, key


# ---------------------------------------------------------------------------
# Human-readable formatters — KEPT for debug only. They write to STDERR so they
# can never contaminate the JSON on stdout. Not called in the normal path.
# ---------------------------------------------------------------------------
def _debug_print_show(rows):
    for r in rows:
        v = r["value"]
        vs = f"{v:.6f}" if isinstance(v, float) else str(v)
        print(f"  {r['var']:<10s} {str(r['idx']):<32s} {vs}", file=sys.stderr)


def _debug_print_residuals(rows):
    for r in rows:
        print(f"  {r['eq']:<28s} idx={str(r['idx']):<28s} "
              f"resid={r['resid']:.4e}", file=sys.stderr)


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


# ---------------------------------------------------------------------------
# Mode runners. Each returns a dict {status, headline, violations, meta, period}
# consumed by run_tool. Canonical seed-at-GAMS fills the common schema; the
# other modes use status clean/dirty/error with their own meta.mode shape.
# ---------------------------------------------------------------------------
def _run_seed_gams(adapter, args) -> dict:
    """Canonical cascade mode: seed Python at the GAMS point, scan residuals."""
    if not args.gdx_ref:
        return dict(status="error", period=args.seed_gams,
                    headline="--seed-gams requires --gdx-ref",
                    violations=[], meta={"mode": "seed_gams"})
    model = adapter.build_warmstarted_model(args.dataset, args.scenario)
    print("[seed-gams] fresh warm-started build (no cache)")  # -> stderr
    res = seed_gams_point(model, Path(args.gdx_ref), args.seed_gams,
                          threshold=args.seed_threshold)
    rows = query_residuals(model, top_n=args.top, family=args.family)
    tol = args.tol_rel
    viols = [make_violation(r["eq"], r["idx"], "residual", r["resid"])
             for r in rows if abs(r["resid"]) > tol]
    top_resid = max((abs(r["resid"]) for r in rows), default=0.0)
    status = "dirty" if viols else "clean"
    if viols:
        worst = viols[0]
        headline = (
            f"seed-at-GAMS {args.seed_gams}: coverage {res['coverage']:.0%}, "
            f"{len(viols)} eq residual(s) > {tol:g}; worst "
            f"{worst['entity']}{worst['index']}={worst['value']:.3e} — "
            f"the GAMS point violates Python equations at this layer")
    else:
        headline = (
            f"seed-at-GAMS {args.seed_gams}: coverage {res['coverage']:.0%}, "
            f"all residuals <= {tol:g} (max {top_resid:.3e}) — "
            f"this layer does not explain the gap")
    return dict(
        status=status, period=args.seed_gams, headline=headline,
        violations=viols,
        meta={
            "mode": "seed_gams", "coverage": res["coverage"],
            "cells_set": res["cells_set"],
            "exportable_cells": res["exportable_cells"],
            "total_free_cells": res["total_free_cells"],
            "below_threshold": res["below_threshold"],
            "seed_threshold": args.seed_threshold,
            "residual_tol": tol, "scenario": args.scenario,
            "gdx_ref": str(args.gdx_ref),
            "n_residuals_scanned": len(rows),
        },
    )


def _run_params(adapter, args) -> dict:
    """--params: diff Pyomo Params vs the GAMS GDX."""
    if not args.gdx_ref:
        return dict(status="error", period=None,
                    headline="--params requires --gdx-ref",
                    violations=[], meta={"mode": "params"})
    model = adapter.build_warmstarted_model(args.dataset, args.scenario)
    period = args.params_period or _PARAMS_PERIOD.get(args.scenario, "base")
    res = diff_params_vs_gams(model, Path(args.gdx_ref), period,
                              tol_rel=args.tol_rel)
    viols = []
    for r in res["diverge"]:
        w = r["worst"]
        idx = [str(w[0])] if w else []
        viols.append(make_violation(r["param"], idx, "param_rel", r["max_rel"]))
    n_verifiable = len(res["diverge"]) + len(res["ok"])
    status = "dirty" if res["diverge"] else "clean"
    headline = (
        f"param diff vs GAMS (period={period}): {len(res['diverge'])} diverge / "
        f"{n_verifiable} verifiable, {len(res['no_match'])} no GAMS counterpart")
    return dict(
        status=status, period=normalize_period(period), headline=headline,
        violations=viols,
        meta={"mode": "params", "gams_period": period,
              "n_verifiable": n_verifiable,
              "n_no_match": len(res["no_match"]),
              "tol_rel": args.tol_rel, "scenario": args.scenario,
              "gdx_ref": str(args.gdx_ref)},
    )


def _run_params_compare_builds(args) -> dict:
    """--params-compare-builds: Params that change with/without t0_snapshot."""
    changed = diff_param_builds(args.dataset, tol_rel=args.tol_rel)
    viols = [make_violation(r["param"], [], "build_rel", r["max_rel"])
             for r in changed]
    status = "dirty" if changed else "clean"
    headline = (
        f"{len(changed)} build-dependent Param(s) change with/without "
        f"t0_snapshot" if changed
        else "no build-dependent Params; calibration is consistent")
    return dict(
        status=status, period=None, headline=headline, violations=viols,
        meta={"mode": "params_compare_builds", "tol_rel": args.tol_rel,
              "detail": changed[:args.top]},
    )


def _run_query(adapter, args) -> dict:
    """--show / --residuals (without seed): read-only value/residual dump.

    Not part of the cascade signal — status is always 'clean' (a read query
    does not 'violate' anything). Values go to meta.values for debug.
    """
    model, _ = _build_and_value(adapter, args.dataset, args.scenario,
                                args.cache_dir, args.no_cache)
    vals = _query_to_dict(model, args)
    return dict(
        status="clean", period=None,
        headline=f"read query: {len(vals)} value(s)",
        violations=[],
        meta={"mode": "query", "values": vals,
              "show": args.show, "residuals": bool(args.residuals),
              "scenario": args.scenario},
    )


def _run_compare_ref(adapter, args) -> dict:
    """--compare-ref: A/B the read query HEAD vs a git ref, side-by-side."""
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
    deltas = {}
    viols = []
    for k in keys:
        h = head_vals.get(k)
        rv = ref_vals.get(k)
        if isinstance(h, (int, float)) and isinstance(rv, (int, float)):
            d = h - rv
            deltas[k] = d
            if abs(d) > 0:
                viols.append(make_violation(k, [], "head_minus_ref", d))
    status = "dirty" if viols else "clean"
    headline = (f"compare-ref {args.compare_ref[:12]}: {len(viols)} of "
                f"{len(keys)} keys differ")
    return dict(
        status=status, period=None, headline=headline, violations=viols,
        meta={"mode": "compare_ref", "ref": args.compare_ref,
              "head_values": head_vals, "ref_values": ref_vals,
              "deltas": deltas, "scenario": args.scenario},
    )


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
    ap.add_argument("--params", action="store_true",
                    help="diff all Pyomo Params vs the GAMS GDX (3 groups)")
    ap.add_argument("--params-compare-builds", dest="params_compare_builds",
                    action="store_true",
                    help="report Params that change with/without t0_snapshot (no GAMS needed)")
    ap.add_argument("--params-period", dest="params_period", default=None,
                    help="GAMS period for --params (default: derived from scenario)")
    ap.add_argument("--tol-rel", type=float, default=1e-3)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"],
                    help="altertax CD (default) or pure-gtap real-CES")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1],
                    help="ifSUB mode (pure-gtap only)")
    args = ap.parse_args()

    # mode=gtap UNSUPPORTED: probe drives the parity ADAPTER (altertax scenarios);
    # the pure-gtap MP solve is not an adapter scenario. Accept the flag (the
    # orchestrator passes it) but emit an honest mode_unsupported. seed_and_solve
    # --mode gtap is the faithful pure-gtap seed-and-solve path.
    if args.mode == "gtap" and not args.emit_json:
        def _unsupported() -> dict:
            return dict(
                status="error", period=args.seed_gams,
                headline=("probe does not support --mode gtap (drives the altertax "
                          "parity adapter); use seed_and_solve --mode gtap."),
                violations=[],
                meta={"error_kind": "mode_unsupported", "mode": "gtap",
                      "ifsub": args.ifsub})
        return run_tool("probe", args.dataset, _unsupported,
                        period_hint=args.seed_gams)

    # --emit-json is the LEGACY internal channel used by --compare-ref's
    # subprocess. It must stay a __JSON__-prefixed line (NOT the common schema)
    # because the parent parses it that way. It runs OUTSIDE run_tool. To keep
    # stdout clean for the parent, capture build/solve chatter to stderr.
    if args.emit_json:
        from _parity_json import stdout_to_stderr, _REAL_STDOUT
        try:
            with stdout_to_stderr():
                adapter = AdapterRegistry.get(args.template)()
                if args.seed_gams:
                    model = adapter.build_warmstarted_model(args.dataset,
                                                            args.scenario)
                    seed_gams_point(model, Path(args.gdx_ref), args.seed_gams,
                                    threshold=args.seed_threshold)
                else:
                    model, _ = _build_and_value(adapter, args.dataset,
                                                args.scenario, args.cache_dir,
                                                args.no_cache)
                payload = _query_to_dict(model, args)
            _REAL_STDOUT.write("__JSON__" + json.dumps(payload) + "\n")
            _REAL_STDOUT.flush()
            return 0
        except Exception as exc:  # noqa: BLE001
            _REAL_STDOUT.write("__JSON__" + json.dumps({"__error__": str(exc)}) + "\n")
            _REAL_STDOUT.flush()
            return 2

    if args.clear_cache:
        # No model build; emit directly (still pure JSON).
        n = clear_cache(cache_dir=args.cache_dir)
        emit(build_payload(tool="probe", dataset=args.dataset, period=None,
                           status="clean", headline=f"cleared {n} cache file(s)",
                           violations=[], meta={"mode": "clear_cache",
                                                "cleared": n}))
        return 0

    def _work() -> dict:
        try:
            adapter = AdapterRegistry.get(args.template)()
        except KeyError as e:
            return dict(status="error", period=None,
                        headline=f"unknown template: {e}", violations=[],
                        meta={"mode": "init"})
        if (args.dataset, args.scenario) not in adapter.enumerate_combinations():
            return dict(
                status="error", period=None,
                headline=(f"({args.dataset!r}, {args.scenario!r}) not available"),
                violations=[],
                meta={"mode": "init",
                      "available": [list(c) for c in
                                    adapter.enumerate_combinations()]})

        if args.params_compare_builds:
            return _run_params_compare_builds(args)
        if args.params:
            return _run_params(adapter, args)
        if args.compare_ref:
            return _run_compare_ref(adapter, args)
        if args.seed_gams:
            return _run_seed_gams(adapter, args)
        return _run_query(adapter, args)

    return run_tool("probe", args.dataset, _work)


if __name__ == "__main__":
    raise SystemExit(main())
