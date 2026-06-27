"""Phase-2 cascade orchestrator. Runs the six JSON cascade tools across periods,
stops at the first layer that explains the gap, and records provenance. The KKT
layer is wired in Task 7."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cascade_config import resolve_ref_gdx, resolve_periods, scenario_for
from cascade_run import sweep_period
from cascade_classify import EXPLAIN_STOP

_SOURCE_LABEL = {"durable": "durable", "adapter_output": "adapter-fallback",
                 "missing": "missing"}


def build_report(dataset, requested_periods, *, ref, period_results, kkt_reader):
    available, dropped = resolve_periods(dataset, requested_periods)
    periods = {}
    for period, results in period_results.items():
        first_dirty = next((r.name for r in results if r.action == EXPLAIN_STOP), None)
        periods[period] = {
            "scenario": scenario_for(dataset, period),
            "first_dirty_layer": first_dirty,
            "layers": [
                {"tool": r.name, "status": r.status, "error_kind": r.error_kind,
                 "action": r.action, "headline": r.headline}
                for r in results
            ],
        }
    explained = {p: d["first_dirty_layer"] for p, d in periods.items()
                 if d["first_dirty_layer"]}
    verdict = ("gap explained: " + ", ".join(f"{p}->{l}" for p, l in explained.items())
               if explained else "no layer explained the gap")
    return {
        "dataset": dataset,
        "ref": {"path": str(ref.path) if ref.path else None,
                "source": ref.source},
        "kkt_reader": kkt_reader,
        "dropped_periods": dropped,
        "periods": periods,
        "verdict": verdict,
    }


def render_tree(report: dict) -> str:
    lines = []
    src = _SOURCE_LABEL.get(report["ref"]["source"], report["ref"]["source"])
    lines.append(f"ref: {report['ref']['path']} [{src}], "
                 f"kkt_reader: [{report['kkt_reader']}]")
    if report["dropped_periods"]:
        lines.append(f"dropped periods (not in dataset): {report['dropped_periods']}")
    lines.append(f"dataset: {report['dataset']}")
    for period, d in report["periods"].items():
        lines.append(f"  period {period} (scenario={d['scenario']})")
        for layer in d["layers"]:
            mark = {
                "explain_stop": "DIRTY (explains gap)",
                "continue": "clean",
                "vacuous_continue": "did not opine (no common constraints)",
                "upstream_stop": "not measurable (upstream no_convergence)",
                "tool_broken_continue": "tool broken",
                "blocking_stop": "BLOCKING error",
            }.get(layer["action"], layer["action"])
            lines.append(f"    [{layer['tool']:18}] {mark} — {layer['headline']}")
    lines.append(f"verdict: {report['verdict']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="parity cascade orchestrator (Phase 2)")
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--periods", nargs="+", default=["check", "shock"])
    ap.add_argument("--no-stop", action="store_true",
                    help="run all layers without pruning (debug)")
    ap.add_argument("--tool-timeout", type=float, default=600.0)
    args = ap.parse_args()

    ref = resolve_ref_gdx(args.dataset)
    print(f"[orchestrator] ref: {ref.note}", file=sys.stderr)
    available, dropped = resolve_periods(args.dataset, args.periods)
    if not ref.usable:
        # No usable ref -> never build a GDX-dependent command. Abort with a
        # structured report so 'could not measure' never reads as clean.
        report = build_report(args.dataset, args.periods, ref=ref,
                              period_results={}, kkt_reader="n/a")
        report["verdict"] = "aborted: no usable reference GDX"
        print(json.dumps(report))
        print(render_tree(report), file=sys.stderr)
        return 2

    period_results = {}
    for period in available:
        period_results[period] = sweep_period(
            args.dataset, period, ref.path,
            stop=not args.no_stop, timeout=args.tool_timeout)
    # kkt_reader is set to 'pure-python' once Task 7 wires the KKT layer; until
    # then the report carries the resolved reader path it WILL use.
    report = build_report(args.dataset, args.periods, ref=ref,
                          period_results=period_results, kkt_reader="pure-python")
    print(json.dumps(report))
    print(render_tree(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
