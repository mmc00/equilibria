#!/usr/bin/env python
"""Compare canonical SimpleOpen closures against GAMS benchmark GDX artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from equilibria.templates import (  # noqa: E402
    compare_simple_open_gams_parity,
)

DEFAULT_GDX_BY_CLOSURE: dict[str, Path] = {
    "simple_open_default": REPO_ROOT / "output" / "simple_open_v1_benchmark_default.gdx",
    "flexible_external_balance": REPO_ROOT / "output" / "simple_open_v1_benchmark_flexible.gdx",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closures",
        nargs="+",
        default=["simple_open_default", "flexible_external_balance"],
        help="Canonical SimpleOpen closures to compare.",
    )
    parser.add_argument(
        "--gdx",
        action="append",
        default=[],
        metavar="CLOSURE=PATH",
        help="Override the GDX path for one closure.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for benchmark, level, parameter, and residual checks.",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Return non-zero if any requested closure fails parity.",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser


def _resolve_gdx_map(entries: list[str]) -> dict[str, Path]:
    gdx_map = dict(DEFAULT_GDX_BY_CLOSURE)
    for raw in entries:
        closure, sep, raw_path = str(raw).partition("=")
        if not sep:
            raise ValueError(f"invalid --gdx override: {raw!r}")
        name = closure.strip().lower()
        if not name:
            raise ValueError(f"invalid --gdx closure name: {raw!r}")
        gdx_map[name] = Path(raw_path).expanduser().resolve()
    return gdx_map


def _print_entry(name: str, entry: dict[str, Any]) -> None:
    print("-" * 92)
    print(f"Closure: {name}")
    print(
        "  parity: passed={passed} active_closure_match={closure_match} "
        "modelstat={modelstat} solvestat={solvestat}".format(
            passed=entry["passed"],
            closure_match=entry["active_closure_match"],
            modelstat=entry["modelstat"],
            solvestat=entry["solvestat"],
        )
    )
    print(
        "  benchmark: compared={compared} mismatches={mismatches} max_abs_diff={max_abs:.3e}".format(
            compared=entry["benchmark_compared"],
            mismatches=entry["benchmark_mismatches"],
            max_abs=float(entry["benchmark_max_abs_diff"]),
        )
    )
    print(
        "  levels   : compared={compared} mismatches={mismatches} max_abs_diff={max_abs:.3e}".format(
            compared=entry["level_compared"],
            mismatches=entry["level_mismatches"],
            max_abs=float(entry["level_max_abs_diff"]),
        )
    )
    print(
        "  residuals: compared={compared} mismatches={mismatches} max_abs={max_abs:.3e}".format(
            compared=entry["residual_compared"],
            mismatches=entry["residual_mismatches"],
            max_abs=float(entry["residual_max_abs"]),
        )
    )
    print(
        "  params   : compared={compared} mismatches={mismatches} max_abs_diff={max_abs:.3e}".format(
            compared=entry["parameter_compared"],
            mismatches=entry["parameter_mismatches"],
            max_abs=float(entry["parameter_max_abs_diff"]),
        )
    )


def main() -> int:
    args = _build_parser().parse_args()
    try:
        gdx_map = _resolve_gdx_map(args.gdx)
    except Exception as exc:
        print(f"Invalid --gdx override: {exc}")
        return 2

    closures = tuple(dict.fromkeys(str(item).strip().lower() for item in args.closures if str(item).strip()))
    report: dict[str, Any] = {
        "metadata": {
            "model": "simple_open_v1",
            "abs_tol": float(args.abs_tol),
            "gate": bool(args.gate),
            "closures": list(closures),
        },
        "closures": {},
    }

    failed = False
    for closure_name in closures:
        gdx_path = gdx_map.get(closure_name)
        if gdx_path is None:
            print(f"Missing GDX mapping for closure '{closure_name}'")
            return 2
        if not gdx_path.exists():
            print(f"GDX file not found for closure '{closure_name}': {gdx_path}")
            return 2

        comparison = compare_simple_open_gams_parity(
            contract={"closure": {"name": closure_name}},
            gdx_path=gdx_path,
            abs_tol=float(args.abs_tol),
        ).to_dict()
        report["closures"][closure_name] = comparison
        failed = failed or (not bool(comparison["passed"]))
        _print_entry(closure_name, comparison)

    report["gate"] = {"passed": not failed, "failed_closures": [name for name, entry in report["closures"].items() if not entry["passed"]]}

    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.gate and failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
