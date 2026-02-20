#!/usr/bin/env python3
"""Print compact summaries for systemic parity JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_bool(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "n/a"


def _summarize_report(path: Path) -> str:
    payload = json.loads(path.read_text())
    classification = payload.get("classification", {})
    kind = classification.get("kind", "unknown")
    reason = classification.get("reason", "unknown")
    first_failed = classification.get("first_failed_block")
    if not first_failed:
        first_failed = payload.get("init", {}).get("gates", {}).get("first_failed_block")
    if not first_failed:
        first_failed = "none"

    sam_qa = payload.get("sam_qa", {}).get("passed")
    init_ok = payload.get("init", {}).get("gates", {}).get("overall_passed")
    solve_converged = payload.get("solve", {}).get("converged")

    return (
        f"{path}: kind={kind} reason={reason} first_failed_block={first_failed} "
        f"sam_qa={_fmt_bool(sam_qa)} init_passed={_fmt_bool(init_ok)} "
        f"solve_converged={_fmt_bool(solve_converged)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="JSON report files")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any report is missing or invalid JSON",
    )
    args = parser.parse_args()

    had_error = False
    for path in args.reports:
        if not path.exists():
            had_error = True
            print(f"{path}: missing")
            continue

        try:
            print(_summarize_report(path))
        except Exception as exc:  # pragma: no cover - defensive script path
            had_error = True
            print(f"{path}: invalid ({exc})")

    return 1 if had_error and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
