#!/usr/bin/env python3
"""Run YAML-based SAM transformation workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from equilibria.sam_tools import run_sam_transform_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YAML-driven SAM transformation workflow")
    parser.add_argument("--config", type=Path, required=True, help="Path to workflow YAML")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path override")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path override",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_sam_transform_workflow(
        config_file=args.config,
        output_override=args.output,
        report_override=args.report,
    )

    summary = report["summary"]
    final_balance = summary["final_balance"]
    print("SAM workflow complete")
    print(f"  workflow: {report['workflow']['name']}")
    print(f"  input:    {report['input']['path']}")
    print(f"  output:   {report['output']['path']}")
    print(f"  steps:    {summary['steps']}")
    print(f"  total:    {final_balance['total']:.6f}")
    print(f"  max|row-col|: {final_balance['max_row_col_abs_diff']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
