#!/usr/bin/env python3
"""Run structural SAM QA gates before PEP calibration/solve."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.qa.reporting import format_report_summary
from equilibria.qa.sam_checks import run_sam_qa_from_file


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 SAM QA contracts")
    parser.add_argument("--sam-file", type=Path, required=True)
    parser.add_argument("--dynamic-sam", action="store_true")
    parser.add_argument("--acc-gvt", type=str, default="gvt")
    parser.add_argument("--acc-row", type=str, default="row")
    parser.add_argument("--acc-td", type=str, default="td")
    parser.add_argument("--acc-ti", type=str, default="ti")
    parser.add_argument("--acc-tm", type=str, default="tm")
    parser.add_argument("--acc-tx", type=str, default="tx")
    parser.add_argument("--acc-inv", type=str, default="inv")
    parser.add_argument("--acc-vstk", type=str, default="vstk")
    parser.add_argument("--balance-rel-tol", type=float, default=1e-6)
    parser.add_argument("--gdp-rel-tol", type=float, default=0.08)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--save-report",
        type=Path,
        default=Path("output/sam_qa_report.json"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if not args.sam_file.exists():
        print(f"SAM file not found: {args.sam_file}")
        return 1

    accounts = {
        "gvt": args.acc_gvt,
        "row": args.acc_row,
        "td": args.acc_td,
        "ti": args.acc_ti,
        "tm": args.acc_tm,
        "tx": args.acc_tx,
        "inv": args.acc_inv,
        "vstk": args.acc_vstk,
    }

    report = run_sam_qa_from_file(
        sam_file=args.sam_file,
        dynamic_sam=args.dynamic_sam,
        accounts=accounts,
        balance_rel_tol=args.balance_rel_tol,
        gdp_rel_tol=args.gdp_rel_tol,
        max_samples=args.max_samples,
    )

    print(format_report_summary(report))
    if not report.passed:
        print("Failed checks:")
        for check in report.checks:
            if check.passed:
                continue
            print(
                f"  - {check.code} [{check.category}] "
                f"failures={check.failures} max_rel={check.max_rel_delta:.3e}"
            )

    args.save_report.parent.mkdir(parents=True, exist_ok=True)
    report.save_json(args.save_report)
    print(f"Saved report: {args.save_report}")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
