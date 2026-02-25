"""Utility script that executes the manual IEEM->PEP pipeline step by step."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from equilibria.sam_tools.manual_pipeline import run_from_excel

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "tmp_ieem_test" / "raw.xlsx"


def main() -> None:
    parser = ArgumentParser(description="Run the manual IEEM->PEP pipeline on a raw SAM workbook.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the raw IEEM Excel file")
    parser.add_argument("--sheet", type=str, default="MCS2016", help="Sheet name within the workbook")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input SAM file not found: {input_path}")

    summary = run_from_excel(input_path, sheet_name=args.sheet)
    print("Manual pipeline completed")
    print("Matrix shape:", summary.sam.matrix.shape)
    print("Total flow:", summary.total_flow)
    for step in summary.steps:
        details = {k: v for k, v in step.items() if k != "step"}
        print(f" - {step['step']}: {details}")


if __name__ == "__main__":
    main()
