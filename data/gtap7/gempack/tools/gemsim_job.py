"""GEMSIM runs of a TAB that tablo compiled: Horridge's SIMPLE and the levels TABs.

``horridge``  tablo + gemsim + sltoht on data/horridge_simple (step 3; Task 6).
``levels6``   gemsim on gtapv7_levels for a matrix job, levels cross-checked
              against the recorded sl4dump.har (step 4; Task 7).
``gragg``     gemsim on gtapuv7_levels for a Burfisher .exp, Johansen 1-step
              rewritten to Gragg 2-4-6 (step 5; Task 8).

``compile_levels_tab`` compiles data/gtap7/gempack_rungtap375_levels into a
work folder outside data/ (Tasks 7 and 8).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _job_cli import common, not_yet


def compile_levels_tab(
    name: str, *, levels_model: Path, work: Path, tablo: Path, root: Path
) -> None:
    raise NotImplementedError("compile_levels_tab: Tasks 7 and 8 of plan-phase-3a2.md")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    horridge = sub.add_parser("horridge")
    horridge.add_argument("--experiment", required=True)
    horridge.add_argument("--model-dir", type=Path, required=True)
    for exe in ("--tablo", "--gemsim", "--sltoht"):
        horridge.add_argument(exe, type=Path, required=True)
    common(horridge)
    levels6 = sub.add_parser("levels6")
    levels6.add_argument("--dataset", required=True)
    levels6.add_argument("--rordelta", type=int, required=True)
    levels6.add_argument("--aux", type=Path, required=True)
    levels6.add_argument("--sl4dump", type=Path, required=True)
    levels6.add_argument("--gemsim", type=Path, required=True)
    levels6.add_argument("--sltoht", type=Path, required=True)
    common(levels6)
    gragg = sub.add_parser("gragg")
    gragg.add_argument("--experiment", required=True)
    gragg.add_argument("--package", type=Path, required=True)
    gragg.add_argument("--aux", type=Path, required=True)
    gragg.add_argument("--gemsim", type=Path, required=True)
    common(gragg)
    args = ap.parse_args(argv)
    task = {"horridge": "Task 6", "levels6": "Task 7", "gragg": "Task 8"}[args.mode]
    return not_yet(f"gemsim_job {args.mode}", task)


if __name__ == "__main__":
    sys.exit(main())
