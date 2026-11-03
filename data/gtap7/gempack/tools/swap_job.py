"""RunGTAP's gtapv7.exe with the retired swap closure (dpsave = del_tbalry).

``swap``   runs/{nus333,9x10}_compare/rungtap/tm10.cmf with only the shock and
           the file names changed (Task 10).
``abort``  runs/gempack_20x41_validation/tm10.cmf: recorded only when GEMPACK
           aborts with the known signature (Task 11).
The .cmf templates are copied into tools/cmf/ by those tasks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _job_cli import common, not_yet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("swap", "abort"):
        p = sub.add_parser(mode)
        p.add_argument("--dataset", required=True)
        p.add_argument("--shock", type=float, required=True)
        p.add_argument("--template", type=Path, required=True)
        p.add_argument("--gtapv7", type=Path, required=True)
        if mode == "swap":
            p.add_argument("--sltoht", type=Path, required=True)
        common(p)
    args = ap.parse_args(argv)
    return not_yet(f"swap_job {args.mode}", "Task 10" if args.mode == "swap" else "Task 11")


if __name__ == "__main__":
    sys.exit(main())
