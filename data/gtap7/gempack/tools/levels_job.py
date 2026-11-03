"""sltoht on a GEMPACK solution: its sl4dump.har and, with --levels, its levels.

``dump``     a recorded solution folder (steps 1, 6a, 6b, 7; Tasks 4 and 9).
``resolve``  re-solves a gtapv7.exe job keeping its .sl4, then exports levels
             cross-checked against the recorded sl4dump.har (step 2 and the
             swap runs; Tasks 5 and 10).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _job_cli import common, not_yet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    dump = sub.add_parser("dump")
    dump.add_argument("--solution", type=Path, required=True)
    dump.add_argument("--sltoht", type=Path, required=True)
    dump.add_argument("--levels", action="store_true")
    common(dump)
    resolve = sub.add_parser("resolve")
    resolve.add_argument("--dataset", required=True)
    resolve.add_argument("--shock", type=float, required=True)
    resolve.add_argument("--steps", required=True)
    how = resolve.add_mutually_exclusive_group(required=True)
    how.add_argument("--rordelta", type=int)
    how.add_argument("--template", type=Path, help="a swap-closure .cmf")
    resolve.add_argument("--sl4dump", type=Path, required=True)
    resolve.add_argument("--gtapv7", type=Path, required=True)
    resolve.add_argument("--sltoht", type=Path, required=True)
    common(resolve)
    args = ap.parse_args(argv)
    if args.mode == "dump":
        return not_yet("levels_job dump", "Tasks 4 and 9")
    return not_yet("levels_job resolve", "Tasks 5 and 10")


if __name__ == "__main__":
    sys.exit(main())
