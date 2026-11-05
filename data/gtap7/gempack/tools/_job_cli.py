"""What every job module shares on its command line.

``--out`` the result folder, ``--expect-release`` the GEMPACK release of the
program that makes the folder's primary output, ``--release NAME=REL`` the
release of each other program it runs (gtapv7.exe 12.1.003 next to sltoht
11.3.004), and ``--work`` the git-ignored folder where programs run (never
data/ or %TEMP%: GEMPACK writes its current directory into consoles and .sl4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

NOT_YET = 3


def common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", type=Path, required=True, help="the result folder")
    ap.add_argument("--expect-release", required=True,
                    help="GEMPACK release the primary program must print")
    ap.add_argument("--release", action="append", default=[], metavar="NAME=REL",
                    help="release another program must print, e.g. sltoht.exe=11.3.004")
    ap.add_argument("--work", type=Path, default=None,
                    help="where programs run (git-ignored, never under data/)")  # fmt: skip


def releases(args: argparse.Namespace) -> dict[str, str]:
    """``--release`` as {program name (lower case): release}."""
    out = {}
    for pair in args.release:
        name, _, rel = pair.partition("=")
        out[name.lower()] = rel
    return out


def not_yet(what: str, task: str) -> int:
    print(f"{what}: not implemented yet ({task} of plan-phase-3a2.md)", file=sys.stderr)
    return NOT_YET
