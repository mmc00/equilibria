"""What every job module shares: its ``--out``/``--expect-release`` and the stub exit.

Phase 3a-2 builds the job list first (Task 2) and each module's real run in
its own task; until then a module parses its arguments and exits 3 naming the
task that implements it, so a run can never record a placeholder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

NOT_YET = 3


def common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", type=Path, required=True, help="the result folder")
    ap.add_argument("--expect-release", required=True,
                    help="GEMPACK release the program must print")  # fmt: skip


def not_yet(what: str, task: str) -> int:
    print(f"{what}: not implemented yet ({task} of plan-phase-3a2.md)", file=sys.stderr)
    return NOT_YET
