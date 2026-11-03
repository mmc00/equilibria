"""One of Burfisher's book experiments solved by RunGTAP's gtapuv7.exe (or gtapv7.exe).

The .cmf is a header (sets, base data, output files) plus the .exp verbatim, as
tools/gempack-oracle/01-run-all.bat wrote it; me8 and tbl813 run their -dir.exp.
Phase 3a-2 Task 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _job_cli import common, not_yet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--package", type=Path, required=True, help="data/gtap7/nus333")
    ap.add_argument("--exe", type=Path, required=True, help="gtapuv7.exe or gtapv7.exe")
    common(ap)
    ap.parse_args(argv)
    return not_yet("burfisher_job", "Task 3")


if __name__ == "__main__":
    sys.exit(main())
