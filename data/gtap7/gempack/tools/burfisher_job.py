"""One of Burfisher's book experiments solved by RunGTAP's gtapuv7.exe (or gtapv7.exe).

The input of corridas_paper.ps1 steps 1 and 6a, as tools/gempack-oracle/
01-run-all.bat made it on 2026-09-24: the .cmf is a header (aux files, sets,
base data, output files) plus the .exp verbatim, so the book's method stays
(Johansen 1-step where the book has it); me8 and tbl813 run their -dir.exp.
gtapv7.exp declares the closure of the condensed model and runs gtapv7.exe.
The run counts only with a normal end, both outputs and the expected release.
Phase 3a-2 Task 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import target_shocks
from _job_cli import common
from gemsim_job import NOTE_FILE, cmf_text, exp_file, finish, run_in_package_copy
from gemsim_job import run as _run

run = _run  # tests replace it
__all__ = ["NOTE_FILE", "main", "run"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--package", type=Path, required=True, help="data/gtap7/nus333")
    ap.add_argument("--exe", type=Path, required=True, help="gtapuv7.exe or gtapv7.exe")
    ap.add_argument("--work", type=Path, required=True,
                    help="where the package copy runs (git-ignored, never under data/)")
    common(ap)
    args = ap.parse_args(argv)  # fmt: skip

    out: Path = args.out.resolve()
    exp = exp_file(args.package, args.experiment).read_bytes().decode(target_shocks.ENCODING)
    cmf = out / "run.cmf"
    # RunGTAP's compiled model: gtapuv7.exe next to its gtapuv7.axs/.axt
    cmf.write_bytes(cmf_text(args.exe.with_suffix(""), out, exp).encode(target_shocks.ENCODING))
    console = out / "solve_console.txt"
    rc = run_in_package_copy(args.exe, cmf, package=args.package, work=args.work,
                             name=args.experiment, console=console, runner=run)  # fmt: skip
    if not finish(out, console, rc, expect_release=args.expect_release, name=args.experiment):
        return 1
    print(f"OK {args.experiment}: {args.exe.name}, method as the book")
    return 0


if __name__ == "__main__":
    sys.exit(main())
