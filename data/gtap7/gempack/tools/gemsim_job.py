"""GEMSIM runs of a TAB that tablo compiled: Horridge's SIMPLE and the levels TABs.

``horridge``  tablo + gemsim + sltoht on data/horridge_simple (step 3; Task 6).
``levels6``   gemsim on gtapv7_levels for a matrix job, levels cross-checked
              against the recorded sl4dump.har (step 4; Task 7).
``gragg``     gemsim on gtapuv7_levels (gtapv7_levels for gtapv7.exp) for one
              Burfisher .exp (step 5; Task 8). Johansen 1-step becomes Gragg
              2-4-6, TBL53's target% becomes plain shocks (target_shocks.py),
              and the run counts only if GEMSIM ends normally.

``compile_levels_tab`` compiles one TAB of data/gtap7/gempack_rungtap375_levels
into a work folder outside data/, once per content (Tasks 7 and 8).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import target_shocks
from _job_cli import common, not_yet
from release_check import ReleaseMismatch, require

from equilibria.sources import check_tree, own_files

DIR_EXP = ("me8", "tbl813")  # run their -dir.exp (rate% translated)
NORMAL_END = "completed without error"
OUTPUTS = ("solution.sl4", "updated.har")  # a gragg run without both did not finish


class JobError(RuntimeError):
    """A job that cannot be run faithfully."""


def run(cmd: list[str], cwd: Path, console: Path) -> int:
    """Run a GEMPACK program in ``cwd``, its stdout and stderr into ``console``."""
    with console.open("wb") as fh:
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                              check=False).returncode  # fmt: skip


def _digest(*files: Path) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def compile_levels_tab(
    name: str, *, levels_model: Path, work: Path, tablo: Path, root: Path
) -> None:
    """tablo on ``levels_model/<name>.tab`` in ``work``; skipped if done for the same files.

    Refuses when the model folder's record does not verify: what is compiled
    must be what data/ records.
    """
    where = levels_model.relative_to(root)
    problems = [m for m in check_tree(root) if m.startswith(str(where) + ":")]
    if problems:
        raise JobError(f"{where} does not verify: " + "; ".join(problems))
    sources = [levels_model / f"{name}.tab", levels_model / f"{name}.sti"]
    digest = _digest(*sources)
    stamp = work / f"{name}.compiled"
    ready = all((work / f"{name}{e}").exists() for e in (".gss", ".gst"))
    if ready and stamp.exists() and stamp.read_text() == digest:
        return
    work.mkdir(parents=True, exist_ok=True)
    for f in sources:
        shutil.copy2(f, work / f.name)
    rc = run([str(tablo), "-sti", f"{name}.sti"], work, work / "tablo_console.txt")
    if rc != 0 or not all((work / f"{name}{e}").exists() for e in (".gss", ".gst")):
        raise JobError(f"tablo {name}: rc={rc}, see {work / 'tablo_console.txt'}")
    stamp.write_text(digest)


def gragg_method(exp: str) -> tuple[str, str]:
    """The .exp with Johansen 1-step rewritten to Gragg 2-4-6, and how it was run."""
    johansen = re.compile(r"(?im)^\s*Method\s*=\s*Johansen\s*;")
    how = "as the .exp"
    if johansen.search(exp):
        exp = johansen.sub("Method = Gragg;", exp)
        exp = re.sub(r"(?im)^\s*Steps\s*=[^;]*;", "Steps = 2 4 6;", exp)
        how = "Johansen->Gragg 2-4-6"
    if len(re.findall(r"(?im)^\s*Method\s*=", exp)) != 1:
        raise JobError("the .exp must have exactly one Method line")
    return exp, how


def exp_file(package: Path, experiment: str) -> Path:
    return package / (f"{experiment}-dir.exp" if experiment in DIR_EXP else f"{experiment}.exp")


def _cmf(aux: Path, out: Path, exp: str) -> str:
    head = [
        f'auxiliary files = "{aux}";',
        "file GTAPSETS = sets.har;",
        "file GTAPDATA = basedata.har;",
        f'file GTAPSUM = "{out / "summary.har"}";',
        f'file WELVIEW = "{out / "decomp.har"}";',
        f'file GTAPVOL = "{out / "volume.har"}";',
        f'updated file GTAPDATA = "{out / "updated.har"}";',
        f'solution file = "{out / "solution"}";',
        "CPU = yes;",
        "NDS = yes;",
        "",
    ]
    return "\r\n".join(head) + exp


def run_gragg(args: argparse.Namespace) -> int:
    out: Path = args.out.resolve()
    source = exp_file(args.package, args.experiment)
    exp, how = gragg_method(source.read_text(encoding=target_shocks.ENCODING))
    exp, targets = target_shocks.translate(exp, args.package)
    cmf = out / "run.cmf"
    cmf.write_bytes(_cmf(args.aux, out, exp).encode(target_shocks.ENCODING))
    console = out / "gemsim_console.txt"
    # the .exp names its .prm and .shk without a path: run in a copy of the package
    with tempfile.TemporaryDirectory(prefix="gragg_") as tmp:
        pkg = Path(tmp)
        for f in own_files(args.package):
            target = pkg / f.relative_to(args.package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        rc = run([str(args.gemsim), "-cmf", str(cmf)], pkg, console)
    for scratch in out.glob("*GPXXX*"):  # GEMSIM's work files (run-GPXXX7.ud8)
        scratch.unlink()
    text = console.read_text(encoding=target_shocks.ENCODING, errors="replace")
    try:
        require(console, args.expect_release)
    except ReleaseMismatch as err:
        print(f"FAILED {args.experiment}: {err}", file=sys.stderr)
        return 1
    missing = [f for f in OUTPUTS if not (out / f).exists()]
    if rc != 0 or NORMAL_END not in text or missing:
        print(f"FAILED {args.experiment}: rc={rc}, normal end={NORMAL_END in text}, "
              f"missing={missing} (see {console.name})", file=sys.stderr)  # fmt: skip
        return 1
    if targets:
        bad = target_shocks.check(out / "updated.har", targets)
        if bad:
            print(f"FAILED {args.experiment}: target% not reached:\n  " + "\n  ".join(bad),
                  file=sys.stderr)  # fmt: skip
            return 1
    print(f"OK {args.experiment}: method {how}"
          + (f", {len(targets)} target% cells reached" if targets else ""))  # fmt: skip
    return 0


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
    if args.mode == "gragg":
        try:
            return run_gragg(args)
        except (JobError, target_shocks.TargetError) as err:
            print(f"FAILED {args.experiment}: {err}", file=sys.stderr)
            return 1
    task = {"horridge": "Task 6", "levels6": "Task 7"}[args.mode]
    return not_yet(f"gemsim_job {args.mode}", task)


if __name__ == "__main__":
    sys.exit(main())
