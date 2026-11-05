"""GEMSIM runs of a TAB that tablo compiled, and what every job module shares.

``horridge``  tablo + gemsim + sltoht on Horridge's SIMPLE (step 3; Task 6).
``levels6``   gemsim on gtapv7_levels for a matrix job; its levels exported
              against the recorded (unpatched) sl4dump.har, drops named (step 4;
              Task 7).
``gragg``     gemsim on gtapuv7_levels (gtapv7_levels for gtapv7.exp) for one
              Burfisher .exp (step 5; Task 8). Johansen 1-step becomes Gragg
              2-4-6, TBL53's target% becomes plain shocks (target_shocks.py).

Shared by every job module: ``run`` (the one place a program runs; tests
replace it), ``run_retry`` (one retry when a program never starts, said in the
record), ``finish``/``check_release`` (a run counts only with the expected
release, a normal end and its outputs), ``sltoht_dump`` and ``export_levels``.
``compile_levels_tab`` compiles one TAB of data/gtap7/gempack_rungtap375_levels
into the work folder, once per content.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import target_shocks
from _job_cli import common, releases
from release_check import ReleaseMismatch, release, require
from run_gempack_matrix import prepare

from equilibria.sources import check_tree, own_files

TOOLS = Path(__file__).resolve().parent
DIR_EXP = ("me8", "tbl813")  # run their -dir.exp (rate% translated)
NORMAL_END = "completed without error"
OUTPUTS = ("solution.sl4", "updated.har")  # a Burfisher run without both did not finish
NOTE_FILE = "retry_note.txt"  # write_source --notes-from: appended to the record's notes
RETRY_PAUSE_S = 15  # before running a program that did not start once more
_MISMATCH = re.compile(r"xcheck-mismatch=(\d+)")


class JobError(RuntimeError):
    """A job that cannot be run faithfully."""


def run(cmd: list[str], cwd: Path, console: Path) -> int:
    """Run a program in ``cwd``, its stdout and stderr into ``console``."""
    with console.open("wb") as fh:
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                              check=False).returncode  # fmt: skip


def _runner(runner: Callable[..., int] | None) -> Callable[..., int]:
    return runner or sys.modules[__name__].run  # looked up now: tests replace it


def run_retry(
    cmd: list[str], cwd: Path, console: Path, *, note_dir: Path,
    setup: Callable[[], None] | None = None, runner: Callable[..., int] | None = None,
) -> int:
    """``cmd`` in ``cwd``; once more if the program never started.

    Now and then a GEMPACK program never starts (3a-1, me4b: no banner in the
    console, rc 0xC0000142). It is retried once, and ``note_dir/NOTE_FILE`` says
    so for the record (write_source --notes-from). A program that started and
    failed is a result, not retried. ``setup`` rebuilds ``cwd`` before each attempt.
    """
    for attempt in (1, 2):
        if setup is not None:
            setup()
        if attempt == 2:
            time.sleep(RETRY_PAUSE_S)  # an immediate retry failed like the first attempt
        rc = _runner(runner)(cmd, cwd, console)
        if attempt == 2 or release(console) is not None:
            return rc
        note = note_dir / NOTE_FILE
        earlier = note.read_text(encoding="utf-8") + "; " if note.exists() else ""
        note.write_text(
            earlier + f"{Path(cmd[0]).name} did not start on the first attempt "
            f"(no GEMPACK banner, rc={rc}); retried once",
            encoding="utf-8",
        )
    raise AssertionError("unreachable")


def check_release(console: Path, expected: str, name: str) -> bool:
    try:
        require(console, expected)
    except ReleaseMismatch as err:
        print(f"FAILED {name}: {err}", file=sys.stderr)
        return False
    return True


def _normal_end(console: Path) -> bool:
    return NORMAL_END in console.read_text(encoding=target_shocks.ENCODING, errors="replace")


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
    rc = run_retry([str(tablo), "-sti", f"{name}.sti"], work, work / "tablo_console.txt",
                   note_dir=work)  # fmt: skip
    if rc != 0 or not all((work / f"{name}{e}").exists() for e in (".gss", ".gst")):
        raise JobError(f"tablo {name}: rc={rc}, see {work / 'tablo_console.txt'}")
    stamp.write_text(digest)


def sltoht_dump(sl4: Path, sltoht: Path, out_name: str, *, console: Path, note_dir: Path) -> int:
    """sltoht in batch on ``sl4``: its %-changes as the HAR ``out_name`` beside it.

    The answers of corridas_paper.ps1's Dump-Changes and run_gempack_matrix's
    convert_sl4: <solution> / c / n / har / <out> / e.
    """
    sti = sl4.parent / f"{Path(out_name).stem}.sti"
    sti.write_text(f"\n{sl4.stem}\nc\nn\nhar\n{out_name}\ne\n", encoding="ascii")
    return run_retry([str(sltoht), "-sti", sti.name], sl4.parent, console, note_dir=note_dir)


def export_levels(
    sl4: Path, sl4dump: Path, sltoht: Path, out_har: Path, *, console: Path, note_dir: Path
) -> tuple[int, int | None]:
    """export_sl4_levels.py on ``sl4`` against ``sl4dump``: (rc, xcheck mismatches).

    It leaves ``<stem>_levels.txt`` (sltoht -SHL -SIC) beside ``sl4``.
    """
    cmd = [sys.executable, str(TOOLS / "export_sl4_levels.py"), str(sl4),
           "--sl4dump", str(sl4dump), "--sltoht", str(sltoht), "--out", str(out_har)]  # fmt: skip
    rc = _runner(None)(cmd, sl4.parent, console)
    m = _MISMATCH.search(console.read_text(encoding="utf-8", errors="replace"))
    return rc, (int(m.group(1)) if m else None)


def _fresh(folder: Path) -> Path:
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    return folder


def _collect(out: Path, pairs: list[tuple[Path, str]]) -> None:
    for src, name in pairs:
        shutil.copy2(src, out / name)


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


def cmf_text(aux: Path, out: Path, exp: str) -> str:
    """The header of 01-run-all.bat and step 5 (aux files, data, outputs) + the .exp."""
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


def run_in_package_copy(
    exe: Path, cmf: Path, *, package: Path, work: Path, name: str, console: Path,
    runner: Callable[..., int] | None = None,
) -> int:
    """``exe -cmf cmf`` in a copy of ``package`` under ``work/package/<name>``.

    The .exp names its .prm and .shk without a path. GEMPACK writes its current
    directory into the console and the .sl4, so the copy lives in the work
    folder (git-ignored, no user name), not %TEMP%; it is removed afterwards.
    """
    pkg = work.resolve() / "package" / name

    def setup() -> None:
        _fresh(pkg)
        for f in own_files(package):
            target = pkg / f.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)

    try:
        return run_retry([str(exe), "-cmf", str(cmf)], pkg, console,
                         note_dir=console.parent, setup=setup, runner=runner)  # fmt: skip
    finally:
        shutil.rmtree(pkg, ignore_errors=True)


def finish(out: Path, console: Path, rc: int, *, expect_release: str, name: str,
           outputs: tuple[str, ...] = OUTPUTS) -> bool:  # fmt: skip
    """Drop the work files, then whether the run counts: release, normal end, outputs."""
    # GEMPACK's work files (run-GPXXX7.ud8), the solution's .slc and the automatic
    # accuracy log .aal (times of day; the accuracy is in the console) are not
    # results: the .sl4 and the updated data are the record (maintainer, 2026-11-03)
    for scratch in [*out.glob("*GPXXX*"), *out.glob("*.slc"), *out.glob("*.aal")]:
        scratch.unlink()
    if not check_release(console, expect_release, name):
        return False
    missing = [f for f in outputs if not (out / f).exists()]
    if rc != 0 or not _normal_end(console) or missing:
        print(f"FAILED {name}: rc={rc}, normal end={_normal_end(console)}, "
              f"missing={missing} (see {console.name})", file=sys.stderr)  # fmt: skip
        return False
    return True


def _work(args: argparse.Namespace) -> Path:
    if args.work is None:
        raise JobError("--work is required")
    return args.work.resolve()


def run_gragg(args: argparse.Namespace) -> int:
    out: Path = args.out.resolve()
    source = exp_file(args.package, args.experiment)
    exp, how = gragg_method(source.read_text(encoding=target_shocks.ENCODING))
    exp, targets = target_shocks.translate(exp, args.package)
    cmf = out / "run.cmf"
    cmf.write_bytes(cmf_text(args.aux, out, exp).encode(target_shocks.ENCODING))
    console = out / "gemsim_console.txt"
    rc = run_in_package_copy(args.gemsim, cmf, package=args.package,
                             work=args.aux.resolve().parent.parent, name=args.experiment,
                             console=console)  # fmt: skip
    if not finish(out, console, rc, expect_release=args.expect_release, name=args.experiment):
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


def run_horridge(args: argparse.Namespace) -> int:
    """tablo -pgs simple, gemsim -cmf <exp>.cmf, sltoht: Horridge's SIMPLE (step 3)."""
    out: Path = args.out.resolve()
    exp = args.experiment
    wd = _fresh(_work(args) / "horridge" / exp)
    for f in ("simple.tab", "simdata.har", f"{exp}.cmf"):
        shutil.copy2(args.model_dir / f, wd / f)
    tablo_console = out / "tablo_console.txt"
    rc = run_retry([str(args.tablo), "-pgs", "simple"], wd, tablo_console, note_dir=out)
    if rc != 0 or not (wd / "simple.gss").exists():
        print(f"FAILED {exp}: tablo rc={rc} (see {tablo_console.name})", file=sys.stderr)
        return 1
    console = out / "gemsim_console.txt"
    rc = run_retry([str(args.gemsim), "-cmf", f"{exp}.cmf"], wd, console, note_dir=out)
    sl4, upd = wd / f"{exp}.sl4", wd / f"{exp}.UPD"  # the .cmf writes <cmf>.UPD
    if not check_release(console, args.expect_release, exp):
        return 1
    if rc != 0 or not _normal_end(console) or not (sl4.exists() and upd.exists()):
        print(f"FAILED {exp}: gemsim rc={rc}, normal end={_normal_end(console)}",
              file=sys.stderr)  # fmt: skip
        return 1
    dump_console = out / "sltoht_console.txt"
    rc = sltoht_dump(sl4, args.sltoht, "sl4dump.har", console=dump_console, note_dir=out)
    sltoht_rel = releases(args).get("sltoht.exe", args.expect_release)
    if not check_release(dump_console, sltoht_rel, exp) or not (wd / "sl4dump.har").exists():
        print(f"FAILED {exp}: sltoht rc={rc}", file=sys.stderr)
        return 1
    _collect(out, [(wd / f"{exp}.cmf", "run.cmf"), (sl4, "solution.sl4"),
                   (upd, "updated.har"), (wd / "sl4dump.har", "sl4dump.har")])  # fmt: skip
    shutil.rmtree(wd, ignore_errors=True)
    print(f"OK horridge {exp}")
    return 0


def run_levels6(args: argparse.Namespace) -> int:
    """gemsim on the patched condensed TAB for one matrix job (step 4)."""
    out: Path = args.out.resolve()
    name = f"{args.dataset} rordelta={args.rordelta}"
    root = _fresh(_work(args) / "levels6" / f"{args.dataset}_rdlt{args.rordelta}")
    run_dir, tag = prepare(args.dataset, 10.0, "8 16 32", args.rordelta, run_root=root,
                           aux=str(args.aux))  # fmt: skip
    console = out / "gemsim_console.txt"
    rc = run_retry([str(args.gemsim), "-cmf", f"{tag}.cmf"], run_dir, console, note_dir=out)
    sl4 = run_dir / f"{tag}.sl4"
    if not check_release(console, args.expect_release, name):
        return 1
    if rc != 0 or not _normal_end(console) or not sl4.exists():
        print(f"FAILED {name}: gemsim rc={rc}, normal end={_normal_end(console)}",
              file=sys.stderr)  # fmt: skip
        return 1
    # against the UNPATCHED run's sl4dump, as step 4: a variable whose %-change
    # differs is dropped and named (xcheck-mismatch=N [...]), and the run counts
    rc, mismatch = export_levels(sl4, args.sl4dump, args.sltoht, run_dir / "sl4levels.har",
                                 console=out / "export_console.txt", note_dir=out)  # fmt: skip
    if rc != 0 or not (run_dir / f"{tag}_levels.txt").exists():
        print(f"FAILED {name}: levels export rc={rc} (see export_console.txt)", file=sys.stderr)
        return 1
    # levels.txt (sltoht -SHL -SIC) is not kept: 60 MB for 20x41, and sltoht
    # rebuilds it from solution.sl4; sl4levels.har is the record (maintainer, 2026-11-04)
    pairs = [(run_dir / f"{tag}.cmf", "run.cmf"), (sl4, "solution.sl4"),
             (run_dir / "sl4levels.har", "sl4levels.har")]  # fmt: skip
    if args.rordelta == 0:  # the .prm GEMSIM read (RDLT 0), from _force_rdlt
        pairs.append((run_dir / f"default_{tag}.prm", "default_rdlt0.prm"))
    _collect(out, pairs)
    shutil.rmtree(root, ignore_errors=True)
    print(f"OK levels6 {name}: xcheck-mismatch={mismatch}")
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
    mode = {"horridge": run_horridge, "levels6": run_levels6, "gragg": run_gragg}[args.mode]
    try:
        return mode(args)
    except (JobError, target_shocks.TargetError) as err:
        print(f"FAILED {args.mode}: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
