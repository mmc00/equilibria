"""sltoht on a GEMPACK solution: its sl4dump.har and, with --levels, its levels.

``dump``     a recorded solution folder (its solution.sl4): sl4dump.har and,
             with --levels, the levels exported against that same dump
             (steps 1, 6a, 6b and 7; Tasks 4 and 9).
``resolve``  re-solves a gtapv7.exe job keeping its .sl4 (run_gempack_matrix's
             prepare, or a swap-closure template), then exports the levels
             against the RECORDED sl4dump.har. Every variable must match: a
             mismatch means the re-solve is not the recorded solution, and
             nothing is recorded (step 2 and the swap levels; Tasks 5 and 10).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import target_shocks
from _job_cli import common, releases
from gemsim_job import JobError, _collect, _fresh, _normal_end, _work
from gemsim_job import check_release, export_levels, run_retry, sltoht_dump
from run_gempack_matrix import INPUT_FILES, prepare
from swap_job import crlf, from_template

from equilibria.sources import dataset_dir


def run_dump(args: argparse.Namespace) -> int:
    out: Path = args.out.resolve()
    name = f"{args.solution.parent.name}/{args.solution.name}"
    wd = _fresh(_work(args) / "dump" / f"{out.parent.name}_{out.name}")
    sl4 = wd / "solution.sl4"
    shutil.copy2(args.solution / "solution.sl4", sl4)
    console = out / "sltoht_console.txt"
    sltoht_dump(sl4, args.sltoht, "sl4dump.har", console=console, note_dir=out)
    if not check_release(console, args.expect_release, name):
        return 1
    if not (wd / "sl4dump.har").exists():
        print(f"FAILED {name}: no sl4dump.har (see {console.name})", file=sys.stderr)
        return 1
    pairs = [(wd / "sl4dump.har", "sl4dump.har")]
    if args.levels:
        rc, mismatch = export_levels(sl4, wd / "sl4dump.har", args.sltoht, wd / "sl4levels.har",
                                     console=out / "export_console.txt", note_dir=out)  # fmt: skip
        # against its own dump every variable must agree
        if rc != 0 or mismatch != 0 or not (wd / "solution_levels.txt").exists():
            print(f"FAILED {name}: levels export rc={rc}, xcheck-mismatch={mismatch}",
                  file=sys.stderr)  # fmt: skip
            return 1
        # levels.txt (sltoht -SHL -SIC) is not kept: sltoht rebuilds it from
        # solution.sl4; sl4levels.har is the record (maintainer, 2026-11-04)
        pairs.append((wd / "sl4levels.har", "sl4levels.har"))
    _collect(out, pairs)
    shutil.rmtree(wd, ignore_errors=True)
    print(f"OK dump {name}" + (" with levels" if args.levels else ""))
    return 0


def run_resolve(args: argparse.Namespace) -> int:
    out: Path = args.out.resolve()
    name = f"{args.dataset} {out.name}"
    root = _fresh(_work(args) / "resolve" / f"{args.dataset}_{out.name}")
    aux = args.gtapv7.with_suffix("")
    if args.template is None:
        run_dir, tag = prepare(args.dataset, args.shock, args.steps.replace("-", " "),
                               args.rordelta, run_root=root, aux=str(aux))  # fmt: skip
    else:
        run_dir, tag = root, "run"
        for f in INPUT_FILES:
            shutil.copy2(dataset_dir(args.dataset) / f, run_dir / f)
        text = from_template(args.template.read_bytes().decode(target_shocks.ENCODING),
                             shock=args.shock, aux=aux)  # fmt: skip
        (run_dir / "run.cmf").write_bytes(crlf(text).encode(target_shocks.ENCODING))
    console = out / "solve_console.txt"
    rc = run_retry([str(args.gtapv7), "-cmf", f"{tag}.cmf"], run_dir, console, note_dir=out)
    gtapv7_rel = releases(args).get("gtapv7.exe")
    if gtapv7_rel is None:
        raise JobError("--release gtapv7.exe=<release> is required: gtapv7.exe re-solves")
    sl4 = run_dir / f"{tag}.sl4"
    if not check_release(console, gtapv7_rel, name):
        return 1
    if rc != 0 or not _normal_end(console) or not sl4.exists():
        print(f"FAILED {name}: gtapv7 rc={rc}, normal end={_normal_end(console)}",
              file=sys.stderr)  # fmt: skip
        return 1
    export_console = out / "export_console.txt"
    rc, mismatch = export_levels(sl4, args.sl4dump, args.sltoht, run_dir / "sl4levels.har",
                                 console=export_console, note_dir=out)  # fmt: skip
    if rc != 0 or mismatch != 0 or not (run_dir / f"{tag}_levels.txt").exists():
        print(f"FAILED {name}: the re-solve is not the recorded solution "
              f"(export rc={rc}, xcheck-mismatch={mismatch}; see {export_console.name})",
              file=sys.stderr)  # fmt: skip
        return 1
    # levels.txt (sltoht -SHL -SIC) is not kept: 60 MB for 20x41, and sltoht
    # rebuilds it from solution.sl4; sl4levels.har is the record (maintainer, 2026-11-04)
    _collect(out, [(run_dir / f"{tag}.cmf", "run.cmf"), (sl4, "solution.sl4"),
                   (run_dir / "sl4levels.har", "sl4levels.har")])  # fmt: skip
    shutil.rmtree(root, ignore_errors=True)
    print(f"OK resolve {name}: levels match the recorded sl4dump")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    dump = sub.add_parser("dump")
    dump.add_argument("--solution", type=Path, required=True, help="a recorded result folder")
    dump.add_argument("--sltoht", type=Path, required=True)
    dump.add_argument("--levels", action="store_true")
    common(dump)
    resolve = sub.add_parser("resolve")
    resolve.add_argument("--dataset", required=True)
    resolve.add_argument("--shock", type=float, required=True)
    resolve.add_argument("--steps", required=True, help='as recorded, e.g. "8-16-32"')
    how = resolve.add_mutually_exclusive_group(required=True)
    how.add_argument("--rordelta", type=int)
    how.add_argument("--template", type=Path, help="a swap-closure .cmf")
    resolve.add_argument("--sl4dump", type=Path, required=True)
    resolve.add_argument("--gtapv7", type=Path, required=True)
    resolve.add_argument("--sltoht", type=Path, required=True)
    common(resolve)
    args = ap.parse_args(argv)
    try:
        return run_dump(args) if args.mode == "dump" else run_resolve(args)
    except JobError as err:
        print(f"FAILED {args.mode}: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
