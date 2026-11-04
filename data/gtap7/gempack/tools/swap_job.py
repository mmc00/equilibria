"""RunGTAP's gtapv7.exe with the retired swap closure (dpsave = del_tbalry).

``swap``   runs/{nus333,9x10}_compare/rungtap/tm10.cmf (kept in tools/cmf/) with
           only the Auxiliary files line and the shock changed; collected as the
           3a-1 matrix records (Task 10).
``abort``  runs/gempack_20x41_validation/tm10.cmf: recorded only when GEMPACK
           aborts with the known signature (LOGE of a negative number in E_u) and its
           assert-arith dump; a converged run or another error records nothing
           (Task 11).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import target_shocks
from _job_cli import common, releases
from gemsim_job import NORMAL_END, JobError, _collect, _fresh, _normal_end, _work
from gemsim_job import check_release, run_retry, sltoht_dump
from run_gempack_matrix import INPUT_FILES

from equilibria.sources import dataset_dir

_AUX = re.compile(r"(?im)^(?P<k>[ \t]*Auxiliary files[ \t]*=[ \t]*)[^;\r\n]*?(?P<e>[ \t]*;)")
_SHOCK = re.compile(r"(?im)^(?P<k>[ \t]*Shock tm[ \t]*=[ \t]*uniform[ \t]+)[-+\d.eE]+(?P<e>[ \t]*;)")
# runs/gempack_20x41_validation/solve_console.txt lines 2725-2727 (RESULTADO.md)
ABORT_SIGNATURE = ('in equation "E_u"', "Attempt to take LOGE of zero or of a negative number")


def crlf(text: str) -> str:
    """``text`` with CRLF line ends: every .cmf gtapv7.exe solved in 3a-1 and in the
    3a-2 smoke runs was CRLF; the templates keep git's LF bytes of runs/."""
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def from_template(template: str, *, shock: float, aux: Path) -> str:
    """``template`` with only its Auxiliary files line and its tm shock changed."""
    for pattern, what in ((_AUX, "Auxiliary files"), (_SHOCK, "Shock tm = uniform")):
        if len(pattern.findall(template)) != 1:
            raise JobError(f"the template must have exactly one '{what}' line")
    out = _AUX.sub(lambda m: f"{m['k']}{aux}{m['e']}", template)
    return _SHOCK.sub(lambda m: f"{m['k']}{shock:g}{m['e']}", out)


def _setup(args: argparse.Namespace, stem: str) -> tuple[Path, Path]:
    """A run folder in --work with the dataset's inputs and ``<stem>.cmf``."""
    out: Path = args.out.resolve()
    wd = _fresh(_work(args) / "swap" / f"{args.dataset}_{out.name}")
    for f in INPUT_FILES:
        shutil.copy2(dataset_dir(args.dataset) / f, wd / f)
    text = from_template(args.template.read_bytes().decode(target_shocks.ENCODING),
                         shock=args.shock, aux=args.gtapv7.with_suffix(""))  # fmt: skip
    (wd / f"{stem}.cmf").write_bytes(crlf(text).encode(target_shocks.ENCODING))
    return out, wd


def run_swap(args: argparse.Namespace) -> int:
    out, wd = _setup(args, "run")
    name = f"{args.dataset} tm{args.shock:g}"
    console = out / "solve_console.txt"
    rc = run_retry([str(args.gtapv7), "-cmf", "run.cmf"], wd, console, note_dir=out)
    if not check_release(console, args.expect_release, name):
        return 1
    if rc != 0 or not _normal_end(console) or not (wd / "run.sl4").exists():
        print(f"FAILED {name}: rc={rc}, normal end={_normal_end(console)}", file=sys.stderr)
        return 1
    dump_console = out / "sltoht_console.txt"
    sltoht_dump(wd / "run.sl4", args.sltoht, "sl4dump.har", console=dump_console, note_dir=out)
    sltoht_rel = releases(args).get("sltoht.exe", args.expect_release)
    if not check_release(dump_console, sltoht_rel, name) or not (wd / "sl4dump.har").exists():
        print(f"FAILED {name}: no sl4dump.har (see {dump_console.name})", file=sys.stderr)
        return 1
    _collect(out, [(wd / "updated.har", "updated.har"), (wd / "decomp.har", "decomp.har"),
                   (wd / "sl4dump.har", "sl4dump.har"), (wd / "run.cmf", "run.cmf")])  # fmt: skip
    shutil.rmtree(wd, ignore_errors=True)
    print(f"OK swap {name}")
    return 0


def run_abort(args: argparse.Namespace) -> int:
    # the .cmf keeps its name, tm10: gtapv7.exe calls the dump <cmf>-assert-arith-fail.har
    out, wd = _setup(args, "tm10")
    name = f"{args.dataset} tm{args.shock:g} (expected abort)"
    console = out / "solve_console.txt"
    rc = run_retry([str(args.gtapv7), "-cmf", "tm10.cmf"], wd, console, note_dir=out)
    if not check_release(console, args.expect_release, name):
        return 1
    text = console.read_text(encoding=target_shocks.ENCODING, errors="replace")
    dump = wd / "tm10-assert-arith-fail.har"
    signature = all(s in text for s in ABORT_SIGNATURE)
    if NORMAL_END in text or not signature or not dump.exists():
        print(f"FAILED {name}: rc={rc}, signature={signature}, "
              f"dump={dump.exists()}, normal end={NORMAL_END in text}", file=sys.stderr)  # fmt: skip
        return 1
    _collect(out, [(wd / "tm10.cmf", "run.cmf"), (dump, "assert_arith_fail.har")])
    shutil.rmtree(wd, ignore_errors=True)
    print(f"OK {name}: GEMPACK aborted as recorded (rc={rc})")
    return 0


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
    try:
        return run_swap(args) if args.mode == "swap" else run_abort(args)
    except JobError as err:
        print(f"FAILED {args.mode}: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
