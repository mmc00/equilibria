"""Solve one GAMS GTAP reference into a folder (ADR-0010, phases 2a and 2b).

Builds the self-contained bundle of a dataset from the model in
data/gtap7/gams (pure: build_gtap7_pure_local_bundle.py, altertax:
build_gtap7_altertax_neos_bundle.py), solves it with the local GAMS and
leaves in --out only what replays the run: ``model.gms`` (the exact file
solved, data inlined), ``path.opt`` when the bundle writes one, and
``out.gdx``. Nothing else, so write_source.py can record the folder as is.

--where neos solves the same bundle on NEOS (neos_job.py, e-mail from
$NEOS_EMAIL) and adds ``model.lst``, ``neos.log`` and ``neos_job.json``, which
write_source.py --remote neos records. --out is created only on success.

--form mcp solves the bundle as built (PATH; the local license stops at 1000
rows, so only gtap7_3x3 ifSUB=1 fits). --form nlp switches ifMCP off and
solves ``gtap using nlp maximizing walras`` with IPOPT (no size limit).

Exit status is not 0, and --out holds no out.gdx, when GAMS fails or any
model status is not 1 (Optimal) or 2 (Locally Optimal).

    uv run python data/gtap7/gams/tools/regen_gams_reference.py --dataset gtap7_3x3 \\
        --mode altertax --form nlp --ifsub 1 --out /tmp/ref
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import neos_job

ROOT = Path(__file__).resolve().parents[4]
BUILDERS = {
    "gtap": Path(__file__).with_name("build_gtap7_pure_local_bundle.py"),
    "altertax": Path(__file__).with_name("build_gtap7_altertax_neos_bundle.py"),
}
_IFMCP = re.compile(r'(ifMCP\s+"[^"]*"\s+/\s*)1(\s*/)')
_NLP_SOLVE = re.compile(
    r"^(\s*)(solve (?:gtap|dynCal|dynGTAP) using nlp maximizing walras ;)", re.MULTILINE
)
_OK_STATUS = {"1", "2"}


def to_nlp(text: str) -> str:
    """The bundle solved as NLP with IPOPT (ifMCP = 0)."""
    text, n = _IFMCP.subn(r"\g<1>0\g<2>", text, count=1)
    if n != 1:
        raise SystemExit("bundle has no `ifMCP ... / 1 /` switch to turn off")
    return _NLP_SOLVE.sub(r"\1option nlp=ipopt;\n\1\2", text)


def _build(mode: str, dataset: str, ifsub: int, work: Path, fbep: str) -> Path:
    subprocess.run(
        [
            sys.executable,
            str(BUILDERS[mode]),
            "--dataset",
            dataset,
            "--ifsub",
            str(ifsub),
            "--out-dir",
            str(work),
            "--fbep",
            fbep,
        ],
        check=True,
        cwd=ROOT,
    )
    built = sorted(work.glob("comp_*.gms"))
    if len(built) != 1:
        raise SystemExit(f"builder left {len(built)} comp_*.gms in {work}, expected 1")
    return built[0]


def _ytax(dump: str, stream: str, period: str | None = None) -> list[float]:
    return [
        abs(float(line.rsplit(",", 1)[-1]))
        for line in dump.splitlines()
        if f'"{stream}"' in line and (period is None or f'"{period}"' in line)
    ]


def _verify_subsidy_aware(gdxdump: str, gdx: Path, fbep: str) -> None:
    """The altertax reference carries the factor subsidy only when it was injected.

    header: nonzero ytax(ft) and ytax(fs) at check. zero: ytax(fs) is zero in
    every period, since the official getData.gms sets fbep = 0 (#187).
    """
    dump = subprocess.run(
        [gdxdump, str(gdx), "Symb=ytax", "Format=csv"], capture_output=True, text=True
    ).stdout
    if fbep == "zero":
        if any(v > 1e-6 for v in _ytax(dump, "fs")):
            raise SystemExit("ytax(fs) is nonzero with fbep=zero: FBEP was injected")
        return
    for stream in ("ft", "fs"):
        if not any(v > 1e-6 for v in _ytax(dump, stream, "check")):
            raise SystemExit(
                f"ytax({stream}) is zero at check: subsidy-blind reference"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", choices=sorted(BUILDERS), required=True)
    ap.add_argument("--form", choices=("nlp", "mcp"), required=True)
    ap.add_argument("--ifsub", type=int, choices=(0, 1), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fbep", choices=("header", "zero"), required=True,
                    help="header: FBEP/FTRV injected from basedata.har; zero: official getData")
    ap.add_argument("--where", choices=("local", "neos"), default="local",
                    help="neos: solve on NEOS (PATH, --form mcp only)")
    args = ap.parse_args(argv)

    if args.where == "neos" and args.form != "mcp":
        ap.error("--where neos solves --form mcp only")
    gams = shutil.which("gams")
    if gams is None and args.where == "local":
        ap.error("the gams binary must be on PATH")
    gdxdump = shutil.which("gdxdump") or (gams and str(Path(gams).with_name("gdxdump")))
    if args.mode == "altertax" and not gdxdump:
        ap.error("the altertax subsidy check needs gdxdump on PATH")
    out = args.out.resolve()

    with tempfile.TemporaryDirectory(prefix="regen_gams_") as tmp:
        work = Path(tmp)
        text = _build(args.mode, args.dataset, args.ifsub, work, args.fbep).read_text()
        model = work / "model.gms"
        model.write_text(to_nlp(text) if args.form == "nlp" else text)
        if args.where == "neos":
            return _solve_on_neos(args, work, model, out, gdxdump)
        assert gams is not None
        proc = subprocess.run(
            [gams, model.name, "lo=2"], cwd=work, capture_output=True, text=True
        )
        lst = work / "model.lst"
        statuses = (
            re.findall(r"MODEL STATUS\s+(\d+)", lst.read_text()) if lst.exists() else []
        )
        gdx = work / "out.gdx"
        if (
            proc.returncode != 0
            or not gdx.exists()
            or not statuses
            or set(statuses) - _OK_STATUS
        ):
            print(
                f"GAMS failed: rc={proc.returncode}, model statuses {statuses}, "
                f"out.gdx {'present' if gdx.exists() else 'missing'}"
            )
            if lst.exists():
                print("\n".join(lst.read_text().splitlines()[-30:]))
            return 1
        if args.mode == "altertax":
            _verify_subsidy_aware(str(gdxdump), gdx, args.fbep)
        _keep(work, out, ("model.gms", "path.opt", "out.gdx"))
    print(f"model statuses {statuses}; wrote {out}")
    return 0


def _keep(work: Path, out: Path, names: tuple[str, ...]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (work / name).exists():
            shutil.copyfile(work / name, out / name)


def _solve_on_neos(
    args: argparse.Namespace, work: Path, model: Path, out: Path, gdxdump: str | None
) -> int:
    """The bundle on NEOS; ``out`` only gets files when NEOS solved it."""
    # a builder that writes path.opt next to the bundle (altertax --homotopy)
    # sends it as the NEOS options; the others write their own with $onecho
    opt = work / "path.opt"
    solved = work / "neos"
    try:
        neos_job.run(model, solved, options=opt.read_text() if opt.exists() else "")
    except neos_job.NeosError as err:
        print(err)
        return 1
    if args.mode == "altertax":
        _verify_subsidy_aware(str(gdxdump), solved / "out.gdx", args.fbep)
    shutil.copyfile(model, solved / "model.gms")
    if opt.exists():
        shutil.copyfile(opt, solved / "path.opt")
    _keep(solved, out, ("model.gms", "path.opt", "out.gdx", "model.lst", "neos.log",
                        "neos_job.json"))  # fmt: skip
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
