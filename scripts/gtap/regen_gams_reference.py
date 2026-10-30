"""Solve one GAMS GTAP reference into a folder (ADR-0010, phase 2a).

Builds the self-contained bundle of a dataset from the model in
data/gtap7/gams (pure: build_gtap7_pure_local_bundle.py, altertax:
build_gtap7_altertax_neos_bundle.py), solves it with the local GAMS and
leaves in --out only what replays the run: ``model.gms`` (the exact file
solved, data inlined), ``path.opt`` when the bundle writes one, and
``out.gdx``. Nothing else, so write_source.py can record the folder as is.

--form mcp solves the bundle as built (PATH; the local license stops at 1000
rows, so only gtap7_3x3 ifSUB=1 fits). --form nlp switches ifMCP off and
solves ``gtap using nlp maximizing walras`` with IPOPT (no size limit).

Exit status is not 0, and --out holds no out.gdx, when GAMS fails or any
model status is not 1 (Optimal) or 2 (Locally Optimal).

    uv run python scripts/gtap/regen_gams_reference.py --dataset gtap7_3x3 \\
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

ROOT = Path(__file__).resolve().parents[2]
BUILDERS = {
    "gtap": ROOT / "scripts" / "gtap" / "build_gtap7_pure_local_bundle.py",
    "altertax": ROOT / "scripts" / "gtap" / "build_gtap7_altertax_neos_bundle.py",
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


def _build(mode: str, dataset: str, ifsub: int, work: Path) -> Path:
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
        ],
        check=True,
        cwd=ROOT,
    )
    built = sorted(work.glob("comp_*.gms"))
    if len(built) != 1:
        raise SystemExit(f"builder left {len(built)} comp_*.gms in {work}, expected 1")
    return built[0]


def _verify_subsidy_aware(gdxdump: str, gdx: Path) -> None:
    """The altertax reference must carry nonzero ytax(ft) and ytax(fs) at check."""
    dump = subprocess.run(
        [gdxdump, str(gdx), "Symb=ytax", "Format=csv"], capture_output=True, text=True
    ).stdout
    for stream in ("ft", "fs"):
        values = [
            abs(float(line.rsplit(",", 1)[-1]))
            for line in dump.splitlines()
            if f'"{stream}"' in line and "check" in line
        ]
        if not any(v > 1e-6 for v in values):
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
    args = ap.parse_args(argv)

    gams = shutil.which("gams")
    if gams is None:
        ap.error("the gams binary must be on PATH")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="regen_gams_") as tmp:
        work = Path(tmp)
        text = _build(args.mode, args.dataset, args.ifsub, work).read_text()
        model = work / "model.gms"
        model.write_text(to_nlp(text) if args.form == "nlp" else text)
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
            _verify_subsidy_aware(str(Path(gams).with_name("gdxdump")), gdx)
        for name in ("model.gms", "path.opt", "out.gdx"):
            if (work / name).exists():
                shutil.copyfile(work / name, out / name)
    print(f"model statuses {statuses}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
