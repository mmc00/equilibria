"""Solve the nus333 GAMS reference locally: its HARs to the three GDX files
getData.gms reads, then comp_nus333.gms (MCP, ifSUB=0). Keeps COMP.gdx as out.gdx.

nus333's FBEP is zero in its data, so there is one variant. The model (651 rows)
fits the local PATH license. This was the bundle path of
scripts/gtap/bench_nus333_dual.py, which made output/nus333_neos/out.gdx (beec5a0).

Usage:
    uv run python data/gtap7/gams/tools/regen_nus333.py \
        --out data/gtap7/nus333/gams_mcp_nus333
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from nus333_bundle import write_nus333_gdx_bundle

from equilibria.sources import GAMS_MODEL_DIR, dataset_dir

MODEL = GAMS_MODEL_DIR / "comp_nus333.gms"
OK_STATUS = {"1", "2"}  # optimal, locally optimal


def build_bundle(work: Path) -> list[Path]:
    """nus333Sets/Dat/Prm.gdx in ``work``, from data/gtap7/nus333's HARs."""
    return list(write_nus333_gdx_bundle(dataset_dir("nus333"), work).values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gams", default=shutil.which("gams"), help="the gams binary")
    ap.add_argument("--fbep", choices=("zero",), default="zero",
                    help="FBEP is zero in nus333's data: the official getData is its one variant")
    args = ap.parse_args(argv)
    if not args.gams:
        ap.error("the gams binary must be on PATH (or --gams)")
    out = args.out.resolve()
    with tempfile.TemporaryDirectory(prefix="regen_nus333_") as tmp:
        work = Path(tmp)
        build_bundle(work)
        lst = work / "model.lst"
        # includes resolve against the model folder; every output goes to work
        proc = subprocess.run(
            [args.gams, str(MODEL), f"idir1={GAMS_MODEL_DIR}", f"curDir={work}",
             f"--inDir={work}", "--baseName=nus333", f"--outDir={work}",
             f"o={lst}", "lo=2"],
            capture_output=True, text=True,
        )  # fmt: skip
        statuses = (
            re.findall(r"MODEL STATUS\s+(\d+)", lst.read_text()) if lst.exists() else []
        )
        comp = work / "COMP.gdx"
        if proc.returncode != 0 or not comp.exists() or not statuses or set(statuses) - OK_STATUS:
            print(
                f"GAMS failed: rc={proc.returncode}, model statuses {statuses}, "
                f"COMP.gdx {'present' if comp.exists() else 'missing'}"
            )
            if lst.exists():
                print("\n".join(lst.read_text().splitlines()[-30:]))
            return 1
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(comp, out / "out.gdx")
    print(f"model statuses {statuses}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
