"""Hosoe's stdcge, GAMS side: the reference of simple_open (base, check, shock).

``stdcge_points.gms`` is the gamslib model (``../../gamslib/stdcge.gms``) with
three ``execute_unload`` lines: before the first solve (the initialized levels,
which are the calibrated base), after it (the check: no shock) and at the end
(the shock: every import tariff set to zero, with ``EV``). GAMS 53 on the Mac,
CONOPT, Demo licence.

    run         solve stdcge_points.gms in a scratch folder, keep the three GDX
    sam         write sam.csv (the SAM table of the .gms) from a GDX
    regenerate  record the two model folders, hosoe_2x2/sam.csv and the result
                folder through write_source.py; a recorded folder is skipped

    uv run python data/stdcge/gams/tools/gams_job.py regenerate --tool-version 53.5.0
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from equilibria.babel.gdx.reader import (
    read_gdx,
    read_parameter_values,
    read_set_elements,
)
from equilibria.sources import DATA_ROOT, SOURCE_FILE

ROOT = Path(__file__).resolve().parents[4]
WRITE_SOURCE = ROOT / "scripts" / "dev" / "write_source.py"
MODEL = "stdcge"
DATASET = "hosoe_2x2"
GMS = "stdcge_points.gms"
POINTS = ("base.gdx", "check.gdx", "shock.gdx")
NOTE_FILE = "solve_note.txt"
_RELEASE = re.compile(r"^GAMS (\d+\.\d+\.\d+)", re.M)
_STATUS = re.compile(r"^\*\*\*\* (SOLVER STATUS|MODEL STATUS) +(\d+) (.+?)\s*$", re.M)
_OPTIMAL = {"1", "2"}  # Optimal, Locally Optimal
CHANGES = """\
- file: stdcge_points.gms
  change: added
  what: copy of gamslib/stdcge.gms with three lines, execute_unload 'base.gdx'
    before the first solve, 'check.gdx' after it, 'shock.gdx' at the end
  why: simple_open is compared with GAMS at base, check and shock; nothing else
    in the model changes
  commit: this commit
- file: tools/gams_job.py
  change: added
  what: runs stdcge_points.gms, writes hosoe_2x2/sam.csv and records every folder
  why: generators live with the data (ADR-0010)
  commit: this commit
"""


def gams_release(gams: str) -> str:
    """The release the ``gams`` executable prints ('' if it prints none)."""
    text = subprocess.run([gams], capture_output=True, text=True, check=False).stdout
    m = re.search(r"GAMS Base Module (\d+\.\d+\.\d+)", text)
    return m.group(1) if m else ""


def solve_statuses(listing: str) -> list[tuple[str, str]]:
    """(solver status, model status) codes of each solve, in listing order."""
    found: list[dict[str, str]] = []
    for kind, code, _text in _STATUS.findall(listing):
        if kind == "SOLVER STATUS" or not found:
            found.append({})
        found[-1][kind] = code
    return [(s.get("SOLVER STATUS", "?"), s.get("MODEL STATUS", "?")) for s in found]


def solve(gams: str, gms: Path, wd: Path) -> tuple[str, list[str]]:
    """Run ``gms`` in ``wd``; return its listing and why it does not count."""
    rc = subprocess.run(
        [gams, str(gms), f"wdir={wd}", "lo=2"], capture_output=True, check=False
    ).returncode
    lst = wd / gms.with_suffix(".lst").name
    listing = lst.read_text(errors="replace") if lst.exists() else ""
    problems = [f"gams exited {rc}"] if rc != 0 else []
    statuses = solve_statuses(listing)
    if len(statuses) != 2:
        problems.append(f"{len(statuses)} solves in the listing, not 2")
    for i, (solver, model) in enumerate(statuses, start=1):
        if solver != "1" or model not in _OPTIMAL:
            problems.append(f"solve {i}: solver status {solver}, model status {model}")
    problems += [f"no {p}" for p in POINTS if not (wd / p).exists()]
    return listing, problems


def write_sam(gdx: Path, out: Path) -> None:
    """The SAM table of the .gms as a square CSV, accounts in set order."""
    data = read_gdx(gdx)
    accounts = [e[0] for e in read_set_elements(data, "u")]
    sam = read_parameter_values(data, "SAM")
    with out.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["", *accounts])
        for r in accounts:
            w.writerow([r, *(repr(sam.get((r, c), 0.0)) for c in accounts)])


def run(args: argparse.Namespace) -> int:
    gms = args.model_dir.resolve() / GMS
    out: Path = args.out.resolve()
    with tempfile.TemporaryDirectory(prefix="stdcge_gams_") as tmp:
        wd = Path(tmp)
        listing, problems = solve(args.gams, gms, wd)
        m = _RELEASE.search(listing)
        if (m.group(1) if m else None) != args.tool_version:
            problems.append(f"the listing is not GAMS {args.tool_version}")
        if problems:
            print("FAILED: " + "; ".join(problems), file=sys.stderr)
            return 1
        out.mkdir(parents=True, exist_ok=True)
        for p in POINTS:
            shutil.copy2(wd / p, out / p)
        (out / NOTE_FILE).write_text(
            f"{GMS}: solve 1 (check) and solve 2 (shock) optimal; base.gdx is "
            "unloaded before any solve. The SAM is the Table inside the .gms; "
            f"{DATASET}/sam.csv is that same table, hashed as the data.\n"
        )
    print("OK")
    return 0


def sam(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="stdcge_sam_") as tmp:
        wd = Path(tmp)
        _listing, problems = solve(args.gams, args.model_dir.resolve() / GMS, wd)
        if problems:
            print("FAILED: " + "; ".join(problems), file=sys.stderr)
            return 1
        write_sam(wd / "base.gdx", args.out.resolve())
    return 0


def _ws(*args: str) -> int:
    return subprocess.run(
        [sys.executable, str(WRITE_SOURCE), *args], check=False
    ).returncode


def regenerate(args: argparse.Namespace) -> int:
    found = gams_release(args.gams)
    if found != args.tool_version:
        print(f"gams is release {found or '?'}, not {args.tool_version}",
              file=sys.stderr)  # fmt: skip
        return 1
    root: Path = args.data_root
    base = root / MODEL
    lib, gams_dir, data = base / "gamslib", base / "gams", base / DATASET
    result = data / "gams_nlp_tariff0"
    rc = 0
    if not (lib / SOURCE_FILE).exists():
        rc |= _ws("--data-root", str(root), "--folder", str(lib), "--kind", "model",
                  "--model", MODEL, "--no-run", "--source",
                  "GAMS model library, stdcge (SEQ=276), copied unchanged from "
                  f"GAMS {args.tool_version} gamslib_ml/stdcge.276. Hosoe, Gasawa "
                  "and Hashimoto, Handbook of CGE Modeling (2004), ch. 6")  # fmt: skip
    if not (gams_dir / SOURCE_FILE).exists():
        changes = Path(tempfile.mkstemp(suffix=".yaml")[1])
        changes.write_text(CHANGES)
        rc |= _ws("--data-root", str(root), "--folder", str(gams_dir), "--kind",
                  "model", "--model", MODEL, "--no-run", "--derived-from",
                  f"{MODEL}/gamslib", "--changes", str(changes), "--source",
                  "gamslib stdcge with three execute_unload lines; tools/gams_job.py "
                  "(this repo) runs it")  # fmt: skip
    if not (data / SOURCE_FILE).exists():
        data.mkdir(parents=True, exist_ok=True)
        rc |= sam(argparse.Namespace(gams=args.gams, model_dir=gams_dir,
                                     out=data / "sam.csv"))  # fmt: skip
        rc |= _ws("--data-root", str(root), "--folder", str(data), "--kind", "data",
                  "--model", MODEL, "--dataset", DATASET, "--no-run", "--license",
                  "distributed with GAMS in its model library", "--source",
                  "the SAM table of gamslib stdcge (Hosoe et al. 2004, ch. 6), "
                  "written by tools/gams_job.py sam")  # fmt: skip
    if not (result / SOURCE_FILE).exists():
        rc |= _ws("--data-root", str(root), "--folder", str(result), "--kind",
                  "result", "--model", MODEL, "--dataset", DATASET, "--tool", "gams",
                  "--tool-version", args.tool_version, "--variant", "form=nlp",
                  "--variant", "experiment=tariff0", "--model-file", str(gams_dir),
                  "--data-dir", str(data), "--solver", "CONOPT", "--notes-from",
                  NOTE_FILE, "--", sys.executable, str(Path(__file__).resolve()),
                  "run", "--model-dir", str(gams_dir), "--out", str(result),
                  "--gams", args.gams, "--tool-version", args.tool_version)  # fmt: skip
    return 1 if rc else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    here = Path(__file__).resolve().parents[1]
    one = sub.add_parser("run", help="solve stdcge_points.gms, keep the three GDX")
    one.add_argument("--model-dir", type=Path, default=here)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--gams", default="gams")
    one.add_argument("--tool-version", required=True)
    s = sub.add_parser("sam", help="write sam.csv from the .gms")
    s.add_argument("--model-dir", type=Path, default=here)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--gams", default="gams")
    reg = sub.add_parser("regenerate", help="record every folder via write_source")
    reg.add_argument("--data-root", type=Path, default=DATA_ROOT)
    reg.add_argument("--gams", default="gams")
    reg.add_argument("--tool-version", required=True)
    args = ap.parse_args(argv)
    return {"run": run, "sam": sam, "regenerate": regenerate}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
