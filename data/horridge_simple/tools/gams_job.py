"""Horridge's SIMPLE, GAMS side: the five results of phase 3a-2 Task 6 (decision D4).

The GAMS files of the tpmh0103 archive run unchanged, on the Mac with GAMS 53.5:

    gams_nlp_fixcap     simpleNLP.gms          CONOPT
    gams_mcp_fixcap     simpleMCP.gms          PATH
    gams_mpsge_fixcap   mpsgevh.gms            MPSGE (PATH)
    gams_nlp_tariff10   simpleNLP_tariff.gms   CONOPT
    gams_mcp_tariff10   simpleMCP_tariff.gms   PATH

``run`` solves one of them in a scratch folder (``wdir``; ``idir`` finds
``input.gdx``) and keeps its ``RESULTS.gdx`` as ``results.gdx``. The listing is
not kept: it holds local paths. The last solve (the simulation) must end
normally and (locally) optimal; the earlier ones are written to the record's
notes (simpleNLP's first solve is Horridge's iterlim=0 calibration check, which
CONOPT reports as infeasible).

``regenerate`` records the five folders through write_source.py; a folder that
already has a source.yaml is skipped.

    uv run python data/horridge_simple/tools/gams_job.py regenerate --tool-version 53.5.0
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from equilibria.sources import DATA_ROOT, SOURCE_FILE

ROOT = Path(__file__).resolve().parents[3]
WRITE_SOURCE = ROOT / "scripts" / "dev" / "write_source.py"
MODEL = "horridge_simple"
NOTE_FILE = "solve_note.txt"  # write_source --notes-from: appended to the notes
_RELEASE = re.compile(r"^GAMS (\d+\.\d+\.\d+)", re.M)
_STATUS = re.compile(r"^\*\*\*\* (SOLVER STATUS|MODEL STATUS) +(\d+) (.+?)\s*$", re.M)
_OPTIMAL = {"1", "2"}  # Optimal, Locally Optimal


@dataclass(frozen=True)
class Job:
    form: str
    experiment: str
    gms: str
    solver: str

    @property
    def folder(self) -> str:
        return f"gams_{self.form}_{self.experiment}"


JOBS = (
    Job("nlp", "fixcap", "simpleNLP.gms", "CONOPT"),
    Job("mcp", "fixcap", "simpleMCP.gms", "PATH"),
    Job("mpsge", "fixcap", "mpsgevh.gms", "PATH"),
    Job("nlp", "tariff10", "simpleNLP_tariff.gms", "CONOPT"),
    Job("mcp", "tariff10", "simpleMCP_tariff.gms", "PATH"),
)


def release(listing: str) -> str | None:
    """The GAMS release a listing's page header prints, e.g. 53.5.0."""
    m = _RELEASE.search(listing)
    return m.group(1) if m else None


def gams_release(gams: str) -> str:
    """The release the ``gams`` executable prints ('' if it prints none)."""
    text = subprocess.run([gams], capture_output=True, text=True, check=False).stdout
    m = re.search(r"GAMS Base Module (\d+\.\d+\.\d+)", text)
    return m.group(1) if m else ""


def _solves(listing: str) -> list[dict[str, tuple[str, str]]]:
    """Each solve's SOLVER STATUS and MODEL STATUS, in listing order."""
    solves: list[dict[str, tuple[str, str]]] = []
    for kind, code, text in _STATUS.findall(listing):
        if kind == "SOLVER STATUS" or not solves:
            solves.append({})
        solves[-1][kind] = (code, text)
    return solves


def solve_problems(listing: str) -> list[str]:
    """Why the last solve does not count as a solution ([] if it does)."""
    solves = _solves(listing)
    if not solves:
        return ["no SOLVER STATUS in the listing"]
    last = solves[-1]
    problems = []
    solver = last.get("SOLVER STATUS")
    if solver is None or solver[0] != "1":
        problems.append(f"SOLVER STATUS     {' '.join(solver or ('?', 'missing'))}")
    model = last.get("MODEL STATUS")
    if model is None or model[0] not in _OPTIMAL:
        problems.append(f"MODEL STATUS      {' '.join(model or ('?', 'missing'))}")
    return problems


def earlier_solves(listing: str) -> list[str]:
    """The statuses of every solve before the last, one line each."""
    solves = _solves(listing)
    return [
        f"solve {i} of {len(solves)}: "
        + ", ".join(
            f"{k} {' '.join(s[k])}" for k in ("SOLVER STATUS", "MODEL STATUS") if k in s
        )
        for i, s in enumerate(solves[:-1], start=1)
    ]


def _job(form: str, experiment: str) -> Job:
    for job in JOBS:
        if (job.form, job.experiment) == (form, experiment):
            return job
    raise SystemExit(f"no job {form} {experiment}")


def run(args: argparse.Namespace) -> int:
    job = _job(args.form, args.experiment)
    model = args.model_dir.resolve()
    out: Path = args.out.resolve()
    with tempfile.TemporaryDirectory(prefix="horridge_gams_") as tmp:
        wd = Path(tmp)
        rc = subprocess.run(
            [args.gams, str(model / job.gms), f"wdir={wd}", f"idir={model}", "lo=2"],
            capture_output=True,
            check=False,
        ).returncode
        lst = wd / job.gms.replace(".gms", ".lst")
        listing = lst.read_text(errors="replace") if lst.exists() else ""
        found = release(listing)
        problems = ([f"gams exited {rc}"] if rc != 0 else []) + solve_problems(listing)
        if found != args.tool_version:
            problems.append(f"the listing says GAMS {found}, not {args.tool_version}")
        if not (wd / "RESULTS.gdx").exists():
            problems.append("no RESULTS.gdx")
        if problems:
            print(f"FAILED {job.folder}: " + "; ".join(problems), file=sys.stderr)
            return 1
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wd / "RESULTS.gdx", out / "results.gdx")
        notes = earlier_solves(listing) or ["one solve"]
        (out / NOTE_FILE).write_text(f"{job.gms}: " + "; ".join(notes) + "\n")
    print(f"OK {job.folder}")
    return 0


def job_command(
    job: Job, *, tool_version: str, gams: str, root: Path = DATA_ROOT
) -> list[str]:
    """write_source.py recording one job (the model folder is hashed whole)."""
    model = root / MODEL
    folder = model / job.folder
    return [
        sys.executable, str(WRITE_SOURCE),
        "--data-root", str(root),
        "--folder", str(folder),
        "--kind", "result", "--model", MODEL, "--dataset", MODEL,
        "--tool", "gams", "--tool-version", tool_version,
        "--variant", f"form={job.form}", "--variant", f"experiment={job.experiment}",
        "--model-file", str(model),
        "--data-dir", str(model),
        "--solver", job.solver,
        "--notes-from", NOTE_FILE,
        "--",
        sys.executable, str(Path(__file__).resolve()), "run",
        "--form", job.form, "--experiment", job.experiment,
        "--model-dir", str(model), "--out", str(folder),
        "--gams", gams, "--tool-version", tool_version,
    ]  # fmt: skip


def regenerate(args: argparse.Namespace) -> int:
    found = gams_release(args.gams)
    if found != args.tool_version:
        print(f"gams at {args.gams} is release {found or '?'}, not {args.tool_version}",
              file=sys.stderr)  # fmt: skip
        return 1
    failed = 0
    for job in JOBS:
        if (args.data_root / MODEL / job.folder / SOURCE_FILE).exists():
            print(f"skip {job.folder} (recorded)")
            continue
        cmd = job_command(job, tool_version=args.tool_version, gams=args.gams,
                          root=args.data_root)  # fmt: skip
        failed += subprocess.run(cmd, check=False).returncode != 0
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    one = sub.add_parser("run", help="solve one .gms and keep its results.gdx")
    one.add_argument("--form", required=True, choices=sorted({j.form for j in JOBS}))
    one.add_argument("--experiment", required=True,
                     choices=sorted({j.experiment for j in JOBS}))  # fmt: skip
    one.add_argument("--model-dir", type=Path, default=DATA_ROOT / MODEL)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--gams", default=shutil.which("gams") or "gams")
    one.add_argument("--tool-version", required=True, help="the release gams prints")
    allj = sub.add_parser("regenerate", help="record the five folders")
    allj.add_argument("--data-root", type=Path, default=DATA_ROOT)
    allj.add_argument("--gams", default=shutil.which("gams") or "gams")
    allj.add_argument("--tool-version", required=True, help="the release gams prints")
    args = ap.parse_args(argv)
    return run(args) if args.mode == "run" else regenerate(args)


if __name__ == "__main__":
    sys.exit(main())
