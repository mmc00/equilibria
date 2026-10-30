"""Regenerate the local GAMS GTAP references into data/ (ADR-0010, phase 2a).

Each job writes one folder data/gtap7/<ds>/gams_<form>_<mode>_ifsub<k>/ through
write_source.py, which runs it, times it and records its source.yaml:

    nl   the GAMS .nl of every phase, local CONVERT (nl_compare --gams-local)
    nlp  the NLP solve with IPOPT (regen_gams_reference --form nlp)
    mcp  the MCP solve with PATH (regen_gams_reference --form mcp); the local
         license stops PATH at 1000 rows, so only gtap7_3x3 ifSUB=1 is here,
         the rest is phase 2b (NEOS)

A folder that already has a source.yaml is skipped, so a run that dies resumes
where it stopped. A failed job leaves no folder.

    uv run python scripts/gtap/regenerate_gams.py --tool-version 53.5.0 --dry-run
    uv run python scripts/gtap/regenerate_gams.py --tool-version 53.5.0 \\
        --datasets gtap7_3x3 --only gams_nl_altertax_ifsub0
    uv run python scripts/gtap/regenerate_gams.py --tool-version 53.5.0
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from equilibria.sources import (
    DATA_ROOT,
    SOURCE_FILE,
    SourceError,
    dataset_dir,
    load_source,
)

ROOT = Path(__file__).resolve().parents[2]
WRITE_SOURCE = ROOT / "scripts" / "dev" / "write_source.py"
NL_COMPARE = ROOT / "scripts" / "gtap" / "nl_compare.py"
REGEN = ROOT / "scripts" / "gtap" / "regen_gams_reference.py"

# The phases of today's .nl fixtures (adding check to 3x4/15x10 is phase 7).
NL_PHASES = {
    "gtap7_3x3": ("base", "check", "shock", "altertax"),
    "gtap7_3x4": ("base", "shock", "altertax"),
    "gtap7_5x5": ("base", "check", "shock", "altertax"),
    "gtap7_10x7": ("base", "check", "shock", "altertax"),
    "gtap7_15x10": ("base", "shock", "altertax"),
}
NLP_DATASETS = {
    "altertax": ("gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7"),
    "gtap": ("gtap7_3x3", "gtap7_5x5", "gtap7_10x7"),
}
MCP_JOBS = (("gtap7_3x3", "gtap", 1), ("gtap7_3x3", "altertax", 1))
SOLVER = {"nl": "CONVERT", "nlp": "IPOPT", "mcp": "PATH"}
SOLVE_PHASES = ("base", "check", "shock")


@dataclass(frozen=True)
class Job:
    dataset: str
    form: str
    mode: str
    ifsub: int
    phases: tuple[str, ...]
    folder: Path


def _job(
    ds: str, form: str, mode: str, k: int, phases: tuple[str, ...], root: Path
) -> Job:
    folder = dataset_dir(ds, root=root) / f"gams_{form}_{mode}_ifsub{k}"
    return Job(ds, form, mode, k, phases, folder)


def jobs(datasets: list[str] | None = None, root: Path = DATA_ROOT) -> list[Job]:
    """The 26 jobs of the plan's table, restricted to ``datasets`` if given."""
    keep = (lambda ds: True) if datasets is None else (lambda ds: ds in datasets)
    out = [
        _job(ds, "nl", "altertax", k, phases, root)
        for ds, phases in NL_PHASES.items()
        for k in (0, 1)
        if keep(ds)
    ]
    out += [
        _job(ds, "nlp", mode, k, SOLVE_PHASES, root)
        for mode, names in NLP_DATASETS.items()
        for ds in names
        for k in (0, 1)
        if keep(ds)
    ]
    out += [
        _job(ds, "mcp", mode, k, SOLVE_PHASES, root)
        for ds, mode, k in MCP_JOBS
        if keep(ds)
    ]
    return out


def job_command(job: Job, *, tool_version: str, root: Path = DATA_ROOT) -> list[str]:
    """write_source.py recording one job (the model folder is hashed whole)."""
    py = sys.executable
    if job.form == "nl":
        inner = [py, str(NL_COMPARE), "--dataset", job.dataset, "--phase", *job.phases,
                 "--ifsub", str(job.ifsub), "--gams-local", "--skip-python",
                 "--out-dir", str(job.folder)]  # fmt: skip
    else:
        inner = [py, str(REGEN), "--dataset", job.dataset, "--mode", job.mode,
                 "--form", job.form, "--ifsub", str(job.ifsub), "--out", str(job.folder)]  # fmt: skip
    return [
        py, str(WRITE_SOURCE),
        "--data-root", str(root),
        "--folder", str(job.folder),
        "--kind", "result", "--model", "gtap7", "--dataset", job.dataset,
        "--tool", "gams", "--tool-version", tool_version,
        "--variant", f"form={job.form}",
        "--variant", f"mode={job.mode}",
        "--variant", f"ifsub={job.ifsub}",
        "--phases", *job.phases,
        "--model-file", str(root / "gtap7" / "gams"),
        "--data-dir", str(job.folder.parent),
        "--solver", SOLVER[job.form],
        "--",
        *inner,
    ]  # fmt: skip


def gams_version(gams: str) -> str:
    """The release GAMS prints, e.g. 53.5.0 ('' if it prints none)."""
    text = subprocess.run([gams], capture_output=True, text=True, check=False).stdout
    m = re.search(r"GAMS Base Module (\d+\.\d+\.\d+)", text)
    return m.group(1) if m else ""


def drop_broken_records(root: Path, datasets: list[str]) -> list[Path]:
    """Remove gams_* folders whose source.yaml does not load as verified."""
    dropped = []
    for ds in datasets:
        for record in sorted(root.glob(f"*/{ds}/gams_*/{SOURCE_FILE}")):
            try:
                ok = load_source(record.parent).status == "verified"
            except SourceError:
                ok = False
            if not ok:
                shutil.rmtree(record.parent)
                dropped.append(record.parent)
    return dropped


def run_job(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--tool-version", required=True, help="the release `gams` prints, e.g. 53.5.0"
    )
    ap.add_argument(
        "--gams",
        default=shutil.which("gams"),
        help="the gams binary (default: on PATH)",
    )
    ap.add_argument("--datasets", nargs="*", default=list(NL_PHASES))
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        help="run only this folder name (e.g. gams_nl_altertax_ifsub0); repeatable",
    )
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args(argv)

    all_jobs = jobs(args.datasets, root=args.data_root)
    unknown = sorted(set(args.only) - {j.folder.name for j in all_jobs})
    if unknown:
        valid = sorted({j.folder.name for j in all_jobs})
        print(f"unknown --only {unknown}; valid: {valid}", file=sys.stderr)
        return 2
    selected = [j for j in all_jobs if not args.only or j.folder.name in args.only]
    print(f"{len(selected)} job(s)")
    if args.dry_run:
        for j in selected:
            print(
                shlex.join(
                    job_command(j, tool_version=args.tool_version, root=args.data_root)
                )
            )
        return 0

    if not args.gams:
        print("gams not found: put GAMS on PATH or pass --gams", file=sys.stderr)
        return 2
    found = gams_version(args.gams)
    if found != args.tool_version:
        print(f"gams at {args.gams} is release {found or '?'}, not --tool-version "
              f"{args.tool_version}: record the release that runs", file=sys.stderr)  # fmt: skip
        return 2
    # the jobs find gams on PATH
    os.environ["PATH"] = (
        f"{Path(args.gams).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    )
    for folder in drop_broken_records(args.data_root, args.datasets):
        print(f"removed an incomplete record: {folder}")

    failed: list[str] = []
    for n, j in enumerate(selected, 1):
        name = f"{j.dataset}/{j.folder.name}"
        if (j.folder / SOURCE_FILE).exists():
            print(f"[{n}/{len(selected)}] {name}: already recorded, skipped")
            continue
        if j.folder.exists():  # left by a run that died: not an input of anything
            shutil.rmtree(j.folder)
        print(f"[{n}/{len(selected)}] {name}: running", flush=True)
        rc = run_job(
            job_command(j, tool_version=args.tool_version, root=args.data_root)
        )
        if rc != 0:
            shutil.rmtree(j.folder, ignore_errors=True)
            failed.append(name)
            print(f"[{n}/{len(selected)}] {name}: FAILED (exit {rc}), folder removed")
    print(f"\nrecorded {len(selected) - len(failed)}/{len(selected)}")
    for name in failed:
        print(f"  failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
