"""Regenerate the GAMS GTAP references into data/ (ADR-0010, phases 2a and 2b).

Each job writes one folder data/gtap7/<ds>/gams_<form>_<mode>_ifsub<k>_fbep-<v>/
through write_source.py, which runs it, times it and records its source.yaml:

    nl   the GAMS .nl of every phase, local CONVERT (tools/nl_standalone.py)
    nlp  the NLP solve with IPOPT (regen_gams_reference --form nlp); gtap7_20x41
         with CONOPT on NEOS and zcons scaled (NEOS_NLP_JOBS)
    mcp  the MCP solve with PATH (regen_gams_reference --form mcp); the local
         license stops PATH at 1000 rows, so only gtap7_3x3 ifSUB=1 runs here,
         the rest on NEOS (--where neos, e-mail from $NEOS_EMAIL)

Every job comes in two variants (#187): fbep=header, the FBEP/FTRV headers of
basedata.har injected after getData.gms's fbep = 0, and fbep=zero, the
official getData.gms. --where local|neos|all picks the rows (default local).

A folder that already has a source.yaml is skipped, so a run that dies resumes
where it stopped. A failed job leaves no folder.

    uv run python data/gtap7/gams/tools/regenerate_gams.py --tool-version 53.5.0 --dry-run
    uv run python data/gtap7/gams/tools/regenerate_gams.py --tool-version 53.5.0 \\
        --datasets gtap7_3x3 --only gams_nl_altertax_ifsub0
    uv run python data/gtap7/gams/tools/regenerate_gams.py --tool-version 53.5.0
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

ROOT = Path(__file__).resolve().parents[4]
WRITE_SOURCE = ROOT / "scripts" / "dev" / "write_source.py"
NL_STANDALONE = Path(__file__).with_name("nl_standalone.py")
REGEN = Path(__file__).with_name("regen_gams_reference.py")

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
# above the local PATH license (1000 rows): solved on NEOS
NEOS_JOBS = {
    "gtap": (("gtap7_3x3", 0), ("gtap7_5x5", 0), ("gtap7_5x5", 1), ("gtap7_10x7", 0),
             ("gtap7_10x7", 1), ("gtap7_15x10", 0), ("gtap7_15x10", 1), ("9x10", 1)),
    "altertax": (("gtap7_3x3", 0), ("gtap7_3x4", 0), ("gtap7_3x4", 1), ("gtap7_5x5", 0),
                 ("gtap7_5x5", 1), ("gtap7_10x7", 0), ("gtap7_10x7", 1), ("gtap7_15x10", 0),
                 ("gtap7_15x10", 1)),
}  # fmt: skip
# NLP with CONOPT on NEOS, zcons scaled: in gtap7_20x41 CHE has SUBPAR at the
# GTAP floor (1e-6) in 8 of 20 goods, so zcons(CHE) ~ 1e-10; PATH stalls there
# (even at a 1% shock) and unscaled CONOPT stops at false points.
NEOS_NLP_JOBS = {"gtap": (("gtap7_20x41", 1),)}
ZCONS_NOTES = (
    "zcons and zconseq scaled by their calibrated level (regen --scale-zcons): "
    "SUBPAR at the GTAP floor (1e-6) in CHE puts zcons near 1e-10; the original "
    "model sets scaleopt=1 but no .scale."
)
# one-shot shock with the tight PATH options of the 2026-07-19 references:
# with PATH defaults the 15x10 altertax shock ends Intermediate Infeasible (6)
TIGHT_PATH = {("gtap7_15x10", "altertax", "neos")}
FBEP = ("header", "zero")
WHERE = ("local", "neos")
SOLVER = {"nl": "CONVERT", "mcp": "PATH"}
NLP_SOLVER = {"ipopt": "IPOPT", "conopt": "CONOPT"}
SOLVE_PHASES = ("base", "check", "shock")


@dataclass(frozen=True)
class Job:
    dataset: str
    form: str
    mode: str
    ifsub: int
    phases: tuple[str, ...]
    folder: Path
    fbep: str
    where: str
    path_opt: bool = False
    nlp_solver: str = "ipopt"
    scale_zcons: bool = False


def _job(
    ds: str, form: str, mode: str, k: int, phases: tuple[str, ...], dirs: dict[str, Path],
    fbep: str, where: str = "local", nlp_solver: str = "ipopt", scale_zcons: bool = False,
) -> Job:  # fmt: skip
    folder = dirs[ds] / f"gams_{form}_{mode}_ifsub{k}_fbep-{fbep}"
    return Job(ds, form, mode, k, phases, folder, fbep, where,
               (ds, mode, where) in TIGHT_PATH, nlp_solver, scale_zcons)  # fmt: skip


def jobs(
    datasets: list[str] | None = None, root: Path = DATA_ROOT, where: str = "all"
) -> list[Job]:
    """The 88 jobs of the plan's table, restricted to ``datasets`` and ``where``."""
    keep = (lambda ds: True) if datasets is None else (lambda ds: ds in datasets)
    named = {*NL_PHASES, *(ds for names in NLP_DATASETS.values() for ds in names),
             *(ds for rows in NEOS_JOBS.values() for ds, _ in rows),
             *(ds for rows in NEOS_NLP_JOBS.values() for ds, _ in rows)}  # fmt: skip
    # dataset_dir walks the whole data tree: once per dataset, not per job
    root_dirs = {ds: dataset_dir(ds, root=root) for ds in named if keep(ds)}
    out = [
        _job(ds, "nl", "altertax", k, phases, root_dirs, v)
        for ds, phases in NL_PHASES.items()
        for k in (0, 1)
        for v in FBEP
        if keep(ds)
    ]
    out += [
        _job(ds, "nlp", mode, k, SOLVE_PHASES, root_dirs, v)
        for mode, names in NLP_DATASETS.items()
        for ds in names
        for k in (0, 1)
        for v in FBEP
        if keep(ds)
    ]
    out += [
        _job(ds, "mcp", mode, k, SOLVE_PHASES, root_dirs, v)
        for ds, mode, k in MCP_JOBS
        for v in FBEP
        if keep(ds)
    ]
    out += [
        _job(ds, "mcp", mode, k, SOLVE_PHASES, root_dirs, v, "neos")
        for mode, rows in NEOS_JOBS.items()
        for ds, k in rows
        for v in FBEP
        if keep(ds)
    ]
    out += [
        _job(ds, "nlp", mode, k, SOLVE_PHASES, root_dirs, v, "neos", "conopt", True)
        for mode, rows in NEOS_NLP_JOBS.items()
        for ds, k in rows
        for v in FBEP
        if keep(ds)
    ]
    if where != "all":
        out = [j for j in out if j.where == where]
    return out


def job_command(job: Job, *, tool_version: str, root: Path = DATA_ROOT) -> list[str]:
    """write_source.py recording one job (the model folder is hashed whole)."""
    py = sys.executable
    if job.form == "nl":
        inner = [py, str(NL_STANDALONE), "--dataset", job.dataset, "--phase", *job.phases,
                 "--ifsub", str(job.ifsub), "--out-dir", str(job.folder),
                 "--fbep", job.fbep]  # fmt: skip
    else:
        inner = [py, str(REGEN), "--dataset", job.dataset, "--mode", job.mode,
                 "--form", job.form, "--ifsub", str(job.ifsub), "--out", str(job.folder),
                 "--fbep", job.fbep, "--where", job.where,
                 *(["--path-opt"] if job.path_opt else []),
                 *(["--nlp-solver", job.nlp_solver] if job.nlp_solver != "ipopt" else []),
                 *(["--scale-zcons"] if job.scale_zcons else [])]  # fmt: skip
    # NEOS reports the GAMS it ran; write_source reads it from neos_job.json
    version = ["--remote", "neos"] if job.where == "neos" else ["--tool-version", tool_version]
    return [
        py, str(WRITE_SOURCE),
        "--data-root", str(root),
        "--folder", str(job.folder),
        "--kind", "result", "--model", "gtap7", "--dataset", job.dataset,
        "--tool", "gams", *version,
        "--variant", f"form={job.form}",
        "--variant", f"mode={job.mode}",
        "--variant", f"ifsub={job.ifsub}",
        "--variant", f"fbep={job.fbep}",
        "--phases", *job.phases,
        "--model-file", str(root / "gtap7" / "gams"),
        "--data-dir", str(job.folder.parent),
        "--solver", NLP_SOLVER[job.nlp_solver] if job.form == "nlp" else SOLVER[job.form],
        "--solver-options-file", "path.opt",
        *(["--notes", ZCONS_NOTES] if job.scale_zcons else []),
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


def run_job(cmd: list[str], env: dict[str, str] | None = None) -> int:
    return subprocess.run(cmd, check=False, env=env).returncode


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
    ap.add_argument("--datasets", nargs="*", default=None, help="default: every dataset")
    ap.add_argument("--where", choices=(*WHERE, "all"), default="local",
                    help="local (default), neos, or all rows of the table")
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        help="run only this folder name (e.g. gams_nl_altertax_ifsub0); repeatable",
    )
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args(argv)

    all_jobs = jobs(args.datasets, root=args.data_root, where=args.where)
    datasets = sorted({j.dataset for j in jobs(args.datasets, root=args.data_root)})
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
    # the jobs find gams on their PATH; this process's environment stays as it was
    path = f"{Path(args.gams).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    env = {**os.environ, "PATH": path}
    for folder in drop_broken_records(args.data_root, datasets):
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
            job_command(j, tool_version=args.tool_version, root=args.data_root), env
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
