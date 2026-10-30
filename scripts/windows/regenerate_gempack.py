"""Regenerate the GEMPACK linearization-study matrix into data/ (ADR-0010, phase 3a-1).

Runs on the Windows machine with RunGTAP and GEMPACK. Each of the 66 jobs
(6 datasets x 10 grid configurations in capFlex, plus tm10_s8-16-32 in capFix)
is solved by ``run_gempack_matrix.py --out`` under ``write_source.py``, which
times the run and writes the folder's ``source.yaml``.

A folder that already has a ``source.yaml`` is skipped, so a run that dies
resumes where it stopped. A failed job leaves no folder: its files would
otherwise count as files of the dataset folder.

Usage (PowerShell, from the repo root):
    uv run python scripts/windows/regenerate_gempack.py --tool-version 11.3 --dry-run
    uv run python scripts/windows/regenerate_gempack.py --tool-version 11.3 `
        --datasets gtap7_3x3 --only tm10_s8-16-32_capflex
    uv run python scripts/windows/regenerate_gempack.py --tool-version 11.3
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from equilibria.sources import DATA_ROOT, SOURCE_FILE, SourceError, dataset_dir, load_source

ROOT = Path(__file__).resolve().parents[2]
MATRIX_SCRIPT = ROOT / "scripts" / "gtap" / "run_gempack_matrix.py"
WRITE_SOURCE = ROOT / "scripts" / "dev" / "write_source.py"


def _matrix() -> Any:
    spec = importlib.util.spec_from_file_location("run_gempack_matrix", MATRIX_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_M = _matrix()
DATASETS: list[str] = list(_M.DATASETS)
CLOSURE = {0: "capFix", 1: "capFlex"}


@dataclass(frozen=True)
class Job:
    dataset: str
    shock: float
    steps: str
    closure: str
    rordelta: int
    folder: Path

    @property
    def tag(self) -> str:
        return _M.config_tag(self.shock, self.steps, self.rordelta)


def jobs(datasets: list[str], root: Path = DATA_ROOT) -> list[Job]:
    """The committed matrix: the grid in capFlex, plus tm10_s8-16-32 in capFix."""
    configs = [(shock, steps, 1) for shock, steps in _M._grid_configs()]
    configs.append((10.0, "8 16 32", 0))
    out = []
    for ds in datasets:
        base = dataset_dir(ds, root=root)
        for shock, steps, rd in configs:
            tag = _M.config_tag(shock, steps, rd)
            out.append(Job(ds, shock, steps, CLOSURE[rd], rd, base / f"gempack_{tag}"))
    return out


def job_command(
    job: Job, *, gtapv7: Path, sltoht: Path, tool_version: str, root: Path = DATA_ROOT
) -> list[str]:
    """write_source.py recording run_gempack_matrix.py --out for one job.

    ``steps`` is quoted so write_source keeps it a string ("4" would load as 4).
    """
    py = sys.executable
    return [
        py, str(WRITE_SOURCE),
        "--data-root", str(root),
        "--folder", str(job.folder),
        "--kind", "result", "--model", "gtap7", "--dataset", job.dataset,
        "--tool", "gempack", "--tool-version", tool_version,
        "--variant", f"shock={job.shock:g}",
        "--variant", f'steps="{job.steps.replace(" ", "-")}"',
        "--variant", f"closure={job.closure}",
        "--phases", "shock",
        "--data-dir", str(job.folder.parent),
        "--solver", "gtapv7",
        "--",
        py, str(MATRIX_SCRIPT),
        "--datasets", job.dataset,
        "--shock-pct", f"{job.shock:g}",
        "--steps", job.steps,
        "--rordelta", str(job.rordelta),
        "--out", str(job.folder),
        "--gtapv7", str(gtapv7),
        "--sltoht", str(sltoht),
        "--expect-release", tool_version,
    ]  # fmt: skip


def drop_broken_records(root: Path, datasets: list[str]) -> list[Path]:
    """Remove gempack_* folders whose source.yaml does not load as verified.

    A record cut short by a killed run would make every dataset_dir() call fail
    (all records are read), and must not count as done on resume.
    """
    dropped = []
    for ds in datasets:
        for record in sorted(root.glob(f"*/{ds}/gempack_*/{SOURCE_FILE}")):
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
    ap.add_argument("--tool-version", required=True,
                    help="GEMPACK version as RunGTAP reports it (Help -> About)")
    ap.add_argument("--gtapv7", type=Path, default=_M.DEFAULT_GTAPV7)
    ap.add_argument("--sltoht", type=Path, default=_M.DEFAULT_SLTOHT)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--only", action="append", default=[],
                    help="run only this tag (e.g. tm10_s8-16-32_capflex); repeatable")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args(argv)

    known = {_M.config_tag(s, st, rd) for s, st in _M._grid_configs() for rd in (1,)}
    known.add(_M.config_tag(10.0, "8 16 32", 0))
    unknown = sorted(set(args.only) - known)
    if unknown:
        print(f"unknown --only {unknown}; valid tags: {sorted(known)}", file=sys.stderr)
        return 2
    if not args.dry_run:
        missing = [str(p) for p in (args.gtapv7, args.sltoht) if not p.exists()]
        if missing:
            print(f"not found: {missing} (pass --gtapv7 / --sltoht)", file=sys.stderr)
            return 2
        for folder in drop_broken_records(args.data_root, args.datasets):
            print(f"removed an incomplete record: {folder}")

    selected = [
        j for j in jobs(args.datasets, root=args.data_root)
        if not args.only or j.tag in args.only
    ]  # fmt: skip
    print(f"{len(selected)} job(s)")
    if args.dry_run:
        for j in selected:
            print(shlex.join(job_command(j, gtapv7=args.gtapv7, sltoht=args.sltoht,
                                         tool_version=args.tool_version,
                                         root=args.data_root)))
        return 0

    failed: list[str] = []
    for n, j in enumerate(selected, 1):
        name = f"{j.dataset}/{j.folder.name}"
        if (j.folder / SOURCE_FILE).exists():
            print(f"[{n}/{len(selected)}] {name}: already recorded, skipped")
            continue
        if j.folder.exists():  # left by a run that died: not an input of anything
            shutil.rmtree(j.folder)
        print(f"[{n}/{len(selected)}] {name}: running", flush=True)
        rc = run_job(job_command(j, gtapv7=args.gtapv7, sltoht=args.sltoht,
                                 tool_version=args.tool_version, root=args.data_root))
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
