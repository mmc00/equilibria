"""Regenerate the GEMPACK altertax references into data/ (10 jobs).

One job per dataset that has a GAMS altertax reference and per set of parameters:
``rungtap`` (RunGTAP's AlterTax button) and ``gams`` (the values of comp_altertax.gms).
Each is altertax_job.py under write_source.py. A folder with a source.yaml is skipped,
so a stopped run resumes; a failed job leaves no folder.

Usage (PowerShell, from the repo root):
    uv run python data/gtap7/gempack/tools/regenerate_altertax.py --dry-run
    uv run python data/gtap7/gempack/tools/regenerate_altertax.py --datasets gtap7_3x3
    uv run python data/gtap7/gempack/tools/regenerate_altertax.py
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from equilibria.sources import DATA_ROOT, SOURCE_FILE, dataset_dir

ROOT = Path(__file__).resolve().parents[4]
JOB_SCRIPT = Path(__file__).with_name("altertax_job.py")
WRITE_SOURCE = ROOT / "scripts" / "dev" / "write_source.py"

# gtap7_20x41, 9x10 and nus333 have no GAMS altertax reference to compare against.
DATASETS = ["gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7", "gtap7_15x10"]
PARAMETERS = ("rungtap", "gams")
GTAP_DIR = Path(r"C:\runGTAP375")
GP_DIR = Path(r"C:\GP")
SHOCK = "tms-rate10"
STEPS = "8-16-32"


@dataclass(frozen=True)
class Job:
    dataset: str
    parameters: str
    folder: Path


def folder_name(parameters: str) -> str:
    return f"gempack_altertax-{parameters}_{SHOCK}_s{STEPS}"


def jobs(datasets: list[str], root: Path = DATA_ROOT) -> list[Job]:
    return [
        Job(ds, p, dataset_dir(ds, root=root) / folder_name(p))
        for ds in datasets
        for p in PARAMETERS
    ]


def job_command(job: Job, *, gempack: str, sltoht: str, root: Path = DATA_ROOT) -> list[str]:
    """write_source.py recording altertax_job.py for one job."""
    py = sys.executable
    exes = {
        "gtapv7.exe": (GTAP_DIR / "gtapv7.exe", gempack),
        "altparv7.exe": (GTAP_DIR / "altparv7.exe", gempack),
        "sltoht.exe": (GP_DIR / "sltoht.exe", sltoht),
    }
    externals: list[str] = []
    for name, (path, release) in exes.items():
        externals += ["--external", f"{name}={path}@{release}"]
    return [
        py, str(WRITE_SOURCE),
        "--data-root", str(root),
        "--folder", str(job.folder),
        "--kind", "result", "--model", "gtap7", "--dataset", job.dataset,
        "--tool", "gempack", "--tool-version", gempack,
        "--variant", "mode=altertax",
        "--variant", f"parameters={job.parameters}",
        "--variant", f"shock={SHOCK}",
        "--variant", f'steps="{STEPS}"',
        "--variant", "closure=altertax",
        "--phases", "shock",
        "--model-file", str(root / "gtap7" / "gempack"),
        "--data-dir", str(job.folder.parent),
        "--solver", "gtapv7",
        *externals,
        "--",
        py, str(JOB_SCRIPT),
        "--dataset", job.dataset,
        "--parameters", job.parameters,
        "--out", str(job.folder),
        "--gtapv7", str(exes["gtapv7.exe"][0]),
        "--altparv7", str(exes["altparv7.exe"][0]),
        "--sltoht", str(exes["sltoht.exe"][0]),
        "--expect-release", gempack,
        "--release", f"altparv7.exe={gempack}",
    ]  # fmt: skip


def run_job(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gempack", default="12.1.003", help="release of gtapv7.exe and altparv7.exe")
    ap.add_argument("--sltoht", default="11.3.004", help="release of C:\\GP\\sltoht.exe")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args(argv)

    selected = jobs(args.datasets, root=args.data_root)
    print(f"{len(selected)} job(s)")
    if args.dry_run:
        for j in selected:
            print(shlex.join(job_command(j, gempack=args.gempack, sltoht=args.sltoht, root=args.data_root)))
        return 0

    failed: list[str] = []
    for n, j in enumerate(selected, 1):
        name = f"{j.dataset}/{j.folder.name}"
        if (j.folder / SOURCE_FILE).exists():
            print(f"[{n}/{len(selected)}] {name}: already recorded, skipped")
            continue
        if j.folder.exists():  # left by a run that died
            shutil.rmtree(j.folder)
        print(f"[{n}/{len(selected)}] {name}: running", flush=True)
        if run_job(job_command(j, gempack=args.gempack, sltoht=args.sltoht, root=args.data_root)) != 0:
            shutil.rmtree(j.folder, ignore_errors=True)
            failed.append(name)
            print(f"[{n}/{len(selected)}] {name}: FAILED, folder removed")
    print(f"\nrecorded {len(selected) - len(failed)}/{len(selected)}")
    for name in failed:
        print(f"  failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
