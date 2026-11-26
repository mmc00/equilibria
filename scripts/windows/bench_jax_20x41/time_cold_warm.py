"""Time the 20x41 AlterTax run on the JAX (family) route: one cold run, then warm runs.

    python scripts/windows/bench_jax_20x41/time_cold_warm.py [--dataset gtap7_20x41] [--warm 2]

Runs ``scripts/family/altertax_rungtap_measure.py`` (AlterTax closure, tms +10 %, the
same exercise as GEMPACK's ``gempack_altertax-rungtap_tms-rate10_s8-16-32``) once with an
empty family cache ("cold") and ``--warm`` more times reusing that cache ("warm").
Every run uses MUMPS with one OpenMP thread (``EQUILIBRIA_GTAP_TR_LINSOLVE=mumps``,
``OMP_NUM_THREADS=1``): the script's own default is SuperLU, which runs out of memory on
the 20x41 (MAR-235).

Writes into ``scripts/windows/bench_jax_20x41/out/``: one ``<tag>.log`` per run (ignored by
git) and ``results.json`` (wall minutes, peak memory, solve codes, MUMPS factorization
times, versions and machine), which is the file to commit.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MEASURE = ROOT / "scripts" / "family" / "altertax_rungtap_measure.py"

SOLVE = re.compile(r"^solve (\S+): code (\d+)")
PATH_SOLVE = re.compile(r"^solve path K=(\d+): .*codes \[([0-9, ]+)\]")
SYMBOLIC = re.compile(r"symbolic factorization done in ([0-9.]+)s")
NUMERIC = re.compile(r"numeric factorization done in ([0-9.]+)s")


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for dist in ("equilibria", "jax", "jaxlib", "numpy", "scipy", "pymumps", "mpi4py"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    conda = os.environ.get("CONDA_PREFIX") or sys.prefix
    meta = Path(conda) / "conda-meta"
    for pkg in ("mumps-seq", "libblas", "libopenblas", "metis", "libscotch", "msmpi"):
        hits = sorted(meta.glob(f"{pkg}-[0-9]*.json")) if meta.is_dir() else []
        out[pkg] = json.loads(hits[-1].read_text())["build"] if hits else None
    try:
        out["git_head"] = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out["git_head"] = None
    return out


def _machine() -> dict:
    vm = psutil.virtual_memory()
    return {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gb": round(vm.total / 2**30, 1),
        "ram_free_gb_at_start": round(vm.available / 2**30, 1),
    }


def _peak(proc: psutil.Process) -> int:
    info = proc.memory_info()
    # peak_wset exists only on Windows
    return max(getattr(info, "peak_wset", 0), info.rss)


def _run(tag: str, dataset: str, env: dict, out: Path) -> dict:
    log_path = out / f"{tag}.log"
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, str(MEASURE), dataset, "--out", str(out)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        watched = psutil.Process(proc.pid)
        peak = 0
        while proc.poll() is None:
            try:
                peak = max(peak, _peak(watched))
            except psutil.Error:
                pass
            time.sleep(1)
    wall = time.perf_counter() - t0
    text = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    codes = {m[1]: int(m[2]) for line in text if (m := SOLVE.match(line))}
    codes |= {f"path K={m[1]}": m[2] for line in text if (m := PATH_SOLVE.match(line))}
    return {
        "tag": tag,
        "exit": proc.returncode,
        "wall_min": round(wall / 60, 2),
        "peak_gb": round(peak / 2**30, 2),
        "overall_pass": any(line.strip() == "overall PASS" for line in text),
        "codes": codes,
        "mumps_symbolic_s": [
            float(m[1]) for line in text if (m := SYMBOLIC.search(line))
        ],
        "mumps_numeric_s": [
            float(m[1]) for line in text if (m := NUMERIC.search(line))
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_20x41")
    ap.add_argument("--warm", type=int, default=2, help="warm runs after the cold one")
    a = ap.parse_args()

    out = HERE / "out"
    cache = out / "family_cache"
    shutil.rmtree(cache, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        EQUILIBRIA_GTAP_TR_LINSOLVE="mumps",
        OMP_NUM_THREADS="1",
        EQUILIBRIA_FAMILY_CACHE=str(cache),
        PYTHONUNBUFFERED="1",
        PYTHONUTF8="1",
    )
    result = {
        "dataset": a.dataset,
        "machine": _machine(),
        "versions": _versions(),
        "runs": [],
    }
    for tag in ["cold"] + [f"warm{i}" for i in range(1, a.warm + 1)]:
        run = _run(tag, a.dataset, env, out)
        result["runs"].append(run)
        print(json.dumps(run), flush=True)
        (out / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return 0 if all(r["exit"] == 0 and r["overall_pass"] for r in result["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
