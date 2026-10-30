#!/usr/bin/env python3
"""Run the command that generates a reference and write its source.yaml (ADR-0010).

The record is never typed by hand: this script runs the command in the target
folder, times it, reads the machine (OS, CPU, cores, RAM; never host or user
names), hashes the inputs and every output, and writes ``source.yaml`` only if
the command succeeded. ``--no-run`` records input data or a reference model,
which are not generated.

    python scripts/dev/write_source.py --folder data/gtap7/gtap7_3x3/gams_ifsub0 \
        --kind result --model gtap7 --dataset gtap7_3x3 --tool gams \
        --tool-version 53.1.0 --variant ifsub=0 --variant closure=capFix \
        --model-file data/gtap7/gams/comp.gms --data-dir data/gtap7/gtap7_3x3 \
        -- gams comp.gms --ifsub=0
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from equilibria.sources import (
    DATA_ROOT,
    SOURCE_FILE,
    check_derived,
    load_source,
    own_files,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]


def machine_info() -> dict[str, Any]:
    """OS, CPU model, logical cores and RAM in GB. Nothing that names the machine."""
    system = platform.system()
    cpu, ram = platform.processor() or platform.machine(), 0.0
    if system == "Darwin":
        cpu = _sysctl("machdep.cpu.brand_string") or cpu
        ram = int(_sysctl("hw.memsize") or 0) / 2**30
        os_name = f"macOS {platform.mac_ver()[0]} (Darwin {platform.release()})"
    elif system == "Linux":
        os_name = f"Linux {platform.release()}"
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                ram = int(line.split()[1]) / 2**20
                break
    elif system == "Windows":
        os_name = f"Windows {platform.release()} ({platform.version()})"
        ram = _windows_ram()
    else:
        os_name = f"{system} {platform.release()}"
    return {
        "os": os_name,
        "cpu": cpu,
        "cores": os.cpu_count() or 1,
        "ram_gb": round(ram, 1),
    }


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _windows_ram() -> float:
    class _MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemStatus()
    status.dwLength = ctypes.sizeof(_MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
    return status.ullTotalPhys / 2**30


def _variant(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        out[key] = yaml.safe_load(value)  # "0" -> 0, "capFix" -> "capFix"
    return out


def _outputs(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): sha256_file(p) for p in own_files(folder)}


def _inputs(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Hash the inputs. Raises ValueError for a path outside ``root``."""
    generated_from: dict[str, Any] = {}
    if args.model_file:
        mf = args.model_file.resolve()
        generated_from["model"] = {
            "path": mf.relative_to(root).as_posix(),
            "sha256": sha256_file(mf),
        }
    if args.data_dir:
        dd = args.data_dir.resolve()
        generated_from["data"] = {
            "path": dd.relative_to(root).as_posix(),
            "sha256": {
                p.relative_to(dd).as_posix(): sha256_file(p) for p in own_files(dd)
            },
        }
    return generated_from


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--folder", type=Path, required=True)
    ap.add_argument("--kind", choices=["result", "data", "model"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--tool")
    ap.add_argument("--tool-version")
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--phases", nargs="*")
    ap.add_argument("--model-file", type=Path)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--solver")
    ap.add_argument("--source", help="kind data/model: where it comes from")
    ap.add_argument("--license", help="kind data: redistribution terms")
    ap.add_argument(
        "--derived-from",
        help="kind model: the folder it was derived from, relative to the data root",
    )
    ap.add_argument(
        "--changes",
        type=Path,
        help="kind model: YAML list of {file, change, what, why, commit}",
    )
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:  # only the separator before the command
        command = command[1:]
    root = args.data_root.resolve()
    folder = args.folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {"kind": args.kind, "model": args.model}
    if args.dataset:
        record["dataset"] = args.dataset

    if args.kind == "result":
        if args.no_run or not command:
            ap.error("a result is generated: give the command after --")
        # Validate the inputs before a run that may take hours.
        try:
            generated_from = _inputs(args, root)
        except ValueError as err:
            ap.error(f"an input is outside the data root {root}: {err}")
        # A rerun replaces the record: if it fails, no stale "verified" stays.
        (folder / SOURCE_FILE).unlink(missing_ok=True)
        started = dt.datetime.now().astimezone()
        t0 = time.perf_counter()
        proc = subprocess.run(command, cwd=folder)
        duration = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"command failed (exit {proc.returncode}); no source.yaml written")
            return proc.returncode
        record.update(
            {
                "tool": args.tool,
                **({"tool_version": args.tool_version} if args.tool_version else {}),
                "variant": _variant(args.variant),
                **({"phases": args.phases} if args.phases else {}),
                "generated_from": generated_from,
                **({"solver": {"name": args.solver}} if args.solver else {}),
                "command": shlex.join(command),
                "equilibria_commit": _git_commit(),
                "exit": {"code": proc.returncode},
                "outputs": _outputs(folder),
                "run": {
                    "date": started.isoformat(timespec="seconds"),
                    "duration_s": round(duration, 3),
                    "machine": machine_info(),
                },
            }
        )
    else:
        record["source"] = args.source or ""
        if args.kind == "data":
            record["license"] = args.license or ""
        if args.derived_from or args.changes:
            if args.kind != "model" or not (args.derived_from and args.changes):
                ap.error("--derived-from and --changes go together, for --kind model")
            record["derived_from"] = args.derived_from
            record["changes"] = yaml.safe_load(args.changes.read_text(encoding="utf-8"))
        record["outputs"] = _outputs(folder)
    record["status"] = "verified"
    record["notes"] = args.notes

    (folder / SOURCE_FILE).write_text(
        yaml.safe_dump(record, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    load_source(folder)  # what was written must be a valid record
    problems = check_derived(folder, root)
    if problems:
        (folder / SOURCE_FILE).unlink()
        print("\n".join(problems) + "\nno source.yaml written")
        return 1
    print(f"wrote {folder / SOURCE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
