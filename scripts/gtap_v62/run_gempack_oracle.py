"""GEMPACK oracle for GTAP v6.2 parity validation.

Invokes ``gtap.exe`` (GTAP Model Version 6.2 GEMPACK executable) on a
target experiment file (CMF/EXP) and extracts the post-solve variable
levels via ``sltohta.exe`` (sl4 → HAR conversion). Used as the
authoritative reference when validating equilibria's Python v6.2
implementation.

Requires RunGTAP installation at ``C:\\runGTAP375\\`` (Windows). The
executables involved:

- ``gtap.exe``      : Standard GTAP v6.2 (TABLO-compiled from gtap.tab)
- ``sltohta.exe``   : Converts a ``.sl4`` solution file to ``.har``
- ``seehara.exe``   : Optional HAR diff viewer

The oracle is **deliberately read-only**: experiment files in BOOK3X3 /
NUS333 directories are not modified. A working copy of the experiment
folder is used so the GEMPACK run does not pollute the canonical
dataset.

Typical usage::

    from scripts.gtap_v62.run_gempack_oracle import run_gempack_experiment

    out = run_gempack_experiment(
        experiment="Exp1a",
        dataset_dir=Path("C:/runGTAP375/BOOK3X3"),
        workdir=Path("runs/gtap_v62_oracle/BOOK3X3_Exp1a"),
    )
    print(out["status"])         # "ok"
    print(out["solution_har"])   # Path to <experiment>_sol.har
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_RUNGTAP_DIR = Path(r"C:\runGTAP375")
GTAP_EXE_NAME = "gtap.exe"
SLTOHTA_EXE_NAME = "sltohta.exe"


@dataclass
class GempackOracleResult:
    """Outcome of a GEMPACK run via :func:`run_gempack_experiment`."""

    experiment: str
    workdir: Path
    cmf_path: Path
    sl4_path: Optional[Path]
    solution_har: Optional[Path]
    log_path: Path
    returncode: int
    status: str  # "ok" | "solve_failed" | "convert_failed" | "missing_exe"
    message: str = ""


def _locate_executable(name: str, rungtap_dir: Path) -> Path:
    candidate = rungtap_dir / name
    if not candidate.exists():
        raise FileNotFoundError(
            f"RunGTAP executable {name!r} not found in {rungtap_dir}. "
            f"Install RunGTAP and/or pass an explicit rungtap_dir."
        )
    return candidate


_GTAPSETS_CANDIDATES = ("SETS.HAR", "sets.har", "Sets.har")
_GTAPDATA_CANDIDATES = ("basedata.har", "BASEDATA.HAR", "Basedata.har")
_GTAPPARM_CANDIDATES = ("Default.prm", "default.prm", "DEFAULT.PRM")


def _first_existing(dataset_dir: Path, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if (dataset_dir / name).exists():
            return name
    return None


def _prepare_workdir(
    dataset_dir: Path,
    workdir: Path,
    experiment: str,
    extra_files: Iterable[str],
) -> Path:
    """Copy a minimal slice of the dataset folder into ``workdir``.

    Files copied:
    - The experiment file (``<experiment>.exp`` or ``<experiment>.cmf``)
    - ``SETS.HAR`` (the v6.2 GTAPSETS file)
    - ``basedata.har`` (the SAM / GTAPDATA)
    - ``Default.prm`` / ``default.prm`` (GTAPPARM)
    - ``CMFSTART`` (run-control template)
    - Any extra files listed by the caller (shock files, supplementary
      parameter overrides, etc.)

    The .exp file is rewritten as ``<experiment>.cmf`` with explicit
    ``file GTAPSETS = ...``, ``file GTAPDATA = ...``, ``file GTAPPARM =
    ...``, and ``Auxiliary files = <experiment> ;`` declarations
    prepended. RunGTAP normally injects these via its GUI; running
    ``gtap.exe`` headlessly requires them in the CMF body.
    """
    workdir.mkdir(parents=True, exist_ok=True)

    candidates = [
        dataset_dir / f"{experiment}.exp",
        dataset_dir / f"{experiment}.EXP",
        dataset_dir / f"{experiment}.cmf",
        dataset_dir / f"{experiment}.CMF",
    ]
    exp_file = next((p for p in candidates if p.exists()), None)
    if exp_file is None:
        raise FileNotFoundError(
            f"Experiment {experiment!r} not found in {dataset_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    # Resolve canonical dataset filenames (case-insensitive existence)
    sets_name = _first_existing(dataset_dir, _GTAPSETS_CANDIDATES)
    data_name = _first_existing(dataset_dir, _GTAPDATA_CANDIDATES)
    parm_name = _first_existing(dataset_dir, _GTAPPARM_CANDIDATES)
    if sets_name is None or data_name is None or parm_name is None:
        raise FileNotFoundError(
            f"Missing v6.2 dataset files in {dataset_dir}. "
            f"Need one of {_GTAPSETS_CANDIDATES}, {_GTAPDATA_CANDIDATES}, "
            f"and {_GTAPPARM_CANDIDATES}."
        )

    # Copy supporting files
    for fname in (sets_name, data_name, parm_name, "CMFSTART"):
        src = dataset_dir / fname
        if src.exists():
            shutil.copy2(src, workdir / fname)
    for fname in extra_files:
        src = dataset_dir / fname
        if src.exists():
            shutil.copy2(src, workdir / fname)

    # Build the augmented CMF: prepend file declarations and an
    # auxiliary-files line, then append the original experiment body.
    original_body = exp_file.read_text(encoding="latin-1", errors="replace")

    sol_basename = experiment

    header = (
        f"! Auto-prepended by equilibria GEMPACK oracle\n"
        f"file GTAPSETS = {sets_name} ;\n"
        f"file GTAPDATA = {data_name} ;\n"
        f"file GTAPPARM = {parm_name} ;\n"
        f"Auxiliary files = gtap ;\n"
        f"solution file = {sol_basename} ;\n"
        f"Updated File GTAPDATA = {sol_basename}-upd.har ;\n"
    )
    # Strip any existing file = gtapparm line from the body to avoid
    # duplication (the body's .exp often declares it).
    body_lines = []
    for line in original_body.splitlines():
        lower = line.strip().lower()
        if lower.startswith("file gtapparm") or lower.startswith("file gtapsets") or lower.startswith("file gtapdata"):
            continue
        if lower.startswith("solution file") or lower.startswith("updated file"):
            continue
        if lower.startswith("auxiliary files"):
            continue
        body_lines.append(line)
    cmf_body = "\n".join(body_lines)

    target_cmf = workdir / f"{experiment}.cmf"
    target_cmf.write_text(header + cmf_body, encoding="latin-1")

    return target_cmf


def run_gempack_experiment(
    experiment: str,
    *,
    dataset_dir: Path,
    workdir: Path,
    rungtap_dir: Path = DEFAULT_RUNGTAP_DIR,
    extra_files: Iterable[str] = (),
    timeout_seconds: int = 600,
) -> GempackOracleResult:
    """Run a GTAP v6.2 experiment via ``gtap.exe`` and convert the output.

    Args:
        experiment: Experiment name without extension (e.g. ``"Exp1a"``).
        dataset_dir: Directory containing the dataset (``basedata.har``,
            ``Default.prm``, ``CMFSTART``, ``<experiment>.exp``).
        workdir: Scratch directory for the run. Will be created if
            missing. The function copies the dataset slice here so the
            canonical dataset is never mutated.
        rungtap_dir: Path to the RunGTAP installation.
        extra_files: Optional extra files (shock ``.shk``, alternative
            ``.prm`` overrides) to copy into the workdir.
        timeout_seconds: Hard ceiling on ``gtap.exe`` runtime.

    Returns:
        :class:`GempackOracleResult` with paths to the produced ``.sl4``
        and converted ``.har`` and a status string.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / f"{experiment}.gempack.log"

    # 1. Locate executables. If they don't exist, return early so the
    # caller can decide to skip rather than crash.
    try:
        gtap_exe = _locate_executable(GTAP_EXE_NAME, rungtap_dir)
        sltohta_exe = _locate_executable(SLTOHTA_EXE_NAME, rungtap_dir)
    except FileNotFoundError as exc:
        log_path.write_text(f"missing_exe: {exc}\n", encoding="utf-8")
        return GempackOracleResult(
            experiment=experiment,
            workdir=workdir,
            cmf_path=Path(),
            sl4_path=None,
            solution_har=None,
            log_path=log_path,
            returncode=-1,
            status="missing_exe",
            message=str(exc),
        )

    # 2. Stage the workdir.
    cmf_path = _prepare_workdir(dataset_dir, workdir, experiment, extra_files)

    # 3. Run gtap.exe -cmf <experiment>.cmf in the workdir.
    cmd_gtap = [str(gtap_exe), "-cmf", cmf_path.name]
    logger.info("Running GEMPACK oracle: %s (cwd=%s)", " ".join(cmd_gtap), workdir)
    proc = subprocess.run(
        cmd_gtap,
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )

    log_path.write_text(
        f"$ {' '.join(cmd_gtap)}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
        f"--- returncode: {proc.returncode} ---\n",
        encoding="utf-8",
    )

    sl4_candidates = list(workdir.glob(f"{experiment}*.sl4"))
    sl4_path = sl4_candidates[0] if sl4_candidates else None

    if proc.returncode != 0 or sl4_path is None:
        return GempackOracleResult(
            experiment=experiment,
            workdir=workdir,
            cmf_path=cmf_path,
            sl4_path=sl4_path,
            solution_har=None,
            log_path=log_path,
            returncode=proc.returncode,
            status="solve_failed",
            message=f"gtap.exe failed (rc={proc.returncode}); see {log_path}",
        )

    # 4. Convert .sl4 to .har.
    solution_har = workdir / f"{experiment}_sol.har"
    cmd_conv = [str(sltohta_exe), sl4_path.name, solution_har.name]
    proc_conv = subprocess.run(
        cmd_conv,
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n$ {' '.join(cmd_conv)}\n"
            f"--- stdout ---\n{proc_conv.stdout}\n"
            f"--- stderr ---\n{proc_conv.stderr}\n"
            f"--- returncode: {proc_conv.returncode} ---\n"
        )

    if proc_conv.returncode != 0 or not solution_har.exists():
        return GempackOracleResult(
            experiment=experiment,
            workdir=workdir,
            cmf_path=cmf_path,
            sl4_path=sl4_path,
            solution_har=solution_har if solution_har.exists() else None,
            log_path=log_path,
            returncode=proc_conv.returncode,
            status="convert_failed",
            message=f"sltohta.exe failed (rc={proc_conv.returncode}); see {log_path}",
        )

    return GempackOracleResult(
        experiment=experiment,
        workdir=workdir,
        cmf_path=cmf_path,
        sl4_path=sl4_path,
        solution_har=solution_har,
        log_path=log_path,
        returncode=0,
        status="ok",
        message="",
    )


def is_rungtap_available(rungtap_dir: Path = DEFAULT_RUNGTAP_DIR) -> bool:
    """Quick probe for the RunGTAP install."""
    return (rungtap_dir / GTAP_EXE_NAME).exists() and (
        rungtap_dir / SLTOHTA_EXE_NAME
    ).exists()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a GTAP v6.2 experiment via GEMPACK and dump the solution."
    )
    parser.add_argument(
        "experiment",
        help="Experiment name without extension (e.g. 'Exp1a').",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory (e.g. C:/runGTAP375/BOOK3X3).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="Scratch directory for the run.",
    )
    parser.add_argument(
        "--rungtap-dir",
        type=Path,
        default=DEFAULT_RUNGTAP_DIR,
        help=f"Path to RunGTAP install (default: {DEFAULT_RUNGTAP_DIR}).",
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Optional extra file(s) to copy into workdir from the dataset.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Hard timeout for gtap.exe (seconds).",
    )

    args = parser.parse_args(argv)

    if not is_rungtap_available(args.rungtap_dir):
        print(f"ERROR: RunGTAP not found at {args.rungtap_dir}", flush=True)
        return 2

    result = run_gempack_experiment(
        args.experiment,
        dataset_dir=args.dataset_dir,
        workdir=args.workdir,
        rungtap_dir=args.rungtap_dir,
        extra_files=args.extra_file,
        timeout_seconds=args.timeout,
    )

    print(f"status:      {result.status}")
    print(f"experiment:  {result.experiment}")
    print(f"workdir:     {result.workdir}")
    print(f"cmf:         {result.cmf_path}")
    print(f"sl4:         {result.sl4_path}")
    print(f"solution:    {result.solution_har}")
    print(f"log:         {result.log_path}")
    print(f"returncode:  {result.returncode}")
    if result.message:
        print(f"message:     {result.message}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
