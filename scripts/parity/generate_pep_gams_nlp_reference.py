#!/usr/bin/env python3
"""Generate an official GAMS + IPOPT + NLP reference manifest for PEP."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from equilibria.baseline import build_gams_nlp_reference_manifest


DEFAULT_GMS = (
    REPO_ROOT
    / "src"
    / "equilibria"
    / "templates"
    / "reference"
    / "pep2"
    / "scripts"
    / "PEP-1-1_v2_1_ipopt_excel.gms"
)
DEFAULT_SAM = (
    REPO_ROOT
    / "src"
    / "equilibria"
    / "templates"
    / "reference"
    / "pep2"
    / "data"
    / "SAM-V2_0_connect.xlsx"
)
DEFAULT_VAL_PAR = (
    REPO_ROOT
    / "src"
    / "equilibria"
    / "templates"
    / "reference"
    / "pep2"
    / "data"
    / "VAL_PAR.xlsx"
)


def _parse_scenario_slices(values: list[str] | None) -> dict[str, str]:
    if not values:
        return {"base": "base"}
    out: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --scenario-slice value: {raw!r}. Expected name=slice.")
        name, slice_name = raw.split("=", 1)
        key = name.strip().lower()
        val = slice_name.strip().lower()
        if not key or not val:
            raise ValueError(f"Invalid --scenario-slice value: {raw!r}.")
        out[key] = val
    return out


def _copy_runtime_script_set(gms_script: Path, output_dir: Path) -> tuple[Path, Path]:
    source_dir = gms_script.parent
    runtime_dir = output_dir / "scripts"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    runtime_script = runtime_dir / gms_script.name
    shutil.copy2(gms_script, runtime_script)

    include_file = source_dir / "RESULTS PEP 1-1.GMS"
    if include_file.exists():
        shutil.copy2(include_file, runtime_dir / include_file.name)

    helper_file = source_dir / "generate_dynamic_sets_inc.py"
    if helper_file.exists() and "generate_dynamic_sets_inc.py" in gms_script.read_text():
        shutil.copy2(helper_file, runtime_dir / helper_file.name)

    return runtime_dir, runtime_script


def _run_gams_reference(
    *,
    gams_bin: str,
    runtime_dir: Path,
    runtime_script: Path,
    sam_file: Path,
    val_par_file: Path | None,
    timeout: int,
    solver: str,
) -> None:
    sam_gdx = runtime_dir / f"{sam_file.stem}-from-excel.gdx"
    val_par_gdx = runtime_dir / f"{(val_par_file.stem if val_par_file else 'VAL_PAR')}-from-excel.gdx"

    cmd = [
        gams_bin,
        runtime_script.name,
        f"--PEP_SOLVE_MODE=NLP",
        f"--PEP_SOLVER={solver}",
        f"--SAM_XLS={sam_file}",
        f"--SAM_GDX={sam_gdx}",
    ]
    if val_par_file is not None:
        cmd.extend(
            [
                f"--VAL_PAR_XLS={val_par_file}",
                f"--VAL_PAR_GDX={val_par_gdx}",
            ]
        )

    result = subprocess.run(
        cmd,
        cwd=runtime_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    (runtime_dir / "gams_stdout.log").write_text(result.stdout)
    (runtime_dir / "gams_stderr.log").write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"GAMS NLP reference run failed with exit code {result.returncode}. "
            f"See {runtime_dir / 'gams_stdout.log'} and {runtime_dir / 'gams_stderr.log'}."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate official PEP GAMS NLP reference artifacts")
    parser.add_argument("--gms-script", type=Path, default=DEFAULT_GMS)
    parser.add_argument("--sam-file", type=Path, default=DEFAULT_SAM)
    parser.add_argument("--val-par-file", type=Path, default=DEFAULT_VAL_PAR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/gams_nlp_reference/latest"),
    )
    parser.add_argument("--gams-bin", type=str, default="gams")
    parser.add_argument("--solver", type=str, default="ipopt")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--scenario-slice",
        action="append",
        default=None,
        help="Reference scenario slice mapping, e.g. base=base or government_spending=sim1",
    )
    parser.add_argument(
        "--skip-gams",
        action="store_true",
        help="Do not run GAMS; build the manifest from existing artifacts in output-dir/scripts.",
    )
    parser.add_argument("--results-gdx", type=Path, default=None)
    parser.add_argument("--parameters-gdx", type=Path, default=None)
    parser.add_argument("--presolve-levels-gdx", type=Path, default=None)
    parser.add_argument("--manifest-name", type=str, default="manifest.json")
    args = parser.parse_args()

    scenario_slices = _parse_scenario_slices(args.scenario_slice)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_dir, runtime_script = _copy_runtime_script_set(args.gms_script.resolve(), output_dir)

    if args.skip_gams:
        results_gdx = (args.results_gdx or (runtime_dir / "Results.gdx")).resolve()
        parameters_gdx = (
            args.parameters_gdx.resolve() if args.parameters_gdx else (runtime_dir / "Parameters.gdx").resolve()
        )
        presolve_levels_gdx = (
            args.presolve_levels_gdx.resolve()
            if args.presolve_levels_gdx
            else (runtime_dir / "PreSolveLevels.gdx").resolve()
        )
    else:
        _run_gams_reference(
            gams_bin=args.gams_bin,
            runtime_dir=runtime_dir,
            runtime_script=runtime_script,
            sam_file=args.sam_file.resolve(),
            val_par_file=args.val_par_file.resolve() if args.val_par_file else None,
            timeout=args.timeout,
            solver=args.solver,
        )
        results_gdx = (runtime_dir / "Results.gdx").resolve()
        parameters_gdx = (runtime_dir / "Parameters.gdx").resolve()
        presolve_levels_gdx = (runtime_dir / "PreSolveLevels.gdx").resolve()

    manifest = build_gams_nlp_reference_manifest(
        gms_script=runtime_script,
        sam_file=args.sam_file.resolve(),
        val_par_file=args.val_par_file.resolve() if args.val_par_file else None,
        results_gdx=results_gdx,
        parameters_gdx=parameters_gdx if parameters_gdx.exists() else None,
        presolve_levels_gdx=presolve_levels_gdx if presolve_levels_gdx.exists() else None,
        scenario_slices=scenario_slices,
        metadata={
            "workspace": str(runtime_dir),
            "skip_gams": bool(args.skip_gams),
            "solver": args.solver,
        },
    )

    manifest_path = output_dir / args.manifest_name
    manifest.save_json(manifest_path)

    print(json.dumps({"manifest": str(manifest_path), "scenario_slices": scenario_slices}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
