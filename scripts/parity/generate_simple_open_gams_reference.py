#!/usr/bin/env python3
"""Generate the official GAMS benchmark reference manifest for SimpleOpen."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from equilibria.baseline import (  # noqa: E402
    SimpleOpenClosureReference,
    build_simple_open_gams_reference_manifest,
    file_sha256,
)

DEFAULT_GMS = (
    REPO_ROOT
    / "src"
    / "equilibria"
    / "templates"
    / "reference"
    / "simple_open"
    / "scripts"
    / "simple_open_v1_benchmark.gms"
)
CANONICAL_CLOSURES = (
    "simple_open_default",
    "flexible_external_balance",
)


def _normalize_closures(raw_values: list[str] | None) -> tuple[str, ...]:
    values = list(CANONICAL_CLOSURES) if not raw_values else raw_values
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name not in CANONICAL_CLOSURES:
            allowed = ", ".join(CANONICAL_CLOSURES)
            raise ValueError(f"Unsupported closure '{raw}'. Allowed: {allowed}")
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _parse_gdx_overrides(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return {}
    out: dict[str, Path] = {}
    for raw in values:
        closure, sep, path_text = str(raw).partition("=")
        if not sep:
            raise ValueError(f"Invalid --gdx value: {raw!r}. Expected closure=path.")
        name = closure.strip().lower()
        if name not in CANONICAL_CLOSURES:
            allowed = ", ".join(CANONICAL_CLOSURES)
            raise ValueError(f"Unsupported closure '{closure}'. Allowed: {allowed}")
        out[name] = Path(path_text).expanduser().resolve()
    return out


def _copy_runtime_script(gms_script: Path, runtime_dir: Path) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_script = runtime_dir / gms_script.name
    shutil.copy2(gms_script, runtime_script)
    return runtime_script


def _run_gams_reference(
    *,
    gams_bin: str,
    runtime_dir: Path,
    runtime_script: Path,
    closure: str,
    out_gdx: Path,
    timeout: int,
) -> None:
    cmd = [
        gams_bin,
        runtime_script.name,
        "lo=2",
        f"--CLOSURE={closure}",
        f"--OUT_GDX={out_gdx}",
    ]
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
            f"SimpleOpen GAMS reference run failed for '{closure}' with exit code {result.returncode}. "
            f"See {runtime_dir / 'gams_stdout.log'} and {runtime_dir / 'gams_stderr.log'}."
        )


def _artifact_payload(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate official SimpleOpen GAMS reference artifacts")
    parser.add_argument("--gms-script", type=Path, default=DEFAULT_GMS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/simple_open_gams_reference/latest"),
    )
    parser.add_argument("--gams-bin", type=str, default="gams")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--closure",
        action="append",
        default=None,
        help="Closure to generate. Repeatable. Defaults to both canonical closures.",
    )
    parser.add_argument(
        "--gdx",
        action="append",
        default=None,
        help="When --skip-gams is set, provide one or more closure=path overrides.",
    )
    parser.add_argument(
        "--skip-gams",
        action="store_true",
        help="Do not run GAMS; build the manifest from existing GDX artifacts.",
    )
    parser.add_argument("--manifest-name", type=str, default="manifest.json")
    args = parser.parse_args()

    closures = _normalize_closures(args.closure)
    gdx_overrides = _parse_gdx_overrides(args.gdx)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    closure_references: dict[str, SimpleOpenClosureReference] = {}
    workspaces: dict[str, str] = {}
    for closure in closures:
        workspace_dir = output_dir / "closures" / closure
        runtime_dir = workspace_dir / "scripts"
        runtime_script = _copy_runtime_script(args.gms_script.resolve(), runtime_dir)
        workspaces[closure] = str(runtime_dir)

        if args.skip_gams:
            results_gdx = gdx_overrides.get(closure)
            if results_gdx is None:
                raise ValueError(
                    f"--skip-gams requires a --gdx override for closure '{closure}'."
                )
            if not results_gdx.exists():
                raise FileNotFoundError(f"Missing GDX artifact for closure '{closure}': {results_gdx}")
        else:
            results_gdx = (workspace_dir / "simple_open_v1_benchmark.gdx").resolve()
            _run_gams_reference(
                gams_bin=args.gams_bin,
                runtime_dir=runtime_dir,
                runtime_script=runtime_script,
                closure=closure,
                out_gdx=results_gdx,
                timeout=args.timeout,
            )

        closure_references[closure] = SimpleOpenClosureReference.model_validate(
            {
                "closure": closure,
                "results_gdx": _artifact_payload(results_gdx),
                "metadata": {
                    "workspace": str(runtime_dir),
                    "skip_gams": bool(args.skip_gams),
                },
            }
        )

    manifest = build_simple_open_gams_reference_manifest(
        gms_script=args.gms_script.resolve(),
        closure_references=closure_references,
        metadata={
            "closures": list(closures),
            "workspaces": workspaces,
            "skip_gams": bool(args.skip_gams),
        },
    )
    manifest_path = output_dir / args.manifest_name
    manifest.save_json(manifest_path)

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "closures": list(closures),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
