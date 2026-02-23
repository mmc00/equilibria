"""Load and validate YAML configuration for SAM workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from equilibria.sam_tools.models import SAMWorkflowConfig
from equilibria.sam_tools.selectors import norm_text, norm_text_lower


def _norm_format(fmt: str) -> str:
    value = norm_text_lower(fmt)
    if value in {"xlsx", "xls", "excel", "pep_excel"}:
        return "excel"
    if value == "gdx":
        return "gdx"
    raise ValueError(f"Unsupported SAM format: {fmt}")


def _infer_format_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    if suffix == ".gdx":
        return "gdx"
    raise ValueError(f"Could not infer SAM format from path: {path}")


def _resolve_path(path_value: Any, base_dir: Path, field_name: str) -> Path:
    if not path_value:
        raise ValueError(f"Missing required config field: {field_name}")
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_workflow_config(config_path: Path) -> SAMWorkflowConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Workflow YAML must define a top-level mapping")

    base_dir = config_path.parent
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")

    input_cfg = payload.get("input") or {}
    output_cfg = payload.get("output") or {}
    if not isinstance(input_cfg, dict) or not isinstance(output_cfg, dict):
        raise ValueError("input/output must be mappings")

    input_path = _resolve_path(input_cfg.get("path"), base_dir, "input.path")
    output_path = _resolve_path(output_cfg.get("path"), base_dir, "output.path")

    input_format = _norm_format(input_cfg.get("format") or _infer_format_from_path(input_path))
    output_format = _norm_format(output_cfg.get("format") or _infer_format_from_path(output_path))

    transforms = payload.get("transforms") or []
    if not isinstance(transforms, list):
        raise ValueError("transforms must be a list")

    report_path_value = payload.get("report_path")
    report_path: Path | None = None
    if report_path_value:
        report_path = _resolve_path(report_path_value, base_dir, "report_path")

    return SAMWorkflowConfig(
        name=norm_text(metadata.get("name") or config_path.stem),
        country=metadata.get("country"),
        input_path=input_path,
        input_format=input_format,
        output_path=output_path,
        output_format=output_format,
        transforms=transforms,
        report_path=report_path,
        output_symbol=norm_text(output_cfg.get("symbol") or "SAM"),
    )
