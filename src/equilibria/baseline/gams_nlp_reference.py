"""Manifest and helpers for official GAMS + IPOPT + NLP reference runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from equilibria.baseline.manifest import file_sha256

_SOLVE_USING_RE = re.compile(r"\bSOLVE\s+\w+\s+USING\s+(\w+)\b", re.IGNORECASE)


def extract_gams_solve_model_types(gms_script: Path | str) -> tuple[str, ...]:
    """Return the ordered unique `USING <model_type>` solve directives in a GAMS script."""

    script_path = Path(gms_script)
    text = script_path.read_text()
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _SOLVE_USING_RE.findall(text):
        model_type = match.strip().lower()
        if not model_type or model_type in seen:
            continue
        seen.add(model_type)
        ordered.append(model_type)
    return tuple(ordered)


def ensure_gams_script_uses_nlp(gms_script: Path | str) -> tuple[str, ...]:
    """Validate that the reference GAMS script explicitly solves the model as NLP."""

    model_types = extract_gams_solve_model_types(gms_script)
    if "nlp" not in model_types:
        script_path = Path(gms_script)
        types_text = ", ".join(model_types) if model_types else "none"
        raise ValueError(
            f"GAMS reference script must use NLP, but {script_path.name} declares: {types_text}"
        )
    return model_types


class GAMSReferenceArtifact(BaseModel):
    """One file artifact produced or consumed by a reference run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str

    @field_validator("path", "sha256", mode="before")
    @classmethod
    def _normalize_required_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Reference artifact fields must be non-empty.")
        return text


class GAMSNLPReferenceManifest(BaseModel):
    """Canonical manifest for one official GAMS + IPOPT + NLP reference run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["pep_gams_nlp_reference/v1"] = "pep_gams_nlp_reference/v1"
    generated_at: str
    model: Literal["pep"] = "pep"
    source: Literal["gams"] = "gams"
    problem_type: Literal["nlp"] = "nlp"
    solver: Literal["ipopt"] = "ipopt"
    script_model_types: tuple[str, ...]
    gms_script: GAMSReferenceArtifact
    sam_file: GAMSReferenceArtifact
    val_par_file: GAMSReferenceArtifact | None = None
    results_gdx: GAMSReferenceArtifact
    parameters_gdx: GAMSReferenceArtifact | None = None
    presolve_levels_gdx: GAMSReferenceArtifact | None = None
    scenario_slices: dict[str, str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generated_at", mode="before")
    @classmethod
    def _normalize_generated_at(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("generated_at must be non-empty.")
        return text

    @field_validator("script_model_types", mode="before")
    @classmethod
    def _normalize_model_types(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            raise ValueError("script_model_types must be non-empty.")
        if isinstance(value, str):
            items = [value]
        else:
            items = list(value)

        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            text = str(item).strip().lower()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        if not ordered:
            raise ValueError("script_model_types must be non-empty.")
        return tuple(ordered)

    @field_validator("scenario_slices", mode="before")
    @classmethod
    def _normalize_scenario_slices(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            raise TypeError("scenario_slices must be a mapping.")
        normalized: dict[str, str] = {}
        for raw_name, raw_slice in value.items():
            name = str(raw_name).strip().lower()
            slice_name = str(raw_slice).strip().lower()
            if not name or not slice_name:
                raise ValueError("scenario_slices keys and values must be non-empty.")
            normalized[name] = slice_name
        if not normalized:
            raise ValueError("scenario_slices must be non-empty.")
        return normalized

    @model_validator(mode="after")
    def _check_consistency(self) -> "GAMSNLPReferenceManifest":
        if "nlp" not in self.script_model_types:
            raise ValueError("Reference manifest requires a GAMS script that solves USING NLP.")
        if "base" not in self.scenario_slices:
            raise ValueError("Reference manifest must include a 'base' scenario slice.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def save_json(self, path: Path | str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2))


def load_gams_nlp_reference_manifest(path: Path | str) -> GAMSNLPReferenceManifest:
    """Load one official GAMS NLP reference manifest from disk."""

    return GAMSNLPReferenceManifest.model_validate_json(Path(path).read_text())


def _artifact_from_path(path: Path | str | None) -> GAMSReferenceArtifact | None:
    if path is None:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Reference artifact not found: {artifact_path}")
    return GAMSReferenceArtifact(path=str(artifact_path), sha256=file_sha256(artifact_path))


def build_gams_nlp_reference_manifest(
    *,
    gms_script: Path | str,
    sam_file: Path | str,
    scenario_slices: dict[str, str],
    results_gdx: Path | str,
    val_par_file: Path | str | None = None,
    parameters_gdx: Path | str | None = None,
    presolve_levels_gdx: Path | str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GAMSNLPReferenceManifest:
    """Build the canonical manifest for one GAMS + IPOPT + NLP reference run."""

    script_path = Path(gms_script)
    model_types = ensure_gams_script_uses_nlp(script_path)

    return GAMSNLPReferenceManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        script_model_types=model_types,
        gms_script=_artifact_from_path(script_path),
        sam_file=_artifact_from_path(sam_file),
        val_par_file=_artifact_from_path(val_par_file),
        results_gdx=_artifact_from_path(results_gdx),
        parameters_gdx=_artifact_from_path(parameters_gdx),
        presolve_levels_gdx=_artifact_from_path(presolve_levels_gdx),
        scenario_slices=scenario_slices,
        metadata=metadata or {},
    )
