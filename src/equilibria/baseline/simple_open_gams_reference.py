"""Manifest helpers for official SimpleOpen GAMS benchmark references."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from equilibria.baseline.gams_nlp_reference import (
    GAMSReferenceArtifact,
    ensure_gams_script_uses_nlp,
)
from equilibria.baseline.manifest import file_sha256

_CANONICAL_SIMPLE_OPEN_CLOSURES = (
    "simple_open_default",
    "flexible_external_balance",
)


class SimpleOpenClosureReference(BaseModel):
    """One closure-specific GAMS benchmark reference for SimpleOpen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    closure: str
    results_gdx: GAMSReferenceArtifact
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("closure", mode="before")
    @classmethod
    def _normalize_closure(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("closure must be non-empty.")
        return text


class SimpleOpenGAMSReferenceManifest(BaseModel):
    """Canonical manifest for official SimpleOpen GAMS benchmark artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["simple_open_gams_reference/v1"] = "simple_open_gams_reference/v1"
    generated_at: str
    model: Literal["simple_open_v1"] = "simple_open_v1"
    source: Literal["gams"] = "gams"
    anchor: Literal["stdcge"] = "stdcge"
    problem_type: Literal["nlp"] = "nlp"
    solver: Literal["ipopt"] = "ipopt"
    script_model_types: tuple[str, ...]
    gms_script: GAMSReferenceArtifact
    closure_references: dict[str, SimpleOpenClosureReference]
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
        items = [value] if isinstance(value, str) else list(value or [])
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item).strip().lower()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        if not ordered:
            raise ValueError("script_model_types must be non-empty.")
        return tuple(ordered)

    @field_validator("closure_references", mode="before")
    @classmethod
    def _normalize_closure_references(
        cls,
        value: Any,
    ) -> dict[str, SimpleOpenClosureReference]:
        if not isinstance(value, dict):
            raise TypeError("closure_references must be a mapping.")
        out: dict[str, SimpleOpenClosureReference] = {}
        for raw_name, raw_payload in value.items():
            name = str(raw_name).strip().lower()
            if not name:
                raise ValueError("closure_references keys must be non-empty.")
            if isinstance(raw_payload, SimpleOpenClosureReference):
                out[name] = raw_payload
            elif isinstance(raw_payload, dict):
                payload = dict(raw_payload)
                payload.setdefault("closure", name)
                out[name] = SimpleOpenClosureReference.model_validate(payload)
            else:
                raise TypeError(
                    "closure_references values must be mappings or SimpleOpenClosureReference."
                )
        if not out:
            raise ValueError("closure_references must be non-empty.")
        return out

    @model_validator(mode="after")
    def _check_consistency(self) -> "SimpleOpenGAMSReferenceManifest":
        if "nlp" not in self.script_model_types:
            raise ValueError("SimpleOpen reference script must solve USING NLP.")
        missing = [
            name for name in _CANONICAL_SIMPLE_OPEN_CLOSURES if name not in self.closure_references
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"SimpleOpen reference manifest is missing canonical closures: {joined}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def save_json(self, path: Path | str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2))


def load_simple_open_gams_reference_manifest(
    path: Path | str,
) -> SimpleOpenGAMSReferenceManifest:
    """Load one official SimpleOpen GAMS reference manifest from disk."""

    return SimpleOpenGAMSReferenceManifest.model_validate_json(Path(path).read_text())


def _artifact_from_path(path: Path | str) -> GAMSReferenceArtifact:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Reference artifact not found: {artifact_path}")
    return GAMSReferenceArtifact(path=str(artifact_path), sha256=file_sha256(artifact_path))


def build_simple_open_gams_reference_manifest(
    *,
    gms_script: Path | str,
    closure_references: dict[str, SimpleOpenClosureReference | dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> SimpleOpenGAMSReferenceManifest:
    """Build the canonical manifest for one official SimpleOpen GAMS reference run."""

    script_path = Path(gms_script)
    model_types = ensure_gams_script_uses_nlp(script_path)
    return SimpleOpenGAMSReferenceManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        script_model_types=model_types,
        gms_script=_artifact_from_path(script_path),
        closure_references=closure_references,
        metadata=metadata or {},
    )
