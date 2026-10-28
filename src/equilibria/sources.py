"""Where every reference file under ``data/`` came from (ADR-0010).

Each folder of reference results, each dataset folder and each reference-model
folder carries a ``source.yaml``. This module reads it, rejects typos and
missing fields, and is the only place that knows its shape.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Kind = Literal["result", "data", "model"]
Status = Literal["verified", "pending"]

SOURCE_FILE = "source.yaml"

_ALLOWED = frozenset(
    {
        "kind",
        "model",
        "dataset",
        "tool",
        "tool_version",
        "variant",
        "phases",
        "generated_from",
        "solver",
        "command",
        "equilibria_commit",
        "exit",
        "outputs",
        "run",
        "status",
        "notes",
        "source",
        "license",
    }
)
_REQUIRED: dict[str, tuple[str, ...]] = {
    "result": (
        "model",
        "dataset",
        "tool",
        "variant",
        "generated_from",
        "command",
        "outputs",
        "run",
        "status",
    ),
    "data": ("model", "dataset", "source", "license", "outputs", "status"),
    "model": ("model", "source", "outputs", "status"),
}
_RUN_KEYS = frozenset({"date", "duration_s", "machine"})
# Only these: a host or user name must never be recorded.
_MACHINE_KEYS = frozenset({"os", "cpu", "cores", "ram_gb"})
_STATUSES = ("verified", "pending")


class SourceError(ValueError):
    """A source.yaml that is missing, malformed, or does not match a request."""


@dataclass(frozen=True)
class SourceRecord:
    kind: Kind
    model: str
    dataset: str | None
    tool: str | None
    status: Status
    path: Path
    variant: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(folder: Path) -> SourceRecord:
    """Read and validate ``<folder>/source.yaml``."""
    file = folder / SOURCE_FILE
    if not file.exists():
        raise SourceError(f"{file}: missing")
    data = yaml.safe_load(file.read_text())
    if not isinstance(data, Mapping):
        raise SourceError(f"{file}: not a mapping")

    unknown = sorted(set(data) - _ALLOWED)
    if unknown:
        raise SourceError(f"{file}: unknown keys {unknown}")
    kind = data.get("kind")
    if kind not in _REQUIRED:
        raise SourceError(
            f"{file}: kind must be one of {sorted(_REQUIRED)}, got {kind!r}"
        )
    missing = [k for k in _REQUIRED[kind] if k not in data]
    if missing:
        raise SourceError(f"{file}: kind {kind!r} requires {missing}")
    if data["status"] not in _STATUSES:
        raise SourceError(
            f"{file}: status must be one of {_STATUSES}, got {data['status']!r}"
        )
    if "run" in data:
        _check_run(file, data["run"])
    variant = data.get("variant") or {}
    if not isinstance(variant, Mapping):
        raise SourceError(f"{file}: variant must be a mapping")

    return SourceRecord(
        kind=kind,
        model=str(data["model"]),
        dataset=data.get("dataset"),
        tool=data.get("tool"),
        status=data["status"],
        path=folder,
        variant=dict(variant),
        raw=data,
    )


def _check_run(file: Path, run: Any) -> None:
    if not isinstance(run, Mapping):
        raise SourceError(f"{file}: run must be a mapping")
    if set(run) != _RUN_KEYS:
        raise SourceError(
            f"{file}: run must have exactly {sorted(_RUN_KEYS)}, got {sorted(run)}"
        )
    machine = run["machine"]
    if not isinstance(machine, Mapping) or set(machine) != _MACHINE_KEYS:
        got = sorted(machine) if isinstance(machine, Mapping) else machine
        raise SourceError(
            f"{file}: run.machine must have exactly {sorted(_MACHINE_KEYS)}, got {got}"
        )
