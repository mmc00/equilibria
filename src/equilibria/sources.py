"""Where every reference file under ``data/`` came from (ADR-0010).

Each folder of reference results, each dataset folder and each reference-model
folder carries a ``source.yaml``. This module reads it, rejects typos and
missing fields, and is the only place that knows its shape.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Kind = Literal["result", "data", "model"]
Status = Literal["verified", "pending"]

SOURCE_FILE = "source.yaml"

# The repository's data/ (this file is src/equilibria/sources.py). Overridable
# for a checkout whose data lives elsewhere.
DATA_ROOT = Path(
    os.environ.get("EQUILIBRIA_DATA_DIR")
    or Path(__file__).resolve().parents[2] / "data"
)

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


def iter_sources(root: Path = DATA_ROOT) -> Iterator[SourceRecord]:
    """Every ``source.yaml`` under ``root``, in path order."""
    for file in sorted(root.rglob(SOURCE_FILE)):
        yield load_source(file.parent)


def reference(
    dataset: str, tool: str, *, root: Path = DATA_ROOT, **variant: Any
) -> Path:
    """The one result folder of ``dataset`` made with ``tool`` in ``variant``.

    A folder matches when its dataset and tool are equal and every requested
    variant key is recorded with the same value; keys left out are free. No
    match, or more than one, is an error that names the candidates.
    """
    results = [r for r in iter_sources(root) if r.kind == "result"]
    candidates = [r for r in results if r.dataset == dataset and r.tool == tool]
    matches = [
        r
        for r in candidates
        if all(k in r.variant and r.variant[k] == v for k, v in variant.items())
    ]
    if len(matches) == 1:
        return matches[0].path
    asked = f"reference({dataset!r}, {tool!r}, {variant})"
    if not candidates:
        known = sorted({f"{r.dataset}/{r.tool}" for r in results})
        raise SourceError(
            f"{asked}: no result for that dataset and tool; known: {known}"
        )
    listed = "\n  ".join(f"{r.path}: {dict(r.variant)}" for r in candidates)
    what = "no folder matches" if not matches else f"{len(matches)} folders match"
    raise SourceError(f"{asked}: {what}; candidates:\n  {listed}")


_EXEMPT = frozenset({SOURCE_FILE, "README.md"})


def check_tree(root: Path = DATA_ROOT, *, allow_pending: bool = False) -> list[str]:
    """Every problem under ``root``, one message each; empty when clean.

    1. every file belongs to a folder (or an ancestor) with a source.yaml;
    2. a verified record's inputs (``generated_from``) still have their sha256;
    3. a record's ``outputs`` are exactly its folder's files, with their sha256;
    4. no record is ``pending`` (unless ``allow_pending``).
    """
    msgs: list[str] = []
    records: dict[Path, SourceRecord] = {}
    for file in sorted(root.rglob(SOURCE_FILE)):
        try:
            records[file.parent] = load_source(file.parent)
        except SourceError as err:
            msgs.append(str(err))

    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        if file.name in _EXEMPT and (file.name == SOURCE_FILE or file.parent == root):
            continue
        if not any(parent in records for parent in file.parents):
            msgs.append(
                f"{file.relative_to(root)}: no source.yaml in its folder or above"
            )

    for folder, rec in records.items():
        where = folder.relative_to(root)
        if rec.status == "pending" and not allow_pending:
            msgs.append(f"{where}: status pending")
        if rec.status == "verified":
            msgs.extend(_check_inputs(root, where, rec.raw.get("generated_from") or {}))
        msgs.extend(
            _check_outputs(folder, where, rec.raw.get("outputs") or {}, records)
        )
    return msgs


def _check_inputs(root: Path, where: Path, inputs: Mapping[str, Any]) -> list[str]:
    msgs = []
    for name, spec in inputs.items():
        base = root / spec["path"]
        hashes = spec["sha256"]
        files = {"": hashes} if isinstance(hashes, str) else hashes
        for rel, expected in files.items():
            f = base / rel if rel else base
            if not f.exists():
                msgs.append(f"{where}: input {name} {f.relative_to(root)} is missing")
            elif sha256_file(f) != expected:
                msgs.append(
                    f"{where}: input {name} {f.relative_to(root)} changed after the "
                    "result was generated"
                )
    return msgs


def _check_outputs(
    folder: Path,
    where: Path,
    outputs: Mapping[str, str],
    records: Mapping[Path, SourceRecord],
) -> list[str]:
    # A subfolder with its own source.yaml is another record's business.
    own = {
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file()
        and p.name != SOURCE_FILE
        and not any(
            q in records for q in p.parents if q != folder and folder in q.parents
        )
    }
    msgs = [
        f"{where}: {rel} is not listed in outputs" for rel in sorted(own - set(outputs))
    ]
    for rel, expected in outputs.items():
        f = folder / rel
        if not f.exists():
            msgs.append(f"{where}: output {rel} is missing")
        elif sha256_file(f) != expected:
            msgs.append(f"{where}: output {rel} does not match its sha256 (edited?)")
    return msgs
