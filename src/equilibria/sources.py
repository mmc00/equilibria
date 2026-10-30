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

# The GAMS GTAP v7 model (comp.gms and its includes) that generates the results.
GAMS_MODEL_DIR = DATA_ROOT / "gtap7" / "gams"

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
        "derived_from",
        "changes",
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
# A model folder derived from another (data/gtap7/gams from gams_jgea) lists
# every file it adds or modifies, with what and why.
_CHANGE_KEYS = frozenset({"file", "change", "what", "why", "commit"})
_CHANGE_KINDS = ("added", "modified")


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
    data = yaml.safe_load(file.read_text(encoding="utf-8"))
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
    if "derived_from" in data or "changes" in data:
        _check_changes(file, data)
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


def _check_changes(file: Path, data: Mapping[str, Any]) -> None:
    if data["kind"] != "model":
        raise SourceError(f"{file}: derived_from/changes are for kind 'model' only")
    if not isinstance(data.get("derived_from"), str) or not isinstance(
        data.get("changes"), list
    ):
        raise SourceError(
            f"{file}: derived_from (a path) and changes (a list) go together"
        )
    for i, entry in enumerate(data["changes"]):
        where = f"{file}: changes[{i}]"
        if not isinstance(entry, Mapping) or set(entry) != _CHANGE_KEYS:
            got = sorted(entry) if isinstance(entry, Mapping) else entry
            raise SourceError(
                f"{where} must have exactly {sorted(_CHANGE_KEYS)}, got {got}"
            )
        if entry["change"] not in _CHANGE_KINDS:
            raise SourceError(f"{where}: change must be one of {_CHANGE_KINDS}")
        empty = [k for k in sorted(_CHANGE_KEYS) if not str(entry[k]).strip()]
        if empty:
            raise SourceError(f"{where}: empty {empty}")


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


def _data_folders(dataset: str, root: Path) -> tuple[list[Path], list[SourceRecord]]:
    data = [r for r in iter_sources(root) if r.kind == "data"]
    return [r.path for r in data if r.dataset == dataset], data


def dataset_dir(dataset: str, *, root: Path = DATA_ROOT) -> Path:
    """The one ``kind: data`` folder of ``dataset``.

    No folder, or more than one, is an error that names what exists.
    """
    matches, data = _data_folders(dataset, root)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        known = sorted({str(r.dataset) for r in data})
        raise SourceError(f"no data folder for dataset {dataset!r}; known: {known}")
    listed = "\n  ".join(str(p) for p in matches)
    raise SourceError(
        f"{len(matches)} data folders for dataset {dataset!r}:\n  {listed}"
    )


def has_dataset(dataset: str, *, root: Path = DATA_ROOT) -> bool:
    """Whether ``dataset`` has a data folder; two folders is still an error."""
    matches, _ = _data_folders(dataset, root)
    if len(matches) > 1:
        dataset_dir(dataset, root=root)
    return bool(matches)


_EXEMPT = frozenset({SOURCE_FILE, "README.md"})


def own_files(folder: Path) -> list[Path]:
    """The files a record in ``folder`` answers for.

    Everything under ``folder`` except its ``source.yaml`` and the files of any
    subfolder that has its own ``source.yaml`` (that one is another record's).
    The writer and the checker both use this, so they agree on what a record
    covers.
    """
    nested = {f.parent for f in folder.rglob(SOURCE_FILE) if f.parent != folder}
    return [
        p
        for p in sorted(folder.rglob("*"))
        if p.is_file()
        and p.name != SOURCE_FILE
        and not any(q in nested for q in p.parents)
    ]


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
        msgs.extend(_check_outputs(folder, where, rec.raw.get("outputs") or {}))
        if "derived_from" in rec.raw:
            msgs.extend(_check_derived(root, folder, where, rec.raw))
    return msgs


def check_derived(folder: Path, root: Path = DATA_ROOT) -> list[str]:
    """check_tree's derived_from/changes check for one model folder."""
    rec = load_source(folder)
    if "derived_from" not in rec.raw:
        return []
    return _check_derived(root, folder, folder.relative_to(root), rec.raw)


def _same_text(a: Path, b: Path) -> bool:
    """Equal up to line endings (the GTAP distribution ships CRLF files)."""
    return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(
        b"\r\n", b"\n"
    )


def _check_derived(
    root: Path, folder: Path, where: Path, raw: Mapping[str, Any]
) -> list[str]:
    base = root / raw["derived_from"]
    if not (base / SOURCE_FILE).is_file():
        return [f"{where}: derived_from {raw['derived_from']} is not a recorded folder"]
    listed = {c["file"]: c["change"] for c in raw["changes"]}
    msgs = []
    for f in own_files(folder):
        rel = f.relative_to(folder).as_posix()
        if rel == SOURCE_FILE:
            continue
        original = base / rel
        actual = (
            "added"
            if not original.is_file()
            else (None if _same_text(f, original) else "modified")
        )
        if actual is None:
            if rel in listed:
                msgs.append(
                    f"{where}: {rel} is listed in changes but identical to the base"
                )
        elif rel not in listed:
            msgs.append(f"{where}: {rel} is {actual} but not listed in changes")
        elif listed[rel] != actual:
            msgs.append(f"{where}: {rel} is listed as {listed[rel]} but is {actual}")
    for rel in sorted(
        set(listed) - {f.relative_to(folder).as_posix() for f in own_files(folder)}
    ):
        msgs.append(f"{where}: changes lists {rel}, which is not in the folder")
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


def _check_outputs(folder: Path, where: Path, outputs: Mapping[str, str]) -> list[str]:
    own = {p.relative_to(folder).as_posix() for p in own_files(folder)}
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
