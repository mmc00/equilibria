"""Builders of minimal valid source.yaml records, shared by the tests of data/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def result_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "result",
        "model": "gtap7",
        "dataset": "gtap7_3x3",
        "tool": "gams",
        "variant": {"ifsub": 0},
        "generated_from": {},
        "command": "gams comp.gms",
        "outputs": {},
        "run": {
            "date": "2026-10-28T21:14:03-06:00",
            "duration_s": 1.5,
            "machine": {"os": "macOS", "cpu": "Apple M3", "cores": 8, "ram_gb": 16.0},
        },
        "status": "verified",
    }
    record.update(overrides)
    return record


def data_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "data",
        "model": "gtap7",
        "dataset": "gtap7_3x3",
        "source": "test",
        "license": "test",
        "outputs": {},
        "status": "verified",
    }
    record.update(overrides)
    return record


def write_record(folder: Path, record: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    return folder
