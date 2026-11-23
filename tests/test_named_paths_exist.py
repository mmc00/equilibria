"""Every repo path the hooks, CI and pytest config name exists.

A deleted script or test left in ``scripts/dev/pre_push_family.sh`` aborts every
push (pytest exits 4 on a missing file); one left in a workflow or a marker's
description points the reader at nothing. The files below are scanned for
``scripts/...`` and ``tests/...`` paths.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "pyproject.toml",
    ROOT / ".pre-commit-config.yaml",
    ROOT / "scripts" / "dev" / "pre_push_family.sh",
    *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
]
PATH = re.compile(r"(?<![\w./-])((?:scripts|tests)/[\w./-]+\.(?:py|sh))")


def named_paths(text: str) -> list[str]:
    return sorted(set(PATH.findall(text)))


@pytest.mark.parametrize("f", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_named_path_exists(f: Path) -> None:
    missing = [p for p in named_paths(f.read_text()) if not (ROOT / p).exists()]
    assert missing == []


def test_the_scan_sees_a_path() -> None:
    assert named_paths("run tests/family/test_x.py and scripts/a/b.sh;") == [
        "scripts/a/b.sh",
        "tests/family/test_x.py",
    ]
