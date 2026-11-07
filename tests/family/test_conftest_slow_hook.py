"""The slow-skip hook of tests/family/conftest.py marks only the tests of this directory."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from tests.family import conftest as family_conftest

HERE = Path(family_conftest.__file__).parent


def _hook(items: list) -> None:
    config = cast(pytest.Config, _Config())
    family_conftest.pytest_collection_modifyitems(config, items)


class _Config:
    def getoption(self, name):
        return ""


class _Item:
    def __init__(self, path: Path, slow: bool):
        self.path = path
        self._slow = slow
        self.marks: list = []

    def get_closest_marker(self, name):
        return object() if (self._slow and name == "slow") else None

    def add_marker(self, mark):
        self.marks.append(mark)


def test_slow_test_outside_tests_family_is_not_skipped(monkeypatch):
    monkeypatch.delenv("EQUILIBRIA_RUN_SLOW", raising=False)
    inside = _Item(HERE / "test_core.py", slow=True)
    outside = _Item(
        HERE.parent / "templates" / "gtap_julia" / "test_capital_eqs.py", slow=True
    )
    fast = _Item(HERE / "test_core.py", slow=False)
    _hook([inside, outside, fast])
    assert len(inside.marks) == 1
    assert outside.marks == []
    assert fast.marks == []


def test_slow_runs_when_opted_in(monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_RUN_SLOW", "1")
    inside = _Item(HERE / "test_core.py", slow=True)
    _hook([inside])
    assert inside.marks == []


def test_marker_is_a_skip(monkeypatch):
    monkeypatch.delenv("EQUILIBRIA_RUN_SLOW", raising=False)
    inside = _Item(HERE / "test_core.py", slow=True)
    _hook([inside])
    assert inside.marks[0].name == "skip"
    _ = pytest
