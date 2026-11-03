"""A model folder derived from another lists every file it changed, with why."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from equilibria.sources import SourceError, check_tree, load_source, sha256_file
from tests.data._records import write_record


def _model(folder: Path, files: dict[str, bytes], **extra: Any) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (folder / name).write_bytes(body)
    record = {
        "kind": "model",
        "model": "gtap7",
        "source": "test",
        "outputs": {n: sha256_file(folder / n) for n in files},
        "status": "verified",
        **extra,
    }
    return write_record(folder, record)


def _change(file: str, change: str = "modified") -> dict[str, str]:
    return {
        "file": file,
        "change": change,
        "what": "w",
        "why": "y",
        "commit": "abc1234",
    }


def _tree(root: Path, derived: dict[str, bytes], changes: list[dict[str, str]]) -> None:
    _model(root / "gtap7" / "base", {"a.gms": b"one\r\ntwo\r\n", "b.gms": b"b\r\n"})
    _model(
        root / "gtap7" / "derived",
        derived,
        derived_from="gtap7/base",
        changes=changes,
    )


def test_every_change_listed_is_clean(tmp_path: Path) -> None:
    # a.gms equal up to line endings, b.gms edited, c.gms added: listed as such
    _tree(
        tmp_path,
        {"a.gms": b"one\ntwo\n", "b.gms": b"B\n", "c.gms": b"c\n"},
        [_change("b.gms"), _change("c.gms", "added")],
    )
    assert check_tree(tmp_path) == []


def test_unlisted_edit_is_red(tmp_path: Path) -> None:
    _tree(tmp_path, {"a.gms": b"one\nTWO\n"}, [])
    msgs = check_tree(tmp_path)
    assert any("a.gms" in m and "not listed in changes" in m for m in msgs), msgs


def test_unlisted_new_file_is_red(tmp_path: Path) -> None:
    _tree(tmp_path, {"c.gms": b"c\n"}, [])
    msgs = check_tree(tmp_path)
    assert any("c.gms" in m and "not listed in changes" in m for m in msgs), msgs


def test_change_on_identical_file_is_red(tmp_path: Path) -> None:
    # a listed change must be a real one: a stale entry hides nothing but lies
    _tree(tmp_path, {"a.gms": b"one\ntwo\n"}, [_change("a.gms")])
    msgs = check_tree(tmp_path)
    assert any("a.gms" in m and "identical" in m for m in msgs), msgs


def test_wrong_change_kind_is_red(tmp_path: Path) -> None:
    _tree(tmp_path, {"b.gms": b"B\n"}, [_change("b.gms", "added")])
    msgs = check_tree(tmp_path)
    assert any("b.gms" in m and "modified" in m for m in msgs), msgs


def test_missing_base_is_red(tmp_path: Path) -> None:
    _model(
        tmp_path / "gtap7" / "derived",
        {"a.gms": b"a"},
        derived_from="gtap7/nope",
        changes=[],
    )
    msgs = check_tree(tmp_path)
    assert any("gtap7/nope" in m for m in msgs), msgs


@pytest.mark.parametrize(
    "entry",
    [
        {"file": "b.gms", "change": "modified", "what": "w", "why": "y"},  # no commit
        {"file": "b.gms", "change": "renamed", "what": "w", "why": "y", "commit": "c"},
        {"file": "b.gms", "change": "modified", "what": "w", "why": "", "commit": "c"},
    ],
)
def test_malformed_change_entry_raises(tmp_path: Path, entry: dict[str, str]) -> None:
    folder = _model(
        tmp_path / "gtap7" / "derived",
        {"b.gms": b"B"},
        derived_from="gtap7/base",
        changes=[entry],
    )
    with pytest.raises(SourceError, match=r"changes\[0\]"):
        load_source(folder)


def test_changes_only_on_model(tmp_path: Path) -> None:
    folder = tmp_path / "x"
    write_record(
        folder,
        {
            "kind": "data",
            "model": "gtap7",
            "dataset": "x",
            "source": "s",
            "license": "l",
            "outputs": {},
            "status": "verified",
            "derived_from": "gtap7/base",
            "changes": [],
        },
    )
    with pytest.raises(SourceError, match="kind 'model'"):
        load_source(folder)
