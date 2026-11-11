"""load_source() parses each source.yaml once until the file changes.

reference() and dataset_dir() walk every source.yaml under data/ on each call;
without a cache, a caller that asks for many references (scripts/paper/run_tables.py,
40 calls over ~400 files) spent minutes in the YAML parser.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import equilibria.sources as sources
from equilibria.sources import SourceError, reference
from tests.data._records import result_record, write_record


def _count_parses(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    seen: list[Path] = []
    real = sources._read_yaml

    def counting(file: Path) -> object:
        seen.append(file)
        return real(file)

    monkeypatch.setattr(sources, "_read_yaml", counting)
    return seen


def test_repeated_lookups_parse_each_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_record(tmp_path / "gams_ifsub0", result_record(variant={"ifsub": 0}))
    write_record(tmp_path / "gams_ifsub1", result_record(variant={"ifsub": 1}))
    seen = _count_parses(monkeypatch)
    for _ in range(5):
        reference("gtap7_3x3", "gams", root=tmp_path, ifsub=0)
    assert len(seen) == 2


def test_an_edited_file_is_read_again(tmp_path: Path) -> None:
    folder = write_record(tmp_path / "gams", result_record(variant={"ifsub": 0}))
    assert reference("gtap7_3x3", "gams", root=tmp_path, ifsub=0) == folder
    write_record(folder, result_record(variant={"ifsub": 1}))
    st = (folder / "source.yaml").stat()
    os.utime(folder / "source.yaml", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert reference("gtap7_3x3", "gams", root=tmp_path, ifsub=1) == folder
    with pytest.raises(SourceError):
        reference("gtap7_3x3", "gams", root=tmp_path, ifsub=0)


def test_an_invalid_file_raises_every_time(tmp_path: Path) -> None:
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "source.yaml").write_text("kind: result\n", encoding="utf-8")
    for _ in range(2):
        with pytest.raises(SourceError):
            sources.load_source(folder)
