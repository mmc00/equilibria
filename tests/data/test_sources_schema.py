"""The source.yaml schema (ADR-0010): a typo or a missing field is an error."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.sources import SourceError, SourceRecord, load_source, sha256_file
from tests.data._records import result_record, write_record


def test_valid_result_loads(tmp_path: Path) -> None:
    rec = load_source(write_record(tmp_path / "r", result_record()))
    assert isinstance(rec, SourceRecord)
    assert rec.kind == "result"
    assert rec.dataset == "gtap7_3x3"
    assert rec.variant == {"ifsub": 0}
    assert rec.path == tmp_path / "r"


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    folder = write_record(tmp_path / "r", result_record(varient={"ifsub": 0}))
    with pytest.raises(SourceError, match="varient"):
        load_source(folder)


def test_missing_required_key_is_an_error(tmp_path: Path) -> None:
    record = result_record()
    del record["command"]
    with pytest.raises(SourceError, match="command"):
        load_source(write_record(tmp_path / "r", record))


def test_machine_rejects_host_name(tmp_path: Path) -> None:
    record = result_record()
    record["run"]["machine"]["host"] = "my-laptop"
    with pytest.raises(SourceError, match="host"):
        load_source(write_record(tmp_path / "r", record))


def test_status_must_be_verified_or_pending(tmp_path: Path) -> None:
    folder = write_record(tmp_path / "r", result_record(status="done"))
    with pytest.raises(SourceError, match="status"):
        load_source(folder)


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"abc")
    assert sha256_file(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
