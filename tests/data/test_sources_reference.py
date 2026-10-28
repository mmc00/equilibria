"""reference(): a test asks for a reference by what it is, never by path."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.sources import SourceError, iter_sources, reference
from tests.data._records import result_record, write_record


def _result(root: Path, name: str, **variant: object) -> Path:
    return write_record(root / name, result_record(variant=dict(variant)))


def test_single_match_returns_its_folder(tmp_path: Path) -> None:
    folder = _result(tmp_path, "gams_ifsub0", ifsub=0)
    _result(tmp_path, "gams_ifsub1", ifsub=1)
    assert reference("gtap7_3x3", "gams", root=tmp_path, ifsub=0) == folder


def test_omitted_keys_are_free(tmp_path: Path) -> None:
    folder = _result(tmp_path, "gams_ifsub0", ifsub=0, closure="capFix")
    assert reference("gtap7_3x3", "gams", root=tmp_path, ifsub=0) == folder


def test_two_matches_raise_and_name_both(tmp_path: Path) -> None:
    _result(tmp_path, "gams_capfix", ifsub=0, closure="capFix")
    _result(tmp_path, "gams_capflex", ifsub=0, closure="capFlex")
    with pytest.raises(SourceError) as err:
        reference("gtap7_3x3", "gams", root=tmp_path, ifsub=0)
    assert "gams_capfix" in str(err.value)
    assert "gams_capflex" in str(err.value)


def test_wrong_variant_raises(tmp_path: Path) -> None:
    _result(tmp_path, "gams_ifsub1", ifsub=1)
    with pytest.raises(SourceError, match="gams_ifsub1"):
        reference("gtap7_3x3", "gams", root=tmp_path, ifsub=0)


def test_no_match_lists_candidates(tmp_path: Path) -> None:
    _result(tmp_path, "gams_ifsub0", ifsub=0)
    with pytest.raises(SourceError, match="gtap7_3x3"):
        reference("gtap7_99x99", "gams", root=tmp_path, ifsub=0)


def test_iter_sources_finds_nested_records(tmp_path: Path) -> None:
    _result(tmp_path / "gtap7" / "gtap7_3x3", "gams_ifsub0", ifsub=0)
    assert [r.path.name for r in iter_sources(tmp_path)] == ["gams_ifsub0"]
