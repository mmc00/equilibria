"""dataset_dir(): code finds an input dataset by its name, never by path."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.sources import SourceError, dataset_dir, has_dataset
from tests.data._records import data_record, result_record, write_record


def test_finds_the_data_folder(tmp_path: Path) -> None:
    folder = write_record(tmp_path / "gtap7" / "gtap7_3x3", data_record())
    assert dataset_dir("gtap7_3x3", root=tmp_path) == folder
    assert has_dataset("gtap7_3x3", root=tmp_path)


def test_unknown_dataset_lists_known(tmp_path: Path) -> None:
    write_record(tmp_path / "gtap7" / "gtap7_3x3", data_record())
    with pytest.raises(SourceError, match=r"gtap7_9x9.*known: \['gtap7_3x3'\]"):
        dataset_dir("gtap7_9x9", root=tmp_path)
    assert not has_dataset("gtap7_9x9", root=tmp_path)


def test_two_folders_same_dataset_raise(tmp_path: Path) -> None:
    write_record(tmp_path / "gtap6" / "x", data_record(model="gtap6", dataset="x"))
    write_record(tmp_path / "gtap7" / "x", data_record(dataset="x"))
    with pytest.raises(SourceError) as e:
        dataset_dir("x", root=tmp_path)
    assert "gtap6" in str(e.value) and "gtap7" in str(e.value)


def test_result_folder_is_not_data(tmp_path: Path) -> None:
    write_record(tmp_path / "gtap7" / "gtap7_3x3" / "gams_ifsub0", result_record())
    assert not has_dataset("gtap7_3x3", root=tmp_path)
