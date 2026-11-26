"""``equilibria.gtap.load``: a bundled base, a ``data/`` dataset or any folder of GTAP HAR files."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from equilibria.datasets import dataset_path, load_bundled
from equilibria.gtap import load
from equilibria.sources import dataset_dir, has_dataset
from equilibria.templates.gtap import GTAPParameters

FILES = ("basedata.har", "sets.har", "default.prm", "baserate.har")


PARTS = ("benchmark", "elasticities", "taxes", "shares", "shifts", "calibrated")


def _same(a: GTAPParameters, b: GTAPParameters) -> None:
    for part in PARTS:
        assert vars(getattr(a, part)) == vars(getattr(b, part)), part


def _copy(dst: Path, files=FILES) -> Path:
    src = dataset_path("gtap", "nus333")
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(src / f, dst / f)
    return dst


def test_bundled_equals_load_bundled() -> None:
    _same(load("nus333"), load_bundled("gtap", "nus333"))


def test_folder_equals_bundled(tmp_path: Path) -> None:
    _same(load(_copy(tmp_path / "base")), load("nus333"))
    _same(load(str(_copy(tmp_path / "base"))), load("nus333"))


def test_data_dataset() -> None:
    if not has_dataset("gtap7_10x7"):
        pytest.skip("gtap7_10x7 not in data/ (only in a checkout)")
    d = dataset_dir("gtap7_10x7")
    want = GTAPParameters()
    want.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    _same(load("gtap7_10x7"), want)


def test_prm_selects_elasticities() -> None:
    d, c = load("nus333"), load("nus333", prm="3x3ces.prm")
    assert vars(d.benchmark) == vars(c.benchmark)
    assert vars(d.elasticities) != vars(c.elasticities)


def test_missing_file_names_it(tmp_path: Path) -> None:
    folder = _copy(tmp_path / "base", ("basedata.har", "default.prm"))
    with pytest.raises(FileNotFoundError) as e:
        load(folder)
    assert str(folder) in str(e.value) and "sets.har" in str(e.value)


def test_missing_prm_names_it(tmp_path: Path) -> None:
    folder = _copy(tmp_path / "base")
    with pytest.raises(FileNotFoundError) as e:
        load(folder, prm="nope.prm")
    assert str(folder) in str(e.value) and "nope.prm" in str(e.value)


def test_unknown_source_lists_where_it_looked() -> None:
    with pytest.raises(FileNotFoundError) as e:
        load("no_such_base_anywhere")
    msg = str(e.value)
    assert "bundled" in msg and "data/" in msg and "folder" in msg


def test_bundled_name_wins_over_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative folder named like a bundled base does not shadow it (documented order)."""
    (tmp_path / "nus333").mkdir()  # an empty folder: loading it would fail
    monkeypatch.chdir(tmp_path)
    _same(load("nus333"), load_bundled("gtap", "nus333"))


def test_folder_without_baserate_differs(tmp_path: Path) -> None:
    """``baserate.har`` is read when the folder has it: without it the tax rates differ,
    so the equality tests above would see a load that drops it."""
    without = load(
        _copy(tmp_path / "base", ("basedata.har", "sets.har", "default.prm"))
    )
    assert vars(without.taxes) != vars(load("nus333").taxes)
