"""load_bundled reads nus333/9x10 from data/ in a checkout and from the wheel copy."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria import datasets
from equilibria.datasets import dataset_path, list_bundled, load_bundled
from equilibria.sources import dataset_dir

# Read with load_bundled on main (4e9ea88), before the files moved.
PINNED = {
    "nus333": (
        (("ROW", "CAPITAL", "AGR"), 0.443187),
        (("USA", "SER", "USA"), 1.4999999621068128e-11),
        (("ROW", "AGR"), 0.24807396531105042),
    ),
    "9x10": (
        (("EU_28", "Capital", "a_Extraction"), 0.080848078125),
        (("SouthAsia", "c_Util_Cons", "SouthAsia"), 1.0602821350097655e-05),
        (("EU_28", "a_Extraction"), 0.20000000298023224),
    ),
}


def test_list_bundled_gtap() -> None:
    assert list_bundled("gtap") == ["9x10", "nus333"]


@pytest.mark.parametrize("name", ["nus333", "9x10"])
def test_checkout_reads_data_folder(name: str) -> None:
    assert dataset_path("gtap", name) == dataset_dir(name)


def test_bundled_copy_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets, "_PACKAGE_DIR", tmp_path)
    (tmp_path / "_bundled" / "gtap" / "nus333").mkdir(parents=True)
    assert dataset_path("gtap", "nus333") == tmp_path / "_bundled" / "gtap" / "nus333"


@pytest.mark.parametrize("name", ["nus333", "9x10"])
def test_load_bundled_same_numbers(name: str) -> None:
    p = load_bundled("gtap", name)
    (k1, vfm), (k2, vxsb), (k3, esubva) = PINNED[name]
    assert p.benchmark.vfm[k1] == vfm
    assert p.benchmark.vxsb[k2] == vxsb
    assert p.elasticities.esubva[k3] == esubva
