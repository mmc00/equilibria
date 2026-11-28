"""The GDX reader returns every element of every set (record count and gdxdump).

Measured on 2026-11-27 (feat/manage-template, 936ef76f): ``read_set_elements``
read 810 sets wrong in 106 of the repository's 171 GDX. A set element with an
explanatory text stores the text's number as a value token (0x06 for text 1, or
0x0A and an 8-byte double), not the one-byte 0x05 of an element without text;
the decoder skipped one byte and lost the rest of the set (MANAGE ``g``: 1 of 16
elements; every GTAP ``out.gdx``: 8 sets). Sets written by equilibria's own
writer may repeat an element (``v7_consolidated.gdx``, ``MCOM``: 65 records, 3
distinct), and the reader keeps them as written.
"""

from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path

import pytest

from equilibria.babel.gdx.gdxdump import locate_gdxdump
from equilibria.babel.gdx.reader import get_sets, read_gdx, read_set_elements
from equilibria.sources import dataset_dir, reference

ROOT = Path(__file__).resolve().parents[3]


def _bau() -> Path:
    return (
        reference(
            "gha_gtap9", "gams", form="mcp", task="baseline", year=2015,
            mcp_solver="nlpec-ipopt",
        )
        / "GHA_BaU.gdx"
    )  # fmt: skip


CASES = {
    "manage_bau": _bau,
    "manage_bridge": lambda: dataset_dir("gha_gtap9") / "BaseBridge_GHA_GTAP_9.gdx",
    "gtap7_3x3_out": lambda: (
        ROOT / "data" / "gdx_tests" / "gtap7_3x3_out_gtap_shock_ifsub0.gdx"
    ),
    "gtap7_3x4_consolidated": lambda: dataset_dir("gtap7_3x4") / "v7_consolidated.gdx",
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_set_has_its_record_count(case: str) -> None:
    data = read_gdx(CASES[case]())
    wrong = {
        s["name"]: (len(read_set_elements(data, s["name"])), s["records"])
        for s in get_sets(data)
        if s["records"] and s["dimension"]
        and len(read_set_elements(data, s["name"])) != s["records"]
    }  # fmt: skip
    assert not wrong, f"(read, records) of the sets read wrong: {wrong}"


def test_texted_sets_of_manage() -> None:
    data = read_gdx(_bau())
    g = read_set_elements(data, "g")
    assert len(g) == 16
    assert ("FacLand",) in g and ("Paf",) in g
    assert len(read_set_elements(data, "l")) == 19


def test_repeated_elements_are_kept() -> None:
    data = read_gdx(CASES["gtap7_3x4_consolidated"]())
    mcom = read_set_elements(data, "MCOM")
    assert len(mcom) == 65
    assert len(set(mcom)) == 3


def _gdxdump_set(gdx: Path, name: str, dim: int) -> list[tuple[str, ...]]:
    exe = locate_gdxdump()
    assert exe is not None
    out = subprocess.run(
        [exe, str(gdx), f"symb={name}", "format=csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = list(csv.reader(io.StringIO(out)))
    return [tuple(r[:dim]) for r in rows[1:]]


@pytest.mark.needs_gdxdump
@pytest.mark.parametrize("case", ["manage_bau", "manage_bridge"])
def test_every_set_matches_gdxdump(case: str) -> None:
    gdx = CASES[case]()
    data = read_gdx(gdx)
    for s in get_sets(data):
        if not s["records"] or not s["dimension"]:
            continue
        ours = read_set_elements(data, s["name"])
        assert ours == _gdxdump_set(gdx, s["name"], s["dimension"]), s["name"]
