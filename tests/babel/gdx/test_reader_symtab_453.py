import json
from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import read_gdx

HERE = Path(__file__).resolve().parent
REF = Path("/Users/marmol/proyectos2/equilibria_refs/"
           "gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx")
FIX = HERE / "fixtures_symtab_453.json"
pytestmark = pytest.mark.skipif(not REF.exists(), reason="durable ref GDX absent")

_CLASS = {"set": "Set", "parameter": "Par", "variable": "Var",
          "equation": "Equ", "alias": "Alias"}


def test_reads_all_453_symbols():
    data = read_gdx(str(REF))
    assert len(data["symbols"]) == 453


def test_no_garbage_names():
    data = read_gdx(str(REF))
    for s in data["symbols"]:
        assert s["name"].isprintable() and len(s["name"]) <= 64, s["name"]


def test_every_type_matches_gdxdump():
    truth = json.loads(FIX.read_text())["by_name"]
    data = read_gdx(str(REF))
    mismatches = []
    for s in data["symbols"]:
        want = truth.get(s["name"])
        got = _CLASS.get(s["type_name"])
        if want and got != want:
            mismatches.append((s["name"], got, want))
    assert not mismatches, f"{len(mismatches)} type mismatches: {mismatches[:5]}"
