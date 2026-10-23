"""Los ejercicios de Burfisher: equilibria contra GEMPACK en niveles, pre y post, con
scripts/gtap/compare_burfisher_gempack.py. Oraculo: ejemplo 18 de
tests/compare/_oracle.py (el .sl4 leido a mano, sin equilibria.compare)."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

from tests.compare import _oracle as O

SCRIPT = O.ROOT / "scripts" / "gtap" / "compare_burfisher_gempack.py"
EXERCISES = sorted(O.EX18_EQ)


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("compare_burfisher_gempack", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("exp", EXERCISES)
def test_equilibria_contra_gempack_en_un_ejercicio(exp: str) -> None:
    mod = _script()
    r = mod.run(exp, *mod.cg.load_eq_json(O.EX18_EQ[exp]), O.EX18_FIX)
    assert {k: s["cells"] for k, s in r.summary.items()} == O.EX18_CELLS
    o = r.overall
    assert o["cells"] == O.EX18_TOTAL
    for thr, n in O.EX18_WITHIN[exp].items():
        assert o["pct_within"][thr] == pytest.approx(100.0 * n / O.EX18_TOTAL)
    assert o["max"] == pytest.approx(O.EX18_MAX[exp], rel=1e-4)
    assert (o["max_variable"], o["max_cell"]) == O.EX18_MAX_CELL[exp]
    assert not r.unmatched


@pytest.mark.parametrize("exp", EXERCISES)
def test_el_metodo_sale_del_log_de_gempack(exp: str) -> None:
    assert _script().method(exp, O.EX18_FIX) == O.EX18_METHOD[exp]


def test_las_variables_sin_nivel_se_excluyen_con_motivo() -> None:
    mod = _script()
    r = mod.run("tbl46a", *mod.cg.load_eq_json(O.EX18_EQ["tbl46a"]), O.EX18_FIX)
    for var in O.EX18_NO_LEVEL:
        assert "sin nivel" in r.excluded[var], var
        assert var not in r.summary


def test_un_log_sin_metodo_es_un_error(tmp_path) -> None:
    import gzip

    with gzip.open(tmp_path / "x.log.gz", "wt") as fh:
        fh.write("Steps = 2 4 6;\n")
    with pytest.raises(ValueError, match="no dice el metodo"):
        _script().method("X", tmp_path)


def test_hay_solucion_gempack_para_los_37_ejercicios() -> None:
    mod = _script()
    con = {e for e in mod.rb.EXERCISES if (O.EX18_FIX / f"{e.lower()}.sl4.gz").exists()}
    assert set(mod.rb.EXERCISES) - con == {"TBL53"}
    johansen = {e for e in con if mod.method(e, O.EX18_FIX) == "Johansen"}
    assert johansen == {
        "TBL45A", "TBL45B", "TBL45C", "TBL64", "ME4A", "ME4B", "GTAPV7"
    }  # fmt: skip


def test_con_dos_lineas_method_vale_la_ultima(tmp_path) -> None:
    """GEMPACK toma la ultima ``Method`` del command file (tools/gempack-oracle)."""
    import gzip

    with gzip.open(tmp_path / "x.log.gz", "wt") as fh:
        fh.write(" Method = Gragg;\n Steps = 2 4 6;\n Method = Johansen;\n")
    assert _script().method("X", tmp_path) == "Johansen"
