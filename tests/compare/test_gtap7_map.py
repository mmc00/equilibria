"""El mapeo GTAP que viene con equilibria (maps/gtap7.yaml, #132): equilibria contra
GEMPACK en niveles, pre y post, con scripts/gtap/compare_gempack_levels.py.
Oraculo: ejemplo 15 de tests/compare/_oracle.py (calculado sin equilibria.compare)."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

from equilibria.compare import (
    CompareError,
    Mapping,
    MappingError,
    compare,
    from_dict,
)
from tests.compare import _oracle as O

SCRIPT = O.ROOT / "scripts" / "gtap" / "compare_gempack_levels.py"
BASEDATA = O.ROOT / "datasets" / "gtap7_3x3" / "basedata.har"
UPDATED = O.ROOT / "tests" / "fixtures" / "gtap7_gempack" / "updated_gtap7_3x3_tm10.har"

pytestmark = pytest.mark.skipif(
    not (O.EX15_HAR.exists() and BASEDATA.exists()), reason="falta el 3x3 de GEMPACK"
)


def _script() -> Any:
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("compare_gempack_levels", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_equilibria_contra_gempack_3x3_en_niveles() -> None:
    r = _script().run(O.EX15_EQ, O.EX15_HAR, BASEDATA, UPDATED)
    assert r.periods_compared == ["pre", "post"]
    assert {k: s["cells"] for k, s in r.summary.items()} == O.EX15_CELLS
    o = r.overall
    assert o["cells"] == O.EX15_TOTAL
    for thr, n in O.EX15_WITHIN.items():
        assert o["pct_within"][thr] == pytest.approx(100.0 * n / O.EX15_TOTAL)
    assert o["max"] == pytest.approx(O.EX15_MAX, rel=1e-4)
    assert (o["max_variable"], o["max_cell"]) == O.EX15_MAX_CELL
    assert r.unmatched == []


def test_cada_exclusion_sale_en_el_reporte_con_su_motivo() -> None:
    r = _script().run(O.EX15_EQ, O.EX15_HAR, BASEDATA, UPDATED)
    assert [(e.variable, e.where, e.cells) for e in r.exclusions] == [
        ("pe", {"ENDW": ["NatRes"]}, O.EX15_NATRES_EXCLUDED)
    ]
    assert "ENDF" in r.exclusions[0].why
    assert {x.index[-1] for x in r.excluded_cells} == {"NatRes"}
    assert r.excluded == {"ppriv": "indice Divisia: sin forma en niveles"}


def test_sin_las_variables_armadas_es_error_no_exclusion() -> None:
    """El modelo crudo no trae xds_net ni vdfb: pedirle el mapeo es error."""
    mod = _script()
    levels, xscale = mod.load_eq_json(O.EX15_EQ)
    raw = from_dict(levels, period_dim=-1, params={"xscale": xscale})
    with pytest.raises(CompareError, match="xds_net|vdfb"):
        compare(
            raw,
            mod.gempack_side(O.EX15_HAR, BASEDATA, UPDATED),
            Mapping.builtin("gtap7"),
        )


def test_un_mapeo_que_no_existe_es_error() -> None:
    with pytest.raises(MappingError, match="gtap7"):
        Mapping.builtin("gtap99")
