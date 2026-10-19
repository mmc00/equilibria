"""compare(a, b, mapping): alinear, medir y reportar, contra los ejemplos a mano."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from equilibria.compare import (
    CompareError,
    Mapping,
    MappingError,
    compare,
    from_dict,
)
from tests.compare import _oracle as O


def _d_by_cell(report, variable):
    return {(c.period, c.index): c.d for c in report.cells if c.variable == variable}


def _run(a, b, mapping, **kw):
    return compare(
        from_dict(a, period_dim=-1),
        from_dict(b, period_dim=-1),
        Mapping.from_dict(mapping),
        **kw,
    )


def test_dos_conjuntos_con_elementos_de_igual_nombre_no_cruzan_celdas():
    r = _run(O.EX1_A, O.EX1_B, O.EX1_MAP)
    got = _d_by_cell(r, "x")
    assert set(got) == set(O.EX1_D)
    for cell, want in O.EX1_D.items():
        assert got[cell] == pytest.approx(want, rel=1e-9, abs=1e-15), cell
    assert r.unmatched == []


def test_piso_absoluto_umbrales_mediana_y_maximo():
    r = _run(O.EX2_A, O.EX2_B, O.EX2_MAP)
    got = _d_by_cell(r, "y")
    for cell, want in O.EX2_D_REF_B.items():
        assert got[cell] == pytest.approx(want, rel=1e-12, abs=1e-15), cell
    s = r.summary["y"]
    assert s["cells"] == 3
    for thr, pct in O.EX2_PCT_WITHIN_REF_B.items():
        assert s["pct_within"][thr] == pytest.approx(pct)
    assert s["pass_pct"] == pytest.approx(O.EX2_PASS_PCT_REF_B)
    assert s["median"] == pytest.approx(O.EX2_MEDIAN_REF_B)
    assert s["max"] == pytest.approx(O.EX2_MAX_REF_B)
    assert s["max_cell"] == O.EX2_MAX_CELL_REF_B
    # el global de una sola variable es el de esa variable
    assert r.overall["pct_within"][0.01] == pytest.approx(O.EX2_PCT_WITHIN_REF_B[0.01])
    assert r.overall["max_variable"] == "y"


def test_la_referencia_a_cambia_el_denominador():
    r = _run(O.EX2_A, O.EX2_B, O.EX2_MAP, reference="a")
    got = _d_by_cell(r, "y")
    for cell, want in O.EX2_D_REF_A.items():
        assert got[cell] == pytest.approx(want, rel=1e-12, abs=1e-15), cell


def test_referencia_invalida_es_error():
    with pytest.raises(CompareError, match="reference"):
        _run(O.EX2_A, O.EX2_B, O.EX2_MAP, reference="c")


def test_periodos_renombrados_y_celdas_sin_pareja():
    r = _run(O.EX3_A, O.EX3_B, O.EX3_MAP)
    assert r.summary["z"]["cells"] == O.EX3_MATCHED
    faltan_en_b = [
        (u.variable, u.period, u.index) for u in r.unmatched if u.missing_in == "b"
    ]
    faltan_en_a = [
        (u.variable, u.period, u.index) for u in r.unmatched if u.missing_in == "a"
    ]
    assert faltan_en_b == O.EX3_MISSING_IN_B
    assert faltan_en_a == O.EX3_MISSING_IN_A
    assert r.summary["z"]["missing_in_a"] == 1
    assert r.summary["z"]["missing_in_b"] == 1
    assert r.periods_only_in_a == O.EX3_PERIODS_ONLY_IN_A
    assert r.periods_only_in_b == O.EX3_PERIODS_ONLY_IN_B
    assert r.excluded == {"ppriv": "indice Divisia: sin forma en niveles"}


def test_una_celda_sin_pareja_nunca_cuenta_como_que_pasa():
    """El periodo "pre" (= check) existe en B (por otra variable), pero a z le faltan
    todas sus celdas ahi: son "sin pareja", no celdas que pasan."""
    b = {"z": {("USA", "post"): 5.0}, "w": {("USA", "pre"): 1.0}}
    r = _run(O.EX3_A, b, O.EX3_MAP)
    assert r.summary["z"]["cells"] == 1  # solo shock/USA
    assert r.summary["z"]["missing_in_b"] == 2  # check/USA y check/ROW


def test_el_pct_que_cuenta_las_sin_pareja_esta_en_el_reporte():
    """Ejemplo 3: 2 parejas que pasan + 2 sin pareja -> 50 % (oraculo 11)."""
    r = _run(O.EX3_A, O.EX3_B, O.EX3_MAP)
    assert r.overall["pass_pct_all"] == pytest.approx(O.EX11_EX3_PASS_PCT_ALL)
    assert r.summary["z"]["pass_pct_all"] == pytest.approx(O.EX11_EX3_PASS_PCT_ALL)


def test_un_periodo_que_falta_entero_en_un_lado_no_se_compara():
    """Q18: solo se comparan los periodos presentes en los dos lados; el resto se
    declara como periodo de un solo lado, no como miles de celdas sin pareja."""
    b = {"z": {("USA", "post"): 5.0}}  # B no tiene "pre" (= check) en absoluto
    r = _run(O.EX3_A, b, O.EX3_MAP)
    assert r.periods_compared == ["shock"]
    assert r.periods_only_in_a == ["base", "check"]
    assert r.summary["z"]["missing_in_b"] == 0


def test_variable_del_mapeo_que_no_existe_en_la_fuente_es_error():
    m = cast(Any, copy.deepcopy(O.EX2_MAP))
    m["variables"][0]["b"]["var"] = "no_existe"
    with pytest.raises(MappingError, match="no_existe"):
        _run(O.EX2_A, O.EX2_B, m)


def test_dims_que_no_coinciden_con_la_fuente_es_error():
    m = cast(Any, copy.deepcopy(O.EX2_MAP))
    m["variables"][0]["a"]["dims"] = ["REG", "ACT"]
    m["variables"][0]["b"]["dims"] = ["REG", "ACT"]
    with pytest.raises(MappingError, match="dims"):
        _run(O.EX2_A, O.EX2_B, m)


def test_reporte_a_json():
    r = _run(O.EX3_A, O.EX3_B, O.EX3_MAP)
    j = json.loads(r.to_json())
    assert j["reference"] == "b"
    assert j["summary"]["z"]["cells"] == O.EX3_MATCHED
    assert j["excluded"] == [
        {"var": "ppriv", "why": "indice Divisia: sin forma en niveles"}
    ]
    assert {
        "variable": "z",
        "period": "check",
        "index": ["ROW"],
        "missing_in": "b",
    } in j["unmatched"]
    assert j["periods_only_in_a"] == ["base"]
    assert len(j["cells"]) == O.EX3_MATCHED


def _csv_rows(text):
    import csv
    import io

    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]

    def num(x):
        return None if x == "" else float(x)

    return header, [
        (v, p, i, num(a), num(b), num(d), s) for v, p, i, a, b, d, s in body
    ]


def _assert_rows(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g[:3] == w[:3] and g[6] == w[6], (g, w)
        for x, y in zip(g[3:6], w[3:6], strict=True):
            assert (x is None and y is None) or x == pytest.approx(y, rel=1e-12), (g, w)


def test_reporte_a_csv_una_fila_por_celda_con_pass_y_fail():
    header, rows = _csv_rows(_run(O.EX2_A, O.EX2_B, O.EX2_MAP).to_csv())
    assert header == O.CSV_HEADER
    _assert_rows(rows, O.EX2_CSV_ROWS)


def test_reporte_a_csv_incluye_las_celdas_sin_pareja(tmp_path):
    f = tmp_path / "r.csv"
    text = _run(O.EX3_A, O.EX3_B, O.EX3_MAP).to_csv(f)
    assert f.read_text() == text
    header, rows = _csv_rows(text)
    assert header == O.CSV_HEADER
    _assert_rows(rows, O.EX3_CSV_ROWS)


def test_reporte_a_csv_une_los_indices_con_punto():
    _, rows = _csv_rows(_run(O.EX1_A, O.EX1_B, O.EX1_MAP).to_csv())
    assert [r[2] for r in rows] == O.EX1_CSV_INDEX


def _pairs(report, variable):
    return {
        (c.period, c.index): (c.a, c.b, c.d)
        for c in report.cells
        if c.variable == variable
    }


def _unmatched(report, side):
    return [
        (u.variable, u.period, u.index)
        for u in report.unmatched
        if u.missing_in == side
    ]


def test_celda_ausente_de_un_gdx_vale_cero_si_la_variable_tiene_ese_periodo():
    """GAMS no escribe registros con nivel 0: en una fuente GDX (absent_is_zero) la
    celda que el otro lado tiene vale 0. Si la variable no tiene registros en ese
    periodo, o si la fuente que no la tiene es un dict, sigue sin pareja."""
    from equilibria.compare import Solution

    a = Solution(O.EX6_A_LEVELS, O.EX6_A_NDIM, "gdx a mano", absent_is_zero=True)
    r = compare(a, from_dict(O.EX6_B, period_dim=-1), Mapping.from_dict(O.EX6_MAP))
    got = _pairs(r, "w")
    assert set(got) == set(O.EX6_PAIRS)
    for cell, (wa, wb, wd) in O.EX6_PAIRS.items():
        assert got[cell] == pytest.approx((wa, wb, wd), rel=1e-12, abs=1e-15), cell
    assert _unmatched(r, "b") == O.EX6_MISSING_IN_B
    assert _unmatched(r, "a") == O.EX6_MISSING_IN_A


def test_un_dict_no_rellena_ceros():
    """Una fuente dict no es GDX: lo que le falta queda sin pareja, nunca 0."""
    a = from_dict(
        {"w": {("USA", "check"): 2.0}, "v": {("USA", "base"): 1.0}}, period_dim=-1
    )
    r = compare(a, from_dict(O.EX6_B, period_dim=-1), Mapping.from_dict(O.EX6_MAP))
    assert ("w", "check", ("ROW",)) in _unmatched(r, "a")
    assert all(c.index != ("ROW",) for c in r.cells)


def test_los_errores_del_modulo_son_una_sola_familia():
    assert issubclass(MappingError, CompareError)


def test_celdas_excluidas_no_se_comparan_ni_quedan_sin_pareja():
    """Ejemplo 9: las celdas excluidas salen de la comparacion (aunque fallarian o
    pasarian) y de "sin pareja"; nunca cuentan como que pasan."""
    r = _run(O.EX9_A, O.EX9_B, O.EX9_MAP)
    s = r.summary["q"]
    assert s["cells"] == O.EX9_MATCHED
    assert s["pass_pct"] == pytest.approx(O.EX9_PASS_PCT)
    assert s["excluded"] == O.EX9_EXCLUDED_TOTAL
    assert r.overall["excluded"] == O.EX9_EXCLUDED_TOTAL
    assert r.unmatched == []
    assert [(e.why, e.cells) for e in r.exclusions] == O.EX9_EXCLUSION_COUNTS
    assert len(r.excluded_cells) == O.EX9_EXCLUDED_TOTAL


def test_exclusiones_por_celda_en_el_json():
    j = json.loads(_run(O.EX9_A, O.EX9_B, O.EX9_MAP).to_json())
    assert [(e["why"], e["cells"]) for e in j["exclusions"]] == O.EX9_EXCLUSION_COUNTS
    assert j["exclusions"][0]["where"] == {"FP": ["NatRes"]}
    assert len(j["excluded_cells"]) == O.EX9_EXCLUDED_TOTAL


def test_reporte_a_csv_marca_las_celdas_excluidas():
    _, rows = _csv_rows(_run(O.EX9_A, O.EX9_B, O.EX9_MAP).to_csv())
    excl = {r for r in rows if r[6] == "excluded"}
    assert excl == O.EX9_CSV_EXCLUDED
    assert len(rows) == O.EX9_MATCHED + O.EX9_EXCLUDED_TOTAL
