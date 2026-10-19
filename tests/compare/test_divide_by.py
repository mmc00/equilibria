"""divide_by: deshacer las escalas de una fuente (decision 2026-10-18). Oraculo: ejemplo 13."""

from __future__ import annotations

import copy
from typing import Any, cast

import pytest

from equilibria.compare import CompareError, Mapping, MappingError, compare, from_dict
from tests.compare import _oracle as O


def _a(params=O.EX13_A_PARAMS):
    return from_dict(O.EX13_A, period_dim=-1, params=params)


def _b():
    return from_dict(O.EX13_B, period_dim=-1)


def test_deshace_las_escalas_y_compara_en_las_unidades_de_b():
    r = compare(_a(), _b(), Mapping.from_dict(O.EX13_MAP))
    a_levels = {(c.period, c.index): c.a for c in r.cells}
    assert a_levels == pytest.approx(O.EX13_A_UNSCALED, rel=1e-12)
    food = next(c for c in r.cells if c.index == ("USA", "Food"))
    assert food.d == pytest.approx(O.EX13_D_FOOD, rel=1e-9)
    assert r.summary["output"]["pass_pct"] == O.EX13_PASS_PCT


def test_el_reporte_declara_las_escalas_deshechas():
    r = compare(_a(), _b(), Mapping.from_dict(O.EX13_MAP))
    scales = r.summary["output"]["divide_by"]
    assert scales["b"] == []
    assert any("xscale" in s for s in scales["a"])
    assert any(O.EX13_INSCALE_WHY in s for s in scales["a"])


def test_sin_el_parametro_es_error_nunca_vale_1():
    with pytest.raises(MappingError, match="xscale"):
        compare(_a(params={}), _b(), Mapping.from_dict(O.EX13_MAP))


def test_sin_una_celda_del_parametro_es_error():
    params = {"xscale": {("USA", "a_Food"): 1.0}}
    with pytest.raises(MappingError, match="a_Mnfcs"):
        compare(_a(params=params), _b(), Mapping.from_dict(O.EX13_MAP))


def test_una_constante_sin_why_es_error_al_cargar():
    m = cast(Any, copy.deepcopy(O.EX13_MAP))
    m["variables"][0]["a"]["divide_by"][1].pop("why")
    with pytest.raises(MappingError, match="why"):
        Mapping.from_dict(m)


def test_las_dims_del_parametro_deben_ser_dims_de_la_variable():
    m = cast(Any, copy.deepcopy(O.EX13_MAP))
    m["variables"][0]["a"]["divide_by"][0]["dims"] = ["REG", "COMM"]
    with pytest.raises(MappingError, match="COMM"):
        Mapping.from_dict(m)


def test_mapping_lista_los_parametros_que_pide_cada_lado():
    m = Mapping.from_dict(O.EX13_MAP)
    assert m.params("a") == ["xscale"]
    assert m.params("b") == []


def test_un_parametro_con_celdas_de_distinto_largo_es_error():
    with pytest.raises(CompareError, match="xscale"):
        from_dict(
            O.EX13_A,
            period_dim=-1,
            params={"xscale": {("USA", "a_Food"): 1.0, ("USA",): 0.1}},
        )


def test_una_constante_con_claves_desconocidas_es_error():
    m = cast(Any, copy.deepcopy(O.EX13_MAP))
    m["variables"][0]["a"]["divide_by"][1]["dims"] = ["REG"]
    with pytest.raises(MappingError, match="dims"):
        Mapping.from_dict(m)
