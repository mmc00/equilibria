"""Mapping: lo que declara el usuario, validado al cargar (spec 3.3, 3.7)."""

from __future__ import annotations

import copy
from typing import Any, cast

import pytest

from equilibria.compare import Mapping, MappingError
from tests.compare import _oracle as O


def _sin(path: list, key: str) -> dict:
    d = cast(Any, copy.deepcopy(O.EX2_MAP))
    node = d
    for p in path:
        node = node[p]
    del node[key]
    return d


def test_una_variable_sin_tol_es_error():
    with pytest.raises(MappingError, match="tol"):
        Mapping.from_dict(_sin(["variables", 0], "tol"))


def test_tol_sin_piso_absoluto_es_error():
    with pytest.raises(MappingError, match="abs_floor"):
        Mapping.from_dict(_sin(["variables", 0, "tol"], "abs_floor"))


def test_tol_sin_rel_es_error():
    with pytest.raises(MappingError, match="rel"):
        Mapping.from_dict(_sin(["variables", 0, "tol"], "rel"))


def test_sin_umbrales_es_error():
    with pytest.raises(MappingError, match="thresholds"):
        Mapping.from_dict(_sin([], "thresholds"))


def test_exclusion_sin_motivo_es_error():
    d = cast(Any, copy.deepcopy(O.EX3_MAP))
    d["exclude"] = [{"var": "ppriv"}]
    with pytest.raises(MappingError, match="why"):
        Mapping.from_dict(d)


def test_dims_repetidas_en_un_lado_son_error():
    d = cast(Any, copy.deepcopy(O.EX2_MAP))
    d["variables"][0]["a"]["dims"] = ["REG", "REG"]
    d["variables"][0]["b"]["dims"] = ["REG", "REG"]
    with pytest.raises(MappingError, match="dims"):
        Mapping.from_dict(d)


def test_dims_de_a_y_b_deben_ser_las_mismas():
    d = cast(Any, copy.deepcopy(O.EX1_MAP))
    d["variables"][0]["b"]["dims"] = ["ACT", "REG"]
    with pytest.raises(MappingError, match="dims"):
        Mapping.from_dict(d)


def test_carga_desde_yaml(tmp_path):
    import yaml

    f = tmp_path / "m.yaml"
    f.write_text(yaml.safe_dump(O.EX3_MAP))
    m = Mapping.from_yaml(f)
    assert [v.name for v in m.variables] == ["z"]
    assert m.excluded == {"ppriv": "indice Divisia: sin forma en niveles"}
    assert m.thresholds == (1e-3,)


def _con_exclusion(e: dict) -> dict:
    d = cast(Any, copy.deepcopy(O.EX9_MAP))
    d["exclude"] = [e]
    return d


def test_exclusion_por_celda_sin_motivo_es_error():
    with pytest.raises(MappingError, match="why"):
        Mapping.from_dict(_con_exclusion({"var": "q", "where": {"FP": ["NatRes"]}}))


def test_exclusion_por_celda_con_dim_que_la_variable_no_tiene_es_error():
    e = {"var": "q", "where": {"ACT": ["Food"]}, "why": "x"}
    with pytest.raises(MappingError, match="ACT"):
        Mapping.from_dict(_con_exclusion(e))


def test_exclusion_por_celda_de_variable_que_no_esta_en_el_mapeo_es_error():
    e = {"var": "no_mapeada", "where": {"FP": ["NatRes"]}, "why": "x"}
    with pytest.raises(MappingError, match="no_mapeada"):
        Mapping.from_dict(_con_exclusion(e))


def test_exclusion_por_celda_con_where_vacio_es_error():
    with pytest.raises(MappingError, match="where"):
        Mapping.from_dict(_con_exclusion({"var": "q", "where": {}, "why": "x"}))


def test_exclusion_por_celda_con_elementos_que_no_son_lista_es_error():
    e = {"var": "q", "where": {"FP": "NatRes"}, "why": "x"}
    with pytest.raises(MappingError, match="lista"):
        Mapping.from_dict(_con_exclusion(e))
