"""Fuentes -> Solution: GDX de GAMS (contra gdxdump), Pyomo y dict."""

from __future__ import annotations

from typing import Any, cast

import pytest

from equilibria.compare import from_dict, from_gdx, from_pyomo
from tests.compare import _oracle as O

gdx = pytest.mark.skipif(not O.GTAP3X3_GDX.exists(), reason="falta el GDX de gtap7_3x3")


@gdx
def test_from_gdx_lee_los_niveles_que_da_gdxdump():
    s = from_gdx(O.GTAP3X3_GDX, period_dim=-1, variables=["xp", "pft"])
    xp = s.levels["xp"]
    assert len(xp) == len(O.GDXDUMP_XP)
    for (r, a, t), want in O.GDXDUMP_XP.items():
        assert xp[(t, (r, a))] == pytest.approx(want, rel=1e-13), (r, a, t)
    for (r, f, t), want in O.GDXDUMP_PFT_SOME.items():
        assert s.levels["pft"][(t, (r, f))] == pytest.approx(want, rel=1e-13)
    assert s.ndim == {"xp": 2, "pft": 2}
    assert s.periods() == ["base", "check", "shock"]


@gdx
def test_from_gdx_variable_inexistente_es_error():
    with pytest.raises(KeyError, match="no_existe"):
        from_gdx(O.GTAP3X3_GDX, period_dim=-1, variables=["no_existe"])


def test_hay_que_declarar_el_periodo():
    with pytest.raises(ValueError, match="period"):
        from_dict({"x": {("a", "t"): 1.0}})
    with pytest.raises(ValueError, match="period"):
        from_dict({"x": {("a", "t"): 1.0}}, period_dim=-1, period="t")


def test_from_dict_con_periodo_fijo_y_periodo_en_otra_posicion():
    s = from_dict({"x": {"USA": 1.0}, "y": {("t1", "USA"): 2.0}}, period="post")
    assert s.levels["x"] == {("post", ("USA",)): 1.0}
    s = from_dict({"y": {("t1", "USA", "Food"): 2.0}}, period_dim=0)
    assert s.levels["y"] == {("t1", ("USA", "Food")): 2.0}


def test_from_pyomo_lee_valores_y_omite_los_que_no_tienen():
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=["check", "shock"])
    m.R = pyo.Set(initialize=["USA", "ROW"])
    m.x = pyo.Var(m.R, m.T)
    m.s = pyo.Var(m.T)
    x = cast(Any, m.x)
    sc = cast(Any, m.s)
    x["USA", "check"].value = 1.5
    x["ROW", "shock"].value = 2.5
    sc["shock"].value = 7.0
    s = from_pyomo(m, period_dim=-1)
    assert s.levels["x"] == {("check", ("USA",)): 1.5, ("shock", ("ROW",)): 2.5}
    assert s.levels["s"] == {("shock", ()): 7.0}
    assert s.ndim == {"x": 1, "s": 0}


@gdx
def test_from_gdx_declara_las_variables_sin_dimension_de_periodo():
    """walras es escalar en el GDX: leyendo todo, queda declarada como omitida;
    pedida a mano, es error."""
    s = from_gdx(O.GTAP3X3_GDX, period_dim=-1)
    assert "walras" in s.skipped and "walras" not in s.levels
    with pytest.raises(ValueError, match="walras"):
        from_gdx(O.GTAP3X3_GDX, period_dim=-1, variables=["walras"])
