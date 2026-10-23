"""Homogeneidad: el numerario sube 10 % y nada real se mueve (GTAPV7 del libro).

``nus333/gtapv7.exp`` (default.prm, Johansen 1 paso): ``Shock pfactwld = uniform 10``.
En equilibria el numerario es ``pnum`` y ``eq_pnum`` lo ata a ``pwfact`` (el
``pfactwld`` de GEMPACK) en el periodo del shock; el shock es
``apply_shock(m, {"pnum": {(): 10}})``.

Oraculo (teoria, no el codigo): el modelo es homogeneo de grado 0 en precios, asi que
en el shock cada precio y cada valor es 1,10 veces el del check y cada cantidad,
indice real y utilidad es igual. Y la solucion GEMPACK GTAPV7.sl4: con un shock
homogeneo la linealizacion de Johansen es exacta, asi que coincide en niveles.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from tests.templates.gtap._nus333 import ROOT, nus333_params

pytestmark = pytest.mark.integration

TOL = 1e-6
# El residuo de Walras deberia ser 0 en los dos periodos: no escala.
SIN_ESCALA = {"walras"}
# Un par por clase, a mano: si el test clasificara al reves, fallan.
NOMINALES = ("pa", "pd", "pf", "pft", "regy", "yc", "pwfact", "pnum")
REALES = ("xa", "xd", "xf", "xp", "xw", "u")


def _cb() -> Any:
    sys.path[:0] = [str(ROOT / "scripts" / "gtap")]
    return importlib.import_module("compare_burfisher_gempack")


@pytest.fixture(scope="module")
def solved():
    nus333_params("default.prm")  # SKIP sin el dataset
    cb = _cb()
    from equilibria._local_refs import nus333_dir, path_capi_src

    sys.path.insert(0, str(path_capi_src()))
    m, p, codes = cb.rb.solve_exercise("GTAPV7", nus333_dir())
    return m, p, codes


def _clases(m) -> dict[str, set[str]]:
    """Por variable: las clases de sus celdas ('x1.10', 'x1.00' u 'otra:<r>')."""
    from pyomo.environ import Var, value

    out: dict[str, set[str]] = {}
    for var in m.component_objects(Var, active=True):
        name = var.local_name
        if name in SIN_ESCALA:
            continue
        for idx in var:
            key = idx if isinstance(idx, tuple) else (idx,)
            if key[-1] != "shock":
                continue
            chk = (*key[:-1], "check")
            ck = chk if len(chk) > 1 else chk[0]
            if ck not in var:
                continue
            a, b = value(var[ck], exception=False), value(var[idx], exception=False)
            if a is None or b is None or abs(a) < 1e-9:
                continue
            r = b / a
            if abs(r - 1.1) <= TOL * 1.1:
                c = "x1.10"
            elif abs(r - 1.0) <= TOL:
                c = "x1.00"
            else:
                c = f"otra:{r:.9g}"
            out.setdefault(name, set()).add(c)
    return out


def test_resuelve(solved):
    _, _, codes = solved
    assert codes == {"base": 1, "check": 1, "shock": 1}


def test_cada_celda_escala_o_queda_igual(solved):
    clases = _clases(solved[0])
    malas = {v: c for v, c in clases.items() if any(x.startswith("otra") for x in c)}
    assert not malas, malas


def test_cada_variable_es_toda_nominal_o_toda_real(solved):
    clases = _clases(solved[0])
    mezcladas = {v: c for v, c in clases.items() if len(c) > 1}
    assert not mezcladas, mezcladas


def test_precios_y_valores_escalan_cantidades_no(solved):
    clases = _clases(solved[0])
    for v in NOMINALES:
        assert clases[v] == {"x1.10"}, v
    for v in REALES:
        assert clases[v] == {"x1.00"}, v


def test_coincide_con_gempack(solved):
    from equilibria.blocks.gtap._derived_params import xscale_data

    m, p, _ = solved
    cb = _cb()
    r = cb.run("GTAPV7", cb.cg.model_levels(m), dict(xscale_data(p, p.sets)))
    cells = [c for c in r.cells if c.period == "post"]
    assert len(cells) == 207
    worst = max(cells, key=lambda c: c.d)
    assert worst.d < 1e-4, worst
    assert not r.unmatched
