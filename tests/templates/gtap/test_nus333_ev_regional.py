"""EV del hogar regional (la definicion de GTAP) contra GEMPACK.

Oraculo: ``EV`` y ``WEV`` de las corridas Gragg de GEMPACK
(``tests/fixtures/nus333_bienestar_gempack.json.gz``, de
``scripts/gtap/welfare_gempack_fixture.py``). En ``gtapv7.tab`` (seccion 11) el EV es
el ingreso que, con los precios de antes del shock fijos, alcanza la utilidad ``u``
de despues, menos el ingreso de antes: ``EV(r) = INCOMEEV(r)/100 * yev(r)``.

Criterio: el de los niveles del paper, 0,1 %, sobre el ingreso equivalente; o sea
``|EV - EV_gempack| <= 0,1 % del ingreso del check``. ``regy`` va en millones de
millones de dolares (inScale = 1e-6) y el EV de GEMPACK en millones.

LOCAL-only: SKIP si falta nus333.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.templates.gtap._nus333 import ROOT, nus333_params, run_burfisher

pytestmark = pytest.mark.integration

FIXTURE = ROOT / "tests" / "fixtures" / "nus333_bienestar_gempack.json.gz"
EXERCISES = ("TBL62A", "TBL45A", "TBL63A", "ME7A")
IN_SCALE = 1e-6
TOL = 1e-3


def _gempack(exp: str) -> dict:
    import gzip
    import json

    return json.loads(gzip.decompress(FIXTURE.read_bytes()))[exp]


@pytest.fixture(scope="module", params=EXERCISES)
def solved(request) -> tuple[str, Any]:
    import sys

    exp = request.param
    rb = run_burfisher()
    nus333_params(rb.EXERCISES[exp][0])  # SKIP sin el dataset
    from equilibria._local_refs import nus333_dir, path_capi_src

    sys.path.insert(0, str(path_capi_src()))
    m, _, codes = rb.solve_exercise(exp, nus333_dir())
    assert codes == {"base": 1, "check": 1, "shock": 1}, codes
    return exp, m


def test_el_oraculo_no_es_trivial() -> None:
    """Un EV de GEMPACK casi nulo haria pasar cualquier cosa."""
    ev = _gempack("TBL62A")["EV"]
    assert abs(ev["USA"]) > 1e4 and abs(ev["ROW"]) > 1e4


def test_ev_regional_igual_a_gempack(solved) -> None:
    from pyomo.environ import value

    from equilibria.templates.gtap.welfare_regional import regional_ev

    exp, m = solved
    ev = regional_ev(m)
    g = _gempack(exp)
    malas = []
    for r in m.r:
        income = value(m.regy[r, "check"]) / IN_SCALE
        e = ev[r] / IN_SCALE
        if abs(e - g["EV"][r]) > TOL * income:
            malas.append((r, e, g["EV"][r], income))
    assert not malas, malas
    wev = sum(ev.values()) / IN_SCALE
    world = sum(value(m.regy[r, "check"]) for r in m.r) / IN_SCALE
    assert abs(wev - g["WEV"]) <= TOL * world, (wev, g["WEV"])


def test_ev_regional_no_cambia_el_modelo(solved) -> None:
    """El sistema sombra se resuelve sobre el modelo y lo deja como estaba."""
    from pyomo.environ import Var, value

    from equilibria.templates.gtap.welfare_regional import regional_ev

    _, m = solved
    antes = {
        (v.name, v.fixed): value(v, exception=False)
        for v in m.component_data_objects(Var)
    }
    regional_ev(m)
    despues = {
        (v.name, v.fixed): value(v, exception=False)
        for v in m.component_data_objects(Var)
    }
    assert antes == despues
