"""Bienestar contra GAMS: ``ev``, ``cv``, ``u``, ``ug``, ``us`` y ``uh`` en el check y el shock.

Oraculo: los niveles que resolvio GAMS en el MCP de cada ejercicio
(``tests/fixtures/nus333_bienestar_gams_levels.json.gz``, de
``scripts/gtap/gams_welfare_fixture.py``). En GAMS ``eveq``, ``cveq``, ``ueq``, ``ugeq``,
``useq`` y ``uheq`` estan en ``model gtap``; el driver de equilibria saca las cinco
primeras del sistema simultaneo (son hojas: nada depende de ellas) y las resuelve
despues de cada periodo. Las dos cosas dan el mismo punto.

Tres ejercicios de tipo distinto: arancel (TBL62A), tecnologia (TBL45A) y dotacion
(TBL63A). Criterio: el de la Tabla 3, 0,1 % relativo.

``us`` no entra: lo deflacta ``psave``, que sigue a GTAP v7 y no a GAMS (GAMS usa
``chiSave**2``). Su oraculo es GEMPACK: ``test_nus333_psave_gtap.py`` (psave y u).

LOCAL-only: SKIP si falta nus333.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.templates.gtap._nus333 import ROOT, gams_levels, nus333_params, run_burfisher

pytestmark = pytest.mark.integration

FIXTURE = ROOT / "tests" / "fixtures" / "nus333_bienestar_gams_levels.json.gz"
EXERCISES = ("TBL62A", "TBL45A", "TBL63A")
VARS = ("ev", "cv", "u", "ug", "uh")
TOL = 1e-3


@pytest.fixture(scope="module", params=EXERCISES)
def solved(request) -> tuple[str, Any]:
    exp = request.param
    rb = run_burfisher()
    nus333_params(rb.EXERCISES[exp][0])  # SKIP sin el dataset
    import sys

    from equilibria._local_refs import nus333_dir, path_capi_src

    sys.path.insert(0, str(path_capi_src()))
    m, _, codes = rb.solve_exercise(exp, nus333_dir())
    assert codes == {"base": 1, "check": 1, "shock": 1}, codes
    return exp, m


def _key(var: str, gkey: tuple) -> tuple:
    # GAMS indexa ev/cv/uh por hogar (r, 'hhd', t); equilibria por (r, t).
    return (gkey[0], gkey[-1]) if var in ("ev", "cv", "uh") else gkey


def test_el_oraculo_se_mueve_con_el_shock() -> None:
    """Si GAMS dejara el bienestar quieto, el test de abajo no probaria nada."""
    g = gams_levels(FIXTURE, "TBL62A")["ev"]
    assert abs(g[("USA", "hhd", "shock")] / g[("USA", "hhd", "check")] - 1) > 1e-3


@pytest.mark.parametrize("var", VARS)
def test_bienestar_igual_a_gams(solved, var: str) -> None:
    from pyomo.environ import value

    exp, m = solved
    comp = getattr(m, var)
    malas = []
    for gkey, g in gams_levels(FIXTURE, exp)[var].items():
        if gkey[-1] not in ("check", "shock"):
            continue
        e = value(comp[_key(var, gkey)])
        if abs(e - g) > TOL * max(abs(g), 1e-6):
            malas.append((gkey, e, g))
    assert not malas, malas
