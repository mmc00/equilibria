"""El precio del ahorro como en GTAP v7: ahorro mundial = inversion mundial en valor.

``gtapv7.tab`` (E_psave): ``psave(r) = pinv(r) + sum_s [(NETINV(s)-SAVE(s))/GLOBINV]
* pinv(s)``. El ajuste comun hace que ``sum_r SAVE(r)*psave(r) = sum_r NETINV(r)*pinv(r)``.
En niveles: ``psave = chiSave * pi`` con ``chiSave = sum invwgt*pi / sum savwgt*pi``.

La forma de GAMS (``model.gms`` chiSaveeq) pone ``psave`` en el denominador, lo que da
``chiSave**2``: la mitad del ajuste en logaritmos. Contra GEMPACK eso movia la
utilidad del ahorro y con ella ``u`` y el EV (TBL62A: ``u`` de ROW -0,0650 % contra
-0,0765 % de GEMPACK).

Los pesos ``NETINV(s)/GLOBINV`` y ``SAVE(s)/GLOBINV`` de GTAP son los del dato
actualizado (un indice de Divisia que Gragg integra). Con los del ano base fijos el
ajuste se queda corto cuando el shock mueve el reparto de la inversion: TBL45A
(+10 % lambdava en SER de USA) lleva el peso de USA en NETINV de 20,7 % a 24,3 % y
psave/pinv da -0,218 % contra -0,251 %; TBL65A, de 21 % a 29 % (+1,49 % contra +1,87 %).

Oraculos: ``psave``, ``pinv`` y ``u`` de GEMPACK Gragg
(``tests/fixtures/nus333_bienestar_gempack.json.gz``).

LOCAL-only: SKIP si falta nus333.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.templates.gtap._nus333 import ROOT, nus333_params, run_burfisher

pytestmark = pytest.mark.integration

FIXTURE = ROOT / "tests" / "fixtures" / "nus333_bienestar_gempack.json.gz"


def _solve(exp: str) -> Any:
    import sys

    rb = run_burfisher()
    nus333_params(rb.EXERCISES[exp][0])  # SKIP sin el dataset
    from equilibria._local_refs import nus333_dir, path_capi_src

    sys.path.insert(0, str(path_capi_src()))
    m, _, codes = rb.solve_exercise(exp, nus333_dir())
    assert codes == {"base": 1, "check": 1, "shock": 1}, codes
    return m


@pytest.fixture(scope="module")
def solved() -> Any:
    return _solve("TBL62A")


# Arancel (chico), tecnologia y el que mas mueve el reparto de NETINV.
@pytest.fixture(scope="module", params=("TBL62A", "TBL45A", "TBL65A"))
def solved_exp(request, solved) -> tuple[str, Any]:
    if request.param == "TBL62A":
        return "TBL62A", solved
    return request.param, _solve(request.param)


def test_ajuste_comun_igual_a_gempack(solved_exp) -> None:
    """psave/pinv (el ajuste comun) de equilibria contra el de GEMPACK."""
    import gzip
    import json

    from pyomo.environ import value

    exp, m = solved_exp
    g = json.loads(gzip.decompress(FIXTURE.read_bytes()))[exp]
    for r in m.r:
        e = (value(m.psave[r, "shock"]) / value(m.psave[r, "check"])) / (
            value(m.pi[r, "shock"]) / value(m.pi[r, "check"])
        )
        gp = (1 + g["psave"][r] / 100) / (1 + g["pinv"][r] / 100)
        assert abs(e / gp - 1) < 1e-4, (r, 100 * (e - 1), 100 * (gp - 1))


def test_u_igual_a_gempack(solved) -> None:
    import gzip
    import json

    from pyomo.environ import value

    m = solved
    g = json.loads(gzip.decompress(FIXTURE.read_bytes()))["TBL62A"]["u"]
    for r in m.r:
        e = value(m.u[r, "shock"]) / value(m.u[r, "check"])
        assert abs(e / (1 + g[r] / 100) - 1) < 5e-5, (r, 100 * (e - 1), g[r])
