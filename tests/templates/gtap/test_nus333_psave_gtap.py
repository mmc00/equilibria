"""El precio del ahorro como en GTAP v7: ahorro mundial = inversion mundial en valor.

``gtapv7.tab`` (E_psave): ``psave(r) = pinv(r) + sum_s [(NETINV(s)-SAVE(s))/GLOBINV]
* pinv(s)``. El ajuste comun hace que ``sum_r SAVE(r)*psave(r) = sum_r NETINV(r)*pinv(r)``.
En niveles: ``psave = chiSave * pi`` con ``chiSave = sum invwgt*pi / sum savwgt*pi``.

La forma de GAMS (``model.gms`` chiSaveeq) pone ``psave`` en el denominador, lo que da
``chiSave**2``: la mitad del ajuste en logaritmos. Contra GEMPACK eso movia la
utilidad del ahorro y con ella ``u`` y el EV (TBL62A: ``u`` de ROW -0,0650 % contra
-0,0765 % de GEMPACK).

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


@pytest.fixture(scope="module")
def solved() -> Any:
    import sys

    rb = run_burfisher()
    nus333_params(rb.EXERCISES["TBL62A"][0])  # SKIP sin el dataset
    from equilibria._local_refs import nus333_dir, path_capi_src

    sys.path.insert(0, str(path_capi_src()))
    m, _, codes = rb.solve_exercise("TBL62A", nus333_dir())
    assert codes == {"base": 1, "check": 1, "shock": 1}, codes
    return m


def test_ajuste_comun_igual_a_gempack(solved) -> None:
    """psave/pinv (el ajuste comun) de equilibria contra el de GEMPACK."""
    import gzip
    import json

    from pyomo.environ import value

    m = solved
    g = json.loads(gzip.decompress(FIXTURE.read_bytes()))["TBL62A"]
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
