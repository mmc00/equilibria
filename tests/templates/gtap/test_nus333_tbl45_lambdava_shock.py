"""nus333 / Burfisher Tabla 4.5A: shock de productividad del VA solo en el periodo shock.

El experimento TBL45A (RunGTAP, ``nus333/TBL45A.EXP``) es ``avaall("SER","USA") = 10``
bajo el cierre estandar de GTAPv7 (RORDELTA=1 -> ``savf_flag="capFlex"``). En niveles
eso es ``lambdava[USA,SER] = 1.10`` aplicado SOLO en el periodo shock.

Oraculo: la solucion Gragg de GEMPACK (``TBL45A.sl4``), copiada abajo a 6 decimales.
GAMS (``comp_nus333.gms`` con ``avaall.fx('USA','a_SER','shock')=0.10``, capFlex) la
reproduce a <=0.0002pp en estas celdas, asi que la tolerancia de 0.002pp deja 10x de
margen sobre la diferencia Gragg-vs-niveles y no mas.

LOCAL-only: SKIP si el dataset nus333 no esta (ver ``equilibria._local_refs``).
"""

from __future__ import annotations

import pytest
from tests.templates.gtap._nus333 import closure, nus333_params, pct
from tests.templates.gtap._routes import solve

pytestmark = pytest.mark.integration

TOL_PP = 0.002

# GEMPACK Gragg, TBL45A.sl4 — % cambio shock/check.
ORACLE = {
    "xp": {
        ("USA", "AGR"): 1.090831,
        ("USA", "MFG"): 2.965437,
        ("USA", "SER"): 9.305929,
        ("ROW", "AGR"): 0.567111,
        ("ROW", "MFG"): 0.892758,
        ("ROW", "SER"): -0.375154,
    },
    "rore": {("USA",): 1.850056, ("ROW",): 1.850056},
    "regy": {("USA",): 4.310648, ("ROW",): -1.270465},
    "pi": {("USA",): -3.052336, ("ROW",): -1.267265},
    "xiagg": {("USA",): 10.851814, ("ROW",): -2.242549},
}


@pytest.fixture(
    scope="module",
    params=[
        ("core", "apply_shock"),
        ("core", "fix_instrument_shock"),
    ],
    ids=lambda c: "-".join(c),
)
def solved(request):
    """Both ways give the same: ``apply_shock`` (GEMPACK's %) and
    ``fix_instrument_shock`` on the built model before the recipe flies (no
    tariff)."""
    route, how = request.param
    p = nus333_params()
    ac = closure()
    if how == "fix_instrument_shock":
        from equilibria.templates.gtap.instruments import fix_instrument_shock

        s = solve(
            route,
            p,
            ac,
            "ROW",
            base_calibrated=True,
            skip_base_solve=True,
            solve_check=True,
            edit_model=lambda m: fix_instrument_shock(
                m, "lambdava", ("USA", "SER"), factor=1.10
            ),
        )
        assert s.codes["shock"] == 1, s.codes
        return s.model
    s = solve(
        route,
        p,
        ac,
        "ROW",
        shock={"lambdava": {("USA", "SER"): 10.0}},
        base_calibrated=True,
        skip_base_solve=True,
        solve_check=True,
    )
    assert s.codes["shock"] == 1, s.codes
    return s.model


def test_el_shock_no_entra_al_check(solved):
    """base y check deben ser el mismo benchmark: el shock va solo en 'shock'."""
    m = solved
    for key in ORACLE["xp"]:
        assert abs(pct(m, "xp", key, num="check", den="base")) < 1e-6, key


@pytest.mark.parametrize(
    ("var", "key"), [(v, k) for v, cells in ORACLE.items() for k in cells]
)
def test_iguala_a_gempack_gragg(solved, var, key):
    m = solved
    got = pct(m, var, key)
    want = ORACLE[var][key]
    assert abs(got - want) <= TOL_PP, (
        f"{var}{key}: equilibria {got:+.6f} vs GEMPACK {want:+.6f}"
    )
