"""nus333 / Burfisher Tabla 4.5B: el shock de TBL45A con ``3x3ces.prm``.

``TBL45B.EXP``: ``avaall("SER","USA") = 10`` con ``3x3ces.prm`` (CDE, SUBPAR=0,5),
cierre estandar GTAPv7 -> ``savf_flag="capFlex"``. Por ``apply_shock`` (el % de
GEMPACK) y por ``fix_instrument_shock`` directo: las dos vias tienen que dar lo mismo.

Oraculo: GAMS en niveles (``comp_nus333.gms`` con ``avaall.fx('USA','a_SER',
'shock')=0.10``, capFlex, ``3x3ces.prm``), % shock/check a 6 decimales. En TBL45A
el mismo GAMS reproduce GEMPACK Gragg a <=0.0002pp (xi USA 10,851853 vs 10,851814);
aca no hay Gragg (requiere Windows), asi que la tolerancia es la de TBL45A: 0.002pp.

SKIP si falta nus333; FALLA si falta su ``3x3ces.prm``.
"""

from __future__ import annotations

import pytest
from tests.templates.gtap._nus333 import closure, nus333_params, pct
from tests.templates.gtap._routes import solve

pytestmark = pytest.mark.integration

TOL_PP = 0.002

# GAMS CDE, 3x3ces.prm, capFlex — % cambio shock/check.
ORACLE = {
    "xp": {
        ("USA", "AGR"): 1.829891,
        ("USA", "MFG"): 3.596916,
        ("USA", "SER"): 9.150424,
        ("ROW", "AGR"): 0.569298,
        ("ROW", "MFG"): 0.892446,
        ("ROW", "SER"): -0.375221,
    },
    "rore": {("USA",): 1.830253, ("ROW",): 1.830253},
    "regy": {("USA",): 4.197924, ("ROW",): -1.225676},
    "pi": {("USA",): -3.137535, ("ROW",): -1.228211},
    "xiagg": {("USA",): 10.811101, ("ROW",): -2.211162},
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
    route, how = request.param
    p = nus333_params("3x3ces.prm")
    ac = closure()
    if how == "fix_instrument_shock":
        from equilibria.templates.gtap.instruments import fix_instrument_shock

        s = solve(
            route,
            p,
            ac,
            "ROW",
            base_calibrated=False,
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
        base_calibrated=False,
        skip_base_solve=True,
        solve_check=True,
    )
    assert s.codes["shock"] == 1, s.codes
    return s.model


@pytest.mark.parametrize(
    ("var", "key"), [(v, k) for v, cells in ORACLE.items() for k in cells]
)
def test_iguala_a_gams(solved, var, key):
    m = solved
    got = pct(m, var, key)
    want = ORACLE[var][key]
    assert abs(got - want) <= TOL_PP, (
        f"{var}{key}: equilibria {got:+.6f} vs GAMS {want:+.6f}"
    )
