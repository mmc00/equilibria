"""nus333 / Burfisher Tabla 5.5: +10% de la dotacion de capital de USA.

``TBL55.EXP``: ``qe("CAPITAL","USA") = 10`` con ``default.prm``, cierre estandar
GTAPv7 -> ``savf_flag="capFlex"``. En GAMS la oferta agregada de un factor movil es
``xft = aft*(pft/pabs)**etaf`` (model.gms:1073) con ``etaf=0``: el shock es
``aft(USA,CAPITAL,'shock') *= 1.10``, SOLO en el periodo shock.

Oraculo: GAMS en niveles (``gams_qe/comp_qe.gms`` = ``comp_nus333.gms`` con ese
``aft``, capFlex), % shock/check a 6 decimales. GAMS reproduce la solucion Gragg de
GEMPACK (``TBL55.sl4``) en 32 celdas a <=0.0036pp (0.0006pp salvo xds[ROW,SER]).
Tolerancia: la misma de TBL45, 0.002pp.

LOCAL-only: SKIP si falta nus333.
"""

from __future__ import annotations

import pytest
from tests.templates.gtap._nus333 import closure, nus333_params, pct

pytestmark = pytest.mark.integration

TOL_PP = 0.002

# GAMS capFlex, aft(USA,CAPITAL,shock)*1.10 — % cambio shock/check.
ORACLE = {
    "xp": {
        ("USA", "AGR"): 3.643473,
        ("USA", "MFG"): 5.670788,
        ("USA", "SER"): 2.087705,
        ("ROW", "AGR"): -0.174434,
        ("ROW", "MFG"): -0.511713,
        ("ROW", "SER"): 0.202479,
    },
    "rore": {("USA",): -1.355559, ("ROW",): -1.355559},
    "regy": {("USA",): 0.170797, ("ROW",): 0.683223},
    "pi": {("USA",): -1.257146, ("ROW",): 0.589155},
    "xiagg": {("USA",): -1.057441, ("ROW",): 1.787295},
    "pft": {
        ("USA", "CAPITAL"): -6.794616,
        ("USA", "LABOR"): 0.260528,
        ("USA", "LAND"): 17.766734,
        ("ROW", "CAPITAL"): 0.672682,
        ("ROW", "LABOR"): 0.694255,
        ("ROW", "LAND"): -0.227202,
    },
    # El shock mismo: la dotacion sube 10% y el resto no se mueve.
    "xft": {
        ("USA", "CAPITAL"): 10.0,
        ("USA", "LABOR"): 0.0,
        ("ROW", "CAPITAL"): 0.0,
    },
    "kstock": {("USA",): 10.0, ("ROW",): 0.0},
}


@pytest.fixture(scope="module", params=["apply_shock", "fix_instrument_shock"])
def solved(request):
    """Las dos vias dan lo mismo: ``apply_shock`` (el % de GEMPACK) y
    ``fix_instrument_shock`` directo antes de ``solve_multiperiod`` (ninguna suma el
    arancel +10%)."""
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.shocks import apply_shock

    p = nus333_params()
    ac = closure()
    m, _mp = build_block_model(p, p.sets, ac, "ROW", base_calibrated=True, ref_gdx=None)
    if request.param == "fix_instrument_shock":
        from equilibria.templates.gtap.instruments import fix_instrument_shock

        fix_instrument_shock(m, "aft", ("USA", "CAPITAL"), factor=1.10)
    else:
        apply_shock(m, {"aft": {("USA", "CAPITAL"): 10.0}})
    res = solve_multiperiod(
        m,
        p,
        ac,
        ref_gdx=None,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        mode="gtap",
        solve_check=True,
    )
    assert int(res["shock"]["code"]) == 1, res["shock"]
    return m


def test_el_shock_no_entra_al_check(solved):
    """base y check deben ser el mismo benchmark: el shock va solo en 'shock'."""
    m = solved
    for var in ("xp", "xft", "kstock", "pft"):
        for key in ORACLE[var]:
            assert abs(pct(m, var, key, num="base", den="check")) < 1e-6, (var, key)


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
