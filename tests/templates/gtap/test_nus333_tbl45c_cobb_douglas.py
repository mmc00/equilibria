"""nus333 / Burfisher Tabla 4.5C: el shock de TBL45A con utilidad Cobb-Douglas.

``TBL45C.EXP``: ``avaall("SER","USA") = 10`` con ``3x3CobbDouglas.prm`` (SUBPAR=0,
INCPAR=1), cierre estandar GTAPv7 -> ``savf_flag="capFlex"``.

Oraculo: GAMS en modo ``--utility=CD`` (``comp_nus333.gms`` con
``avaall.fx('USA','a_SER','shock')=0.10``, capFlex), % shock/check a 6 decimales.
Con CDE (``--utility=CDE``) GAMS divide por cero en este .prm: Cobb-Douglas es la
otra rama de model.gms:761-795, no la CDE con bh=0. En TBL45A el mismo GAMS
reproduce GEMPACK Gragg a <=0.0002pp; aca no hay Gragg (requiere Windows), asi que
la tolerancia es la misma que en TBL45A: 0.002pp.

LOCAL-only: SKIP si falta nus333 o su ``3x3CobbDouglas.prm``.
"""

from __future__ import annotations

import pytest
from tests.templates.gtap._nus333 import closure, nus333_params, pct

pytestmark = pytest.mark.integration

TOL_PP = 0.002

# GAMS --utility=CD, capFlex — % cambio shock/check.
ORACLE = {
    "xp": {
        ("USA", "AGR"): 1.143736,
        ("USA", "MFG"): 2.740057,
        ("USA", "SER"): 9.356648,
        ("ROW", "AGR"): 0.469037,
        ("ROW", "MFG"): 0.885746,
        ("ROW", "SER"): -0.364424,
    },
    "rore": {("USA",): 1.85611, ("ROW",): 1.85611},
    "regy": {("USA",): 4.33883, ("ROW",): -1.280096},
    "pi": {("USA",): -3.029648, ("ROW",): -1.274519},
    "xiagg": {("USA",): 10.856966, ("ROW",): -2.243197},
}


@pytest.fixture(scope="module", params=[True, False], ids=["mute", "sin_mute"])
def solved(request):
    """``mute_welfare=True`` fija ev/cv por su cuenta; ``False`` deja que las fije
    ``_fix_cd_welfare`` (eq_ev/eq_cv no existen bajo CD), que es lo que se prueba."""
    from pyomo.environ import value

    from equilibria.blocks.gtap import _derived_params as dp
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p = nus333_params("3x3CobbDouglas.prm")
    assert dp.cd_regions(p, p.sets) == frozenset(p.sets.r)
    ac = closure()
    m, _mp = build_block_model(
        p, p.sets, ac, "ROW", base_calibrated=False, ref_gdx=None
    )
    res = solve_multiperiod(
        m,
        p,
        ac,
        ref_gdx=None,
        skip_base_solve=True,
        mute_welfare=request.param,
        seed_from_prior=False,
        mode="gtap",
        solve_check=True,
        lambdava_shock={("USA", "SER"): 1.10},
    )
    assert int(res["shock"]["code"]) == 1, res["shock"]
    return m, value


@pytest.mark.parametrize(
    ("var", "key"), [(v, k) for v, cells in ORACLE.items() for k in cells]
)
def test_iguala_a_gams_cobb_douglas(solved, var, key):
    m, value = solved
    got = pct(m, var, key)
    want = ORACLE[var][key]
    assert abs(got - want) <= TOL_PP, (
        f"{var}{key}: equilibria {got:+.6f} vs GAMS-CD {want:+.6f}"
    )


@pytest.mark.parametrize("t", ["check", "shock"])
def test_ev_cv_fijados_al_ingreso_de_calibracion(solved, t):
    """Bajo CD GAMS no tiene eveq/cveq (model.gms:1322/1328): ev/cv quedan en
    ``ev.l = cv.l = yc.l`` de la calibracion (cal.gms:245-246), en TODO periodo."""
    m, value = solved
    for r in m.r:
        yc0 = float(value(m.yc[r, "base"]))
        for nombre in ("ev", "cv"):
            vd = getattr(m, nombre)[r, t]
            assert vd.fixed, f"{nombre}[{r},{t}] libre sin su ecuacion"
            assert float(value(vd)) == pytest.approx(yc0, rel=1e-9), (
                f"{nombre}[{r},{t}]={float(value(vd))} vs yc base {yc0}"
            )
