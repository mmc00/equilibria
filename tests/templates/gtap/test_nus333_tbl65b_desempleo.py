"""nus333 / Burfisher TBL65B (= ME3C): subsidio a MFG de USA con desempleo, contra GAMS.

Mismo shock que TBL65A (``to(MFG,MFG,USA) = -10.9414``), pero con el cierre de
``nus333/ME3C.EXP``: ``swap qe("LABOR","USA") = pebfactreal("LABOR","USA")``. El salario
real de USA queda fijo y el empleo se ajusta. En equilibria el cierre se arma con
``@overwrite``, igual que en el notebook del ejercicio:

- ``ShockBlock``: ``aft[USA,LABOR]`` endogeno;
- ``ClosureBlock``: ``eq_wreal``, ``pft = pft0 * T``, con T el indice Tornqvist del
  consumo de hogares contra la base.

Deflactor: ``pebfactreal = peb/ppriv`` (medido en TBL65B.sl4), y ``ppriv`` es un Divisia.
El Tornqvist es su version en un paso. Medido en GAMS: en TBL65A, el Divisia sobre el
camino de GAMS (20 pasos) da +9,2576 vs ``ppriv`` +9,2580. Oraculo GAMS TBL65B
(gams_shock/gen_gams.py, UNEMP, deflactor ``tornq``) vs GEMPACK TBL65B.sl4:

    qe LABOR USA   GAMS +38,7809   GEMPACK +38,7830
    qo AGR/MFG/SER GAMS +4,0016/+28,3125/+26,8804   GEMPACK +4,0016/+28,3127/+26,8821

Con ``pcons`` (pconseq) GAMS da +37,01; con ``cv`` o ``pabs``, +38,84. Los precios nominales
difieren de GEMPACK por un factor uniforme de 1,0002 (el numerario). En GAMS la base es
igual al check (<=3e-6 en pft/pa/xa), asi que anclar a la base equivale al cierre
del oraculo, que ancla al check.

Todas las celdas: los niveles GAMS de check y shock de TBL65B y ME3C (cada uno
con su GDX, gen_gams UNEMP) estan en ``tests/fixtures/nus333_desempleo_gams_levels.json.gz``,
extraidos con ``_diff_core.gams_levels``, y se comparan con ``run_burfisher.compare``
(la misma logica que los 35 ejercicios). ME3C es el mismo .EXP que TBL65B.

LOCAL-only: SKIP si falta nus333.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from tests.templates.gtap._desempleo import register_desempleo_hooks
from tests.templates.gtap._nus333 import (
    closure,
    gams_levels,
    nus333_params,
    pct,
    run_burfisher,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
LEVELS = ROOT / "tests" / "fixtures" / "nus333_desempleo_gams_levels.json.gz"
TOL_PP = 0.002

# GAMS capFlex, % shock/check (oraculo de arriba).
ORACLE = {
    "xft": {("USA", "LABOR"): 38.780869},
    "pft": {
        ("USA", "LAND"): 52.645631,
        ("USA", "LABOR"): 7.691635,
        ("USA", "CAPITAL"): 38.567014,
        ("ROW", "LAND"): 5.810518,
        ("ROW", "LABOR"): -6.539468,
        ("ROW", "CAPITAL"): -6.525959,
    },
    "xp": {
        ("USA", "AGR"): 4.001585,
        ("USA", "MFG"): 28.312494,
        ("USA", "SER"): 26.880396,
        ("ROW", "AGR"): 2.374047,
        ("ROW", "MFG"): 1.449308,
        ("ROW", "SER"): -0.731717,
    },
    "rore": {("USA",): 9.33956, ("ROW",): 9.33956},
    "regy": {("USA",): 42.26937, ("ROW",): -6.482061},
    "pi": {("USA",): 4.499024, ("ROW",): -6.294348},
}


@pytest.fixture(scope="module", params=["TBL65B", "ME3C"])
def solved(request):
    from pyomo.environ import value

    from equilibria.blocks.gtap import overwrite
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.instruments import fix_instrument_shock

    p = nus333_params()
    ac = closure()
    register_desempleo_hooks()
    try:
        m, _ = cast(
            Any,
            build_block_model(
                p, p.sets, ac, "ROW", base_calibrated=False, ref_gdx=None
            ),
        )
        idx = ("USA", "MFG", "MFG")
        t0 = float(value(m.prdtx_rai[(*idx, "check")]))
        fix_instrument_shock(m, "prdtx_rai", idx, value=(1 + t0) * (1 - 0.109414) - 1)
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
        yield request.param, m, p, res, value
    finally:
        overwrite.clear()


def test_resuelve(solved):
    _, _, _, res, _ = solved
    for t in ("check", "shock"):
        assert int(res[t]["code"]) == 1, (t, res[t])


def test_salario_real_fijo(solved):
    """pft[USA,LABOR] sube lo mismo que el Tornqvist de hogares en el shock."""
    _, m, _, _, value = solved
    got = pct(m, "pft", ("USA", "LABOR"))
    assert abs(got - 7.691635) < TOL_PP
    # Celda libre en el check: sin shock, el empleo queda en el de la base (a
    # precision del solver, relativa; en GAMS base vs check difiere hasta 3e-6).
    a_chk = float(value(m.aft["USA", "LABOR", "check"]))
    a_base = float(value(m.aft["USA", "LABOR", "base"]))
    assert abs(a_chk / a_base - 1) < 1e-6


def test_iguala_a_gams(solved):
    exp, m, _, _, value = solved
    malas = []
    for var, cells in ORACLE.items():
        for key, want in cells.items():
            got = pct(m, var, key)
            if abs(got - want) > TOL_PP:
                malas.append(f"{var}{key}: equilibria {got:+.6f} vs GAMS {want:+.6f}")
    n = sum(len(c) for c in ORACLE.values())
    assert not malas, (
        f"{exp}: {len(malas)}/{n} celdas fuera de {TOL_PP}pp:\n" + "\n".join(malas)
    )


def test_todas_las_celdas_contra_gams(solved):
    """Check y shock, todas las celdas, contra los niveles de GAMS (tolerancia 0,1%)."""
    exp, m, p, _, _ = solved
    r = run_burfisher().compare(m, p, gams_levels(LEVELS, exp))
    for period in ("check", "shock"):
        got = r[period]
        assert got["cells"] > 500, (exp, period, got["cells"])
        assert got["match_pct"]["0.1%"] == 100.0, (exp, period, got["worst"])
        assert got["instr_bad"] == [], (exp, period, got["instr_bad"])
