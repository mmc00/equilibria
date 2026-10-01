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

LOCAL-only: SKIP si falta nus333.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

import pytest

pytestmark = pytest.mark.integration

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


@pytest.fixture(scope="module")
def solved():
    from pyomo.environ import value

    from equilibria._local_refs import nus333_dir

    har = nus333_dir()
    if not (har / "basedata.har").exists():
        pytest.skip(f"nus333 no disponible en {har}")

    from equilibria.blocks.gtap import ClosureBlock, ShockBlock
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.instruments import fix_instrument_shock

    ow = cast(Any, importlib.import_module("equilibria.blocks.gtap.overwrite"))

    @ow.overwrite(ShockBlock)
    def desempleo(b):
        b.endogeno("aft", ("USA", "LABOR"))

    @ow.overwrite(ClosureBlock)
    def salario_real(b):
        b.ecuacion(
            "eq_wreal",
            ("USA", "LABOR"),
            lambda m, r, f: m.pft[r, f]
            == value(m.pft[r, f]) * ow.ppriv_tornqvist(m, r),
        )

    try:
        p = GTAPParameters()
        p.load_from_har(
            basedata_path=har / "basedata.har",
            sets_path=har / "sets.har",
            default_path=har / "default.prm",
            baserate_path=har / "baserate.har",
        )
        ac = GTAPClosureConfig(
            name="base",
            closure_type="MCP",
            capital_mobility="sluggish",
            fix_endowments=False,
            fix_taxes=False,
            fix_technology=False,
            if_sub=False,
            savf_flag="capFlex",
            numeraire="pnum",
        )
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
        yield m, res, value, ow
    finally:
        ow.overwrite.clear()


def _pct(m, value, var, key):
    comp = getattr(m, var)
    return 100.0 * (
        float(value(comp[(*key, "shock")])) / float(value(comp[(*key, "check")])) - 1.0
    )


def test_resuelve(solved):
    _, res, _, _ = solved
    for t in ("check", "shock"):
        assert int(res[t]["code"]) == 1, (t, res[t])


def test_salario_real_fijo(solved):
    """El Tornqvist de hogares sube lo mismo que pft[USA,LABOR] en el shock."""
    m, _, value, ow = solved
    got = _pct(m, value, "pft", ("USA", "LABOR"))
    assert abs(got - 7.691635) < TOL_PP
    # Celda libre en el check: sin shock, el empleo queda en el de la base (a
    # precision del solver, relativa; en GAMS base vs check difiere hasta 3e-6).
    a_chk = float(value(m.aft["USA", "LABOR", "check"]))
    a_base = float(value(m.aft["USA", "LABOR", "base"]))
    assert abs(a_chk / a_base - 1) < 1e-6


def test_iguala_a_gams(solved):
    m, _, value, _ = solved
    malas = []
    for var, cells in ORACLE.items():
        for key, want in cells.items():
            got = _pct(m, value, var, key)
            if abs(got - want) > TOL_PP:
                malas.append(f"{var}{key}: equilibria {got:+.6f} vs GAMS {want:+.6f}")
    n = sum(len(c) for c in ORACLE.values())
    assert not malas, (
        f"TBL65B: {len(malas)}/{n} celdas fuera de {TOL_PP}pp:\n" + "\n".join(malas)
    )
