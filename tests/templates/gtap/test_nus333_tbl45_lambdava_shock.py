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


@pytest.fixture(scope="module")
def solved():
    from pyomo.environ import value

    from equilibria._local_refs import nus333_dir

    har = nus333_dir()
    if not (har / "basedata.har").exists():
        pytest.skip(f"nus333 no disponible en {har}")

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

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
    m, _mp = build_block_model(p, p.sets, ac, "ROW", base_calibrated=True, ref_gdx=None)
    res = solve_multiperiod(
        m,
        p,
        ac,
        ref_gdx=None,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        holdfix_cd=True,
        mode="gtap",
        solve_check=True,
        lambdava_shock={("USA", "SER"): 1.10},
    )
    assert int(res["shock"]["code"]) == 1, res["shock"]
    return m, value


def _pct(m, value, var, key, a="check", b="shock"):
    comp = getattr(m, var)
    return 100.0 * (float(value(comp[(*key, b)])) / float(value(comp[(*key, a)])) - 1.0)


def test_el_shock_no_entra_al_check(solved):
    """base y check deben ser el mismo benchmark: el shock va solo en 'shock'."""
    m, value = solved
    for key in ORACLE["xp"]:
        assert abs(_pct(m, value, "xp", key, "base", "check")) < 1e-6, key


@pytest.mark.parametrize(
    ("var", "key"), [(v, k) for v, cells in ORACLE.items() for k in cells]
)
def test_iguala_a_gempack_gragg(solved, var, key):
    m, value = solved
    got = _pct(m, value, var, key)
    want = ORACLE[var][key]
    assert abs(got - want) <= TOL_PP, (
        f"{var}{key}: equilibria {got:+.6f} vs GEMPACK {want:+.6f}"
    )
