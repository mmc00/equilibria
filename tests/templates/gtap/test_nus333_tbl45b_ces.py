"""nus333 / Burfisher Tabla 4.5B: el shock de TBL45A con ``3x3CES.prm``.

``TBL45B.EXP``: ``avaall("SER","USA") = 10`` con ``3x3CES.prm`` (CDE, SUBPAR=0,5),
cierre estandar GTAPv7 -> ``savf_flag="capFlex"``. Por el kwarg ``lambdava_shock``
y por ``fix_instrument_shock`` directo: las dos vias tienen que dar lo mismo.

Oraculo: GAMS en niveles (``comp_nus333.gms`` con ``avaall.fx('USA','a_SER',
'shock')=0.10``, capFlex, ``3x3CES.prm``), % shock/check a 6 decimales. En TBL45A
el mismo GAMS reproduce GEMPACK Gragg a <=0.0002pp (xi USA 10,851853 vs 10,851814);
aca no hay Gragg (requiere Windows), asi que la tolerancia es la de TBL45A: 0.002pp.

LOCAL-only: SKIP si falta nus333 o su ``3x3CES.prm``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

TOL_PP = 0.002

# GAMS CDE, 3x3CES.prm, capFlex — % cambio shock/check.
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


@pytest.fixture(scope="module", params=["kwarg", "fix_instrument_shock"])
def solved(request):
    from pyomo.environ import value

    from equilibria._local_refs import nus333_dir

    har = nus333_dir()
    prm = har / "3x3CES.prm"
    if not (har / "basedata.har").exists() or not prm.exists():
        pytest.skip(f"nus333 o 3x3CES.prm no disponible en {har}")

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har / "basedata.har",
        sets_path=har / "sets.har",
        default_path=prm,
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
    m, _mp = build_block_model(
        p, p.sets, ac, "ROW", base_calibrated=False, ref_gdx=None
    )
    shock_kw: dict = {"lambdava_shock": {("USA", "SER"): 1.10}}
    if request.param == "fix_instrument_shock":
        from equilibria.templates.gtap.instruments import fix_instrument_shock

        fix_instrument_shock(m, "lambdava", ("USA", "SER"), factor=1.10)
        shock_kw = {}
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
        **shock_kw,
    )
    assert int(res["shock"]["code"]) == 1, res["shock"]
    return m, value


def _pct(m, value, var, key):
    comp = getattr(m, var)
    return 100.0 * (
        float(value(comp[(*key, "shock")])) / float(value(comp[(*key, "check")])) - 1.0
    )


@pytest.mark.parametrize(
    ("var", "key"), [(v, k) for v, cells in ORACLE.items() for k in cells]
)
def test_iguala_a_gams(solved, var, key):
    m, value = solved
    got = _pct(m, value, var, key)
    want = ORACLE[var][key]
    assert abs(got - want) <= TOL_PP, (
        f"{var}{key}: equilibria {got:+.6f} vs GAMS {want:+.6f}"
    )
