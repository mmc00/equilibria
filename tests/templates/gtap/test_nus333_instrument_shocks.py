"""nus333 / Burfisher: un ejercicio por instrumento del ShockBlock, contra GAMS.

Cada ejercicio aplica su shock con ``fix_instrument_shock`` (solo la celda 'shock')
y se compara el % shock/check contra GAMS en niveles, capFlex, con el MISMO shock
fijado en el periodo shock (``gams_shock/comp_shock.gms`` + ``shocks/<EXP>.inc``).

GAMS se valido antes contra GEMPACK (``.sl4`` del .EXP), 32 celdas por ejercicio:
TBL46A 0,0079pp, TBL65A 0,0287pp, TBL54A 0,0021pp, TBL62A 0,0054pp, TBL78 0,0464pp,
TBL93 0,0042pp. TBL64 es Johansen en GEMPACK (1 paso lineal): GAMS con el shock a
0,1% x 100 lo reproduce a 0,0163pp, asi que el mapeo del shock es correcto y los
0,89pp a 10% son la linealizacion.

Traduccion GEMPACK -> niveles:
- ``tms``/``to``/``tpdall``: % de cambio en la potencia (1+t) ->
  ``t_shock = (1+t_check)*(1+x/100) - 1``.
- ``tfe``: % de cambio en (1+fctts+fcttx) -> ``fcttx`` absorbe el cambio.
- ``afeall``/``aoall``/``ams``: % directo del shifter -> ``factor = 1+x/100``.

LOCAL-only: SKIP si falta nus333 o el .prm.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

TOL_PP = 0.002
ACTS = ("AGR", "MFG", "SER")

# EXP: (prm, [(instrumento, indice, tipo, x)])
SHOCKS = {
    "TBL46A": ("esubd0.8.prm", [("imptx", ("ROW", "MFG", "USA"), "power", 8.6637)]),
    "TBL65A": (
        "default.prm",
        [("prdtx_rai", ("USA", "MFG", "MFG"), "power", -10.9414)],
    ),
    "TBL54A": (
        "ESUBVAmfg1.2.prm",
        [("fcttx", ("USA", "LABOR", "MFG"), "power_fct", 4.2969)],
    ),
    "TBL62A": (
        "default.prm",
        [("dintx_tgt", ("USA", "MFG", "hhd"), "power", -13.7359)],
    ),
    "TBL64": (
        "default.prm",
        [("lambdaf", ("USA", "LABOR", a), "pct", 10.0) for a in ACTS],
    ),
    "TBL78": ("default.prm", [("axp", ("ROW", "MFG"), "pct", -6.0)]),
    "TBL93": ("default.prm", [("lambdam", ("ROW", "MFG", "USA"), "pct", 2.0)]),
}

# GAMS capFlex con el shock del .EXP solo en 'shock' — % cambio shock/check.
ORACLES = {
    "TBL46A": {
        "xp": {
            ("USA", "AGR"): -1.102346,
            ("USA", "MFG"): -0.143071,
            ("USA", "SER"): 0.044455,
            ("ROW", "AGR"): 0.026724,
            ("ROW", "MFG"): -0.175558,
            ("ROW", "SER"): 0.062319,
        },
        "rore": {("USA",): -0.412183, ("ROW",): -0.412183},
        "regy": {("USA",): 2.116299, ("ROW",): -0.348298},
        "pi": {("USA",): 2.381245, ("ROW",): -0.240182},
        "xiagg": {("USA",): -2.264433, ("ROW",): 0.314798},
    },
    "TBL65A": {
        "xp": {
            ("USA", "AGR"): -6.594479,
            ("USA", "MFG"): 4.444661,
            ("USA", "SER"): -0.838426,
            ("ROW", "AGR"): 1.034076,
            ("ROW", "MFG"): 0.229022,
            ("ROW", "SER"): -0.170144,
        },
        "rore": {("USA",): 4.168839, ("ROW",): 4.168839},
        "regy": {("USA",): 11.518476, ("ROW",): -6.119913},
        "pi": {("USA",): 5.915109, ("ROW",): -5.91796},
        "xiagg": {("USA",): 14.976589, ("ROW",): -5.279593},
    },
    "TBL54A": {
        "xp": {
            ("USA", "AGR"): 0.371033,
            ("USA", "MFG"): -0.719073,
            ("USA", "SER"): 0.140529,
            ("ROW", "AGR"): -0.05815,
            ("ROW", "MFG"): 0.052905,
            ("ROW", "SER"): -0.014644,
        },
        "rore": {("USA",): -0.169718, ("ROW",): -0.169718},
        "regy": {("USA",): -0.831667, ("ROW",): 0.455407},
        "pi": {("USA",): -0.382592, ("ROW",): 0.442194},
        "xiagg": {("USA",): -0.679255, ("ROW",): 0.22753},
    },
    "TBL62A": {
        "xp": {
            ("USA", "AGR"): 0.596907,
            ("USA", "MFG"): 3.660332,
            ("USA", "SER"): -0.768637,
            ("ROW", "AGR"): 0.165891,
            ("ROW", "MFG"): 0.034386,
            ("ROW", "SER"): -0.026357,
        },
        "rore": {("USA",): 0.193546, ("ROW",): 0.193546},
        "regy": {("USA",): -0.760413, ("ROW",): -0.384668},
        "pi": {("USA",): 0.760322, ("ROW",): -0.329535},
        "xiagg": {("USA",): 0.439806, ("ROW",): -0.330642},
    },
    "TBL64": {
        "xp": {
            ("USA", "AGR"): 2.194449,
            ("USA", "MFG"): 5.497713,
            ("USA", "SER"): 7.512136,
            ("ROW", "AGR"): 0.405557,
            ("ROW", "MFG"): 0.436907,
            ("ROW", "SER"): -0.194175,
        },
        "rore": {("USA",): 1.498529, ("ROW",): 1.498529},
        "regy": {("USA",): 5.99769, ("ROW",): -1.852146},
        "pi": {("USA",): -1.722383, ("ROW",): -1.84043},
        "xiagg": {("USA",): 8.562886, ("ROW",): -1.82761},
    },
    "TBL78": {
        "xp": {
            ("USA", "AGR"): -4.220607,
            ("USA", "MFG"): 1.550835,
            ("USA", "SER"): -0.266978,
            ("ROW", "AGR"): 1.907194,
            ("ROW", "MFG"): -0.465341,
            ("ROW", "SER"): -2.323702,
        },
        "rore": {("USA",): -4.137303, ("ROW",): -4.137303},
        "regy": {("USA",): 10.23643, ("ROW",): -3.714367},
        "pi": {("USA",): 9.944173, ("ROW",): 2.95006},
        "xiagg": {("USA",): 5.759681, ("ROW",): -6.314724},
    },
    "TBL93": {
        "xp": {
            ("USA", "AGR"): -0.088131,
            ("USA", "MFG"): -0.653483,
            ("USA", "SER"): 0.136969,
            ("ROW", "AGR"): -0.017458,
            ("ROW", "MFG"): 0.026306,
            ("ROW", "SER"): -0.008227,
        },
        "rore": {("USA",): 0.100703, ("ROW",): 0.100703},
        "regy": {("USA",): -0.025728, ("ROW",): 0.022367},
        "pi": {("USA",): -0.431929, ("ROW",): 0.004041},
        "xiagg": {("USA",): 0.613834, ("ROW",): -0.090248},
    },
}


def _level(m, p, name, idx, kind, x):
    """El valor en niveles de la celda 'shock' para el shock GEMPACK ``x``."""
    from pyomo.environ import value

    chk = float(value(getattr(m, name)[(*idx, "check")]))
    if kind == "pct":
        return chk * (1 + x / 100)
    if kind == "power":
        return (1 + chk) * (1 + x / 100) - 1
    assert kind == "power_fct"
    from equilibria.blocks.gtap import _derived_params as dp

    fs = dp.fctts_data(p, p.sets).get(idx, 0.0)
    return (1 + fs + chk) * (1 + x / 100) - 1 - fs


@pytest.fixture(scope="module", params=sorted(SHOCKS))
def solved(request):
    from pyomo.environ import value

    from equilibria._local_refs import nus333_dir

    exp = request.param
    prm_name, shocks = SHOCKS[exp]
    har = nus333_dir()
    prm = har / prm_name
    if not (har / "basedata.har").exists() or not prm.exists():
        pytest.skip(f"nus333 o {prm_name} no disponible en {har}")

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.instruments import fix_instrument_shock

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
    m, _mp = build_block_model(p, p.sets, ac, "ROW", base_calibrated=True, ref_gdx=None)
    for name, idx, kind, x in shocks:
        fix_instrument_shock(m, name, idx, value=_level(m, p, name, idx, kind, x))
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
    assert int(res["shock"]["code"]) == 1, (exp, res["shock"])
    return exp, m, value


def _pct(m, value, var, key):
    comp = getattr(m, var)
    return 100.0 * (
        float(value(comp[(*key, "shock")])) / float(value(comp[(*key, "check")])) - 1.0
    )


def test_iguala_a_gams(solved):
    exp, m, value = solved
    malas = []
    for var, cells in ORACLES[exp].items():
        for key, want in cells.items():
            got = _pct(m, value, var, key)
            if abs(got - want) > TOL_PP:
                malas.append(f"{var}{key}: equilibria {got:+.6f} vs GAMS {want:+.6f}")
    n = sum(len(c) for c in ORACLES[exp].values())
    assert not malas, (
        f"{exp}: {len(malas)}/{n} celdas fuera de {TOL_PP}pp:\n" + "\n".join(malas)
    )
