"""nus333 / Burfisher: un ejercicio por instrumento del ShockBlock, contra GAMS.

Cada ejercicio aplica su shock con ``fix_instrument_shock`` (solo la celda 'shock')
y se compara el % shock/check contra GAMS en niveles, capFlex, con el MISMO shock
fijado en el periodo shock (``gams_shock/comp_shock.gms`` + ``shocks/<EXP>.inc``).

GAMS se valido antes contra GEMPACK (``.sl4`` del .EXP), 32 celdas por ejercicio:
TBL46A 0,0079pp, TBL65A 0,0287pp, TBL54A 0,0021pp, TBL62A 0,0054pp, TBL78 0,0464pp,
TBL93 0,0042pp, ME5 0,0004pp, ME8 0,0004pp (ME8-DIR.sl4).
TBL79: qxs/pcif/pfob de los 12 flujos y qst a <=0,003pp (qxw de SER difiere por
definicion: GEMPACK suma la oferta de margenes qst, GAMS xet no).
ME9B/ME9C: GAMS vs GEMPACK hasta 0,85pp sobre cambios de ~215% (brecha en estudio,
prueba de escala ME9B-S10 pendiente); aca se mide equilibria contra GAMS. TBL64 es Johansen en GEMPACK (1 paso lineal): GAMS con el shock a
0,1% x 100 lo reproduce a 0,0163pp, asi que el mapeo del shock es correcto y los
0,89pp a 10% son la linealizacion.

Traduccion GEMPACK -> niveles:
- ``tms``/``to``/``tpdall``: % de cambio en la potencia (1+t) ->
  ``t_shock = (1+t_check)*(1+x/100) - 1``.
- ``tfd``/``tfm``: igual, sobre ``dintx_tgt``/``mintx_tgt`` del agente comprador.
- ``tfe``: % de cambio en (1+fctts+fcttx) -> ``fcttx`` absorbe el cambio.
- ``tinc``: % de cambio en la potencia 1/(1-kappaf) (cal.gms:143) ->
  ``kappaf_shock = 1 - (1-kappaf_check)/(1+x/100)``.
- ``txs``: % de la potencia 1+exptx, como ``tms``.
- ``pop``/``aoreg``: % directo -> ``factor = 1+x/100`` (aoreg en ``axp`` de cada actividad).
- ``rate% N from file X.shk`` (ME8): el .shk trae el shock de ELIMINAR cada
  impuesto; subir la tasa N% es ``x = -N/100 x`` ese valor, celda por celda.
- ``afeall``/``aoall``/``ams``: % directo del shifter -> ``factor = 1+x/100``.
- ``target% N from file X.shk`` (TBL53): lleva la tasa ad valorem a N% (libro
  p.155). El .shk trae e = % de la potencia que elimina el impuesto, 1+t0 =
  1/(1+e/100) -> potencia ``x = 100*((1+N/100)*(1+e/100) - 1)``. gtap.exe no
  entiende target% (lo traduce RunGTAP): no hay .sl4; GAMS calza con la Tabla 5.3
  del libro (qint USA +1,04 / -0,12 / +0,01) a 2 decimales.
- ``atd`` (TBL79): % directo de la eficiencia del transporte hacia el destino ->
  ``lambdamg(m,r,i,d)`` (model.gms:1000/1007) en las celdas con margen (GAMS declara
  ``atd`` pero ninguna ecuacion lo usa).

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
    # TBL53: tfd(i,ACTS,"USA") = target% 5 (AGR) / 10 (MFG) / 2 (SER) -> dintx_tgt
    # de cada actividad. Potencias desde nus333/tfd.shk (AGR->MFG: t0 = 0, x = 5).
    "TBL53": (
        "esubd4.prm",
        [
            ("dintx_tgt", ("USA", "AGR", "AGR"), "power", 9.3219304),
            ("dintx_tgt", ("USA", "AGR", "MFG"), "power", 5.0),
            ("dintx_tgt", ("USA", "AGR", "SER"), "power", 5.0000125),
            ("dintx_tgt", ("USA", "MFG", "AGR"), "power", 10.6509107),
            ("dintx_tgt", ("USA", "MFG", "MFG"), "power", 9.4152162),
            ("dintx_tgt", ("USA", "MFG", "SER"), "power", 6.8348582),
            ("dintx_tgt", ("USA", "SER", "AGR"), "power", 6.3567546),
            ("dintx_tgt", ("USA", "SER", "MFG"), "power", 1.7135180),
            ("dintx_tgt", ("USA", "SER", "SER"), "power", 1.8647366),
        ],
    ),
    # atd("USA") = 10 -> lambdamg(margen, origen, bien, USA). En nus333 solo SER es
    # margen y solo ROW->USA lleva margen (amgm); el resto de las celdas es inerte.
    "TBL79": (
        "default.prm",
        [("lambdamg", ("SER", "ROW", i, "USA"), "pct", 10.0) for i in ("AGR", "MFG")],
    ),
    # ME5: tfe -> fcttx, tfd -> dintx_tgt, tfm -> mintx_tgt (agente: actividad AGR).
    "ME5": (
        "default.prm",
        [
            ("fcttx", ("USA", "LAND", "AGR"), "power_fct", 4.2896),
            ("fcttx", ("USA", "CAPITAL", "AGR"), "power_fct", 3.2700),
            ("dintx_tgt", ("USA", "AGR", "AGR"), "power", 4.1161),
            ("dintx_tgt", ("USA", "MFG", "AGR"), "power", 0.5917),
            ("dintx_tgt", ("USA", "SER", "AGR"), "power", 4.2713),
            ("mintx_tgt", ("USA", "AGR", "AGR"), "power", 4.2910),
            ("mintx_tgt", ("USA", "MFG", "AGR"), "power", 1.9788),
            ("mintx_tgt", ("USA", "SER", "AGR"), "power", 4.7064),
        ],
    ),
    # ME8: +1% a todas las tasas de USA; las 39 celdas distintas de 0 de los 11
    # .shk (tinc -> kappaf, txs -> exptx, el resto con los instrumentos de arriba).
    "ME8": (
        "ballard.prm",
        [
            ("fcttx", ("USA", "LAND", "AGR"), "power_fct", -0.04289619),
            ("fcttx", ("USA", "LABOR", "AGR"), "power_fct", 0.075687323),
            ("fcttx", ("USA", "LABOR", "MFG"), "power_fct", 0.13085938),
            ("fcttx", ("USA", "LABOR", "SER"), "power_fct", 0.13085938),
            ("fcttx", ("USA", "CAPITAL", "AGR"), "power_fct", -0.032699599),
            ("fcttx", ("USA", "CAPITAL", "MFG"), "power_fct", 0.031535168),
            ("fcttx", ("USA", "CAPITAL", "SER"), "power_fct", 0.031535168),
            ("dintx_tgt", ("USA", "AGR", "AGR"), "power", -0.041161242),
            ("dintx_tgt", ("USA", "AGR", "SER"), "power", -1.192e-07),
            ("dintx_tgt", ("USA", "MFG", "AGR"), "power", -0.0059173703),
            ("dintx_tgt", ("USA", "MFG", "MFG"), "power", 0.0053162163),
            ("dintx_tgt", ("USA", "MFG", "SER"), "power", 0.028774016),
            ("dintx_tgt", ("USA", "SER", "AGR"), "power", -0.04271328),
            ("dintx_tgt", ("USA", "SER", "MFG"), "power", 0.0028086472),
            ("dintx_tgt", ("USA", "SER", "SER"), "power", 0.0013261114),
            ("mintx_tgt", ("USA", "AGR", "AGR"), "power", -0.042909613),
            ("mintx_tgt", ("USA", "MFG", "AGR"), "power", -0.019788332),
            ("mintx_tgt", ("USA", "MFG", "MFG"), "power", 0.0043851116),
            ("mintx_tgt", ("USA", "MFG", "SER"), "power", 0.022134778),
            ("mintx_tgt", ("USA", "SER", "AGR"), "power", -0.047064238),
            ("prdtx_rai", ("USA", "AGR", "AGR"), "power", 0.0024332063),
            ("prdtx_rai", ("USA", "MFG", "MFG"), "power", 0.010459609),
            ("prdtx_rai", ("USA", "SER", "SER"), "power", 0.028050888),
            ("dintx_tgt", ("USA", "AGR", "hhd"), "power", 0.042836466),
            ("dintx_tgt", ("USA", "MFG", "hhd"), "power", 0.091956644),
            ("dintx_tgt", ("USA", "SER", "hhd"), "power", 0.0064833277),
            ("mintx_tgt", ("USA", "AGR", "hhd"), "power", 0.058277063),
            ("mintx_tgt", ("USA", "MFG", "hhd"), "power", 0.079622064),
            ("mintx_tgt", ("USA", "SER", "hhd"), "power", 0.0001112099),
            ("kappaf", ("USA", "LAND", "AGR"), "power_kappa", 0.082899618),
            ("kappaf", ("USA", "LABOR", "AGR"), "power_kappa", 0.21230932),
            ("kappaf", ("USA", "LABOR", "MFG"), "power_kappa", 0.21230932),
            ("kappaf", ("USA", "LABOR", "SER"), "power_kappa", 0.21230932),
            ("kappaf", ("USA", "CAPITAL", "AGR"), "power_kappa", 0.082899523),
            ("kappaf", ("USA", "CAPITAL", "MFG"), "power_kappa", 0.082899523),
            ("kappaf", ("USA", "CAPITAL", "SER"), "power_kappa", 0.082899618),
            ("imptx", ("ROW", "AGR", "USA"), "power", 0.015394439),
            ("imptx", ("ROW", "MFG", "USA"), "power", 0.012148236),
            ("exptx", ("USA", "MFG", "ROW"), "power", 0.0029257515),
        ],
    ),
    # ME9B/ME9C (2010-2050, climatechange.prm): cierre estandar. aoreg -> axp de
    # todas las actividades (en ME9C x 0,89 en AGR: aoall y aoreg se multiplican en
    # niveles), pop -> pop, qe -> aft.
    "ME9B": (
        "climatechange.prm",
        [
            ("axp", ("USA", "AGR"), "pct", 31.94),
            ("axp", ("USA", "MFG"), "pct", 31.94),
            ("axp", ("USA", "SER"), "pct", 31.94),
            ("axp", ("ROW", "AGR"), "pct", 42.31),
            ("axp", ("ROW", "MFG"), "pct", 42.31),
            ("axp", ("ROW", "SER"), "pct", 42.31),
            ("pop", ("USA",), "pct", 32.3),
            ("pop", ("ROW",), "pct", 37.3),
            ("aft", ("USA", "LABOR"), "pct", 24.1),
            ("aft", ("ROW", "LABOR"), "pct", 38.4),
            ("aft", ("USA", "CAPITAL"), "pct", 60.6),
            ("aft", ("ROW", "CAPITAL"), "pct", 213.1),
            ("aft", ("USA", "LAND"), "pct", -0.93),
            ("aft", ("ROW", "LAND"), "pct", 4.4),
        ],
    ),
    "ME9C": (
        "climatechange.prm",
        [
            ("axp", ("USA", "AGR"), "pct", 17.4266),
            ("axp", ("USA", "MFG"), "pct", 31.94),
            ("axp", ("USA", "SER"), "pct", 31.94),
            ("axp", ("ROW", "AGR"), "pct", 26.6559),
            ("axp", ("ROW", "MFG"), "pct", 42.31),
            ("axp", ("ROW", "SER"), "pct", 42.31),
            ("pop", ("USA",), "pct", 32.3),
            ("pop", ("ROW",), "pct", 37.3),
            ("aft", ("USA", "LABOR"), "pct", 24.1),
            ("aft", ("ROW", "LABOR"), "pct", 38.4),
            ("aft", ("USA", "CAPITAL"), "pct", 60.6),
            ("aft", ("ROW", "CAPITAL"), "pct", 213.1),
            ("aft", ("USA", "LAND"), "pct", 10.07),
            ("aft", ("ROW", "LAND"), "pct", 15.4),
        ],
    ),
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
    "ME5": {
        "xp": {
            ("USA", "AGR"): -1.416146,
            ("USA", "MFG"): 0.055787,
            ("USA", "SER"): 0.007262,
            ("ROW", "AGR"): 0.141865,
            ("ROW", "MFG"): -0.023271,
            ("ROW", "SER"): -0.003176,
        },
        "rore": {("USA",): -0.035358, ("ROW",): -0.035358},
        "regy": {("USA",): -0.053396, ("ROW",): 0.038195},
        "pi": {("USA",): -0.037497, ("ROW",): 0.038001},
        "xiagg": {("USA",): -0.09493, ("ROW",): 0.028993},
    },
    "TBL53": {
        "xp": {
            ("USA", "AGR"): 1.040791,
            ("USA", "MFG"): -0.1206,
            ("USA", "SER"): 0.010949,
            ("ROW", "AGR"): -0.240524,
            ("ROW", "MFG"): -0.370481,
            ("ROW", "SER"): 0.156015,
        },
        # qint de la Tabla 5.3 del libro: +1,04 / -0,12 / +0,01
        "nd": {
            ("USA", "AGR"): 1.040791,
            ("USA", "MFG"): -0.1206,
            ("USA", "SER"): 0.010949,
        },
        "rore": {("USA",): -1.926356, ("ROW",): -1.926356},
        "regy": {("USA",): -5.241516, ("ROW",): 2.924634},
        "pi": {("USA",): -3.018604, ("ROW",): 2.694196},
        "xiagg": {("USA",): -8.853645, ("ROW",): 2.682595},
    },
    "TBL79": {
        "xp": {
            ("USA", "AGR"): -0.212697,
            ("USA", "MFG"): -0.113089,
            ("USA", "SER"): 0.026357,
            ("ROW", "AGR"): 0.027396,
            ("ROW", "MFG"): 0.036812,
            ("ROW", "SER"): -0.015794,
        },
        "rore": {("USA",): 0.020113, ("ROW",): 0.020113},
        "regy": {("USA",): -0.002613, ("ROW",): 0.003995},
        "pi": {("USA",): -0.085911, ("ROW",): -0.001057},
        "xiagg": {("USA",): 0.12907, ("ROW",): -0.019117},
        # el canal del shock: precio cif y volumen ROW->USA, oferta de margenes
        "pmcif": {("ROW", "AGR", "USA"): -1.471087, ("ROW", "MFG", "USA"): -0.391736},
        "xw": {("ROW", "AGR", "USA"): 2.667191, ("ROW", "MFG", "USA"): 0.786517},
        "xaa": {("USA", "SER", "tmg"): -1.2606, ("ROW", "SER", "tmg"): -1.287219},
    },
    "ME8": {
        "xp": {
            ("USA", "AGR"): 0.19338,
            ("USA", "MFG"): 0.045786,
            ("USA", "SER"): -0.012117,
            ("ROW", "AGR"): -0.024914,
            ("ROW", "MFG"): -0.016303,
            ("ROW", "SER"): 0.008051,
        },
        "rore": {("USA",): -0.059708, ("ROW",): -0.059708},
        "regy": {("USA",): 0.032833, ("ROW",): 0.047464},
        "pi": {("USA",): 0.023267, ("ROW",): 0.045714},
        "xiagg": {("USA",): -0.266482, ("ROW",): 0.074028},
    },
    "ME9B": {
        "xp": {
            ("USA", "AGR"): 52.745462,
            ("USA", "MFG"): 33.815545,
            ("USA", "SER"): 85.158579,
            ("ROW", "AGR"): 70.757195,
            ("ROW", "MFG"): 158.560469,
            ("ROW", "SER"): 212.715214,
        },
        "rore": {("USA",): 33.065987, ("ROW",): 33.065987},
        "regy": {("USA",): 35.842836, ("ROW",): 109.819413},
        "pi": {("USA",): -41.240148, ("ROW",): -47.064437},
        "xiagg": {("USA",): 117.401573, ("ROW",): 267.970934},
    },
    "ME9C": {
        "xp": {
            ("USA", "AGR"): 53.063436,
            ("USA", "MFG"): 33.270063,
            ("USA", "SER"): 85.053661,
            ("ROW", "AGR"): 70.237086,
            ("ROW", "MFG"): 157.120398,
            ("ROW", "SER"): 211.538958,
        },
        "rore": {("USA",): 32.351242, ("ROW",): 32.351242},
        "regy": {("USA",): 36.466163, ("ROW",): 109.238355},
        "pi": {("USA",): -40.88362, ("ROW",): -46.941339},
        "xiagg": {("USA",): 117.972772, ("ROW",): 265.824712},
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
    if kind == "power_kappa":
        return 1 - (1 - chk) / (1 + x / 100)
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
    m, _mp = build_block_model(
        p, p.sets, ac, "ROW", base_calibrated=False, ref_gdx=None
    )
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


def test_los_shocks_son_los_de_run_burfisher():
    """SHOCKS es copia de ``run_burfisher.EXERCISES`` (de ahi sale el oraculo GAMS,
    gen_gams.py): si una copia cambia y la otra no, el test y el oraculo miden
    shocks distintos sin avisar."""
    import importlib
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[3]
    # scripts/gtap no es un paquete: se carga por ruta (como lo hace el script).
    sys.path.insert(0, str(root / "scripts" / "gtap"))
    exercises = importlib.import_module("run_burfisher").EXERCISES
    distintos = [k for k in SHOCKS if exercises.get(k) != SHOCKS[k]]
    assert not distintos, f"SHOCKS distinto de run_burfisher.EXERCISES: {distintos}"
