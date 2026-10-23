"""Burfisher 3e / nus333: corre cada ejercicio con equilibria y lo compara contra GAMS.

Cada ejercicio es una fila de datos: el ``.prm`` y la lista de shocks del ``.EXP``
traducidos a instrumentos del ShockBlock: (instrumento, indice, % GEMPACK). El
shock se aplica con ``shocks.apply_shock`` (solo la celda 'shock'), que decide la
conversion a niveles segun el instrumento; base y check son el benchmark.
Despues se comparan TODAS las celdas de check y shock contra el GDX de GAMS del
mismo ejercicio (``<gams-dir>/<EXP>_capFlex.gdx``), con las exclusiones de
``measure_gtap_pure_tols.py``.

Uso:
    .venv/bin/python scripts/gtap/run_burfisher.py --gams-dir <dir> [--only TBL46A,TBL78]

Lee el paquete del libro de ``datasets/nus333`` (o de ``EQUILIBRIA_NUS333_DIR``) y
necesita gdxdump en el PATH.
Los GDX de GAMS los genera ``gen_burfisher_gams.py`` con la misma tabla EXERCISES.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

ACTS = ("AGR", "MFG", "SER")


def me9(land, ao_agr, afe):
    """Shocks de ME9B/C/D. ``land``: qe LAND (USA, ROW); ``ao_agr``: aoall(AGR) %;
    ``afe``: afeall LABOR (USA, ROW) en todas las actividades, o None."""
    aoreg = {"USA": 31.94, "ROW": 42.31}
    out: list[tuple[str, tuple, float]] = []
    for r in ("USA", "ROW"):
        for a in ACTS:
            f = 1 + aoreg[r] / 100
            if a == "AGR":
                f *= 1 + ao_agr / 100
            out.append(("axp", (r, a), round(100 * (f - 1), 10)))
    out += [("pop", ("USA",), 32.3), ("pop", ("ROW",), 37.3)]
    out += [
        ("aft", ("USA", "LABOR"), 24.1),
        ("aft", ("ROW", "LABOR"), 38.4),
        ("aft", ("USA", "CAPITAL"), 60.6),
        ("aft", ("ROW", "CAPITAL"), 213.1),
        ("aft", ("USA", "LAND"), land[0]),
        ("aft", ("ROW", "LAND"), land[1]),
    ]
    if afe is not None:
        for r, x in zip(("USA", "ROW"), afe, strict=True):
            out += [("lambdaf", (r, "LABOR", a), x) for a in ACTS]
    return out


# EXP: (prm, [(instrumento, indice, % GEMPACK)]); la conversion la decide el instrumento
# (shocks.GEMPACK_KIND).
EXERCISES: dict[str, tuple[str, list[tuple[str, tuple, float]]]] = {
    # avaall -> lambdava
    "TBL45A": ("default.prm", [("lambdava", ("USA", "SER"), 10.0)]),
    "TBL45B": ("3x3CES.prm", [("lambdava", ("USA", "SER"), 10.0)]),
    "TBL45C": ("3x3CobbDouglas.prm", [("lambdava", ("USA", "SER"), 10.0)]),
    # qe -> aft
    "TBL55": ("default.prm", [("aft", ("USA", "CAPITAL"), 10.0)]),
    "TBL63A": ("default.prm", [("aft", ("USA", "LABOR"), 10.0)]),
    "TBL63B": ("esubvaklassubstitutes.prm", [("aft", ("USA", "LABOR"), 10.0)]),
    "TBL66": ("default.prm", [("aft", ("USA", "LABOR"), 2.0)]),
    "TBL77": ("esubva4allsects.prm", [("aft", ("USA", "LAND"), 10.0)]),
    # tms -> imptx (exportador, bien, importador)
    "TBL46A": ("esubd0.8.prm", [("imptx", ("ROW", "MFG", "USA"), 8.6637)]),
    "TBL46B": ("esubd1.2.prm", [("imptx", ("ROW", "MFG", "USA"), 8.6637)]),
    "TBL46C": ("esubd4.prm", [("imptx", ("ROW", "MFG", "USA"), 8.6637)]),
    "TBL75A": ("usmfgesubm3.prm", [("imptx", ("ROW", "MFG", "USA"), 13.6030)]),
    "TBL75B": (
        "usmfgesubm10.prm",
        [("imptx", ("ROW", "MFG", "USA"), 13.6030)],
    ),
    "ME7A": ("default.prm", [("imptx", ("ROW", "MFG", "USA"), 4.9395)]),
    "ME7B": (
        "default.prm",
        [
            ("imptx", ("ROW", "MFG", "USA"), 4.9395),
            ("imptx", ("USA", "MFG", "ROW"), 4.8637),
        ],
    ),
    # to -> prdtx_rai (region, actividad, bien)
    "TBL65A": (
        "default.prm",
        [("prdtx_rai", ("USA", "MFG", "MFG"), -10.9414)],
    ),
    "TBL67": ("default.prm", [("prdtx_rai", ("USA", "SER", "SER"), -7.6648)]),
    "ME3A": ("default.prm", [("prdtx_rai", ("USA", "MFG", "MFG"), -10.9414)]),
    "ME3B": (
        "esubva20mfg.prm",
        [("prdtx_rai", ("USA", "MFG", "MFG"), -10.9414)],
    ),
    # tfe -> fcttx (region, factor, actividad)
    "TBL54A": (
        "ESUBVAmfg1.2.prm",
        [("fcttx", ("USA", "LABOR", "MFG"), 4.2969)],
    ),
    "TBL54B": (
        "esubvamfg.8.prm",
        [("fcttx", ("USA", "LABOR", "MFG"), 4.2969)],
    ),
    # tpdall -> dintx_tgt (region, bien, agente)
    "TBL62A": (
        "default.prm",
        [("dintx_tgt", ("USA", "MFG", "hhd"), -13.7359)],
    ),
    "TBL62B": (
        "SectorspecificK.prm",
        [("dintx_tgt", ("USA", "MFG", "hhd"), -13.7359)],
    ),
    "TBL62C": (
        "SluggishK.prm",
        [("dintx_tgt", ("USA", "MFG", "hhd"), -13.7359)],
    ),
    # afeall -> lambdaf (region, factor, actividad)
    "TBL64": (
        "default.prm",
        [("lambdaf", ("USA", "LABOR", a), 10.0) for a in ACTS],
    ),
    "ME4A": ("default.prm", [("lambdaf", ("ROW", "LAND", "AGR"), -81.0)]),
    "ME4B": ("3x3CobbDouglas.prm", [("lambdaf", ("ROW", "LAND", "AGR"), -81.0)]),
    # aoall -> axp (region, actividad)
    "TBL78": ("default.prm", [("axp", ("ROW", "MFG"), -6.0)]),
    # ams -> lambdam (origen, bien, destino)
    "TBL93": ("default.prm", [("lambdam", ("ROW", "MFG", "USA"), 2.0)]),
    # tfd = target% N from file tfd.shk -> dintx_tgt: la tasa pasa a N% (libro p.155);
    # potencia x = 100*((1+N/100)*(1+e/100) - 1), e del .shk (elimina el impuesto).
    "TBL53": (
        "esubd4.prm",
        [
            ("dintx_tgt", ("USA", "AGR", "AGR"), 9.3219304),
            ("dintx_tgt", ("USA", "AGR", "MFG"), 5.0),
            ("dintx_tgt", ("USA", "AGR", "SER"), 5.0000125),
            ("dintx_tgt", ("USA", "MFG", "AGR"), 10.6509107),
            ("dintx_tgt", ("USA", "MFG", "MFG"), 9.4152162),
            ("dintx_tgt", ("USA", "MFG", "SER"), 6.8348582),
            ("dintx_tgt", ("USA", "SER", "AGR"), 6.3567546),
            ("dintx_tgt", ("USA", "SER", "MFG"), 1.7135180),
            ("dintx_tgt", ("USA", "SER", "SER"), 1.8647366),
        ],
    ),
    # atd -> lambdamg (margen, origen, bien, destino): en nus333 solo SER es margen y
    # solo ROW->USA lleva margen hacia USA; el resto de las celdas es inerte.
    "TBL79": (
        "default.prm",
        [("lambdamg", ("SER", "ROW", i, "USA"), 10.0) for i in ("AGR", "MFG")],
    ),
    # TBL813 (Tabla 8.13): +1% a la tasa del impuesto al consumo privado de MFG
    # domestico en USA. GEMPACK: `tpdall = rate% 1 from file tpdall.shk`, y el .shk
    # trae el shock que ELIMINA cada impuesto (-9.1956644 = 1/1.101269 - 1). Subir la
    # tasa 1% es -0.01 x ese valor sobre la potencia (exacto, no lineal en t).
    "TBL813": (
        "ballard.prm",
        [("dintx_tgt", ("USA", "MFG", "hhd"), 0.091956644)],
    ),
    # ME5: quitar los subsidios agricolas de USA. tfe -> fcttx, tfd -> dintx_tgt y
    # tfm -> mintx_tgt, con la actividad AGR como agente comprador.
    "ME5": (
        "default.prm",
        [
            ("fcttx", ("USA", "LAND", "AGR"), 4.2896),
            ("fcttx", ("USA", "CAPITAL", "AGR"), 3.2700),
            ("dintx_tgt", ("USA", "AGR", "AGR"), 4.1161),
            ("dintx_tgt", ("USA", "MFG", "AGR"), 0.5917),
            ("dintx_tgt", ("USA", "SER", "AGR"), 4.2713),
            ("mintx_tgt", ("USA", "AGR", "AGR"), 4.2910),
            ("mintx_tgt", ("USA", "MFG", "AGR"), 1.9788),
            ("mintx_tgt", ("USA", "SER", "AGR"), 4.7064),
        ],
    ),
    # ME8: +1% a TODAS las tasas de impuesto de USA (rate% 1 from file X.shk, 11
    # shocks). Celda por celda, -0.01 x el .shk (que trae el shock de ELIMINAR cada
    # impuesto); solo las celdas distintas de 0. tinc -> kappaf (potencia
    # 1/(1-kappaf)), txs -> exptx, to(i,a) -> prdtx_rai(r,a,i), tgd/tgm -> gov,
    # tpdall/tpmall -> hhd. Desplegado leyendo los nombres de cada eje del .shk.
    "ME8": (
        "ballard.prm",
        [
            ("fcttx", ("USA", "LAND", "AGR"), -0.04289619),
            ("fcttx", ("USA", "LABOR", "AGR"), 0.075687323),
            ("fcttx", ("USA", "LABOR", "MFG"), 0.13085938),
            ("fcttx", ("USA", "LABOR", "SER"), 0.13085938),
            ("fcttx", ("USA", "CAPITAL", "AGR"), -0.032699599),
            ("fcttx", ("USA", "CAPITAL", "MFG"), 0.031535168),
            ("fcttx", ("USA", "CAPITAL", "SER"), 0.031535168),
            ("dintx_tgt", ("USA", "AGR", "AGR"), -0.041161242),
            ("dintx_tgt", ("USA", "AGR", "SER"), -1.192e-07),
            ("dintx_tgt", ("USA", "MFG", "AGR"), -0.0059173703),
            ("dintx_tgt", ("USA", "MFG", "MFG"), 0.0053162163),
            ("dintx_tgt", ("USA", "MFG", "SER"), 0.028774016),
            ("dintx_tgt", ("USA", "SER", "AGR"), -0.04271328),
            ("dintx_tgt", ("USA", "SER", "MFG"), 0.0028086472),
            ("dintx_tgt", ("USA", "SER", "SER"), 0.0013261114),
            ("mintx_tgt", ("USA", "AGR", "AGR"), -0.042909613),
            ("mintx_tgt", ("USA", "MFG", "AGR"), -0.019788332),
            ("mintx_tgt", ("USA", "MFG", "MFG"), 0.0043851116),
            ("mintx_tgt", ("USA", "MFG", "SER"), 0.022134778),
            ("mintx_tgt", ("USA", "SER", "AGR"), -0.047064238),
            ("prdtx_rai", ("USA", "AGR", "AGR"), 0.0024332063),
            ("prdtx_rai", ("USA", "MFG", "MFG"), 0.010459609),
            ("prdtx_rai", ("USA", "SER", "SER"), 0.028050888),
            ("dintx_tgt", ("USA", "AGR", "hhd"), 0.042836466),
            ("dintx_tgt", ("USA", "MFG", "hhd"), 0.091956644),
            ("dintx_tgt", ("USA", "SER", "hhd"), 0.0064833277),
            ("mintx_tgt", ("USA", "AGR", "hhd"), 0.058277063),
            ("mintx_tgt", ("USA", "MFG", "hhd"), 0.079622064),
            ("mintx_tgt", ("USA", "SER", "hhd"), 0.0001112099),
            ("kappaf", ("USA", "LAND", "AGR"), 0.082899618),
            ("kappaf", ("USA", "LABOR", "AGR"), 0.21230932),
            ("kappaf", ("USA", "LABOR", "MFG"), 0.21230932),
            ("kappaf", ("USA", "LABOR", "SER"), 0.21230932),
            ("kappaf", ("USA", "CAPITAL", "AGR"), 0.082899523),
            ("kappaf", ("USA", "CAPITAL", "MFG"), 0.082899523),
            ("kappaf", ("USA", "CAPITAL", "SER"), 0.082899618),
            ("imptx", ("ROW", "AGR", "USA"), 0.015394439),
            ("imptx", ("ROW", "MFG", "USA"), 0.012148236),
            ("exptx", ("USA", "MFG", "ROW"), 0.0029257515),
        ],
    ),
    # ME9B-D (2010-2050, climatechange.prm): cierre ESTANDAR, solo instrumentos.
    # aoreg -> axp de todas las actividades; pop -> pop; qe -> aft; afeall -> lambdaf.
    # aoall(AGR) y aoreg se SUMAN en % en GEMPACK (ao = aoall + aosec + aoreg), o sea
    # que en niveles se MULTIPLICAN: axp(AGR) = 1,3194 x 0,89 en USA.
    "ME9B": ("climatechange.prm", me9(land=(-0.93, 4.4), ao_agr=0.0, afe=None)),
    "ME9C": ("climatechange.prm", me9(land=(10.07, 15.4), ao_agr=-11.0, afe=None)),
    "ME9D": (
        "climatechange.prm",
        me9(land=(10.07, 15.4), ao_agr=-11.0, afe=(-0.73, -2.0)),
    ),
}

# Como GEMPACK (Gragg), el shock no se aplica de una vez: se sube en pasos y cada
# solve arranca del anterior. Sin pasos, ME9B/C (capital ROW +213 %) a veces no
# resuelven, con PATH o con Newton-TR.
SHOCK_STEPS = (0.25, 0.5, 0.75, 1.0)

SKIP = {"walras", "ev", "cv", "uh", "u", "ug", "us"}
RF = {
    "pfa", "pfy", "pm", "pmcif", "pefob", "pwmg", "pp", "pdp", "pmp", "xwmg", "xmgm",
    "lambdamg", "imptx", "exptx",
}  # fmt: skip
ALIAS = {
    "xa": "xaa",
    "xd": "xda",
    "xm": "xma",
    "pp": "pp_rai",
    "p": "p_rai",
    "ytaxInd": "ytax_ind",
    "ytaxind": "ytax_ind",
    "xi": "xiagg",
}
TOLS = (1e-3, 5e-3, 1e-2)


def as_shock(rows: list[tuple[str, tuple, float]]) -> dict[str, dict[tuple, float]]:
    """Las filas (instrumento, indice, %) de un ejercicio como shock de apply_shock."""
    out: dict[str, dict[tuple, float]] = {}
    for name, idx, pct in rows:
        cells = out.setdefault(name, {})
        if idx in cells:
            raise ValueError(f"{name}{idx}: shock repetido")
        cells[idx] = pct
    return out


def solve_exercise(exp: str, har: Path):
    """Construye, aplica los shocks del ejercicio y resuelve base/check/shock."""
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.shocks import apply_shock

    prm, shocks = EXERCISES[exp]
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har / "basedata.har",
        sets_path=har / "sets.har",
        default_path=har / prm,
        baserate_path=har / "baserate.har",
    )
    gc = GTAPClosureConfig(
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
    # GAMS compStat no resuelve la base: sus niveles (cal.gms) son la referencia de
    # los indices de Fisher del check y del shock.
    m, _ = build_block_model(p, p.sets, gc, "ROW", base_calibrated=False, ref_gdx=None)
    apply_shock(m, as_shock(shocks))
    res = solve_multiperiod(
        m,
        p,
        gc,
        ref_gdx=None,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        mode="gtap",
        solve_check=True,
        shock_steps=SHOCK_STEPS,
    )
    return m, p, {k: int(res[k]["code"]) for k in res}


GAMS_PREFIXES = ["a_", "c_", "f_", "r_"]
WHY_SKIP = (
    "bienestar/Walras: GAMS no los resuelve en el MCP comparado (SKIP de los gates)"
)
WHY_RF = (
    "fijada a su macro con ifSUB (RF de measure_gtap_pure_tols): reporte, no resultado"
)


def compare(m, p, gdx: Path | dict) -> dict:
    """Todas las celdas de check y shock contra GAMS, con ``equilibria.compare``: GAMS
    (A, referencia) contra el modelo (B).

    ``gdx``: el GDX de GAMS, o los niveles ya leidos ``{var: {clave: nivel}}`` (las
    claves como las da ``_diff_core.gams_levels``; asi lo usan los tests, sin gdxdump).

    Los instrumentos exogenos del ShockBlock son datos de entrada, no resultados: no
    entran al %, se cuentan aparte (instr_cells / instr_bad) para ver que el shock es
    el de GAMS. Una celda que un hook @overwrite volvio endogena es un resultado mas.

    El mapeo se arma por ejercicio: cada variable de GAMS con registros que el modelo
    tiene (``ALIAS``), las mismas dims en el mismo orden, ``strip_prefix`` para los
    prefijos de GAMS y SKIP/RF excluidas con su motivo. Antes de comparar se sacan,
    de los dos lados, las celdas que :func:`compare` no cuenta como resultado: los
    instrumentos exogenos del ShockBlock (se cuentan aparte) y ``p`` fuera de la
    diagonal sin produccion (``makb = 0``). Devuelve, por periodo, ``cells``,
    ``match_pct``, ``worst`` (las 10 peores fuera del 0,1 %: (% , var, celda)),
    ``instr_cells`` e ``instr_bad``."""
    from pyomo.environ import value

    from equilibria.compare import Mapping, from_dict, from_gdx
    from equilibria.compare import compare as cmp
    from equilibria.templates.gtap.instruments import exogenous_test

    if isinstance(gdx, dict):
        gams = from_dict(gdx, period_dim=-1, origin="gams")
    else:
        gams = from_gdx(gdx, period_dim=-1)
    variables, excluded, eq_cells = [], {}, {}
    instr: dict[str, int] = {"check": 0, "shock": 0}
    instr_bad: dict[str, list] = {"check": [], "shock": []}
    mk = p.benchmark.makb
    for vn, cells in gams.levels.items():
        if vn.lower() in SKIP:
            excluded[vn] = WHY_SKIP
            continue
        if vn.lower() in RF:
            excluded[vn] = WHY_RF
            continue
        if not cells:
            continue
        pv = getattr(m, ALIAS.get(vn, vn), None) or getattr(m, vn.lower(), None)
        if pv is None:
            continue
        exo = exogenous_test(m, pv.local_name)
        in_gams = {
            (*(_gams_strip(e) for e in idx), t): float(x)
            for (t, idx), x in cells.items()
        }
        n = gams.ndim[vn]
        dims = [f"D{i}" for i in range(n)]
        mine = {}
        for idx, c in pv.items():
            key = idx if isinstance(idx, tuple) else (idx,)
            t = key[-1]
            if t not in ("check", "shock") or len(key) != n + 1:
                continue
            body = key[:-1]
            if vn == "p" and len(body) == 3 and body[1] != body[2]:
                if not float(
                    mk.get(body, 0) or mk.get((body[2], body[1], body[0]), 0) or 0
                ):
                    continue
            if exo(key):
                if key in in_gams:  # solo las que GAMS tiene
                    instr[t] += 1
                    v, g = float(value(c)), in_gams[key]
                    if abs(v - g) > 1e-6 and abs(v - g) > TOLS[0] * abs(g):
                        instr_bad[t].append((vn, list(body), v, g))
                continue
            try:
                mine[key] = float(value(c))
            except (ValueError, TypeError):
                continue
        if not mine:
            continue
        eq_cells[f"{vn}__eq"] = mine
        variables.append(
            {
                "name": vn,
                "a": {"var": vn, "dims": dims},
                "b": {"var": f"{vn}__eq", "dims": dims},
                "tol": {"rel": TOLS[0], "abs_floor": 1e-3},
            }
        )
    mapping = Mapping.from_dict(
        {
            "sets": {
                d: {"a": {"strip_prefix": GAMS_PREFIXES}}
                for d in {d for v in variables for d in v["a"]["dims"]}
            },
            "variables": variables,
            "exclude": [{"var": k, "why": w} for k, w in excluded.items()],
            "thresholds": list(TOLS),
        }
    )
    model = from_dict(eq_cells, period_dim=-1, origin="equilibria")
    # GAMS sin base; y sin las celdas que el modelo no cuenta (instrumentos, p fuera
    # de la diagonal): solo se miden las celdas de GAMS que tienen pareja en el modelo.
    keep = {
        v["name"]: {
            (k[-1], tuple(_gams_strip(x) for x in k[:-1]))
            for k in eq_cells[v["b"]["var"]]
        }
        for v in variables
    }
    gams_f = from_dict(
        {
            vn: {
                (*idx, t): x
                for (t, idx), x in cells.items()
                if t in ("check", "shock")
                and (t, tuple(_gams_strip(e) for e in idx)) in keep[vn]
            }
            for vn, cells in gams.levels.items()
            if vn in keep
        },
        period_dim=-1,
        origin="gams",
    )
    r = cmp(gams_f, model, mapping, reference="a")
    out = {}
    for t in ("check", "shock"):
        cs = [c for c in r.cells if c.period == t]
        ds = [c.d for c in cs]
        out[t] = {
            "cells": len(ds),
            "match_pct": {
                f"{thr:.1%}": round(
                    100 * sum(d <= thr for d in ds) / max(len(ds), 1), 2
                )
                for thr in TOLS
            },
            "worst": sorted(
                (
                    (round(100 * c.d, 4), c.variable, list(c.index))
                    for c in cs
                    if c.d > TOLS[0]
                ),
                reverse=True,
            )[:10],
            "instr_cells": instr[t],
            "instr_bad": instr_bad[t],
        }
    return out


def _gams_strip(e: str) -> str:
    for pre in GAMS_PREFIXES:
        if e.startswith(pre):
            return e[len(pre) :]
    return e


def main() -> int:
    from equilibria._local_refs import nus333_dir

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gams-dir", type=Path, required=True)
    ap.add_argument("--only", default="", help="EXP separados por coma")
    ap.add_argument("--json", type=Path, help="guardar los resultados en JSON")
    args = ap.parse_args()

    har = nus333_dir()
    todo = [e for e in args.only.split(",") if e] or list(EXERCISES)
    results = {}
    for exp in todo:
        gdx = args.gams_dir / f"{exp}_capFlex.gdx"
        if not gdx.exists():
            print(f"{exp:7s} sin GDX de GAMS ({gdx.name})")
            continue
        m, p, codes = solve_exercise(exp, har)
        r = {"codes": codes, **compare(m, p, gdx)}
        results[exp] = r
        c, s = r["check"], r["shock"]
        print(
            f"{exp:7s} codes {codes['base']}/{codes['check']}/{codes['shock']}  "
            f"check {c['cells']} {c['match_pct']['0.1%']}%  "
            f"shock {s['cells']} {s['match_pct']['0.1%']}% "
            f"({s['match_pct']['0.5%']}% a 0,5%)  peores {s['worst'][:3]}  "
            f"instrumentos {s['instr_cells']} ({len(s['instr_bad'])} distintos de GAMS)",
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
