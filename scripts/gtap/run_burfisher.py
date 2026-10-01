"""Burfisher 3e / nus333: corre cada ejercicio con equilibria y lo compara contra GAMS.

Cada ejercicio es una fila de datos: el ``.prm`` y la lista de shocks del ``.EXP``
traducidos a instrumentos del ShockBlock. El shock se aplica con
``fix_instrument_shock`` (solo la celda 'shock'); base y check son el benchmark.
Despues se comparan TODAS las celdas de check y shock contra el GDX de GAMS del
mismo ejercicio (``<gams-dir>/<EXP>_capFlex.gdx``), con las exclusiones de
``measure_gtap_pure_tols.py``.

Traduccion GEMPACK -> niveles (``kind``):
- ``pct``: % directo del instrumento (avaall, qe, afeall, aoall, ams) -> x*(1+p/100).
- ``power``: % de la potencia 1+t (tms, to, tpdall) -> (1+t)*(1+p/100)-1.
- ``power_fct``: % de 1+fctts+fcttx (tfe) -> fcttx absorbe el cambio.
- ``power_kappa``: % de la potencia 1/(1-kappaf) (tinc) -> 1-(1-k)/(1+p/100).

Uso:
    .venv/bin/python scripts/gtap/run_burfisher.py --gams-dir <dir> [--only TBL46A,TBL78]

Requiere el dataset nus333 (``EQUILIBRIA_NUS333_DIR``) y gdxdump en el PATH.
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


def _me9(land, ao_agr, afe):
    """Shocks de ME9B/C/D. ``land``: qe LAND (USA, ROW); ``ao_agr``: aoall(AGR) %;
    ``afe``: afeall LABOR (USA, ROW) en todas las actividades, o None."""
    aoreg = {"USA": 31.94, "ROW": 42.31}
    out: list[tuple[str, tuple, str, float]] = []
    for r in ("USA", "ROW"):
        for a in ACTS:
            f = 1 + aoreg[r] / 100
            if a == "AGR":
                f *= 1 + ao_agr / 100
            out.append(("axp", (r, a), "pct", round(100 * (f - 1), 10)))
    out += [("pop", ("USA",), "pct", 32.3), ("pop", ("ROW",), "pct", 37.3)]
    out += [
        ("aft", ("USA", "LABOR"), "pct", 24.1),
        ("aft", ("ROW", "LABOR"), "pct", 38.4),
        ("aft", ("USA", "CAPITAL"), "pct", 60.6),
        ("aft", ("ROW", "CAPITAL"), "pct", 213.1),
        ("aft", ("USA", "LAND"), "pct", land[0]),
        ("aft", ("ROW", "LAND"), "pct", land[1]),
    ]
    if afe is not None:
        for r, x in zip(("USA", "ROW"), afe, strict=True):
            out += [("lambdaf", (r, "LABOR", a), "pct", x) for a in ACTS]
    return out


# EXP: (prm, [(instrumento, indice, kind, porcentaje)])
EXERCISES: dict[str, tuple[str, list[tuple[str, tuple, str, float]]]] = {
    # avaall -> lambdava
    "TBL45A": ("default.prm", [("lambdava", ("USA", "SER"), "pct", 10.0)]),
    "TBL45B": ("3x3CES.prm", [("lambdava", ("USA", "SER"), "pct", 10.0)]),
    "TBL45C": ("3x3CobbDouglas.prm", [("lambdava", ("USA", "SER"), "pct", 10.0)]),
    # qe -> aft
    "TBL55": ("default.prm", [("aft", ("USA", "CAPITAL"), "pct", 10.0)]),
    "TBL63A": ("default.prm", [("aft", ("USA", "LABOR"), "pct", 10.0)]),
    "TBL63B": ("esubvaklassubstitutes.prm", [("aft", ("USA", "LABOR"), "pct", 10.0)]),
    "TBL66": ("default.prm", [("aft", ("USA", "LABOR"), "pct", 2.0)]),
    "TBL77": ("esubva4allsects.prm", [("aft", ("USA", "LAND"), "pct", 10.0)]),
    # tms -> imptx (exportador, bien, importador)
    "TBL46A": ("esubd0.8.prm", [("imptx", ("ROW", "MFG", "USA"), "power", 8.6637)]),
    "TBL46B": ("esubd1.2.prm", [("imptx", ("ROW", "MFG", "USA"), "power", 8.6637)]),
    "TBL46C": ("esubd4.prm", [("imptx", ("ROW", "MFG", "USA"), "power", 8.6637)]),
    "TBL75A": ("usmfgesubm3.prm", [("imptx", ("ROW", "MFG", "USA"), "power", 13.6030)]),
    "TBL75B": (
        "usmfgesubm10.prm",
        [("imptx", ("ROW", "MFG", "USA"), "power", 13.6030)],
    ),
    "ME7A": ("default.prm", [("imptx", ("ROW", "MFG", "USA"), "power", 4.9395)]),
    "ME7B": (
        "default.prm",
        [
            ("imptx", ("ROW", "MFG", "USA"), "power", 4.9395),
            ("imptx", ("USA", "MFG", "ROW"), "power", 4.8637),
        ],
    ),
    # to -> prdtx_rai (region, actividad, bien)
    "TBL65A": (
        "default.prm",
        [("prdtx_rai", ("USA", "MFG", "MFG"), "power", -10.9414)],
    ),
    "TBL67": ("default.prm", [("prdtx_rai", ("USA", "SER", "SER"), "power", -7.6648)]),
    "ME3A": ("default.prm", [("prdtx_rai", ("USA", "MFG", "MFG"), "power", -10.9414)]),
    "ME3B": (
        "esubva20mfg.prm",
        [("prdtx_rai", ("USA", "MFG", "MFG"), "power", -10.9414)],
    ),
    # tfe -> fcttx (region, factor, actividad)
    "TBL54A": (
        "ESUBVAmfg1.2.prm",
        [("fcttx", ("USA", "LABOR", "MFG"), "power_fct", 4.2969)],
    ),
    "TBL54B": (
        "esubvamfg.8.prm",
        [("fcttx", ("USA", "LABOR", "MFG"), "power_fct", 4.2969)],
    ),
    # tpdall -> dintx_tgt (region, bien, agente)
    "TBL62A": (
        "default.prm",
        [("dintx_tgt", ("USA", "MFG", "hhd"), "power", -13.7359)],
    ),
    "TBL62B": (
        "SectorspecificK.prm",
        [("dintx_tgt", ("USA", "MFG", "hhd"), "power", -13.7359)],
    ),
    "TBL62C": (
        "SluggishK.prm",
        [("dintx_tgt", ("USA", "MFG", "hhd"), "power", -13.7359)],
    ),
    # afeall -> lambdaf (region, factor, actividad)
    "TBL64": (
        "default.prm",
        [("lambdaf", ("USA", "LABOR", a), "pct", 10.0) for a in ACTS],
    ),
    "ME4A": ("default.prm", [("lambdaf", ("ROW", "LAND", "AGR"), "pct", -81.0)]),
    "ME4B": ("3x3CobbDouglas.prm", [("lambdaf", ("ROW", "LAND", "AGR"), "pct", -81.0)]),
    # aoall -> axp (region, actividad)
    "TBL78": ("default.prm", [("axp", ("ROW", "MFG"), "pct", -6.0)]),
    # ams -> lambdam (origen, bien, destino)
    "TBL93": ("default.prm", [("lambdam", ("ROW", "MFG", "USA"), "pct", 2.0)]),
    # TBL813 (Tabla 8.13): +1% a la tasa del impuesto al consumo privado de MFG
    # domestico en USA. GEMPACK: `tpdall = rate% 1 from file tpdall.shk`, y el .shk
    # trae el shock que ELIMINA cada impuesto (-9.1956644 = 1/1.101269 - 1). Subir la
    # tasa 1% es -0.01 x ese valor sobre la potencia (exacto, no lineal en t).
    "TBL813": (
        "ballard.prm",
        [("dintx_tgt", ("USA", "MFG", "hhd"), "power", 0.091956644)],
    ),
    # ME5: quitar los subsidios agricolas de USA. tfe -> fcttx, tfd -> dintx_tgt y
    # tfm -> mintx_tgt, con la actividad AGR como agente comprador.
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
    # ME8: +1% a TODAS las tasas de impuesto de USA (rate% 1 from file X.shk, 11
    # shocks). Celda por celda, -0.01 x el .shk (que trae el shock de ELIMINAR cada
    # impuesto); solo las celdas distintas de 0. tinc -> kappaf (potencia
    # 1/(1-kappaf)), txs -> exptx, to(i,a) -> prdtx_rai(r,a,i), tgd/tgm -> gov,
    # tpdall/tpmall -> hhd. Desplegado leyendo los nombres de cada eje del .shk.
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
    # ME9B-D (2010-2050, climatechange.prm): cierre ESTANDAR, solo instrumentos.
    # aoreg -> axp de todas las actividades; pop -> pop; qe -> aft; afeall -> lambdaf.
    # aoall(AGR) y aoreg se SUMAN en % en GEMPACK (ao = aoall + aosec + aoreg), o sea
    # que en niveles se MULTIPLICAN: axp(AGR) = 1,3194 x 0,89 en USA.
    "ME9B": ("climatechange.prm", _me9(land=(-0.93, 4.4), ao_agr=0.0, afe=None)),
    "ME9C": ("climatechange.prm", _me9(land=(10.07, 15.4), ao_agr=-11.0, afe=None)),
    "ME9D": (
        "climatechange.prm",
        _me9(land=(10.07, 15.4), ao_agr=-11.0, afe=(-0.73, -2.0)),
    ),
}

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


def level(m, p, name: str, idx: tuple, kind: str, pct: float) -> float:
    """Valor en niveles de la celda 'shock' para el shock GEMPACK ``pct``."""
    from pyomo.environ import value

    chk = float(value(getattr(m, name)[(*idx, "check")]))
    if kind == "pct":
        return chk * (1 + pct / 100)
    if kind == "power":
        return (1 + chk) * (1 + pct / 100) - 1
    if kind == "power_kappa":
        return 1 - (1 - chk) / (1 + pct / 100)
    if kind == "power_fct":
        from equilibria.blocks.gtap import _derived_params as dp

        fs = dp.fctts_data(p, p.sets).get(idx, 0.0)
        return (1 + fs + chk) * (1 + pct / 100) - 1 - fs
    raise ValueError(f"kind desconocido: {kind!r}")


def solve_exercise(exp: str, har: Path):
    """Construye, aplica los shocks del ejercicio y resuelve base/check/shock."""
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.instruments import fix_instrument_shock

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
    for name, idx, kind, pct in shocks:
        fix_instrument_shock(m, name, idx, value=level(m, p, name, idx, kind, pct))
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
    )
    return m, p, {k: int(res[k]["code"]) for k in res}


def _strip(s):
    return (
        s[2:]
        if isinstance(s, str) and len(s) > 2 and s[1] == "_" and s[0] in "acfr"
        else s
    )


def compare(m, p, gdx: Path) -> dict:
    """Todas las celdas de check y shock contra GAMS (misma logica que los gates).

    Los instrumentos del ShockBlock son datos de entrada, no resultados: no entran al %,
    se cuentan aparte (instr_cells / instr_bad) para ver que el shock es el de GAMS.
    """
    from _diff_core import gams_levels, list_populated_vars, split_t
    from pyomo.environ import value

    instruments = getattr(m, "_exogenous_instruments", frozenset())
    out = {}
    for period in ("check", "shock"):
        tot = 0
        instr_cells = 0
        instr_bad = []
        match = dict.fromkeys(TOLS, 0)
        worst = []
        for vn in list_populated_vars(gdx):
            if vn.lower() in SKIP or vn.lower() in RF:
                continue
            try:
                g = gams_levels(gdx, vn)
            except Exception:
                continue
            pv = getattr(m, ALIAS.get(vn, vn), None) or getattr(m, vn.lower(), None)
            if pv is None:
                continue
            for fk, gval in g.items():
                body, t = split_t(fk)
                if t != period:
                    continue
                st = tuple(_strip(x) for x in body)
                # p(r,a,i) fuera de la diagonal: makb=0, la celda no existe.
                if vn == "p" and len(st) == 3 and st[1] != st[2]:
                    mk = p.benchmark.makb
                    if not float(
                        mk.get(st, 0) or mk.get((st[2], st[1], st[0]), 0) or 0
                    ):
                        continue
                try:
                    val = float(value(pv[(*st, period) if st else (period,)]))
                except Exception:
                    continue
                d = abs(val - gval)
                rel = d / abs(gval) if abs(gval) > 1e-12 else (0.0 if d < 1e-6 else 9e9)
                if pv.local_name in instruments:
                    instr_cells += 1
                    if d > 1e-6 and rel > TOLS[0]:
                        instr_bad.append((vn, list(st), val, gval))
                    continue
                tot += 1
                ok = [d <= 1e-6 or rel <= t for t in TOLS]
                for t, o in zip(TOLS, ok, strict=True):
                    match[t] += o
                if not ok[0]:
                    worst.append((round(100 * rel, 4), vn, list(st)))
        worst.sort(reverse=True)
        out[period] = {
            "cells": tot,
            "match_pct": {
                f"{t:.1%}": round(100 * match[t] / max(tot, 1), 2) for t in TOLS
            },
            "worst": worst[:10],
            "instr_cells": instr_cells,
            "instr_bad": instr_bad,
        }
    return out


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
