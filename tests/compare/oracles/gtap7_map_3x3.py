"""Reproduce el oraculo del ejemplo 15 (tests/compare/_oracle.py) SIN equilibria.compare.

El mapeo GTAP de equilibria (#132): equilibria resuelto contra GEMPACK en NIVELES,
pre y post. A = ``tests/compare/fixtures/eq_gtap7_3x3_levels.json.gz``
(``make_eq_gtap7_levels.py``: ``_solve_shock("gtap7_3x3", 0)``, capFlex, el cierre del
fixture de GEMPACK); B = ``sl4levels6_gtap7_3x3_tm10.har`` (LEVB = pre, LEVA = post).
Periodos de A: check -> pre, shock -> post. Todo a mano, con numpy:

- cantidades de A / (1e-6 [inScale] * xscale[r, a] si la variable es por actividad);
- qds = xds - xda[r, c, 'tmg']: xds incluye la oferta de margenes (VST), qds no
  (medido el 2026-10-21: xds - qds = VST en las 3 regiones, a < 2 unidades);
- VDFB = pd[r, c] * xda[r, c, a] / (1e-6 * xscale[r, a]) contra VDFB de basedata.har
  (pre) y de updated_gtap7_3x3_tm10.har (post);
- pe: GEMPACK no trae NatRes (factor especifico, ENDF); esas celdas no se miden;
- las demandas de hogar/gobierno/inversion/margenes son xda/xma con el agente fijo
  (hhd/gov/inv/tmg); qfd/qfm solo las actividades;
- qfe = xf * pf[check] (GEMPACK mide la cantidad al precio del agente en la base);
- qxw = sum_d xw * pefob[check] + xda[r,c,'tmg'] (pesos FOB de la base, mas margenes);
- tpd = 1 + dintx[r,c,'hhd']; pr = pd / pmt.
Las 18 variables restantes del .har son indices encadenados (precio, volumen,
terminos de intercambio): sin forma en niveles, no se miden.

``d = |A - B| / max(|B|, 1e-6)``. Lee los HAR con ``babel.har.reader``.

Uso (desde la raiz): ``uv run python tests/compare/oracles/gtap7_map_3x3.py``
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from equilibria.babel.har.reader import read_har

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "tests" / "fixtures" / "gtap7_gempack"
EQ = ROOT / "tests" / "compare" / "fixtures" / "eq_gtap7_3x3_levels.json.gz"
THRESHOLDS = (1e-3, 5e-3, 1e-2)
PERIODS = {"pre": "check", "post": "shock"}
INSCALE = 1e-6
GEMPACK_VARS = (
    "qo",
    "rore",
    "y",
    "pinv",
    "qinv",
    "pva",
    "qds",
    "pe",
    "pds",
    "pfactor",
    "pfactwld",
    "pgdp",
    "pms",
    "pxw",
    "qc",
    "qes",
    "qfd",
    "qfm",
    "qgd",
    "qgdp",
    "qgm",
    "qid",
    "qim",
    "qms",
    "qpd",
    "qpm",
    "qst",
    "qxs",
    "qfe",
    "qxw",
    "tpd",
    "pr",
)


def _gempack_levels() -> dict[str, dict[tuple[str, tuple[str, ...]], float]]:
    """{var: {(periodo, indices): nivel}} desde las cabeceras LEVA/LEVB crudas."""
    out = _leva_levb(FIX / "sl4levels6_gtap7_3x3_tm10.har")
    _add_vdfb(
        out,
        ROOT / "data/gtap7/gtap7_3x3/basedata.har",
        ROOT / "tests/fixtures/gtap7_gempack/updated_gtap7_3x3_tm10.har",
    )
    return out


def _leva_levb(path: Path) -> dict[str, dict[tuple[str, tuple[str, ...]], float]]:
    har = read_har(str(path))
    out: dict[str, dict] = {}
    for ha in har.values():
        name = (ha.long_name or "").split("#", 1)[0].strip()
        if not name or name.startswith("variables with NO level"):
            continue
        period = "pre" if name.startswith("LEVB ") else "post"
        name = name.removeprefix("LEVB ").strip()
        if name not in GEMPACK_VARS:
            continue
        arr = np.asarray(ha.array, dtype=float)
        cells = out.setdefault(name, {})
        if not ha.set_elements:  # escalar (pfactwld)
            cells[(period, ())] = float(arr.ravel()[0])
            continue
        for idx in np.ndindex(arr.shape):
            key = tuple(ha.set_elements[d][i] for d, i in enumerate(idx))
            cells[(period, key)] = float(arr[idx])
    return out


def _add_vdfb(out: dict, basedata: Path, updated: Path) -> None:
    """VDFB[c, a, r] de basedata (pre) y del .har actualizado (post)."""
    for period, f in (("pre", basedata), ("post", updated)):
        ha = read_har(str(f))["VDFB"]
        arr = np.asarray(ha.array, dtype=float)
        for idx in np.ndindex(arr.shape):
            key = tuple(ha.set_elements[d][i] for d, i in enumerate(idx))
            out.setdefault("VDFB", {})[(period, key)] = float(arr[idx])


def load_eq(path: Path) -> tuple[dict, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    eq = {v: {tuple(k): x for k, x in rows} for v, rows in raw["vars"].items()}
    xs = {tuple(k): x for k, x in raw["params"]["xscale"]}
    return eq, xs


def rows_of(eq: dict, xs: dict, B: dict) -> list[tuple]:
    """(variable, periodo, indices en orden de A, d) de cada celda que B trae."""
    rows = []  # (variable, periodo, indices en orden de A, d)

    def add(name, period, a_idx, a, b):
        rows.append((name, period, a_idx, abs(a - b) / max(abs(b), 1e-6)))

    X = lambda r, a: INSCALE * xs[(r, a)]  # noqa: E731
    acts = {a for (_, a, _) in eq["xp"]}
    regs = sorted({r for (r, _) in eq["rore"]})
    # gempack -> (nombre, indices de A desde la clave de B, nivel de A)
    spec = {
        "qo": lambda k, t: ((k[1], k[0]), eq["xp"][(k[1], k[0], t)] / X(k[1], k[0])),
        "rore": lambda k, t: (k, eq["rore"][(k[0], t)]),
        "y": lambda k, t: (k, eq["regy"][(k[0], t)] / INSCALE),
        "pinv": lambda k, t: (k, eq["pi"][(k[0], t)]),
        "qinv": lambda k, t: (k, eq["xiagg"][(k[0], t)] / INSCALE),
        "pva": lambda k, t: ((k[1], k[0]), eq["pva"][(k[1], k[0], t)]),
        "qds": lambda k, t: (
            (k[1], k[0]),
            (eq["xds"][(k[1], k[0], t)] - eq["xda"].get((k[1], k[0], "tmg", t), 0.0))
            / INSCALE,
        ),
        "pe": lambda k, t: ((k[1], k[0]), eq["pft"][(k[1], k[0], t)]),
        "pds": lambda k, t: ((k[1], k[0]), eq["pd"][(k[1], k[0], t)]),
        "pfactor": lambda k, t: (k, eq["pfact"][(k[0], t)]),
        "pfactwld": lambda k, t: ((), eq["pwfact"][(t,)]),
        "pgdp": lambda k, t: (k, eq["pgdpmp"][(k[0], t)]),
        "pms": lambda k, t: ((k[1], k[0]), eq["pmt"][(k[1], k[0], t)]),
        "pxw": lambda k, t: ((k[1], k[0]), eq["pet"][(k[1], k[0], t)]),
        "qc": lambda k, t: ((k[1], k[0]), eq["xs"][(k[1], k[0], t)] / INSCALE),
        "qes": lambda k, t: (
            (k[2], k[0], k[1]),
            eq["xf"][(k[2], k[0], k[1], t)] / X(k[2], k[1]),
        ),
        "qfd": lambda k, t: (
            (k[2], k[0], k[1]),
            eq["xda"][(k[2], k[0], k[1], t)] / X(k[2], k[1]),
        ),
        "qfm": lambda k, t: (
            (k[2], k[0], k[1]),
            eq["xma"][(k[2], k[0], k[1], t)] / X(k[2], k[1]),
        ),
        "qgdp": lambda k, t: (k, eq["rgdpmp"][(k[0], t)] / INSCALE),
        "qms": lambda k, t: ((k[1], k[0]), eq["xmt"][(k[1], k[0], t)] / INSCALE),
        "qxs": lambda k, t: (
            (k[1], k[0], k[2]),
            eq["xw"][(k[1], k[0], k[2], t)] / INSCALE,
        ),
        "qfe": lambda k, t: (
            (k[2], k[0], k[1]),
            eq["xf"][(k[2], k[0], k[1], t)]
            * eq["pf"][(k[2], k[0], k[1], "check")]
            / X(k[2], k[1]),
        ),
        "qxw": lambda k, t: (
            (k[1], k[0]),
            (
                sum(
                    eq["xw"][(k[1], k[0], d, t)] * eq["pefob"][(k[1], k[0], d, "check")]
                    for d in regs
                    if (k[1], k[0], d, t) in eq["xw"]
                )
                + eq["xda"].get((k[1], k[0], "tmg", t), 0.0)
            )
            / INSCALE,
        ),
        "tpd": lambda k, t: ((k[1], k[0]), 1.0 + eq["dintx"][(k[1], k[0], "hhd", t)]),
        "pr": lambda k, t: (
            (k[1], k[0]),
            eq["pd"][(k[1], k[0], t)] / eq["pmt"][(k[1], k[0], t)],
        ),
    }
    for g, var, agent in (
        ("qpd", "xda", "hhd"),
        ("qpm", "xma", "hhd"),
        ("qgd", "xda", "gov"),
        ("qgm", "xma", "gov"),
        ("qid", "xda", "inv"),
        ("qim", "xma", "inv"),
        ("qst", "xda", "tmg"),
    ):
        spec[g] = (
            lambda var, agent: lambda k, t: (
                (k[1], k[0]),
                eq[var][(k[1], k[0], agent, t)] / INSCALE,
            )
        )(var, agent)

    for period, t in PERIODS.items():
        for g, f in spec.items():
            for (p, k), b in B.get(g, {}).items():
                if p != period:
                    continue
                if g == "pe" and k[0] == "NatRes":
                    continue
                if g in ("qfd", "qfm") and k[1] not in acts:
                    continue
                a_idx, a = f(k, t)
                add(g, period, a_idx, a, b)
        for (p, (c, a, r)), b in B["VDFB"].items():
            if p == period:
                v = eq["pd"][(r, c, t)] * eq["xda"][(r, c, a, t)]
                add("vdfb", period, (r, c, a), v / X(r, a), b)
    return rows


def report(eq: dict, rows: list[tuple]) -> None:
    print(f"celdas={len(rows)}")
    for t in THRESHOLDS:
        print(f"  <= {t:g}: {sum(row[3] <= t for row in rows)}")
    for name in dict.fromkeys(row[0] for row in rows):
        mine = [row for row in rows if row[0] == name]
        w = max(mine, key=lambda row: row[3])
        print(f"  {name}: {len(mine)} celdas, max d = {w[3]:.6g} en {w[1]} {w[2]}")
    worst = max(rows, key=lambda row: row[3])
    print(f"  maximo: {worst}")
    natres = {(r, e) for r, e, _ in eq["pft"] if e == "NatRes"}
    print(f"  pft NatRes en A (pre+post): {2 * len(natres)} celdas sin pareja en B")


def main() -> None:
    eq, xs = load_eq(EQ)
    report(eq, rows_of(eq, xs, _gempack_levels()))


if __name__ == "__main__":
    main()
