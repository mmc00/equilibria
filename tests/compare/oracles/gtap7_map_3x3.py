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
- pe: GEMPACK no trae NatRes (factor especifico, ENDF); esas celdas no se miden.

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
GEMPACK_VARS = ("qo", "rore", "y", "pinv", "qinv", "pva", "qds", "pe")


def _gempack_levels() -> dict[str, dict[tuple[str, tuple[str, ...]], float]]:
    """{var: {(periodo, indices): nivel}} desde las cabeceras LEVA/LEVB crudas."""
    har = read_har(str(FIX / "sl4levels6_gtap7_3x3_tm10.har"))
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
        for idx in np.ndindex(arr.shape):
            key = tuple(ha.set_elements[d][i] for d, i in enumerate(idx))
            cells[(period, key)] = float(arr[idx])
    for period, f in (
        ("pre", "datasets/gtap7_3x3/basedata.har"),
        ("post", "tests/fixtures/gtap7_gempack/updated_gtap7_3x3_tm10.har"),
    ):
        ha = read_har(str(ROOT / f))["VDFB"]
        arr = np.asarray(ha.array, dtype=float)
        for idx in np.ndindex(arr.shape):
            key = tuple(ha.set_elements[d][i] for d, i in enumerate(idx))
            out.setdefault("VDFB", {})[(period, key)] = float(arr[idx])
    return out


def main() -> None:
    with gzip.open(EQ, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    eq = {v: {tuple(k): x for k, x in rows} for v, rows in raw["vars"].items()}
    xs = {tuple(k): x for k, x in raw["params"]["xscale"]}
    B = _gempack_levels()

    rows = []  # (variable, periodo, indices en orden de A, d)

    def add(name, period, a_idx, a, b):
        rows.append((name, period, a_idx, abs(a - b) / max(abs(b), 1e-6)))

    for period, t in PERIODS.items():
        for (p, (c, a)), b in B["qo"].items():
            if p == period:
                add(
                    "qo",
                    period,
                    (a, c),
                    eq["xp"][(a, c, t)] / (INSCALE * xs[(a, c)]),
                    b,
                )
        for name, var, scale in (
            ("rore", "rore", 1.0),
            ("y", "regy", INSCALE),
            ("pinv", "pi", 1.0),
            ("qinv", "xiagg", INSCALE),
        ):
            for (p, (r,)), b in B[name].items():
                if p == period:
                    add(name, period, (r,), eq[var][(r, t)] / scale, b)
        for (p, (a, r)), b in B["pva"].items():
            if p == period:
                add("pva", period, (r, a), eq["pva"][(r, a, t)], b)
        for (p, (c, r)), b in B["qds"].items():
            if p == period:
                net = eq["xds"][(r, c, t)] - eq["xda"].get((r, c, "tmg", t), 0.0)
                add("qds", period, (r, c), net / INSCALE, b)
        for (p, (e, r)), b in B["pe"].items():
            if p == period:
                add("pe", period, (r, e), eq["pft"][(r, e, t)], b)
        for (p, (c, a, r)), b in B["VDFB"].items():
            if p == period:
                v = eq["pd"][(r, c, t)] * eq["xda"][(r, c, a, t)]
                add("vdfb", period, (r, c, a), v / (INSCALE * xs[(r, a)]), b)

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


if __name__ == "__main__":
    main()
