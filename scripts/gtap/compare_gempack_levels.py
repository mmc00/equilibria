"""equilibria contra GEMPACK en NIVELES con el mapeo GTAP (``maps/gtap7.yaml``, #132).

Arma las dos variables que el mapeo pide y ninguna fuente trae tal cual:

- lado equilibria: ``xds_net[r,c] = xds[r,c] - xda[r,c,'tmg']`` (``xds`` incluye la
  oferta de margenes VST y ``qds`` de GEMPACK no) y ``vdfb[r,c,a] = pd[r,c] *
  xda[r,c,a]`` para las actividades ``a`` (las de ``xp``);
- lado GEMPACK: ``VDFB[c,a,r]`` de ``basedata.har`` (pre) y del ``.har`` actualizado
  de la corrida (post).

Uso (desde la raiz)::

    uv run python scripts/gtap/compare_gempack_levels.py \\
        --eq tests/compare/fixtures/eq_gtap7_3x3_levels.json.gz \\
        --har tests/fixtures/gtap7_gempack/sl4levels6_gtap7_3x3_tm10.har \\
        --basedata datasets/gtap7_3x3/basedata.har \\
        --updated tests/fixtures/gtap7_gempack/updated_gtap7_3x3_tm10.har

``--eq`` tiene la forma de ``tests/compare/fixtures/make_eq_gtap7_levels.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
from collections.abc import Mapping as AbcMapping
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.babel.har.reader import read_har
from equilibria.compare import Mapping, Report, Solution, compare, from_dict, from_har

Levels = AbcMapping[str, AbcMapping[tuple[str, ...], float]]
MARGIN_AGENT = "tmg"


def equilibria_side(levels: Levels, xscale: AbcMapping[Any, float]) -> Solution:
    """``levels[var][(indices..., periodo)] = nivel`` del modelo de equilibria, mas
    ``xds_net`` y ``vdfb``. ``xscale[(r, a)]`` va como parametro de la fuente."""
    xds, xda, pd = levels["xds"], levels["xda"], levels["pd"]
    xds_net = {
        (r, c, t): x - xda.get((r, c, MARGIN_AGENT, t), 0.0)
        for (r, c, t), x in xds.items()
    }
    # VDFB es solo de las actividades (las de xp); xda trae ademas hhd, gov, inv, tmg.
    acts = {a for (_, a, _) in levels["xp"]}
    vdfb = {
        (r, c, a, t): pd[(r, c, t)] * q for (r, c, a, t), q in xda.items() if a in acts
    }
    data = {**levels, "xds_net": xds_net, "vdfb": vdfb}
    return from_dict(
        data, period_dim=-1, origin="equilibria", params={"xscale": dict(xscale)}
    )


def gempack_side(har: Path, basedata: Path, updated: Path) -> Solution:
    """El ``.har`` de niveles de GEMPACK mas ``VDFB`` (pre: basedata; post: updated)."""
    sol = from_har(har)
    vdfb: dict[tuple[str, tuple[str, ...]], float] = {}
    for period, path in (("pre", basedata), ("post", updated)):
        ha = read_har(str(path))["VDFB"]
        arr = np.asarray(ha.array, dtype=float)
        for idx in np.ndindex(arr.shape):
            key = tuple(ha.set_elements[d][i] for d, i in enumerate(idx))
            vdfb[(period, key)] = float(arr[idx])
    return dataclasses.replace(
        sol,
        levels={**sol.levels, "VDFB": vdfb},
        ndim={**sol.ndim, "VDFB": 3},
    )


def load_eq_json(path: Path) -> tuple[dict, dict]:
    """(levels, xscale) desde el JSON de ``make_eq_gtap7_levels.py``."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    levels = {v: {tuple(k): x for k, x in rows} for v, rows in raw["vars"].items()}
    xscale = {tuple(k): x for k, x in raw["params"]["xscale"]}
    return levels, xscale


def run(eq: Path, har: Path, basedata: Path, updated: Path) -> Report:
    levels, xscale = load_eq_json(eq)
    return compare(
        equilibria_side(levels, xscale),
        gempack_side(har, basedata, updated),
        Mapping.builtin("gtap7"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eq", type=Path, required=True)
    ap.add_argument("--har", type=Path, required=True)
    ap.add_argument("--basedata", type=Path, required=True)
    ap.add_argument("--updated", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    r = run(a.eq, a.har, a.basedata, a.updated)
    for name, s in r.summary.items():
        print(
            f"{name:6s} celdas={s['cells']:4d} <=0,1%={s['pct_within'][1e-3]:6.2f}% "
            f"max={s['max']:.3g} en {s['max_cell']}"
        )
    o = r.overall
    print(f"TOTAL  celdas={o['cells']:4d} <=0,1%={o['pct_within'][1e-3]:6.2f}%")
    for e in r.exclusions:
        print(f"excluidas: {e.variable} {e.where} ({e.cells} celdas): {e.why}")
    for var, why in r.excluded.items():
        print(f"excluida: {var}: {why}")
    if a.out:
        a.out.write_text(r.to_json())


if __name__ == "__main__":
    main()
