"""equilibria contra GEMPACK en NIVELES con el mapeo GTAP (``maps/gtap7.yaml``).

Arma las variables que el mapeo pide y ninguna fuente trae tal cual (formulas
medidas contra GEMPACK en el 3x3: ejemplo 15 de ``tests/compare/_oracle.py``).
Lado equilibria (``check`` es el periodo base del modelo):

- ``xds_net[r,c] = xds[r,c] - xda[r,c,'tmg']``: ``xds`` incluye la oferta de
  margenes (VST) y ``qds`` no;
- ``xda_act``/``xma_act[r,c,a]``: ``xda``/``xma`` solo para las actividades (las de
  ``xp``); ``xda_hhd``, ``xma_hhd``, ``xda_gov``, ``xma_gov``, ``xda_inv``,
  ``xma_inv``, ``xda_tmg[r,c]``: el agente fijo (``qpd``, ``qgm``, ``qst``...);
  ``xda_tmg`` solo donde hay oferta de margenes en la base (MARG);
- ``xf_pf0[r,e,a] = xf * pf[check]``: GEMPACK mide ``qfe`` al precio del agente en
  la base;
- ``xw_fob[r,c] = sum_d xw[r,c,d] * pefob[r,c,d,check] + xda[r,c,'tmg']``: ``qxw``
  (pesos FOB de la base, mas margenes);
- ``tpd[r,c] = 1 + dintx[r,c,'hhd']``; ``pd_pmt[r,c] = pd / pmt`` (``pr``);
- ``vdfb[r,c,a] = pd[r,c] * xda[r,c,a]`` para las actividades.

Lado GEMPACK: ``VDFB[c,a,r]`` de ``basedata.har`` (pre) y del ``.har`` actualizado
de la corrida (post).

Uso (desde la raiz). Con niveles de equilibria ya guardados::

    uv run python scripts/gtap/compare_gempack_levels.py \\
        --eq tests/compare/fixtures/eq_gtap7_3x3_levels.json.gz \\
        --har tests/fixtures/gtap7_gempack/sl4levels6_gtap7_3x3_tm10.har \\
        --basedata datasets/gtap7_3x3/basedata.har \\
        --updated tests/fixtures/gtap7_gempack/updated_gtap7_3x3_tm10.har

o resolviendo equilibria en el momento (necesita PATH; ``--solver newton`` es el
Newton-TR + MUMPS del 20x41, que necesita pymumps)::

    PYTHONPATH=. uv run python scripts/gtap/compare_gempack_levels.py \\
        --dataset gtap7_20x41 --closure capFix --solver newton --save-eq eq.json.gz \\
        --har tests/fixtures/gtap7_gempack/sl4levels6_gtap7_20x41_tm10_s8-16-32_capfix.har \\
        --basedata datasets/gtap7_20x41/basedata.har \\
        --updated tests/fixtures/gtap7_gempack/updated_gtap7_20x41_tm10_s8-16-32_capfix.har \\
        --out reporte.json

``--eq``/``--save-eq`` tienen la forma de ``tests/compare/fixtures/make_eq_gtap7_levels.py``.
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

ROOT = Path(__file__).resolve().parents[2]
Levels = AbcMapping[str, AbcMapping[tuple[str, ...], float]]
MARGIN_AGENT = "tmg"
FINAL_AGENTS = ("hhd", "gov", "inv", "tmg")
BASE = "check"  # el periodo base del modelo de equilibria (el pre de GEMPACK)


def equilibria_side(levels: Levels, xscale: AbcMapping[Any, float]) -> Solution:
    """``levels[var][(indices..., periodo)] = nivel`` del modelo de equilibria, mas
    las variables armadas (ver el docstring del modulo). ``xscale[(r, a)]`` va como
    parametro de la fuente."""
    xds, xda, xma, pd = levels["xds"], levels["xda"], levels["xma"], levels["pd"]
    acts = {a for (_, a, _) in levels["xp"]}
    regs = {r for (r, _) in levels["rore"]}
    data: dict[str, AbcMapping[tuple[str, ...], float]] = dict(levels)
    data["xds_net"] = {
        (r, c, t): x - xda.get((r, c, MARGIN_AGENT, t), 0.0)
        for (r, c, t), x in xds.items()
    }
    for var, table in (("xda", xda), ("xma", xma)):
        data[f"{var}_act"] = {k: x for k, x in table.items() if k[2] in acts}
        for agent in FINAL_AGENTS:
            data[f"{var}_{agent}"] = {
                (r, c, t): x for (r, c, a, t), x in table.items() if a == agent
            }
    # qst es solo de los productos de margenes (MARG): los que tienen oferta de
    # margenes en la base; el modelo guarda un 0 para los demas.
    data["xda_tmg"] = {
        (r, c, t): x
        for (r, c, t), x in data["xda_tmg"].items()
        if xda[(r, c, MARGIN_AGENT, BASE)] != 0.0
    }
    data["xf_pf0"] = {
        (r, e, a, t): x * levels["pf"][(r, e, a, BASE)]
        for (r, e, a, t), x in levels["xf"].items()
    }
    xw, pefob = levels["xw"], levels["pefob"]
    data["xw_fob"] = {
        (r, c, t): sum(
            xw[(r, c, d, t)] * pefob[(r, c, d, BASE)]
            for d in regs
            if (r, c, d, t) in xw
        )
        + xda.get((r, c, MARGIN_AGENT, t), 0.0)
        for (r, c, t) in xds
    }
    data["tpd"] = {
        (r, c, t): 1.0 + x for (r, c, a, t), x in levels["dintx"].items() if a == "hhd"
    }
    data["pd_pmt"] = {k: x / levels["pmt"][k] for k, x in pd.items()}
    data["vdfb"] = {
        (r, c, a, t): pd[(r, c, t)] * q for (r, c, a, t), q in xda.items() if a in acts
    }
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


# Variables del modelo que usan el mapeo y equilibria_side.
EQ_VARS = (
    "xp", "rore", "regy", "pi", "xiagg", "pva", "xds", "xda", "pd", "pft",
    "pfact", "pwfact", "pgdpmp", "pmt", "pet", "xs", "xf", "pf", "xma", "rgdpmp",
    "xmt", "xw", "pefob", "dintx",
)  # fmt: skip

# Newton-TR + MUMPS: el solver del 20x41 en el paper (IPOPT se queda sin memoria).
NEWTON_MUMPS = {
    "EQUILIBRIA_GTAP_SOLVE_NLP": "1",
    "EQUILIBRIA_GTAP_SOLVER": "scipy_newton_tr",
    "EQUILIBRIA_GTAP_TR_LINSOLVE": "mumps",
    "EQUILIBRIA_GTAP_NLP_NO_JACSCALE": "1",
    "EQUILIBRIA_GTAP_TR_GATE": "1",
    "EQUILIBRIA_GTAP_TR_FTOL": "1e-7",
    "EQUILIBRIA_GTAP_SCIPY_MAXITER": "300",
    "EQUILIBRIA_GTAP_TR_DELTA0": "10.0",
    "EQUILIBRIA_GTAP_GMIN": "1e-9",
    "EQUILIBRIA_GTAP_TR_RELTOL": "1e-6",
}


def model_levels(m: Any) -> dict[str, dict[tuple[str, ...], float]]:
    """``{var: {(indices..., periodo): nivel}}`` de las ``EQ_VARS`` de un modelo
    resuelto (``pyomo.value``)."""
    from pyomo.environ import value

    return {
        v: {
            (i if isinstance(i, tuple) else (i,)): float(value(c))
            for i, c in getattr(m, v).items()
        }
        for v in EQ_VARS
    }


def dataset_xscale(dataset: str, va_subsidy_basis: str = "gempack") -> dict:
    """``xscale[(r, aa)]`` con que el modelo escala las cantidades por actividad.

    Hay que calcularlo con la MISMA base de subsidios al valor agregado que usa el
    modelo (``_solve_shock`` construye con ``va_subsidy_basis="gempack"`` y
    ``build_block_model`` la fija en los parametros): cambia ``xp`` de la base y, en
    sectores chicos, la potencia de 10 de la escala. Con la base por defecto el
    20x41 salia 10 veces mayor en CHE CropsNe y MEX Restofagri."""
    from equilibria.blocks.gtap._derived_params import xscale_data
    from equilibria.templates.gtap import GTAPParameters

    d = ROOT / "datasets" / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    p.set_va_subsidy_basis(va_subsidy_basis)
    return dict(xscale_data(p, p.sets))


def solve(dataset: str, closure: str, solver: str) -> tuple[dict, dict]:
    """Resuelve equilibria (base -> check -> shock, tm uniforme 10) con el cierre del
    fixture de GEMPACK y devuelve (levels, xscale)."""
    import os
    import sys

    from equilibria._local_refs import path_capi_src

    if solver == "newton":
        os.environ.update(NEWTON_MUMPS)
    sys.path.insert(0, str(path_capi_src()))
    sys.path.insert(0, str(ROOT))
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    m, code = _solve_shock(dataset, 0, closure, solve_nlp=solver in ("newton", "ipopt"))
    if code != 1:
        raise SystemExit(f"{dataset}: el solve termino con code={code}")
    return model_levels(m), dataset_xscale(dataset)


def save_eq_json(path: Path, levels: dict, xscale: dict) -> None:
    out = {
        "vars": {v: [[list(k), x] for k, x in t.items()] for v, t in levels.items()},
        "params": {"xscale": [[list(k), x] for k, x in xscale.items()]},
        "code": 1,
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(out, fh)


def load_eq_json(path: Path) -> tuple[dict, dict]:
    """(levels, xscale) desde el JSON de ``make_eq_gtap7_levels.py``."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    levels = {v: {tuple(k): x for k, x in rows} for v, rows in raw["vars"].items()}
    xscale = {tuple(k): x for k, x in raw["params"]["xscale"]}
    return levels, xscale


def run(levels: dict, xscale: dict, har: Path, basedata: Path, updated: Path) -> Report:
    return compare(
        equilibria_side(levels, xscale),
        gempack_side(har, basedata, updated),
        Mapping.builtin("gtap7"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--eq", type=Path, help="niveles de equilibria ya guardados")
    src.add_argument("--dataset", help="resolver equilibria sobre datasets/<dataset>")
    ap.add_argument("--closure", default="capFlex", choices=["capFlex", "capFix"])
    ap.add_argument("--solver", default="path", choices=["path", "ipopt", "newton"])
    ap.add_argument("--save-eq", type=Path, help="guardar los niveles resueltos")
    ap.add_argument("--har", type=Path, required=True)
    ap.add_argument("--basedata", type=Path, required=True)
    ap.add_argument("--updated", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.eq:
        levels, xscale = load_eq_json(a.eq)
    else:
        levels, xscale = solve(a.dataset, a.closure, a.solver)
        if a.save_eq:
            save_eq_json(a.save_eq, levels, xscale)
    r = run(levels, xscale, a.har, a.basedata, a.updated)
    for name, s in r.summary.items():
        print(
            f"{name:9s} celdas={s['cells']:6d} <=0,1%={s['pct_within'][1e-3]:6.2f}% "
            f"max={s['max']:.3g} en {s['max_cell']}"
        )
    o = r.overall
    print(
        f"TOTAL     celdas={o['cells']:6d} <=0,1%={o['pct_within'][1e-3]:6.2f}% "
        f"<=0,5%={o['pct_within'][5e-3]:6.2f}% <=1%={o['pct_within'][1e-2]:6.2f}% "
        f"max={o['max']:.3g} en {o['max_variable']} {o['max_cell']}"
    )
    print(f"sin pareja: {len(r.unmatched)}")
    for e in r.exclusions:
        print(f"excluidas: {e.variable} {e.where} ({e.cells} celdas): {e.why}")
    for var, why in r.excluded.items():
        print(f"excluida: {var}: {why}")
    if a.out:
        a.out.write_text(r.to_json())


if __name__ == "__main__":
    main()
