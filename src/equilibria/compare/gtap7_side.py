"""The equilibria side of the GTAP map (``maps/gtap7.yaml``): levels in GEMPACK's terms.

Builds the variables the map asks for that no source carries as is (formulas measured
against GEMPACK on the 3x3: example 15 of ``tests/compare/_oracle.py``). ``check`` is
the model's base period:

- ``xds_net[r,c] = xds[r,c] - xda[r,c,'tmg']``: ``xds`` includes the margin supply
  (VST) and ``qds`` does not;
- ``xda_act``/``xma_act[r,c,a]``: ``xda``/``xma`` for the activities only (those of
  ``xp``); ``xda_hhd``, ``xma_hhd``, ``xda_gov``, ``xma_gov``, ``xda_inv``,
  ``xma_inv``, ``xda_tmg[r,c]``: the fixed agent (``qpd``, ``qgm``, ``qst``...);
  ``xda_tmg`` only where the base has margin supply (MARG);
- ``xf_pf0[r,e,a] = xf * pf[check]``: GEMPACK measures ``qfe`` at the agent's base
  price;
- ``xw_fob[r,c] = sum_d xw[r,c,d] * pefob[r,c,d,check] + xda[r,c,'tmg']``: ``qxw``
  (base FOB weights, plus margins);
- ``tpd[r,c] = 1 + dintx[r,c,'hhd']``; ``pd_pmt[r,c] = pd / pmt`` (``pr``);
- ``vdfb[r,c,a] = pd[r,c] * xda[r,c,a]`` for the activities.

It lived in ``scripts/gtap/compare_gempack_levels.py`` until MAR-249, which moved it
here so installed code does not load a repo script; the script re-exports these names.
"""

from __future__ import annotations

from collections.abc import Collection
from collections.abc import Mapping as AbcMapping
from typing import Any

from equilibria.compare import Mapping, Solution, from_dict
from equilibria.compare.mapping import VariableMap

Levels = AbcMapping[str, AbcMapping[tuple[str, ...], float]]
Pct = dict[str, dict[tuple[str, ...], float]]
MARGIN_AGENT = "tmg"
FINAL_AGENTS = ("hhd", "gov", "inv", "tmg")
BASE = "check"  # the base period of the equilibria model (GEMPACK's pre)

# Model variables the map and equilibria_side use.
EQ_VARS = (
    "xp", "rore", "regy", "pi", "xiagg", "pva", "xds", "xda", "pd", "pft",
    "pfact", "pwfact", "pgdpmp", "pmt", "pet", "xs", "xf", "pf", "xma", "rgdpmp",
    "xmt", "xw", "pefob", "dintx",
)  # fmt: skip


def equilibria_side(
    levels: Levels, xscale: AbcMapping[Any, float], *, linear: bool = False
) -> Solution:
    """``levels[var][(indices..., period)] = level`` of the equilibria model, plus the
    built variables (see the module docstring). ``xscale[(r, a)]`` goes in as a
    parameter of the source.

    ``linear``: the levels are a Johansen solution (``shock_method="johansen"``).
    There a product or a ratio moves by the first-order change of its factors, as
    GEMPACK reports it, not by the product of the new levels; the other derived
    variables are already linear in the levels."""
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
    # qst is only for the margin commodities (MARG): those with margin supply in the
    # base; the model stores a 0 for the others.
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
    pmt = levels["pmt"]
    data["pd_pmt"] = {k: x / pmt[k] for k, x in pd.items()}
    data["vdfb"] = {
        (r, c, a, t): pd[(r, c, t)] * q for (r, c, a, t), q in xda.items() if a in acts
    }
    if linear:
        # d(p/m) = dp/m0 - p0·dm/m0²  and  d(p·q) = q0·dp + p0·dq, from check.
        for (r, c, t), x in pd.items():
            p0, m0 = pd[(r, c, BASE)], pmt[(r, c, BASE)]
            dp, dm = x - p0, pmt[(r, c, t)] - m0
            data["pd_pmt"][(r, c, t)] = p0 / m0 + dp / m0 - p0 * dm / m0**2
        for r, c, a, t in data["vdfb"]:
            p0, q0 = pd[(r, c, BASE)], xda[(r, c, a, BASE)]
            dp, dq = pd[(r, c, t)] - p0, xda[(r, c, a, t)] - q0
            data["vdfb"][(r, c, a, t)] = p0 * q0 + q0 * dp + p0 * dq
    return from_dict(
        data, period_dim=-1, origin="equilibria", params={"xscale": dict(xscale)}
    )


def model_levels(m: Any) -> dict[str, dict[tuple[str, ...], float]]:
    """``{var: {(indices..., period): level}}`` of the ``EQ_VARS`` of a solved Pyomo
    model (``pyomo.value``)."""
    from pyomo.environ import value

    return {
        v: {
            (i if isinstance(i, tuple) else (i,)): float(value(c))
            for i, c in getattr(m, v).items()
        }
        for v in EQ_VARS
    }


# GEMPACK set name of each dimension label of gtap7.yaml
_SET = {
    "REG": "REG",
    "SRC": "REG",
    "DST": "REG",
    "COMM": "COMM",
    "ACT": "ACTS",
    "ENDW": "ENDW",
    "MARG": "MARG",
}

# gtap7.yaml rows that are not GEMPACK variables, with the reason
NOT_VARIABLES = {
    "vdfb": "VDFB is a data coefficient (basedata.har), not a variable of the .sl4"
}


def _ratio(pre: float, post: float) -> float:
    return 0.0 if pre == 0.0 else 100.0 * (post / pre - 1.0)


def gtap7_rows() -> dict[str, VariableMap]:
    """The compared rows of ``gtap7.yaml`` by GEMPACK variable name."""
    return {
        v.b.var: v
        for v in Mapping.builtin("gtap7").variables
        if v.name not in NOT_VARIABLES
    }


def gempack_pct(
    levels: Levels,
    xscale: AbcMapping[Any, float],
    pre: str = "check",
    post: str = "shock",
    *,
    skip: Collection[str] = (),
) -> tuple[Pct, dict[str, set]]:
    """``(% changes, cells whose pre value is 0)`` of every compared row of
    ``gtap7.yaml``: ``{GEMPACK variable: {cell: 100 * (post / pre - 1)}}`` with cells in
    GEMPACK order, from the levels of :data:`EQ_VARS` (``model_levels`` or
    ``family_levels``) through :func:`equilibria_side`. A cell whose ``pre`` is 0
    reports 0 and is listed in the second dict (only variables that have one). Scales
    cancel in a ratio, so ``divide_by`` is ignored. ``skip``: GEMPACK names left out
    (AlterTax computes its fixed variables itself)."""
    side = equilibria_side(levels, xscale)
    out: Pct = {}
    zero: dict[str, set] = {}
    for name, row in gtap7_rows().items():
        if name in skip:
            continue
        cells = side.levels[row.a.var]
        perm = [row.a.dims.index(d) for d in row.b.dims]
        out[name], zero[name] = {}, set()
        for period, idx in cells:
            if period != pre:
                continue
            key = tuple(idx[p] for p in perm)
            out[name][key] = _ratio(cells[(pre, idx)], cells[(post, idx)])
            if cells[(pre, idx)] == 0.0:
                zero[name].add(key)
    return out, {n: c for n, c in zero.items() if c}
