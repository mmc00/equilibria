"""``results``: the view of a solved GTAP exercise, in GEMPACK's names."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from equilibria.compare.gtap7_side import (
    _SET,
    EQ_VARS,
    gempack_pct,
    gtap7_rows,
    model_levels,
)

_log = logging.getLogger(__name__)

SUMMARY_COLUMNS = ("EV", "qgdp", "exports", "imports")


@dataclass
class GTAPResults:
    """% changes of a solved exercise by GEMPACK variable, EV and a summary.

    ``tables[var]``: one ``DataFrame`` per GEMPACK variable (``qo``, ``pm``, ``qxs``...),
    one column ``pct`` (``100 * (post / pre - 1)``), indexed by GEMPACK's sets in
    GEMPACK's order (``qxs``: ``COMM, REG, REG``, exporter then importer); a scalar
    (``pfactwld``) has one row. ``ev``: the EV of each regional household
    (``regional_ev``, in units of ``regy``); empty on the family backend.
    ``summary``: one row per region with ``EV``, ``qgdp`` (real GDP, % change) and
    ``exports``/``imports``, the % change of the sum of the region's ``xw`` out and in.
    In a terminal or a notebook the object shows its ``summary``.
    """

    tables: dict[str, pd.DataFrame]
    ev: pd.Series
    summary: pd.DataFrame

    def __repr__(self) -> str:
        return repr(self.summary)

    def _repr_html_(self) -> str:
        return self.summary.to_html()


def _table(name: str, cells: dict[tuple[str, ...], float]) -> pd.DataFrame:
    dims = [_SET[d] for d in gtap7_rows()[name].b.dims]
    keys = list(cells)
    if not dims:
        index = pd.RangeIndex(len(keys))
    elif len(dims) == 1:
        index = pd.Index([k[0] for k in keys], name=dims[0])
    else:
        index = pd.MultiIndex.from_tuples(keys, names=dims)
    return pd.DataFrame({"pct": list(cells.values())}, index=index)


def _trade(levels, regions, pre: str, post: str) -> tuple[dict, dict]:
    xw = levels["xw"]
    out = {r: {pre: 0.0, post: 0.0} for r in regions}
    inn = {r: {pre: 0.0, post: 0.0} for r in regions}
    for (s, _i, d, t), x in xw.items():
        if t in (pre, post):
            out[s][t] += x
            inn[d][t] += x

    def pct(tot):
        return {
            r: 100.0 * (v[post] / v[pre] - 1.0) if v[pre] else 0.0
            for r, v in tot.items()
        }

    return pct(out), pct(inn)


def results(ex: Any, pre: str = "check", post: str = "shock") -> GTAPResults:
    """The % changes from ``pre`` to ``post`` of a solved exercise ``ex``.

    The tables are the GEMPACK translation of ``compare/maps/gtap7.yaml``
    (:func:`equilibria.compare.gtap7_side.gempack_pct`, the rule the AlterTax gates hold
    against GEMPACK): a cell whose ``pre`` is 0 reports 0. Levels come from the Pyomo
    model, or from the family arrays when ``ex.template.backend == "family"``. The EV
    is :func:`~equilibria.templates.gtap.welfare_regional.regional_ev` on Pyomo; the
    family backend has no EV yet (MAR-251): ``ev`` is empty and a WARNING says so.
    Raises ``RuntimeError`` before ``ex.run()``.
    """
    if ex.model is None:
        raise RuntimeError("results: run the exercise first (ex.run())")
    p = ex.template.params
    regions = list(p.sets.r)
    if ex.template.backend == "family":
        from equilibria.family.altertax.levels import family_levels

        levels = family_levels(ex.model.mp, EQ_VARS)
        ev = pd.Series(dtype=float, name="EV")
        _log.warning("results: the family backend has no EV yet (MAR-251); ev is empty")
    else:
        from equilibria.templates.gtap.welfare_regional import regional_ev

        m = ex.model.pyomo
        levels = model_levels(m)
        ev = pd.Series(regional_ev(m, post, pre), name="EV", dtype=float)
    from equilibria.blocks.gtap._derived_params import xscale_data

    pct, _ = gempack_pct(levels, dict(xscale_data(p, p.sets)), pre, post)
    tables = {name: _table(name, cells) for name, cells in pct.items()}
    exports, imports = _trade(levels, regions, pre, post)
    qgdp = tables["qgdp"]["pct"]
    summary = pd.DataFrame(
        {
            "EV": [ev.get(r, float("nan")) for r in regions],
            "qgdp": [float(qgdp[r]) for r in regions],
            "exports": [exports[r] for r in regions],
            "imports": [imports[r] for r in regions],
        },
        index=pd.Index(regions, name="REG"),
    )
    _log.info(
        "results: %d GEMPACK variables, %s -> %s, EV %s",
        len(tables),
        pre,
        post,
        {r: round(float(x), 6) for r, x in ev.items()},
    )
    return GTAPResults(tables=tables, ev=ev, summary=summary)
