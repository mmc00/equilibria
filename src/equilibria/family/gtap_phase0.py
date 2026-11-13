"""Phase 0 GTAP families: ``eq_pndeq``/``eq_pvaeq`` (CES prices) and ``eq_facty`` (sum).

Each family is written once over set-indexed arrays. Parameters come from the same
``_derived_params`` recipes the blocks use (``ProductionSupplyBlock``, ``IncomeBlock``),
never from a built Pyomo model; the variable state comes in through ``VarLayout``.

Pyomo bodies reproduced (``lhs == rhs`` evaluates as ``lhs - rhs``):

- ``eq_pndeq[r,a]`` (production_supply.py): ``pnd**e == sum_i io*(pa[r,i,a]/lambdaio)**e``
  with ``e = 1 - sigmand``, ``lambdaio`` floored at 1e-8, terms only where ``io > 0``
  and rows only where some term exists.
- ``eq_pvaeq[r,a]`` (production_supply.py, ifSUB=0): ``pva**e == sum_f af*(pfa/lambdaf)**e``
  with ``e = 1 - sigmav``, terms only where ``af > 0``; ``lambdaf`` is a (normally
  fixed) variable, so it enters the term as a second slot.
- ``eq_facty[r]`` (income.py): ``facty == sum_{f,a: xfflag>0} pf*xf/xscale
  - fdepr*pi*kstock``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from equilibria.blocks.gtap import _derived_params as dp
from equilibria.family.core import Family, FamilySystem
from equilibria.family.layout import VarLayout
from equilibria.templates.gtap.gtap_block_model import _set_elems

# Variables the Phase 0 families touch, with their block-declared domains.
PHASE0_VARS: dict[str, tuple[str, ...]] = {
    "pa": ("r", "i", "aa"),
    "pnd": ("r", "a"),
    "pva": ("r", "a"),
    "pfa": ("r", "f", "a"),
    "lambdaf": ("r", "f", "a"),
    "pf": ("r", "f", "a"),
    "xf": ("r", "f", "a"),
    "facty": ("r",),
    "pi": ("r",),
    "kstock": ("r",),
}


def _pndeq_row(xs, e):
    return xs[0] ** e


def _pndeq_term(xs, io, lam, e):
    return io * (xs[0] / lam) ** e


def _pvaeq_term(xs, af, e):
    return af * (xs[0] / xs[1]) ** e


def _facty_row(xs, fdepr):
    return xs[0] + fdepr * xs[1] * xs[2]


def _facty_term(xs, xscale):
    return xs[0] * xs[1] / xscale


def _positions(sub: list[str], full: list[str]) -> np.ndarray:
    where = {e: k for k, e in enumerate(full)}
    return np.array([where[e] for e in sub], dtype=np.int64)


def _labels(*elems_by_dim: list[str], idx: tuple[np.ndarray, ...]) -> list[str]:
    return [
        ",".join(e[k] for e, k in zip(elems_by_dim, cell, strict=True))
        for cell in zip(*idx, strict=True)
    ]


@dataclass
class Phase0Data:
    """Parameter arrays of the Phase 0 families, from the blocks' ``_derived_params``."""

    sets: dict[str, list[str]]
    arrays: dict[str, np.ndarray]


def phase0_data(params: Any, sets: Any, residual_region: str) -> Phase0Data:
    el = _set_elems(sets)
    r, i, a, f, aa = el["r"], el["i"], el["a"], el["f"], el["aa"]

    def expo(sig: dict) -> np.ndarray:
        return 1.0 - np.array([[sig.get((rr, x), 1.0) for x in a] for rr in r], float)

    io = dp.to_array(dp.io_param_data(params, sets), [r, i, a], 0.0)
    if not params.shifts.lambdaio:
        io = dp.to_array(dp.p_io_data(params, sets), [r, i, a], 0.0)
    calib = dp.demand_income_params(params, sets, residual_region=residual_region)
    xscale = dp.to_array(dp.xscale_data(params, sets), [r, aa], 1.0)
    return Phase0Data(
        sets=el,
        arrays={
            "io": io,
            "lambdaio": np.maximum(
                dp.to_array(dp.lambdaio_data(params, sets), [r, i, a], 1.0), 1e-8
            ),
            "e_nd": expo(params.elasticities.sigmand),
            "af": dp.to_array(dp.af_param_data(params, sets), [r, f, a], 0.0),
            "e_va": expo(params.elasticities.sigmav),
            "xfflag": dp.to_array(dp.xfflag_data(params, sets), [r, f, a], 0.0),
            "xscale": xscale[:, _positions(a, aa)],
            "fdepr": dp.to_array(calib["fdepr"], [r], 0.0),
        },
    )


def _rows_and_terms(term_mask: np.ndarray):
    """Rows = (r, a) cells with any term; term index arrays in C order; row ids."""
    row_mask = term_mask.any(axis=1)
    row_id = np.full(row_mask.shape, -1, dtype=np.int64)
    row_id[row_mask] = np.arange(int(row_mask.sum()))
    return row_mask, row_id, np.nonzero(term_mask)


def pndeq_family(data: Phase0Data, layout: VarLayout) -> Family:
    el, d = data.sets, data.arrays
    r, a, aa = el["r"], el["a"], el["aa"]
    e = d["e_nd"]
    row_mask, row_id, (tr, ti, ta) = _rows_and_terms(d["io"] > 0.0)
    pa_slots = layout.col("pa")[:, :, _positions(a, aa)]
    return Family(
        name="eq_pndeq",
        row_fn=_pndeq_row,
        row_cols=layout.col("pnd")[row_mask].reshape(-1, 1),
        row_args={"e": e[row_mask]},
        term_fn=_pndeq_term,
        term_rows=row_id[tr, ta],
        term_cols=pa_slots[tr, ti, ta].reshape(-1, 1),
        term_args={
            "io": d["io"][tr, ti, ta],
            "lam": d["lambdaio"][tr, ti, ta],
            "e": e[tr, ta],
        },
        row_labels=_labels(r, a, idx=np.nonzero(row_mask)),
    )


def pvaeq_family(data: Phase0Data, layout: VarLayout) -> Family:
    el, d = data.sets, data.arrays
    r, a = el["r"], el["a"]
    e = d["e_va"]
    row_mask, row_id, (tr, tf, ta) = _rows_and_terms(d["af"] > 0.0)
    return Family(
        name="eq_pvaeq",
        row_fn=_pndeq_row,
        row_cols=layout.col("pva")[row_mask].reshape(-1, 1),
        row_args={"e": e[row_mask]},
        term_fn=_pvaeq_term,
        term_rows=row_id[tr, ta],
        term_cols=np.stack(
            [layout.col("pfa")[tr, tf, ta], layout.col("lambdaf")[tr, tf, ta]], axis=1
        ),
        term_args={"af": d["af"][tr, tf, ta], "e": e[tr, ta]},
        row_labels=_labels(r, a, idx=np.nonzero(row_mask)),
    )


def facty_family(data: Phase0Data, layout: VarLayout) -> Family:
    d = data.arrays
    tr, tf, ta = np.nonzero(d["xfflag"] > 0.0)
    return Family(
        name="eq_facty",
        row_fn=_facty_row,
        row_cols=np.stack(
            [layout.col("facty"), layout.col("pi"), layout.col("kstock")], axis=1
        ),
        row_args={"fdepr": d["fdepr"]},
        term_fn=_facty_term,
        term_rows=tr,
        term_cols=np.stack(
            [layout.col("pf")[tr, tf, ta], layout.col("xf")[tr, tf, ta]], axis=1
        ),
        term_args={"xscale": d["xscale"][tr, ta]},
        row_labels=list(data.sets["r"]),
    )


def build_phase0_system(
    params: Any,
    sets: Any,
    layout: VarLayout,
    residual_region: str,
    data: Phase0Data | None = None,
) -> FamilySystem:
    data = phase0_data(params, sets, residual_region) if data is None else data
    families = [
        pndeq_family(data, layout),
        pvaeq_family(data, layout),
        facty_family(data, layout),
    ]
    return FamilySystem(
        families, n_cols=layout.n_cols, fixed_values=layout.fixed_values
    )
