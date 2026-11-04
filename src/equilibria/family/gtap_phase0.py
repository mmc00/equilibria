"""Phase 0 GTAP families: ``eq_pndeq`` (CES price) and ``eq_facty`` (factor income sum).

Each family is written once over set-indexed arrays. Parameters come from the same
``_derived_params`` recipes the blocks use (``ProductionSupplyBlock``, ``IncomeBlock``),
never from a built Pyomo model; the variable state comes in through ``VarLayout``.

Pyomo bodies reproduced (``lhs == rhs`` evaluates as ``lhs - rhs``):

- ``eq_pndeq[r,a]`` (production_supply.py): ``pnd**e == sum_i io*(pa[r,i,a]/lambdaio)**e``
  with ``e = 1 - sigmand``, ``lambdaio`` floored at 1e-8, terms only where ``io > 0``
  and rows only where some term exists.
- ``eq_facty[r]`` (income.py): ``facty == sum_{f,a: xfflag>0} pf*xf/xscale
  - fdepr*pi*kstock``.
"""

from __future__ import annotations

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


def _facty_row(xs, fdepr):
    return xs[0] + fdepr * xs[1] * xs[2]


def _facty_term(xs, xscale):
    return xs[0] * xs[1] / xscale


def _positions(sub: list[str], full: list[str]) -> np.ndarray:
    where = {e: k for k, e in enumerate(full)}
    return np.array([where[e] for e in sub], dtype=np.int64)


def pndeq_family(params: Any, sets: Any, layout: VarLayout) -> Family:
    el = _set_elems(sets)
    r, i, a, aa = el["r"], el["i"], el["a"], el["aa"]
    io = dp.to_array(dp.io_param_data(params, sets), [r, i, a], 0.0)
    if not params.shifts.lambdaio:
        io = dp.to_array(dp.p_io_data(params, sets), [r, i, a], 0.0)
    lam = np.maximum(dp.to_array(dp.lambdaio_data(params, sets), [r, i, a], 1.0), 1e-8)
    sig = params.elasticities.sigmand
    expo = 1.0 - np.array([[sig.get((rr, x), 1.0) for x in a] for rr in r], dtype=float)

    term_mask = io > 0.0
    row_mask = term_mask.any(axis=1)
    row_id = np.full(row_mask.shape, -1, dtype=np.int64)
    row_id[row_mask] = np.arange(int(row_mask.sum()))
    tr, ti, ta = np.nonzero(term_mask)

    pa_slots = layout.col("pa")[:, :, _positions(a, aa)]
    return Family(
        name="eq_pndeq",
        row_fn=_pndeq_row,
        row_cols=layout.col("pnd")[row_mask].reshape(-1, 1),
        row_args={"e": expo[row_mask]},
        term_fn=_pndeq_term,
        term_rows=row_id[tr, ta],
        term_cols=pa_slots[tr, ti, ta].reshape(-1, 1),
        term_args={"io": io[tr, ti, ta], "lam": lam[tr, ti, ta], "e": expo[tr, ta]},
    )


def facty_family(
    params: Any, sets: Any, layout: VarLayout, residual_region: str
) -> Family:
    el = _set_elems(sets)
    r, f, a, aa = el["r"], el["f"], el["a"], el["aa"]
    xfflag = dp.to_array(dp.xfflag_data(params, sets), [r, f, a], 0.0)
    xscale = dp.to_array(dp.xscale_data(params, sets), [r, aa], 1.0)[
        :, _positions(a, aa)
    ]
    calib = dp.demand_income_params(params, sets, residual_region=residual_region)
    fdepr = dp.to_array(calib["fdepr"], [r], 0.0)

    tr, tf, ta = np.nonzero(xfflag > 0.0)
    return Family(
        name="eq_facty",
        row_fn=_facty_row,
        row_cols=np.stack(
            [layout.col("facty"), layout.col("pi"), layout.col("kstock")], axis=1
        ),
        row_args={"fdepr": fdepr},
        term_fn=_facty_term,
        term_rows=tr,
        term_cols=np.stack(
            [layout.col("pf")[tr, tf, ta], layout.col("xf")[tr, tf, ta]], axis=1
        ),
        term_args={"xscale": xscale[tr, ta]},
    )


def build_phase0_system(
    params: Any, sets: Any, layout: VarLayout, residual_region: str
) -> FamilySystem:
    families = [
        pndeq_family(params, sets, layout),
        facty_family(params, sets, layout, residual_region),
    ]
    return FamilySystem(
        families, n_cols=layout.n_cols, fixed_values=layout.fixed_values
    )
