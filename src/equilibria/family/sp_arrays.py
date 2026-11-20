"""The single-period reference arrays of the family route (``SPRef``) and the
period table of an ``MPArrays`` variable. Moved here from the old family period
recipe (PR 4 of gtap-on-core), unchanged.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from equilibria.family.mp_arrays import MPArrays, RowArr, VarArr

# a unique id per SPVar (see SPVar.uid)
_UID = itertools.count()


@dataclass
class SPVar:
    """One reference variable: every cell (key without period), flat in set order."""

    keys: list[tuple]
    value: np.ndarray  # a fixed cell's value (NaN -> 0.0 as in SPSnapshot)
    fixed: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    uid: int = field(default_factory=lambda: next(_UID))


@dataclass
class SPRef:
    vars: dict[str, SPVar]
    exogenous: frozenset[str]  # registered ShockBlock instruments
    endogenous: dict[str, frozenset]  # instrument cells an @overwrite hook freed


def sp_reference_arrays(params, closure, residual_region, *, state=None) -> SPRef:
    """The reference ``SPReferenceCache`` reads, from ``compose_block_state``.

    Same cells as ``sp_snapshot(build_sp_reference(...))``: the Fisher auxiliaries
    the driver deletes from the reference are left out. ``state`` is the composed state
    when the caller already has it (``MPArrays.build`` reads the same one), so the
    composition runs once; it is only read."""
    from equilibria.blocks.gtap.closure import FISHER_AUX_VARS

    if state is None:
        from equilibria.templates.gtap.gtap_block_model import compose_block_state

        _, state = compose_block_state(params, params.sets, closure, residual_region)
    out = {}
    for name, doms in state.var_domains.items():
        if name in FISHER_AUX_VARS:
            continue
        keys = list(itertools.product(*(state.sets[d] for d in doms)))
        value = np.asarray(state.values[name], dtype=float).ravel().copy()
        out[name] = SPVar(
            keys,
            np.where(np.isnan(value), 0.0, value),
            np.asarray(state.fixed[name], dtype=bool).ravel().copy(),
            np.asarray(state.lower[name], dtype=float).ravel().copy(),
            np.asarray(state.upper[name], dtype=float).ravel().copy(),
        )
    endo = getattr(state, "_endogenous_instrument_cells", None) or {}
    return SPRef(
        out,
        frozenset(getattr(state, "_exogenous_instruments", ())),
        {k: frozenset(v) for k, v in endo.items()},
    )


def period_table(obj: VarArr | RowArr, period: str) -> dict[tuple, int]:
    """Position of each cell of ``period``, by its key without the period."""
    cache = obj.__dict__.setdefault("period_table", {})
    if period not in cache:
        sel = np.flatnonzero(obj.period == period)
        cache[period] = {obj.keys[i][:-1]: int(i) for i in sel}
    return cache[period]
