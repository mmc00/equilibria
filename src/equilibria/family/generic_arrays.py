"""``MPArrays`` of any blocks model, with no GTAP step and no Pyomo model.

``MPArrays.build`` is GTAP's (Fisher rows, ``mq_*``, ShockBlock instruments). This is
the general case: the model's ``StateModel`` gives levels and bounds, ``FamilyBackend``
records its rows, and both are laid out once per period. Cells keep their index tuples
with the period last, rows come in recording order, as in ``MPArrays.build``.

Bounds come from the state (``lower``/``upper``), not only from ``NonNegativeReals``:
a level block's floor (``LEVEL_FLOOR``) is a bound on a ``Reals`` variable.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

import numpy as np

from equilibria.backends.family_backend import FamilyBackend
from equilibria.family.mp_arrays import MPArrays, RowArr, VarArr
from equilibria.family.state import StateModel


def _per_period(a: np.ndarray, n_t: int) -> np.ndarray:
    return np.repeat(np.asarray(a, dtype=float).ravel(), n_t)


def build_arrays(
    model: Any, exogenous: Iterable[str], periods: tuple[str, ...]
) -> tuple[MPArrays, FamilyBackend, StateModel]:
    """The arrays of ``model`` over ``periods``; ``exogenous`` variables (the
    instruments) are fixed in every period. Returns the recording and the state too:
    the period model evaluates rows with them."""
    exo = frozenset(exogenous)
    state = StateModel(model)
    backend = FamilyBackend()
    backend.build(model)
    n_t = len(periods)

    vars_: dict[str, VarArr] = {}
    for name, doms in state.var_domains.items():
        keys = [
            (*k, t)
            for k in itertools.product(*(state.sets[d] for d in doms))
            for t in periods
        ]
        value = _per_period(state.values[name], n_t)
        value = np.where(np.isnan(value), 1.0, value)
        fixed = np.full(len(keys), name in exo)
        lb = _per_period(state.lower[name], n_t)
        ub = _per_period(state.upper[name], n_t)
        vars_[name] = VarArr(keys, value, fixed, lb, ub)

    rows = {
        fam: RowArr(
            [(*idx, t) for t in periods for idx, _ in recs],
            np.ones(len(recs) * n_t, dtype=bool),
        )
        for fam, recs in backend.records.items()
    }
    return MPArrays(vars_, rows, exogenous=exo), backend, state
