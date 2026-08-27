"""Jacobian sparsity pattern for the JAX residual, as a BCOO for sparsejac.

sparsejac.jacrev(F, sparsity=pattern) needs the (row, col) pattern of dF/dz. We derive it
structurally: for each constraint row i, the nonzero columns = the free-variable slots that
row's expression tree touches (from `collect_var_slots`). This is exact — it equals Pyomo's
actual Jacobian nnz (no numerical probing) — so sparsejac computes exactly the right entries.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

from equilibria.templates.gtap_jax.expr_to_jax import collect_var_slots


def jacobian_pattern(cons_order: list, var_index: dict[int, int]):
    """Return a BCOO of shape (n_eq, n_var) whose stored positions are the Jacobian nnz.

    The values are 1.0 (a pattern, not the numeric Jacobian). Row order = cons_order index;
    col order = var_index slot.
    """
    n_eq = len(cons_order)
    n_var = len(var_index)
    rows: list[int] = []
    cols: list[int] = []
    for i, c in enumerate(cons_order):
        for j in sorted(collect_var_slots(c.body, var_index)):
            rows.append(i)
            cols.append(j)
    indices = np.stack([np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)], axis=1)
    data = np.ones(len(rows), dtype=float)
    return jsparse.BCOO(
        (jnp.asarray(data), jnp.asarray(indices)),
        shape=(n_eq, n_var),
    )
