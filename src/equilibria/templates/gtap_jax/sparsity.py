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


def build_jacobian_fn(F, cons_order, var_index):
    """Return jac(z) -> scipy.sparse.csr_matrix, computed FAMILY-BY-FAMILY.

    F carries `F.per_family` (a list of (row_positions, family_fn)). Instead of one giant
    sparsejac.jacrev over the whole 395k-output F (whose XLA compile OOMs at 20x41 scale), we
    take jax.jacrev of EACH family_fn separately (a small graph: n_cells_fam outputs) and
    scatter its nonzeros into the global CSR. Each family's dense block is n_cells_fam × n; we
    keep only the structurally-nonzero columns (from the family's own var-leaves) so the memory
    is bounded by the family's real sparsity, not n.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import scipy.sparse as sp

    per_family = getattr(F, "per_family", None)
    n_eq = getattr(F, "n_eq", len(cons_order))
    n_var = len(var_index)
    if per_family is None:
        # fallback: whole-F sparse jacrev (small systems only)
        import sparsejac
        pat = jacobian_pattern(cons_order, var_index)
        jac = jax.jit(sparsejac.jacrev(F, sparsity=pat))

        def _fn(z):
            J = jac(jnp.asarray(z, dtype=float))
            idx = np.asarray(J.indices)
            return sp.csr_matrix((np.asarray(J.data), (idx[:, 0], idx[:, 1])),
                                 shape=(n_eq, n_var))
        return _fn

    # precompute, per family: the row positions (np), the family jac fn (jitted jacrev), and
    # the column slots the family touches (to sparsify its dense block).
    from equilibria.templates.gtap_jax.expr_to_jax import collect_var_slots
    fam_specs = []
    # group cons_order by family to get each family's touched columns
    from collections import OrderedDict
    fam_cols = OrderedDict()
    fam_rowlist = OrderedDict()
    for i, c in enumerate(cons_order):
        fam = str(c.name).split("[")[0]
        fam_cols.setdefault(fam, set()).update(collect_var_slots(c.body, var_index))
        fam_rowlist.setdefault(fam, []).append(i)

    fam_names = list(fam_rowlist.keys())
    for (pos, fam_fn), fam in zip(per_family, fam_names):
        cols = np.array(sorted(fam_cols[fam]), dtype=np.int64)
        rows = np.asarray(pos, dtype=np.int64)
        jfn = jax.jit(jax.jacrev(fam_fn))  # small: n_cells_fam × n, but only `cols` are nonzero
        fam_specs.append((rows, cols, jfn))

    def _fn(z):
        zj = jnp.asarray(z, dtype=float)
        all_r, all_c, all_v = [], [], []
        for rows, cols, jfn in fam_specs:
            Jb = np.asarray(jfn(zj))  # (n_cells_fam, n_var) dense block for this family
            sub = Jb[:, cols]         # keep only structurally-nonzero columns
            rr, cc = np.nonzero(np.abs(sub) > 0)
            all_r.append(rows[rr]); all_c.append(cols[cc]); all_v.append(sub[rr, cc])
        R = np.concatenate(all_r); C = np.concatenate(all_c); V = np.concatenate(all_v)
        return sp.csr_matrix((V, (R, C)), shape=(n_eq, n_var))

    return _fn


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
