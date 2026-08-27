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

    import sparsejac
    from jax.experimental import sparse as jsparse

    fam_names = list(fam_rowlist.keys())
    for (pos, fam_fn), fam in zip(per_family, fam_names):
        rows = np.asarray(pos, dtype=np.int64)
        n_cells = len(rows)
        # Build the family's OWN sparsity pattern (n_cells × n_var), then use sparsejac so the
        # family jacobian is computed SPARSELY — never the huge dense n_cells×n_var block that
        # jax.jacrev would materialize (e.g. eq_xda: 2940×11277 = 33M floats at 10x7 → ~31GB at
        # 20x41). sparsejac gives only the structurally-nonzero entries.
        prow, pcol = [], []
        for local_i, gi in enumerate(rows.tolist()):
            for j in sorted(collect_var_slots(cons_order[gi].body, var_index)):
                prow.append(local_i); pcol.append(j)
        if not prow:  # a family with no free-var deps (all constants) — empty jacobian block
            fam_specs.append((rows, None, None))
            continue
        idx = jnp.asarray(np.stack([np.asarray(prow, np.int32), np.asarray(pcol, np.int32)], 1))
        sub_pat = jsparse.BCOO((jnp.ones(len(prow)), idx), shape=(n_cells, n_var))
        jfn = jax.jit(sparsejac.jacrev(fam_fn, sparsity=sub_pat))
        fam_specs.append((rows, jfn, None))

    def _fn(z):
        zj = jnp.asarray(z, dtype=float)
        all_r, all_c, all_v = [], [], []
        for rows, jfn, _ in fam_specs:
            if jfn is None:
                continue
            Jb = jfn(zj)  # BCOO (n_cells_fam × n_var), sparse
            bidx = np.asarray(Jb.indices); bdat = np.asarray(Jb.data)
            all_r.append(rows[bidx[:, 0]]); all_c.append(bidx[:, 1]); all_v.append(bdat)
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
