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

    # SHAPE-GROUPED Jacobian, consistent with F. Each group's meta has cell_fn(z, consts, slots)
    # + the per-cell (consts, slots) stacks. The derivative of cell i wrt its OWN variables is
    # grad(cell_fn wrt the z[slots] it reads) → a small dense n_slots-vector per cell; vmap over
    # the group gives an (n_cells × n_slots) block whose COLUMNS are exactly the cell's slots.
    # No dense n_cells×n_var, no sparsejac-per-family graph explosion — one small vmapped grad
    # graph per shape-group (same as F). Rows/cols scattered into the global CSR by (pos, slots).
    grp_specs = []
    for pos, fam_fn, meta in per_family:
        rows = np.asarray(pos, dtype=np.int64)
        if meta is None:
            # non-uniform fallback group (rare): use whole-block jacrev, dense small
            grp_specs.append((rows, None, None, None))
            continue
        cell_fn = meta["cell_fn"]; consts = meta["consts"]; slots = meta["slots"]
        if slots.shape[1] == 0:  # group touches no free vars → no jacobian entries
            grp_specs.append((rows, None, None, None))
            continue

        # grad of one cell wrt its own variable-values vector zc (length n_slots)
        def _mk(cf):
            def cell_val(zc, cst, sl):
                # cf indexes z by sl; feed zc as a local z and identity slots 0..k-1
                k = zc.shape[0]
                return cf(zc, cst, jnp.arange(k))
            return jax.jit(jax.vmap(jax.grad(cell_val, argnums=0)))
        gfn = _mk(cell_fn)
        grp_specs.append((rows, gfn, consts, slots))

    def _fn(z):
        zj = jnp.asarray(z, dtype=float)
        all_r, all_c, all_v = [], [], []
        for rows, gfn, consts, slots in grp_specs:
            if gfn is None:
                continue
            zc = zj[slots]              # (n_cells, n_slots) each cell's own var values
            G = np.asarray(gfn(zc, consts, slots))  # (n_cells, n_slots) gradient block
            sl = np.asarray(slots)     # (n_cells, n_slots) the real column indices
            # Emit the FULL STRUCTURAL pattern (every (cell, slot) pair), NOT only numerically
            # nonzero entries — the Jacobian sparsity must be FIXED across Newton steps (cuDSS
            # reuses the analyzed pattern; a value that is 0 at the seed but nonzero later must
            # keep its slot). One caveat: a slot can repeat within a cell (same var appears
            # twice in the tree) — grad already SUMS those, but here each column-occurrence would
            # double-count; dedup per cell by summing G over repeated slots.
            ncell, nsl = sl.shape
            ci = np.repeat(np.arange(ncell), nsl)
            all_r.append(rows[ci]); all_c.append(sl.reshape(-1)); all_v.append(G.reshape(-1))
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
