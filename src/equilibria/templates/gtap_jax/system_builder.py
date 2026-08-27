"""Assemble the per-cell Pyomo->JAX translations into ONE vectorized residual F(z).

`build_F(m)` walks the active constraints of a (squared) Pyomo model and returns a single
jitted F(z) that stacks every constraint residual in Pyomo's constraint order, matching
`nlp.evaluate_eq_constraints()` elementwise.

Two build modes:
  vectorize=False (default): one translated callable per row, stacked. Correct and simple —
    the parity ORACLE. Compile cost grows with #rows (fine at 3x3/10x7, not at 395k).
  vectorize=True: within each family (all cells share ONE tree shape — verified), evaluate the
    family's cells with `jax.vmap` over per-cell (constant, var-slot) arrays, so compile cost
    is O(#families) not O(#rows). This is what scales to 395k. Values are identical to the
    non-vectorized path (same math, just batched).

Returns (F, var_index, cons_order) — F jitted, var_index: id(VarData)->slot over free vars,
cons_order: the row order (row i of F == cons_order[i]).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import jax
import jax.numpy as jnp
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var

from equilibria.templates.gtap_jax.expr_to_jax import (
    translate,
    translate_constraint,
    translate_parametric,
)


def build_F(m: Any, vectorize: bool = False, var_index=None, cons_order=None):
    """Build the vectorized JAX residual F(z).

    By default var_index/cons_order are derived from the model's active vars/constraints. Pass
    them explicitly (e.g. from a PyomoNLP's get_pyomo_variables()/get_pyomo_constraints()) to
    align the JAX system to an EXTERNAL ordering — required when replacing a PyomoNLP's eval in
    the driver, where z and the Jacobian rows/cols must follow the NLP's order.
    """
    if var_index is None:
        free_vars = [v for v in m.component_data_objects(Var, active=True) if not v.fixed]
        var_index = {id(v): i for i, v in enumerate(free_vars)}
    if cons_order is None:
        cons_order = list(m.component_data_objects(Constraint, active=True))

    if not vectorize:
        row_fns = [translate_constraint(c, var_index) for c in cons_order]

        def F(z):
            return jnp.stack([f(z) for f in row_fns])

        return jax.jit(F), var_index, cons_order

    # --- vectorized: group by family, vmap each family's shared tree over its cells ---
    # family order preserved; within a family, cells keep their cons_order positions.
    fam_cells: "OrderedDict[str, list]" = OrderedDict()
    fam_pos: "OrderedDict[str, list]" = OrderedDict()
    for i, c in enumerate(cons_order):
        fam = str(c.name).split("[")[0]
        fam_cells.setdefault(fam, []).append(c)
        fam_pos.setdefault(fam, []).append(i)

    # For each family, translate ONE representative cell into a parametric function
    #   cell_fn(z, consts) -> scalar
    # where `consts` is the vector of numeric leaves for that cell (in tree order), and z is
    # the global variable vector (var slots are baked structurally, shared across cells because
    # the family shares the tree shape). Then vmap cell_fn over the stacked consts of all cells.
    from pyomo.environ import value as _value

    def _rhs(c):
        if c.equality or (c.lower is not None and c.lower is c.upper):
            return float(_value(c.lower))
        if c.upper is not None:
            return float(_value(c.upper))
        if c.lower is not None:
            return float(_value(c.lower))
        return 0.0

    n = len(cons_order)
    per_family = []  # (pos, cell_fn|None, consts, slots, rhs) or (pos, None, row_fns, None, None)
    for fam, cells in fam_cells.items():
        # cell_fn(z, consts, slots) from the representative; consts AND var-slots differ per
        # cell but the tree STRUCTURE is shared across the family (verified: 1 shape/family),
        # so one cell_fn works for all cells — vmap over the stacked per-cell (consts, slots).
        # The per-cell RHS (nonzero for `sum==1.0`-style eqs) is subtracted to get the residual.
        cell_fn, (n_consts, n_slots) = translate_parametric(cells[0].body, var_index)
        consts_list, slots_list, rhs_list = [], [], []
        ok = True
        for c in cells:
            k, s = translate_parametric(c.body, var_index, extract_only=True)
            if len(k) != n_consts or len(s) != n_slots:
                ok = False
                break
            consts_list.append(k)
            slots_list.append(s)
            rhs_list.append(_rhs(c))
        pos = jnp.asarray(fam_pos[fam])
        if ok:
            consts_stack = (jnp.asarray(consts_list, dtype=float) if n_consts > 0
                            else jnp.zeros((len(cells), 0)))
            slots_stack = (jnp.asarray(slots_list, dtype=int) if n_slots > 0
                           else jnp.zeros((len(cells), 0), dtype=int))
            rhs_stack = jnp.asarray(rhs_list, dtype=float)
            # jit ONE small graph PER FAMILY (vmap over the family's cells), NOT one giant jit
            # over all 96 families. A single jax.jit(F) fuses everything into one XLA graph of
            # ~395k outputs → the compile blows RAM at 20x41 scale (measured: OOM in solve while
            # model+jax = 5.6GB). Per-family jit keeps each graph small (one family's cells);
            # the assembly (scatter into the full vector) is done in Python, outside jit.
            def _mk_fam_fn(cf, cs, ss, rs):
                @jax.jit
                def fam_fn(z):
                    return jax.vmap(lambda c, s: cf(z, c, s))(cs, ss) - rs
                return fam_fn
            per_family.append((pos, _mk_fam_fn(cell_fn, consts_stack, slots_stack, rhs_stack)))
        else:
            row_fns = [translate_constraint(c, var_index) for c in cells]
            def _mk_rows_fn(fns):
                @jax.jit
                def rows_fn(z):
                    return jnp.stack([f(z) for f in fns])
                return rows_fn
            per_family.append((pos, _mk_rows_fn(row_fns)))

    def F(z):
        # assemble the full residual by scattering each family's (separately-jitted) block.
        # Not jitted at the top level — that is the whole point: no single 395k-output graph.
        out = jnp.zeros(n)
        for pos, fam_fn in per_family:
            out = out.at[pos].set(fam_fn(z))
        return out

    # expose the per-family blocks so the Jacobian can be computed family-by-family (small
    # graphs) instead of one giant sparsejac.jacrev over the whole 395k-output F (which OOMs
    # the XLA compile at 20x41 scale). Each entry: (row_positions, family_fn).
    F.per_family = per_family
    F.n_eq = n
    return F, var_index, cons_order
