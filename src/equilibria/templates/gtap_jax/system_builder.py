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
            per_family.append((pos, cell_fn, consts_stack, slots_stack, rhs_stack))
        else:
            per_family.append(
                (pos, None, [translate_constraint(c, var_index) for c in cells], None, None)
            )

    def F(z):
        out = jnp.zeros(n)
        for pos, cell_fn, payload, slots_stack, rhs_stack in per_family:
            if cell_fn is not None:
                vals = jax.vmap(lambda cst, sl: cell_fn(z, cst, sl))(payload, slots_stack)
                vals = vals - rhs_stack
            else:
                vals = jnp.stack([f(z) for f in payload])
            out = out.at[pos].set(vals)
        return out

    return jax.jit(F), var_index, cons_order
