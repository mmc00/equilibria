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

    # --- vectorized: group by EXACT SHAPE-KEY (tree structure + n_consts + n_slots), NOT by
    # family. Many GTAP families are NON-uniform (variable-length sums → cells touch different
    # #vars): grouping by family breaks vmap (inhomogeneous shapes). Grouping by shape-key makes
    # every group uniform by construction → one small vmap(cell_fn) graph per shape. 152 shape
    # groups but top-12 cover 91% of cells; each graph is tiny (one shape), so compiles fast and
    # never OOMs (unlike sparsejac-per-family). Cells keep their cons_order positions.
    from pyomo.environ import value as _value
    import pyomo.core.expr.numeric_expr as _ne
    from pyomo.core.base.var import VarData as _VarData

    def _shape_sig(e):
        if isinstance(e, _VarData):
            # CRITICAL: a FIXED var is translated as a CONSTANT (not a var-slot), so it must
            # get a DIFFERENT shape symbol than a free var — otherwise two cells with the same
            # tree but different fixed/free patterns collide in one group with incompatible
            # structure (the cell_fn bakes the representative's var/const positions), producing
            # NaN (an exponent used as a base, etc.). Match translate_parametric's var/const split.
            return "C" if (e.fixed or var_index.get(id(e)) is None) else "V"
        if isinstance(e, (int, float)) or not hasattr(e, "args"):
            return "C"
        return type(e).__name__ + "(" + ",".join(_shape_sig(a) for a in e.args) + ")"

    def _rhs(c):
        if c.equality or (c.lower is not None and c.lower is c.upper):
            return float(_value(c.lower))
        if c.upper is not None:
            return float(_value(c.upper))
        if c.lower is not None:
            return float(_value(c.lower))
        return 0.0

    grp_cells: "OrderedDict[Any, list]" = OrderedDict()
    grp_pos: "OrderedDict[Any, list]" = OrderedDict()
    for i, c in enumerate(cons_order):
        k, s = translate_parametric(c.body, var_index, extract_only=True)
        key = (_shape_sig(c.body), len(k), len(s))
        grp_cells.setdefault(key, []).append(c)
        grp_pos.setdefault(key, []).append(i)

    n = len(cons_order)
    per_family = []  # (pos, cell_fn|None, consts, slots, rhs) or (pos, None, row_fns, None, None)
    for key, cells in grp_cells.items():
        # every cell in this group has IDENTICAL shape + n_consts + n_slots (by construction),
        # so one cell_fn vmaps over the group's stacked (consts, slots) with no padding.
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
        pos = jnp.asarray(grp_pos[key])
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
            # keep cell_fn + per-cell consts/slots so the JACOBIAN can be computed the SAME
            # shape-grouped way (vmap(jacrev(cell_fn)) over the group), consistent with F.
            meta = {"cell_fn": cell_fn, "consts": consts_stack, "slots": slots_stack}
            per_family.append((pos, _mk_fam_fn(cell_fn, consts_stack, slots_stack, rhs_stack), meta))
        else:
            row_fns = [translate_constraint(c, var_index) for c in cells]
            def _mk_rows_fn(fns):
                @jax.jit
                def rows_fn(z):
                    return jnp.stack([f(z) for f in fns])
                return rows_fn
            per_family.append((pos, _mk_rows_fn(row_fns), None))

    def F(z):
        # assemble the full residual by scattering each group's (separately-jitted) block.
        # Not jitted at the top level — that is the whole point: no single 395k-output graph.
        out = jnp.zeros(n)
        for pos, fam_fn, _meta in per_family:
            out = out.at[pos].set(fam_fn(z))
        return out

    # expose the per-family blocks so the Jacobian can be computed family-by-family (small
    # graphs) instead of one giant sparsejac.jacrev over the whole 395k-output F (which OOMs
    # the XLA compile at 20x41 scale). Each entry: (row_positions, family_fn).
    F.per_family = per_family
    F.n_eq = n
    return F, var_index, cons_order
