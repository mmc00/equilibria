"""Assemble the per-cell Pyomo->JAX translations into ONE vectorized residual F(z).

`build_F(m)` walks the active constraints of a (squared) Pyomo model, translates each body to
a JAX callable over the global variable vector z, and returns a single jitted F(z) that stacks
them in Pyomo's constraint order. This is the CORRECTNESS version — one callable per row. The
per-family `jax.vmap` scale optimization is a separate step (Task 3); this establishes the
parity oracle (F_jax == F_pyomo elementwise) that the fast path must preserve.

Returns
-------
(F, var_index, cons_order):
  F           : jitted callable, F(z) -> jnp.ndarray of shape (len(cons_order),)
  var_index   : dict id(VarData) -> slot, over the FREE (non-fixed) variables (z's columns)
  cons_order  : list[ConstraintData], the row order (row i of F == cons_order[i])
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var

from equilibria.templates.gtap_jax.expr_to_jax import translate


def build_F(m: Any):
    # z columns = the free variables of the (squared) system
    free_vars = [v for v in m.component_data_objects(Var, active=True) if not v.fixed]
    var_index = {id(v): i for i, v in enumerate(free_vars)}

    # rows = active equality constraints, in Pyomo's iteration order
    cons_order = list(m.component_data_objects(Constraint, active=True))

    # translate each row body once; missing-var leaves (fixed vars) resolve to constants
    # because translate() treats a leaf without a var_index entry as a float via value().
    row_fns = [translate(c.body, var_index) for c in cons_order]

    def F(z):
        return jnp.stack([f(z) for f in row_fns])

    return jax.jit(F), var_index, cons_order
