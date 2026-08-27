"""Pyomo expression tree -> JAX function (prototype).

The GTAP equations are built in Pyomo (blocks/gtap/*.py). Pyomo already resolves every
build-time conditional (data-dependent if/return None, elasticity-limit branches, sums,
macro expansion) into a concrete expression TREE of pure operations over variables. So we
don't translate the Python logic of build_expression — we translate the resulting tree.

The vocabulary is small and closed (confirmed on the real eq_nd tree):
  SumExpression, NegationExpression, ProductExpression, MonomialTermExpression,
  DivisionExpression, PowExpression, (LinearExpression), plus leaves VarData and float.

`translate(expr, var_index)` walks the tree and returns a Python callable f(z) -> scalar,
where z is a 1-D JAX array of all model variables and `var_index[id(vardata)]` gives each
variable's slot. Because it's built from jnp ops, the result is jittable and autodiff-able,
so jax.grad / sparsejac give the Jacobian for free.
"""
from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import pyomo.core.expr.numeric_expr as ne
from pyomo.core.base.var import VarData


def translate(expr: Any, var_index: dict[int, int]) -> Callable[[Any], Any]:
    """Return f(z) computing `expr` in JAX, z indexed by var_index[id(vardata)] -> slot."""

    def visit(e: Any) -> Callable[[Any], Any]:
        # --- leaves ---
        if isinstance(e, VarData):
            slot = var_index.get(id(e))
            if slot is None:
                # a FIXED variable (not a free column): freeze its current value as a constant,
                # exactly as the squared system treats it (fixed vars are data, not unknowns).
                c = float(e.value)
                return lambda z: c
            return lambda z: z[slot]
        if isinstance(e, (int, float)):
            c = float(e)
            return lambda z: c
        # a fixed Var or Param resolved to a number
        if not hasattr(e, "args"):
            try:
                c = float(e)
                return lambda z: c
            except Exception as exc:  # pragma: no cover
                raise TypeError(f"leaf not translatable: {type(e).__name__}") from exc

        # --- internal nodes (translate children once, combine) ---
        children = [visit(a) for a in e.args]

        if isinstance(e, ne.SumExpression) or isinstance(e, ne.LinearExpression):
            return lambda z: sum(c(z) for c in children)
        if isinstance(e, ne.NegationExpression):
            (c0,) = children
            return lambda z: -c0(z)
        if isinstance(e, ne.ProductExpression):
            c0, c1 = children
            return lambda z: c0(z) * c1(z)
        if isinstance(e, ne.MonomialTermExpression):
            # (coef, var) — coef is numeric, var is a VarData
            c0, c1 = children
            return lambda z: c0(z) * c1(z)
        if isinstance(e, ne.DivisionExpression):
            c0, c1 = children
            return lambda z: c0(z) / c1(z)
        if isinstance(e, ne.PowExpression):
            c0, c1 = children
            return lambda z: c0(z) ** c1(z)
        # unary functions exp/log/sqrt
        if isinstance(e, ne.UnaryFunctionExpression):
            (c0,) = children
            fn = e.getname()
            jfn = {"exp": jnp.exp, "log": jnp.log, "sqrt": jnp.sqrt}.get(fn)
            if jfn is None:
                raise TypeError(f"unsupported unary fn: {fn}")
            return lambda z: jfn(c0(z))
        if isinstance(e, ne.AbsExpression):
            (c0,) = children
            return lambda z: jnp.abs(c0(z))

        raise TypeError(f"unsupported expression node: {type(e).__name__}")

    return visit(expr)
