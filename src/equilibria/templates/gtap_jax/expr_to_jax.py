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
            # large sums → jnp.sum reduction (O(1) compile) not a 1000-deep add tree; see the
            # parametric path for why (macro-closure eqs sum thousands of terms).
            if len(children) > 8:
                return lambda z: jnp.sum(jnp.stack([c(z) for c in children]))
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


def translate_constraint(con: Any, var_index: dict[int, int]) -> Callable[[Any], Any]:
    """Translate a Pyomo equality CONSTRAINT to its residual f(z) = body - rhs.

    pynumero's evaluate_eq_constraints returns `body - rhs`; `body` alone omits a nonzero RHS
    (e.g. `sum(terms) == 1.0`), which shows up as a constant-1.0 mismatch. This wraps
    `translate(con.body)` and subtracts the constant RHS so the residual matches pynumero."""
    from pyomo.environ import value as _value

    body_fn = translate(con.body, var_index)
    rhs = 0.0
    if con.equality or (con.lower is not None and con.lower is con.upper):
        rhs = float(_value(con.lower))
    elif con.upper is not None:
        rhs = float(_value(con.upper))
    elif con.lower is not None:
        rhs = float(_value(con.lower))
    if rhs == 0.0:
        return body_fn
    return lambda z: body_fn(z) - rhs


def collect_var_slots(expr: Any, var_index: dict[int, int]) -> set[int]:
    """Return the set of free-variable slots that `expr` depends on (its Jacobian columns).

    Walks the tree gathering `var_index[id(VarData)]` for every non-fixed VarData leaf. This IS
    the structural sparsity of the row (which columns are nonzero), matching Pyomo's
    identify_variables. Fixed vars / Params are excluded (they're not free columns)."""
    slots: set[int] = set()

    def walk(e: Any) -> None:
        if isinstance(e, VarData):
            s = var_index.get(id(e))
            if s is not None:
                slots.add(s)
            return
        if hasattr(e, "args") and not isinstance(e, (int, float)):
            for a in e.args:
                walk(a)

    walk(expr)
    return slots


def translate_parametric(expr: Any, var_index: dict[int, int], extract_only: bool = False):
    """Split a cell's expression into (structure, constants, var-slots) so a whole family —
    all cells sharing the same tree SHAPE — can be evaluated with one vmapped function over
    per-cell data.

    Walks the tree in a fixed pre-order. Each numeric leaf → `consts[k]`; each free-var leaf →
    `z[slots[j]]` where the SLOT is per-cell data (cells of a family touch DIFFERENT variables,
    so slots must be batched, not baked). A fixed var freezes to a constant. Returns:
      - default: `(cell_fn, packing)` where cell_fn(z, cst, slots) -> scalar, and
        packing=(n_consts, n_slots).
      - extract_only=True: `(consts_list, slots_list)` — this cell's constants and var-slots in
        walk order, used to check shape-compatibility and to stack the family's data.
    """
    consts: list[float] = []
    slots: list[int] = []

    def build(e: Any):
        # leaves
        if isinstance(e, VarData):
            slot = var_index.get(id(e))
            if slot is None:
                k = len(consts); consts.append(float(e.value))
                return ("const", k)
            j = len(slots); slots.append(int(slot))
            return ("var", j)
        if isinstance(e, (int, float)) or not hasattr(e, "args"):
            k = len(consts); consts.append(float(e))
            return ("const", k)
        # internal
        return ("op", e, [build(a) for a in e.args])

    tree = build(expr)

    if extract_only:
        return consts, slots

    n_consts = len(consts)
    n_slots = len(slots)

    def cell_fn(z, cst, sl):
        def ev(node):
            tag = node[0]
            if tag == "var":
                return z[sl[node[1]]]
            if tag == "const":
                return cst[node[1]]
            e = node[1]; kids = [ev(c) for c in node[2]]
            if isinstance(e, (ne.SumExpression, ne.LinearExpression)):
                # CRITICAL for compile speed: a sum with THOUSANDS of terms (macro-closure eqs
                # like eq_walras/eq_pwfact sum over all r×i×f — up to 8401 terms) must lower to a
                # single jnp.sum REDUCTION, not a left-folded tree of 8400 adds. The tree makes
                # XLA compile a graph with one node per term → 100s per such equation. A stacked
                # reduction compiles in O(1). (Small sums: python sum is fine and avoids stack
                # overhead.) Threshold 8 keeps tiny sums cheap.
                if len(kids) > 8:
                    return jnp.sum(jnp.stack(kids))
                return sum(kids)
            if isinstance(e, ne.NegationExpression):
                return -kids[0]
            if isinstance(e, (ne.ProductExpression, ne.MonomialTermExpression)):
                return kids[0] * kids[1]
            if isinstance(e, ne.DivisionExpression):
                return kids[0] / kids[1]
            if isinstance(e, ne.PowExpression):
                return kids[0] ** kids[1]
            if isinstance(e, ne.UnaryFunctionExpression):
                jfn = {"exp": jnp.exp, "log": jnp.log, "sqrt": jnp.sqrt}[e.getname()]
                return jfn(kids[0])
            if isinstance(e, ne.AbsExpression):
                return jnp.abs(kids[0])
            raise TypeError(f"unsupported node in parametric: {type(e).__name__}")

        return ev(tree)

    return cell_fn, (n_consts, n_slots)
