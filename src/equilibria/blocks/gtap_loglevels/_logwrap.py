"""Log-wrap the EXISTING levels GTAP blocks (equilibria.blocks.gtap).

This template is our own levels model (blocks/gtap/) re-expressed in log form —
NOT the mivanic Julia port (that is blocks/gtap_logvalue/). Rather than transcribe
every equation by hand, we WRAP each levels block: its build_expression already
returns a Pyomo relational `lhs == rhs`; we rebuild it as `log(lhs) == log(rhs)`
when both sides are provably positive, and leave it raw otherwise.

`log(a)==log(b) ⟺ a==b` only for a,b>0. Many GTAP balances have a side that can be
≤0 (market clearing with net exports, savings, a constant 0.0 RHS, differences). For
those the log form is invalid, so `wrap` keeps the original equality. The log form is
applied only where it is both valid and numerically useful (products / CES / value
balances of strictly-positive quantities and prices).
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo
from pyomo.core.base.var import VarData
from pyomo.core.expr.numeric_expr import (
    DivisionExpression,
    MonomialTermExpression,
    PowExpression,
    ProductExpression,
)
from pyomo.core.expr.relational_expr import EqualityExpression

from equilibria.core.symbolic_equations import SymbolicEquation

# Expression types that are STRICTLY MULTIPLICATIVE (product / quotient / power /
# monomial c·x) — safe to log, since the model's positive lower bounds keep every
# factor > 0. Sums (LinearExpression / SumExpression) are NOT safe: a value balance
# with net exports, margins that are 0 on some routes, or the Walras residual (== 0
# in equilibrium) evaluates to 0 → log(0) = NaN. Those keep the raw equality.
_SAFE = (ProductExpression, DivisionExpression, PowExpression, MonomialTermExpression)


def _positive_side(expr: Any) -> bool:
    """True only if `expr` is a strictly-positive multiplicative term (or a single
    positive-bounded Var / positive constant) — never a sum/difference/zero."""
    if isinstance(expr, int | float):
        return expr > 0.0
    if isinstance(expr, VarData):
        return True  # model vars carry positive lower bounds (prices/quantities)
    return isinstance(expr, _SAFE)


def _eval_positive(expr: Any) -> bool:
    """True if `expr` currently evaluates to a strictly-positive finite number at the
    seed point. Log-wrapping is only valid — and only numerically safe for the solver's
    initial evaluation — when BOTH sides are >0 here (a side that is 0 at the seed, like
    rsav / walras / a zero-margin route, would give log(0)=NaN)."""
    from pyomo.environ import value

    if isinstance(expr, int | float):
        return expr > 0.0
    try:
        v = value(expr, exception=False)
    except Exception:
        return False
    return v is not None and v == v and v > 0.0


def _logify(rel: Any) -> Any:
    """Turn `lhs == rhs` into `log(lhs) == log(rhs)` when both sides are strictly
    positive multiplicative terms AND evaluate >0 at the seed; else keep it raw."""
    if rel is None:
        return None
    if not isinstance(rel, EqualityExpression):
        return rel
    lhs, rhs = rel.args
    if (
        _positive_side(lhs)
        and _positive_side(rhs)
        and _eval_positive(lhs)
        and _eval_positive(rhs)
    ):
        return pyo.log(lhs) == pyo.log(rhs)
    return rel


def log_wrap_block(levels_block_cls: type) -> type:
    """Return a subclass of a levels Block whose every emitted SymbolicEquation
    returns the log form where valid. The levels block's setup() is reused verbatim
    (same vars, params, masks); only each equation's build_expression is wrapped."""

    class _LogBlock(levels_block_cls):  # type: ignore[valid-type,misc]
        def setup(self, set_manager, parameters, variables):
            eqs = super().setup(set_manager, parameters, variables)
            wrapped: list[SymbolicEquation] = []
            for eq in eqs:
                wrapped.append(_wrap_equation(eq))
            return wrapped

    _LogBlock.__name__ = f"Log{levels_block_cls.__name__}"
    _LogBlock.__qualname__ = _LogBlock.__name__
    return _LogBlock


def _wrap_equation(eq: SymbolicEquation) -> SymbolicEquation:
    """Wrap one SymbolicEquation so build_expression returns the log form."""
    orig_build = eq.build_expression

    def _build(pyomo_model, indices, _orig=orig_build):
        return _logify(_orig(pyomo_model, indices))

    # bind the wrapped builder onto this instance
    object.__setattr__(eq, "build_expression", _build)
    return eq
