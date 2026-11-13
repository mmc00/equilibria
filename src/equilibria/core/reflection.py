"""Reflect a single-period model into periods (ADR-0006).

Blocks write their equations for one period, with no time index. ``reflect``
copies that single-period (SP) Pyomo model into a new one where every variable
and every equation gets a trailing period index ``t``: one copy of each row per
period, each reading its own period's variables. The equations are never
rewritten between periods; what changes between periods is the value of a
fixed variable (an instrument, ADR-0003).

A general version of ``GTAPMultiPeriodModel.build_vars`` and
``build_equations_all_periods`` (``templates/gtap/gtap_model_multiperiod.py``),
without the GTAP-only pieces.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any, cast

import pyomo.environ as pyo
from pyomo.core.expr.visitor import ExpressionReplacementVisitor


def _astuple(key: Any) -> tuple:
    return key if isinstance(key, tuple) else (key,)


def period_of(index: Any) -> Any:
    """The period of a reflected index: its last element."""
    return index[-1] if isinstance(index, tuple) else index


def exogenous(m: Any) -> frozenset[tuple[str, tuple]]:
    """``(variable name, index without t)`` of every cell fixed in the SP model."""
    return getattr(m, "_exogenous", frozenset())


def _check_periods(periods: Sequence[str]) -> list[str]:
    out = list(periods)
    if not out:
        raise ValueError("reflect needs at least one period")
    if len(set(out)) != len(out):
        raise ValueError(f"period names must be unique, got {out}")
    return out


def _reflect_vars(sp: Any, m: Any, periods: list[str]) -> set[tuple[str, tuple]]:
    fixed: set[tuple[str, tuple]] = set()
    for v in sp.component_objects(pyo.Var, active=True):
        cells = {_astuple(k): v[k] for k in v} if v.is_indexed() else {(): v[None]}
        index = [(*k, t) for k in cells for t in periods]
        first = next(iter(cells.values()))
        new: Any = pyo.Var(index, within=first.domain, dense=True)
        m.add_component(v.name, new)
        for k, cell in cells.items():
            if cell.fixed:
                fixed.add((v.name, k))
            for t in periods:
                target = cast("Any", new[(*k, t)])
                target.domain = cell.domain
                target.setlb(cell.lb)
                target.setub(cell.ub)
                target.set_value(cell.value, skip_validation=True)
                if cell.fixed:
                    target.fix()
    return fixed


def _substitutes(sp: Any, m: Any, period: str) -> dict[int, Any]:
    sub: dict[int, Any] = {}
    for v in sp.component_objects(pyo.Var, active=True):
        target = m.component(v.name)
        cells = {_astuple(k): v[k] for k in v} if v.is_indexed() else {(): v[None]}
        for k, cell in cells.items():
            sub[id(cell)] = target[(*k, period)]
    # a mutable parameter would stay a live reference to the throwaway SP model;
    # fold it to its value so the reflected rows are self-contained
    for prm in sp.component_objects(pyo.Param, active=True):
        if prm.mutable:
            for k in prm:
                with contextlib.suppress(ValueError, TypeError):
                    sub[id(prm[k])] = float(pyo.value(prm[k]))
    return sub


def _emit(body: Any, lb: Any, ub: Any) -> Any:
    if lb is not None and ub is not None and lb == ub:
        return body == lb
    if lb is not None and ub is not None:
        return (lb, body, ub)
    if lb is not None:
        return body >= lb
    if ub is not None:
        return body <= ub
    return pyo.Constraint.Skip


def _reflect_constraints(sp: Any, m: Any, periods: list[str]) -> None:
    visitors = {
        t: ExpressionReplacementVisitor(substitute=_substitutes(sp, m, t))
        for t in periods
    }
    for con in sp.component_objects(pyo.Constraint, active=True):
        rows: dict[tuple, tuple] = {}
        cells = (
            {_astuple(k): con[k] for k in con} if con.is_indexed() else {(): con[None]}
        )
        for t in periods:
            for k, cell in cells.items():
                if not cell.active:
                    continue
                rows[(*k, t)] = (
                    visitors[t].walk_expression(cell.body),
                    cell.lower,
                    cell.upper,
                )

        def rule(_m: Any, *key: Any, rows: dict = rows) -> Any:
            return _emit(*rows[key])

        m.add_component(con.name, pyo.Constraint(list(rows), rule=rule))


def reflect(sp: Any, periods: Sequence[str]) -> Any:
    """A new model with every variable and equation of ``sp`` copied per period.

    Each variable cell keeps its value, bounds, domain and fixed state in every
    period; cells fixed in ``sp`` are listed by :func:`exogenous`. Each equation
    is copied once per period, reading that period's variables. Objectives are
    not copied.
    """
    names = _check_periods(periods)
    m = pyo.ConcreteModel(name=sp.name)
    m.t = pyo.Set(initialize=names, ordered=True)
    m._exogenous = frozenset(_reflect_vars(sp, m, names))
    _reflect_constraints(sp, m, names)
    return m


def period_residual(m: Any, period: str) -> float:
    """Largest ``|body - bound|`` over the active equation cells of ``period``."""
    worst = 0.0
    for con in m.component_objects(pyo.Constraint, active=True):
        for k in con:
            cell = con[k]
            if not cell.active or period_of(k) != period:
                continue
            body = pyo.value(cell.body)
            lb, ub = cell.lb, cell.ub
            gap = 0.0
            if lb is not None and body < lb:
                gap = lb - body
            if ub is not None and body > ub:
                gap = max(gap, body - ub)
            if lb is not None and ub is not None and lb == ub:
                gap = abs(body - lb)
            worst = max(worst, gap)
    return worst
