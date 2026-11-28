"""``b.target`` rows anchored to the check on the family backend (MAR-251).

A hook's ``b.target`` row is recorded on the single-period model as
``quantity == target x Q``, with ``Q`` the quantity folded at recording time: the
BASE level. Pyomo's ``overwrite.shock_only`` rewrites it in the shock as
``quantity[shock] == target[shock] x quantity[check]``. Family solves period by
period, so when the shock is solved the check is a known point: ``anchor_targets``
puts ``quantity`` evaluated at the check in place of ``Q``, in the run's recording.
The row is active only in the shock (``MPArrays.build``), so no other period reads it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

from equilibria.family.recorder import Node, Residual

Leaf = Callable[[str, int], float]

_FUNCS = {"sqrt": math.sqrt, "log": math.log, "exp": math.exp}


def node_value(x: Any, leaf: Leaf) -> float:
    """A recorded expression's value with the Var levels given by ``leaf(var, flat)``."""
    if not isinstance(x, Node):
        return float(x)
    op, a = x.op, x.args
    if op == "v":
        return leaf(a[0], a[1])
    if op == "add":
        return float(sum(node_value(c, leaf) for c in a[: x._n]))
    if op == "mul":
        return node_value(a[0], leaf) * node_value(a[1], leaf)
    if op == "div":
        return node_value(a[0], leaf) / node_value(a[1], leaf)
    if op == "pow":
        return node_value(a[0], leaf) ** node_value(a[1], leaf)
    if op == "neg":
        return -node_value(a[0], leaf)
    return _FUNCS[op](node_value(a[0], leaf))


def leaf_reader(backend: Any, mp: Any, period: str) -> Leaf:
    """``leaf(var, flat)``: the level in ``mp`` at ``period`` of the single-period cell
    ``flat`` of ``var`` (its position in the product of the var's domains)."""
    keys: dict[str, list[tuple]] = {}

    def leaf(var: str, flat: int) -> float:
        if var not in keys:
            doms = backend.var_domains[var]
            keys[var] = list(itertools.product(*(backend.sets[d] for d in doms)))
        arr = mp.vars[var]
        return float(arr.value[arr.pos((*keys[var][flat], period))])

    return leaf


def _target_leaf(rhs: Any) -> Node:
    """The target Var of ``target x Q`` as ``BlockEdit.target`` records it."""
    if isinstance(rhs, Node) and rhs.op == "mul":
        for arg in rhs.args:
            if isinstance(arg, Node) and arg.op == "v":
                return arg
    raise ValueError(f"b.target row: expected target x constant, got {rhs!r}")


def anchor_targets(run: Any, mp: Any, period: str) -> None:
    """In the shock, each target row of ``mp.targets`` becomes
    ``quantity == target x quantity[check]`` in ``run.backend.records``; a no-op in any
    other period. Idempotent: ``quantity`` is read at the check every time."""
    from equilibria.blocks.gtap.overwrite import target_row
    from equilibria.blocks.gtap.periods import CHECK, SHOCK

    if period != SHOCK or not mp.targets:
        return
    records = run.backend.records
    leaf = leaf_reader(run.backend, mp, CHECK)
    for name in mp.targets:
        fam = target_row(name)
        if fam not in records:
            continue
        # a new list: the backend's compiled tables are keyed by the list itself
        records[fam] = [
            (idx, Residual(res.lhs, _target_leaf(res.rhs) * node_value(res.lhs, leaf)))
            for idx, res in records[fam]
        ]
