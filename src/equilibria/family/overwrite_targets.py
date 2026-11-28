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


def _target_leaf(backend: Any, name: str, idx: tuple) -> Node:
    """The leaf of the target Var ``name`` at the single-period cell ``idx``, built
    from its domains: the recorded right-hand side is not parsed, since the recorder
    folds ``target x 1.0`` to ``target`` (every GTAP price is 1.0 at the base)."""
    doms = backend.var_domains[name]
    cells = list(itertools.product(*(backend.sets[d] for d in doms)))
    return Node("v", (name, cells.index(tuple(idx)), 1.0, not doms))


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
        # target x quantity[check] as a node of its own: ``*`` would fold a 1.0 (or 0.0)
        # away and the row would lose its target; a new list for the backend's tables
        records[fam] = [
            (
                idx,
                Residual(
                    res.lhs,
                    Node(
                        "mul",
                        (
                            _target_leaf(run.backend, name, idx),
                            node_value(res.lhs, leaf),
                        ),
                    ),
                ),
            )
            for idx, res in records[fam]
        ]
