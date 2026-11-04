"""Record block equations as small expression trees, without Pyomo.

``RecordingModel`` stands in for the Pyomo model a block's ``build_expression`` expects:
sets iterate as element lists, parameters index to floats, variables index to leaf
``Node``\\ s. Running a rule against it leaves a tree whose leaves are variable cells and
constants. The tree mirrors the structural simplifications of Pyomo's ``.nl`` writer that
change the Jacobian pattern (a product with a literal 0 vanishes, ``x**0`` is 1).

Structure is recorded once and does not depend on which cells are fixed: the variable
layout is applied afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Number
from typing import Any

import numpy as np

_NUM = (int, float, np.floating, np.integer)


def _is_num(x) -> bool:
    return isinstance(x, _NUM)


class Node:
    """Expression node. ``op`` in {'v', 'add', 'mul', 'div', 'pow', 'neg', 'sqrt', 'log', 'exp'}.

    A leaf ('v') holds ``(var_name, flat_index, level)``. An 'add' node holds a list that
    may be shared with the node it was extended from (``_n`` is its own length), so
    ``sum()`` over n terms is O(n), as in Pyomo.
    """

    __slots__ = ("op", "args", "_n")

    def __init__(self, op: str, args, n: int = 0):
        self.op = op
        self.args = args
        self._n = n

    # --- Pyomo duck typing so pyomo.environ.value() evaluates at build-time levels ---
    def is_numeric_type(self) -> bool:
        return True

    def is_constant(self) -> bool:
        return False

    def is_expression_type(self, *_a, **_k) -> bool:
        return self.op != "v"

    def __call__(self, exception: bool = True) -> float:
        return self.evaluate()

    def evaluate(self) -> float:
        op, a = self.op, self.args
        if op == "v":
            return float(a[2])
        if op == "add":
            return float(sum(_ev(c) for c in a[: self._n]))
        if op == "mul":
            return _ev(a[0]) * _ev(a[1])
        if op == "div":
            return _ev(a[0]) / _ev(a[1])
        if op == "pow":
            return _ev(a[0]) ** _ev(a[1])
        if op == "neg":
            return -_ev(a[0])
        return _FUNCS[op](_ev(a[0]))

    def children(self):
        return self.args[: self._n] if self.op == "add" else self.args

    def var_leaves(self) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        _collect_leaves(self, out)
        return out

    # --- algebra ---
    def __add__(self, o):
        return _add(self, o)

    def __radd__(self, o):
        return _add(o, self)

    def __sub__(self, o):
        return _add(self, _neg(o))

    def __rsub__(self, o):
        return _add(o, _neg(self))

    def __mul__(self, o):
        return _mul(self, o)

    def __rmul__(self, o):
        return _mul(o, self)

    def __truediv__(self, o):
        return _div(self, o)

    def __rtruediv__(self, o):
        return _div(o, self)

    def __pow__(self, o):
        return _pow(self, o)

    def __rpow__(self, o):
        return _pow(o, self)

    def __neg__(self):
        return _neg(self)

    def __pos__(self):
        return self

    def __eq__(self, o):  # type: ignore[override]
        return Residual(self, o)

    __hash__ = object.__hash__

    def __repr__(self) -> str:
        if self.op == "v":
            return f"{self.args[0]}#{self.args[1]}"
        return f"{self.op}({', '.join(map(repr, self.children()))})"


class Residual:
    """``lhs == rhs`` as returned by a rule; evaluates to ``lhs - rhs`` like PyomoNLP."""

    __slots__ = ("lhs", "rhs")

    def __init__(self, lhs, rhs):
        self.lhs, self.rhs = lhs, rhs

    def evaluate(self) -> float:
        return _ev(self.lhs) - _ev(self.rhs)


def _ev(x) -> float:
    return x.evaluate() if isinstance(x, Node) else float(x)


def _collect_leaves(x, out) -> None:
    if not isinstance(x, Node):
        return
    if x.op == "v":
        out.append((x.args[0], x.args[1]))
        return
    for c in x.children():
        _collect_leaves(c, out)


def _add(a, b):
    if _is_num(a) and _is_num(b):
        return float(a) + float(b)
    if _is_num(b):
        a, b = b, a
    if _is_num(a):
        if a == 0:
            return b
        return Node("add", [b, float(a)], 2)
    # both nodes
    if a.op == "add" and a._n == len(a.args):
        a.args.append(b)
        return Node("add", a.args, a._n + 1)
    if a.op == "add":
        lst = list(a.args[: a._n]) + [b]
        return Node("add", lst, len(lst))
    return Node("add", [a, b], 2)


def _neg(a):
    if _is_num(a):
        return -float(a)
    if a.op == "neg":
        return a.args[0]
    return Node("neg", (a,))


def _mul(a, b):
    if _is_num(a) and _is_num(b):
        return float(a) * float(b)
    if _is_num(a):
        if a == 0:
            return 0.0
        if a == 1:
            return b
        return Node("mul", (float(a), b))
    if _is_num(b):
        if b == 0:
            return 0.0
        if b == 1:
            return a
        return Node("mul", (a, float(b)))
    return Node("mul", (a, b))


def _div(a, b):
    if _is_num(a) and _is_num(b):
        return float(a) / float(b)
    if _is_num(a) and a == 0:
        return 0.0
    if _is_num(b) and b == 1:
        return a
    return Node("div", (float(a) if _is_num(a) else a, float(b) if _is_num(b) else b))


def _pow(a, b):
    if _is_num(a) and _is_num(b):
        return float(a) ** float(b)
    if _is_num(b):
        if b == 0:
            return 1.0
        if b == 1:
            return a
        return Node("pow", (a, float(b)))
    return Node("pow", (float(a) if _is_num(a) else a, b))


_FUNCS = {"sqrt": math.sqrt, "log": math.log, "exp": math.exp}


def intrinsic(name: str):
    """A drop-in for pyomo's ``sqrt``/``log``/``exp`` that records on nodes."""
    fn = _FUNCS[name]

    def f(x):
        return Node(name, (x,)) if isinstance(x, Node) else fn(float(x))

    f.__name__ = name
    return f


def residual_terms(res: Residual) -> list[tuple[float, Any]]:
    """Signed additive terms of ``lhs - rhs`` (numbers or non-additive nodes)."""
    out: list[tuple[float, Any]] = []
    _terms(res.lhs, 1.0, out)
    _terms(res.rhs, -1.0, out)
    return out


def _terms(x, s: float, out) -> None:
    if _is_num(x):
        if x != 0:
            out.append((s, float(x)))
        return
    if x.op == "add":
        for c in x.args[: x._n]:
            _terms(c, s, out)
    elif x.op == "neg":
        _terms(x.args[0], -s, out)
    else:
        out.append((s, x))


# ----------------------------------------------------------------------------------
# The stand-in model
# ----------------------------------------------------------------------------------


class _Indexed:
    """Index labels -> flat position over a fixed domain (C order)."""

    __slots__ = ("name", "domains", "pos", "strides", "data")

    def __init__(self, name, domains, pos_maps, data):
        self.name = name
        self.domains = domains
        self.pos = [pos_maps[d] for d in domains]
        shape = [len(p) for p in self.pos]
        strides, acc = [], 1
        for n in reversed(shape):
            strides.append(acc)
            acc *= n
        self.strides = strides[::-1]
        self.data = np.asarray(data, dtype=float).ravel()

    def flat(self, key) -> int:
        if len(self.pos) == 1:
            return self.pos[0][key[0] if isinstance(key, tuple) else key]
        return sum(
            p[k] * s for p, k, s in zip(self.pos, key, self.strides, strict=True)
        )

    def __contains__(self, key) -> bool:
        try:
            self.flat(key)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def keys(self) -> Iterable:
        import itertools

        labels = [list(p) for p in self.pos]
        for k in itertools.product(*labels):
            yield k[0] if len(k) == 1 else k

    def __iter__(self):
        return self.keys()


class _ParamView(_Indexed):
    def __getitem__(self, key) -> float:
        return float(self.data[self.flat(key)])


class _VarView(_Indexed):
    def __getitem__(self, key) -> Node:
        k = self.flat(key)
        return Node("v", (self.name, k, self.data[k]))


class RecordingModel:
    """Pyomo-like view of an equilibria Model's sets, parameters and variables."""

    def __init__(
        self,
        sets: dict[str, list[str]],
        params: dict[str, tuple[tuple[str, ...], np.ndarray]],
        variables: dict[str, tuple[tuple[str, ...], np.ndarray]],
    ):
        d = self.__dict__
        d["_sets"] = {k: list(v) for k, v in sets.items()}
        pos = {k: {e: i for i, e in enumerate(v)} for k, v in d["_sets"].items()}
        d["_pos"] = pos
        d["_params"] = params
        d["_vars"] = variables
        d["var_domains"] = {k: tuple(v[0]) for k, v in variables.items()}

    @classmethod
    def from_model(cls, model: Any) -> RecordingModel:
        sm, pm, vm = model.set_manager, model.parameter_manager, model.variable_manager
        sets = {n: list(sm.get(n).iter_elements()) for n in sm.list_sets()}
        params = {
            n: (tuple(p.domains), np.asarray(p.value, float))
            for n in pm.list_params()
            for p in [pm.get(n)]
        }
        variables = {
            n: (tuple(v.domains), np.asarray(v.value, float))
            for n in vm.list_vars()
            for v in [vm.get(n)]
        }
        return cls(sets, params, variables)

    def __getattr__(self, name: str):
        d = self.__dict__
        if name in d["_sets"]:
            return d["_sets"][name]
        if name in d["_params"]:
            doms, arr = d["_params"][name]
            view = (
                float(arr.ravel()[0])
                if not doms
                else _ParamView(name, doms, d["_pos"], arr)
            )
        elif name in d["_vars"]:
            doms, arr = d["_vars"][name]
            view = (
                Node("v", (name, 0, float(arr.ravel()[0])))
                if not doms
                else _VarView(name, doms, d["_pos"], arr)
            )
        else:
            raise AttributeError(name)
        d[name] = view  # cache: later accesses skip __getattr__
        return view

    def __setattr__(self, name, val):
        raise AttributeError(f"RecordingModel is read-only ({name})")

    @property
    def sets(self) -> dict[str, list[str]]:
        return self.__dict__["_sets"]


def var_shape(rec: RecordingModel, name: str) -> tuple[int, ...]:
    return tuple(len(rec.sets[d]) for d in rec.var_domains[name])


__all__ = [
    "Node",
    "Number",
    "RecordingModel",
    "Residual",
    "intrinsic",
    "residual_terms",
    "var_shape",
]
