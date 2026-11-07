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

    A leaf ('v') holds ``(var_name, flat_index, level, is_scalar)``. An 'add' node holds a list that
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
        # Pyomo's ScalarVar subclasses VarData: `indexed_cell == scalar_var` dispatches
        # to the scalar's __eq__ first, so Pyomo's body is scalar - cell. Mirror it.
        if (
            self.op == "v"
            and not self.args[3]
            and isinstance(o, Node)
            and o.op == "v"
            and o.args[3]
        ):
            return Residual(o, self)
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


class _MutableParamView(_Indexed):
    """A mutable Parameter stays symbolic in Pyomo: here a leaf whose value comes from
    the state at evaluation time (a fixed slot), not a folded constant."""

    def __getitem__(self, key) -> Node:
        k = self.flat(key)
        return Node("v", (self.name, k, self.data[k], False))


class _VarView(_Indexed):
    def __getitem__(self, key) -> Node:
        k = self.flat(key)
        return Node("v", (self.name, k, self.data[k], False))


class RecordingModel:
    """Pyomo-like view of an equilibria Model's sets, parameters and variables."""

    def __init__(
        self,
        sets: dict[str, list[str]],
        params: dict[str, tuple],
        variables: dict[str, tuple[tuple[str, ...], np.ndarray]],
    ):
        d = self.__dict__
        d["_sets"] = {k: list(v) for k, v in sets.items()}
        pos = {k: {e: i for i, e in enumerate(v)} for k, v in d["_sets"].items()}
        d["_pos"] = pos
        # params: name -> (domains, values[, mutable])
        d["_params"] = {
            k: (v[0], v[1], bool(v[2]) if len(v) > 2 else False)
            for k, v in params.items()
        }
        d["_vars"] = variables
        d["var_domains"] = {k: tuple(v[0]) for k, v in variables.items()}
        d["param_domains"] = {k: tuple(v[0]) for k, v in d["_params"].items() if v[2]}

    @classmethod
    def from_model(cls, model: Any) -> RecordingModel:
        sm, pm, vm = model.set_manager, model.parameter_manager, model.variable_manager
        sets = {n: list(sm.get(n).iter_elements()) for n in sm.list_sets()}
        params = {
            n: (
                tuple(p.domains),
                np.asarray(p.value, float),
                bool(getattr(p, "mutable", False)),
            )
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
            doms, arr, mutable = d["_params"][name]
            if mutable:
                view = (
                    Node("v", (name, 0, float(arr.ravel()[0]), False))
                    if not doms
                    else _MutableParamView(name, doms, d["_pos"], arr)
                )
            else:
                view = (
                    float(arr.ravel()[0])
                    if not doms
                    else _ParamView(name, doms, d["_pos"], arr)
                )
        elif name in d["_vars"]:
            doms, arr = d["_vars"][name]
            view = (
                Node("v", (name, 0, float(arr.ravel()[0]), True))
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


# ----------------------------------------------------------------------------------
# Shapes, templates and term groups
# ----------------------------------------------------------------------------------


def signature(x) -> tuple[tuple, list[tuple[str, int]], list[float]]:
    """(shape key, variable leaves, constants) in one DFS. Leaves and constants become
    the slots and data of the term; the key says which template evaluates it."""
    leaves: list[tuple[str, int]] = []
    consts: list[float] = []
    key = _sig(x, leaves, consts)
    return key, leaves, consts


def _sig(x, leaves, consts):
    if not isinstance(x, Node):
        consts.append(float(x))
        return "c"
    if x.op == "v":
        leaves.append((x.args[0], x.args[1]))
        return "v"
    return (x.op, *(_sig(c, leaves, consts) for c in x.children()))


def fold_zeros(x, is_zero) -> Any:
    """Rebuild ``x`` with the leaves ``is_zero((var, flat))`` replaced by 0.0, through the
    same simplifying algebra (a product with 0 vanishes, ``y**0`` is 1). Mirrors the
    ``.nl`` writer, which substitutes fixed variables by their values."""
    if not isinstance(x, Node):
        return x
    if x.op == "v":
        return 0.0 if is_zero((x.args[0], x.args[1])) else x
    kids = [fold_zeros(c, is_zero) for c in x.children()]
    if x.op == "add":
        out = 0.0
        for k in kids:
            out = _add(out, k)
        return out
    if x.op == "mul":
        return _mul(kids[0], kids[1])
    if x.op == "div":
        return _div(kids[0], kids[1])
    if x.op == "pow":
        return _pow(kids[0], kids[1])
    if x.op == "neg":
        return _neg(kids[0])
    return intrinsic(x.op)(kids[0])


_TEMPLATES: dict[tuple, Any] = {}


def template(key) -> Any:
    """``fn(xs, c, s) = s * term(xs, c)`` for a shape key; one function object per key so
    the jitted kernels reuse their compilation."""
    fn = _TEMPLATES.get(key)
    if fn is None:
        import jax.numpy as jnp

        jfuncs = {"sqrt": jnp.sqrt, "log": jnp.log, "exp": jnp.exp}
        counter = {"v": 0, "c": 0}

        def build(k):
            if k == "v":
                i = counter["v"]
                counter["v"] += 1
                return lambda xs, c: xs[i]
            if k == "c":
                j = counter["c"]
                counter["c"] += 1
                return lambda xs, c: c[j]
            op, *kids = k
            fs = [build(kk) for kk in kids]
            if op == "add":
                return lambda xs, c: sum(f(xs, c) for f in fs)
            if op == "mul":
                return lambda xs, c: fs[0](xs, c) * fs[1](xs, c)
            if op == "div":
                return lambda xs, c: fs[0](xs, c) / fs[1](xs, c)
            if op == "pow":
                return lambda xs, c: fs[0](xs, c) ** fs[1](xs, c)
            if op == "neg":
                return lambda xs, c: -fs[0](xs, c)
            g = jfuncs[op]
            return lambda xs, c: g(fs[0](xs, c))

        body = build(key)

        def fn(xs, c, s):
            return s * body(xs, c)

        fn.__name__ = f"term_{len(_TEMPLATES)}"
        # stable across processes: the executable cache key
        fn.__dict__["_aot_key"] = repr(key)
        _TEMPLATES[key] = fn
    return fn


class _Bucket:
    """The terms of one shape: lists while the table is read, arrays after."""

    __slots__ = ("key", "k", "m", "rows", "codes", "flats", "c", "s", "t", "terms")

    def __init__(self, key: tuple, k: int, m: int):
        self.key, self.k, self.m = key, k, m
        self.rows: Any = []
        self.codes: Any = []
        self.flats: Any = []
        self.c: Any = []
        self.s: Any = []
        self.t: Any = []
        self.terms: list = []

    def freeze(self) -> None:
        n = len(self.rows)
        self.rows = np.asarray(self.rows, dtype=np.int64)
        self.codes = np.asarray(self.codes, dtype=np.int64).reshape(n, self.k)
        self.flats = np.asarray(self.flats, dtype=np.int64).reshape(n, self.k)
        self.c = np.asarray(self.c, dtype=float).reshape(n, self.m)
        self.s = np.asarray(self.s, dtype=float)
        self.t = np.asarray(self.t, dtype=np.int64)


# Terms re-simplified because they read a cell fixed at 0 (a test hook: no output reads it).
fold_stats: dict[str, int] = {"folded": 0}


def reset_fold_stats() -> None:
    fold_stats["folded"] = 0


# A folded term can split into several: position = stream index * _SPLIT + split number.
_SPLIT = 1 << 16


class TermTable:
    """The signed terms of a family's records grouped by shape, built once.

    Which cells are fixed changes with the call, the terms do not: per layout only the
    terms that read a cell fixed at 0 are re-simplified (``fold_zeros``), the others are
    gathered as arrays. The groups are those ``compile_family`` has always produced
    (buckets by first appearance, entries in stream order inside a bucket)."""

    def __init__(self, records) -> None:
        self.labels = [",".join(str(i) for i in idx) for idx, _ in records]
        self.vars: dict[str, int] = {}
        buckets: dict[tuple, _Bucket] = {}
        t = 0
        for row, (_, res) in enumerate(records):
            for sgn, term in residual_terms(res):
                key, leaves, consts = signature(term)
                b = buckets.get(key)
                if b is None:
                    b = buckets[key] = _Bucket(key, len(leaves), len(consts))
                b.rows.append(row)
                for var, flat in leaves:
                    b.codes.append(self.vars.setdefault(var, len(self.vars)))
                    b.flats.append(flat)
                b.c.extend(consts)
                b.s.append(sgn)
                b.t.append(t)
                b.terms.append(term)
                t += 1
        self.buckets = list(buckets.values())
        for b in self.buckets:
            b.freeze()

    def groups(self, layout) -> list:
        from equilibria.family.core import TermGroup

        flat_slots = {v: np.asarray(layout.slots[v]).ravel() for v in self.vars}
        n_cols = layout.n_cols
        fixed = np.asarray(layout.fixed_values, dtype=float)
        sizes = [len(flat_slots[v]) for v in self.vars]
        off = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)]).astype(np.int64)
        all_slots = (
            np.concatenate([flat_slots[v] for v in self.vars])
            if self.vars
            else np.zeros(0, dtype=np.int64)
        )
        zero = np.zeros(len(all_slots), dtype=bool)
        is_fixed = all_slots >= n_cols
        zero[is_fixed] = fixed[all_slots[is_fixed] - n_cols] == 0.0

        def is_fixed_zero(leaf) -> bool:
            slot = flat_slots[leaf[0]][leaf[1]]
            return bool(slot >= n_cols and fixed[slot - n_cols] == 0.0)

        # per shape: parts of (position, rows, slots, c, s), merged by position below
        parts: dict[tuple, list] = {}
        meta: dict[tuple, tuple[int, int]] = {}
        folded: dict[tuple, dict[str, list]] = {}
        for b in self.buckets:
            gl = off[:-1][b.codes] + b.flats  # (n, k) positions in all_slots
            slots = all_slots[gl]
            hit = zero[gl].any(axis=1) if b.k else np.zeros(len(b.rows), dtype=bool)
            meta[b.key] = (b.k, b.m)
            fold_stats["folded"] += int(hit.sum())
            for i in np.flatnonzero(hit).tolist():
                f = fold_zeros(b.terms[i], is_fixed_zero)
                for j, (s2, t2) in enumerate(residual_terms(Residual(f, 0.0))):
                    key2, leaves2, consts2 = signature(t2)
                    meta.setdefault(key2, (len(leaves2), len(consts2)))
                    d = folded.setdefault(
                        key2, {"pos": [], "rows": [], "slots": [], "c": [], "s": []}
                    )
                    d["pos"].append(int(b.t[i]) * _SPLIT + j)
                    d["rows"].append(int(b.rows[i]))
                    d["slots"].extend(flat_slots[v][fl] for v, fl in leaves2)
                    d["c"].extend(consts2)
                    d["s"].append(float(b.s[i]) * s2)
            keep = ~hit
            parts.setdefault(b.key, []).append(
                (b.t[keep] * _SPLIT, b.rows[keep], slots[keep], b.c[keep], b.s[keep])
            )
        for key2, d in folded.items():
            k, m = meta[key2]
            n = len(d["rows"])
            parts.setdefault(key2, []).append(
                (
                    np.asarray(d["pos"], dtype=np.int64),
                    np.asarray(d["rows"], dtype=np.int64),
                    np.asarray(d["slots"], dtype=np.int64).reshape(n, k),
                    np.asarray(d["c"], dtype=float).reshape(n, m),
                    np.asarray(d["s"], dtype=float),
                )
            )
        merged = []
        for key, ps in parts.items():
            pos = np.concatenate([p[0] for p in ps])
            if len(pos) == 0:
                continue  # every term of the shape folded away
            o = np.argsort(pos, kind="stable") if len(ps) > 1 else slice(None)
            merged.append(
                (
                    int(pos[o][0]),
                    key,
                    np.concatenate([p[1] for p in ps])[o],
                    np.concatenate([p[2] for p in ps])[o],
                    np.concatenate([p[3] for p in ps])[o],
                    np.concatenate([p[4] for p in ps])[o],
                )
            )
        merged.sort(key=lambda g: g[0])  # buckets by first appearance
        return [
            TermGroup(rows=rows, fn=template(key), cols=cols, args={"c": c, "s": s})
            for _, key, rows, cols, c, s in merged
        ]


def compile_family(name: str, records, layout, table: TermTable | None = None) -> Any:
    """Recorded cells of one equation -> ``CompiledFamily`` (rows + term groups by shape).

    ``records`` is ``[(indices, Residual), ...]`` for the cells the rule did not skip.
    ``table`` is ``TermTable(records)``, kept by a caller that compiles several layouts.
    """
    from equilibria.family.core import CompiledFamily

    if table is None:
        table = TermTable(records)
    return CompiledFamily(
        name=name, row_labels=list(table.labels), groups=table.groups(layout)
    )


__all__ = [
    "TermTable",
    "compile_family",
    "signature",
    "template",
    "Node",
    "Number",
    "RecordingModel",
    "Residual",
    "intrinsic",
    "residual_terms",
    "var_shape",
]
