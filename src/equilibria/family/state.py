"""Mutable variable state over an equilibria Model's arrays, with a Pyomo-like face.

The composer's post-build steps (benchmark scaling, Fisher snapshot reseed, cell fixes)
are written against a Pyomo model: ``value(m.x[k])``, ``m.x[k].set_value(v)``,
``.fix(v)``, ``.fixed``, ``.lb``/``.setlb``, iteration and ``find_component``.
``StateModel`` offers exactly that over numpy arrays, so the SAME step functions update
levels, fixed flags and bounds that any backend can then read.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


class _Cell:
    """One variable cell: level, fixed flag and bounds live in the parent arrays."""

    __slots__ = ("_v", "_k")

    def __init__(self, var: _StateVar, flat: int):
        self._v = var
        self._k = flat

    # --- Pyomo VarData surface ---
    @property
    def value(self) -> float:
        return float(self._v._vals.flat[self._k])

    @value.setter
    def value(self, v: float) -> None:
        self.set_value(v)

    def set_value(self, v, skip_validation: bool = False) -> None:
        self._v._vals.flat[self._k] = np.nan if v is None else float(v)

    def fix(self, v=None) -> None:
        if v is not None:
            self.set_value(v)
        self._v._fx.flat[self._k] = True

    def unfix(self) -> None:
        self._v._fx.flat[self._k] = False

    @property
    def fixed(self) -> bool:
        return bool(self._v._fx.flat[self._k])

    @property
    def lb(self):
        lo = float(self._v._lo.flat[self._k])
        return None if lo == -np.inf else lo

    @property
    def ub(self):
        hi = float(self._v._hi.flat[self._k])
        return None if hi == np.inf else hi

    def setlb(self, v) -> None:
        self._v._lo.flat[self._k] = -np.inf if v is None else float(v)

    def setub(self, v) -> None:
        self._v._hi.flat[self._k] = np.inf if v is None else float(v)

    @property
    def name(self) -> str:
        return self._v.cell_name(self._k)

    def is_numeric_type(self) -> bool:
        return True

    def is_constant(self) -> bool:
        return False

    def is_fixed(self) -> bool:
        return self.fixed

    def __call__(self, exception: bool = True) -> float:
        return self.value

    def __float__(self) -> float:
        return self.value

    # Arithmetic evaluates at the current level (the steps only ever take value()).
    def __add__(self, o):
        return self.value + float(o)

    __radd__ = __add__

    def __sub__(self, o):
        return self.value - float(o)

    def __rsub__(self, o):
        return float(o) - self.value

    def __mul__(self, o):
        return self.value * float(o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self.value / float(o)

    def __rtruediv__(self, o):
        return float(o) / self.value

    def __pow__(self, o):
        return self.value ** float(o)

    def __neg__(self):
        return -self.value


class _Indexed:
    def __init__(self, name: str, domains: tuple[str, ...], sets: dict[str, list[str]]):
        self.name = name
        self.domains = domains
        self._labels = [sets[d] for d in domains]
        self._pos = [{e: i for i, e in enumerate(lab)} for lab in self._labels]
        self.shape = tuple(len(lab) for lab in self._labels)

    def flat(self, key) -> int:
        if not self.domains:
            return 0
        if len(self.domains) == 1:
            return self._pos[0][key[0] if isinstance(key, tuple) else key]
        return int(
            np.ravel_multi_index(
                tuple(p[k] for p, k in zip(self._pos, key, strict=True)), self.shape
            )
        )

    def keys(self):
        for k in itertools.product(*self._labels):
            yield k[0] if len(k) == 1 else k

    def __iter__(self):
        return self.keys()

    def __contains__(self, key) -> bool:
        try:
            self.flat(key)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def __len__(self) -> int:
        return int(np.prod(self.shape)) if self.domains else 1

    def is_indexed(self) -> bool:
        return bool(self.domains)

    def index_set(self):
        return list(self.keys())

    def cell_name(self, flat: int) -> str:
        if not self.domains:
            return self.name
        idx = np.unravel_index(flat, self.shape)
        return f"{self.name}[{','.join(lab[i] for lab, i in zip(self._labels, idx, strict=True))}]"


class _StateVar(_Indexed):
    def __init__(self, name, domains, sets, values, fixed, lower, upper):
        super().__init__(name, domains, sets)
        self._vals, self._fx, self._lo, self._hi = values, fixed, lower, upper

    def __getitem__(self, key) -> _Cell:
        return _Cell(self, self.flat(key))

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def values(self):
        for k in self.keys():
            yield self[k]

    # Scalar variables: the component is its own cell (as a Pyomo ScalarVar).
    def _scalar(self) -> _Cell:
        return _Cell(self, 0)

    def set_value(self, v, skip_validation: bool = False) -> None:
        self._scalar().set_value(v, skip_validation)

    def fix(self, v=None) -> None:
        self._scalar().fix(v)

    @property
    def value(self) -> float:
        return self._scalar().value

    def is_numeric_type(self) -> bool:
        return not self.domains

    def __call__(self, exception: bool = True) -> float:
        return self._scalar().value

    def __float__(self) -> float:
        return self._scalar().value


class _ParamCell:
    """A mutable Parameter cell (Pyomo ParamData): float-like, with ``set_value``."""

    __slots__ = ("_p", "_k")

    def __init__(self, param: _StateParam, flat: int):
        self._p = param
        self._k = flat

    @property
    def value(self) -> float:
        return float(self._p.data.flat[self._k])

    def set_value(self, v, skip_validation: bool = False) -> None:
        self._p.data.flat[self._k] = float(v)

    def is_numeric_type(self) -> bool:
        return True

    def is_constant(self) -> bool:
        return False

    def __call__(self, exception: bool = True) -> float:
        return self.value

    def __float__(self) -> float:
        return self.value

    def __add__(self, o):
        return self.value + float(o)

    __radd__ = __add__

    def __sub__(self, o):
        return self.value - float(o)

    def __rsub__(self, o):
        return float(o) - self.value

    def __mul__(self, o):
        return self.value * float(o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self.value / float(o)

    def __rtruediv__(self, o):
        return float(o) / self.value

    def __lt__(self, o):
        return self.value < float(o)

    def __le__(self, o):
        return self.value <= float(o)

    def __gt__(self, o):
        return self.value > float(o)

    def __ge__(self, o):
        return self.value >= float(o)


class _StateParam(_Indexed):
    """Parameter values. Immutable cells index to floats (Pyomo folds them); mutable
    ones to ``_ParamCell`` so steps can ``set_value`` them, as on a mutable Param."""

    def __init__(self, name, domains, sets, data, mutable: bool = False):
        super().__init__(name, domains, sets)
        self.data = data
        self.mutable = mutable

    def __getitem__(self, key):
        k = self.flat(key)
        return _ParamCell(self, k) if self.mutable else float(self.data.flat[k])

    def set_value(self, v, skip_validation: bool = False) -> None:
        self.data.flat[0] = float(v)

    @property
    def value(self) -> float:
        return float(self.data.flat[0])

    def is_numeric_type(self) -> bool:
        return not self.domains

    def __call__(self, exception: bool = True) -> float:
        return float(self.data.flat[0])

    def __float__(self) -> float:
        return float(self.data.flat[0])


def _as_shape(x, shape: tuple[int, ...]) -> np.ndarray:
    """``x`` as a writable float array of ``shape`` (exact reshape or broadcast)."""
    arr = np.asarray(x, dtype=float)
    if arr.size == int(np.prod(shape)):
        return arr.reshape(shape).copy()
    return np.broadcast_to(arr, shape).copy()


class StateModel:
    """Levels, fixed flags and bounds of every variable of an equilibria Model."""

    def __init__(self, model: Any):
        sm, pm, vm = model.set_manager, model.parameter_manager, model.variable_manager
        self.__dict__["sets"] = {
            n: list(sm.get(n).iter_elements()) for n in sm.list_sets()
        }
        sets = self.sets
        self.__dict__["var_domains"] = {}
        self.__dict__["values"] = {}
        self.__dict__["fixed"] = {}
        self.__dict__["lower"] = {}
        self.__dict__["upper"] = {}
        self.__dict__["_components"] = {}
        # Variables Pyomo declares NonNegativeReals (any other domain is free): the
        # multiperiod reflection takes its only bound, lb 0, from this.
        self.__dict__["nonneg_vars"] = set()
        for n in vm.list_vars():
            v = vm.get(n)
            doms = tuple(v.domains)
            shape = tuple(len(sets[d]) for d in doms)
            vals = _as_shape(v.value, shape)
            lo = _as_shape(v.lower, shape)
            if getattr(v, "domain", "Reals") == "NonNegativeReals":
                lo = np.maximum(lo, 0.0)
                self.nonneg_vars.add(n)
            hi = _as_shape(v.upper, shape)
            fx = np.zeros(shape, dtype=bool)
            self.var_domains[n] = doms
            self.values[n], self.fixed[n], self.lower[n], self.upper[n] = (
                vals,
                fx,
                lo,
                hi,
            )
            self._components[n] = _StateVar(n, doms, sets, vals, fx, lo, hi)
        for n in pm.list_params():
            p = pm.get(n)
            doms = tuple(p.domains)
            shape = tuple(len(sets[d]) for d in doms)
            self._components[n] = _StateParam(
                n,
                doms,
                sets,
                _as_shape(p.value, shape),
                bool(getattr(p, "mutable", False)),
            )

    def __getattr__(self, name: str):
        comps = self.__dict__["_components"]
        if name in comps:
            return comps[name]
        sets = self.__dict__["sets"]
        if name in sets:
            return sets[name]
        raise AttributeError(name)

    def __setattr__(self, name, val) -> None:
        # Steps may tag the model (pm._something = ...); keep those as plain attributes.
        self.__dict__[name] = val

    def find_component(self, name: str):
        try:
            return getattr(self, name)
        except AttributeError:
            return None

    component = find_component
