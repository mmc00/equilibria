"""Adapter presenting Pyomo's ``ConcreteModel`` surface over PyOptInterface.

The GTAP blocks write their equations in Pyomo syntax::

    model.pf[r, f, a] * model.xf[r, f, a]
    sum(model.xa[r, i, a] for a in model.a)
    if value(model.xfflag[r, f, a]):

Those bodies never name Pyomo directly — they only use ``__getattr__`` for
variables, sets and parameters, ``__getitem__`` for indexing, and Python's own
operators. Presenting the same surface over PyOptInterface lets the 5,910 lines of
block equations run unmodified against a second backend, which is the assumption
Fase 0 exists to test.

Three details of Pyomo's semantics matter enough to state:

* **Variable identity.** ``model.px['USA', 'Food']`` is one variable no matter how
  often it is read. Handles are cached per key; minting a fresh one per access
  would split a shared cell and silently enlarge the system.
* **Parameters are values, not handles.** Blocks branch on parameters at build
  time. A handle would make every such condition truthy and change which
  constraints exist.
* **A missing name is an error.** Returning ``None`` would defer the failure to an
  inscrutable expression error much later.
"""

from __future__ import annotations

from typing import Any


class PoiVarProxy:
    """One variable family: ``model.px`` here, ``model.px[r, i]`` a POI handle.

    Handles are created lazily — the blocks touch a subset of any family's index
    space, and materializing the full cartesian product would build variables no
    equation ever mentions.
    """

    __slots__ = ("_model", "_name", "_domains", "_cache")

    def __init__(self, poi_model: Any, name: str, domains: tuple[str, ...]) -> None:
        self._model = poi_model
        self._name = name
        self._domains = domains
        self._cache: dict[tuple[Any, ...], Any] = {}

    def __getitem__(self, key: Any) -> Any:
        k = key if isinstance(key, tuple) else (key,)
        handle = self._cache.get(k)
        if handle is None:
            label = f"{self._name}[{','.join(map(str, k))}]" if k else self._name
            handle = self._model.add_variable(name=label)
            self._cache[k] = handle
        return handle

    def __iter__(self):
        """Iterate the keys built so far, like a Pyomo indexed Var."""
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def domains(self) -> tuple[str, ...]:
        return self._domains

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"PoiVarProxy({self._name!r}, domains={self._domains}, built={len(self._cache)})"


class PoiModelAdapter:
    """Stands in for the ``pyomo_model`` argument of ``build_expression``.

    Resolution order on attribute access is variables, then sets, then parameters.
    It mirrors Pyomo, where all three share one namespace on the model object, and
    the GTAP names do not collide across the three kinds.
    """

    def __init__(
        self,
        poi_model: Any,
        sets: dict[str, list[str]],
        params: Any,
        var_specs: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        # Bypass __getattr__ during construction: these are real instance
        # attributes, and a partially built adapter must not recurse.
        object.__setattr__(self, "_model", poi_model)
        object.__setattr__(self, "_sets", dict(sets))
        object.__setattr__(self, "_params", params)
        object.__setattr__(
            self,
            "_vars",
            {
                name: PoiVarProxy(poi_model, name, tuple(domains))
                for name, domains in (var_specs or {}).items()
            },
        )
        object.__setattr__(self, "constraints", {})

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal attribute lookup fails, so the instance
        # attributes set above never come through here.
        d = object.__getattribute__(self, "__dict__")

        var = d.get("_vars", {}).get(name)
        if var is not None:
            return var

        elems = d.get("_sets", {})
        if name in elems:
            return elems[name]

        params = d.get("_params")
        if params is not None:
            if isinstance(params, dict):
                if name in params:
                    return params[name]
            elif hasattr(params, name):
                return getattr(params, name)

        raise AttributeError(
            f"{name!r} is not on the POI adapter (not a variable, set or parameter)"
        )

    def add_constraint(self, name: str, expr: Any) -> Any:
        """Attach one constraint, routing by the expression type POI produced.

        POI splits its constraint API by degree: products of two variables come
        back as ``ScalarQuadraticFunction`` and have a dedicated entry point, while
        anything transcendental (the CES powers and logs throughout GTAP) must go
        through the nonlinear path. Dispatching on the type keeps each expression on
        the cheapest route that can represent it exactly.
        """
        model = object.__getattribute__(self, "_model")
        kind = type(expr).__name__

        if kind in ("ScalarAffineFunction", "VariableIndex"):
            con = model.add_linear_constraint(expr)
        elif kind == "ScalarQuadraticFunction":
            con = model.add_quadratic_constraint(expr)
        else:
            con = model.add_nl_constraint(expr)

        object.__getattribute__(self, "constraints")[name] = con
        return con

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        d = object.__getattribute__(self, "__dict__")
        return (
            f"PoiModelAdapter(vars={len(d['_vars'])}, sets={len(d['_sets'])}, "
            f"constraints={len(d['constraints'])})"
        )
