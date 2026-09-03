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


def _affine_nnz(body: Any) -> int:
    """Distinct variables in an affine expression — its Jacobian row's nonzeros."""
    variables = getattr(body, "variables", None)
    if variables is None:
        # A bare VariableIndex is one variable, so one nonzero.
        return 1
    return len(set(variables))


def _nl_row_nnz(adapter: Any, expr: Any) -> int:
    """Distinct variables in one nonlinear row.

    POI hands back an opaque graph handle for a nonlinear constraint, and its own
    counters describe deduplicated group representatives rather than rows. The
    variables are instead taken from what the row's ``build_expression`` actually
    read, which the adapter records as it hands out handles — the same quantity
    Pyomo reports via ``identify_variables``.
    """
    return len(object.__getattribute__(adapter, "_touched"))


def _quadratic_nnz(body: Any) -> int:
    """Distinct variables in a quadratic expression.

    A product ``x*y`` differentiates to nonzeros in both x and y, so both factors
    count, together with anything in the affine part.
    """
    seen: set[Any] = set()
    seen.update(getattr(body, "variable_1s", ()) or ())
    seen.update(getattr(body, "variable_2s", ()) or ())
    affine = getattr(body, "affine_part", None)
    if affine is not None:
        seen.update(getattr(affine, "variables", ()) or ())
    return len(seen)


class PoiVarProxy:
    """One variable family: ``model.px`` here, ``model.px[r, i]`` a POI handle.

    Handles are created lazily — the blocks touch a subset of any family's index
    space, and materializing the full cartesian product would build variables no
    equation ever mentions.
    """

    __slots__ = ("_model", "_name", "_domains", "_cache", "_touched")

    def __init__(
        self,
        poi_model: Any,
        name: str,
        domains: tuple[str, ...],
        touched: set | None = None,
    ) -> None:
        self._model = poi_model
        self._name = name
        self._domains = domains
        self._cache: dict[tuple[Any, ...], Any] = {}
        # Shared with the adapter: every handle handed out is recorded so a row's
        # variables can be attributed even when POI keeps the expression opaque.
        self._touched = touched

    def __getitem__(self, key: Any) -> Any:
        k = key if isinstance(key, tuple) else (key,)
        handle = self._cache.get(k)
        if handle is None:
            # A scalar variable is cached under the empty key and keeps its bare
            # name; an indexed one is labelled with its cell.
            label = f"{self._name}[{','.join(map(str, k))}]" if k else self._name
            handle = self._model.add_variable(name=label)
            self._cache[k] = handle
        if self._touched is not None:
            self._touched.add((self._name, k))
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
            {},
        )
        object.__setattr__(self, "constraints", {})
        # Structural nonzeros per row, counted while building.
        #
        # POI's own counters cannot supply this: it deduplicates identical graphs
        # and keeps ONE representative per group, so the published nnz covers the
        # representatives rather than every row (measured: 409 nonlinear rows
        # collapse to 32 representatives). Counting here, per row, matches how the
        # Pyomo side is counted — distinct variables per constraint body — so the
        # two totals describe the same matrix.
        object.__setattr__(self, "linear_nnz", [])
        object.__setattr__(self, "nl_nnz", [])
        # Variable handles touched since the last reset, used to attribute the
        # variables of a nonlinear row that POI exposes only as an opaque graph.
        touched: set = set()
        object.__setattr__(self, "_touched", touched)
        object.__setattr__(
            self,
            "_vars",
            {
                name: PoiVarProxy(poi_model, name, tuple(domains), touched)
                for name, domains in (var_specs or {}).items()
            },
        )

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal attribute lookup fails, so the instance
        # attributes set above never come through here.
        d = object.__getattribute__(self, "__dict__")

        var = d.get("_vars", {}).get(name)
        if var is not None:
            # A scalar variable is used bare (``m.chiSave * m.pi[r]``), never
            # indexed, so hand back the single handle rather than the proxy —
            # matching Pyomo, where a non-indexed Var IS the variable.
            if not var.domains:
                return var[()]
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
        """Attach one constraint, routing by what POI's ``==`` actually produced.

        The block bodies all write ``lhs == rhs``, but POI answers that in two
        different types depending on the operands (measured, POI 0.6.1):

        * affine and quadratic comparisons -> ``ComparisonConstraint``, which
          carries its own sense and bounds and goes to the linear/quadratic entry
          points;
        * anything transcendental — the CES powers and logs throughout GTAP —
          -> ``ExpressionHandle``, where the comparison is folded into the graph
          and ``add_nl_constraint`` unpacks it.

        Routing each to the cheapest entry point that represents it exactly keeps
        the linear rows out of the autodiff graph, which is where POI's compile
        cost lives.
        """
        model = object.__getattribute__(self, "_model")

        from pyoptinterface._src.comparison_constraint import ComparisonConstraint

        linear_nnz = object.__getattribute__(self, "linear_nnz")

        if isinstance(expr, ComparisonConstraint):
            body = expr.lhs - expr.rhs
            kind = type(body).__name__
            if kind in ("ScalarAffineFunction", "VariableIndex"):
                con = model.add_linear_constraint(body, expr.sense, 0.0)
                linear_nnz.append(_affine_nnz(body))
            elif kind == "ScalarQuadraticFunction":
                con = model.add_quadratic_constraint(body, expr.sense, 0.0)
                linear_nnz.append(_quadratic_nnz(body))
            else:
                con = model.add_nl_constraint(expr)
                object.__getattribute__(self, "nl_nnz").append(_nl_row_nnz(self, expr))
        else:
            con = model.add_nl_constraint(expr)
            object.__getattribute__(self, "nl_nnz").append(_nl_row_nnz(self, expr))

        object.__getattribute__(self, "constraints")[name] = con
        return con

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        d = object.__getattribute__(self, "__dict__")
        return (
            f"PoiModelAdapter(vars={len(d['_vars'])}, sets={len(d['_sets'])}, "
            f"constraints={len(d['constraints'])})"
        )
