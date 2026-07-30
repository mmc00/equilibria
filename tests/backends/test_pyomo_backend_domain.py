"""The Pyomo bridge must forward a Variable's domain as ``within=`` (F3 task 3.6).

Before this task ``core.variables.Variable`` had no domain field and
``pyomo_backend._build_variables`` emitted ``Var(bounds=...)`` with Pyomo's
default domain ``Reals`` on every variable. The GTAP monolith declares 78
price/quantity vars ``within=NonNegativeReals``, so ``domain_bounds_diff``
reported a label mismatch (``Reals`` vs ``NonNegativeReals``) on every one of
them even when the feasible region was identical — the Task-4 domain gate was
unpassable for a byte-perfect port.

These tests build a tiny model with one ``NonNegativeReals`` var and one
default (``Reals``) var, run it through ``PyomoBackend.build``, and assert:

  - the ``NonNegativeReals`` var's emitted ``.domain`` is ``NonNegativeReals``;
  - the default var's emitted ``.domain`` is still ``Reals`` (non-regression
    for every existing 1-D/2-D caller: PEP blocks, simple_open, task-3.5 tests);
  - the ``bounds`` are still forwarded alongside ``within=`` (Pyomo intersects
    the domain and the box; the monolith sets both, so we must too).

Plus a synthetic ``domain_bounds_diff`` check: a NonNegativeReals-block var vs a
NonNegativeReals monolith var now reports ZERO domain mismatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from equilibria import Model
from equilibria.backends import PyomoBackend
from equilibria.core import Set, Variable
from equilibria.core.symbolic_equations import SymbolicEquation


class _QtyEqualsPrice(SymbolicEquation):
    """One trivial scalar-per-(I) constraint: qty[i] == price[i].

    Makes the model non-empty (``_build_constraints`` raises otherwise) without
    touching the domains under test.
    """

    name: str = "qty_eq_price"
    domains: tuple = ("I",)
    description: str = "Trivial identity so the model has a constraint."

    def build_expression(self, pyomo_model: Any, indices: tuple[str, ...]) -> Any:
        (i,) = indices
        return pyomo_model.qty[i] == pyomo_model.price[i]


def _domain_model() -> Model:
    """A Model with a NonNegativeReals var (``qty``) and a default var (``price``)."""
    model = Model(name="DomainModel")
    model.add_set(Set(name="I", elements=("i1", "i2")))

    # Explicit NonNegativeReals, with an explicit lower bound (matches the
    # monolith, which sets BOTH within and .setlb on its price/qty vars).
    model.add_variable(
        Variable(
            name="qty",
            value=np.array([1.0, 2.0]),
            domains=("I",),
            lower=0.0,
            upper=float("inf"),
            domain="NonNegativeReals",
        )
    )
    # Default domain — no `domain=` kwarg → must stay Reals (non-regression).
    model.add_variable(
        Variable(
            name="price",
            value=np.array([1.0, 2.0]),
            domains=("I",),
            lower=0.0,
            upper=float("inf"),
        )
    )

    model.add_equation(_QtyEqualsPrice())  # ty: ignore[invalid-argument-type]
    return model


@pytest.fixture
def built_backend() -> PyomoBackend:
    """A PyomoBackend with build() already run on the domain model."""
    backend = PyomoBackend()
    backend.build(_domain_model())
    return backend


def test_variable_domain_default_is_reals() -> None:
    """A Variable created without ``domain=`` defaults to the string 'Reals'."""
    v = Variable(name="p", value=np.array([1.0]), domains=("I",))
    assert v.domain == "Reals"


def test_nonnegative_var_emits_nonnegativereals(built_backend: PyomoBackend) -> None:
    """A domain='NonNegativeReals' Variable emits a NonNegativeReals Pyomo Var."""
    pm = built_backend.pyomo_model
    for i in ("i1", "i2"):
        assert pm.qty[i].domain.name == "NonNegativeReals"


def test_default_var_still_emits_reals(built_backend: PyomoBackend) -> None:
    """A default-domain Variable still emits a Reals Pyomo Var (non-regression)."""
    pm = built_backend.pyomo_model
    for i in ("i1", "i2"):
        assert pm.price[i].domain.name == "Reals"


def test_bounds_still_passed_alongside_within(built_backend: PyomoBackend) -> None:
    """The lower bound is still forwarded alongside within= (domain ∩ box)."""
    pm = built_backend.pyomo_model
    # lower=0.0 was passed as bounds; it must survive on both vars.
    assert pm.qty["i1"].lb == 0.0
    assert pm.price["i1"].lb == 0.0


def test_domain_bounds_diff_agrees_for_matching_domains() -> None:
    """domain_bounds_diff reports ZERO mismatch between two NonNegativeReals vars.

    Confirms Blocker B closes the diag gap: with the block able to set
    ``domain='NonNegativeReals'``, a NonNegativeReals-block var vs a
    NonNegativeReals monolith var no longer false-diffs on the domain label.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root / "scripts" / "gtap") not in sys.path:
        sys.path.insert(0, str(root / "scripts" / "gtap"))
    from blocks_diag import domain_bounds_diff  # ty: ignore[unresolved-import]
    from pyomo.environ import ConcreteModel, NonNegativeReals, Var

    def _build():
        m = ConcreteModel()
        m.x = Var(within=NonNegativeReals, bounds=(0.0, None), initialize=1.0)
        return m

    diffs = domain_bounds_diff(_build(), _build())
    assert diffs == []


def test_per_index_array_bounds_emit_percell_floors() -> None:
    """An ndarray lower bound emits the exact per-cell lower bound (F3 task 4).

    GTAP applies a runtime relative price floor ``setlb(1e-3*init)`` to price
    vars whose benchmark init differs per cell (``p_rai``/``pf``/``pfa``). The
    exact per-cell floor is not expressible via Pyomo's scalar ``bounds=`` tuple,
    so the bridge builds a bounds RULE when ``lower``/``upper`` is an ndarray.
    A scalar side is broadcast. This is what makes the Step-C domain gate
    reachable for those vars.
    """
    from equilibria import Model, Set, Variable
    from equilibria.backends import PyomoBackend
    from equilibria.core.symbolic_equations import SymbolicEquation

    class _Eq(SymbolicEquation):
        name: str = "e"
        domains: tuple = ("I",)

        def build_expression(self, pm: Any, indices: tuple) -> Any:
            (i,) = indices
            return pm.p[i] >= 0

    m = Model(name="t")
    m.add_set(Set(name="I", elements=["a", "b", "c"]))
    m.add_variable(
        Variable(
            name="p",
            value=np.array([1.0, 2.0, 3.0]),
            domains=("I",),
            domain="NonNegativeReals",
            lower=np.array([0.1, 0.2, 0.3]),
            upper=float("inf"),
        )
    )
    m.add_equation(_Eq())
    b = PyomoBackend()
    b.build(m)
    assert b.pyomo_model.p["a"].bounds == (0.1, None)
    assert b.pyomo_model.p["b"].bounds == (0.2, None)
    assert b.pyomo_model.p["c"].bounds == (0.3, None)
