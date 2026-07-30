"""The bridge must RAISE on an un-translatable equation, not swallow it.

These are the strict-mode guarantees for the Pyomo bridge (F3 task 1):
a dropped equation is invisible and fatal for parity, so the bridge must
fail loudly instead of skipping constraints or stubbing a trivial one.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from equilibria import Model
from equilibria.backends.pyomo_backend import BridgeTranslationError, PyomoBackend
from equilibria.blocks import CESValueAdded
from equilibria.core import Set
from equilibria.core.symbolic_equations import SymbolicEquation


class _FailingEquation(SymbolicEquation):
    """A scalar equation whose translation always fails.

    ``build_expression`` accepts any call signature (the backend invokes it
    as ``build_expression(pyomo_model, indices)``) and always raises, so the
    bridge's per-constraint build step is forced to surface the failure.
    """

    name: str = "always_fails"
    domains: tuple = ()
    description: str = "Equation whose build_expression raises."

    def build_expression(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("intentional translation failure")


class _EmptyDomainEquation(SymbolicEquation):
    """An equation domained over an empty set — resolves to zero indices.

    Legitimate in GTAP (an equation over an empty region/sector subset).
    The bridge must skip it with a visible warning, not raise.
    """

    name: str = "empty_domain_eq"
    domains: tuple = ("EMPTY",)
    description: str = "Equation over an empty set."

    def build_expression(self, *args: Any, **kwargs: Any) -> Any:
        # Never reached: get_indices() returns [] for an empty domain.
        raise AssertionError("build_expression must not be called")


class _AllNoneEquation(SymbolicEquation):
    """An equation with real indices whose build always returns None.

    Every ``build_expression`` call yields ``None`` so the whole equation
    builds to nothing — a true invisible drop the bridge must raise on.
    """

    name: str = "all_none_eq"
    domains: tuple = ("J",)
    description: str = "Equation whose build_expression always returns None."

    def build_expression(self, *args: Any, **kwargs: Any) -> Any:
        return None


def _backend_with_one_failing_equation() -> tuple[PyomoBackend, Model]:
    """A Model with exactly one equation whose build_expression raises."""
    model = Model(name="FailingModel")
    # add_equation is typed for the legacy Equation ABC, but the runtime (and
    # every block) registers SymbolicEquation instances the same way.
    model.add_equation(_FailingEquation())  # ty: ignore[invalid-argument-type]
    return PyomoBackend(), model


def _backend_with_no_equations() -> tuple[PyomoBackend, Model]:
    """A Model with sets but zero buildable constraints."""
    model = Model(name="EmptyModel")
    model.add_set(Set(name="J", elements=("sec1", "sec2")))
    return PyomoBackend(), model


def _backend_with_empty_domain_equation() -> tuple[PyomoBackend, Model]:
    """A Model with a buildable block plus one empty-domain equation.

    The empty-domain equation must be skipped (with a warning); the block
    supplies real constraints so ``build()`` succeeds overall.
    """
    model = Model(name="EmptyDomainModel")
    model.add_set(Set(name="J", elements=("sec1", "sec2")))
    model.add_set(Set(name="F", elements=("labor", "capital")))
    model.add_set(Set(name="EMPTY", elements=()))
    model.add_block(CESValueAdded(sigma=0.8))
    model.add_equation(_EmptyDomainEquation())  # ty: ignore[invalid-argument-type]
    return PyomoBackend(), model


def _backend_with_all_none_equation() -> tuple[PyomoBackend, Model]:
    """A Model whose single equation builds to nothing (all None)."""
    model = Model(name="AllNoneModel")
    model.add_set(Set(name="J", elements=("sec1", "sec2")))
    model.add_equation(_AllNoneEquation())  # ty: ignore[invalid-argument-type]
    return PyomoBackend(), model


def test_bridge_raises_on_unbuildable_equation():
    # a block whose build_expression raises must surface, not be skipped
    backend, model = _backend_with_one_failing_equation()
    with pytest.raises(BridgeTranslationError):
        backend.build(model)


def test_bridge_raises_when_zero_constraints():
    # must NOT inject dummy_constraint = 1 == 1
    backend, model = _backend_with_no_equations()
    with pytest.raises(BridgeTranslationError):
        backend.build(model)


def test_bridge_warns_and_skips_empty_domain_equation(caplog):
    # an equation over an empty set is legitimate (zero scalar constraints):
    # skip it with a VISIBLE warning, do NOT raise. The block supplies real
    # constraints so build() succeeds.
    backend, model = _backend_with_empty_domain_equation()
    with caplog.at_level(logging.WARNING, logger="equilibria.backends.pyomo_backend"):
        backend.build(model)  # must not raise

    # the empty-domain equation was skipped: no constraint object for it
    assert not hasattr(backend.pyomo_model, "empty_domain_eq_con")
    # but the drop is visible in the logs
    assert any(
        "empty_domain_eq" in rec.message and "empty set" in rec.message
        for rec in caplog.records
    )


def test_bridge_raises_when_equation_builds_to_nothing():
    # non-empty indices but every build_expression returned None ->
    # true invisible drop of the whole equation -> must raise.
    backend, model = _backend_with_all_none_equation()
    with pytest.raises(BridgeTranslationError, match="produced no constraints"):
        backend.build(model)
