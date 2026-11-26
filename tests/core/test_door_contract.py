"""Every model door keeps the door contract (ADR-0012, ``equilibria.core.door``)."""

from __future__ import annotations

import sys
import types

import pytest

from equilibria.core.door import DOORS, DoorSpec, check_door


@pytest.mark.parametrize("spec", DOORS, ids=lambda s: s.module)
def test_every_door_keeps_the_contract(spec: DoorSpec) -> None:
    assert check_door(spec) == []


def test_gtap_is_a_door() -> None:
    assert DoorSpec("equilibria.gtap", "GTAP", "gtap_static", "gtap_default") in DOORS


def test_check_door_catches_a_broken_door(monkeypatch: pytest.MonkeyPatch) -> None:
    """A door missing ``results`` and with a ``ClosureConfig`` that is not a
    ``ModelClosureConfig`` breaks two rules, and the check names both."""
    import equilibria.gtap as good

    fake = types.ModuleType("equilibria_fake_door")
    for name in ("load", "GTAP", "gtap_static", "Exercise"):
        setattr(fake, name, getattr(good, name))

    class NotAClosure:
        @classmethod
        def gtap_default(cls, **changes):
            return cls()

    vars(fake).update(
        ClosureConfig=NotAClosure,
        __all__=["load", "GTAP", "gtap_static", "Exercise", "ClosureConfig"],
    )
    monkeypatch.setitem(sys.modules, "equilibria_fake_door", fake)
    msgs = check_door(
        DoorSpec("equilibria_fake_door", "GTAP", "gtap_static", "gtap_default")
    )
    assert any("results" in m for m in msgs), msgs
    assert any("ModelClosureConfig" in m for m in msgs), msgs
