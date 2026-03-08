from __future__ import annotations

import pytest
from pydantic import ValidationError

from equilibria.simulations import Scenario, Shock


def test_scenario_normalizes_reference_slice() -> None:
    scenario = Scenario(
        name="import",
        shocks=[Shock(var="PWM", op="scale", values={"*": 1.25})],
        reference_slice="SIM1",
    )
    assert scenario.reference_slice == "sim1"


def test_scenario_requires_non_empty_name() -> None:
    with pytest.raises(ValidationError):
        Scenario(
            name="   ",
            shocks=[Shock(var="G", op="set", values=100.0)],
        )


def test_shock_requires_non_empty_var() -> None:
    with pytest.raises(ValidationError):
        Shock(var="  ", op="set", values=1.0)
