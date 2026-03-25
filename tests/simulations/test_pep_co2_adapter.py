from __future__ import annotations

from typing import Any

import pytest

from equilibria.simulations.adapters.pep import PepAdapter
from equilibria.simulations.adapters.pep_co2 import PepCO2Adapter
from equilibria.simulations.types import Shock
from equilibria.templates.pep_calibration_unified import PEPModelState
from equilibria.templates.pep_co2_data import get_state_co2_block, set_state_co2_block


def test_pep_co2_adapter_fit_base_state_attaches_validated_co2_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_fit(self: PepAdapter) -> PEPModelState:
        state = PEPModelState(sets={"J": ["agr", "ser"]})
        self._sets = dict(state.sets)
        return state

    monkeypatch.setattr(PepAdapter, "fit_base_state", _fake_fit)

    adapter = PepCO2Adapter(
        sam_file="dummy.gdx",
        val_par_file=None,
        co2_intensity={"agr": 1.5},
        tco2b={"ser": 2.0},
        tco2scal=1.25,
    )

    state = adapter.fit_base_state()
    block = get_state_co2_block(state)

    assert block["co2_intensity"] == {"agr": 1.5, "ser": 0.0}
    assert block["tco2b"] == {"agr": 0.0, "ser": 2.0}
    assert block["tco2scal"] == pytest.approx(1.25)


def test_pep_co2_adapter_available_shocks_include_j_domain_members() -> None:
    adapter = PepCO2Adapter(sam_file="dummy.gdx", val_par_file=None)
    adapter._sets = {"J": ["agr", "ser"], "I": ["agr"]}

    catalog = {item.var: item for item in adapter.available_shocks()}

    assert catalog["tco2b"].members == ("agr", "ser")
    assert catalog["co2_intensity"].members == ("agr", "ser")
    assert catalog["tco2scal"].kind == "scalar"


def test_pep_co2_adapter_applies_sector_and_scalar_shocks() -> None:
    state = PEPModelState(sets={"J": ["agr", "ser"]})
    set_state_co2_block(
        state,
        {
            "co2_intensity": {"agr": 1.0, "ser": 2.0},
            "tco2b": {"agr": 1.0, "ser": 2.0},
            "tco2scal": 1.0,
        },
    )
    adapter = PepCO2Adapter(sam_file="dummy.gdx", val_par_file=None)

    adapter.apply_shock(state, Shock(var="tco2b", op="scale", values={"*": 2.0, "agr": 3.0}))
    adapter.apply_shock(state, Shock(var="tco2scal", op="add", values=0.5))
    adapter.apply_shock(state, Shock(var="co2_intensity", op="set", values={"agr": 4.0}))

    block = get_state_co2_block(state)
    assert block["tco2b"]["agr"] == pytest.approx(6.0)
    assert block["tco2b"]["ser"] == pytest.approx(4.0)
    assert block["tco2scal"] == pytest.approx(1.5)
    assert block["co2_intensity"]["agr"] == pytest.approx(4.0)
    assert block["co2_intensity"]["ser"] == pytest.approx(2.0)
