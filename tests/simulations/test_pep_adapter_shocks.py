from __future__ import annotations

import pytest

from equilibria.simulations.adapters.pep import PepAdapter
from equilibria.simulations.types import Shock
from equilibria.templates.pep_calibration_unified import PEPModelState


def test_pep_adapter_applies_scalar_go_shocks() -> None:
    state = PEPModelState(consumption={"GO": 100.0})
    adapter = PepAdapter(sam_file="dummy.gdx", val_par_file=None)

    adapter.apply_shock(state, Shock(var="G", op="scale", values=1.2))
    assert state.consumption["GO"] == pytest.approx(120.0)

    adapter.apply_shock(state, Shock(var="G", op="add", values=5.0))
    assert state.consumption["GO"] == pytest.approx(125.0)

    adapter.apply_shock(state, Shock(var="G", op="set", values=80.0))
    assert state.consumption["GO"] == pytest.approx(80.0)


def test_pep_adapter_applies_indexed_pwm_wildcard_and_specific() -> None:
    state = PEPModelState(
        sets={"I": ["agr", "ser"]},
        trade={"PWMO": {"agr": 1.0, "ser": 2.0}},
    )
    adapter = PepAdapter(sam_file="dummy.gdx", val_par_file=None)

    adapter.apply_shock(
        state,
        Shock(var="PWM", op="scale", values={"*": 1.25, "agr": 2.0}),
    )

    assert state.trade["PWMO"]["agr"] == pytest.approx(2.5)
    assert state.trade["PWMO"]["ser"] == pytest.approx(2.5)


def test_pep_adapter_rejects_unknown_index_for_indexed_shock() -> None:
    state = PEPModelState(
        sets={"I": ["agr"]},
        trade={"PWMO": {"agr": 1.0}},
    )
    adapter = PepAdapter(sam_file="dummy.gdx", val_par_file=None)

    with pytest.raises(ValueError, match="unknown indices"):
        adapter.apply_shock(
            state,
            Shock(var="PWM", op="set", values={"bad": 1.0}),
        )


def test_pep_adapter_syncs_export_tax_aggregates_after_ttix_shock() -> None:
    state = PEPModelState(
        sets={"I": ["agr", "ser"]},
        trade={
            "ttixO": {"agr": 0.1, "ser": 0.2},
            "EXDO": {"agr": 10.0, "ser": 5.0},
            "PEO": {"agr": 2.0, "ser": 3.0},
            "tmrg_X": {
                ("agr", "agr"): 0.1,
                ("ser", "agr"): 0.2,
                ("agr", "ser"): 0.0,
                ("ser", "ser"): 0.1,
            },
            "PCO": {"agr": 1.0, "ser": 2.0},
        },
        income={
            "TICTO": 10.0,
            "TIMTO": 20.0,
            "TIXTO": 0.0,
            "TPRCTSO": 0.0,
            "YGKO": 100.0,
            "TDHTO": 5.0,
            "TDFTO": 7.0,
            "TPRODNO": 8.0,
            "YGTRO": 9.0,
            "YGO": 0.0,
        },
    )
    adapter = PepAdapter(sam_file="dummy.gdx", val_par_file=None)

    adapter.apply_shock(state, Shock(var="ttix", op="scale", values={"*": 2.0}))

    assert state.trade["ttixO"]["agr"] == pytest.approx(0.2)
    assert state.trade["ttixO"]["ser"] == pytest.approx(0.4)
    assert state.trade["TIXO"]["agr"] == pytest.approx(5.0)
    assert state.trade["TIXO"]["ser"] == pytest.approx(6.4)
    assert state.income["TIXTO"] == pytest.approx(11.4)
    assert state.income["TPRCTSO"] == pytest.approx(41.4)
    assert state.income["YGO"] == pytest.approx(170.4)


def test_pep_adapter_available_shocks_include_domain_members_when_sets_known() -> None:
    adapter = PepAdapter(sam_file="dummy.gdx", val_par_file=None)
    adapter._sets = {"I": ["agr", "ser"]}  # internal cache filled by fit_base_state
    catalog = adapter.available_shocks()
    by_var = {item.var: item for item in catalog}
    assert by_var["G"].members is None
    assert by_var["PWM"].members == ("agr", "ser")
    assert by_var["PWX"].members == ("agr", "ser")
    assert by_var["ttix"].members == ("agr", "ser")
