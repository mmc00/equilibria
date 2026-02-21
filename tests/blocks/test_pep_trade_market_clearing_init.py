"""Unit tests for PEP trade market-clearing blockwise initializer."""

from __future__ import annotations

from equilibria.blocks.trade import PEPTradeMarketClearingInit
from equilibria.core import Set, SetManager


def _sets() -> SetManager:
    sm = SetManager()
    sm.add(Set(name="I", elements=("c1",)))
    sm.add(Set(name="J", elements=("s1",)))
    return sm


def test_initialize_and_validate_market_clearing_full_alpha() -> None:
    block = PEPTradeMarketClearingInit()
    sm = _sets()

    params = {
        "DDO0": {"c1": 1.0},
        "IMO0": {"c1": 1.0},
        "DSO0": {("s1", "c1"): 1.0},
        "beta_M": {"c1": 0.2},
        "sigma_M": {"c1": 1.0},
        "trade_market_alpha": 1.0,
    }
    vars_map = {
        "DS": {("s1", "c1"): 100.0},
        "DD": {"c1": 0.0},
        "IM": {"c1": 0.0},
        "PD": {"c1": 1.0},
        "PM": {"c1": 1.0},
    }

    block.initialize_levels(
        set_manager=sm,
        parameters=params,
        variables=vars_map,
        mode="gams_blockwise",
    )

    # EQ88 => DD = sum DS = 100
    assert abs(vars_map["DD"]["c1"] - 100.0) < 1e-12
    # EQ64 with beta=0.2, sigma=1, PD/PM=1 => IM = 0.25 * DD
    assert abs(vars_map["IM"]["c1"] - 25.0) < 1e-12

    residuals = block.validate_initialization(
        set_manager=sm,
        parameters=params,
        variables=vars_map,
    )
    for _, value in residuals.items():
        assert abs(value) < 1e-12
