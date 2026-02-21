"""Unit tests for PEP trade-transformation blockwise initializer."""

from __future__ import annotations

from equilibria.blocks.trade import PEPTradeTransformationInit
from equilibria.core import Set, SetManager


def _sets() -> SetManager:
    sm = SetManager()
    sm.add(Set(name="I", elements=("c1", "c2")))
    sm.add(Set(name="J", elements=("s1",)))
    return sm


def test_initialize_levels_eq58_updates_xst() -> None:
    block = PEPTradeTransformationInit()
    sm = _sets()

    params = {
        "XSO0": {("s1", "c1"): 60.0, ("s1", "c2"): 40.0},
        "rho_XT": {"s1": 1.0},
        "beta_XT": {("s1", "c1"): 0.6, ("s1", "c2"): 0.4},
        "B_XT": {"s1": 1.0},
        "XSTO0": {"s1": 100.0},
        "sigma_XT": {"s1": 2.0},
    }
    vars_map = {
        "XS": {("s1", "c1"): 70.0, ("s1", "c2"): 30.0},
        "XST": {"s1": 0.0},
        "P": {("s1", "c1"): 1.0, ("s1", "c2"): 1.0},
        "PT": {"s1": 1.0},
    }

    block.initialize_levels(
        set_manager=sm,
        parameters=params,
        variables=vars_map,
        mode="gams_blockwise",
    )

    # rho=1 => XST = B * sum(beta*XS) = 0.6*70 + 0.4*30 = 54
    assert abs(vars_map["XST"]["s1"] - 54.0) < 1e-12


def test_validate_initialization_eq58_eq59_zero() -> None:
    block = PEPTradeTransformationInit()
    sm = _sets()

    # Build a configuration where EQ58 and EQ59 are exactly satisfied.
    params = {
        "XSO0": {("s1", "c1"): 60.0, ("s1", "c2"): 40.0},
        "XSTO0": {"s1": 100.0},
        "rho_XT": {"s1": 1.0},
        "sigma_XT": {"s1": 2.0},
        "beta_XT": {("s1", "c1"): 0.6, ("s1", "c2"): 0.4},
        "B_XT": {"s1": 1.0},
    }
    vars_map = {
        "XS": {("s1", "c1"): 70.0, ("s1", "c2"): 30.0},
        "XST": {"s1": 54.0},  # satisfies EQ58 with rho=1
        "PT": {"s1": 1.0},
        "P": {
            ("s1", "c1"): ((70.0 / 54.0) ** 0.5) * 0.6,
            ("s1", "c2"): ((30.0 / 54.0) ** 0.5) * 0.4,
        },
    }

    residuals = block.validate_initialization(
        set_manager=sm,
        parameters=params,
        variables=vars_map,
    )

    for _, value in residuals.items():
        assert abs(value) < 1e-9
