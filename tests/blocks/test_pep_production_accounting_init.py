"""Unit tests for PEP production-accounting blockwise initializer."""

from __future__ import annotations

from equilibria.blocks.production import PEPProductionAccountingInit
from equilibria.core import Set, SetManager


def _sets() -> SetManager:
    sm = SetManager()
    sm.add(Set(name="I", elements=("c1", "c2")))
    sm.add(Set(name="J", elements=("s1",)))
    return sm


def test_initialize_levels_production_accounting() -> None:
    block = PEPProductionAccountingInit()
    sm = _sets()

    params = {
        "io": {"s1": 0.4},
        "aij": {("c1", "s1"): 0.5, ("c2", "s1"): 0.5},
        "ttip": {"s1": 0.1},
    }
    vars_map = {
        "XST": {"s1": 100.0},
        "PVA": {"s1": 1.2},
        "VA": {"s1": 60.0},
        "DI": {("c1", "s1"): 20.0, ("c2", "s1"): 20.0},
        "PC": {"c1": 1.0, "c2": 1.0},
        "CI": {"s1": 0.0},
        "PP": {"s1": 9.0},
        "PT": {"s1": 10.0},
        "PCI": {"s1": 0.0},
        "DIT": {"c1": 0.0, "c2": 0.0},
    }

    block.initialize_levels(
        set_manager=sm,
        parameters=params,
        variables=vars_map,
        mode="gams_blockwise",
    )

    assert abs(vars_map["CI"]["s1"] - 40.0) < 1e-12
    assert abs(vars_map["DI"][("c1", "s1")] - 20.0) < 1e-12
    assert abs(vars_map["DI"][("c2", "s1")] - 20.0) < 1e-12
    assert abs(vars_map["DIT"]["c1"] - 20.0) < 1e-12
    assert abs(vars_map["DIT"]["c2"] - 20.0) < 1e-12
    assert abs(vars_map["PCI"]["s1"] - 1.0) < 1e-12
    assert abs(vars_map["PP"]["s1"] - 1.12) < 1e-12
    assert abs(vars_map["PT"]["s1"] - 10.0) < 1e-12


def test_validate_initialization_production_accounting_zero_residuals() -> None:
    block = PEPProductionAccountingInit()
    sm = _sets()

    params = {
        "io": {"s1": 0.4},
        "aij": {("c1", "s1"): 0.5, ("c2", "s1"): 0.5},
        "ttip": {"s1": 0.1},
    }
    vars_map = {
        "XST": {"s1": 100.0},
        "PVA": {"s1": 1.2},
        "VA": {"s1": 60.0},
        "DI": {("c1", "s1"): 20.0, ("c2", "s1"): 20.0},
        "PC": {"c1": 1.0, "c2": 1.0},
        "CI": {"s1": 40.0},
        "PCI": {"s1": 1.0},
        "PP": {"s1": 1.12},
        "PT": {"s1": 1.232},
        "DIT": {"c1": 20.0, "c2": 20.0},
    }

    residuals = block.validate_initialization(
        set_manager=sm,
        parameters=params,
        variables=vars_map,
    )
    assert set(residuals.keys()) == {"EQ2_s1", "EQ9_c1_s1", "EQ9_c2_s1", "EQ67_s1", "EQ65_s1", "EQ56_c1", "EQ56_c2"}
    for _, value in residuals.items():
        assert abs(value) < 1e-12
