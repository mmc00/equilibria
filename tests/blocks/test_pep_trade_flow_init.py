"""Unit tests for PEP trade-flow blockwise initializer."""

from __future__ import annotations

import math

from equilibria.blocks.trade import PEPTradeFlowInit
from equilibria.core import Set, SetManager


def _sets() -> SetManager:
    sm = SetManager()
    sm.add(Set(name="I", elements=("c1",)))
    sm.add(Set(name="J", elements=("s1",)))
    return sm


def test_initialize_levels_from_benchmarks() -> None:
    block = PEPTradeFlowInit()
    sm = _sets()

    params = {
        "QO0": {"c1": 80.0},
        "DDO0": {"c1": 100.0},
        "IMO0": {"c1": 20.0},
        "EXDO0": {"c1": 30.0},
        "PCO0": {"c1": 1.1},
        "PDO0": {"c1": 1.2},
        "PMO0": {"c1": 1.3},
        "MRGNO0": {"c1": 18.0},
        "DITO0": {"c1": 7.0},
    }
    vars_map: dict[str, dict] = {}

    block.initialize_levels(
        set_manager=sm,
        parameters=params,
        variables=vars_map,
        mode="gams_blockwise",
    )

    assert vars_map["Q"]["c1"] == 80.0
    assert vars_map["DD"]["c1"] == 100.0
    assert vars_map["IM"]["c1"] == 20.0
    assert vars_map["EXD"]["c1"] == 30.0
    assert vars_map["PC"]["c1"] == 1.1
    assert vars_map["PD"]["c1"] == 1.2
    assert vars_map["PM"]["c1"] == 1.3
    assert vars_map["MRGN"]["c1"] == 18.0
    assert vars_map["DIT"]["c1"] == 7.0


def test_validate_initialization_eq57_to_eq64_consistent_case() -> None:
    block = PEPTradeFlowInit()
    sm = _sets()

    beta_m = 1.0 / 6.0
    dd = 100.0
    im = 20.0
    q = (beta_m * math.sqrt(im) + (1.0 - beta_m) * math.sqrt(dd)) ** 2

    params = {
        "DDO0": {"c1": dd},
        "IMO0": {"c1": im},
        "EXDO0": {"c1": 30.0},
        "EXDO": {"c1": 30.0},
        "XSO0": {("s1", "c1"): 150.0},
        "XSTO0": {"s1": 150.0},
        "DSO0": {("s1", "c1"): 120.0},
        "EXO0": {("s1", "c1"): 30.0},
        "tmrg": {("c1", "c1"): 0.1},
        "tmrg_X": {("c1", "c1"): 0.2},
        "rho_XT": {"s1": 1.0},
        "sigma_XT": {"s1": 2.0},
        "beta_XT": {("s1", "c1"): 1.0},
        "B_XT": {"s1": 1.0},
        "rho_X": {("s1", "c1"): 1.0},
        "beta_X": {("s1", "c1"): 0.8},
        "sigma_X": {("s1", "c1"): 1.0},
        "B_X": {("s1", "c1"): 3.125},
        "rho_M": {"c1": -0.5},
        "beta_M": {"c1": beta_m},
        "sigma_M": {"c1": 1.0},
        "B_M": {"c1": 1.0},
        "sigma_XD": {"c1": 1.0},
        "PWX": {"c1": 2.0},
        "e": 1.0,
    }

    vars_map = {
        "Q": {"c1": q},
        "DD": {"c1": dd},
        "IM": {"c1": im},
        "EXD": {"c1": 30.0},
        "PC": {"c1": 1.0},
        "PD": {"c1": 1.0},
        "PM": {"c1": 1.0},
        "MRGN": {"c1": 18.0},
        "XS": {("s1", "c1"): 150.0},
        "XST": {"s1": 150.0},
        "DS": {("s1", "c1"): 120.0},
        "EX": {("s1", "c1"): 30.0},
        "P": {("s1", "c1"): 1.0},
        "PT": {"s1": 1.0},
        "PE": {"c1": 1.0},
        "PL": {"c1": 1.0},
        "PWM": {"c1": 2.0},
        "PE_FOB": {"c1": 2.0},
    }

    residuals = block.validate_initialization(
        set_manager=sm,
        parameters=params,
        variables=vars_map,
    )

    for name, val in residuals.items():
        assert abs(val) < 1e-9, f"{name} should be ~0, got {val}"
