"""Unit tests for PEP macro closure blockwise initializer."""

from __future__ import annotations

from equilibria.blocks.equilibrium import PEPMacroClosureInit
from equilibria.core import Set, SetManager


def _sets() -> SetManager:
    sm = SetManager()
    sm.add(Set(name="I", elements=("c1",)))
    sm.add(Set(name="J", elements=("s1",)))
    sm.add(Set(name="K", elements=("cap",)))
    sm.add(Set(name="H", elements=("h1",)))
    sm.add(Set(name="F", elements=("firm",)))
    sm.add(Set(name="AGD", elements=("h1", "firm", "gvt")))
    return sm


def test_macro_closure_full_alpha_consistent() -> None:
    block = PEPMacroClosureInit()
    sm = _sets()

    params = {
        "IMO0": {"c1": 1.0},
        "EXDO0": {"c1": 1.0},
        "KDO0": {("cap", "s1"): 1.0},
        "lambda_RK": {("row", "cap"): 0.0},
        "macro_alpha": 1.0,
    }
    vars_map = {
        "PWM": {"c1": 2.0},
        "IM": {"c1": 5.0},
        "R": {("cap", "s1"): 1.0},
        "KD": {("cap", "s1"): 10.0},
        "TR": {("row", "h1"): 0.0, ("row", "firm"): 0.0, ("row", "gvt"): 0.0, ("h1", "row"): 0.0, ("firm", "row"): 0.0, ("gvt", "row"): 0.0},
        "PE_FOB": {"c1": 3.0},
        "EXD": {"c1": 4.0},
        "SH": {"h1": 10.0},
        "SF": {"firm": 20.0},
        "SG": 30.0,
        "PC": {"c1": 1.0},
        "C": {("c1", "h1"): 40.0},
        "CG": {"c1": 5.0},
        "INV": {"c1": 6.0},
        "VSTK": {"c1": 1.0},
        "e": 1.0,
        "YROW": 0.0,
        "SROW": 0.0,
        "CAB": 0.0,
        "IT": 0.0,
        "GFCF": 0.0,
        "GDP_FD": 0.0,
    }

    block.initialize_levels(
        set_manager=sm,
        parameters=params,
        variables=vars_map,
        mode="gams_blockwise",
    )

    residuals = block.validate_initialization(
        set_manager=sm,
        parameters=params,
        variables=vars_map,
    )
    for _, value in residuals.items():
        assert abs(value) < 1e-10
