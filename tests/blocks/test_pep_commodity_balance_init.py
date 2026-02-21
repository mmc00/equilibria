"""Unit tests for PEP commodity-balance blockwise initializer."""

from __future__ import annotations

from equilibria.blocks.trade import PEPCommodityBalanceInit
from equilibria.core import Set, SetManager


def _sets() -> SetManager:
    sm = SetManager()
    sm.add(Set(name="I", elements=("c1",)))
    sm.add(Set(name="J", elements=("s1",)))
    return sm


def test_initialize_levels_single_source_domestic() -> None:
    block = PEPCommodityBalanceInit()
    sm = _sets()

    params = {
        "DDO0": {"c1": 1.0},
        "IMO0": {"c1": 0.0},
        "EXDO0": {"c1": 0.0},
        "tmrg": {},
        "tmrg_X": {},
        "rho_M": {"c1": -0.5},
        "beta_M": {"c1": 0.2},
        "B_M": {"c1": 1.5625},  # 1/(1-beta)^2 so Q=DD when only domestic
    }
    vars_map = {
        "Q": {"c1": 0.0},
        "DD": {"c1": 0.0},
        "IM": {"c1": 0.0},
        "EXD": {"c1": 0.0},
        "PC": {"c1": 1.0},
        "PD": {"c1": 1.0},
        "PM": {"c1": 1.0},
        "MRGN": {"c1": 0.0},
        "C": {},
        "CG": {"c1": 100.0},
        "INV": {"c1": 0.0},
        "VSTK": {"c1": 0.0},
        "DIT": {"c1": 0.0},
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
        assert abs(value) < 1e-8
