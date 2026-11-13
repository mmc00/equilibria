"""Each block's equations hold exactly at the calibrated base (stdcge levels).

The base is the SAM itself, so a wrong sign, a missing wedge or a share computed
on the wrong total leaves a nonzero residual.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.templates.simple_open._helpers import block_at_base, max_residual

from equilibria.blocks.demand import CobbDouglasHousehold
from equilibria.blocks.equilibrium import FactorMarket, GoodsMarket, Numeraire
from equilibria.blocks.institutions import ExternalSector, TaxGovernment
from equilibria.blocks.production import (
    CETOutput,
    CobbDouglasValueAdded,
    LeontiefComposite,
)
from equilibria.blocks.trade import ArmingtonComposite
from equilibria.templates.simple_open.blocks import StdcgeInvestment

TOL = 1e-10


def _blocks(cal: Any) -> dict[str, tuple[Any, list[str]]]:
    kw = {"values": cal.values, "start": cal.base}
    return {
        "value_added": (CobbDouglasValueAdded(**kw), ["eq_va", "eq_fd"]),
        "leontief": (LeontiefComposite(**kw), ["eq_xi", "eq_va_z", "eq_pz"]),
        "cet": (CETOutput(**kw), ["eq_cet", "eq_e", "eq_ds"]),
        "armington": (ArmingtonComposite(**kw), ["eq_arm", "eq_m", "eq_d"]),
        "external": (ExternalSector(**kw), ["eq_pe", "eq_pm", "eq_bop"]),
        "household": (CobbDouglasHousehold(**kw), ["eq_sp", "eq_xp", "eq_uu"]),
        "government": (
            TaxGovernment(**kw),
            ["eq_td", "eq_tz", "eq_tm", "eq_xg", "eq_sg"],
        ),
        "investment": (StdcgeInvestment(**kw), ["eq_xv"]),
        "goods_market": (GoodsMarket(**kw), ["eq_q"]),
        "factor_market": (FactorMarket(numeraire="LAB", **kw), ["eq_ff"]),
        "numeraire": (Numeraire(numeraire="LAB", **kw), ["eq_num"]),
    }


BLOCKS = [
    "value_added",
    "leontief",
    "cet",
    "armington",
    "external",
    "household",
    "government",
    "investment",
    "goods_market",
    "factor_market",
    "numeraire",
]


@pytest.mark.parametrize("which", BLOCKS)
def test_block_residual_is_zero_at_base(calibration: Any, which: str) -> None:
    block, equations = _blocks(calibration)[which]
    pm = block_at_base(block, calibration)
    assert max_residual(pm, equations) <= TOL
