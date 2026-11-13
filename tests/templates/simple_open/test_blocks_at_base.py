"""Each block's equations hold exactly at the calibrated base (stdcge levels).

The base is the SAM itself, so a wrong sign, a missing wedge or a share computed
on the wrong total leaves a nonzero residual.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.templates.simple_open._helpers import block_at_base, max_residual

from equilibria.blocks.institutions import ExternalSector
from equilibria.blocks.production import (
    CETOutput,
    CobbDouglasValueAdded,
    LeontiefComposite,
)
from equilibria.blocks.trade import ArmingtonComposite

TOL = 1e-10


def _blocks(cal: Any) -> dict[str, tuple[Any, list[str]]]:
    kw = {"values": cal.values, "start": cal.base}
    return {
        "value_added": (CobbDouglasValueAdded(**kw), ["eq_va", "eq_fd"]),
        "leontief": (LeontiefComposite(**kw), ["eq_xi", "eq_va_z", "eq_pz"]),
        "cet": (CETOutput(**kw), ["eq_cet", "eq_e", "eq_ds"]),
        "armington": (ArmingtonComposite(**kw), ["eq_arm", "eq_m", "eq_d"]),
        "external": (ExternalSector(**kw), ["eq_pe", "eq_pm", "eq_bop"]),
    }


@pytest.mark.parametrize(
    "which", ["value_added", "leontief", "cet", "armington", "external"]
)
def test_block_residual_is_zero_at_base(calibration: Any, which: str) -> None:
    block, equations = _blocks(calibration)[which]
    pm = block_at_base(block, calibration)
    assert max_residual(pm, equations) <= TOL
