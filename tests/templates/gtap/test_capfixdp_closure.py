"""capFixDp closure (dpsave↔del_tbalry: betaS endogenous, trade balance fixed) — tests.

The GEMPACK fixture used `swap dpsave(r)=del_tbalry(r)` (saving distribution endogenous,
real trade balance fixed). Our capFix `savf=pigbl·savf_bar` IS the trade-balance-fixed
condition; the only difference is betaS: fixed (capFix) vs free (capFixDp).
"""

import contextlib
import io
from pathlib import Path

from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

DS = Path("datasets/gtap7_3x3")


def test_capfixdp_is_valid_savf_flag():
    gc = GTAPClosureConfig(name="base", closure_type="MCP", savf_flag="capFixDp")
    assert gc.savf_flag == "capFixDp"
