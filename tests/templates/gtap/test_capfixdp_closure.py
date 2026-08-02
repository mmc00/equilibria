"""capFixDp closure (dpsave↔del_tbalry: betaS endogenous, trade balance fixed) — tests.

The GEMPACK fixture used `swap dpsave(r)=del_tbalry(r)` (saving distribution endogenous,
real trade balance fixed). Our capFix `savf=pigbl·savf_bar` IS the trade-balance-fixed
condition; the only difference is betaS: fixed (capFix) vs free (capFixDp).
"""

import contextlib
import io
from pathlib import Path

from pyomo.environ import Var

from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

DS = Path("datasets/gtap7_3x3")


def _build(sf):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DS / "basedata.har",
        sets_path=DS / "sets.har",
        default_path=DS / "default.prm",
        baserate_path=DS / "baserate.har",
    )
    gc = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        if_sub=False,
        savf_flag=sf,
    )
    with (
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        m, mp = build_block_model(p, p.sets, gc, list(p.sets.r)[-1])
    return m


def test_capfixdp_is_valid_savf_flag():
    gc = GTAPClosureConfig(name="base", closure_type="MCP", savf_flag="capFixDp")
    assert gc.savf_flag == "capFixDp"


def test_capfixdp_betas_is_variable():
    m = _build("capFixDp")
    assert hasattr(m, "betas") and isinstance(m.betas, Var), (
        "capFixDp: betas must be a Var"
    )


def test_capfix_betas_not_a_variable():
    m = _build("capFix")
    b = getattr(m, "betas", None)
    assert b is None or not isinstance(b, Var), "capFix: betas stays a folded Param"


def test_capfixdp_eq_phi_rsav_live_betas():
    # eq_phi + eq_rsav must carry the endogenous betas Var symbolically (not a frozen number).
    m = _build("capFixDp")
    body = " ".join(str(m.eq_phi[k].body) for k in m.eq_phi)
    body += " ".join(str(m.eq_rsav[k].body) for k in m.eq_rsav)
    assert "betas" in body, "capFixDp: eq_phi/eq_rsav must reference the betas Var"
