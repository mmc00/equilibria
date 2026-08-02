"""capFixDp closure (dpsave↔del_tbalry: betaS endogenous, trade balance fixed) — tests.

The GEMPACK fixture used `swap dpsave(r)=del_tbalry(r)` (saving distribution endogenous,
real trade balance fixed). Our capFix `savf=pigbl·savf_bar` IS the trade-balance-fixed
condition; the only difference is betaS: fixed (capFix) vs free (capFixDp).
"""

import contextlib
import io
from pathlib import Path

import pytest
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


REF = Path("tests/fixtures/gtap7/gtap7_3x3/out_gtap_shock_ifsub1.gdx")


def _solve(sf):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

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
        m, mp = build_block_model(
            p, p.sets, gc, list(p.sets.r)[-1], base_calibrated=True
        )
        res = solve_multiperiod(
            m, p, gc, ref_gdx=REF, skip_base_solve=True, mode="gtap"
        )
    return m, max(r["code"] for r in res.values())


def test_capfixdp_solves_code1():
    # capFixDp CONVERGES (the betaS-endogenous swap re-squares the MCP 1-for-1).
    _m, code = _solve("capFixDp")
    assert code == 1, f"capFixDp shock must converge (code=1), got {code}"


@pytest.mark.xfail(
    reason="capFixDp converges but the free betaS runs away (qsave USA -99%, EU -141%; "
    "match vs GEMPACK 34% << capFix 96.3%). Freeing the saving distribution alone does "
    "NOT reproduce the fixture's dpsave↔del_tbalry equilibrium — the swap needs the real "
    "del_tbalry equation (trade balance as %-of-world-income), not just betaS free + capFix "
    "savf. Anchoring betaS at base/residual did not tame it. Hypothesis refuted by measurement.",
    strict=True,
)
def test_capfixdp_matches_gempack_savings():
    from pyomo.environ import value as V

    m, code = _solve("capFixDp")
    assert code == 1
    # GEMPACK EU qsave -10%: capFixDp should land near it (within a few pp), NOT blow up.
    b = float(V(m.rsav[("EU_28", "base")]))
    s = float(V(m.rsav[("EU_28", "shock")]))
    eu = 100 * (s / b - 1)
    assert -20.0 < eu < -2.0, f"capFixDp qsave EU should approach GEMPACK -10, got {eu}"
