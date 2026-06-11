"""Tests for the GTAP shadow demand system integrator (welfare_shadow).

Pure-function tests on the linearised chain (E_qpev → E_ueprivev →
E_dpavev → E_uelasev → E_ypev/E_ygev/E_ysaveev → E_yev → E_EV) plus a
calibration smoke test against the known NUS333 USA baseline coefficients
to make sure the four integrators (Euler / midpoint / Gragg / Bulirsch-
Stoer) converge to the expected order-of-magnitude EV.

No external solver or HAR I/O — uses synthetic NUS333-shaped baselines.
"""
from __future__ import annotations

import math

import pytest

from equilibria.templates.gtap.welfare_shadow import (
    ShadowBaseline,
    ShadowResult,
    ShadowState,
    integrate,
    _make_state,
    _recompute_derived,
    _step,
)


# ---------------------------------------------------------------------------
# NUS333-like USA baseline (numbers extracted from basedata.har + default.prm).
# Kept here as constants so the test stays standalone — does not require any
# .har / .xls / RunGTAP artefacts on disk.
# ---------------------------------------------------------------------------

COMMS_NUS333 = ("AGR", "MFG", "SER")


def nus333_usa_baseline() -> ShadowBaseline:
    return ShadowBaseline(
        region="USA",
        commodities=COMMS_NUS333,
        PRIVEXP=9_949_302.0,
        GOVEXP=2_258_359.0,
        SAVE=594_183.0,
        INCOME=12_801_844.0,
        VPP={"AGR": 68_434.0, "MFG": 2_036_548.0, "SER": 7_844_320.0},
        INCPAR={"AGR": 0.1694, "MFG": 0.8781, "SER": 1.0389},
        ALPHA={"AGR": 0.8182, "MFG": 0.1979, "SER": 0.1719},
        DPARSUM=1.0,
    )


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------


def test_make_state_initialises_coefs_at_baseline():
    base = nus333_usa_baseline()
    st = _make_state(base)

    # Shadow expenditures = baseline expenditures.
    assert st.PRIVEXPEV == base.PRIVEXP
    assert st.GOVEXPEV == base.GOVEXP
    assert st.SAVEEV == base.SAVE
    assert st.INCOMEEV == base.INCOME

    # Utility activity levels start at 1.
    assert st.UTILPRIVEV == 1.0
    assert st.UTILGOVEV == 1.0
    assert st.UTILSAVEEV == 1.0
    assert st.UTILELASEV == 1.0

    # Main shares from baseline ratios.
    assert st.XSHRPRIV == pytest.approx(base.PRIVEXP / base.INCOME)
    assert st.XSHRGOV == pytest.approx(base.GOVEXP / base.INCOME)
    assert st.XSHRSAVE == pytest.approx(base.SAVE / base.INCOME)
    assert st.XSHRPRIV + st.XSHRGOV + st.XSHRSAVE == pytest.approx(1.0)


def test_recompute_derived_matches_known_uelaspriv():
    """UELASPRIV = Σ CONSHR · INCPAR with NUS333 USA values ≈ 1.0 (calibrated)."""
    base = nus333_usa_baseline()
    st = _make_state(base)

    # CONSHR_c × INCPAR_c summed: 0.0069×0.1694 + 0.2047×0.8781 + 0.7884×1.0389
    expected = sum(
        (base.VPP[c] / base.PRIVEXP) * base.INCPAR[c] for c in COMMS_NUS333
    )
    assert st.UELASPRIV == pytest.approx(expected, rel=1e-9)
    # NUS333 is calibrated so UELASPRIV ≈ 1 by construction.
    assert st.UELASPRIV == pytest.approx(1.0, rel=1e-3)


def test_recompute_derived_xwconshr_sums_to_one():
    base = nus333_usa_baseline()
    st = _make_state(base)
    assert sum(st.XWCONSHR.values()) == pytest.approx(1.0, rel=1e-9)


def test_recompute_derived_eyev_weighted_avg_is_one():
    """Σ CONSHR · EYEV = 1 by the EYEV definition in gtapv7.tab:3617-3623."""
    base = nus333_usa_baseline()
    st = _make_state(base)
    weighted = sum(st.CONSHR[c] * st.EYEV[c] for c in COMMS_NUS333)
    assert weighted == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Single-step chain — degenerate cases
# ---------------------------------------------------------------------------


def test_step_zero_shock_yields_zero_increments():
    base = nus333_usa_baseline()
    st = _make_state(base)
    step = _step(st, base, du=0.0, dpop=0.0, ddppriv=0.0, ddpgov=0.0, ddpsave=0.0)
    for v in step.values():
        assert v == pytest.approx(0.0, abs=1e-12)


def test_step_zero_dpsave_no_log_term_yev_equals_u():
    """With dpsave=0 and baseline coefs, log_term=0 → yev = UTILELAS·u = u (≈1)."""
    base = nus333_usa_baseline()
    st = _make_state(base)
    step = _step(st, base, du=0.1, dpop=0.0, ddppriv=0.0, ddpgov=0.0, ddpsave=0.0)
    # UTILELAS ≈ 1 at baseline → yev ≈ u
    assert step["yev"] == pytest.approx(0.1, rel=1e-3)


def test_step_dpsave_alone_produces_negative_uelasev():
    """A positive dpsave shift implies a negative uelasev (savings preference rose).

    For ddpsave=1 alone: uelasev = XSHRPRIV·ueprivev − dpavev. The dpavev
    piece is XSHRSAVE·1, and the ueprivev piece is small (CDE income
    elasticity correction). Together they put uelasev within a few percent
    of −XSHRSAVE.
    """
    base = nus333_usa_baseline()
    st = _make_state(base)
    step = _step(st, base, du=0.0, dpop=0.0, ddppriv=0.0, ddpgov=0.0, ddpsave=1.0)
    # Sign: positive dpsave → negative uelasev.
    assert step["uelasev"] < 0.0
    # Magnitude: within 5% of −XSHRSAVE (the leading-order term).
    assert step["uelasev"] == pytest.approx(-base.SAVE / base.INCOME, rel=5e-2)


def test_step_qsaveev_equals_ysaveev():
    """E_qsaveev: qsaveev = ysaveev (psave constant in shadow)."""
    base = nus333_usa_baseline()
    st = _make_state(base)
    step = _step(st, base, du=0.05, dpop=0.0, ddppriv=0.0, ddpgov=0.0, ddpsave=2.0)
    assert step["qsaveev"] == pytest.approx(step["ysaveev"])


def test_step_ypev_minus_yev_identity():
    """E_ypev: ypev − yev = -(ueprivev − uelasev) + dppriv."""
    base = nus333_usa_baseline()
    st = _make_state(base)
    step = _step(st, base, du=0.05, dpop=0.0, ddppriv=0.02, ddpgov=0.0, ddpsave=1.0)
    lhs = step["ypev"] - step["yev"]
    rhs = -(step["ueprivev"] - step["uelasev"]) + 0.02
    assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-12)


def test_step_ygev_minus_yev_identity():
    """E_ygev: ygev − yev = uelasev + dpgov."""
    base = nus333_usa_baseline()
    st = _make_state(base)
    step = _step(st, base, du=0.05, dpop=0.0, ddppriv=0.0, ddpgov=0.01, ddpsave=1.0)
    lhs = step["ygev"] - step["yev"]
    rhs = step["uelasev"] + 0.01
    assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-12)


def test_step_dpavev_share_weighted_sum():
    """E_dpavev: dpavev = XSHRPRIV·dppriv + XSHRGOV·dpgov + XSHRSAVE·dpsave."""
    base = nus333_usa_baseline()
    st = _make_state(base)
    step = _step(st, base, du=0.0, dpop=0.0, ddppriv=0.5, ddpgov=0.3, ddpsave=1.2)
    expected = (
        st.XSHRPRIV * 0.5 + st.XSHRGOV * 0.3 + st.XSHRSAVE * 1.2
    )
    assert step["dpavev"] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Integrator API — degenerate + linearity properties
# ---------------------------------------------------------------------------


def test_integrate_zero_shock_yields_zero_ev():
    base = nus333_usa_baseline()
    res = integrate(base, u_pct=0.0, dpsave_pct=0.0, n_steps=10)
    assert res.EV_USDm == pytest.approx(0.0, abs=1e-9)
    assert res.yev_pct == pytest.approx(0.0, abs=1e-9)


def test_integrate_returns_shadow_result_with_region():
    base = nus333_usa_baseline()
    res = integrate(base, u_pct=0.1, dpsave_pct=0.0, n_steps=5)
    assert isinstance(res, ShadowResult)
    assert res.region == base.region


def test_integrate_linear_in_inputs_for_small_shocks():
    """With dppriv=dpgov=dpsave=0 and tiny u, EV ≈ INCOME · u / 100 (linear)."""
    base = nus333_usa_baseline()
    small = 0.001
    res = integrate(base, u_pct=small, dpsave_pct=0.0, n_steps=10)
    expected = base.INCOME * small / 100.0
    assert res.EV_USDm == pytest.approx(expected, rel=1e-3)


def test_integrate_doubling_u_doubles_ev_in_linear_regime():
    base = nus333_usa_baseline()
    r1 = integrate(base, u_pct=0.001, dpsave_pct=0.0, n_steps=10)
    r2 = integrate(base, u_pct=0.002, dpsave_pct=0.0, n_steps=10)
    assert r2.EV_USDm == pytest.approx(2.0 * r1.EV_USDm, rel=1e-4)


def test_integrate_negative_u_yields_negative_ev():
    base = nus333_usa_baseline()
    res = integrate(base, u_pct=-0.5, dpsave_pct=0.0, n_steps=10)
    assert res.EV_USDm < 0.0


# ---------------------------------------------------------------------------
# Integration methods — convergence properties
# ---------------------------------------------------------------------------


def test_integrate_default_method_is_euler_n25():
    """Documented default: Euler N=25 — calibrated to GEMPACK Gragg-8-16-32."""
    base = nus333_usa_baseline()
    default = integrate(base, u_pct=0.1725, dpsave_pct=16.18)
    explicit = integrate(
        base, u_pct=0.1725, dpsave_pct=16.18, method="euler", n_steps=25,
    )
    assert default.EV_USDm == pytest.approx(explicit.EV_USDm, rel=1e-9)


def test_all_methods_agree_on_zero_shock():
    base = nus333_usa_baseline()
    for method in ("euler", "midpoint", "gragg", "bulirsch_stoer"):
        res = integrate(
            base, u_pct=0.0, dpsave_pct=0.0, n_steps=8, method=method,
        )
        assert res.EV_USDm == pytest.approx(0.0, abs=1e-9), method


def test_methods_diverge_on_large_dpsave():
    """For the NUS333 10pct USA shock (capFix-swap closure), the four
    integrators give different EVs because GEMPACK's Richardson kernel
    doesn't converge to the asymptotic ODE fixed point. Default Euler-25
    is the calibration we ship.
    """
    base = nus333_usa_baseline()
    kw = dict(u_pct=0.1725, dpsave_pct=16.18)
    euler25 = integrate(base, **kw, method="euler", n_steps=25)
    bs      = integrate(base, **kw, method="bulirsch_stoer")
    # Both should sit in the right neighbourhood of $14k USD M.
    assert 13_000 < euler25.EV_USDm < 16_000
    assert 13_000 < bs.EV_USDm      < 16_000
    # And they should differ by the documented ~2% structural gap.
    assert abs(euler25.EV_USDm - bs.EV_USDm) > 100.0


def test_integrate_invalid_n_steps_raises():
    base = nus333_usa_baseline()
    with pytest.raises(ValueError):
        integrate(base, u_pct=0.1, n_steps=0)


def test_integrate_invalid_method_raises():
    base = nus333_usa_baseline()
    with pytest.raises(ValueError):
        integrate(base, u_pct=0.1, n_steps=5, method="not_a_method")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Calibration sanity: NUS333 USA reference value
# ---------------------------------------------------------------------------


def test_nus333_usa_default_matches_rungtap_within_1pct():
    """Default integrator (Euler N=25) reproduces RunGTAP's EV for the
    NUS333 USA 10% tariff shock under capFix-swap closure to <1%.
    Reference: RunGTAP gtapv7.exe with `Steps=8 16 32 Subintervals=1`
    plus `swap dpsave("USA")=del_tbalry("USA")`.
    """
    base = nus333_usa_baseline()
    res = integrate(base, u_pct=0.1725, dpsave_pct=16.18)
    rungtap_target = 14_933.34  # USD millions, from sl4dump.har header 0208
    gap_pct = abs(res.EV_USDm - rungtap_target) / rungtap_target * 100
    assert gap_pct < 1.0, (
        f"EV USA = {res.EV_USDm:,.2f} vs RunGTAP {rungtap_target:,.2f} "
        f"(gap {gap_pct:.2f}%); calibration drifted"
    )


def test_nus333_usa_sub_components_match_expected_signs():
    """Under a positive tariff shock for USA:
        - dpsave is positive (savings preference rose)
        - log(UTILSAVEEV) grows → log_term positive → yev < u
        - uelasev negative
        - qsaveev positive (savings volume rose in shadow system)
    """
    base = nus333_usa_baseline()
    res = integrate(base, u_pct=0.1725, dpsave_pct=16.18)
    assert res.yev_pct > 0.0          # USA gains welfare
    assert res.yev_pct < 0.1725        # log_term shaves it from raw u
    assert res.uelasev_pct < 0.0       # dpavev > 0 dominates
    assert res.qsaveev_pct > 0.0       # virtual savings volume rises
