"""Phase 1 scaffolding tests — t_set, Sets, prev_t helper."""
import os
import sys
import pytest
from pathlib import Path
os.environ['EQUILIBRIA_GTAP_RRES'] = 'USA'
sys.path.insert(0, str(Path('src').resolve()))

from equilibria.templates.gtap.gtap_parameters import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations, prev_t
from equilibria.templates.gtap.gtap_contract import default_gtap_contract

GDX_15X10 = Path('datasets/gtap7_15x10/GDX/basedata.gdx').resolve()


def _build(t_set):
    contract = default_gtap_contract()
    p = GTAPParameters(); p.load_from_gdx(GDX_15X10)
    eq = GTAPModelEquations(p.sets, p, contract.closure, t_set=t_set)
    return eq.build_model()


def test_default_t_set_is_base_only():
    m = _build(("base",))
    assert list(m.t) == ["base"]
    assert list(m.t0) == ["base"]
    assert list(m.ts) == []


def test_three_period_t_set():
    m = _build(("base", "check", "shock"))
    assert list(m.t) == ["base", "check", "shock"]
    assert list(m.t0) == ["base"]
    assert list(m.ts) == ["check", "shock"]


def test_prev_t_helper():
    t_set = ("base", "check", "shock")
    assert prev_t("base", t_set) is None
    assert prev_t("check", t_set) == "base"
    assert prev_t("shock", t_set) == "check"


def test_t_set_validation():
    contract = default_gtap_contract()
    p = GTAPParameters(); p.load_from_gdx(GDX_15X10)
    with pytest.raises(ValueError, match="t_set"):
        GTAPModelEquations(p.sets, p, contract.closure, t_set=())
    with pytest.raises(ValueError, match="base"):
        GTAPModelEquations(p.sets, p, contract.closure, t_set=("shock", "check"))


def test_prev_t_unknown_raises():
    t_set = ("base", "check", "shock")
    with pytest.raises(ValueError, match="not in t_set"):
        prev_t("foo", t_set)


def _build_3p():
    """Build with t_set=(base, check, shock) — for verifying t-indexing."""
    return _build(("base", "check", "shock"))


def test_production_vars_have_t_dim():
    m = _build_3p()
    for period in ("base", "check", "shock"):
        assert ('USA', 'VegFruit', period) in m.xp, f"xp missing for {period}"
        assert ('USA', 'VegFruit', period) in m.va
        assert ('USA', 'VegFruit', period) in m.nd
        assert ('USA', 'VegFruit', 'VegFruit', period) in m.x


def test_production_vars_single_period_still_works():
    m = _build(("base",))
    assert ('USA', 'VegFruit', 'base') in m.xp
    assert ('USA', 'VegFruit', 'check') not in m.xp


def test_lift_to_t_broadcasts_init_across_periods():
    """_lift_to_t must give every period the same initial value (warm start)."""
    from pyomo.environ import value as pyo_value
    m = _build_3p()
    for r, a in [('USA', 'VegFruit'), ('EU_28', 'Chemicals')]:
        v0 = pyo_value(m.xp[r, a, 'base'])
        v1 = pyo_value(m.xp[r, a, 'check'])
        v2 = pyo_value(m.xp[r, a, 'shock'])
        assert v0 == v1 == v2, f"xp[{r},{a}] not broadcast: {v0}/{v1}/{v2}"


def test_trade_vars_have_t_dim():
    m = _build_3p()
    assert ('USA', 'VegFruit', 'VegFruit', 'base') in m.pa
    assert ('USA', 'VegFruit', 'CAN', 'base') in m.xw
    assert ('USA', 'VegFruit', 'CAN', 'base') in m.pe
    assert ('USA', 'VegFruit', 'CAN', 'shock') in m.pefob


def test_trade_vars_init_broadcasts_across_periods():
    """Verify _lift_to_t broadcasts trade-Var inits to every period (warm start)."""
    from pyomo.environ import value as pyo_value
    m = _build_3p()
    for (idx_base, idx_check) in [
        (('USA', 'VegFruit', 'VegFruit', 'base'),  ('USA', 'VegFruit', 'VegFruit', 'check')),
        (('USA', 'VegFruit', 'CAN', 'base'),        ('USA', 'VegFruit', 'CAN', 'check')),
    ]:
        if idx_base in m.pa and idx_check in m.pa:
            assert pyo_value(m.pa[idx_base]) == pyo_value(m.pa[idx_check])
        if idx_base in m.xw and idx_check in m.xw:
            # xw had ~2e-9 roundoff per Phase 2.2 report — keep approx for this Var only.
            assert pyo_value(m.xw[idx_base]) == pytest.approx(pyo_value(m.xw[idx_check]), rel=1e-8)


def test_factor_vars_have_t_dim():
    m = _build_3p()
    assert ('USA', 'UnSkLab', 'VegFruit', 'base') in m.pf
    assert ('USA', 'UnSkLab', 'VegFruit', 'shock') in m.pfa
    assert ('USA', 'UnSkLab', 'VegFruit', 'check') in m.xf
    assert ('USA', 'UnSkLab', 'base') in m.pft
    assert ('USA', 'UnSkLab', 'shock') in m.pft
    assert ('USA', 'UnSkLab', 'base') in m.xft


def test_factor_vars_init_broadcasts_across_periods():
    """_lift_to_t broadcasts factor inits to every period (warm start)."""
    import pytest
    from pyomo.environ import value as pyo_value
    m = _build_3p()
    for (idx_base, idx_chk) in [
        (('USA', 'UnSkLab', 'VegFruit', 'base'), ('USA', 'UnSkLab', 'VegFruit', 'check')),
        (('USA', 'UnSkLab', 'base'),              ('USA', 'UnSkLab', 'check')),
    ]:
        if idx_base in m.pf and idx_chk in m.pf:
            assert pyo_value(m.pf[idx_base]) == pyo_value(m.pf[idx_chk])
        if idx_base in m.pft and idx_chk in m.pft:
            assert pyo_value(m.pft[idx_base]) == pyo_value(m.pft[idx_chk])


def test_income_vars_have_t_dim():
    m = _build_3p()
    for vname in ('yc', 'yg', 'rsav', 'regy', 'facty', 'ytax_ind',
                  'yi', 'xi', 'xiagg', 'pi', 'kstock'):
        v = getattr(m, vname, None)
        if v is None:
            continue  # not declared — note in report
        # xi has 2 r/i indices; the rest are r-indexed.
        if vname == 'xi':
            assert ('USA', 'VegFruit', 'base') in v, f"{vname} missing base"
            assert ('USA', 'VegFruit', 'check') in v, f"{vname} missing check"
            assert ('USA', 'VegFruit', 'shock') in v, f"{vname} missing shock"
        else:
            assert ('USA', 'base') in v, f"{vname} missing base"
            assert ('USA', 'check') in v, f"{vname} missing check"
            assert ('USA', 'shock') in v, f"{vname} missing shock"
    # ytax has extra (r, gy) index
    assert ('USA', 'dt', 'base') in m.ytax


def test_regy_fixed_only_at_base():
    """eq_regy must be Constraint.Skip at base (regy anchored); active for ts."""
    m = _build_3p()
    assert m.regy['USA', 'base'].fixed
    assert not m.regy['USA', 'check'].fixed
    assert not m.regy['USA', 'shock'].fixed


def test_price_agg_vars_have_t_dim():
    m = _build(("base", "check", "shock"))
    # Scalar vars (indexed by t only)
    assert 'base' in m.pmuv and 'check' in m.pmuv and 'shock' in m.pmuv
    assert 'base' in m.pwfact and 'check' in m.pwfact
    assert 'base' in m.pnum
    assert 'base' in m.walras
    # Per-region aggregates
    assert ('USA', 'base') in m.pabs
    assert ('USA', 'check') in m.pabs
    assert ('USA', 'base') in m.pfact
    assert ('USA', 'base') in m.pgdpmp
    assert ('USA', 'base') in m.gdpmp
    assert ('USA', 'base') in m.rgdpmp


def test_welfare_savings_vars_have_t_dim():
    m = _build(("base", "check", "shock"))
    # Per-region utility/savings vars
    for period in ("base", "check", "shock"):
        assert ('USA', period) in m.uh, f"uh missing for {period}"
        assert ('USA', period) in m.ug, f"ug missing for {period}"
        assert ('USA', period) in m.us, f"us missing for {period}"
        assert ('USA', period) in m.u, f"u missing for {period}"
        assert ('USA', period) in m.pcons, f"pcons missing for {period}"
        assert ('USA', period) in m.psave, f"psave missing for {period}"
        assert ('USA', period) in m.savf, f"savf missing for {period}"
    # Global rate of return: scalar-per-period
    assert 'base' in m.rorg, "rorg missing base"
    assert 'check' in m.rorg, "rorg missing check"
    assert 'shock' in m.rorg, "rorg missing shock"
