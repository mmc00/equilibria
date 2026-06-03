"""Tests for gtap_iterloop.py — Phase 3 in-place period fixings."""
import os
import sys
from pathlib import Path

import pytest

os.environ['EQUILIBRIA_GTAP_RRES'] = 'USA'
sys.path.insert(0, str(Path('src').resolve()))

from pyomo.environ import value as pyo_value
from equilibria.templates.gtap.gtap_parameters import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_contract import default_gtap_contract
from equilibria.templates.gtap.gtap_iterloop import apply_iterloop_fixings

GDX = Path('datasets/gtap7_15x10/GDX/basedata.gdx').resolve()


def _build_15x10_multi():
    contract = default_gtap_contract()
    p = GTAPParameters()
    p.load_from_gdx(GDX)
    eq = GTAPModelEquations(
        p.sets, p, contract.closure, t_set=("base", "check", "shock")
    )
    return eq.build_model(), p


# --- Task 3.1 ---------------------------------------------------------------


def test_fix_tax_instruments_at_tsim():
    m, p = _build_15x10_multi()
    apply_iterloop_fixings(
        m, "check",
        t_set=("base", "check", "shock"),
        sets=p.sets, params=p, flags={}, first_year="base",
    )
    assert m.imptx['USA', 'VegFruit', 'CAN', 'check'].fixed
    assert m.prdtx['USA', 'VegFruit', 'VegFruit', 'check'].fixed
    assert m.fcttx['USA', 'UnSkLab', 'VegFruit', 'check'].fixed


def test_fix_trade_margins_at_tsim():
    m, p = _build_15x10_multi()
    apply_iterloop_fixings(
        m, "shock",
        t_set=("base", "check", "shock"),
        sets=p.sets, params=p, flags={}, first_year="base",
    )
    assert ('USA', 'VegFruit', 'CAN', 'shock') in m.tmarg


# --- Task 3.2 ---------------------------------------------------------------


def test_price_lower_bounds_set():
    from equilibria.templates.gtap.gtap_iterloop import _set_price_lower_bounds
    m, p = _build_15x10_multi()
    for r in p.sets.r:
        for f in p.sets.f:
            for a in p.sets.a:
                m.pf[r, f, a, "base"].set_value(1.0)
    _set_price_lower_bounds(m, "check", ("base", "check", "shock"))
    for r in p.sets.r:
        for f in p.sets.f:
            for a in p.sets.a:
                assert abs(m.pf[r, f, a, "check"].lb - 0.001) < 1e-9


# --- Task 3.3 ---------------------------------------------------------------


def test_fix_inactive_flows_xf_zero_flag():
    from equilibria.templates.gtap.gtap_iterloop import _fix_inactive_flows
    m, p = _build_15x10_multi()
    for r in p.sets.r:
        for f in p.sets.f:
            for a in p.sets.a:
                m.pf[r, f, a, "base"].set_value(1.0)
    flags = {'xfFlag': {('USA', 'UnSkLab', 'VegFruit'): False}}
    _fix_inactive_flows(m, 'check', ('base', 'check', 'shock'), flags)
    assert m.xf['USA', 'UnSkLab', 'VegFruit', 'check'].fixed
    assert abs(pyo_value(m.xf['USA', 'UnSkLab', 'VegFruit', 'check'])) < 1e-12
    assert m.pf['USA', 'UnSkLab', 'VegFruit', 'check'].fixed
    assert abs(pyo_value(m.pf['USA', 'UnSkLab', 'VegFruit', 'check']) - 1.0) < 1e-9


# --- Task 3.4 ---------------------------------------------------------------


def test_fix_lagged_state_at_check_pins_base_vars():
    from equilibria.templates.gtap.gtap_iterloop import _fix_lagged_state
    m, p = _build_15x10_multi()
    for r in p.sets.r:
        for f in p.sets.f:
            for a in p.sets.a:
                m.pf[r, f, a, "base"].set_value(1.0)
                m.xf[r, f, a, "base"].set_value(2.0)
        m.pabs[r, "base"].set_value(1.5)
    m.pmuv["base"].set_value(1.1)
    m.pwfact["base"].set_value(1.2)
    _fix_lagged_state(m, "check", ("base", "check", "shock"), "base")
    assert m.pf['USA', 'UnSkLab', 'VegFruit', 'base'].fixed
    assert abs(pyo_value(m.pf['USA', 'UnSkLab', 'VegFruit', 'base']) - 1.0) < 1e-12
    assert m.xf['USA', 'UnSkLab', 'VegFruit', 'base'].fixed
    assert m.pmuv['base'].fixed
    assert m.pwfact['base'].fixed
    assert m.pabs['USA', 'base'].fixed


def test_fix_lagged_state_skips_first_year():
    from equilibria.templates.gtap.gtap_iterloop import _fix_lagged_state
    m, p = _build_15x10_multi()
    _fix_lagged_state(m, "base", ("base", "check", "shock"), "base")
    assert not m.pf['USA', 'UnSkLab', 'VegFruit', 'base'].fixed


# --- Task 3.5 ---------------------------------------------------------------


def test_build_flags_dict_returns_15_flags():
    sys.path.insert(0, str(Path('scripts/gtap').resolve()))
    from run_gtap import _build_flags_dict
    p = GTAPParameters()
    p.load_from_gdx(GDX)
    flags = _build_flags_dict(p)
    expected = {
        'xfFlag', 'xftFlag', 'xFlag', 'xaFlag', 'ndFlag', 'vaFlag',
        'xwFlag', 'tmgFlag', 'xpFlag', 'alphad', 'alpham', 'xsFlag',
        'xmtFlag', 'xdFlag', 'xetFlag',
    }
    assert set(flags.keys()) >= expected
    assert isinstance(flags['xfFlag'], dict)
    sample_key = next(iter(flags['xfFlag']))
    assert isinstance(sample_key, tuple) and len(sample_key) == 3
